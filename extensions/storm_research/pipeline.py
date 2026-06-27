"""Core STORM pipeline — reimplementation of the STORM algorithm.

This module implements the 6-stage research pipeline without any dependency
on dspy, litellm, or other heavy packages.  All LLM interactions go through
a simple callable interface that the extension wires up to tau's provider.

Reference: Shao et al., "Assisting in Writing Wikipedia-like Articles From
Scratch with Large Language Models" (NAACL 2024, arXiv:2402.14207).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable

from storm_research import prompts
from storm_research.data import (
    Conversation,
    DialogueTurn,
    InformationTable,
    SearchResult,
    SectionNode,
    StormArticle,
    parse_outline_markdown,
)
from storm_research.search import SearchBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Outline response cleaner
# ---------------------------------------------------------------------------

# Phrases that indicate a heading is meta-commentary, not a real section.
_META_HEADING_PATTERNS = re.compile(
    r"(?:here is|here\'s|my outline|master framework|i will|"
    r"outline for|the following|below is|let me|as requested|"
    r"expect from|makes a good|template preview|ready to|proposed outline)",
    re.IGNORECASE,
)


def _clean_outline_response(text: str) -> str:
    """Strip meta-commentary from an outline response.

    Some models (esp. Gemma) produce lines like::

        ## Here is the "Master Framework" I will use for your outline:

    which are not real section headings.  This function removes:
      - Non-heading lines (preamble/commentary)
      - Heading lines whose text looks like meta-commentary
    """
    cleaned_lines: list[str] = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        # Keep only heading lines (## ...)
        heading_match = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if not heading_match:
            # Non-heading line: skip unless it's blank (preserve spacing)
            continue
        heading_text = heading_match.group(2).strip()
        # Skip meta-commentary headings
        if _META_HEADING_PATTERNS.search(heading_text):
            continue
        cleaned_lines.append(stripped)
    return "\n".join(cleaned_lines)


# Phrases that indicate the model is roleplaying as an editor/assistant
# instead of writing actual article content.
_BROKEN_LEAD_PATTERNS = re.compile(
    r"(?:paste your (?:article|text)|i am ready to (?:assist|help|edit)|"
    r"please (?:provide|paste|send|share) (?:your|the) |"
    r"proofreading|copyediting|substantive editing|"
    r"developmental editing|level of editing|"
    r"target audience|desired tone|whenever you are ready)",
    re.IGNORECASE,
)


def _is_broken_lead(text: str, topic: str = "") -> bool:
    """Return True if the lead section looks like a broken response.

    Checks for:
      - Empty or very short responses
      - Editing-service roleplay patterns
      - Off-topic content (topic name not mentioned)
      - Multi-section structure (headings indicate a mini-article, not a summary)
    """
    if not text or len(text.strip()) < 50:
        return True
    if _BROKEN_LEAD_PATTERNS.search(text):
        return True

    # Off-topic check: the topic should appear somewhere in the lead
    if topic and topic.lower() not in text.lower():
        logger.warning(
            "STORM: lead section doesn't mention topic %r", topic,
        )
        return True

    # Structure check: a lead section should be plain paragraphs,
    # not a multi-section document with markdown headings
    heading_count = len(re.findall(r"^#{1,4}\s+", text, re.MULTILINE))
    if heading_count >= 3:
        logger.warning(
            "STORM: lead section has %d headings (expected plain paragraphs)",
            heading_count,
        )
        return True

    return False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StormConfig:
    """Tuneable parameters for the STORM pipeline."""

    num_perspectives: int = 3
    conv_turns: int = 3
    search_top_k: int = 5
    num_search_queries: int = 3
    max_info_chars: int = 30000  # max chars of collected info sent to LLM


# ---------------------------------------------------------------------------
# Type alias for the LLM call function
# ---------------------------------------------------------------------------

# The pipeline expects a callable with signature:
#   llm_call(system_prompt: str, user_prompt: str) -> str
LLMCallFn = Callable[[str, str], str]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class StormPipeline:
    """Reimplementation of the STORM algorithm using simple LLM calls.

    Parameters
    ----------
    llm_call : callable
        ``(system_prompt, user_prompt) -> response_text``.
        The extension wires this to tau's provider.
    search : SearchBackend
        Pluggable web search (DuckDuckGo, Tavily, etc.).
    config : StormConfig
        Pipeline parameters.
    on_progress : callable, optional
        ``(stage_name, detail_msg) -> None`` for progress reporting.
    """

    def __init__(
        self,
        llm_call: LLMCallFn,
        search: SearchBackend,
        config: StormConfig | None = None,
        on_progress: Callable[[str, str], None] | None = None,
    ) -> None:
        self.llm = llm_call
        self.search = search
        self.config = config or StormConfig()
        self._progress = on_progress or (lambda *_: None)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, topic: str) -> StormArticle:
        """Run the full STORM pipeline and return a polished article."""
        self._progress("start", f"Starting research on: {topic}")

        # Stage 1: Find related topics
        self._progress("related_topics", "Discovering related topics…")
        related = self._find_related_topics(topic)

        # Stage 2: Generate personas
        self._progress("personas", "Generating research perspectives…")
        personas = self._generate_personas(topic, related)

        # Stage 3: Simulated conversations
        self._progress("conversations", f"Running {len(personas)} research conversations…")
        conversations = self._run_conversations(topic, personas)

        # Stage 4: Generate outline
        self._progress("outline", "Generating article outline…")
        info_table = InformationTable(conversations=conversations)
        outline = self._generate_outline(topic, info_table)

        # Stage 5: Generate article
        self._progress("article", "Writing article sections…")
        article = self._generate_article(topic, outline, info_table)

        # Stage 6: Polish
        self._progress("polish", "Polishing final article…")
        polished = self._polish_article(topic, article, info_table)

        self._progress("done", "Research complete!")
        return polished

    def run_outline_only(self, topic: str) -> tuple[SectionNode, InformationTable]:
        """Run only the pre-writing stage (stages 1-4)."""
        self._progress("start", f"Starting outline research on: {topic}")

        related = self._find_related_topics(topic)
        personas = self._generate_personas(topic, related)
        conversations = self._run_conversations(topic, personas)
        info_table = InformationTable(conversations=conversations)
        outline = self._generate_outline(topic, info_table)

        self._progress("done", "Outline research complete!")
        return outline, info_table

    # ------------------------------------------------------------------
    # Stage 1: Find related topics
    # ------------------------------------------------------------------

    def _find_related_topics(self, topic: str) -> list[str]:
        """Ask the LLM to suggest related topics for perspective discovery."""
        prompt = prompts.FIND_RELATED_TOPICS.format(topic=topic)
        response = self.llm(
            "You are a knowledgeable research assistant.",
            prompt,
        )
        # Parse numbered list
        lines = []
        for line in response.strip().splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                # Remove numbering / bullets
                cleaned = re.sub(r"^[\d\.\-\*\)]+\s*", "", line).strip()
                if cleaned:
                    lines.append(cleaned)
        return lines or [topic]  # Fallback to the topic itself

    # ------------------------------------------------------------------
    # Stage 2: Generate perspectives / personas
    # ------------------------------------------------------------------

    def _generate_personas(self, topic: str, related: list[str]) -> list[str]:
        """Generate N distinct editor personas for the topic."""
        related_text = "\n".join(f"- {r}" for r in related)
        prompt = prompts.GENERATE_PERSONAS.format(
            topic=topic,
            num_perspectives=self.config.num_perspectives,
            related_topics=related_text,
        )
        response = self.llm(
            "You are assembling a diverse research team.",
            prompt,
        )
        # Parse numbered list of personas
        personas: list[str] = []
        for line in response.strip().splitlines():
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith("-")):
                cleaned = re.sub(r"^[\d\.\-\*\)]+\s*", "", line).strip()
                if cleaned:
                    personas.append(cleaned)

        if not personas:
            # Fallback: use the raw response as a single persona
            personas = [response.strip()[:200]]

        return personas[: self.config.num_perspectives]

    # ------------------------------------------------------------------
    # Stage 3: Simulated conversations
    # ------------------------------------------------------------------

    def _run_conversations(
        self, topic: str, personas: list[str]
    ) -> list[Conversation]:
        """Run a multi-turn conversation for each persona."""
        conversations: list[Conversation] = []

        for i, persona in enumerate(personas):
            self._progress(
                "conversation",
                f"Perspective {i + 1}/{len(personas)}: {persona[:60]}…",
            )
            conv = self._run_single_conversation(topic, persona)
            conversations.append(conv)

        return conversations

    def _run_single_conversation(
        self, topic: str, persona: str
    ) -> Conversation:
        """Simulate a multi-turn dialogue between writer and expert."""
        turns: list[DialogueTurn] = []

        for turn_idx in range(self.config.conv_turns):
            # Writer asks a question
            question = self._ask_question(topic, persona, turns)
            if not question or "thank you so much" in question.lower():
                break

            # Generate search queries
            queries = self._generate_search_queries(topic, question)

            # Search the web
            all_results: list[SearchResult] = []
            for query in queries:
                results = self.search.search(query, top_k=self.config.search_top_k)
                all_results.extend(results)

            # Deduplicate by URL
            seen_urls: set[str] = set()
            unique_results: list[SearchResult] = []
            for r in all_results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    unique_results.append(r)

            # Expert answers grounded in search results
            answer = self._expert_answer(topic, question, unique_results)

            turns.append(
                DialogueTurn(
                    question=question,
                    answer=answer,
                    search_queries=queries,
                    search_results=unique_results,
                )
            )

        return Conversation(persona=persona, turns=turns)

    def _ask_question(
        self, topic: str, persona: str, history: list[DialogueTurn]
    ) -> str:
        """Generate the writer's next question based on persona and history."""
        if not history:
            prompt = prompts.ASK_QUESTION_INITIAL.format(
                topic=topic, persona=persona
            )
        else:
            conv_text = self._format_conversation_history(history)
            prompt = prompts.ASK_QUESTION_WITH_PERSONA.format(
                topic=topic,
                persona=persona,
                conversation_history=conv_text,
            )

        response = self.llm("You are a curious research writer.", prompt)
        return response.strip()

    def _generate_search_queries(self, topic: str, question: str) -> list[str]:
        """Generate web search queries for a given question."""
        prompt = prompts.GENERATE_SEARCH_QUERIES.format(
            topic=topic,
            question=question,
            num_queries=self.config.num_search_queries,
        )
        response = self.llm("You are a search query generator.", prompt)
        queries = [
            line.strip()
            for line in response.strip().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        # Remove any numbering
        queries = [re.sub(r"^[\d\.\-\*\)]+\s*", "", q).strip() for q in queries]
        return queries[: self.config.num_search_queries] or [f"{topic} {question}"]

    def _expert_answer(
        self, topic: str, question: str, results: list[SearchResult]
    ) -> str:
        """Generate an expert answer grounded in search results."""
        # Format search results for the prompt
        results_text = self._format_search_results(results)
        prompt = prompts.EXPERT_ANSWER.format(
            topic=topic,
            question=question,
            search_results=results_text,
        )
        return self.llm("You are a knowledgeable topic expert.", prompt)

    # ------------------------------------------------------------------
    # Stage 4: Generate outline
    # ------------------------------------------------------------------

    def _generate_outline(
        self, topic: str, info_table: InformationTable
    ) -> SectionNode:
        """Generate a hierarchical outline from collected information."""
        collected_text = info_table.all_snippets_text(
            max_chars=self.config.max_info_chars
        )
        prompt = prompts.GENERATE_OUTLINE.format(
            topic=topic,
            collected_info=collected_text,
        )
        system_prompt = (
            f"You are a raw data generator. Generate only the Markdown headings (##, ###, ####) "
            f"for a detailed hierarchical outline about {topic}. Never output any intro, preamble, "
            f"explanations, questions, or conversational commentary. Start directly with the first heading."
        )
        response = self.llm(
            system_prompt,
            prompt,
        )

        # Clean the response: strip meta-commentary lines before parsing
        response = _clean_outline_response(response)

        outline = parse_outline_markdown(topic, response)

        # Count total leaf nodes — these are the sections that get content
        def _count_leaves(node: SectionNode) -> int:
            if not node.children:
                return 1
            return sum(_count_leaves(c) for c in node.children)

        leaf_count = _count_leaves(outline)

        # Fallback: if the outline has too few sections OR is too flat
        if len(outline.children) < 3:
            logger.warning(
                "STORM: outline only has %d sections, generating default",
                len(outline.children),
            )
            outline = self._default_outline(topic)
        elif leaf_count < 10:
            logger.warning(
                "STORM: outline is too flat (%d leaf sections), "
                "generating default with subsections",
                leaf_count,
            )
            outline = self._default_outline(topic)

        return outline

    def _default_outline(self, topic: str) -> SectionNode:
        """Generate a sensible default outline when the LLM produces too few sections.

        The outline must have subsections so that ``_fill_sections`` generates
        content for each leaf node, resulting in a comprehensive article.
        """
        root = SectionNode(name=topic)

        # Build a rich hierarchical outline
        structure = {
            "Introduction": [
                "Definition and Overview",
                "Core Concepts",
            ],
            "History and Development": [
                "Origins and Background",
                "Key Milestones",
                "Evolution Over Time",
            ],
            "Technical Architecture": [
                "System Design",
                "Core Algorithm",
                "Key Components",
            ],
            "Key Features and Innovations": [
                "Novel Contributions",
                "Comparison with Prior Approaches",
            ],
            "Applications and Impact": [
                "Primary Use Cases",
                "Broader Influence",
            ],
            "Criticism and Limitations": [
                "Known Challenges",
                "Areas for Improvement",
            ],
            "Legacy and Influence": [
                "Impact on the Field",
                "Future Directions",
            ],
        }

        for section_name, subsections in structure.items():
            section = SectionNode(name=section_name)
            for sub_name in subsections:
                section.add_child(SectionNode(name=sub_name))
            root.add_child(section)

        return root

    # ------------------------------------------------------------------
    # Stage 5: Generate article sections
    # ------------------------------------------------------------------

    def _generate_article(
        self,
        topic: str,
        outline: SectionNode,
        info_table: InformationTable,
    ) -> StormArticle:
        """Generate content for each section in the outline."""
        # Assign citation indices
        info_table.assign_citation_indices()

        # Render outline for context
        outline_text = self._render_outline_text(outline)

        # Generate content for each leaf/section
        self._fill_sections(topic, outline, outline_text, info_table)

        return StormArticle(
            topic=topic,
            outline=outline,
            conversations=info_table.conversations,
            references=dict(info_table.url_to_info),
        )

    def _fill_sections(
        self,
        topic: str,
        node: SectionNode,
        outline_text: str,
        info_table: InformationTable,
        written_sections: list[tuple[str, str]] | None = None,
    ) -> None:
        """Recursively generate content for sections that have no children or for all nodes."""
        if written_sections is None:
            written_sections = []

        if node.children:
            for child in node.children:
                self._fill_sections(
                    topic, child, outline_text, info_table, written_sections
                )
        else:
            # Leaf section — generate content
            self._progress("section", f"Writing: {node.name}")
            relevant = info_table.retrieve(f"{topic} {node.name}", top_k=8)
            sources_text = self._format_search_results_with_citations(relevant)

            # Format previously written sections to pass as context
            if written_sections:
                prev_text = ""
                for name, content in written_sections:
                    prev_text += f"## {name}\n{content}\n\n"
            else:
                prev_text = "No sections have been written yet."

            prompt = prompts.WRITE_SECTION.format(
                topic=topic,
                outline=outline_text,
                section_name=node.name,
                relevant_sources=sources_text,
                previous_sections=prev_text,
            )
            node.content = self.llm(
                "You are a direct, factual writer. Write the section content directly based on the sources. "
                "Never include any greetings, intro, self-reference, roleplay, or confirmation like 'I accept this role' or 'I am ready'. "
                "Start writing the first paragraph of the section content immediately.",
                prompt,
            )
            # Add to written sections for subsequent runs
            written_sections.append((node.name, node.content))

    # ------------------------------------------------------------------
    # Stage 6: Polish
    # ------------------------------------------------------------------

    def _polish_article(
        self,
        topic: str,
        article: StormArticle,
        info_table: InformationTable,
    ) -> StormArticle:
        """Add a lead section and optionally polish the article."""
        # Generate lead/summary section.
        # Truncate article text to prevent context-window overload which
        # causes some models to generate off-topic content.
        article_text = article.to_markdown()
        truncated = article_text[:4000]
        if len(article_text) > 4000:
            truncated += "\n\n[... article continues ...]"

        prompt = prompts.WRITE_LEAD_SECTION.format(
            topic=topic,
            article_text=truncated,
        )
        lead_content = self.llm(
            f"You are writing an encyclopedia article about {topic}. "
            f"Write only article content — never offer services, ask "
            f"questions, or provide instructions.",
            prompt,
        )

        # Validate: detect broken lead sections (off-topic, editing
        # services, multi-section structure, etc.)
        if _is_broken_lead(lead_content, topic):
            logger.warning("STORM: lead section looks broken, regenerating")
            lead_content = self.llm(
                f"You are writing an encyclopedia article about {topic}.",
                f"Write exactly 2-3 paragraphs summarizing {topic}. "
                f"The first sentence MUST mention {topic} by name. "
                f"No headings, no citations, no preamble — just the "
                f"summary paragraphs.\n\n"
                f"Key facts from the article:\n{truncated[:2000]}",
            )
            # If still broken, generate a minimal fallback
            if _is_broken_lead(lead_content, topic):
                lead_content = (
                    f"{topic} is a topic in artificial intelligence and "
                    f"machine learning. See the sections below for details."
                )

        # Strip any markdown headings from the lead (should be plain prose)
        lead_content = re.sub(
            r"^#{1,6}\s+.*$", "", lead_content, flags=re.MULTILINE
        ).strip()

        # Insert lead as the first child of the root
        lead_node = SectionNode(name="Summary", content=lead_content)
        article.outline.children.insert(0, lead_node)

        return article

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_conversation_history(turns: list[DialogueTurn]) -> str:
        """Format dialogue history for inclusion in prompts."""
        lines: list[str] = []
        for turn in turns[-4:]:  # Keep last 4 turns to manage context
            lines.append(f"You: {turn.question}")
            # Truncate long answers
            answer = turn.answer
            if len(answer) > 500:
                answer = answer[:500] + "…"
            lines.append(f"Expert: {answer}")
        return "\n".join(lines) or "N/A"

    @staticmethod
    def _format_search_results(results: list[SearchResult]) -> str:
        """Format search results for inclusion in expert-answer prompts."""
        if not results:
            return "No search results found."
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.title}")
            lines.append(f"    URL: {r.url}")
            for snippet in r.snippets[:2]:  # Max 2 snippets per source
                lines.append(f"    {snippet}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _format_search_results_with_citations(
        results: list[SearchResult],
    ) -> str:
        """Format results using their assigned citation indices."""
        if not results:
            return "No relevant sources found."
        lines: list[str] = []
        for r in results:
            idx = r.citation_index if r.citation_index > 0 else "?"
            lines.append(f"[{idx}] {r.title} ({r.url})")
            for snippet in r.snippets[:2]:
                lines.append(f"    {snippet}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _render_outline_text(node: SectionNode, level: int = 1) -> str:
        """Render the outline tree as a simple text representation."""
        lines: list[str] = []
        prefix = "#" * level
        lines.append(f"{prefix} {node.name}")
        for child in node.children:
            lines.append(StormPipeline._render_outline_text(child, level + 1))
        return "\n".join(lines)
