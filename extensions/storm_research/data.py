"""Data classes for the STORM research pipeline.

Lightweight replacements for knowledge_storm's dataclasses — no numpy,
sklearn, or sentence-transformers dependencies.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single search result from a web search."""

    url: str
    title: str
    description: str
    snippets: list[str] = field(default_factory=list)
    citation_index: int = -1  # assigned during article generation

    def to_dict(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "snippets": self.snippets,
            "citation_index": self.citation_index,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SearchResult":
        return cls(
            url=d["url"],
            title=d.get("title", ""),
            description=d.get("description", ""),
            snippets=d.get("snippets", []),
            citation_index=d.get("citation_index", -1),
        )


# ---------------------------------------------------------------------------
# Conversation structures
# ---------------------------------------------------------------------------

@dataclass
class DialogueTurn:
    """A single turn in a simulated conversation between writer and expert."""

    question: str
    answer: str
    search_queries: list[str] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "search_queries": self.search_queries,
            "search_results": [r.to_dict() for r in self.search_results],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DialogueTurn":
        return cls(
            question=d["question"],
            answer=d["answer"],
            search_queries=d.get("search_queries", []),
            search_results=[
                SearchResult.from_dict(r) for r in d.get("search_results", [])
            ],
        )


@dataclass
class Conversation:
    """A full conversation for one persona/perspective."""

    persona: str
    turns: list[DialogueTurn] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "persona": self.persona,
            "turns": [t.to_dict() for t in self.turns],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Conversation":
        return cls(
            persona=d["persona"],
            turns=[DialogueTurn.from_dict(t) for t in d.get("turns", [])],
        )


# ---------------------------------------------------------------------------
# Article / outline structures
# ---------------------------------------------------------------------------

@dataclass
class SectionNode:
    """A node in the hierarchical article outline tree."""

    name: str
    content: str = ""
    children: list["SectionNode"] = field(default_factory=list)

    def add_child(self, child: "SectionNode") -> None:
        self.children.append(child)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "content": self.content,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SectionNode":
        node = cls(name=d["name"], content=d.get("content", ""))
        for child_d in d.get("children", []):
            node.add_child(cls.from_dict(child_d))
        return node


# ---------------------------------------------------------------------------
# Information table
# ---------------------------------------------------------------------------

@dataclass
class InformationTable:
    """Collected information from all conversations, indexed by URL.

    Replaces StormInformationTable — uses simple keyword matching instead
    of embedding-based retrieval (modern 128k+ context windows make this
    viable without sentence-transformers).
    """

    conversations: list[Conversation] = field(default_factory=list)
    url_to_info: dict[str, SearchResult] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.conversations and not self.url_to_info:
            self.url_to_info = self._build_index()

    def _build_index(self) -> dict[str, SearchResult]:
        """Merge all search results from conversations, deduping by URL."""
        index: dict[str, SearchResult] = {}
        for conv in self.conversations:
            for turn in conv.turns:
                for result in turn.search_results:
                    if result.url in index:
                        existing = index[result.url]
                        for snippet in result.snippets:
                            if snippet not in existing.snippets:
                                existing.snippets.append(snippet)
                    else:
                        index[result.url] = SearchResult(
                            url=result.url,
                            title=result.title,
                            description=result.description,
                            snippets=list(result.snippets),
                        )
        return index

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Simple keyword-based retrieval from collected information.

        Scores each source by how many query terms appear in its
        title + description + snippets.  Good enough for retrieval
        when the full context is also available to the LLM.
        """
        query_terms = set(query.lower().split())
        scored: list[tuple[float, SearchResult]] = []

        for info in self.url_to_info.values():
            text = " ".join(
                [info.title, info.description] + info.snippets
            ).lower()
            # Simple term-frequency score
            score = sum(1 for t in query_terms if t in text)
            if score > 0:
                scored.append((score, info))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [info for _, info in scored[:top_k]]

    def all_snippets_text(self, max_chars: int = 30000) -> str:
        """Return a formatted string of all collected information."""
        parts: list[str] = []
        total = 0
        for url, info in self.url_to_info.items():
            block = f"Source: {info.title} ({url})\n"
            for snippet in info.snippets:
                block += f"  - {snippet}\n"
            if total + len(block) > max_chars:
                break
            parts.append(block)
            total += len(block)
        return "\n".join(parts)

    def assign_citation_indices(self) -> dict[str, int]:
        """Assign sequential citation numbers [1], [2], ... to all sources."""
        mapping: dict[str, int] = {}
        idx = 1
        for url, info in self.url_to_info.items():
            info.citation_index = idx
            mapping[url] = idx
            idx += 1
        return mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversations": [c.to_dict() for c in self.conversations],
            "url_to_info": {u: r.to_dict() for u, r in self.url_to_info.items()},
        }


# ---------------------------------------------------------------------------
# Final article
# ---------------------------------------------------------------------------

@dataclass
class StormArticle:
    """The final output of the STORM pipeline."""

    topic: str
    outline: SectionNode
    conversations: list[Conversation] = field(default_factory=list)
    references: dict[str, SearchResult] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render the article as a Markdown string with citations."""
        lines: list[str] = []
        self._render_section(self.outline, lines, level=1)

        # References section
        if self.references:
            lines.append("\n---\n")
            lines.append("## References\n")
            for url, ref in self.references.items():
                idx = ref.citation_index
                if idx > 0:
                    lines.append(f"[{idx}] {ref.title}. {url}")
                else:
                    lines.append(f"- {ref.title}. {url}")

        return "\n".join(lines)

    def _render_section(
        self, node: SectionNode, lines: list[str], level: int
    ) -> None:
        """Recursively render the outline tree to markdown."""
        heading = "#" * level
        lines.append(f"{heading} {node.name}\n")
        if node.content:
            lines.append(node.content)
            lines.append("")
        for child in node.children:
            self._render_section(child, lines, level + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "outline": self.outline.to_dict(),
            "conversations": [c.to_dict() for c in self.conversations],
            "references": {u: r.to_dict() for u, r in self.references.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StormArticle":
        return cls(
            topic=d["topic"],
            outline=SectionNode.from_dict(d["outline"]),
            conversations=[
                Conversation.from_dict(c) for c in d.get("conversations", [])
            ],
            references={
                u: SearchResult.from_dict(r)
                for u, r in d.get("references", {}).items()
            },
        )


# ---------------------------------------------------------------------------
# Outline parsing helpers
# ---------------------------------------------------------------------------

def parse_outline_markdown(topic: str, outline_text: str) -> SectionNode:
    """Parse a markdown outline (with # headings) into a SectionNode tree.

    Example input::

        # Topic Name
        ## Introduction
        ## Background
        ### Early History
        ### Modern Development
        ## Conclusion

    Returns a tree rooted at a SectionNode named *topic*.
    """
    root = SectionNode(name=topic)
    stack: list[tuple[int, SectionNode]] = [(0, root)]

    for line in outline_text.strip().splitlines():
        line = line.strip()
        match = re.match(r"^(#{1,6})\s+(.*)", line)
        if not match:
            continue
        level = len(match.group(1))
        name = match.group(2).strip()

        # Skip the root heading if it matches the topic
        if level == 1 and name.lower() == topic.lower():
            continue

        node = SectionNode(name=name)

        # Pop back to the correct parent
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()

        parent = stack[-1][1]
        parent.add_child(node)
        stack.append((level, node))

    return root
