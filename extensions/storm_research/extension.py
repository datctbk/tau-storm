"""storm_research — deep research extension using the STORM algorithm.

Generates comprehensive, long-form articles with citations by:
1. Discovering diverse perspectives on the topic
2. Simulating multi-perspective research conversations (grounded in web search)
3. Generating a structured outline from collected information
4. Writing each section with inline citations
5. Polishing with a lead summary

Reference: Shao et al., "Assisting in Writing Wikipedia-like Articles From
Scratch with Large Language Models" (NAACL 2024, arXiv:2402.14207).

Tools:
  - storm_research  — full research pipeline → article with citations
  - storm_outline   — pre-writing stage only → outline + references

Slash commands:
  - /research <topic>     — quick-start a full research run
  - /research-status      — show search backend info
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure storm_research is importable when loaded dynamically as a file
_package_root = Path(__file__).resolve().parent.parent
if str(_package_root) not in sys.path:
    sys.path.insert(0, str(_package_root))

from tau.core.extension import Extension, ExtensionContext
from tau.core.types import (
    ExtensionManifest,
    SlashCommand,
    ToolDefinition,
    ToolParameter,
)

logger = logging.getLogger(__name__)


class StormResearchExtension(Extension):
    manifest = ExtensionManifest(
        name="storm_research",
        version="0.1.0",
        description=(
            "Deep research tool — generates long-form articles with citations "
            "using the STORM algorithm (multi-perspective research conversations "
            "grounded in web search)."
        ),
        author="tau",
        system_prompt_fragment=(
            "You have a deep research tool (`storm_research`) that generates "
            "comprehensive, long-form articles with citations on any topic. "
            "It works by discovering multiple perspectives, running simulated "
            "research conversations grounded in web search, and synthesizing "
            "the results into a well-organized article. Use it when the user "
            "asks for thorough research, in-depth analysis, or a comprehensive "
            "overview of a topic. You also have `storm_outline` for generating "
            "just the research outline and references without the full article."
        ),
    )

    def __init__(self) -> None:
        self._ext_context: ExtensionContext | None = None
        self._workspace_root: str = "."

    def on_load(self, context: ExtensionContext) -> None:
        self._ext_context = context
        if hasattr(context, "_agent_config") and context._agent_config:
            self._workspace_root = (
                getattr(context._agent_config, "workspace_root", ".") or "."
            )

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="storm_research",
                description=(
                    "Run a full STORM deep-research pipeline on a topic. "
                    "Discovers multiple perspectives, runs simulated research "
                    "conversations grounded in web search, generates an outline, "
                    "writes each section with inline citations, and polishes "
                    "the final article. Returns a comprehensive Markdown article. "
                    "This typically takes 1-3 minutes."
                ),
                parameters={
                    "topic": ToolParameter(
                        type="string",
                        description="The topic to research.",
                    ),
                    "num_perspectives": ToolParameter(
                        type="integer",
                        description="Number of research perspectives/personas (default 3).",
                        required=False,
                    ),
                    "conv_turns": ToolParameter(
                        type="integer",
                        description="Conversation turns per perspective (default 3).",
                        required=False,
                    ),
                },
                handler=self._handle_storm_research,
            ),
            ToolDefinition(
                name="storm_outline",
                description=(
                    "Run the pre-writing stage of the STORM pipeline: "
                    "discovers perspectives, runs research conversations, "
                    "and generates a structured outline with references. "
                    "Faster than storm_research (no article writing). "
                    "Use this to preview the research before committing to "
                    "a full article."
                ),
                parameters={
                    "topic": ToolParameter(
                        type="string",
                        description="The topic to research.",
                    ),
                    "num_perspectives": ToolParameter(
                        type="integer",
                        description="Number of research perspectives (default 3).",
                        required=False,
                    ),
                },
                handler=self._handle_storm_outline,
            ),
        ]

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def slash_commands(self) -> list[SlashCommand]:
        return [
            SlashCommand(
                name="research",
                description="Run STORM deep research on a topic.",
                usage="/research <topic>",
            ),
            SlashCommand(
                name="research-status",
                description="Show search backend and config info.",
                usage="/research-status",
            ),
        ]

    def handle_slash(
        self, command: str, args: str, context: ExtensionContext
    ) -> bool:
        if command == "research":
            if not args.strip():
                context.print("[dim]Usage: /research <topic>[/dim]")
                return True
            context.print(f"[cyan]Starting STORM research on:[/cyan] {args.strip()}")
            try:
                result = self._handle_storm_research(topic=args.strip())
                # Save to file
                output_path = self._save_research(args.strip(), result)
                context.print(
                    f"[green]✓ Research complete![/green] "
                    f"Saved to: {output_path}"
                )
                # Enqueue the result as a follow-up for the agent
                context.enqueue(
                    f"I just completed research on '{args.strip()}'. "
                    f"The article has been saved to {output_path}. "
                    f"Here is a summary of what I found:\n\n"
                    f"{result[:2000]}"
                )
            except Exception as exc:
                context.print(f"[red]Research failed: {exc}[/red]")
                logger.exception("STORM research failed")
            return True

        if command == "research-status":
            from storm_research.search import get_search_backend

            backend = get_search_backend()
            context.print(f"[cyan]Search backend:[/cyan] {backend.name}")
            context.print(
                f"[cyan]Output directory:[/cyan] {self._output_dir()}"
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Tool handlers
    # ------------------------------------------------------------------

    def _handle_storm_research(
        self,
        topic: str,
        num_perspectives: int = 3,
        conv_turns: int = 3,
    ) -> str:
        """Run the full STORM pipeline and return the article as markdown."""
        from storm_research.pipeline import StormConfig, StormPipeline
        from storm_research.search import get_search_backend

        config = StormConfig(
            num_perspectives=max(1, min(num_perspectives, 7)),
            conv_turns=max(1, min(conv_turns, 6)),
        )
        search = get_search_backend()
        llm_fn = self._make_llm_callable()

        pipeline = StormPipeline(
            llm_call=llm_fn,
            search=search,
            config=config,
            on_progress=self._report_progress,
        )

        article = pipeline.run(topic)

        # Save outputs
        output_path = self._save_research(topic, article.to_markdown())
        self._save_research_data(topic, article)

        return (
            f"# Research Article: {topic}\n\n"
            f"{article.to_markdown()}\n\n"
            f"---\n"
            f"*Saved to: {output_path}*\n"
            f"*Sources: {len(article.references)} references collected*"
        )

    def _handle_storm_outline(
        self,
        topic: str,
        num_perspectives: int = 3,
    ) -> str:
        """Run the pre-writing stage and return the outline."""
        from storm_research.pipeline import StormConfig, StormPipeline
        from storm_research.search import get_search_backend

        config = StormConfig(
            num_perspectives=max(1, min(num_perspectives, 7)),
        )
        search = get_search_backend()
        llm_fn = self._make_llm_callable()

        pipeline = StormPipeline(
            llm_call=llm_fn,
            search=search,
            config=config,
            on_progress=self._report_progress,
        )

        outline, info_table = pipeline.run_outline_only(topic)

        # Render outline
        lines: list[str] = [f"# Research Outline: {topic}\n"]
        self._render_outline_summary(outline, lines, level=2)

        lines.append(f"\n## Collected References ({len(info_table.url_to_info)})\n")
        for url, info in info_table.url_to_info.items():
            lines.append(f"- [{info.title}]({url})")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # LLM integration
    # ------------------------------------------------------------------

    def _make_llm_callable(self):
        """Create a simple LLM callable using tau's sub-session SDK.

        Returns a function ``(system_prompt, user_prompt) -> str``.
        """
        ctx = self._ext_context
        if ctx is None:
            raise RuntimeError("Extension not loaded — no context available.")

        def llm_call(system_prompt: str, user_prompt: str) -> str:
            """Call the LLM via a disposable tau sub-session."""
            from tau.core.types import TextDelta

            with ctx.create_sub_session(
                system_prompt=system_prompt,
                load_skills=False,
                load_extensions=False,
                load_context_files=False,
                max_turns=1,
                allowed_tools=[],  # No tools — pure LLM call
            ) as sub:
                events = sub.prompt_sync(user_prompt)
                # Collect all text deltas
                text_parts: list[str] = []
                for event in events:
                    if isinstance(event, TextDelta):
                        text_parts.append(event.text)
                result = "".join(text_parts)
                if not result:
                    # Fallback: check for content in ProviderResponse-like events
                    for event in events:
                        if hasattr(event, "content") and event.content:
                            return event.content
                return result

        return llm_call

    # ------------------------------------------------------------------
    # Progress reporting
    # ------------------------------------------------------------------

    def _report_progress(self, stage: str, detail: str) -> None:
        """Report progress to the user via the extension context."""
        if self._ext_context is not None:
            try:
                self._ext_context.set_spinner(f"🔬 {detail}", key="storm")
            except Exception:  # noqa: BLE001
                pass
        logger.info("STORM [%s] %s", stage, detail)

    # ------------------------------------------------------------------
    # File I/O
    # ------------------------------------------------------------------

    def _output_dir(self) -> Path:
        path = Path(self._workspace_root).resolve() / ".tau" / "research"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _safe_filename(self, topic: str) -> str:
        """Convert topic to a safe filename."""
        safe = topic.lower().strip()
        safe = safe.replace(" ", "_")
        safe = "".join(c for c in safe if c.isalnum() or c in "_-")
        return safe[:80] or "research"

    def _save_research(self, topic: str, markdown: str) -> str:
        """Save the research article to a markdown file."""
        out_dir = self._output_dir()
        filename = self._safe_filename(topic)
        path = out_dir / f"{filename}.md"
        path.write_text(markdown, encoding="utf-8")
        return str(path)

    def _save_research_data(self, topic: str, article) -> None:
        """Save the raw research data (conversations, references) as JSON."""
        out_dir = self._output_dir()
        filename = self._safe_filename(topic)
        path = out_dir / f"{filename}_data.json"
        try:
            path.write_text(
                json.dumps(article.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not save research data: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render_outline_summary(
        node, lines: list[str], level: int = 2
    ) -> None:
        prefix = "#" * level
        lines.append(f"{prefix} {node.name}")
        for child in node.children:
            StormResearchExtension._render_outline_summary(
                child, lines, level + 1
            )


EXTENSION = StormResearchExtension()
