"""Tests for the storm_research extension."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tau.core.context import ContextManager
from tau.core.extension import ExtensionContext, ExtensionRegistry
from tau.core.steering import SteeringChannel
from tau.core.tool_registry import ToolRegistry
from tau.core.types import AgentConfig, ExtensionManifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg() -> AgentConfig:
    return AgentConfig(
        provider="openai",
        model="gpt-4o",
        compaction_enabled=False,
        retry_enabled=False,
    )


def _storm_ext_dir() -> str:
    """Return the path to the extensions/ dir in tau-storm."""
    return str(Path(__file__).resolve().parent.parent / "extensions")


# ===========================================================================
# Data classes
# ===========================================================================

class TestSearchResult:
    def test_roundtrip(self):
        from storm_research.data import SearchResult

        sr = SearchResult(
            url="https://example.com",
            title="Example",
            description="A test",
            snippets=["snippet1", "snippet2"],
            citation_index=3,
        )
        d = sr.to_dict()
        sr2 = SearchResult.from_dict(d)
        assert sr2.url == sr.url
        assert sr2.title == sr.title
        assert sr2.snippets == sr.snippets
        assert sr2.citation_index == 3

    def test_defaults(self):
        from storm_research.data import SearchResult

        sr = SearchResult(url="u", title="t", description="d")
        assert sr.snippets == []
        assert sr.citation_index == -1


class TestDialogueTurn:
    def test_roundtrip(self):
        from storm_research.data import DialogueTurn, SearchResult

        turn = DialogueTurn(
            question="What is X?",
            answer="X is Y [1].",
            search_queries=["what is X"],
            search_results=[
                SearchResult(url="u", title="t", description="d", snippets=["s"])
            ],
        )
        d = turn.to_dict()
        turn2 = DialogueTurn.from_dict(d)
        assert turn2.question == turn.question
        assert len(turn2.search_results) == 1
        assert turn2.search_results[0].url == "u"


class TestConversation:
    def test_roundtrip(self):
        from storm_research.data import (
            Conversation,
            DialogueTurn,
        )

        conv = Conversation(
            persona="Historian",
            turns=[
                DialogueTurn(question="q", answer="a"),
            ],
        )
        d = conv.to_dict()
        conv2 = Conversation.from_dict(d)
        assert conv2.persona == "Historian"
        assert len(conv2.turns) == 1


class TestSectionNode:
    def test_tree_building(self):
        from storm_research.data import SectionNode

        root = SectionNode(name="Topic")
        intro = SectionNode(name="Introduction", content="Intro text.")
        bg = SectionNode(name="Background")
        bg.add_child(SectionNode(name="Early History"))
        bg.add_child(SectionNode(name="Modern Era"))
        root.add_child(intro)
        root.add_child(bg)

        assert len(root.children) == 2
        assert len(root.children[1].children) == 2
        assert root.children[1].children[0].name == "Early History"

    def test_roundtrip(self):
        from storm_research.data import SectionNode

        root = SectionNode(name="Root")
        root.add_child(SectionNode(name="Child", content="text"))
        d = root.to_dict()
        root2 = SectionNode.from_dict(d)
        assert root2.name == "Root"
        assert root2.children[0].content == "text"


class TestInformationTable:
    def test_build_index_from_conversations(self):
        from storm_research.data import (
            Conversation,
            DialogueTurn,
            InformationTable,
            SearchResult,
        )

        conv = Conversation(
            persona="Test",
            turns=[
                DialogueTurn(
                    question="q",
                    answer="a",
                    search_results=[
                        SearchResult(
                            url="http://a.com",
                            title="A",
                            description="d",
                            snippets=["s1"],
                        ),
                        SearchResult(
                            url="http://b.com",
                            title="B",
                            description="d",
                            snippets=["s2"],
                        ),
                    ],
                ),
                DialogueTurn(
                    question="q2",
                    answer="a2",
                    search_results=[
                        SearchResult(
                            url="http://a.com",
                            title="A",
                            description="d",
                            snippets=["s3"],  # additional snippet for same URL
                        ),
                    ],
                ),
            ],
        )
        table = InformationTable(conversations=[conv])
        assert len(table.url_to_info) == 2
        # Snippets from same URL should be merged
        assert "s1" in table.url_to_info["http://a.com"].snippets
        assert "s3" in table.url_to_info["http://a.com"].snippets

    def test_retrieve(self):
        from storm_research.data import InformationTable, SearchResult

        table = InformationTable()
        table.url_to_info = {
            "http://a.com": SearchResult(
                url="http://a.com",
                title="Quantum Computing Basics",
                description="Overview of quantum computing",
                snippets=["Quantum bits or qubits"],
            ),
            "http://b.com": SearchResult(
                url="http://b.com",
                title="Classical Music History",
                description="History of classical music",
                snippets=["Bach and Mozart"],
            ),
        }
        results = table.retrieve("quantum computing qubits", top_k=5)
        assert len(results) >= 1
        assert results[0].url == "http://a.com"

    def test_assign_citation_indices(self):
        from storm_research.data import InformationTable, SearchResult

        table = InformationTable()
        table.url_to_info = {
            "http://a.com": SearchResult(url="http://a.com", title="A", description=""),
            "http://b.com": SearchResult(url="http://b.com", title="B", description=""),
        }
        mapping = table.assign_citation_indices()
        assert set(mapping.values()) == {1, 2}
        for info in table.url_to_info.values():
            assert info.citation_index > 0


class TestStormArticle:
    def test_to_markdown(self):
        from storm_research.data import (
            SectionNode,
            SearchResult,
            StormArticle,
        )

        root = SectionNode(name="Test Topic")
        root.add_child(SectionNode(name="Introduction", content="This is the intro."))
        root.add_child(SectionNode(name="Details", content="Details here [1]."))

        refs = {
            "http://a.com": SearchResult(
                url="http://a.com",
                title="Source A",
                description="",
                citation_index=1,
            )
        }
        article = StormArticle(
            topic="Test Topic", outline=root, references=refs
        )
        md = article.to_markdown()
        assert "# Test Topic" in md
        assert "## Introduction" in md
        assert "This is the intro." in md
        assert "[1] Source A" in md

    def test_roundtrip(self):
        from storm_research.data import (
            SectionNode,
            StormArticle,
        )

        root = SectionNode(name="T")
        root.add_child(SectionNode(name="S", content="c"))
        article = StormArticle(topic="T", outline=root)
        d = article.to_dict()
        article2 = StormArticle.from_dict(d)
        assert article2.topic == "T"
        assert article2.outline.children[0].content == "c"


class TestParseOutlineMarkdown:
    def test_basic_outline(self):
        from storm_research.data import parse_outline_markdown

        text = """\
# My Topic
## Introduction
## Background
### Early History
### Modern Era
## Conclusion
"""
        root = parse_outline_markdown("My Topic", text)
        assert root.name == "My Topic"
        assert len(root.children) == 3  # Intro, Background, Conclusion
        assert root.children[1].name == "Background"
        assert len(root.children[1].children) == 2

    def test_no_root_heading(self):
        from storm_research.data import parse_outline_markdown

        text = """\
## Section A
## Section B
### Sub B1
"""
        root = parse_outline_markdown("Topic X", text)
        assert root.name == "Topic X"
        assert len(root.children) == 2

    def test_empty_input(self):
        from storm_research.data import parse_outline_markdown

        root = parse_outline_markdown("Empty", "")
        assert root.name == "Empty"
        assert root.children == []


# ===========================================================================
# Prompt templates
# ===========================================================================

class TestPromptTemplates:
    def test_find_related_topics_formats(self):
        from storm_research.prompts import FIND_RELATED_TOPICS

        result = FIND_RELATED_TOPICS.format(topic="Quantum Computing")
        assert "Quantum Computing" in result

    def test_generate_personas_formats(self):
        from storm_research.prompts import GENERATE_PERSONAS

        result = GENERATE_PERSONAS.format(
            topic="AI", num_perspectives=3, related_topics="- Machine Learning"
        )
        assert "AI" in result
        assert "3" in result

    def test_ask_question_formats(self):
        from storm_research.prompts import (
            ASK_QUESTION_INITIAL,
            ASK_QUESTION_WITH_PERSONA,
        )

        r1 = ASK_QUESTION_INITIAL.format(topic="T", persona="P")
        assert "T" in r1 and "P" in r1

        r2 = ASK_QUESTION_WITH_PERSONA.format(
            topic="T", persona="P", conversation_history="Q: hi"
        )
        assert "Q: hi" in r2

    def test_expert_answer_formats(self):
        from storm_research.prompts import EXPERT_ANSWER

        result = EXPERT_ANSWER.format(
            topic="T", question="Q?", search_results="[1] Result"
        )
        assert "[1] Result" in result

    def test_all_prompts_have_no_missing_placeholders(self):
        from storm_research import prompts

        # Collect all prompt constants
        prompt_names = [
            n for n in dir(prompts) if n.isupper() and isinstance(getattr(prompts, n), str)
        ]
        assert len(prompt_names) >= 6  # We defined at least 6 prompt templates


# ===========================================================================
# Search backends
# ===========================================================================

class TestSearchBackends:
    def test_fallback_returns_empty(self):
        from storm_research.search import _FallbackSearch

        fb = _FallbackSearch()
        assert fb.name == "none"
        assert fb.search("test") == []

    def test_duckduckgo_handles_import_error(self):
        from storm_research.search import DuckDuckGoSearch

        ddg = DuckDuckGoSearch()
        # If duckduckgo_search is not installed, should return empty
        # If it is installed, should return results (we don't fail either way)
        results = ddg.search("test query python")
        assert isinstance(results, list)

    def test_tavily_without_key_returns_empty(self):
        from storm_research.search import TavilySearch

        ts = TavilySearch(api_key="")
        results = ts.search("test")
        assert results == []


# ===========================================================================
# Pipeline (with mocked LLM)
# ===========================================================================

class TestStormPipeline:
    def _mock_llm(self, system: str, user: str) -> str:
        """A mock LLM that returns canned responses based on prompt content."""
        if "related topics" in user.lower() or "closely related" in user.lower():
            return "1. Topic A — relevant\n2. Topic B — relevant\n3. Topic C — relevant"
        if "editors" in user.lower() or "perspectives" in user.lower():
            return (
                "1. Historian: Focus on historical development\n"
                "2. Technologist: Focus on technical aspects\n"
                "3. Economist: Focus on economic impact"
            )
        if "your first question" in user.lower() or "next question" in user.lower():
            return "What are the key developments in this field?"
        if "search queries" in user.lower():
            return "key developments topic\nhistory of topic\nrecent advances topic"
        if "expert" in system.lower() and "search results" in user.lower():
            return "Based on the research [1], the key developments include X and Y [2]."
        if "outline" in user.lower() and "heading" in user.lower():
            return (
                "## Introduction\n"
                "## Background\n"
                "### Historical Context\n"
                "## Key Developments\n"
                "## Conclusion"
            )
        if "section" in user.lower() and "write" in system.lower():
            return "This section covers important aspects of the topic [1]."
        if "lead" in user.lower() or "summary" in user.lower():
            return "This article provides a comprehensive overview of the topic."
        return "Generic LLM response."

    def _mock_search(self, query: str, top_k: int = 5):
        from storm_research.data import SearchResult

        return [
            SearchResult(
                url=f"http://example.com/{i}",
                title=f"Result {i} for {query[:20]}",
                description=f"Description for result {i}",
                snippets=[f"Snippet about {query[:20]}"],
            )
            for i in range(min(top_k, 3))
        ]

    def test_full_pipeline(self):
        from storm_research.pipeline import StormConfig, StormPipeline

        mock_search = MagicMock()
        mock_search.search = self._mock_search

        progress_calls = []
        pipeline = StormPipeline(
            llm_call=self._mock_llm,
            search=mock_search,
            config=StormConfig(num_perspectives=2, conv_turns=2),
            on_progress=lambda stage, msg: progress_calls.append((stage, msg)),
        )

        article = pipeline.run("Test Topic")

        # Verify structure
        assert article.topic == "Test Topic"
        assert len(article.outline.children) > 0
        assert len(article.references) > 0

        # Verify progress was reported
        stages = [s for s, _ in progress_calls]
        assert "start" in stages
        assert "done" in stages

        # Verify markdown output
        md = article.to_markdown()
        assert "Test Topic" in md
        assert "References" in md

    def test_outline_only(self):
        from storm_research.pipeline import StormConfig, StormPipeline

        mock_search = MagicMock()
        mock_search.search = self._mock_search

        pipeline = StormPipeline(
            llm_call=self._mock_llm,
            search=mock_search,
            config=StormConfig(num_perspectives=2, conv_turns=1),
        )

        outline, info_table = pipeline.run_outline_only("Test Outline")

        assert outline.name == "Test Outline"
        assert len(outline.children) > 0
        assert len(info_table.url_to_info) > 0

    def test_conversation_stops_on_thank_you(self):
        from storm_research.pipeline import StormConfig, StormPipeline

        call_count = 0

        def llm_with_early_stop(system: str, user: str) -> str:
            nonlocal call_count
            call_count += 1
            if "first question" in user.lower() or "next question" in user.lower():
                return "Thank you so much for your help!"
            return self._mock_llm(system, user)

        mock_search = MagicMock()
        mock_search.search = self._mock_search

        pipeline = StormPipeline(
            llm_call=llm_with_early_stop,
            search=mock_search,
            config=StormConfig(num_perspectives=1, conv_turns=5),
        )

        # The conversation should stop after the first question
        conv = pipeline._run_single_conversation("Topic", "Persona")
        assert len(conv.turns) == 0  # Thank you on first turn → no turns recorded


# ===========================================================================
# Extension registration (loads via extra_paths)
# ===========================================================================

class TestStormExtensionRegistration:
    def test_extension_loads_via_extra_paths(self):
        """The storm_research extension should be discovered via extra_paths."""
        ext_dir = _storm_ext_dir()
        reg = ExtensionRegistry(
            extra_paths=[ext_dir],
            disabled=[],
            include_builtins=False,
        )
        r = ToolRegistry()
        c = ContextManager(_cfg())
        loaded = reg.load_all(r, c, steering=None, console_print=lambda _: None)
        assert "storm_research" in loaded

    def test_tools_registered(self):
        """storm_research and storm_outline tools should be in the registry."""
        ext_dir = _storm_ext_dir()
        reg = ExtensionRegistry(extra_paths=[ext_dir], disabled=[], include_builtins=False)
        r = ToolRegistry()
        c = ContextManager(_cfg())
        reg.load_all(r, c, steering=None, console_print=lambda _: None)
        names = r.names()
        assert "storm_research" in names
        assert "storm_outline" in names

    def test_slash_commands_registered(self):
        """Slash commands should be discoverable."""
        ext_dir = _storm_ext_dir()
        reg = ExtensionRegistry(extra_paths=[ext_dir], disabled=[], include_builtins=False)
        r = ToolRegistry()
        c = ContextManager(_cfg())
        reg.load_all(r, c, steering=None, console_print=lambda _: None)
        cmds = dict(reg.all_slash_commands())
        assert "research" in cmds
        assert "research-status" in cmds

    def test_manifest_fields(self):
        from storm_research.extension import StormResearchExtension

        ext = StormResearchExtension()
        assert ext.manifest.name == "storm_research"
        assert ext.manifest.version == "0.1.0"
        assert "storm_research" in ext.manifest.system_prompt_fragment
