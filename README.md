# tau-storm

Deep research extension for [tau](../tau) — generates comprehensive, long-form articles with citations using the **STORM** algorithm.

Based on the paper: [*"Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models"*](https://arxiv.org/abs/2402.14207) (Shao et al., NAACL 2024).

## What it does

Given a topic, tau-storm:
1. **Discovers diverse perspectives** on the topic
2. **Simulates multi-perspective research conversations** grounded in web search
3. **Generates a structured outline** from collected information
4. **Writes each section** with inline citations [1], [2], ...
5. **Polishes** with a lead summary

Zero dependency on the original `knowledge-storm` package — fully rewritten using tau's provider system.

## Installation

```bash
tau install git:https://github.com/<your-org>/tau-storm
```

Or add the path to your `~/.tau/config.toml`:

```toml
[extensions]
paths = ["/path/to/tau-storm/extensions"]
```

## Requirements

Install a search backend (at least one):

```bash
# Free (no API key needed)
pip install duckduckgo-search

# Or use Tavily (higher quality, requires API key)
export TAVILY_API_KEY=your-key-here
```

## Usage

### As a tool (the agent uses it automatically)
The agent will call `storm_research` when you ask for deep research:
> "Research quantum computing and write me a comprehensive article"

### Via slash command
```
/research Quantum Computing
/research-status
```

### Tools available
| Tool | Description |
|------|-------------|
| `storm_research` | Full pipeline → article with citations (1-3 min) |
| `storm_outline` | Pre-writing stage → outline + references (faster) |

## Output

Research articles are saved to `.tau/research/` in the workspace:
- `<topic>.md` — the article in Markdown
- `<topic>_data.json` — raw conversations and references

## Running tests

```bash
cd tau-storm
python -m pytest tests/ -v
```
