# 🎓 Smart Semantic Scholar MCP

[![MCP](https://img.shields.io/badge/MCP-Ready-blue)](https://modelcontextprotocol.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

An intelligent [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for the **Semantic Scholar API**, specifically engineered for AI agents to conduct deep, rigorous academic literature reviews without exhausting token limits or hitting API rate blocks.

## 🧠 Why is it "Smart"?

This server is designed as a stateful research layer for **autonomous agent workflows** over Semantic Scholar:
- **Discovery-first retrieval:** Agents start with lightweight search metadata and only fetch deep paper details when needed.
- **Persistent local memory:** A SQLite cache (WAL mode) stores fetched papers and citation metadata to reduce repeated API calls.
- **Graceful API behavior:** Built-in semaphores, async rate limiting, and retry/backoff help avoid hard failures under throttling.
- **Agent-facing guidance:** The MCP server includes internal instructions describing Semantic Scholar identifiers (`paperId`, `authorId`), recommended workflow, and tool usage constraints.

## ✨ Key Features

- 🔍 **Broad Search + Deep Fetch Workflow**: Search first for compact metadata, then fetch full paper details in batch.
- 🧠 **SQLite Cache with Enrichment**: Cached records are merged with newly fetched fields, preserving previously known attributes.
- ❄️ **Citation Snowballing**: Explore forward citations and backward references from a seed paper.
- 📊 **Author Profiling**: Retrieve an author profile and top-cited papers.
- 🧬 **Semantic Recommendations**: Retrieve related papers using positive and optional negative paper examples.
- 📄 **Smart PDF Retrieval**: Resolve direct PDF links from open-access metadata, DOI redirects, and supported landing-page patterns.
- 🧾 **BibTeX Export**: Generate reference-ready BibTeX entries from one or more paper IDs.
- 🛡️ **Authenticated/Unauthenticated Operation**: Runs with or without API key, applying stricter throttling in unauthenticated mode.

## 🛠️ Available MCP Tools

Agents connected to this server have access to the following tools:

- `search_literature_broad`: Broad query over Semantic Scholar with lightweight fields (`paperId`, `title`, `year`, `citationCount`, `venue`).
- `get_papers_batch`: Batch fetch of detailed paper metadata (`abstract`, `tldr`, `authors`, `isOpenAccess`, `openAccessPdf`).
- `trace_citations_snowball`: Citation graph traversal in forward or backward direction with minimum citation filtering.
- `generate_author_graph`: Author profile lookup with top-cited works.
- `fetch_pdf`: Best-effort PDF download workflow with fallback guidance when direct retrieval is unavailable.
- `export_citations_bibtex`: BibTeX export for a list of paper IDs using cache + API fallback.
- `get_recommended_papers`: Semantic-paper recommendation endpoint using positive and optional negative paper IDs.

## 🚀 Quick Start

### 1. API Key (Recommended)
While the server works without an API key, we highly recommend getting a free Semantic Scholar API key to significantly increase your rate limits.
Get one here: [Semantic Scholar API](https://www.semanticscholar.org/product/api)

### 2. Configuration for Claude Desktop

Add the following to your `claude_desktop_config.json` (or your preferred MCP client config):

```json
{
  "mcpServers": {
    "smart-semantic-scholar-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/spideryzarc/smart-semantic-scholar-mcp",
        "smart-semantic-scholar-mcp"
      ],
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": "your_api_key_here"
      }
    }
  }
}
```

*Note: You can optionally add `"MCP_CACHE_DIR": "/path/to/custom/dir"` inside the `env` block to change where the SQLite cache and downloaded PDFs are stored. If omitted, it defaults to `~/.semantic_scholar_mcp/`.*

### 3. Runtime Behavior

- Cache directory: `MCP_CACHE_DIR` (default: `~/.semantic_scholar_mcp/`)
- Cache database: `papers_cache.sqlite` in WAL mode
- API key variable: `SEMANTIC_SCHOLAR_API_KEY`
- Without API key: server uses stricter throttling and prepends a system warning in tool outputs

## 💻 Development & Testing

This project includes integration workflows for live Semantic Scholar interactions.

1. Clone the repository and navigate to the folder.
2. Create your virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -e .`
4. Create a `.env` file at the root with your `SEMANTIC_SCHOLAR_API_KEY`.
5. Run the tests:
   ```bash
   pytest tests/ -v
   ```
*(Integration tests write `tests/inspection_log.md` with full tool responses for manual inspection.)*

Entry point script:
- `smart-semantic-scholar-mcp = smart_semantic_scholar_mcp.server:main`

## 📜 License
MIT