# 🎓 Smart Semantic Scholar MCP

[![MCP](https://img.shields.io/badge/MCP-Ready-blue)](https://modelcontextprotocol.io/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)

An intelligent [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server for the **Semantic Scholar API**, specifically engineered for AI agents to conduct deep, rigorous academic literature reviews without exhausting token limits or hitting API rate blocks.

## 🧠 Why is it "Smart"?

Standard API wrappers simply pass data back and forth. This MCP acts as a stateful, intelligent bridge designed specifically for **autonomous agent workflows**:
- **Token Economy:** It splits discovery from deep-reading. Agents receive lightweight metadata first and selectively drill down into heavy abstracts, preserving valuable context windows.
- **Stateful Memory:** A local SQLite cache remembers every paper fetched. Redundant queries are served instantly locally, saving API quotas and drastically speeding up multi-turn agent conversations.
- **Self-Healing:** Built-in semaphores and asynchronous rate-limiters ensure that even without an API key, the server gracefully paces itself to prevent catastrophic `429 Too Many Requests` crashes.

## ✨ Key Features

- 🔍 **Drill-Down & Bulk Search Workflow**: Optimized for LLM contexts. Performs broad searches first (fetching only IDs and basic metadata), allowing agents to select relevant papers before downloading heavy abstracts and TL;DRs in bulk.
- 🧠 **Intelligent Local Caching**: Uses a robust SQLite WAL-mode database to cache paper metadata persistently across `uvx` restarts. Reduces redundant API calls, speeds up workflows, and saves your API quota.
- ❄️ **Citation Snowballing**: Advanced endpoint to trace a paper's citations (forward) and references (backward) to organically map the research landscape.
- 📊 **Author Impact Graphs**: Maps an author's top works and calculates their global citation impact.
- 📄 **Smart PDF Fetching**: Automatically detects Open Access status, performs `content-type` validation to avoid downloading HTML landing pages, and saves PDFs directly to your local machine.
- 🛡️ **Graceful Degradation & Rate Limiting**: Features built-in concurrency semaphores and asynchronous rate-limiting. Operates smoothly with or without an API key (automatically slowing down queries to avoid `429 Too Many Requests` blocks when unauthenticated).

## 🛠️ Available MCP Tools

Agents connected to this server have access to the following tools:

- `search_literature_broad`: Initial broad search for discovering papers. Returns only basic metadata to save tokens.
- `get_papers_batch`: Fetches deep details (abstract, tldr, authors, open access links) for multiple papers simultaneously using the local cache.
- `trace_citations_snowball`: Explores the citation graph. Use `direction='forward'` for papers that cited the target, or `'backward'` for its references.
- `generate_author_graph`: Analyzes an author's relevance and top works.
- `fetch_pdf`: Attempts to download the Open Access PDF of a paper directly to the local cache directory.

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

## 💻 Development & Testing

This project includes rigorous integration workflows to ensure robust API compatibility.

1. Clone the repository and navigate to the folder.
2. Create your virtual environment: `python -m venv .venv && source .venv/bin/activate`
3. Install dependencies: `pip install -e .` (and `pytest` for testing)
4. Create a `.env` file at the root with your `SEMANTIC_SCHOLAR_API_KEY`.
5. Run the tests:
   ```bash
   pytest tests/ -v
   ```
*(Tests output a detailed `inspection_log.md` with raw API responses to manually verify what your AI agent will receive.)*

## 📜 License
MIT