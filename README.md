# Semantic Scholar MCP

An MCP server for Semantic Scholar, optimized for AI agents to perform deep literature reviews without exceeding token or API limits.

## Features
- **Drill-Down and Bulk Search**: Extreme token savings for large discovery queries.
- **Intelligent Local Cache (SQLite)**: Persists across short-lived `uvx` restarts (stored in `~/.semantic_scholar_mcp/`).
- **Snowballing API**: Advanced citation and reference discovery.
- **Smart PDF Downloading**: Open-access checks and `content-type` validation.
- **Graceful Degradation**: Strict request limits when no API key is present to avoid being blocked.

## How to Run
1. Initialize a git repository and push this code.
2. (Optional but recommended) Set the environment variable:
   `export SEMANTIC_SCHOLAR_API_KEY="your_api_key_here"`
3. Run via `uvx`:
   `uvx --from git+https://github.com/your-user/your-repo semantic-scholar-mcp`

## Config 
```json
"smart-semantic-scholar-mcp": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/spideryzarc/smart-semantic-scholar-mcp",
        "smart-semantic-scholar-mcp"
      ],
      "env": {
        "SEMANTIC_SCHOLAR_API_KEY": "my_API_key"
      }
    }
```