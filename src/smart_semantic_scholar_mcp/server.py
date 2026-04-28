import os
import sys
import json
import sqlite3
import asyncio
import urllib.parse
from pathlib import Path
import httpx
from aiolimiter import AsyncLimiter
from mcp.server.fastmcp import FastMCP

# Absolute cache directory settings
HOME = Path.home()
MCP_DIR = HOME / ".semantic_scholar_mcp"
MCP_DIR.mkdir(exist_ok=True)
DB_PATH = MCP_DIR / "papers_cache.sqlite"

# Initialize the secure database (WAL mode for concurrency)
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                data TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

init_db()

# API and rate limiting settings (graceful degradation)
API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
API_BASE = "https://api.semanticscholar.org/graph/v1"

if API_KEY:
    rate_limiter = AsyncLimiter(100, 60)  # 100 requests per 60 seconds
    concurrency_semaphore = asyncio.Semaphore(10)
    HEADERS = {"x-api-key": API_KEY}
    SYSTEM_WARNING = ""
else:
    rate_limiter = AsyncLimiter(1, 4)  # 1 request every 4 seconds (safe fallback)
    concurrency_semaphore = asyncio.Semaphore(1)
    HEADERS = {}
    SYSTEM_WARNING = "[SYSTEM WARNING: MCP running without API key. Queries intentionally slowed to avoid rate limits (HTTP 429). Recommend setting SEMANTIC_SCHOLAR_API_KEY in the environment.]\n\n"

# Inicialização do FastMCP
mcp = FastMCP("semantic-scholar")

# Utility: API fetch with rate limiting and concurrency backoff
async def fetch_api(client: httpx.AsyncClient, method: str, endpoint: str, **kwargs):
    async with concurrency_semaphore:
        async with rate_limiter:
            url = f"{API_BASE}{endpoint}"
            response = await client.request(method, url, headers=HEADERS, timeout=15.0, **kwargs)
            response.raise_for_status()
            return response.json()

# Database utilities
def get_cached(paper_ids: list[str]) -> dict:
    if not paper_ids: return {}
    results = {}
    with sqlite3.connect(DB_PATH) as conn:
        placeholders = ','.join(['?'] * len(paper_ids))
        query = f"SELECT paper_id, data FROM papers WHERE paper_id IN ({placeholders})"
        for row in conn.execute(query, paper_ids):
            results[row[0]] = json.loads(row[1])
    return results

def save_cached(papers: dict):
    if not papers: return
    with sqlite3.connect(DB_PATH) as conn:
        for pid, data in papers.items():
            existing = {}
            cur = conn.execute("SELECT data FROM papers WHERE paper_id = ?", (pid,))
            row = cur.fetchone()
            if row:
                existing = json.loads(row[0])
            
            # Cache Enrichment (Merge inteligente de atributos novos sem apagar os antigos)
            # Cache enrichment: merge new attributes without deleting old ones
            existing.update(data)
            conn.execute(
                "INSERT OR REPLACE INTO papers (paper_id, data, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                (pid, json.dumps(existing))
            )

# =======================================================
# TOOLS EXPOSED TO THE AGENT
# =======================================================

@mcp.tool()
async def search_literature_broad(query: str, year_range: str = None, limit: int = 10) -> str:
    """Initial broad search for discovering papers. Returns only IDs and basic metadata to save tokens."""
    params = {"query": query, "limit": limit, "fields": "paperId,title,year,citationCount,venue"}
    if year_range:
        params["year"] = year_range
        
    async with httpx.AsyncClient() as client:
        try:
            data = await fetch_api(client, "GET", "/paper/search", params=params)
            papers = data.get("data", [])
            
            # Pre-cache for future use
            save_cached({p["paperId"]: p for p in papers if "paperId" in p})
            return SYSTEM_WARNING + json.dumps(papers, indent=2)
        except Exception as e:
            return f"Error during search: {str(e)}"

@mcp.tool()
async def get_papers_batch(paper_ids: list[str]) -> str:
    """Fetches details for multiple papers (abstract, tldr, authors). Processes in batch using the local cache."""
    cached = get_cached(paper_ids)
    
    missing_ids = []
    for pid in paper_ids:
        # Partial cache hit validation (verifica se já temos os dados profundos)
        if pid not in cached or ("abstract" not in cached[pid] and "tldr" not in cached[pid]):
            missing_ids.append(pid)
            
    if missing_ids:
        async with httpx.AsyncClient() as client:
            try:
                # Semantic scholar limita bulk a 500 ids. 
                payload = {"ids": missing_ids[:500]}
                params = {"fields": "paperId,title,abstract,tldr,authors,isOpenAccess,openAccessPdf"}
                data = await fetch_api(client, "POST", "/paper/batch", json=payload, params=params)
                
                new_papers = {p["paperId"]: p for p in data if p and "paperId" in p}
                save_cached(new_papers)
                cached.update(new_papers)
            except Exception as e:
                return f"Error fetching batch from API: {str(e)}"
                
    results = [cached.get(pid) for pid in paper_ids if cached.get(pid)]
    return SYSTEM_WARNING + json.dumps(results, indent=2)

@mcp.tool()
async def trace_citations_snowball(paper_id: str, direction: str = "forward", min_citations: int = 10) -> str:
    """Snowballing: direction='forward' (papers that cited the given paper) or 'backward' (references of the paper)."""
    endpoint_map = {"forward": "citations", "backward": "references"}
    if direction not in endpoint_map:
        return "Error: direction must be 'forward' or 'backward'."
        
    endpoint = f"/paper/{paper_id}/{endpoint_map[direction]}"
    params = {"fields": "paperId,title,citationCount,year", "limit": 1000}
    
    async with httpx.AsyncClient() as client:
        try:
            data = await fetch_api(client, "GET", endpoint, params=params)
            papers = data.get("data", [])
            
            key = "citingPaper" if direction == "forward" else "citedPaper"
            filtered = []
            
            for p in papers:
                p_data = p.get(key, {})
                if not p_data or not p_data.get("paperId"): continue
                
                c_count = p_data.get("citationCount", 0) or 0
                if c_count >= min_citations:
                    filtered.append({
                        "paperId": p_data["paperId"],
                        "title": p_data.get("title"),
                        "year": p_data.get("year"),
                        "citationCount": c_count
                    })
            
            # Sort by citation count (highest impact first)
            filtered = sorted(filtered, key=lambda x: x["citationCount"], reverse=True)
            save_cached({p["paperId"]: p for p in filtered})
            
            return SYSTEM_WARNING + json.dumps(filtered, indent=2)
        except Exception as e:
            return f"Error in snowballing: {str(e)}"

@mcp.tool()
async def generate_author_graph(author_id: str) -> str:
    """Maps an author's top works to understand their relevance."""
    params = {"fields": "authorId,name,paperCount,citationCount,papers.paperId,papers.title,papers.citationCount"}
    async with httpx.AsyncClient() as client:
        try:
            data = await fetch_api(client, "GET", f"/author/{author_id}", params=params)
            papers = data.get("papers", [])
            papers = sorted([p for p in papers if p.get("citationCount")], key=lambda x: x["citationCount"], reverse=True)[:5]
            data["top_papers"] = papers
            if "papers" in data: del data["papers"]
            
            return SYSTEM_WARNING + json.dumps(data, indent=2)
        except Exception as e:
            return f"Error fetching author: {str(e)}"

@mcp.tool()
async def fetch_pdf(paper_id: str, save_directory: str = None) -> str:
    """Attempts to download the PDF via open access or returns a manual/Google Scholar URL."""
    cached = get_cached([paper_id])
    paper = cached.get(paper_id, {})
    
    # Se os detalhes do pdf não estiverem no cache, busca na API
    if "isOpenAccess" not in paper or "title" not in paper:
        async with httpx.AsyncClient() as client:
            try:
                params = {"fields": "paperId,title,isOpenAccess,openAccessPdf"}
                data = await fetch_api(client, "GET", f"/paper/{paper_id}", params=params)
                save_cached({paper_id: data})
                paper = data
            except Exception as e:
                return f"API error: {str(e)}"
                
    title = paper.get("title", f"paper {paper_id}")
    google_scholar_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote_plus(title)}"
    
    if not paper.get("isOpenAccess"):
        return f"PAYWALL: The paper is behind a paywall.\nAlternative search (Google Scholar): {google_scholar_url}"
        
    pdf_info = paper.get("openAccessPdf")
    if not pdf_info or not pdf_info.get("url"):
        return f"NOT_FOUND: Paper is open access but no official link on Semantic Scholar.\nAlternative search: {google_scholar_url}"
        
    url = pdf_info["url"]
    
    # Validação Melhor Esforço (Verifica os Headers sem baixar tudo)
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        async with httpx.AsyncClient(follow_redirects=True, headers=headers) as client:
            async with client.stream("GET", url, timeout=10.0) as response:
                content_type = response.headers.get("content-type", "").lower()
                
                if response.status_code in [403, 503]:
                    return f"BLOCKED: Publisher anti-bot protection detected.\nURL: {url}\nAlternative search: {google_scholar_url}"
                    
                if "application/pdf" in content_type:
                    content = await response.aread()
                    save_path = Path(save_directory) if save_directory else MCP_DIR / f"{paper_id}.pdf"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    save_path.write_bytes(content)
                    return f"SUCCESS: PDF successfully downloaded to -> {save_path}"
                else:
                    return f"MANUAL_DOWNLOAD: The official link is an HTML landing page.\nPlease download manually: {url}"
    except Exception as e:
        return f"MANUAL_DOWNLOAD: Automated connection failed ({str(e)}).\nTry accessing: {url}"

def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
