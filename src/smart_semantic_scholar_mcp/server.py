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
MCP_DIR = Path(os.environ.get("MCP_CACHE_DIR", Path.home() / ".semantic_scholar_mcp"))
MCP_DIR.mkdir(parents=True, exist_ok=True)
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

API_BASE = "https://api.semanticscholar.org/graph/v1"

def get_api_config():
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        return {"headers": {"x-api-key": api_key}, "warning": ""}
    return {
        "headers": {}, 
        "warning": "[SYSTEM WARNING: MCP running without API key. Queries intentionally slowed to avoid rate limits (HTTP 429). Recommend setting SEMANTIC_SCHOLAR_API_KEY in the environment.]\n\n"
    }

_limiters = {}
def get_rate_limiter():
    loop = asyncio.get_running_loop()
    if loop not in _limiters:
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            _limiters[loop] = AsyncLimiter(100, 60)
        else:
            _limiters[loop] = AsyncLimiter(1, 4)
    return _limiters[loop]

_semaphores = {}
def get_semaphore():
    loop = asyncio.get_running_loop()
    if loop not in _semaphores:
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if api_key:
            _semaphores[loop] = asyncio.Semaphore(10)
        else:
            _semaphores[loop] = asyncio.Semaphore(1)
    return _semaphores[loop]

# Inicialização do FastMCP
mcp = FastMCP("semantic-scholar")

# Utility: API fetch with rate limiting and concurrency backoff
async def fetch_api(client: httpx.AsyncClient, method: str, endpoint: str, **kwargs):
    config = get_api_config()
    max_retries = 3
    
    for attempt in range(max_retries + 1):
        async with get_semaphore():
            async with get_rate_limiter():
                url = f"{API_BASE}{endpoint}"
                response = await client.request(method, url, headers=config["headers"], timeout=15.0, **kwargs)
                
                if response.status_code == 429 and attempt < max_retries:
                    # Se fomos bloqueados (429), aguardamos e tentamos novamente
                    await asyncio.sleep(5 * (2 ** attempt))
                    continue
                    
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
            config = get_api_config()
            data = await fetch_api(client, "GET", "/paper/search", params=params)
            papers = data.get("data", [])
            
            # Pre-cache for future use
            save_cached({p["paperId"]: p for p in papers if "paperId" in p})
            return config["warning"] + json.dumps(papers, indent=2)
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
    config = get_api_config()
    return config["warning"] + json.dumps(results, indent=2)

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
            papers = data.get("data") or []
            
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
            
            config = get_api_config()
            return config["warning"] + json.dumps(filtered, indent=2)
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
            
            config = get_api_config()
            return config["warning"] + json.dumps(data, indent=2)
        except Exception as e:
            return f"Error fetching author: {str(e)}"

import re
import urllib.parse

async def extract_direct_pdf_url(url: str, html_text: str = None, client: httpx.AsyncClient = None) -> str | None:
    """
    Tenta extrair um link direto para o PDF a partir de uma URL de landing page acadêmica.
    Aplica regras de reescrita de URL ou busca pela meta tag 'citation_pdf_url' no HTML.
    
    Retorna:
        str: A URL direta para o PDF, ou None se não for possível extrair.
    """
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc.lower()
    
    # Tratamento ArXiv
    if "arxiv.org" in domain and "/abs/" in parsed_url.path:
        return url.replace("/abs/", "/pdf/") + ".pdf"
        
    # Tratamento BioRxiv / MedRxiv
    if "biorxiv.org" in domain or "medrxiv.org" in domain:
        if "/content/" in parsed_url.path and not parsed_url.path.endswith(".full.pdf"):
            return url + ".full.pdf"
            
    # Tratamento OpenReview
    if "openreview.net" in domain and "/forum" in parsed_url.path:
        return url.replace("/forum", "/pdf")

    # Se o HTML não foi passado previamente, tenta baixar
    if not html_text and client:
        try:
            response = await client.get(url, timeout=10.0, follow_redirects=True)
            if response.status_code == 200:
                html_text = response.text
        except Exception:
            return None

    if html_text:
        match = re.search(
            r'<meta[^>]*?(?:name|property)=["\']citation_pdf_url["\'][^>]*?content=["\']([^"\']+)["\']', 
            html_text, 
            re.IGNORECASE
        )
        if not match:
            match = re.search(
                r'<meta[^>]*?content=["\']([^"\']+)["\'][^>]*?(?:name|property)=["\']citation_pdf_url["\']', 
                html_text, 
                re.IGNORECASE
            )
            
        if match:
            extracted_path = match.group(1)
            return urllib.parse.urljoin(url, extracted_path)

    return None

async def _download_direct_pdf(url: str, paper_id: str, save_directory: str, client: httpx.AsyncClient) -> str | None:
    """Helper interno para baixar e salvar o PDF a partir de uma URL direta."""
    try:
        async with client.stream("GET", url, timeout=15.0) as response:
            if response.status_code == 200 and "application/pdf" in response.headers.get("content-type", "").lower():
                content = await response.aread()
                save_path = Path(save_directory) if save_directory else MCP_DIR / f"{paper_id}.pdf"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(content)
                return f"SUCCESS: PDF successfully downloaded to -> {save_path}"
    except Exception:
        pass
    return None

@mcp.tool()
async def fetch_pdf(paper_id: str, save_directory: str = None) -> str:
    """Attempts to download the PDF via open access or returns a manual/Google Scholar URL."""
    cached = get_cached([paper_id])
    paper = cached.get(paper_id, {})
    
    if "isOpenAccess" not in paper or "title" not in paper:
        async with httpx.AsyncClient() as client:
            try:
                params = {"fields": "paperId,title,url,externalIds,isOpenAccess,openAccessPdf"}
                data = await fetch_api(client, "GET", f"/paper/{paper_id}", params=params)
                save_cached({paper_id: data})
                paper = data
            except Exception as e:
                return f"API error: {str(e)}"
                
    title = paper.get("title", f"paper {paper_id}")
    google_scholar_url = f"https://scholar.google.com/scholar?q={urllib.parse.quote_plus(title)}"
    
    # Busca por URLs alternativas (ex: landing page, publisher, arxiv no disclaimer)
    # Domínios que são endpoints de API de metadados, não páginas reais de papers
    BLACKLISTED_DOMAINS = ["api.unpaywall.org", "api.crossref.org", "api.openalex.org"]
    
    def _is_useful_url(u: str) -> bool:
        try:
            domain = urllib.parse.urlparse(u).netloc.lower()
            return not any(bl in domain for bl in BLACKLISTED_DOMAINS)
        except Exception:
            return False

    alternative_urls = []
    if paper.get("url"):
        alternative_urls.append(f"Semantic Scholar Page: {paper['url']}")
        
    pdf_info = paper.get("openAccessPdf") or {}
    disclaimer = pdf_info.get("disclaimer", "")
    urls_in_disclaimer = []
    if disclaimer:
        urls_in_disclaimer = [u for u in re.findall(r'https?://[^\s,]+', disclaimer) if _is_useful_url(u)]
        if urls_in_disclaimer:
            alternative_urls.append(f"Publisher/Source Page: {urls_in_disclaimer[0]}")
            
    alternatives_text = "\n".join(alternative_urls)
    if alternatives_text:
        alternatives_text += f"\nAlternative search (Google Scholar): {google_scholar_url}"
    else:
        alternatives_text = f"Alternative search (Google Scholar): {google_scholar_url}"
    
    target_urls_to_try = []
    
    # DOI é a fonte mais confiável — prioridade máxima
    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI")
    if doi:
        target_urls_to_try.append(f"https://doi.org/{doi}")
    
    if pdf_info and pdf_info.get("url") and _is_useful_url(pdf_info["url"]):
        target_urls_to_try.append(pdf_info["url"])
    if urls_in_disclaimer:
        target_urls_to_try.append(urls_in_disclaimer[0])

    # Deduplicar mantendo a ordem
    seen = set()
    target_urls_to_try = [u for u in target_urls_to_try if not (u in seen or seen.add(u))]

    if not target_urls_to_try:
        if not paper.get("isOpenAccess"):
            return f"LANDING_PAGE: Direct PDF link not found by Semantic Scholar.\n{alternatives_text}"
        else:
            return f"NOT_FOUND: Paper is open access but no official link on Semantic Scholar.\n{alternatives_text}"
            
    url = target_urls_to_try[0]
    
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
                    # É uma landing page HTML. Vamos tentar extrair o PDF direto!
                    html_text = await response.aread()
                    html_str = html_text.decode('utf-8', errors='ignore')
                    
                    direct_url = await extract_direct_pdf_url(url, html_str, client)
                    if direct_url:
                        download_result = await _download_direct_pdf(direct_url, paper_id, save_directory, client)
                        if download_result:
                            return download_result
                            
                    # Se mesmo com a tentativa extra não achou
                    return f"MANUAL_DOWNLOAD: The link is an HTML landing page.\nPlease download manually: {url}\n{alternatives_text}"
    except Exception as e:
        return f"MANUAL_DOWNLOAD: Automated connection failed ({str(e)}).\nTry accessing: {url}"

def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
