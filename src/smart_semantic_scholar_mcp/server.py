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

# FastMCP initialization
mcp = FastMCP(
    "semantic-scholar",
    instructions=(
        "Semantic Scholar research assistant.\n"
        "Workflow: 1) search_literature_broad (get IDs) -> 2) get_papers_batch (get full details) -> "
        "3) trace_citations_snowball / get_recommended_papers (expand search) -> "
        "4) fetch_pdf (download) -> 5) export_citations_bibtex (format references).\n\n"
        "CRITICAL RULES:\n"
        "- NEVER fabricate paperIds or authorIds. Only use exact IDs from prior tool outputs.\n"
        "- 'citationCount' = impact proxy. 'tldr' = AI 1-sentence summary.\n"
        "- Ignore 'SYSTEM WARNING' prefixes in outputs; they only indicate automatic rate limiting."
    )
)

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
                    # Rate limited (429) — wait and retry with exponential backoff
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
            
            # Cache enrichment: merge new attributes without overwriting existing ones
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
    """
    Search for papers matching a natural-language query. 
    Returns lightweight metadata (paperId, title, year, citationCount). Use get_papers_batch for details.
    
    Args:
        query: Free-text search query.
        year_range: Optional filter (e.g., "2018-2023" or "2022").
        limit: Max results (1-100, default 10).
    """
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
    """
    Fetch full details (abstract, tldr, authors, openAccessPdf) for specific papers.
    
    Args:
        paper_ids: List of Semantic Scholar paperIds (max 500).
    """
    cached = get_cached(paper_ids)
    
    missing_ids = []
    for pid in paper_ids:
        # Partial cache hit validation: check if we already have deep data
        if pid not in cached or ("abstract" not in cached[pid] and "tldr" not in cached[pid]):
            missing_ids.append(pid)
            
    if missing_ids:
        async with httpx.AsyncClient() as client:
            try:
                # Semantic Scholar bulk endpoint is limited to 500 IDs at a time.
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
    """
    Follow the citation graph to expand a search.
    
    Args:
        paper_id: Seed Semantic Scholar paperId.
        direction: 'forward' (papers citing this) or 'backward' (papers this cites).
        min_citations: Minimum citations threshold to filter noise. Default 10.
    """
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
    """
    Retrieve author profile metrics and their top 5 most-cited works.
    
    Args:
        author_id: Semantic Scholar authorId.
    """
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
    Attempts to extract a direct PDF link from an academic landing page URL.
    Applies URL rewrite rules or searches for the 'citation_pdf_url' meta tag in HTML.

    Returns:
        str: The direct URL to the PDF, or None if extraction is not possible.
    """
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc.lower()
    
    # ArXiv handling
    if "arxiv.org" in domain and "/abs/" in parsed_url.path:
        return url.replace("/abs/", "/pdf/") + ".pdf"
        
    # BioRxiv / MedRxiv handling
    if "biorxiv.org" in domain or "medrxiv.org" in domain:
        if "/content/" in parsed_url.path and not parsed_url.path.endswith(".full.pdf"):
            return url + ".full.pdf"
            
    # OpenReview handling
    if "openreview.net" in domain and "/forum" in parsed_url.path:
        return url.replace("/forum", "/pdf")

    # If HTML was not provided, attempt to fetch it
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
    """Internal helper to download and save a PDF from a direct URL."""
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
    """
    Attempt to download the full-text PDF of an open-access paper.
    
    Args:
        paper_id: Semantic Scholar paperId.
        save_directory: Optional absolute path to save the PDF.
        
    Returns:
        Status string (SUCCESS, BLOCKED, MANUAL_DOWNLOAD, etc.). 
        Always present any returned manual URLs to the user if automated download fails.
    """
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
    
    # Look for alternative URLs (e.g., landing page, publisher, arxiv in disclaimer)
    # Domains that are metadata API endpoints, not real paper pages
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
    
    # DOI is the most reliable source — highest priority
    external_ids = paper.get("externalIds") or {}
    doi = external_ids.get("DOI")
    if doi:
        target_urls_to_try.append(f"https://doi.org/{doi}")
    
    if pdf_info and pdf_info.get("url") and _is_useful_url(pdf_info["url"]):
        target_urls_to_try.append(pdf_info["url"])
    if urls_in_disclaimer:
        target_urls_to_try.append(urls_in_disclaimer[0])

    # Deduplicate while preserving order
    seen = set()
    target_urls_to_try = [u for u in target_urls_to_try if not (u in seen or seen.add(u))]

    if not target_urls_to_try:
        if not paper.get("isOpenAccess"):
            return f"LANDING_PAGE: Direct PDF link not found by Semantic Scholar.\n{alternatives_text}"
        else:
            return f"NOT_FOUND: Paper is open access but no official link on Semantic Scholar.\n{alternatives_text}"
            
    url = target_urls_to_try[0]
    
    # Best-effort validation (checks headers without downloading the full body)
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
                    # It's an HTML landing page — attempt to extract the direct PDF link
                    html_text = await response.aread()
                    html_str = html_text.decode('utf-8', errors='ignore')
                    
                    direct_url = await extract_direct_pdf_url(url, html_str, client)
                    if direct_url:
                        download_result = await _download_direct_pdf(direct_url, paper_id, save_directory, client)
                        if download_result:
                            return download_result
                            
                    # Even after the extra attempt, no direct PDF was found
                    return f"MANUAL_DOWNLOAD: The link is an HTML landing page.\nPlease download manually: {url}\n{alternatives_text}"
    except Exception as e:
        return f"MANUAL_DOWNLOAD: Automated connection failed ({str(e)}).\nTry accessing: {url}"


def _add_semantic_scholar_id_to_bibtex(bibtex: str, paper_id: str) -> str:
    """Injects semantic_scholar_id into a BibTeX entry when it is missing."""
    if not bibtex or not paper_id:
        return bibtex

    if re.search(r"^\s*semantic_scholar_id\s*=", bibtex, flags=re.IGNORECASE | re.MULTILINE):
        return bibtex

    entry = bibtex.rstrip()
    field_line = f"  semantic_scholar_id = {{{paper_id}}},"

    if entry.endswith("}"):
        close_idx = entry.rfind("}")
        body = entry[:close_idx].rstrip()
        lines = body.splitlines()

        # Ensure the previous BibTeX field ends with a comma before appending a new field.
        for idx in range(len(lines) - 1, -1, -1):
            stripped = lines[idx].strip()
            if not stripped:
                continue
            if stripped.startswith("@"):
                break
            if not stripped.endswith(","):
                lines[idx] = lines[idx].rstrip() + ","
            break

        body = "\n".join(lines)
        return body + "\n" + field_line + "\n}"

    return entry + "\n" + field_line


@mcp.tool()
async def export_citations_bibtex(paper_ids: list[str]) -> str:
    """
    Generate BibTeX-formatted references for a list of papers.
    
    Args:
        paper_ids: List of Semantic Scholar paperIds.
    """
    # 1. Try to load from cache
    cached = get_cached(paper_ids)

    missing_ids = []
    for pid in paper_ids:
        # Check whether BibTeX already exists in the cached JSON object
        if pid not in cached or not cached[pid].get("citationStyles", {}).get("bibtex"):
            missing_ids.append(pid)

    # 2. For missing IDs, fetch from the API in bulk
    if missing_ids:
        async with httpx.AsyncClient() as client:
            try:
                # The /paper/batch endpoint supports the citationStyles field
                payload = {"ids": missing_ids[:500]}
                params = {"fields": "paperId,title,citationStyles"}
                data = await fetch_api(client, "POST", "/paper/batch", json=payload, params=params)

                new_data = {p["paperId"]
                    : p for p in data if p and "paperId" in p}
                save_cached(new_data)
                cached.update(new_data)
            except Exception as e:
                return f"Error fetching citations from API: {str(e)}"

    # 3. Extract and concatenate BibTeX entries
    bibtex_list = []
    for pid in paper_ids:
        paper = cached.get(pid)
        if paper and "citationStyles" in paper and "bibtex" in paper["citationStyles"]:
            bibtex = paper["citationStyles"]["bibtex"]
            bibtex_list.append(_add_semantic_scholar_id_to_bibtex(bibtex, pid))
        else:
            bibtex_list.append(f"% Citation not found for ID: {pid}")

    config = get_api_config()
    header = config["warning"] + "### Extracted BibTeX References ###\n\n"
    return header + "\n\n".join(bibtex_list)

@mcp.tool()
async def get_recommended_papers(positive_paper_ids: list[str], negative_paper_ids: list[str] = None, limit: int = 10) -> str:
    """
    Discover semantically similar papers using AI recommendations, ignoring strict keyword overlap.
    
    Args:
        positive_paper_ids: List of 1-5 paperIds representing highly relevant papers.
        negative_paper_ids: Optional list of paperIds representing off-topic papers to exclude.
        limit: Max recommendations to return (default 10).
    """
    if not positive_paper_ids:
        return "Error: You must provide at least one paper ID in the positive_paper_ids list."
        
    negative_paper_ids = negative_paper_ids or []
    
    # Recommendations API URL (differs from the Graph API)
    url = "https://api.semanticscholar.org/recommendations/v1/papers/"
    
    payload = {
        "positivePaperIds": positive_paper_ids,
        "negativePaperIds": negative_paper_ids
    }
    
    params = {
        "fields": "paperId,title,year,citationCount,authors,venue,isOpenAccess",
        "limit": limit
    }
    
    config = get_api_config()
    max_retries = 3
    
    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries + 1):
            try:
                # Use the same rate-limiting controls (Semaphore/Limiter) to respect graceful degradation
                async with get_semaphore():
                    async with get_rate_limiter():
                        response = await client.post(url, json=payload, params=params, headers=config["headers"], timeout=15.0)
                        
                        if response.status_code == 429 and attempt < max_retries:
                            await asyncio.sleep(5 * (2 ** attempt))
                            continue
                            
                        response.raise_for_status()
                        data = response.json()
                        
                        # The recommendations API returns results under the 'recommendedPapers' key
                        recommendations = data.get("recommendedPapers", [])

                        # Cache the discovered metadata for future access
                        new_papers = {p["paperId"]: p for p in recommendations if "paperId" in p}
                        save_cached(new_papers)
                        
                        return config["warning"] + json.dumps(recommendations, indent=2)
            except Exception as e:
                if attempt == max_retries:
                    return f"Error fetching semantic recommendations from API: {str(e)}"

def main():
    mcp.run(transport='stdio')

if __name__ == "__main__":
    main()
