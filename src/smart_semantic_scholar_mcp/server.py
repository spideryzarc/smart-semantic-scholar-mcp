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
        "You are a research assistant with access to Semantic Scholar, a free academic search engine "
        "and knowledge graph covering over 200 million scientific papers across all fields of study. "
        "Papers are identified by a unique 'paperId' string (e.g. '204e3073870fae3d05bcbc2f6a8e263d9b72e776'). "
        "Authors are identified by a unique 'authorId' string (e.g. '1741101'). "
        "\n\n"
        "## Recommended research workflow\n"
        "1. Start with search_literature_broad to discover candidate papers and obtain their paperIds. "
        "   This returns lightweight metadata only (title, year, venue, citationCount).\n"
        "2. Use get_papers_batch to fetch full details (abstract, tldr, authors) for the papers you care about.\n"
        "3. Use trace_citations_snowball to expand coverage via forward (who cited this paper?) or backward "
        "   (what does this paper cite?) snowballing.\n"
        "4. Use generate_author_graph to profile a key author and surface their most-cited works.\n"
        "5. Use get_recommended_papers to discover thematically related papers that keyword search may have missed.\n"
        "6. Use fetch_pdf to retrieve the full text of open-access papers.\n"
        "7. Use export_citations_bibtex to produce ready-to-use BibTeX entries for a final reference list.\n"
        "\n"
        "## Key concepts\n"
        "- citationCount: number of times a paper has been cited — a strong proxy for impact and relevance.\n"
        "- tldr: a one-sentence AI-generated summary of the paper. Prefer this over the abstract when skimming.\n"
        "- isOpenAccess: if true, the full text is legally available for free download.\n"
        "- openAccessPdf.url: direct link to the PDF when available.\n"
        "- venue: the conference or journal where the paper was published.\n"
        "\n"
        "## Important constraints\n"
        "- Always obtain paperIds from search results or user input before calling batch/detail tools.\n"
        "- Do not fabricate or guess paperIds — they must come from API results.\n"
        "- Results are returned as JSON strings. Parse them before reasoning about their content.\n"
        "- A SYSTEM WARNING prefix in a response means the server is running without an API key and "
        "queries are being rate-limited automatically. This does not indicate an error."
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
    Search Semantic Scholar for papers matching a natural-language query.
    Use this as the FIRST step in any research task to discover candidate papers and obtain their paperIds.

    Returns lightweight metadata only (paperId, title, year, citationCount, venue) to keep token usage low.
    After identifying relevant papers here, use get_papers_batch to fetch full details.

    Args:
        query: Free-text search query (e.g. "transformer models for protein folding").
        year_range: Optional publication year filter. Format: "2018-2023" for a range, or "2022" for a
                    single year. Omit to search all years.
        limit: Maximum number of results to return (1-100). Default is 10. Use a small value for
               focused exploration; increase to 50-100 for exhaustive coverage.

    Returns:
        JSON array of paper objects, each with: paperId, title, year, citationCount, venue.
        Results are pre-cached locally — calling get_papers_batch on these IDs is fast.
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
    Fetch full details for one or more papers by their Semantic Scholar paperIds.
    Use this AFTER search_literature_broad or trace_citations_snowball to get abstracts, TLDRs, and author lists.

    This tool uses a local SQLite cache — papers already fetched are returned instantly without an API call.
    Up to 500 IDs can be requested in a single call (the API bulk endpoint limit).

    Args:
        paper_ids: List of Semantic Scholar paperIds to fetch
                   (e.g. ["204e3073870fae3d05bcbc2f6a8e263d9b72e776"]).

    Returns:
        JSON array of paper objects, each with:
          - paperId, title, abstract: full bibliographic data
          - tldr: AI-generated one-sentence summary (prefer this when skimming)
          - authors: list of {authorId, name}
          - isOpenAccess: boolean — if true, the PDF can be fetched with fetch_pdf
          - openAccessPdf: {url, status} — direct link to the PDF if available
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
    Expand a literature review by following the citation graph of a known paper (citation snowballing).

    Two directions are supported:
      - 'forward': find papers that CITED this paper (who built upon this work?).
        Best for finding recent developments and downstream applications.
      - 'backward': find papers this paper REFERENCES (what is this work's foundation?).
        Best for tracing foundational/seminal works in a field.

    Only papers with at least `min_citations` citations are returned, filtering out low-impact noise.
    Results are sorted by citationCount descending (highest impact first).

    Args:
        paper_id: The Semantic Scholar paperId of the seed paper.
        direction: 'forward' (papers that cite this one) or 'backward' (papers this one cites).
                   Default is 'forward'.
        min_citations: Minimum citation count a paper must have to appear in results.
                       Increase this (e.g. 50-100) for highly-cited seed papers to reduce noise.
                       Default is 10.

    Returns:
        JSON array of paper objects sorted by citationCount descending, each with:
        paperId, title, year, citationCount.
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
    Retrieve the profile and top-cited works of a Semantic Scholar author.
    Use this to assess an author's expertise, total output, and influence before deciding how much
    weight to give their papers in a literature review.

    The authorId can be found in the 'authors' field returned by get_papers_batch.

    Args:
        author_id: The Semantic Scholar authorId string (e.g. "1741101").

    Returns:
        JSON object with:
          - authorId, name
          - paperCount: total number of papers indexed on Semantic Scholar
          - citationCount: total citations across all their papers
          - top_papers: list of up to 5 most-cited papers, each with paperId, title, citationCount
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
    Attempt to download the full-text PDF of a paper.
    Prioritises the DOI, then the Semantic Scholar open-access link, then any URL found in the
    openAccessPdf disclaimer. For ArXiv, BioRxiv/MedRxiv, and OpenReview papers, URL rewriting is
    applied automatically to resolve abstract pages to direct PDF links.

    If automated download is not possible, a set of manual fallback URLs is returned.

    Args:
        paper_id: The Semantic Scholar paperId of the paper to download.
        save_directory: Optional absolute path to the directory where the PDF should be saved.
                        If omitted, the file is saved to the MCP cache directory (~/.semantic_scholar_mcp/).

    Returns:
        One of the following status strings:
          - 'SUCCESS: PDF successfully downloaded to -> <path>' — file saved on disk.
          - 'BLOCKED: Publisher anti-bot protection detected.' — publisher rejected automated access.
          - 'MANUAL_DOWNLOAD: ...' — automated download failed; manual URLs are provided.
          - 'LANDING_PAGE: ...' — no open-access link found; manual URLs are provided.
          - 'NOT_FOUND: ...' — paper is open access but Semantic Scholar has no link.
        When a manual URL is included, always present it to the user so they can download it themselves.
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


@mcp.tool()
async def export_citations_bibtex(paper_ids: list[str]) -> str:
    """
    Generate a BibTeX-formatted reference block for a list of papers.
    Use this as the FINAL step of a research task to produce a ready-to-use bibliography.

    BibTeX data is fetched from the Semantic Scholar API and cached locally.
    Calling this on IDs already retrieved earlier in the session is fast (cache hit).

    Args:
        paper_ids: List of Semantic Scholar paperIds for which to generate BibTeX entries.

    Returns:
        A plain-text block containing one BibTeX entry per paper, separated by blank lines,
        prefixed with the header '### Extracted BibTeX References ###'.
        If a BibTeX entry is unavailable for a given ID, a BibTeX comment placeholder is included
        (e.g. '% Citation not found for ID: <id>') so the output remains valid BibTeX.
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
            bibtex_list.append(paper["citationStyles"]["bibtex"])
        else:
            bibtex_list.append(f"% Citation not found for ID: {pid}")

    config = get_api_config()
    header = config["warning"] + "### Extracted BibTeX References ###\n\n"
    return header + "\n\n".join(bibtex_list)

@mcp.tool()
async def get_recommended_papers(positive_paper_ids: list[str], negative_paper_ids: list[str] = None, limit: int = 10) -> str:
    """
    Discover semantically similar papers using Semantic Scholar's AI-powered recommendation engine.

    Unlike keyword search, this tool works by understanding the *meaning* of papers, making it
    ideal for finding related work that uses different terminology, comes from adjacent fields,
    or solves the same problem with a different approach.

    Provide 1-5 papers you consider highly relevant as 'positive' examples. Optionally provide
    papers you consider off-topic as 'negative' examples to steer results away from unwanted themes.

    Args:
        positive_paper_ids: List of 1-5 Semantic Scholar paperIds representing papers that are
                            good examples of what you are looking for. These are used as the
                            semantic anchor for the recommendation query.
        negative_paper_ids: Optional list of Semantic Scholar paperIds for papers that are
                            NOT a good match. Use this to exclude a specific sub-field or approach.
                            Defaults to empty (no negative guidance).
        limit: Maximum number of recommendations to return (1-500). Default is 10.

    Returns:
        JSON array of recommended paper objects, each with:
        paperId, title, year, citationCount, authors, venue, isOpenAccess.
        Use get_papers_batch on any of these paperIds to retrieve full abstracts and TLDRs.
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
