from dotenv import load_dotenv
import os
import sys
from pathlib import Path

# Ensure the local `src/` package is importable when running tests from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import asyncio
import pytest
from smart_semantic_scholar_mcp import server

load_dotenv()



API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
if not API_KEY:
    pytest.skip("Requires SEMANTIC_SCHOLAR_API_KEY environment variable for live API tests", allow_module_level=True)


def _strip_system_warning(s: str) -> str:
    if server.SYSTEM_WARNING and s.startswith(server.SYSTEM_WARNING):
        return s[len(server.SYSTEM_WARNING):]
    return s


def test_search_literature_broad_returns_list():
    resp = asyncio.run(server.search_literature_broad("attention is all you need", limit=3))
    text = _strip_system_warning(resp)
    papers = json.loads(text)
    assert isinstance(papers, list)
    assert len(papers) > 0
    assert "paperId" in papers[0]
    assert "title" in papers[0]


def test_get_papers_batch_returns_details():
    resp = asyncio.run(server.search_literature_broad("attention is all you need", limit=2))
    papers = json.loads(_strip_system_warning(resp))
    assert papers and isinstance(papers, list)
    first_id = papers[0]["paperId"]

    batch_resp = asyncio.run(server.get_papers_batch([first_id]))
    results = json.loads(_strip_system_warning(batch_resp))
    assert isinstance(results, list)
    assert results[0].get("paperId") == first_id


def test_fetch_pdf_returns_status_string():
    resp = asyncio.run(server.search_literature_broad("attention is all you need", limit=2))
    papers = json.loads(_strip_system_warning(resp))
    first_id = papers[0]["paperId"]

    pdf_resp = asyncio.run(server.fetch_pdf(first_id))
    assert isinstance(pdf_resp, str)
    ok_prefixes = ("PAYWALL:", "NOT_FOUND:", "MANUAL_DOWNLOAD:", "BLOCKED:", "SUCCESS:", "API error", "Error")
    assert any(pdf_resp.startswith(p) for p in ok_prefixes) or any(k in pdf_resp for k in ok_prefixes)
