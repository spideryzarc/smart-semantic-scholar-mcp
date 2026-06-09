import inspect
import json
import os
import sys

import pytest

# Ensure src is in the path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from smart_semantic_scholar_mcp import server


def _strip_system_warning(payload: str) -> str:
    if payload.startswith("[SYSTEM WARNING:"):
        parts = payload.split("\n\n", 1)
        return parts[1] if len(parts) > 1 else payload
    return payload


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    db_path = tmp_path / "papers_cache.sqlite"
    monkeypatch.setattr(server, "DB_PATH", db_path)
    server.init_db()
    return db_path


@pytest.mark.asyncio
async def test_get_papers_batch_returns_cached_at_from_cache(isolated_db):
    paper_id = "paper-cache-hit-1"
    server.save_cached(
        {
            paper_id: {
                "paperId": paper_id,
                "title": "Cached title",
                "abstract": "Cached abstract",
            }
        }
    )

    response = await server.get_papers_batch([paper_id])
    payload = json.loads(_strip_system_warning(response))

    assert isinstance(payload, list)
    assert payload[0]["paperId"] == paper_id
    assert isinstance(payload[0].get("cached_at"), str)
    assert payload[0]["cached_at"]


@pytest.mark.asyncio
async def test_get_papers_batch_force_refresh_bypasses_cache(isolated_db, monkeypatch):
    paper_id = "paper-force-refresh-1"
    server.save_cached(
        {
            paper_id: {
                "paperId": paper_id,
                "title": "Stale title",
                "abstract": "Stale abstract",
            }
        }
    )

    async def fake_fetch_api(client, method, endpoint, **kwargs):
        assert method == "POST"
        assert endpoint == "/paper/batch"
        return [
            {
                "paperId": paper_id,
                "title": "Fresh title",
                "abstract": "Fresh abstract",
                "tldr": {"text": "Fresh summary"},
                "authors": [],
                "isOpenAccess": False,
                "openAccessPdf": None,
            }
        ]

    monkeypatch.setattr(server, "fetch_api", fake_fetch_api)

    response = await server.get_papers_batch([paper_id], force_refresh=True)
    payload = json.loads(_strip_system_warning(response))

    assert payload[0]["title"] == "Fresh title"
    assert payload[0]["abstract"] == "Fresh abstract"

    cached_after = server.get_cached([paper_id])[paper_id]
    assert cached_after["title"] == "Fresh title"


def test_force_refresh_parameter_is_exposed_on_search_tools():
    assert inspect.signature(server.search_literature_broad).parameters["force_refresh"].default is False
    assert inspect.signature(server.get_papers_batch).parameters["force_refresh"].default is False
    assert inspect.signature(server.trace_citations_snowball).parameters["force_refresh"].default is False
    assert inspect.signature(server.get_recommended_papers).parameters["force_refresh"].default is False
