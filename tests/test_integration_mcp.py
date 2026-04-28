import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

# Use a temporary directory for testing to avoid polluting the main cache
import tempfile
test_cache_dir = tempfile.mkdtemp()
os.environ["MCP_CACHE_DIR"] = test_cache_dir

# Ensure the local `src/` package is importable when running tests from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import json
import asyncio
import pytest
from datetime import datetime
from smart_semantic_scholar_mcp import server


API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
if not API_KEY:
    pytest.skip("Requires SEMANTIC_SCHOLAR_API_KEY environment variable for live API tests", allow_module_level=True)


def _strip_system_warning(s: str) -> str:
    if s.startswith("[SYSTEM WARNING:"):
        parts = s.split("\n\n", 1)
        return parts[1] if len(parts) > 1 else s
    return s


@pytest.fixture(scope="session")
def inspection_log():
    log_path = Path(__file__).parent / "inspection_log.md"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"# Semantic Scholar MCP - Inspection Log\n")
        f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
        f.write("Este arquivo registra as respostas completas das ferramentas do MCP para inspeção manual.\n\n")
    return log_path


def log_step(log_path: Path, step_name: str, input_data: str, output_data: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"## {step_name}\n")
        f.write(f"**Input:** `{input_data}`\n\n")
        f.write("**Output/Response:**\n")
        if output_data.strip().startswith(("{", "[")):
            f.write(f"```json\n{output_data}\n```\n\n")
        else:
            f.write(f"```text\n{output_data}\n```\n\n")
        f.write("---\n\n")


def test_agent_workflow_scenario(inspection_log):
    """
    Simula o fluxo completo de um agente de IA pesquisando literatura e 
    registra todas as respostas no log de inspeção.
    """
    # Step 1: Broad Search
    query = "attention is all you need"
    resp = asyncio.run(server.search_literature_broad(query, limit=3))
    clean_resp = _strip_system_warning(resp)
    log_step(inspection_log, "Step 1: Broad Search (search_literature_broad)", query, clean_resp)
    
    papers = json.loads(clean_resp)
    assert isinstance(papers, list), "Expected broad search to return a list"
    assert len(papers) > 0, "Expected at least one paper"
    
    first_paper_id = papers[0]["paperId"]
    
    # Step 2: Get Details for the first paper
    batch_resp = asyncio.run(server.get_papers_batch([first_paper_id]))
    clean_batch_resp = _strip_system_warning(batch_resp)
    log_step(inspection_log, "Step 2: Get Paper Details (get_papers_batch)", f"IDs: [{first_paper_id}]", clean_batch_resp)
    
    details = json.loads(clean_batch_resp)
    assert isinstance(details, list), "Expected batch details to return a list"
    assert details[0].get("paperId") == first_paper_id
    
    # Step 3: Snowball - Backward References
    snowball_resp = asyncio.run(server.trace_citations_snowball(first_paper_id, direction="backward", min_citations=100))
    clean_snowball_resp = _strip_system_warning(snowball_resp)
    log_step(inspection_log, "Step 3: Trace Citations (trace_citations_snowball - backward)", f"Paper ID: {first_paper_id}", clean_snowball_resp)
    
    snowball_data = json.loads(clean_snowball_resp)
    assert isinstance(snowball_data, list), "Expected snowball search to return a list"
    
    # Step 4: Author Graph
    authors = details[0].get("authors", [])
    if authors and authors[0].get("authorId"):
        author_id = authors[0]["authorId"]
        author_name = authors[0].get("name", "Unknown")
        author_resp = asyncio.run(server.generate_author_graph(author_id))
        clean_author_resp = _strip_system_warning(author_resp)
        log_step(inspection_log, f"Step 4: Generate Author Graph (generate_author_graph)", f"Author: {author_name} ({author_id})", clean_author_resp)
        
        author_data = json.loads(clean_author_resp)
        assert isinstance(author_data, dict), "Expected author graph to return a dict"
    else:
        log_step(inspection_log, f"Step 4: Generate Author Graph", "No authors found in paper details.", "Skipped")
    
    # Step 5: Fetch PDF
    pdf_resp = asyncio.run(server.fetch_pdf(first_paper_id))
    log_step(inspection_log, "Step 5: Fetch PDF (fetch_pdf)", f"Paper ID: {first_paper_id}", pdf_resp)
    
    assert isinstance(pdf_resp, str), "Expected fetch_pdf to return a status string"
