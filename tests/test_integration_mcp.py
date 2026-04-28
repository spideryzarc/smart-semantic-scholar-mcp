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


from unittest.mock import patch

API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")

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

async def _run_workflow(inspection_log, has_key: bool = True):
    """
    Helper para rodar o cenário completo.
    """
    scenario_name = "Scenario WITH Key" if has_key else "Scenario WITHOUT Key"
    with open(inspection_log, "a", encoding="utf-8") as f:
        f.write(f"\n# RUNNING SCENARIO: {scenario_name}\n\n")

    # Se estiver rodando o cenário sem chave logo após o com chave,
    # aguardamos alguns segundos para o rate limit por IP da API esfriar.
    if not has_key:
        await asyncio.sleep(6)

    # Step 1: Broad Search
    query = "attention is all you need"
    resp = await server.search_literature_broad(query, limit=3)
    
    # Se estivermos sem chave, validamos que o aviso está presente
    if not has_key:
        assert "[SYSTEM WARNING: MCP running without API key" in resp
    
    clean_resp = _strip_system_warning(resp)
    log_step(inspection_log, f"[{scenario_name}] Step 1: Broad Search", query, clean_resp)
    
    papers = json.loads(clean_resp)
    assert isinstance(papers, list), "Expected broad search to return a list"
    assert len(papers) > 0, "Expected at least one paper"
    
    first_paper_id = papers[0]["paperId"]
    
    # Step 2: Get Details for the first paper
    batch_resp = await server.get_papers_batch([first_paper_id])
    clean_batch_resp = _strip_system_warning(batch_resp)
    log_step(inspection_log, f"[{scenario_name}] Step 2: Get Paper Details", f"IDs: [{first_paper_id}]", clean_batch_resp)
    
    details = json.loads(clean_batch_resp)
    assert isinstance(details, list), "Expected batch details to return a list"
    assert details[0].get("paperId") == first_paper_id
    
    # Step 3: Snowball - Backward References
    snowball_resp = await server.trace_citations_snowball(first_paper_id, direction="backward", min_citations=100)
    clean_snowball_resp = _strip_system_warning(snowball_resp)
    log_step(inspection_log, f"[{scenario_name}] Step 3: Trace Citations", f"Paper ID: {first_paper_id}", clean_snowball_resp)
    
    snowball_data = json.loads(clean_snowball_resp)
    assert isinstance(snowball_data, list), "Expected snowball search to return a list"
    
    # Step 4: Author Graph
    authors = details[0].get("authors", [])
    if authors and authors[0].get("authorId"):
        author_id = authors[0]["authorId"]
        author_name = authors[0].get("name", "Unknown")
        author_resp = await server.generate_author_graph(author_id)
        clean_author_resp = _strip_system_warning(author_resp)
        log_step(inspection_log, f"[{scenario_name}] Step 4: Generate Author Graph", f"Author: {author_name} ({author_id})", clean_author_resp)
        
        author_data = json.loads(clean_author_resp)
        assert isinstance(author_data, dict), "Expected author graph to return a dict"
    
    # Step 5: Fetch PDF
    pdf_resp = await server.fetch_pdf(first_paper_id)
    log_step(inspection_log, f"[{scenario_name}] Step 5: Fetch PDF", f"Paper ID: {first_paper_id}", pdf_resp)
    assert isinstance(pdf_resp, str)


def test_agent_workflow_scenario_with_key(inspection_log):
    if not API_KEY:
        pytest.skip("Requires SEMANTIC_SCHOLAR_API_KEY environment variable for live API tests")
    asyncio.run(_run_workflow(inspection_log, has_key=True))


def test_agent_workflow_scenario_no_key(inspection_log):
    """
    Testa o funcionamento REAL sem a chave, atingindo a API com rate limit.
    """
    with patch.dict("os.environ", {}, clear=True):
        # Garantimos que o server use o novo ambiente
        asyncio.run(_run_workflow(inspection_log, has_key=False))
