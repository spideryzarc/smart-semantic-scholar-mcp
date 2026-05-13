import asyncio
import os
import sys

# Ensure src is in the path for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from smart_semantic_scholar_mcp.server import extract_SS_ids_from_bibtex

async def run_test():
    bib_file = os.path.join(os.path.dirname(__file__), 'references.bib')
    print(f"Testing extraction tool on {bib_file}...\n")
    
    result = await extract_SS_ids_from_bibtex(bib_file)
    
    print("\n=== EXTRACTION RESULT ===")
    print(result)

if __name__ == "__main__":
    asyncio.run(run_test())
