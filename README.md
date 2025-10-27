# Agentiops
1. User provides PDF
   ↓
2. Master Agent validates metadata
   ↓
3. Master Agent creates plan with 2 SubMasters
   ↓
4. Orchestrator spawns 2 SubMaster Ray actors
   ↓
5. Each SubMaster spawns 3 Worker Ray actors
   ↓
6. SubMaster extracts text from PDF
   ↓
7. SubMaster distributes pages to Workers (round-robin)
   ↓
8. Workers call LLM API to analyze pages
   ↓
9. Workers return results to SubMasters
   ↓
10. SubMasters aggregate results
   ↓
11. Orchestrator collects all SubMaster results
   ↓
12. Report Generator creates JSON + PDF
