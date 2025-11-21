# main.py
"""Main entry point for AgentOps document processing pipeline."""

import json
import sys
from agents.master_agent import MasterAgent
from orchestrator import spawn_submasters_and_run
from utils.report_generator import generate_analysis_report
from utils.logger import get_logger

logger = get_logger("Main")

def main(metadata_path: str = "metadata.json"):
    """Execute complete processing pipeline."""
    
    print("\n" + "=" * 80)
    print("🚀 AGENTOPS DOCUMENT PROCESSING PIPELINE")
    print("=" * 80)
    
    # Step 1: Generate SubMaster plan
    print("\n[STEP 1/4] Generating SubMaster execution plan...")
    agent = MasterAgent()
    plan = agent.execute(metadata_path)
    
    if plan is None or plan.get("status") != "approved":
        print("\n❌ Plan generation failed or not approved. Exiting.")
        return 1
    
    print(f"\n✅ Plan approved: {plan.get('num_submasters')} SubMasters")
    
    # Step 2: Load metadata
    print("\n[STEP 2/4] Loading metadata...")
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"✅ Loaded: {metadata.get('file_name')} ({metadata.get('num_pages')} pages)")
    except Exception as e:
        print(f"❌ Failed to load metadata: {e}")
        return 1
    
    # Step 3: Execute SubMasters
    print("\n[STEP 3/4] Executing SubMasters in parallel...")
    try:
        results = spawn_submasters_and_run(plan, metadata)
    except Exception as e:
        print(f"❌ SubMaster execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 4: Generate reports
    print("\n[STEP 4/4] Generating analysis reports...")
    try:
        report_files = generate_analysis_report(results, metadata, output_dir="output")
        
        print("\n✅ Reports generated:")
        if 'json' in report_files:
            print(f"   📊 JSON: {report_files['json']}")
        if 'pdf' in report_files:
            print(f"   📄 PDF: {report_files['pdf']}")
        
    except Exception as e:
        print(f"⚠️  Report generation failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 PROCESSING SUMMARY")
    print("=" * 80)
    
    total_success = sum(1 for r in results.values() if r.get('status') == 'ok')
    total_failures = len(results) - total_success
    
    total_llm_success = 0
    total_llm_failures = 0
    total_entities = 0
    total_keywords = 0
    
    for sm_id, info in results.items():
        if info['status'] == 'ok':
            output = info['output']
            total_llm_success += output.get('llm_successes', 0)
            total_llm_failures += output.get('llm_failures', 0)
            total_entities += output.get('total_entities', 0)
            total_keywords += output.get('total_keywords', 0)
            
            print(f"\n✅ {sm_id}")
            print(f"   Pages: {output.get('total_pages', 0)} | "
                  f"Entities: {output.get('total_entities', 0)} | "
                  f"Keywords: {output.get('total_keywords', 0)}")
            
            summary = output.get('aggregate_summary', '')
            if summary and len(summary) > 20:
                preview = summary[:120] + "..." if len(summary) > 120 else summary
                print(f"   📝 {preview}")
        else:
            print(f"\n❌ {sm_id}: {info.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print(f"SubMasters: {total_success} successful, {total_failures} failed")
    print(f"LLM Analyses: {total_llm_success} successful, {total_llm_failures} failed")
    print(f"Extracted: {total_entities} entities, {total_keywords} keywords")
    
    if total_llm_success > 0:
        success_rate = (total_llm_success / (total_llm_success + total_llm_failures)) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    print("=" * 80)
    print("\n✅ Processing complete!\n")
    
    return 0


if __name__ == "__main__":
    metadata_file = sys.argv if len(sys.argv) > 1 else "metadata.json"[1]
    sys.exit(main(metadata_file))
