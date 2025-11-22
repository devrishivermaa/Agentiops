# run_pipeline.py
"""
Complete pipeline: Mapper → MasterAgent → SubMasters → Reports
"""

import os
import sys
import json
from workflows.mapper import Mapper
from agents.master_agent import MasterAgent
from orchestrator import spawn_submasters_and_run
from utils.report_generator import generate_analysis_report
from utils.logger import get_logger

logger = get_logger("Pipeline")

def run_complete_pipeline(pdf_path: str, config: dict = None):
    """Run the complete document processing pipeline."""
    
    print("\n" + "=" * 80)
    print("🚀 AGENTOPS COMPLETE PIPELINE")
    print("=" * 80)
    
    # Default configuration
    if config is None:
        config = {
            "document_type": "research_paper",
            "processing_requirements": [
                "summary_generation",
                "entity_extraction",
                "keyword_indexing"
            ],
            "user_notes": "Extract key findings, methods, and results from this paper.",
            "brief_info": "Research paper analysis",
            "complexity_level": "high",
            "priority": "high",
            "preferred_model": "mistral-small-latest",  # Updated
            "max_parallel_submasters": 3,  # Can increase with Mistral's better limits
            "num_workers_per_submaster": 4,
            "feedback_required": True
        }

    
    # STEP 1: MAPPER - Generate metadata
    print("\n" + "=" * 80)
    print("[STEP 1/5] MAPPER: Validating PDF and extracting metadata...")
    print("=" * 80)
    
    mapper = Mapper(output_dir="./output")
    
    try:
        mapper_result = mapper.execute(pdf_path, config)
        
        if mapper_result["status"] != "success":
            print(f"\n❌ Mapper failed: {mapper_result}")
            return 1
        
        metadata_path = mapper_result["metadata_path"]
        print(f"\n✅ Metadata generated: {metadata_path}")
        print(f"   Pages: {mapper_result['num_pages']}")
        print(f"   Sections: {mapper_result['num_sections']}")
        
    except Exception as e:
        print(f"\n❌ Mapper failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # STEP 2: MASTER AGENT - Generate execution plan
    print("\n" + "=" * 80)
    print("[STEP 2/5] MASTER AGENT: Generating SubMaster execution plan...")
    print("=" * 80)
    
    try:
        master_agent = MasterAgent()
        plan = master_agent.execute(metadata_path)
        
        if plan is None or plan.get("status") != "approved":
            print("\n❌ Plan generation failed or not approved")
            return 1
        
        print(f"\n✅ Plan approved: {plan.get('num_submasters')} SubMasters")
        
    except Exception as e:
        print(f"\n❌ Master Agent failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # STEP 3: Load metadata
    print("\n" + "=" * 80)
    print("[STEP 3/5] Loading metadata for processing...")
    print("=" * 80)
    
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"✅ Loaded: {metadata.get('file_name')} ({metadata.get('num_pages')} pages)")
    except Exception as e:
        print(f"\n❌ Failed to load metadata: {e}")
        return 1
    
    # STEP 4: ORCHESTRATOR - Execute SubMasters
    print("\n" + "=" * 80)
    print("[STEP 4/5] ORCHESTRATOR: Executing SubMasters in parallel...")
    print("=" * 80)
    
    try:
        results = spawn_submasters_and_run(plan, metadata)
        print(f"\n✅ Processing completed: {len(results)} SubMasters finished")
    except Exception as e:
        print(f"\n❌ SubMaster execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # STEP 5: REPORT GENERATOR - Create outputs
    print("\n" + "=" * 80)
    print("[STEP 5/5] REPORT GENERATOR: Creating analysis reports...")
    print("=" * 80)
    
    try:
        report_files = generate_analysis_report(results, metadata, output_dir="output")
        
        print("\n✅ Reports generated:")
        if 'json' in report_files:
            print(f"   📊 JSON: {report_files['json']}")
        if 'pdf' in report_files:
            print(f"   📄 PDF: {report_files['pdf']}")
        elif 'pdf_error' in report_files:
            print(f"   ⚠️  PDF failed: {report_files['pdf_error']}")
            print(f"   💡 JSON report is still available")
        
    except Exception as e:
        print(f"\n⚠️  Report generation failed: {e}")
        print("   (Results are still available)")
        import traceback
        traceback.print_exc()
    
    # FINAL SUMMARY
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    
    total_success = sum(1 for r in results.values() if r.get('status') == 'ok')
    total_failures = len(results) - total_success
    
    total_llm_success = 0
    total_llm_failures = 0
    total_entities = 0
    total_keywords = 0
    total_pages_processed = 0
    
    for sm_id, info in results.items():
        if info['status'] == 'ok':
            output = info['output']
            total_llm_success += output.get('llm_successes', 0)
            total_llm_failures += output.get('llm_failures', 0)
            total_entities += output.get('total_entities', 0)
            total_keywords += output.get('total_keywords', 0)
            total_pages_processed += output.get('total_pages', 0)
            
            print(f"\n✅ {sm_id}")
            print(f"   Role: {output.get('role', 'N/A')[:70]}...")
            print(f"   Pages: {output.get('total_pages', 0)} | "
                  f"Entities: {output.get('total_entities', 0)} | "
                  f"Keywords: {output.get('total_keywords', 0)}")
            
            summary = output.get('aggregate_summary', '')
            if summary and len(summary) > 20 and not summary.startswith("No analysis"):
                preview = summary[:150] + "..." if len(summary) > 150 else summary
                print(f"   📝 {preview}")
        else:
            print(f"\n❌ {sm_id}: {info.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print(f"SubMasters: {total_success} successful, {total_failures} failed")
    print(f"Pages Processed: {total_pages_processed}")
    print(f"LLM Analyses: {total_llm_success} successful, {total_llm_failures} failed")
    print(f"Extracted: {total_entities} entities, {total_keywords} keywords")
    
    if total_llm_success > 0:
        success_rate = (total_llm_success / (total_llm_success + total_llm_failures)) * 100
        print(f"Success Rate: {success_rate:.1f}%")
    
    print("=" * 80)
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!\n")
    
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <pdf_path>")
        print("\nExample:")
        print("  python run_pipeline.py data/2510.02125v1.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    if not os.path.exists(pdf_path):
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    # Optional: Load custom config from JSON file
    config = None
    if len(sys.argv) > 2:
        config_path = sys.argv[2]
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            print(f"✅ Loaded custom config from: {config_path}")
        except Exception as e:
            print(f"⚠️  Failed to load config (using defaults): {e}")
    
    exit_code = run_complete_pipeline(pdf_path, config)
    sys.exit(exit_code)

