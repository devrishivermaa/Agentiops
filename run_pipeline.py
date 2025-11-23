"""
Complete pipeline: Mapper -> MasterAgent -> ResidualAgent -> SubMasters -> Reports
"""

import os
import sys
import json
import ray

from workflows.mapper import Mapper
from agents.master_agent import MasterAgent
from agents.residual_agent import ResidualAgentActor
from orchestrator import spawn_submasters_and_run
from utils.report_generator import generate_analysis_report
from utils.logger import get_logger

logger = get_logger("Pipeline")


def run_complete_pipeline(pdf_path: str, config: dict = None):
    print("\n" + "=" * 80)
    print("AGENTOPS COMPLETE PIPELINE")
    print("=" * 80)

    # default config
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
            "preferred_model": "mistral-small-latest",
            "max_parallel_submasters": 3,
            "num_workers_per_submaster": 4,
            "feedback_required": True
        }

    # STEP 1: MAPPER
    print("\n" + "=" * 80)
    print("[STEP 1] MAPPER")
    print("=" * 80)

    mapper = Mapper(output_dir="./output")

    try:
        mapper_result = mapper.execute(pdf_path, config)
        if mapper_result["status"] != "success":
            print(f"Mapper failed: {mapper_result}")
            return 1

        metadata_path = mapper_result["metadata_path"]
        print(f"Metadata generated at: {metadata_path}")

    except Exception as e:
        print(f"Mapper exception: {e}")
        return 1

    # STEP 2: MASTER AGENT
    print("\n" + "=" * 80)
    print("[STEP 2] MASTER AGENT")
    print("=" * 80)

    try:
        master = MasterAgent()
        plan = master.execute(metadata_path)

        if plan is None or plan.get("status") != "approved":
            print("Plan not approved or not generated")
            return 1

        print(f"Plan approved with {plan.get('num_submasters')} submasters")

    except Exception as e:
        print(f"Master Agent exception: {e}")
        return 1

    # STEP 3: RESIDUAL AGENT (Initialize only)
    print("\n" + "=" * 80)
    print("[STEP 3] RESIDUAL AGENT INITIALIZATION")
    print("=" * 80)

    try:
        with open(metadata_path, "r", encoding="utf8") as f:
            metadata = json.load(f)

        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

        # Create ResidualAgent actor
        residual = ResidualAgentActor.remote()
        print("ResidualAgent actor created")

    except Exception as e:
        print(f"ResidualAgent initialization exception: {e}")
        return 1

    # STEP 4: ORCHESTRATOR (passes residual handle)
    print("\n" + "=" * 80)
    print("[STEP 4] ORCHESTRATOR + RESIDUAL AGENT CONTEXT GENERATION")
    print("=" * 80)

    try:
        # The orchestrator will:
        # 1. Spawn and initialize SubMasters (and workers)
        # 2. Register them with ResidualAgent
        # 3. Call residual.generate_and_distribute()
        # 4. Run processing
        results = spawn_submasters_and_run(plan, metadata, residual_handle=residual)
        print(f"Completed: {len(results)} submasters")
        
        # Print context usage summary
        for sm_id, result in results.items():
            if result.get("status") == "ok":
                context_usage = result.get("output", {}).get("output", {}).get("context_usage", "N/A")
                print(f"  {sm_id}: {context_usage}")
                
    except Exception as e:
        print(f"SubMaster run exception: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # STEP 5: REPORT GENERATOR
    print("\n" + "=" * 80)
    print("[STEP 5] REPORT GENERATOR")
    print("=" * 80)

    try:
        files = generate_analysis_report(results, metadata, output_dir="output")

        print("Reports generated:")
        for k, v in files.items():
            print(f"{k}: {v}")

    except Exception as e:
        print(f"Report generation exception: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_pipeline.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    config = None
    if len(sys.argv) > 2:
        try:
            with open(sys.argv[2], "r") as f:
                config = json.load(f)
            print("Loaded custom config")
        except Exception:
            print("Config load failed, using defaults")

    exit(run_complete_pipeline(pdf_path, config))