from agents.master_agent import MasterAgent
from orchestrator import spawn_submasters_and_run
import json

if __name__ == "__main__":
    # Step 1: Generate and approve SubMaster plan
    agent = MasterAgent()
    plan = agent.execute("metadata.json")

    if plan is None or plan.get("status") != "approved":
        print("Plan generation failed or was not approved.")
    else:
        # Step 2: Load metadata
        with open("metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        # Step 3: Run SubMasters
        results = spawn_submasters_and_run(plan, metadata)

        # Step 4: Print summary
        print("\n✅ DONE. Summary:\n")
        for sm_id, info in results.items():
            print(f" - {sm_id}: {info['status']}")
