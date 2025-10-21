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

        # Step 4: Print detailed summary
        print("\n" + "=" * 80)
        print("✅ PROCESSING COMPLETE")
        print("=" * 80)
        
        for sm_id, info in results.items():
            status = info['status']
            
            if status == 'ok':
                output = info.get('output', {})
                total_pages = output.get('total_pages', 0)
                total_chars = output.get('total_chars', 0)
                role = output.get('role', 'N/A')[:60]
                
                print(f"\n✅ {sm_id}")
                print(f"   Role: {role}...")
                print(f"   Pages: {total_pages} | Characters: {total_chars:,}")
                
                # Check if LLM processing happened
                results_data = output.get('results', [])
                has_analysis = any(
                    'summary' in r or 'entities' in r or 'keywords' in r 
                    for r in results_data
                )
                
                if has_analysis:
                    print(f"   ✨ LLM Analysis: Yes")
                else:
                    print(f"   ⚠️  LLM Analysis: No (only extracted text)")
            else:
                print(f"\n❌ {sm_id}: {info.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 80)
        print("💡 TIP: Run 'python3 inspect_results.py' for detailed analysis")
        print("=" * 80)

