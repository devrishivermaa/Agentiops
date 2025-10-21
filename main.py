from agents.master_agent import MasterAgent
from orchestrator import spawn_submasters_and_run
from utils.report_generator import generate_analysis_report
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
        
        total_llm_successes = 0
        total_llm_failures = 0
        
        for sm_id, info in results.items():
            status = info['status']
            
            if status == 'ok':
                output = info.get('output', {})
                total_pages = output.get('total_pages', 0)
                total_chars = output.get('total_chars', 0)
                total_entities = output.get('total_entities', 0)
                total_keywords = output.get('total_keywords', 0)
                llm_successes = output.get('llm_successes', 0)
                llm_failures = output.get('llm_failures', 0)
                role = output.get('role', 'N/A')[:60]
                
                total_llm_successes += llm_successes
                total_llm_failures += llm_failures
                
                print(f"\n✅ {sm_id}")
                print(f"   Role: {role}...")
                print(f"   Pages: {total_pages} | Characters: {total_chars:,}")
                
                if llm_successes > 0:
                    print(f"   ✨ LLM Analysis: {llm_successes}/{total_pages} pages analyzed")
                    print(f"   📊 Extracted: {total_entities} entities, {total_keywords} keywords")
                    
                    # Show aggregate summary preview
                    agg_summary = output.get('aggregate_summary', '')
                    if agg_summary and not agg_summary.startswith("No analysis"):
                        preview = agg_summary[:150] + "..." if len(agg_summary) > 150 else agg_summary
                        print(f"   📝 Summary: {preview}")
                else:
                    print(f"   ⚠️  LLM Analysis: Failed ({llm_failures} errors)")
            else:
                print(f"\n❌ {sm_id}: {info.get('error', 'Unknown error')}")
        
        print("\n" + "=" * 80)
        print("📊 OVERALL STATISTICS")
        print("=" * 80)
        print(f"   Total SubMasters: {len(results)}")
        print(f"   LLM Analyses: {total_llm_successes} successful, {total_llm_failures} failed")
        
        if total_llm_successes > 0:
            success_rate = (total_llm_successes / (total_llm_successes + total_llm_failures)) * 100
            print(f"   Success Rate: {success_rate:.1f}%")
        
        print("\n💡 TIP: Run 'python3 inspect_results.py' for detailed analysis")
        print("=" * 80)
        
        # Step 5: Generate reports
        print("\n📝 Generating analysis reports...")
        try:
            report_files = generate_analysis_report(results, metadata, output_dir="output")
            
            print("\n✅ Reports generated successfully!")
            
            if 'json' in report_files:
                print(f"   � JSON Data: {report_files['json']}")
            
            if 'pdf' in report_files:
                print(f"   �📄 PDF Report: {report_files['pdf']}")
            elif 'pdf_error' in report_files:
                print(f"   ⚠️  PDF generation failed: {report_files['pdf_error']}")
                print(f"   💡 JSON report is still available")
            
            print("\n💡 View reports: python3 view_reports.py")
            print("=" * 80)
            
        except Exception as e:
            print(f"\n⚠️  Failed to generate reports: {e}")
            print("   (Results are still available in memory)")
            import traceback
            traceback.print_exc()
            print("=" * 80)

