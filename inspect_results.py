#!/usr/bin/env python3
# inspect_results.py
"""
Inspect what SubMasters are actually returning.
"""

import json
import sys

def inspect_results():
    print("=" * 80)
    print("🔍 INSPECTING SUBMASTER RESULTS")
    print("=" * 80)
    
    # Run the workflow
    from agents.master_agent import MasterAgent
    from orchestrator import spawn_submasters_and_run
    
    # Load plan
    with open("submasters_plan.json", "r") as f:
        plan = json.load(f)
    
    # Load metadata
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"\n📋 Plan has {len(plan['submasters'])} SubMasters\n")
    
    # Run SubMasters
    print("🚀 Running SubMasters...\n")
    results = spawn_submasters_and_run(plan, metadata)
    
    # Detailed inspection
    print("\n" + "=" * 80)
    print("📊 DETAILED RESULTS")
    print("=" * 80)
    
    for sm_id, result in results.items():
        print(f"\n{'─' * 80}")
        print(f"🤖 {sm_id}")
        print(f"{'─' * 80}")
        
        if result['status'] == 'error':
            print(f"❌ ERROR: {result['error']}")
            continue
        
        output = result.get('output', {})
        
        print(f"📌 Role: {output.get('role', 'N/A')}")
        print(f"📄 Sections: {', '.join(output.get('assigned_sections', []))}")
        print(f"📖 Page Range: {output.get('page_range', [])}")
        print(f"📊 Total Pages Processed: {output.get('total_pages', 0)}")
        print(f"📝 Total Characters: {output.get('total_chars', 0):,}")
        
        # Show sample of extracted text
        results_data = output.get('results', [])
        if results_data:
            print(f"\n📄 Sample from first page:")
            first_result = results_data[0]
            page_num = first_result.get('page', '?')
            preview = first_result.get('preview', first_result.get('text', '')[:300])
            print(f"   Page {page_num}:")
            print(f"   {preview}")
            print(f"   ...")
            
            if len(results_data) > 1:
                print(f"\n📄 Sample from last page:")
                last_result = results_data[-1]
                page_num = last_result.get('page', '?')
                preview = last_result.get('preview', last_result.get('text', '')[:300])
                print(f"   Page {page_num}:")
                print(f"   {preview}")
                print(f"   ...")
        else:
            print("\n⚠️  No results data found!")
    
    print("\n" + "=" * 80)
    print("🎯 PROBLEMS IDENTIFIED:")
    print("=" * 80)
    
    problems = []
    
    # Check if text is being extracted
    for sm_id, result in results.items():
        if result['status'] == 'ok':
            output = result.get('output', {})
            total_chars = output.get('total_chars', 0)
            
            if total_chars == 0:
                problems.append(f"❌ {sm_id}: No text extracted (0 characters)")
            elif total_chars < 100:
                problems.append(f"⚠️  {sm_id}: Very little text extracted ({total_chars} chars)")
    
    # Check if LLM processing is happening
    has_llm_output = False
    for sm_id, result in results.items():
        if result['status'] == 'ok':
            output = result.get('output', {})
            results_data = output.get('results', [])
            
            for page_result in results_data:
                if 'summary' in page_result or 'entities' in page_result or 'keywords' in page_result:
                    has_llm_output = True
                    break
    
    if not has_llm_output:
        problems.append("❌ NO LLM PROCESSING: SubMasters are only extracting text, not analyzing it!")
    
    if problems:
        for p in problems:
            print(f"   {p}")
    else:
        print("   ✅ All checks passed!")
    
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATIONS:")
    print("=" * 80)
    
    print("""
1. ❌ SubMasters are extracting PDF text but NOT calling LLM
   - Need to add LLM initialization in SubMaster.__init__()
   - Need to add LLM processing in SubMaster.process()
   
2. ❌ No role-based analysis happening
   - Text is extracted but not summarized/analyzed
   - Need to create prompts based on each SubMaster's role
   
3. ❌ Results are just raw text, not structured insights
   - Should return: summaries, entities, keywords, findings
   - Currently returning: raw page text
   
4. ✅ PDF extraction is working correctly
   - Pages are being read successfully
   - Text extraction is functional

NEXT STEP: Implement LLM calls in SubMaster to actually process the text!
    """)

if __name__ == "__main__":
    try:
        inspect_results()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
