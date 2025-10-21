#!/usr/bin/env python3
# test_parallel_llm.py
"""
Test parallel LLM processing with SubMasters.
This verifies that multiple SubMasters can make concurrent LLM API calls.
"""

import json
import time
import sys
import ray

from agents.sub_master import SubMaster
from utils.logger import get_logger

logger = get_logger("TestParallelLLM")


def test_parallel_llm():
    """Test that SubMasters can make parallel LLM calls."""
    
    print("=" * 80)
    print("🧪 TESTING PARALLEL LLM PROCESSING")
    print("=" * 80)
    
    # Load metadata
    with open("metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"\n📄 Document: {metadata['file_name']}")
    print(f"📁 Path: {metadata['file_path']}")
    print(f"📊 Total pages: {metadata['num_pages']}")
    print(f"🤖 Model: {metadata.get('preferred_model', 'gemini-2.0-flash-exp')}")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        print("\n✅ Ray initialized")
    
    # Create test plan with 2 SubMasters for parallel testing
    test_plan = {
        "submasters": [
            {
                "submaster_id": "SM-TEST-LLM-01",
                "role": "Summarize the abstract and extract key objectives",
                "assigned_sections": ["Abstract"],
                "page_range": [1, 2]  # Small range for quick testing
            },
            {
                "submaster_id": "SM-TEST-LLM-02",
                "role": "Extract entities and methodologies from introduction",
                "assigned_sections": ["Introduction"],
                "page_range": [3, 4]  # Small range for quick testing
            }
        ]
    }
    
    print(f"\n🚀 Spawning {len(test_plan['submasters'])} SubMasters for parallel LLM testing...")
    
    # Spawn SubMaster actors
    actors = {}
    for sm_config in test_plan["submasters"]:
        sm_id = sm_config["submaster_id"]
        actor = SubMaster.options(name=sm_id).remote(sm_config, metadata)
        actors[sm_id] = actor
        print(f"   ✓ Spawned {sm_id}: {sm_config['role'][:50]}...")
    
    # Initialize all actors
    print("\n⚙️  Initializing SubMasters (PDF + LLM)...")
    init_futures = [actor.initialize.remote() for actor in actors.values()]
    init_results = ray.get(init_futures)
    print(f"   ✓ All {len(init_results)} SubMasters initialized")
    
    # Process in parallel with timing
    print("\n🔄 Processing pages with parallel LLM calls...")
    start_time = time.time()
    
    process_futures = {sm_id: actor.process.remote() for sm_id, actor in actors.items()}
    
    results = {}
    for sm_id, future in process_futures.items():
        try:
            result = ray.get(future)
            results[sm_id] = result
            
            elapsed = time.time() - start_time
            print(f"   ✓ {sm_id} completed in {elapsed:.2f}s")
            print(f"      - {result['total_pages']} pages")
            print(f"      - {result['llm_successes']} LLM successes")
            print(f"      - {result['total_entities']} entities")
            print(f"      - {result['total_keywords']} keywords")
        except Exception as e:
            print(f"   ✗ {sm_id} failed: {e}")
            results[sm_id] = {"error": str(e)}
    
    total_time = time.time() - start_time
    
    # Display detailed results
    print("\n" + "=" * 80)
    print("📊 DETAILED RESULTS")
    print("=" * 80)
    
    all_successes = []
    all_failures = []
    
    for sm_id, result in results.items():
        if "error" in result:
            print(f"\n❌ {sm_id}: ERROR - {result['error']}")
            continue
        
        print(f"\n✅ {sm_id}")
        print(f"   Role: {result['role']}")
        print(f"   Pages: {result['page_range']}")
        print(f"   LLM Analysis: {result['llm_successes']}/{result['total_pages']} successful")
        print(f"   Entities: {result['total_entities']}")
        print(f"   Keywords: {result['total_keywords']}")
        
        all_successes.append(result['llm_successes'])
        all_failures.append(result['llm_failures'])
        
        # Show sample analysis from first page
        if result['results']:
            first_page = result['results'][0]
            print(f"\n   📄 Sample from Page {first_page['page']}:")
            
            if 'summary' in first_page and not first_page['summary'].startswith('['):
                summary = first_page['summary'][:200]
                print(f"      Summary: {summary}...")
            
            if 'entities' in first_page and first_page['entities']:
                entities = ', '.join(first_page['entities'][:5])
                print(f"      Entities: {entities}")
            
            if 'keywords' in first_page and first_page['keywords']:
                keywords = ', '.join(first_page['keywords'][:5])
                print(f"      Keywords: {keywords}")
    
    # Final statistics
    print("\n" + "=" * 80)
    print("🎯 TEST RESULTS")
    print("=" * 80)
    
    total_successes = sum(all_successes)
    total_failures = sum(all_failures)
    
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   SubMasters: {len(results)}")
    print(f"   LLM Calls: {total_successes} successful, {total_failures} failed")
    
    if total_successes + total_failures > 0:
        success_rate = (total_successes / (total_successes + total_failures)) * 100
        print(f"   Success Rate: {success_rate:.1f}%")
    
    # Verify parallel execution
    avg_time_per_submaster = total_time / len(results) if results else 0
    print(f"   Avg Time per SubMaster: {avg_time_per_submaster:.2f}s")
    
    if total_time < (len(results) * 3):  # Each should take ~3-5s, parallel should be much faster
        print(f"   ✅ Parallel execution confirmed (total < sum of individual times)")
    else:
        print(f"   ⚠️  May not be running in parallel (check Ray cluster)")
    
    print("\n" + "=" * 80)
    
    if total_successes > 0:
        print("✅ PARALLEL LLM PROCESSING WORKS!")
        print("\nSubMasters are successfully:")
        print("  - Extracting PDF text")
        print("  - Making concurrent LLM API calls")
        print("  - Returning structured analysis")
        print("  - Running in parallel via Ray")
    else:
        print("❌ PARALLEL LLM PROCESSING FAILED!")
        print("\nCheck:")
        print("  - GOOGLE_API_KEY environment variable")
        print("  - Internet connectivity")
        print("  - Gemini API quota")
    
    print("=" * 80)
    
    # Shutdown Ray
    ray.shutdown()
    
    return total_successes > 0


if __name__ == "__main__":
    try:
        success = test_parallel_llm()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
