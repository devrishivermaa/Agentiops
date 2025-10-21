#!/usr/bin/env python3
# generate_simple_report.py
"""
Simple text-based report generator (no PDF dependencies).
Fallback when reportlab is not available.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any


def generate_text_report(results: Dict[str, Any], metadata: Dict[str, Any], output_path: str) -> str:
    """
    Generate a simple text report.
    
    Args:
        results: SubMaster analysis results
        metadata: Document metadata
        output_path: Path to save text file
        
    Returns:
        Path to generated file
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Header
        f.write("=" * 80 + "\n")
        f.write(f"DOCUMENT ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # Document info
        f.write(f"Document: {metadata.get('file_name', 'N/A')}\n")
        f.write(f"Type: {metadata.get('document_type', 'N/A').upper()}\n")
        f.write(f"Pages: {metadata.get('num_pages', 'N/A')}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {metadata.get('preferred_model', 'N/A')}\n")
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Executive summary
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-" * 80 + "\n\n")
        
        total_submasters = len(results)
        total_pages = sum(r.get('output', {}).get('total_pages', 0) for r in results.values())
        total_entities = sum(r.get('output', {}).get('total_entities', 0) for r in results.values())
        total_keywords = sum(r.get('output', {}).get('total_keywords', 0) for r in results.values())
        total_successes = sum(r.get('output', {}).get('llm_successes', 0) for r in results.values())
        total_failures = sum(r.get('output', {}).get('llm_failures', 0) for r in results.values())
        
        f.write(f"Total SubMasters: {total_submasters}\n")
        f.write(f"Pages Analyzed: {total_pages}\n")
        f.write(f"Entities Extracted: {total_entities}\n")
        f.write(f"Keywords Extracted: {total_keywords}\n")
        f.write(f"LLM Successes: {total_successes}\n")
        f.write(f"LLM Failures: {total_failures}\n")
        
        if total_successes + total_failures > 0:
            success_rate = (total_successes / (total_successes + total_failures)) * 100
            f.write(f"Success Rate: {success_rate:.1f}%\n")
        
        f.write("\n" + "=" * 80 + "\n\n")
        
        # Detailed results
        f.write("DETAILED ANALYSIS BY SECTION\n")
        f.write("-" * 80 + "\n\n")
        
        for sm_id, result in results.items():
            if result.get('status') != 'ok':
                f.write(f"SubMaster {sm_id}: ERROR - {result.get('error', 'Unknown')}\n\n")
                continue
            
            output = result.get('output', {})
            
            f.write(f"SubMaster: {sm_id}\n")
            f.write(f"Role: {output.get('role', 'N/A')}\n")
            f.write(f"Sections: {', '.join(output.get('assigned_sections', []))}\n")
            f.write(f"Pages: {output.get('page_range', [])}\n")
            f.write(f"Total Pages: {output.get('total_pages', 0)}\n")
            f.write(f"Characters: {output.get('total_chars', 0):,}\n")
            f.write(f"Entities: {output.get('total_entities', 0)}\n")
            f.write(f"Keywords: {output.get('total_keywords', 0)}\n")
            f.write(f"LLM Analysis: {output.get('llm_successes', 0)}/{output.get('total_pages', 0)} successful\n")
            f.write("\n")
            
            # Aggregate summary
            agg_summary = output.get('aggregate_summary', '')
            if agg_summary and not agg_summary.startswith('No analysis'):
                f.write(f"Summary:\n{agg_summary}\n\n")
            
            # Sample findings
            page_results = output.get('results', [])
            if page_results:
                f.write("Sample Findings (First 3 Pages):\n\n")
                
                for page_result in page_results[:3]:
                    page_num = page_result.get('page', '?')
                    f.write(f"  Page {page_num}:\n")
                    
                    if 'summary' in page_result and not page_result['summary'].startswith('['):
                        summary = page_result['summary'][:200]
                        f.write(f"    Summary: {summary}...\n")
                    
                    if page_result.get('entities'):
                        entities = ', '.join(page_result['entities'][:5])
                        f.write(f"    Entities: {entities}\n")
                    
                    if page_result.get('keywords'):
                        keywords = ', '.join(page_result['keywords'][:5])
                        f.write(f"    Keywords: {keywords}\n")
                    
                    f.write("\n")
            
            f.write("-" * 80 + "\n\n")
        
        # Appendix
        f.write("=" * 80 + "\n")
        f.write("APPENDIX: ALL EXTRACTED ENTITIES AND KEYWORDS\n")
        f.write("=" * 80 + "\n\n")
        
        # Collect all entities and keywords
        all_entities = set()
        all_keywords = set()
        
        for result in results.values():
            if result.get('status') != 'ok':
                continue
            
            output = result.get('output', {})
            for page_result in output.get('results', []):
                all_entities.update(page_result.get('entities', []))
                all_keywords.update(page_result.get('keywords', []))
        
        if all_entities:
            f.write("ALL ENTITIES:\n")
            f.write(", ".join(sorted(all_entities)))
            f.write("\n\n")
        
        if all_keywords:
            f.write("ALL KEYWORDS:\n")
            f.write(", ".join(sorted(all_keywords)))
            f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write(f"End of Report - Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n")
    
    return output_path


def main():
    """Generate reports from latest results."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 generate_simple_report.py <results.json>")
        print("\nOr run after main.py to use latest results")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if not os.path.exists(json_file):
        print(f"❌ File not found: {json_file}")
        sys.exit(1)
    
    # Load results
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract results and metadata
    results = data.get('results', data)
    
    # Try to load metadata
    try:
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
    except:
        metadata = {
            'file_name': 'document.pdf',
            'num_pages': 0,
            'document_type': 'pdf'
        }
    
    # Generate report
    output_path = json_file.replace('.json', '.txt')
    
    print("📝 Generating text report...")
    report_path = generate_text_report(results, metadata, output_path)
    print(f"✅ Report saved: {report_path}")
    
    # Show first few lines
    print("\n" + "=" * 80)
    with open(report_path, 'r') as f:
        for i, line in enumerate(f):
            if i < 20:
                print(line, end='')
            else:
                print("\n... (see full report in file)")
                break
    print("=" * 80)


if __name__ == "__main__":
    main()
