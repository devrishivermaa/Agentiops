# workflows/reducer.py
"""
Reducer workflow: Second-pass aggregation and cross-document analysis.
Processes Mapper results to generate consolidated insights.
"""

import json
import os
import time
from typing import Dict, Any, List
from collections import Counter
from datetime import datetime
from utils.logger import get_logger
from utils.monitor import SystemMonitor

logger = get_logger("Reducer")


class Reducer:
    """
    Reducer aggregates SubMaster results into consolidated final output.
    
    Performs:
    - Entity deduplication and frequency analysis
    - Keyword consolidation and ranking
    - Cross-section insight generation
    - Final summary synthesis
    """
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.monitor = SystemMonitor()
        logger.info(f"Reducer initialized: {output_dir}")
    
    def reduce(self, mapper_results: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate and consolidate Mapper results.
        
        Args:
            mapper_results: Results from SubMasters (dict of {sm_id: result})
            metadata: Document metadata
            
        Returns:
            Consolidated analysis results
        """
        start_time = time.time()
        
        logger.info("="*80)
        logger.info("REDUCER WORKFLOW STARTED")
        logger.info("="*80)
        
        # Extract all entities and keywords with frequencies
        all_entities = []
        all_keywords = []
        all_summaries = []
        all_key_points = []
        all_technical_terms = []
        
        total_pages = 0
        total_llm_successes = 0
        total_llm_failures = 0
        
        for sm_id, sm_result in mapper_results.items():
            if sm_result.get('status') != 'ok':
                continue
            
            output = sm_result.get('output', {})
            total_pages += output.get('total_pages', 0)
            total_llm_successes += output.get('llm_successes', 0)
            total_llm_failures += output.get('llm_failures', 0)
            
            # Collect aggregate summary
            agg_summary = output.get('aggregate_summary', '')
            if agg_summary and not agg_summary.startswith('[') and len(agg_summary) > 20:
                all_summaries.append(agg_summary)
            
            # Collect from individual page results
            for page_result in output.get('results', []):
                if page_result.get('status') == 'success':
                    all_entities.extend(page_result.get('entities', []))
                    all_keywords.extend(page_result.get('keywords', []))
                    all_key_points.extend(page_result.get('key_points', []))
                    all_technical_terms.extend(page_result.get('technical_terms', []))
        
        # Frequency analysis
        entity_counts = Counter(all_entities)
        keyword_counts = Counter(all_keywords)
        term_counts = Counter(all_technical_terms)
        
        # Top entities, keywords, terms
        top_entities = entity_counts.most_common(50)
        top_keywords = keyword_counts.most_common(30)
        top_terms = term_counts.most_common(20)
        
        # Deduplicate key points (case-insensitive)
        unique_key_points = list({kp.lower(): kp for kp in all_key_points}.values())[:30]
        
        # Generate consolidated summary
        consolidated_summary = self._generate_consolidated_summary(
            all_summaries, 
            top_entities[:10],
            top_keywords[:10],
            metadata
        )
        
        elapsed = time.time() - start_time
        self.monitor.log_stats()
        
        reduced_result = {
            "status": "completed",
            "document": {
                "file_name": metadata.get('file_name'),
                "total_pages": metadata.get('num_pages'),
                "pages_processed": total_pages,
                "document_type": metadata.get('document_type'),
            },
            "processing_stats": {
                "total_submasters": len(mapper_results),
                "llm_successes": total_llm_successes,
                "llm_failures": total_llm_failures,
                "success_rate": round((total_llm_successes / (total_llm_successes + total_llm_failures)) * 100, 1) if (total_llm_successes + total_llm_failures) > 0 else 0,
                "elapsed_time": elapsed
            },
            "consolidated_analysis": {
                "summary": consolidated_summary,
                "top_entities": [{"entity": e, "count": c} for e, c in top_entities],
                "top_keywords": [{"keyword": k, "count": c} for k, c in top_keywords],
                "top_technical_terms": [{"term": t, "count": c} for t, c in top_terms],
                "key_insights": unique_key_points[:15],
                "total_unique_entities": len(entity_counts),
                "total_unique_keywords": len(keyword_counts),
            },
            "raw_mapper_results": mapper_results,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Reducer completed in {elapsed:.2f}s")
        logger.info(f"   Top Entities: {len(top_entities)}, Top Keywords: {len(top_keywords)}")
        
        return reduced_result
    
    def _generate_consolidated_summary(
        self, 
        summaries: List[str], 
        top_entities: List[tuple],
        top_keywords: List[tuple],
        metadata: Dict[str, Any]
    ) -> str:
        """Generate a consolidated summary from all section summaries."""
        if not summaries:
            return "No analysis summaries available."
        
        doc_name = metadata.get('file_name', 'document')
        doc_type = metadata.get('document_type', 'document')
        
        # Extract key entities and keywords
        entity_names = [e for e, _ in top_entities[:5]]
        keyword_names = [k for k, _ in top_keywords[:5]]
        
        # Combine summaries intelligently
        if len(summaries) == 1:
            combined = summaries[0]
        elif len(summaries) <= 3:
            combined = " ".join(summaries)
        else:
            # Use first, middle, last for very long documents
            combined = f"{summaries[0]} ... {summaries[len(summaries)//2]} ... {summaries[-1]}"
        
        # Trim if too long
        if len(combined) > 800:
            combined = combined[:800] + "..."
        
        # Create consolidated summary
        summary = f"""
This {doc_type} ({doc_name}) has been analyzed across {metadata.get('num_pages', 'multiple')} pages. 
Key entities identified include: {', '.join(entity_names) if entity_names else 'N/A'}. 
Primary keywords: {', '.join(keyword_names) if keyword_names else 'N/A'}. 

{combined}
"""
        
        return summary.strip()
    
    def save_reduced_results(self, results: Dict[str, Any]) -> str:
        """Save reduced results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_name = os.path.splitext(results['document']['file_name'])[0]
        filename = f"{doc_name}_reduced_{timestamp}.json"
        path = os.path.join(self.output_dir, filename)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Reduced results saved: {path}")
        return path
    
    def execute(self, mapper_results: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete Reducer workflow."""
        reduced_results = self.reduce(mapper_results, metadata)
        output_path = self.save_reduced_results(reduced_results)
        
        reduced_results['output_path'] = output_path
        return reduced_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python reducer.py <mapper_results.json>")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        mapper_results = json.load(f)
    
    # Load metadata
    metadata_path = sys.argv[2] if len(sys.argv) > 2 else "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    reducer = Reducer()
    result = reducer.execute(mapper_results, metadata)
    
    print(json.dumps(result['consolidated_analysis'], indent=2))
