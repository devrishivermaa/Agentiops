"""
Merge Supervisor: Combines Mapper and Reducer outputs into final consolidated report.
This is the green robot in Architecture 1 diagram.
"""

import os
import json
import time
from typing import Dict, Any, List
from datetime import datetime
from utils.logger import get_logger
from utils.llm_helper import LLMProcessor

logger = get_logger("MergeSupervisor")


class MergeSupervisor:
    """
    MergeSupervisor combines Mapper stage results with Reducer aggregations
    to produce the final unified analysis report.
    
    Architecture Role: Green "Merge Supervisor" robot combining both pipelines
    """
    
    def __init__(self, use_llm: bool = True):
        """
        Initialize Merge Supervisor.
        
        Args:
            use_llm: If True, use LLM to generate enhanced final summary
        """
        self.use_llm = use_llm
        self.llm = None
        
        if use_llm:
            try:
                self.llm = LLMProcessor(
                    model="mistral-small-latest",
                    temperature=0.3,
                    max_retries=3,
                    caller_id="MergeSupervisor"
                )
                logger.info("MergeSupervisor initialized with LLM enhancement")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM, falling back to rule-based merge: {e}")
                self.use_llm = False
        else:
            logger.info("MergeSupervisor initialized (rule-based mode)")
    
    def merge(
        self, 
        mapper_results: Dict[str, Any], 
        reducer_results: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge Mapper and Reducer outputs into final report.
        
        Args:
            mapper_results: Raw SubMaster results from orchestrator
            reducer_results: Consolidated analysis from Reducer
            metadata: Document metadata
            
        Returns:
            Final merged analysis report
        """
        start_time = time.time()
        
        logger.info("="*80)
        logger.info("MERGE SUPERVISOR: Combining Mapper + Reducer outputs")
        logger.info("="*80)
        
        # Extract key data from both stages
        consolidated = reducer_results.get('consolidated_analysis', {})
        raw_mapper = reducer_results.get('raw_mapper_results', mapper_results)
        
        # Build comprehensive entity and keyword maps
        entity_map = self._build_entity_map(raw_mapper)
        keyword_map = self._build_keyword_map(raw_mapper)
        
        # Generate section-wise analysis
        section_analysis = self._generate_section_analysis(raw_mapper, metadata)
        
        # Create final summary (with optional LLM enhancement)
        final_summary = self._generate_final_summary(
            consolidated.get('summary', ''),
            consolidated.get('top_entities', []),
            consolidated.get('top_keywords', []),
            section_analysis,
            metadata
        )
        
        # Build final merged report
        merged_report = {
            "document_info": {
                "file_name": metadata.get('file_name'),
                "document_type": metadata.get('document_type'),
                "total_pages": metadata.get('num_pages'),
                "file_size_mb": metadata.get('file_size_mb'),
                "processing_date": datetime.now().isoformat()
            },
            
            "executive_summary": final_summary,
            
            "key_findings": {
                "top_entities": consolidated.get('top_entities', [])[:20],
                "top_keywords": consolidated.get('top_keywords', [])[:20],
                "top_technical_terms": consolidated.get('top_technical_terms', [])[:15],
                "key_insights": consolidated.get('key_insights', [])[:10]
            },
            
            "section_analysis": section_analysis,
            
            "detailed_entity_analysis": entity_map,
            "detailed_keyword_analysis": keyword_map,
            
            "processing_statistics": {
                "mapper_stats": reducer_results.get('processing_stats', {}),
                "total_unique_entities": consolidated.get('total_unique_entities', 0),
                "total_unique_keywords": consolidated.get('total_unique_keywords', 0),
                "merge_time": 0  # Will be updated below
            },
            
            "quality_metrics": self._calculate_quality_metrics(
                reducer_results.get('processing_stats', {}),
                consolidated
            ),
            
            "raw_data": {
                "mapper_results": raw_mapper,
                "reducer_results": reducer_results
            },
            
            "metadata": metadata
        }
        
        elapsed = time.time() - start_time
        merged_report['processing_statistics']['merge_time'] = round(elapsed, 2)
        
        logger.info(f"✅ Merge completed in {elapsed:.2f}s")
        logger.info(f"   Final entities: {len(entity_map)}, Final keywords: {len(keyword_map)}")
        
        return merged_report
    
    def _build_entity_map(self, mapper_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive entity map with context."""
        entity_data = {}
        
        for sm_id, sm_result in mapper_results.items():
            if sm_result.get('status') != 'ok':
                continue
            
            output = sm_result.get('output', {})
            
            for page_result in output.get('results', []):
                page_num = page_result.get('page')
                section = page_result.get('section', 'Unknown')
                
                for entity in page_result.get('entities', []):
                    if entity not in entity_data:
                        entity_data[entity] = {
                            "entity": entity,
                            "frequency": 0,
                            "sections": set(),
                            "pages": []
                        }
                    
                    entity_data[entity]['frequency'] += 1
                    entity_data[entity]['sections'].add(section)
                    if page_num not in entity_data[entity]['pages']:
                        entity_data[entity]['pages'].append(page_num)
        
        # Convert sets to lists for JSON serialization
        for entity in entity_data:
            entity_data[entity]['sections'] = list(entity_data[entity]['sections'])
            entity_data[entity]['pages'].sort()
        
        return entity_data
    
    def _build_keyword_map(self, mapper_results: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive keyword map with context."""
        keyword_data = {}
        
        for sm_id, sm_result in mapper_results.items():
            if sm_result.get('status') != 'ok':
                continue
            
            output = sm_result.get('output', {})
            
            for page_result in output.get('results', []):
                page_num = page_result.get('page')
                section = page_result.get('section', 'Unknown')
                
                for keyword in page_result.get('keywords', []):
                    if keyword not in keyword_data:
                        keyword_data[keyword] = {
                            "keyword": keyword,
                            "frequency": 0,
                            "sections": set(),
                            "pages": []
                        }
                    
                    keyword_data[keyword]['frequency'] += 1
                    keyword_data[keyword]['sections'].add(section)
                    if page_num not in keyword_data[keyword]['pages']:
                        keyword_data[keyword]['pages'].append(page_num)
        
        # Convert sets to lists
        for keyword in keyword_data:
            keyword_data[keyword]['sections'] = list(keyword_data[keyword]['sections'])
            keyword_data[keyword]['pages'].sort()
        
        return keyword_data
    
    def _generate_section_analysis(
        self, 
        mapper_results: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate per-section analysis summary."""
        sections = metadata.get('sections', {})
        section_summaries = {}
        
        for section_name in sections.keys():
            section_summaries[section_name] = {
                "section_name": section_name,
                "page_range": f"{sections[section_name].get('page_start')}-{sections[section_name].get('page_end')}",
                "summaries": [],
                "entities": set(),
                "keywords": set()
            }
        
        # Aggregate data by section
        for sm_id, sm_result in mapper_results.items():
            if sm_result.get('status') != 'ok':
                continue
            
            output = sm_result.get('output', {})
            assigned_sections = output.get('assigned_sections', [])
            aggregate_summary = output.get('aggregate_summary', '')
            
            for section_name in assigned_sections:
                if section_name in section_summaries:
                    if aggregate_summary and len(aggregate_summary) > 20:
                        section_summaries[section_name]['summaries'].append(aggregate_summary)
            
            # Collect entities/keywords by section
            for page_result in output.get('results', []):
                section = page_result.get('section', 'Unknown')
                if section in section_summaries:
                    section_summaries[section]['entities'].update(page_result.get('entities', []))
                    section_summaries[section]['keywords'].update(page_result.get('keywords', []))
        
        # Convert sets to lists and combine summaries
        for section_name in section_summaries:
            section_summaries[section_name]['entities'] = list(section_summaries[section_name]['entities'])
            section_summaries[section_name]['keywords'] = list(section_summaries[section_name]['keywords'])
            
            summaries = section_summaries[section_name]['summaries']
            if summaries:
                combined = " ".join(summaries)
                if len(combined) > 500:
                    combined = combined[:500] + "..."
                section_summaries[section_name]['combined_summary'] = combined
            else:
                section_summaries[section_name]['combined_summary'] = "No analysis available for this section."
            
            del section_summaries[section_name]['summaries']
        
        return section_summaries
    
    def _generate_final_summary(
        self,
        reducer_summary: str,
        top_entities: List[Dict],
        top_keywords: List[Dict],
        section_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Generate final executive summary (optionally LLM-enhanced)."""
        
        if not self.use_llm or not self.llm:
            # Rule-based summary
            return self._rule_based_summary(
                reducer_summary, top_entities, top_keywords, metadata
            )
        
        # LLM-enhanced summary
        try:
            prompt = self._create_summary_prompt(
                reducer_summary, top_entities, top_keywords, section_analysis, metadata
            )
            
            enhanced_summary = self.llm.call_with_retry(prompt, parse_json=False)
            logger.info("✅ LLM-enhanced summary generated")
            return enhanced_summary
            
        except Exception as e:
            logger.warning(f"LLM summary failed, falling back to rule-based: {e}")
            return self._rule_based_summary(
                reducer_summary, top_entities, top_keywords, metadata
            )
    
    def _rule_based_summary(
        self,
        reducer_summary: str,
        top_entities: List[Dict],
        top_keywords: List[Dict],
        metadata: Dict[str, Any]
    ) -> str:
        """Generate summary using rules (no LLM)."""
        doc_name = metadata.get('file_name', 'document')
        doc_type = metadata.get('document_type', 'document')
        num_pages = metadata.get('num_pages', 'multiple')
        
        entity_list = [e['entity'] for e in top_entities[:5]]
        keyword_list = [k['keyword'] for k in top_keywords[:5]]
        
        summary = f"""
EXECUTIVE SUMMARY

Document: {doc_name}
Type: {doc_type.upper()}
Pages: {num_pages}

TOP ENTITIES: {', '.join(entity_list) if entity_list else 'N/A'}
TOP KEYWORDS: {', '.join(keyword_list) if keyword_list else 'N/A'}

ANALYSIS:
{reducer_summary}

This analysis was generated by the AgentOps multi-agent document processing pipeline.
"""
        return summary.strip()
    
    def _create_summary_prompt(
        self,
        reducer_summary: str,
        top_entities: List[Dict],
        top_keywords: List[Dict],
        section_analysis: Dict[str, Any],
        metadata: Dict[str, Any]
    ) -> str:
        """Create prompt for LLM-enhanced summary."""
        entity_list = [f"{e['entity']} ({e['count']})" for e in top_entities[:10]]
        keyword_list = [f"{k['keyword']} ({k['count']})" for k in top_keywords[:10]]
        
        section_texts = []
        for section_name, section_data in section_analysis.items():
            section_texts.append(f"{section_name}: {section_data.get('combined_summary', 'N/A')[:200]}")
        
        prompt = f"""You are an expert research analyst. Generate a comprehensive executive summary for this document.

DOCUMENT: {metadata.get('file_name')}
TYPE: {metadata.get('document_type', 'research paper')}
PAGES: {metadata.get('num_pages')}

TOP ENTITIES (with frequency):
{chr(10).join(entity_list)}

TOP KEYWORDS (with frequency):
{chr(10).join(keyword_list)}

SECTION SUMMARIES:
{chr(10).join(section_texts)}

REDUCER ANALYSIS:
{reducer_summary[:1000]}

Generate a well-structured executive summary (300-500 words) covering:
1. Main topic and purpose of the document
2. Key methodologies or approaches discussed
3. Primary findings or contributions
4. Important entities, concepts, and technical terms
5. Overall significance and conclusions

Write in clear, professional language suitable for a technical audience.
"""
        
        return prompt
    
    def _calculate_quality_metrics(
        self,
        processing_stats: Dict[str, Any],
        consolidated: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate quality metrics for the analysis."""
        success_rate = processing_stats.get('success_rate', 0)
        
        # Coverage score (how much content was analyzed)
        total_entities = consolidated.get('total_unique_entities', 0)
        total_keywords = consolidated.get('total_unique_keywords', 0)
        
        coverage_score = min(100, (total_entities + total_keywords) / 2)
        
        # Overall quality score
        quality_score = (success_rate * 0.7) + (coverage_score * 0.3)
        
        return {
            "success_rate": success_rate,
            "coverage_score": round(coverage_score, 1),
            "overall_quality_score": round(quality_score, 1),
            "quality_rating": self._get_quality_rating(quality_score)
        }
    
    def _get_quality_rating(self, score: float) -> str:
        """Convert quality score to rating."""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        else:
            return "Poor"
    
    def save_merged_report(self, merged_report: Dict[str, Any], output_dir: str = "./output") -> str:
        """Save merged report to JSON."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_name = os.path.splitext(merged_report['document_info']['file_name'])[0]
        filename = f"{doc_name}_final_report_{timestamp}.json"
        path = os.path.join(output_dir, filename)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(merged_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Final merged report saved: {path}")
        return path


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python merge_supervisor.py <mapper_results.json> <reducer_results.json> [metadata.json]")
        sys.exit(1)
    
    with open(sys.argv[1], 'r') as f:
        mapper_results = json.load(f)
    
    with open(sys.argv[2], 'r') as f:
        reducer_results = json.load(f)
    
    metadata_path = sys.argv[3] if len(sys.argv) > 3 else "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    supervisor = MergeSupervisor(use_llm=True)
    merged = supervisor.merge(mapper_results, reducer_results, metadata)
    
    print("\n" + "="*80)
    print("FINAL EXECUTIVE SUMMARY")
    print("="*80)
    print(merged['executive_summary'])
    print("\n" + "="*80)
    print(f"Quality Score: {merged['quality_metrics']['overall_quality_score']}/100 ({merged['quality_metrics']['quality_rating']})")
    print("="*80)
