"""
Master Merger Agent - Final Document Synthesis

Takes complete outputs from all reducer submasters along with global context
from ResidualAgent and creates an extremely detailed, comprehensive final summary.

Run from project root:
    python -m agents.master_merger
"""

import os
import sys
import json
import time
from typing import Dict, Any, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ray
from pymongo import MongoClient

from utils.logger import get_logger
from utils.llm_helper import LLMProcessor

logger = get_logger("MasterMerger")


@ray.remote
class MasterMergerAgent:
    """
    Final synthesis agent that creates comprehensive document summary
    by merging all reducer submaster outputs with global context guidance.
    """

    def __init__(self, agent_id: str = "MASTER-MERGER-001", llm_model: str = None):
        self.agent_id = agent_id
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "mistral-small-latest")
        
        # Initialize LLM with higher token capacity for synthesis
        try:
            self.llm = LLMProcessor(
                model=self.llm_model,
                temperature=0.2,  # Lower temperature for consistent synthesis
                caller_id=self.agent_id
            )
            logger.info(f"[{self.agent_id}] LLM initialized: {self.llm_model}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to initialize LLM: {e}")
            raise

        # MongoDB connection
        uri = os.getenv("MONGO_URI")
        dbname = os.getenv("MONGO_DB")
        try:
            client = MongoClient(uri)
            db = client[dbname]
            self.master_coll = db[os.getenv("MONGO_MASTER_COLLECTION", "master_merger_results")]
            self.rsm_coll = db[os.getenv("MONGO_REDUCER_SUBMASTER_COLLECTION", "reducer_submaster_results")]
            logger.info(f"[{self.agent_id}] Connected to MongoDB: {dbname}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] MongoDB connection failed: {e}")
            raise

        logger.info(f"[{self.agent_id}] MasterMergerAgent initialized")

    def synthesize_final_document(
        self,
        reducer_results: Dict[str, Any],
        global_context: Dict[str, Any],
        processing_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create comprehensive final synthesis from all inputs.
        OPTIMIZED: Uses only 1 LLM call for speed.
        """
        logger.info(f"[{self.agent_id}] Starting final document synthesis (FAST MODE)")
        start_time = time.time()
        
        # Load individual RSM outputs
        rsm_outputs = self._load_rsm_outputs(reducer_results.get("rsm_results", []))
        
        # FAST MODE: Single LLM call for executive summary + insights
        # All other sections use direct data extraction (no LLM)
        combined_result = self._create_combined_synthesis(
            reducer_results, global_context, processing_plan
        )
        
        # Extract sections from direct data (no LLM)
        detailed_synthesis = self._create_detailed_synthesis(
            rsm_outputs, global_context, processing_plan
        )
        
        # Extract metadata (no LLM)
        metadata = self._extract_comprehensive_metadata(reducer_results, global_context)
        
        # Compile final output
        final_output = {
            "agent_id": self.agent_id,
            "timestamp": time.time(),
            "processing_time": time.time() - start_time,
            "executive_summary": combined_result.get("executive_summary", ""),
            "detailed_synthesis": detailed_synthesis,
            "metadata": metadata,
            "insights_and_conclusions": combined_result.get("insights_and_conclusions", {}),
            "global_context_used": global_context,
            "processing_plan_used": processing_plan,
            "source_statistics": {
                "num_reducer_submasters": len(rsm_outputs),
                "total_entities": len(reducer_results.get("entities", {})),
                "total_keywords": len(reducer_results.get("keywords", {})),
                "total_key_points": len(reducer_results.get("key_points", [])),
                "total_insights": len(reducer_results.get("insights", []))
            }
        }
        
        # Persist to MongoDB
        self._persist_final_output(final_output)
        
        logger.info(f"[{self.agent_id}] Final synthesis complete in {final_output['processing_time']:.2f}s")
        
        return final_output

    def _create_combined_synthesis(
        self,
        reducer_results: Dict[str, Any],
        global_context: Dict[str, Any],
        processing_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        SINGLE LLM call to generate executive summary + insights.
        This replaces multiple separate calls for speed.
        """
        # Prepare compact input
        summary = reducer_results.get('final_summary', '')[:1000]
        themes = global_context.get('cross_document_themes', [])[:5]
        key_points = reducer_results.get('key_points', [])[:10]
        insights = reducer_results.get('insights', [])[:10]
        top_entities = list(dict(sorted(
            reducer_results.get('entities', {}).items(), 
            key=lambda x: x[1], reverse=True
        )[:8]).keys())
        
        prompt = f"""Analyze this document and provide a structured response.

DOCUMENT SUMMARY: {summary}
THEMES: {themes}
KEY POINTS: {key_points}
INSIGHTS: {insights}
TOP ENTITIES: {top_entities}

Provide JSON with:
{{
  "executive_summary": "200-300 word professional summary covering main themes, findings, and significance",
  "key_findings": ["finding1", "finding2", ...],  // 5 main findings
  "conclusions": "100 word conclusion",
  "recommendations": ["rec1", "rec2", "rec3"]  // 3 recommendations
}}
"""
        
        try:
            result = self.llm.call_with_retry(prompt, parse_json=True, max_tokens=2048)
            if isinstance(result, dict):
                return {
                    "executive_summary": result.get("executive_summary", summary[:500]),
                    "insights_and_conclusions": {
                        "key_findings": result.get("key_findings", key_points[:5]),
                        "conclusions": result.get("conclusions", "Analysis complete."),
                        "recommendations": result.get("recommendations", []),
                        "implications": [],
                        "future_directions": []
                    }
                }
        except Exception as e:
            logger.warning(f"[{self.agent_id}] Combined synthesis failed, using fallback: {e}")
        
        # Fallback: use existing data directly
        return {
            "executive_summary": summary[:500] if summary else "Document analysis complete.",
            "insights_and_conclusions": {
                "key_findings": [str(kp) for kp in key_points[:5]],
                "conclusions": "Analysis complete. See detailed sections for findings.",
                "recommendations": [],
                "implications": [],
                "future_directions": []
            }
        }

    def _load_rsm_outputs(self, rsm_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Load individual RSM outputs from MongoDB for detailed processing."""
        outputs = []
        for rsm in rsm_results:
            rsm_id = rsm.get("rsm_id")
            try:
                doc = self.rsm_coll.find_one({"rsm_id": rsm_id}, {"_id": 0})
                if doc:
                    outputs.append(doc)
                else:
                    outputs.append(rsm)  # Use what we have
            except Exception as e:
                logger.warning(f"Failed to load RSM {rsm_id}: {e}")
                outputs.append(rsm)
        return outputs

    def _create_executive_summary(
        self,
        reducer_results: Dict[str, Any],
        global_context: Dict[str, Any],
        processing_plan: Dict[str, Any]
    ) -> str:
        """Generate high-level executive summary - OPTIMIZED for speed."""
        
        # OPTIMIZATION: Limit context size to reduce tokens
        context_str = json.dumps(global_context, indent=2)[:1500]  # Limit context
        
        prompt = f"""Create an executive summary (300-400 words) for this document analysis.

CONTEXT:
{context_str}

SUMMARY: {reducer_results.get('final_summary', '')[:800]}

THEMES: {global_context.get('cross_document_themes', [])[:5]}

ENTITIES: {list(dict(sorted(reducer_results.get('entities', {}).items(), key=lambda x: x[1], reverse=True)[:8]).keys())}

Write a professional summary covering: overview, main themes, key entities, technical significance, and patterns.
"""
        
        try:
            summary = self.llm.call_with_retry(prompt, parse_json=False, max_tokens=2048)
            return summary.strip()
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to create executive summary: {e}")
            return reducer_results.get('final_summary', '')[:500]

    def _create_detailed_synthesis(
        self,
        rsm_outputs: List[Dict[str, Any]],
        global_context: Dict[str, Any],
        processing_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create detailed section-by-section synthesis - FAST MODE (NO LLM calls)."""
        
        synthesis = {
            "sections": [],
            "cross_section_analysis": "",
            "technical_deep_dive": ""
        }
        
        # FAST MODE: Process ALL sections without LLM - just extract existing data
        for rsm in rsm_outputs:
            section = self._synthesize_sections_batch([rsm], global_context, processing_plan)
            synthesis["sections"].extend(section)
        
        # FAST MODE: Use lite versions (no LLM calls)
        synthesis["cross_section_analysis"] = self._create_cross_section_analysis_lite(
            rsm_outputs, global_context
        )
        synthesis["technical_deep_dive"] = self._create_technical_deep_dive_lite(
            rsm_outputs, global_context
        )
        
        return synthesis

    def _synthesize_sections_batch(
        self,
        rsm_batch: List[Dict[str, Any]],
        global_context: Dict[str, Any],
        processing_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Synthesize multiple RSM outputs - FAST MODE: Skip LLM, use direct extraction."""
        
        # OPTIMIZATION: Skip LLM entirely for speed - just extract and format existing data
        # This is much faster and avoids rate limiting/JSON parsing issues
        sections = []
        for rsm in rsm_batch:
            rsm_id = rsm.get("rsm_id", "Unknown")
            output = rsm.get("output", {})
            
            # Use existing enhanced_summary directly
            summary = output.get('enhanced_summary', '')
            if not summary:
                # Fallback to combining key points
                key_points = output.get('key_points', [])
                summary = '. '.join(str(kp) for kp in key_points[:5]) if key_points else "No summary available."
            
            sections.append({
                "section_id": rsm_id,
                "synthesis": summary[:400],  # Limit length
                "key_entities": list(dict(sorted(
                    output.get('entities', {}).items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]).keys()),
                "main_themes": output.get('key_points', [])[:3]
            })
        
        return sections

    def _create_technical_deep_dive_lite(
        self,
        rsm_outputs: List[Dict[str, Any]],
        global_context: Dict[str, Any]
    ) -> str:
        """Lightweight technical summary without LLM call."""
        # Extract key technical terms from all sections
        all_tech_terms = {}
        for rsm in rsm_outputs:
            output = rsm.get("output", {})
            for term, count in output.get("technical_terms", {}).items():
                all_tech_terms[term] = all_tech_terms.get(term, 0) + count
        
        top_terms = sorted(all_tech_terms.items(), key=lambda x: x[1], reverse=True)[:15]
        
        return f"""Technical Overview:
This document covers {len(rsm_outputs)} major sections with the following key technical concepts:
{', '.join([f"{term} ({count})" for term, count in top_terms[:10]])}.

Primary technical focus areas identified across the document include {', '.join([t[0] for t in top_terms[:5]])}.
"""

    def _create_cross_section_analysis_lite(
        self,
        rsm_outputs: List[Dict[str, Any]],
        global_context: Dict[str, Any]
    ) -> str:
        """Lightweight cross-section analysis without LLM call."""
        all_themes = []
        all_entities = {}
        
        for rsm in rsm_outputs:
            output = rsm.get("output", {})
            all_themes.extend(output.get("key_points", []))
            for entity, count in output.get("entities", {}).items():
                all_entities[entity] = all_entities.get(entity, 0) + count
        
        top_entities = sorted(all_entities.items(), key=lambda x: x[1], reverse=True)[:10]
        unique_themes = list(set(str(t) for t in all_themes))[:8]
        
        themes_from_context = global_context.get('cross_document_themes', [])[:5]
        
        return f"""Cross-Section Analysis:
The document spans {len(rsm_outputs)} major sections with interconnected themes.

Key Recurring Entities: {', '.join([f"{e[0]} ({e[1]})" for e in top_entities[:8]])}.

Common Themes: {', '.join(unique_themes)}.

Document-Wide Patterns: {', '.join(str(t) for t in themes_from_context) if themes_from_context else 'Multiple interconnected topics identified'}.
"""

    def _synthesize_section(
        self,
        rsm_output: Dict[str, Any],
        global_context: Dict[str, Any],
        processing_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize a single RSM output into detailed section."""
        
        rsm_id = rsm_output.get("rsm_id", "Unknown")
        output = rsm_output.get("output", {})
        
        prompt = f"""
Create a detailed synthesis of this document section.

**SECTION ID:** {rsm_id}

**ENHANCED SUMMARY:**
{output.get('enhanced_summary', '')}

**KEY POINTS:**
{output.get('key_points', [])}

**INSIGHTS:**
{output.get('insights', [])}

**TOP ENTITIES:**
{list(dict(sorted(output.get('entities', {}).items(), key=lambda x: x[1], reverse=True)[:10]).keys())}

**GLOBAL CONTEXT:**
Themes: {global_context.get('cross_document_themes', [])}
Strategy: {processing_plan.get('synthesis_strategy', '')}

**TASK:**
Create a comprehensive section synthesis (400-600 words) that:
1. Summarizes main content with technical precision
2. Highlights relationships to other sections
3. Emphasizes key findings and evidence
4. Notes important entities and their significance
5. Identifies patterns and implications

Be specific, detailed, and technically accurate. Include all relevant information.
"""
        
        try:
            section_text = self.llm.call_with_retry(prompt, parse_json=False, max_tokens=2048)
            
            return {
                "section_id": rsm_id,
                "synthesis": section_text.strip(),
                "key_entities": list(dict(sorted(
                    output.get('entities', {}).items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]).keys()),
                "main_themes": output.get('key_points', [])[:5]
            }
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to synthesize section {rsm_id}: {e}")
            return {
                "section_id": rsm_id,
                "synthesis": output.get('enhanced_summary', ''),
                "key_entities": [],
                "main_themes": []
            }

    def _create_cross_section_analysis(
        self,
        rsm_outputs: List[Dict[str, Any]],
        global_context: Dict[str, Any]
    ) -> str:
        """Analyze patterns and connections across all sections."""
        
        all_themes = []
        all_entities = {}
        
        for rsm in rsm_outputs:
            output = rsm.get("output", {})
            all_themes.extend(output.get("key_points", []))
            for entity, count in output.get("entities", {}).items():
                all_entities[entity] = all_entities.get(entity, 0) + count
        
        top_entities = dict(sorted(all_entities.items(), key=lambda x: x[1], reverse=True)[:10])
        
        prompt = f"""Analyze cross-cutting patterns across document sections (200-300 words).

Themes: {global_context.get('cross_document_themes', [])[:5]}
Key Entities: {list(top_entities.keys())[:8]}
Common Topics: {list(set(all_themes))[:10]}

Focus on: patterns across sections, entity relationships, theme evolution, and key connections.
"""
        
        try:
            analysis = self.llm.call_with_retry(prompt, parse_json=False, max_tokens=1500)
            return analysis.strip()
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed cross-section analysis: {e}")
            return "Cross-section analysis unavailable."

    def _create_technical_deep_dive(
        self,
        rsm_outputs: List[Dict[str, Any]],
        global_context: Dict[str, Any]
    ) -> str:
        """Create technical deep dive into key concepts - OPTIMIZED."""
        
        all_technical_terms = {}
        for rsm in rsm_outputs:
            output = rsm.get("output", {})
            for term, count in output.get("technical_terms", {}).items():
                all_technical_terms[term] = all_technical_terms.get(term, 0) + count
        
        top_technical = dict(sorted(all_technical_terms.items(), key=lambda x: x[1], reverse=True)[:12])
        
        prompt = f"""Technical deep dive (200-300 words) on key concepts.

Technical Concepts: {global_context.get('technical_concepts', [])[:5]}
Top Terms: {list(top_technical.keys())}

Cover: key concepts, methodologies, technical relationships, and significance.
"""
        
        try:
            deep_dive = self.llm.call_with_retry(prompt, parse_json=False, max_tokens=1500)
            return deep_dive.strip()
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed technical deep dive: {e}")
            return "Technical deep dive unavailable."

    def _extract_comprehensive_metadata(
        self,
        reducer_results: Dict[str, Any],
        global_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract and organize comprehensive metadata - OPTIMIZED (no LLM call)."""
        
        entities = reducer_results.get("entities", {})
        keywords = reducer_results.get("keywords", {})
        technical_terms = reducer_results.get("technical_terms", {})
        
        # OPTIMIZATION: Limit to top 15 instead of 30
        return {
            "top_entities": dict(sorted(entities.items(), key=lambda x: x[1], reverse=True)[:15]),
            "top_keywords": dict(sorted(keywords.items(), key=lambda x: x[1], reverse=True)[:15]),
            "top_technical_terms": dict(sorted(technical_terms.items(), key=lambda x: x[1], reverse=True)[:15]),
            "document_themes": global_context.get("cross_document_themes", [])[:5],
            "technical_concepts": global_context.get("technical_concepts", [])[:5],
            "total_unique_entities": len(entities),
            "total_unique_keywords": len(keywords),
            "total_unique_technical_terms": len(technical_terms)
        }

    def _generate_insights_conclusions(
        self,
        reducer_results: Dict[str, Any],
        global_context: Dict[str, Any],
        processing_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate final insights and conclusions - OPTIMIZED."""
        
        all_insights = reducer_results.get("insights", [])[:20]  # Limit input
        key_points = reducer_results.get("key_points", [])[:20]  # Limit input
        
        prompt = f"""Generate insights and conclusions (JSON format).

Insights: {all_insights[:15]}
Key Points: {key_points[:15]}
Themes: {global_context.get('cross_document_themes', [])[:3]}

Return JSON:
{{
  "key_findings": ["..."],  // 5-7 findings
  "conclusions": "150-200 word conclusion",
  "implications": ["..."],  // 3-4 implications
  "recommendations": ["..."],  // 3-4 recommendations
  "future_directions": ["..."]  // 2-3 directions
}}
"""
        
        try:
            result = self.llm.call_with_retry(prompt, parse_json=True, max_tokens=1500)
            return result
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to generate insights: {e}")
            return {
                "key_findings": key_points[:5],
                "conclusions": "Analysis complete.",
                "implications": [],
                "recommendations": [],
                "future_directions": []
            }

    def _persist_final_output(self, final_output: Dict[str, Any]):
        """Save final output to MongoDB."""
        try:
            self.master_coll.insert_one(final_output)
            logger.info(f"[{self.agent_id}] Final output persisted to MongoDB")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to persist final output: {e}")


def run_master_merger_standalone():
    """
    Standalone runner that loads reducer results and residual context,
    then creates final synthesis.
    """
    logger.info("Starting MasterMerger standalone mode")
    
    # Connect to MongoDB
    uri = os.getenv("MONGO_URI")
    dbname = os.getenv("MONGO_DB")
    
    if not uri or not dbname:
        raise RuntimeError("MONGO_URI and MONGO_DB must be set")
    
    client = MongoClient(uri)
    db = client[dbname]
    
    # Load latest reducer results
    reducer_coll = db[os.getenv("MONGO_REDUCER_RESULTS_COLLECTION", "reducer_results")]
    latest_reducer = reducer_coll.find_one(sort=[("timestamp", -1)], projection={"_id": 0})
    
    if not latest_reducer:
        logger.error("No reducer results found")
        return None
    
    # Load latest residual context
    residual_coll = db[os.getenv("MONGO_RESIDUAL_COLLECTION", "residual_memory")]
    latest_residual = residual_coll.find_one(sort=[("timestamp", -1)], projection={"_id": 0})
    
    if not latest_residual:
        logger.warning("No residual context found, using empty context")
        global_context = {}
        processing_plan = {}
    else:
        global_context = latest_residual.get("global_context", {})
        # Get processing plan (would normally be generated by residual agent)
        processing_plan = {
            "synthesis_strategy": "Comprehensive integration of all sections",
            "key_focus_areas": global_context.get("cross_document_themes", [])
        }
    
    logger.info("Loaded reducer results and residual context")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Create MasterMerger
    merger = MasterMergerAgent.remote()
    
    # Synthesize final document
    result_future = merger.synthesize_final_document.remote(
        latest_reducer, global_context, processing_plan
    )
    final_result = ray.get(result_future)
    
    logger.info("MasterMerger processing complete")
    
    return final_result


def main():
    """Main entry point."""
    try:
        result = run_master_merger_standalone()
        
        if result:
            print("\n=== MASTER MERGER FINAL OUTPUT ===\n")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()