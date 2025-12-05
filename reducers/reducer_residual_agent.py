"""
Residual Agent - Global Context Manager

Maintains global context across the entire pipeline by:
1. Reading brief summaries from each ReducerSubMaster
2. Tracking cross-document themes and patterns
3. Creating strategic plans for future processing
4. Maintaining evolving understanding of the document corpus

Run from project root:
    python -m agents.residual_agent
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

logger = get_logger("ResidualAgent")


@ray.remote
class ResidualAgent:
    """
    Manages global context and cross-document understanding.
    Reads brief summaries from reducer submasters and maintains
    an evolving knowledge state.
    """

    def __init__(self, agent_id: str = "RESIDUAL-001", llm_model: str = None):
        self.agent_id = agent_id
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "mistral-small-latest")
        
        # Initialize LLM
        try:
            self.llm = LLMProcessor(
                model=self.llm_model,
                temperature=0.3,
                caller_id=self.agent_id
            )
            logger.info(f"[{self.agent_id}] LLM initialized: {self.llm_model}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to initialize LLM: {e}")
            raise

        # Global context state
        self.global_context = {
            "document_overview": "",
            "cross_document_themes": [],
            "key_entities_global": {},
            "technical_concepts": [],
            "processing_strategy": "",
            "insights_accumulated": [],
            "section_summaries": []
        }

        # MongoDB connection
        uri = os.getenv("MONGO_URI")
        dbname = os.getenv("MONGO_DB")
        try:
            client = MongoClient(uri)
            db = client[dbname]
            self.residual_coll = db[os.getenv("MONGO_REDUCER_RESIDUAL_COLLECTION", "reducer_residual_memory")]
            logger.info(f"[{self.agent_id}] Connected to MongoDB: {dbname}")
        except Exception as e:
            logger.error(f"[{self.agent_id}] MongoDB connection failed: {e}")
            raise

        logger.info(f"[{self.agent_id}] ResidualAgent initialized")

    def update_context_from_reducer_results(
        self, 
        reducer_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update global context based on reducer submaster results.
        
        Args:
            reducer_results: Aggregated results from all reducer submasters
            
        Returns:
            Updated global context
        """
        logger.info(f"[{self.agent_id}] Updating context from reducer results")
        
        # Extract brief summaries from reducer submasters
        brief_summaries = reducer_results.get("brief_summaries", [])
        overall_brief = reducer_results.get("overall_brief_summary", "")
        
        # Collect top entities and keywords
        all_entities = reducer_results.get("entities", {})
        all_keywords = reducer_results.get("keywords", {})
        
        # Build context update prompt
        context_prompt = self._build_context_update_prompt(
            brief_summaries=brief_summaries,
            overall_brief=overall_brief,
            entities=all_entities,
            keywords=all_keywords
        )
        
        # Get LLM analysis
        try:
            analysis = self.llm.call_with_retry(context_prompt, parse_json=True)
            
            # Update global context
            self.global_context.update({
                "document_overview": analysis.get("document_overview", overall_brief),
                "cross_document_themes": analysis.get("themes", []),
                "key_entities_global": self._top_n_dict(all_entities, 20),
                "technical_concepts": analysis.get("technical_concepts", []),
                "processing_strategy": analysis.get("processing_strategy", ""),
                "insights_accumulated": analysis.get("insights", []),
                "section_summaries": brief_summaries,
                "last_updated": time.time()
            })
            
            logger.info(f"[{self.agent_id}] Global context updated successfully")
            
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to update context: {e}")
            # Fallback update
            self.global_context.update({
                "document_overview": overall_brief,
                "section_summaries": brief_summaries,
                "key_entities_global": self._top_n_dict(all_entities, 20),
                "last_updated": time.time()
            })
        
        # Persist to MongoDB
        self._persist_context()
        
        return self.global_context

    def _build_context_update_prompt(
        self,
        brief_summaries: List[Dict[str, Any]],
        overall_brief: str,
        entities: Dict[str, int],
        keywords: Dict[str, int]
    ) -> str:
        """Build prompt for LLM to analyze and update global context."""
        
        summaries_text = "\n\n".join([
            f"**Section {s.get('rsm_id')}:**\n{s.get('brief_summary', '')}"
            for s in brief_summaries
        ])
        
        top_entities = self._top_n_dict(entities, 15)
        top_keywords = self._top_n_dict(keywords, 15)
        
        prompt = f"""
You are a Residual Agent maintaining global context across a document analysis pipeline.

**OVERALL DOCUMENT BRIEF:**
{overall_brief}

**SECTION SUMMARIES:**
{summaries_text}

**TOP ENTITIES:** {list(top_entities.keys())}
**TOP KEYWORDS:** {list(top_keywords.keys())}

**YOUR TASK:**
Analyze the above information and provide a comprehensive global context update.

Return a JSON object with:
{{
  "document_overview": "2-3 sentence high-level overview of the entire document corpus",
  "themes": ["theme1", "theme2", ...],  // Cross-cutting themes across all sections
  "technical_concepts": ["concept1", "concept2", ...],  // Key technical concepts
  "processing_strategy": "Strategy for how the master merger should synthesize this information",
  "insights": ["insight1", "insight2", ...]  // Key insights for downstream processing
}}

Focus on:
- Cross-document patterns and connections
- Hierarchical relationships between concepts
- Strategic guidance for final synthesis
- Technical depth and domain specificity
"""
        return prompt

    def _top_n_dict(self, d: Dict[str, int], n: int) -> Dict[str, int]:
        """Return top N items from dictionary by value."""
        sorted_items = sorted(d.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_items[:n])

    def _persist_context(self):
        """Save current global context to MongoDB."""
        try:
            doc = {
                "agent_id": self.agent_id,
                "timestamp": time.time(),
                "global_context": self.global_context
            }
            self.residual_coll.insert_one(doc)
            logger.info(f"[{self.agent_id}] Global context persisted to MongoDB")
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to persist context: {e}")

    def get_context(self) -> Dict[str, Any]:
        """Return current global context."""
        return self.global_context

    def create_processing_plan(self) -> Dict[str, Any]:
        """
        Create a strategic plan for master merger based on current context.
        """
        logger.info(f"[{self.agent_id}] Creating processing plan for master merger")
        
        plan_prompt = f"""
Based on the following global context, create a detailed processing plan for the Master Merger agent.

**GLOBAL CONTEXT:**
{json.dumps(self.global_context, indent=2)}

**YOUR TASK:**
Create a strategic plan that guides how the Master Merger should synthesize all submaster outputs.

Return a JSON object with:
{{
  "synthesis_strategy": "How to combine and organize information",
  "key_focus_areas": ["area1", "area2", ...],
  "narrative_structure": "Recommended structure for final output",
  "cross_references": ["How to link different sections"],
  "depth_requirements": "Level of detail needed in final summary",
  "quality_checks": ["What to verify in final output"]
}}
"""
        
        try:
            plan = self.llm.call_with_retry(plan_prompt, parse_json=True)
            logger.info(f"[{self.agent_id}] Processing plan created")
            return plan
        except Exception as e:
            logger.error(f"[{self.agent_id}] Failed to create processing plan: {e}")
            return {
                "synthesis_strategy": "Combine all sections chronologically",
                "key_focus_areas": list(self.global_context.get("cross_document_themes", [])),
                "narrative_structure": "Sequential with cross-references",
                "depth_requirements": "Comprehensive technical detail"
            }


def run_residual_agent_standalone():
    """
    Standalone runner that loads latest reducer results and updates context.
    """
    logger.info("Starting ResidualAgent standalone mode")
    
    # Connect to MongoDB
    uri = os.getenv("MONGO_URI")
    dbname = os.getenv("MONGO_DB")
    
    if not uri or not dbname:
        raise RuntimeError("MONGO_URI and MONGO_DB must be set")
    
    client = MongoClient(uri)
    db = client[dbname]
    
    # Load latest reducer results
    reducer_coll = db[os.getenv("MONGO_REDUCER_RESULTS_COLLECTION", "reducer_results")]
    latest_reducer = reducer_coll.find_one(sort=[("timestamp", -1)])
    
    if not latest_reducer:
        logger.error("No reducer results found in MongoDB")
        return None
    
    logger.info("Loaded latest reducer results")
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # Create ResidualAgent
    agent = ResidualAgent.remote()
    
    # Update context
    context_future = agent.update_context_from_reducer_results.remote(latest_reducer)
    global_context = ray.get(context_future)
    
    # Create processing plan
    plan_future = agent.create_processing_plan.remote()
    processing_plan = ray.get(plan_future)
    
    logger.info("ResidualAgent processing complete")
    
    return {
        "global_context": global_context,
        "processing_plan": processing_plan
    }


def main():
    """Main entry point."""
    try:
        result = run_residual_agent_standalone()
        
        print("\n=== RESIDUAL AGENT OUTPUT ===\n")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()