# reducers/reducer_worker.py

import time
import uuid
import ray
from typing import Dict, Any, List, Optional
from utils.logger import get_logger
from utils.llm_helper import LLMProcessor

logger = get_logger("ReducerWorker")


@ray.remote
class ReducerWorker:

    def __init__(self, worker_id: str, llm_model: str = "mistral-small-latest"):
        self.worker_id = worker_id
        self.llm_model = llm_model
        self.status = "initialized"

        # LLM Processor interface
        self.llm = LLMProcessor(
            model=llm_model,
            temperature=0.3,
            max_retries=3,
            caller_id=worker_id
        )

        # ResidualAgent context
        self.global_context: Dict[str, Any] = {}

        logger.info(f"[{self.worker_id}] ReducerWorker initialized")

    # ---------------------------------------------------------------------
    # Receive global context from ResidualAgent
    # ---------------------------------------------------------------------
    def set_global_context(self, context: Dict[str, Any]):
        self.global_context = context or {}
        logger.info(f"[{self.worker_id}] Global context received: {list(self.global_context.keys())}")
        return {"status": "ok"}

    # ---------------------------------------------------------------------
    # Worker initialization
    # ---------------------------------------------------------------------
    def initialize(self):
        self.status = "ready"
        logger.info(f"[{self.worker_id}] Worker ready")
        return {"worker_id": self.worker_id, "status": "ready"}

    # ---------------------------------------------------------------------
    # PRIVATE: Generate LLM-enhanced summary using raw + instructions
    # ---------------------------------------------------------------------
    def _enhance_summary_llm(
        self,
        items: List[Dict[str, Any]],
        extra_instructions: str = ""
    ) -> str:

        # Collect all summaries
        raw_summaries = [i.get("summary", "") for i in items if i.get("summary")]
        combined_text = "\n".join(raw_summaries)

        if not combined_text:
            return ""

        # Add residual context + instructions
        context_lines = []

        if self.global_context:
            hc = self.global_context.get("high_level_intent", "")
            rc = self.global_context.get("expected_outputs", {}).get("reasoning_style", "")
            context_lines.append(f"High-Level Goal: {hc}")
            context_lines.append(f"Preferred Style: {rc}")

        if extra_instructions:
            context_lines.append(f"Reducer Instructions: {extra_instructions}")

        context_block = "\n".join(context_lines)

        prompt = f"""
You are an expert document analysis LLM.

Your task is to transform a set of mapper-level summaries into a 
single enhanced, unified, high-quality summary.

CONTEXT:
{context_block}

RAW SUMMARIES (from mapper workers):
{combined_text}

REQUIREMENTS:
- Create a refined, structured summary
- Remove repetition
- Improve clarity
- Preserve all technical meaning
- Include cross-section insights when relevant
- 200 to 350 words
- Professional tone

Return only the improved summary, no JSON.
"""

        try:
            response = self.llm.call_with_retry(prompt, parse_json=False)
            return response.strip()
        except Exception as e:
            logger.error(f"[{self.worker_id}] LLM failure: {e}")
            return combined_text  # fallback

    # ---------------------------------------------------------------------
    # MAIN: Process one reducer chunk (summary enhancement + merging)
    # ---------------------------------------------------------------------
    def process_chunk(self, chunk: List[Dict[str, Any]], instructions: str = "") -> Dict[str, Any]:
        start = time.time()

        logger.info(f"[{self.worker_id}] Processing chunk size: {len(chunk)}")

        entity_count = {}
        keyword_count = {}
        term_count = {}
        key_points = []
        insights = []

        # Merge raw mapper content
        for item in chunk:

            for e in item.get("entities", []):
                entity_count[e] = entity_count.get(e, 0) + 1

            for k in item.get("keywords", []):
                keyword_count[k] = keyword_count.get(k, 0) + 1

            for t in item.get("technical_terms", []):
                term_count[t] = term_count.get(t, 0) + 1

            # carry key points forward
            key_points.extend(item.get("key_points", []))
            insights.extend(item.get("key_points", []))

        # LLM-enhanced summary
        final_summary = self._enhance_summary_llm(
            items=chunk,
            extra_instructions=instructions
        )

        result = {
            "worker_id": self.worker_id,
            "summary": final_summary,
            "entities": entity_count,
            "keywords": keyword_count,
            "technical_terms": term_count,
            "key_points": list(set(key_points)),
            "insights": list(set(insights)),
            "global_context_used": bool(self.global_context),
            "processing_time": round(time.time() - start, 3),
            "status": "success"
        }

        logger.info(f"[{self.worker_id}] Completed chunk in {result['processing_time']} seconds")

        return result
