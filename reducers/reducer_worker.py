# reducers/reducer_worker.py

import time
import uuid
import ray
import hashlib
from typing import Dict, Any, List, Optional
from utils.logger import get_logger
from utils.llm_helper import LLMProcessor

logger = get_logger("ReducerWorker")


@ray.remote
class ReducerWorker:

    def __init__(self, worker_id: str, llm_model: str = "mistral-small-latest"):
        self.worker_id = worker_id
        self.llm_model = llm_model

        self.llm = LLMProcessor(
            model=llm_model,
            temperature=0.3,
            max_retries=3,
            caller_id=worker_id
        )

        # Cache: hash(chunk) -> summary
        self.cache: Dict[str, Any] = {}

        self.global_context: Dict[str, Any] = {}
        self.status = "initialized"

        logger.info(f"[{self.worker_id}] ReducerWorker initialized")

    # ---------------------------------------------------------------
    def set_global_context(self, context: Dict[str, Any]):
        self.global_context = context or {}
        return {"status": "ok"}

    # ---------------------------------------------------------------
    def initialize(self):
        self.status = "ready"
        return {"worker_id": self.worker_id, "status": "ready"}

    # ---------------------------------------------------------------
    def _extract_reducer_instructions(self) -> str:
        """Create reducer level instruction text from global context."""
        if not self.global_context:
            return ""

        wc = self.global_context.get("worker_guidance", {})
        sc = self.global_context.get("submaster_guidance", {})
        style = self.global_context.get("reasoning_style", "")
        outputs = self.global_context.get("expected_outputs", "")

        text = []

        if wc:
            text.append("Worker Guidance:")
            for k, v in wc.items():
                text.append(f"{k}: {v}")

        if sc:
            text.append("Submaster Guidance:")
            for k, v in sc.items():
                text.append(f"{k}: {v}")

        if style:
            text.append(f"Preferred reasoning style: {style}")

        if outputs:
            text.append(f"Expected outputs: {outputs}")

        return "\n".join(text)

    # ---------------------------------------------------------------
    def _chunk_hash(self, chunk: List[Dict[str, Any]]) -> str:
        """Compute a stable hash to detect repeated work."""
        raw = ""
        for item in chunk:
            raw += item.get("summary", "")
            raw += ",".join(item.get("keywords", []))
            raw += ",".join(item.get("entities", []))

        return hashlib.sha256(raw.encode()).hexdigest()

    # ---------------------------------------------------------------
    def _enhance_summary_llm(self, items: List[Dict[str, Any]], extra_instructions: str) -> str:

        raw_summaries = [i.get("summary", "") for i in items if i.get("summary")]
        combined_text = "\n".join(raw_summaries)[:2000]  # OPTIMIZATION: Limit input size

        if not combined_text:
            return ""

        instruction_block = self._extract_reducer_instructions()[:500]  # Limit instructions

        if extra_instructions:
            instruction_block += "\n" + extra_instructions[:200]

        # OPTIMIZED: Shorter, focused prompt
        prompt = f"""Refine these mapper summaries into one unified summary (200-300 words).

Context: {instruction_block}

Summaries:
{combined_text}

Requirements: Clear, non-repetitive, technically accurate, professional tone. Include key data points.

Be thorough and detailed. Do not omit important information.
"""

        try:
            result = self.llm.call_with_retry(prompt, parse_json=False, max_tokens=1500)  # Reduced from 4096
            return result.strip()

        except Exception as e:
            logger.error(f"[{self.worker_id}] LLM error: {e}")
            return combined_text[:500]  # Return truncated on error

    # ---------------------------------------------------------------
    def process_chunk(self, chunk: List[Dict[str, Any]], instructions: str = "") -> Dict[str, Any]:

        start = time.time()

        # Check cache
        h = self._chunk_hash(chunk)
        if h in self.cache:
            logger.info(f"[{self.worker_id}] Cache hit for chunk")
            cached = self.cache[h]
            cached["cache"] = True
            return cached

        # Merge entities, keywords, etc
        entity_count = {}
        keyword_count = {}
        term_count = {}
        key_points = []
        insights = []

        for item in chunk:
            for e in item.get("entities", []):
                entity_count[e] = entity_count.get(e, 0) + 1
            for k in item.get("keywords", []):
                keyword_count[k] = keyword_count.get(k, 0) + 1
            for t in item.get("technical_terms", []):
                term_count[t] = term_count.get(t, 0) + 1

            key_points.extend(item.get("key_points", []))
            insights.extend(item.get("key_points", []))

        summary = self._enhance_summary_llm(
            items=chunk,
            extra_instructions=instructions
        )

        result = {
            "worker_id": self.worker_id,
            "summary": summary,
            "entities": entity_count,
            "keywords": keyword_count,
            "technical_terms": term_count,
            "key_points": list(set(key_points)),
            "insights": list(set(insights)),
            "global_context_used": bool(self.global_context),
            "processing_time": round(time.time() - start, 3),
            "status": "success",
            "cache": False
        }

        # Store in cache
        self.cache[h] = result

        return result
