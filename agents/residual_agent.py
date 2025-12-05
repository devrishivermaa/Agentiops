# agents/residual_agent.py

"""
ResidualAgent (Option B: Active Coordinator with Worker Allocation Awareness)

This Ray actor:
- Generates an initial global_context from metadata + master plan using the LLM
- Broadcasts that context to registered SubMasters
- Waits briefly for submaster task maps (which tell which worker is doing what)
- Enhances the context using gathered updates via the LLM
- Broadcasts enhanced context to registered Worker actors
- Optionally persists snapshots to MongoDB (if MONGO_URI and MONGO_DB set)
- Provides RPC endpoints for SubMasters and Workers to push updates
- Emits events through the API EventBus for real-time visualization
"""

import os
import json
import time
import uuid
import re
import threading
from typing import Dict, Any, List, Optional, Any

import ray
from pymongo.collection import Collection

from utils.logger import get_logger
from utils.llm_helper import LLMProcessor
from utils.mongo_helper import get_mongo_client

logger = get_logger("ResidualAgent")


def _emit_event_safe(event_type, pipeline_id, agent_id, agent_type, data=None):
    """Safely emit events - works from within Ray actors via HTTP"""
    try:
        from api.event_emitter import emit_event_safe
        # event_type can be EventType enum or string
        event_type_str = event_type.value if hasattr(event_type, 'value') else str(event_type)
        emit_event_safe(
            event_type=event_type_str,
            pipeline_id=pipeline_id,
            agent_id=agent_id,
            agent_type=agent_type,
            data=data,
        )
    except Exception as e:
        logger.debug(f"Event emission skipped: {e}")


@ray.remote
class ResidualAgentActor:
    def __init__(self, model: Optional[str] = None, persist: bool = True, pipeline_id: str = None):
        self.id = f"RA-{uuid.uuid4().hex[:8].upper()}"
        self.lock = threading.RLock()
        self.pipeline_id = pipeline_id or "unknown"

        model = model or os.getenv("LLM_MODEL", "mistral-small-latest")
        # initialize LLM processor (LLMProcessor should handle its own exceptions)
        self.llm = LLMProcessor(
            model=model,
            temperature=0.25,
            max_retries=4,
            caller_id=self.id
        )

        # canonical global context
        self.global_context: Dict[str, Any] = {
            "high_level_intent": "",
            "document_context": "",
            "master_strategy": "",
            "section_overview": {"sections": []},
            "worker_guidance": {},
            "submaster_guidance": {},
            "important_constraints": [],
            "expected_outputs": "",
            "reasoning_style": "short factual precise globally consistent",
            "work_allocations": {},       # submaster -> list of worker task entries
            "generated_at": None,
            "residual_id": self.id
        }

        self.update_history: List[Dict[str, Any]] = []
        self.submaster_handles: List[Any] = []
        self.worker_handles: List[Any] = []

        # persistence
        self.persist = persist
        self.mongo_client: Optional[Any] = None
        self.mongo_coll: Optional[Collection] = None

        if persist:
            try:
                uri = os.getenv("MONGO_URI")
                db = os.getenv("MONGO_DB")
                coll = os.getenv("MONGO_RESIDUAL_COLLECTION", "residual_memory")
                if uri and db:
                    self.mongo_client = get_mongo_client(uri)
                    if self.mongo_client:
                        self.mongo_coll = self.mongo_client[db][coll]
                        logger.info(f"[{self.id}] Connected to MongoDB {db}.{coll}")
                        # try to load latest snapshot quietly
                        try:
                            self._load_latest_from_db()
                        except Exception:
                            logger.exception(f"[{self.id}] Failed to load latest snapshot from MongoDB")
                    else:
                        logger.warning(f"[{self.id}] Mongo connection failed, disabling persistence")
                        self.persist = False
                else:
                    logger.warning(f"[{self.id}] Mongo disabled. Missing MONGO_URI or MONGO_DB")
                    self.persist = False
            except Exception:
                logger.exception(f"[{self.id}] Mongo init failed, disabling persistence")
                self.persist = False

        logger.info(f"[{self.id}] ResidualAgentActor initialized (persist={self.persist})")
        
        # Emit initialized event
        self._emit_event("RESIDUAL_INITIALIZED", {"status": "initialized", "persist": self.persist})

    def _emit_event(self, event_name: str, data: dict = None):
        """Helper to emit events from this ResidualAgent"""
        try:
            from api.events import EventType
            event_type = getattr(EventType, event_name, None)
            if event_type:
                _emit_event_safe(event_type, self.pipeline_id, self.id, "residual", data)
        except Exception as e:
            logger.debug(f"[{self.id}] Event emission skipped: {e}")

    # registration helpers

    def register_submasters(self, handles: List[Any]) -> Dict[str, int]:
        with self.lock:
            for h in handles:
                if h not in self.submaster_handles:
                    self.submaster_handles.append(h)
        return {"registered_submasters": len(self.submaster_handles)}

    def register_workers(self, handles: List[Any]) -> Dict[str, int]:
        with self.lock:
            for h in handles:
                if h not in self.worker_handles:
                    self.worker_handles.append(h)
        return {"registered_workers": len(self.worker_handles)}


    # main flow

    def generate_and_distribute(self, metadata: Dict[str, Any], master_plan: Dict[str, Any],
                                wait_for_updates_seconds: int = 10) -> Dict[str, Any]:
        """
        1) create initial global_context from LLM
        2) broadcast to submasters
        3) wait for submaster task maps for up to wait_for_updates_seconds
        4) enhance global_context using the updates
        5) broadcast enhanced context to workers
        6) return final snapshot
        """
        self._emit_event("RESIDUAL_CONTEXT_GENERATING", {"status": "generating"})
        
        try:
            gc = self._generate_initial_context(metadata, master_plan)
            self._emit_event("RESIDUAL_CONTEXT_GENERATED", {
                "context_keys": list(gc.keys())[:10],
                "status": "generated"
            })
        except Exception:
            logger.exception(f"[{self.id}] Failed to generate initial context")
            # return current canonical snapshot
            return self.get_snapshot()

        # broadcast to submasters (fire and forget)
        self._emit_event("RESIDUAL_BROADCASTING", {
            "target": "submasters",
            "count": len(self.submaster_handles)
        })
        self._broadcast_to_submasters(gc)
        self._emit_event("RESIDUAL_BROADCAST_COMPLETE", {"target": "submasters", "status": "complete"})

        # wait for submaster task maps
        self._wait_for_submaster_task_maps(wait_for_updates_seconds)

        # capture updates copy and enhance
        with self.lock:
            updates_copy = list(self.update_history)

        enhanced = self._enhance_context_with_updates(metadata, master_plan, updates_copy)
        self._emit_event("RESIDUAL_CONTEXT_ENHANCED", {
            "num_updates": len(updates_copy),
            "status": "enhanced"
        })

        # broadcast to workers
        self._emit_event("RESIDUAL_BROADCASTING", {
            "target": "workers",
            "count": len(self.worker_handles)
        })
        self._broadcast_to_workers(enhanced)
        self._emit_event("RESIDUAL_BROADCAST_COMPLETE", {"target": "workers", "status": "complete"})

        # persist and return
        if self.persist:
            try:
                self._maybe_persist()
            except Exception:
                logger.exception(f"[{self.id}] Failed to persist final snapshot")

        return self.get_snapshot()


    # RPC endpoints for receiving updates

    def update_from_submaster(self, update: Dict[str, Any], author: str) -> Dict[str, str]:
        """
        Expected update example:
        {
          "submaster_id": "SM-001",
          "section_name": "Introduction",
          "work_distribution": [
               { "worker_id": "SM-001-W1", "task_type": "summary", "page_range": [2,4], "status": "assigned" }
          ]
        }
        """
        with self.lock:
            self.update_history.append({
                "timestamp": time.time(),
                "author": author,
                "source": "submaster",
                "payload": update
            })
            
            # Emit update received event
            self._emit_event("RESIDUAL_UPDATE_RECEIVED", {
                "source": "submaster",
                "author": author,
                "submaster_id": update.get("submaster_id")
            })
            
            try:
                self._merge_submaster_task_map(update)
            except Exception:
                logger.exception(f"[{self.id}] Failed to merge submaster task map from {author}")
            try:
                self._maybe_persist()
            except Exception:
                logger.exception(f"[{self.id}] Failed to persist after submaster update")
        return {"status": "ok"}


    def update_from_worker(self, update: Dict[str, Any], author: str) -> Dict[str, Any]:
        """
        Worker updates can be lightweight page-level results or status updates.
        e.g. {"worker_id": "SM-001-W2", "page": 3, "status": "done", "entities": [...], "keywords": [...], "summary": "..."}
        """
        with self.lock:
            self.update_history.append({
                "timestamp": time.time(),
                "author": author,
                "source": "worker",
                "payload": update
            })
            
            # Emit update received event
            self._emit_event("RESIDUAL_UPDATE_RECEIVED", {
                "source": "worker",
                "author": author,
                "worker_id": update.get("worker_id"),
                "page": update.get("page")
            })
            
            try:
                self._merge_worker_output(update)
            except Exception:
                logger.exception(f"[{self.id}] Failed to merge worker update from {author}")
            try:
                self._maybe_persist()
            except Exception:
                logger.exception(f"[{self.id}] Failed to persist after worker update")
        return {"status": "ok"}


    # initial generation

    def _generate_initial_context(self, metadata: Dict[str, Any], plan: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._build_initial_prompt(metadata, plan)
        raw = self.llm.call_with_retry(prompt, parse_json=False)

        parsed = self._safe_extract_json(raw)
        gc = parsed.get("global_context") if isinstance(parsed, dict) else None

        if not isinstance(gc, dict):
            raise ValueError("LLM did not return a valid 'global_context' object")

        with self.lock:
            gc["generated_at"] = time.time()
            gc["residual_id"] = self.id
            self._merge_generated_context(gc)
            self.update_history.append({"timestamp": time.time(), "source": "residual_llm_init"})
            # persist a snapshot if enabled
            try:
                self._maybe_persist()
            except Exception:
                logger.exception(f"[{self.id}] Failed to persist initial snapshot")

        logger.info(f"[{self.id}] Initial context generated with worker_guidance keys: {list(gc.get('worker_guidance', {}).keys())}")
        logger.info(f"[{self.id}] Initial context generated with submaster_guidance keys: {list(gc.get('submaster_guidance', {}).keys())}")

        return json.loads(json.dumps(self.global_context))


    # broadcast helpers

    def _broadcast_to_submasters(self, context: Dict[str, Any]) -> None:
        if not self.submaster_handles:
            logger.info(f"[{self.id}] No submasters registered")
            return
        logger.info(f"[{self.id}] Broadcasting context to {len(self.submaster_handles)} submasters")
        for h in self.submaster_handles:
            try:
                # submaster actor must implement set_global_context(context)
                h.set_global_context.remote(context)
            except Exception:
                logger.exception(f"[{self.id}] Failed to send context to a submaster actor")

    def _broadcast_to_workers(self, context: Dict[str, Any]) -> None:
        if not self.worker_handles:
            logger.info(f"[{self.id}] No workers registered")
            return
        logger.info(f"[{self.id}] Broadcasting context to {len(self.worker_handles)} workers")
        for h in self.worker_handles:
            try:
                # worker actor should implement set_global_context(context)
                h.set_global_context.remote(context)
            except Exception:
                logger.exception(f"[{self.id}] Failed to send context to a worker actor")


    # waiting for submaster maps

    def _wait_for_submaster_task_maps(self, timeout_seconds: int) -> None:
        deadline = time.time() + max(0, int(timeout_seconds))
        logger.info(f"[{self.id}] Waiting up to {timeout_seconds}s for submaster task maps")
        while time.time() < deadline:
            with self.lock:
                sub_updates = [u for u in self.update_history if u.get("source") == "submaster"]
                distinct_authors = {u.get("author") for u in sub_updates if u.get("author")}
                if len(distinct_authors) >= len(self.submaster_handles):
                    logger.info(f"[{self.id}] Received task maps from {len(distinct_authors)} submasters")
                    return
            time.sleep(0.5)
        logger.info(f"[{self.id}] Wait timeout reached. Collected {len(sub_updates)} submaster updates")


    # enhancement with updates

    def _enhance_context_with_updates(self, metadata: Dict[str, Any], plan: Dict[str, Any], updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            prev = json.dumps(self.global_context, indent=2, ensure_ascii=False)
            upd = json.dumps(updates[-25:], indent=2, ensure_ascii=False)
        except Exception:
            prev = str(self.global_context)
            upd = str(updates)

        prompt = f"""
You are a ResidualAgent responsible for maintaining a global context for a multi-agent PDF processing system.

Previous global_context:
{prev}

Recent updates from submasters and workers:
{upd}

Task: Enhance the global_context by:
1. Merging work allocations from submaster updates
2. Updating section notes with any new information
3. Ensuring worker_guidance and submaster_guidance are properly populated with actionable instructions
4. Maintaining consistency across all fields

CRITICAL: The worker_guidance and submaster_guidance fields MUST contain specific, actionable instructions.

Output ONLY valid JSON with a top-level key "global_context" that includes ALL required fields:
- high_level_intent
- document_context
- master_strategy
- section_overview (with sections array)
- worker_guidance (object with specific instructions for different aspects)
- submaster_guidance (object with coordination instructions)
- important_constraints
- expected_outputs
- reasoning_style

Example worker_guidance structure:
{{
  "entity_extraction": "Extract named entities including people, organizations, methods, and technical terms. Focus on key figures and methodologies.",
  "keyword_indexing": "Identify and index domain-specific keywords, technical terms, and key concepts for searchability.",
  "summary_generation": "Create concise summaries highlighting main findings, methods, and conclusions. Keep summaries under 3 sentences.",
  "consistency_check": "Ensure terminology and references remain consistent with previous sections and the overall document context."
}}

Example submaster_guidance structure:
{{
  "coordination": "Ensure workers in your section maintain consistent terminology and entity references.",
  "quality_control": "Review worker outputs for completeness and alignment with the document's overall narrative.",
  "cross_section_awareness": "Be aware of dependencies between your section and others as noted in section_overview."
}}
"""
        try:
            raw = self.llm.call_with_retry(prompt, parse_json=False)
            parsed = self._safe_extract_json(raw)
            new_gc = parsed.get("global_context")
            if not isinstance(new_gc, dict):
                raise ValueError("Enhancement LLM response missing global_context")
        except Exception:
            logger.exception(f"[{self.id}] Enhancement LLM failed; returning current global_context")
            return json.loads(json.dumps(self.global_context))

        with self.lock:
            new_gc["generated_at"] = time.time()
            new_gc["residual_id"] = self.id
            self._merge_generated_context(new_gc)
            self.update_history.append({"timestamp": time.time(), "source": "residual_llm_enhance"})
            try:
                self._maybe_persist()
            except Exception:
                logger.exception(f"[{self.id}] Failed to persist after enhancement")
            logger.info(f"[{self.id}] Enhanced global_context merged")
            logger.info(f"[{self.id}] Enhanced worker_guidance keys: {list(new_gc.get('worker_guidance', {}).keys())}")
            logger.info(f"[{self.id}] Enhanced submaster_guidance keys: {list(new_gc.get('submaster_guidance', {}).keys())}")

        return json.loads(json.dumps(self.global_context))


    # merging submaster task maps into work_allocations

    def _merge_submaster_task_map(self, update: Dict[str, Any]) -> None:
        sm_id = update.get("submaster_id") or update.get("submaster")
        work_list = update.get("work_distribution", []) or update.get("work_allocation", [])
        section = update.get("section_name") or update.get("section")

        if not sm_id or not work_list:
            return

        with self.lock:
            allocations = self.global_context.setdefault("work_allocations", {})
            sm_alloc = allocations.setdefault(sm_id, [])

            for task in work_list:
                entry = {
                    "worker_id": task.get("worker_id"),
                    "section": section,
                    "task_type": task.get("task_type"),
                    "pages": task.get("page_range") or task.get("pages"),
                    "status": task.get("status", "assigned")
                }
                # avoid duplicates by simple equality check
                if entry not in sm_alloc:
                    sm_alloc.append(entry)

            # store back
            allocations[sm_id] = sm_alloc
            self.global_context["work_allocations"] = allocations


    # merging worker outputs (basic aggregation)

    def _merge_worker_output(self, worker_update: Dict[str, Any]) -> None:
        # aggregate top-level entities/keywords when present, attach short summary into section notes
        if not worker_update:
            return

        with self.lock:
            # aggregate global_entities
            ents = set(self.global_context.get("global_entities", []))
            for e in worker_update.get("entities", []) or []:
                ents.add(e)
            if ents:
                self.global_context["global_entities"] = list(ents)

            # aggregate global_keywords
            kws = set(self.global_context.get("global_keywords", []))
            for k in worker_update.get("keywords", []) or []:
                kws.add(k)
            if kws:
                self.global_context["global_keywords"] = list(kws)

            # attach summary to section notes
            section = worker_update.get("section")
            summary = worker_update.get("summary") or worker_update.get("short_summary")
            page = worker_update.get("page")
            if section and summary:
                secs = self.global_context.setdefault("section_overview", {}).setdefault("sections", [])
                tgt = next((s for s in secs if s.get("name") == section), None)
                if not tgt:
                    tgt = {"name": section, "purpose": "", "page_range": [], "importance": "medium", "dependencies": []}
                    secs.append(tgt)
                old = tgt.get("notes", "")
                addition = f" [p{page}] {summary}" if page else f" {summary}"
                tgt["notes"] = (old + addition).strip()
                # write back
                self.global_context["section_overview"]["sections"] = secs


    # merging generated context from LLM

    def _merge_generated_context(self, new: Dict[str, Any]) -> None:
        # top-level text fields
        for key in ["high_level_intent", "document_context", "master_strategy", "expected_outputs", "reasoning_style"]:
            if key in new and new[key]:
                self.global_context[key] = new[key]

        # replace/merge section overview conservatively
        new_sections = new.get("section_overview", {}).get("sections", [])
        if new_sections:
            # simple replace for now, but preserve any existing notes/allocations if names match
            existing_secs = {s.get("name"): s for s in self.global_context.get("section_overview", {}).get("sections", [])}
            for s in new_sections:
                name = s.get("name")
                if name and name in existing_secs:
                    # merge purpose and dependencies
                    ex = existing_secs[name]
                    ex["purpose"] = (ex.get("purpose", "") + " " + s.get("purpose", "")).strip()
                    # merge dependencies
                    deps = set(ex.get("dependencies", []))
                    for d in s.get("dependencies", []):
                        deps.add(d)
                    ex["dependencies"] = list(deps)
                    # keep other keys from new
                    for k, v in s.items():
                        if k not in ("purpose", "dependencies") and v:
                            ex[k] = v
                    existing_secs[name] = ex
                else:
                    existing_secs[name] = s
            self.global_context["section_overview"]["sections"] = list(existing_secs.values())

        # guidance replacement/merge - IMPROVED to ensure non-empty guidance
        if "worker_guidance" in new and isinstance(new["worker_guidance"], dict) and new["worker_guidance"]:
            dest = self.global_context.get("worker_guidance", {})
            for k, v in new["worker_guidance"].items():
                if v:  # Only merge non-empty values
                    if k not in dest or not dest[k]:
                        dest[k] = v
                    else:
                        # concat short strings
                        if isinstance(dest[k], str) and isinstance(v, str) and len(dest[k]) < 500:
                            dest[k] = (dest[k].strip() + " " + v.strip()).strip()
            self.global_context["worker_guidance"] = dest
            logger.info(f"[{self.id}] Merged worker_guidance with {len(dest)} keys")

        if "submaster_guidance" in new and isinstance(new["submaster_guidance"], dict) and new["submaster_guidance"]:
            dest = self.global_context.get("submaster_guidance", {})
            for k, v in new["submaster_guidance"].items():
                if v:  # Only merge non-empty values
                    if k not in dest or not dest[k]:
                        dest[k] = v
                    else:
                        if isinstance(dest[k], str) and isinstance(v, str) and len(dest[k]) < 500:
                            dest[k] = (dest[k].strip() + " " + v.strip()).strip()
            self.global_context["submaster_guidance"] = dest
            logger.info(f"[{self.id}] Merged submaster_guidance with {len(dest)} keys")

        # constraints
        if "important_constraints" in new and isinstance(new["important_constraints"], list):
            existing = set(self.global_context.get("important_constraints", []))
            for c in new["important_constraints"]:
                existing.add(c)
            self.global_context["important_constraints"] = list(existing)


    # robust JSON extraction from LLM output

    def _safe_extract_json(self, text: str) -> Dict[str, Any]:
        """
        Attempt to extract a single JSON object from the LLM output.
        Strategy:
          - find first '{' and last '}' occurrence and try to load
          - if fails, fallback to regex-based greedy capture
        """
        if not text or not isinstance(text, str):
            raise ValueError("No text to parse")

        # try first/last brace slice
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            candidate = text[first:last+1]
            try:
                return json.loads(candidate)
            except Exception:
                # fallthrough to regex
                pass

        # fallback: regex greedy (less safe but sometimes works)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                raise ValueError("Failed to parse JSON from LLM output")
        raise ValueError("No JSON object found in LLM output")


    def _build_initial_prompt(self, metadata: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """Build a more detailed prompt with concrete examples for guidance fields"""
        try:
            md = json.dumps(metadata, indent=2, ensure_ascii=False)
        except Exception:
            md = str(metadata)
        try:
            pl = json.dumps(plan, indent=2, ensure_ascii=False)
        except Exception:
            pl = str(plan)

        # Extract processing requirements to guide the generation
        proc_reqs = metadata.get("processing_requirements", [])
        req_str = ", ".join(proc_reqs) if proc_reqs else "summary_generation, entity_extraction, keyword_indexing"

        return f"""
You are a ResidualAgent responsible for generating a comprehensive global context for a multi-agent PDF processing system.

Input metadata:
{md}

Input master plan:
{pl}

Task: Generate a complete global_context JSON object that will guide all SubMasters and Workers in processing this document.

The processing requirements are: {req_str}

Output ONLY valid JSON with a top-level key "global_context" that includes:

1. high_level_intent: The user's goal (e.g., "Summarize for a presentation", "Extract research findings")
2. document_context: Brief description of what the document is about
3. master_strategy: How the work is divided among SubMasters
4. section_overview: Object with "sections" array, each containing:
   - name: section name
   - page_start: starting page number
   - page_end: ending page number
   - purpose: what this section covers
   - importance: "high", "medium", or "low"
   - dependencies: array of other section names this depends on
5. worker_guidance: Object with SPECIFIC instructions for each processing task. Must include:
   - For each requirement in {req_str}, provide detailed guidance
   - Example keys: "entity_extraction", "keyword_indexing", "summary_generation", "consistency_check"
6. submaster_guidance: Object with coordination instructions:
   - "coordination": how to coordinate workers
   - "quality_control": what to check in worker outputs
   - "cross_section_awareness": how to handle section dependencies
7. important_constraints: Array of constraints (e.g., "Maintain consistent terminology", "Preserve technical accuracy")
8. expected_outputs: Description of what final outputs should look like
9. reasoning_style: How to approach the analysis (e.g., "Analytical and structured", "Concise and factual")

CRITICAL: worker_guidance and submaster_guidance MUST be populated with specific, actionable instructions. Do NOT leave them empty or use placeholder text.

Example worker_guidance (adapt to the actual document and requirements):
{{
  "entity_extraction": "Extract named entities including authors, methods, key findings, and technical terms. Pay special attention to methodology names and important figures.",
  "keyword_indexing": "Identify domain-specific keywords, technical terms, and key concepts. Focus on terms that appear frequently or are emphasized in headings.",
  "summary_generation": "Create concise 2-3 sentence summaries capturing the main point, methodology, and key result or conclusion of each section.",
  "consistency_check": "Ensure entity names and terminology match previous sections. Cross-reference with the document_context."
}}

Example submaster_guidance:
{{
  "coordination": "Distribute pages evenly among workers. Ensure workers are aware of section boundaries and dependencies.",
  "quality_control": "Review worker outputs for completeness, accuracy, and consistency with the overall document narrative.",
  "cross_section_awareness": "Note any references to other sections and ensure these are captured in the section dependencies."
}}

Generate the complete global_context now:
"""


    # persistence helpers

    def _maybe_persist(self) -> None:
        if not self.persist or self.mongo_coll is None:
            return
        try:
            doc = {
                "residual_id": self.id,
                "timestamp": time.time(),
                "global_context": self.global_context,
                "history_tail": self.update_history[-80:]
            }
            self.mongo_coll.insert_one(doc)
            logger.debug(f"[{self.id}] Persisted global_context to MongoDB")
        except Exception:
            logger.exception(f"[{self.id}] Failed to persist global_context")


    def _load_latest_from_db(self) -> None:
        if self.mongo_coll is None:
            return
        try:
            doc = self.mongo_coll.find_one(sort=[("timestamp", -1)])
            if not doc:
                return
            gc = doc.get("global_context")
            if isinstance(gc, dict):
                with self.lock:
                    self.global_context = gc
                    self.update_history = doc.get("history_tail", []) or []
                logger.info(f"[{self.id}] Loaded existing global_context from MongoDB")
        except Exception:
            logger.exception(f"[{self.id}] Failed to load latest global_context from MongoDB")


    # utility: snapshot getter

    def get_snapshot(self) -> Dict[str, Any]:
        with self.lock:
            # update generated_at
            self.global_context["generated_at"] = time.time()
            return json.loads(json.dumps(self.global_context))