"""
Standalone Reducer orchestrator and ReducerSubMaster actor.

Behavior:
- Loads all mapper SubMaster documents from MongoDB collection `submaster_results`
- Sorts them by `sm_id`
- Groups them in pairs (group size = 2)
- Creates ceil(N/2) ReducerSubMaster actors, each processing a pair
- Each ReducerSubMaster spawns ceil(mapper_workers_per_submaster / 2) reducer workers
- Reducer workers accept chunk + instructions and enhance summaries using LLM
- Per-worker outputs saved to `reducer_worker_results`
- Per-ReducerSubMaster aggregate saved to `reducer_submaster_results`
- Final merged reducer output saved to `reducer_results`

Run from project root:
    python -m reducers.reducer_submaster
"""

import os
import sys
import math
import json
import time
import uuid
from typing import List, Dict, Any

# Ensure project root on sys.path so imports like utils.* work when running file directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import ray
from pymongo import MongoClient

from utils.logger import get_logger
from utils.llm_helper import LLMProcessor
from reducers.reducer_worker import ReducerWorker

logger = get_logger("ReducerOrchestrator")


def sanitize_mongo_doc(doc):
    """
    Recursively convert MongoDB ObjectId and other non-JSON types to strings.
    """
    if isinstance(doc, dict):
        return {k: sanitize_mongo_doc(v) for k, v in doc.items()}
    elif isinstance(doc, list):
        return [sanitize_mongo_doc(item) for item in doc]
    elif hasattr(doc, '__class__') and doc.__class__.__name__ == 'ObjectId':
        return str(doc)
    else:
        return doc


@ray.remote
class ReducerSubMasterActor:
    """
    ReducerSubMaster actor that receives 1 or 2 mapper_submaster documents,
    merges their page-level results, spawns reducer workers, composes LLM
    instructions, dispatches chunks to workers, collects outputs, persists to Mongo,
    and returns aggregated output.
    """

    def __init__(
        self,
        rsm_id: str,
        mapper_docs: List[Dict[str, Any]],
        num_workers: int,
        metadata: Dict[str, Any],
        global_context: Dict[str, Any] = None,
    ):
        self.rsm_id = rsm_id
        self.mapper_docs = mapper_docs or []
        self.metadata = metadata or {}
        self.num_workers = max(1, int(num_workers))
        self.global_context = global_context or {}
        self.status = "initialized"

        # Combine page-level results from mapper submasters assigned to this reducer
        self.page_results: List[Dict[str, Any]] = []
        for md in self.mapper_docs:
            out = md.get("output", {}) or {}
            results = out.get("results", []) or []
            for r in results:
                # record source for traceability
                r["_source_sm_id"] = md.get("sm_id")
            self.page_results.extend(results)

        # Create a small LLM processor for this SubMaster to generate worker instructions
        llm_model = self.metadata.get("preferred_model", os.getenv("LLM_MODEL", "mistral-small-latest"))
        try:
            self.llm = LLMProcessor(model=llm_model, temperature=0.2, caller_id=f"ReducerSubMaster-{self.rsm_id}")
            logger.info(f"[{self.rsm_id}] SubMaster LLM initialized {llm_model}")
        except Exception as e:
            self.llm = None
            logger.warning(f"[{self.rsm_id}] Failed to init SubMaster LLM: {e}")

        # Mongo setup
        uri = os.getenv("MONGO_URI")
        dbname = os.getenv("MONGO_DB")
        self.mongo_ok = False
        try:
            client = MongoClient(uri)
            db = client[dbname]
            self.mongo_worker_coll = db[os.getenv("MONGO_REDUCER_WORKER_COLLECTION", "reducer_worker_results")]
            self.mongo_sm_coll = db[os.getenv("MONGO_REDUCER_SUBMASTER_COLLECTION", "reducer_submaster_results")]
            self.mongo_final_coll = db[os.getenv("MONGO_REDUCER_RESULTS_COLLECTION", "reducer_results")]
            self.mongo_ok = True
            logger.info(f"[{self.rsm_id}] Connected to Mongo DB {dbname}")
        except Exception as e:
            logger.error(f"[{self.rsm_id}] Mongo connection failed: {e}")
            self.mongo_ok = False

        # Spawn reducer workers
        self.workers = []
        self._spawn_workers()

        # forward global context to workers
        if self.global_context and self.workers:
            try:
                ray.get([w.set_global_context.remote(self.global_context) for w in self.workers])
                logger.info(f"[{self.rsm_id}] Global context forwarded to workers")
            except Exception as e:
                logger.warning(f"[{self.rsm_id}] Failed to forward context to workers: {e}")

        self.status = "ready"
        logger.info(f"[{self.rsm_id}] Initialized. pages={len(self.page_results)} workers={len(self.workers)}")

    def initialize(self):
        """
        Explicit initialization method for compatibility with orchestrator.
        Since __init__ already performs all setup, this just confirms readiness.
        """
        logger.info(f"[{self.rsm_id}] Initialize called - already ready")
        return {"rsm_id": self.rsm_id, "status": self.status}

    def _spawn_workers(self):
        llm_model = self.metadata.get("preferred_model", os.getenv("LLM_MODEL", "mistral-small-latest"))
        for i in range(self.num_workers):
            wid = f"{self.rsm_id}-RW{i+1}"
            try:
                w = ReducerWorker.remote(worker_id=wid, llm_model=llm_model)
                self.workers.append(w)
            except Exception as e:
                logger.exception(f"[{self.rsm_id}] Failed to create worker {wid}: {e}")

        try:
            ray.get([w.initialize.remote() for w in self.workers])
            logger.info(f"[{self.rsm_id}] All workers initialized")
        except Exception as e:
            logger.warning(f"[{self.rsm_id}] Worker initialization error: {e}")

    def _make_instructions_for_chunk(self, chunk_items: List[Dict[str, Any]]) -> str:
        """
        Use SubMaster LLM to compose clear instructions for reducer workers.
        If LLM not available, return a fallback instruction string.
        """
        if not self.llm:
            return "Refine the provided mapper summaries. Merge duplicates, improve clarity, extract insights, and return structured output."

        sample_summaries = [it.get("summary", "") for it in chunk_items if it.get("summary")]
        prompt = f"""
You are a reducer SubMaster creating instructions for reducer workers.
Global context:
{json.dumps(self.global_context, indent=2)}

Provide step-by-step instructions for the worker to:
1) Improve and unify the given mapper-level summaries
2) Preserve technical detail
3) Ensure terminology consistency across pages
4) Extract key insights and concise key points
5) Return: enhanced_summary, entities frequency map, keywords map, technical_terms map, key_points, insights

SAMPLE SUMMARIES:
{json.dumps(sample_summaries[:10], indent=2)}

Return only the instruction text.
"""
        try:
            inst = self.llm.call_with_retry(prompt, parse_json=False)
            return inst
        except Exception as e:
            logger.warning(f"[{self.rsm_id}] SubMaster LLM failed to create instructions: {e}")
            return "Refine the summaries, extract insights, and return structured output."

    def _generate_brief_summary(self, combined_output: Dict[str, Any]) -> str:
        """
        Generate a very brief summary (2-4 sentences) for the residual agent.
        This provides high-level context without full detail.
        """
        if not self.llm:
            # Fallback: extract first 200 chars from enhanced summary
            full_summary = combined_output.get("enhanced_summary", "")
            return full_summary[:200] + "..." if len(full_summary) > 200 else full_summary

        # Get top entities and keywords for context
        top_entities = sorted(
            combined_output.get("entities", {}).items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        top_keywords = sorted(
            combined_output.get("keywords", {}).items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        top_insights = combined_output.get("insights", [])[:3]
        
        prompt = f"""
You are generating a very brief summary for a residual agent to understand this document section.

FULL SUMMARY:
{combined_output.get('enhanced_summary', '')[:500]}

TOP ENTITIES: {[e[0] for e in top_entities]}
TOP KEYWORDS: {[k[0] for k in top_keywords]}
KEY INSIGHTS: {top_insights}

Generate a 2-4 sentence brief summary that captures:
1) Main topic/domain
2) Key findings or focus areas
3) Any critical entities or concepts

Keep it concise and informative for high-level context.
"""
        try:
            brief = self.llm.call_with_retry(prompt, parse_json=False)
            return brief.strip()
        except Exception as e:
            logger.warning(f"[{self.rsm_id}] Failed to generate brief summary: {e}")
            # Fallback to truncated summary
            full_summary = combined_output.get("enhanced_summary", "")
            return full_summary[:200] + "..." if len(full_summary) > 200 else full_summary

    def process(self) -> Dict[str, Any]:
        """
        Main entry for this ReducerSubMaster actor.
        Splits page_results among workers, generates instructions,
        dispatches to workers, persists worker outputs and returns aggregated result.
        """
        self.status = "running"
        t0 = time.time()

        total_pages = len(self.page_results)
        if total_pages == 0:
            return {"rsm_id": self.rsm_id, "status": "completed", "output": {}, "elapsed": 0.0}

        # Split page_results evenly among workers
        chunk_size = math.ceil(total_pages / max(1, len(self.workers)))
        chunks = [self.page_results[i:i + chunk_size] for i in range(0, total_pages, chunk_size)]

        # Prepare and dispatch tasks
        futures = []
        for idx, chunk in enumerate(chunks):
            instructions = self._make_instructions_for_chunk(chunk)
            worker = self.workers[idx % len(self.workers)]
            try:
                futures.append(worker.process_chunk.remote(chunk, instructions))
            except Exception as e:
                logger.exception(f"[{self.rsm_id}] Failed to dispatch to worker: {e}")

        # Collect outputs
        try:
            worker_outputs = ray.get(futures)
        except Exception as e:
            logger.error(f"[{self.rsm_id}] Error collecting worker outputs: {e}")
            worker_outputs = []

        # Persist worker outputs
        if self.mongo_ok:
            for out in worker_outputs:
                try:
                    doc = {
                        "rsm_id": self.rsm_id,
                        "worker_id": out.get("worker_id"),
                        "timestamp": time.time(),
                        "output": out
                    }
                    self.mongo_worker_coll.insert_one(doc)
                except Exception as e:
                    logger.error(f"[{self.rsm_id}] Failed to save worker output: {e}")

        # Merge worker outputs
        aggregated_output = self._merge_worker_outputs(worker_outputs)

        aggregated_doc = {
            "rsm_id": self.rsm_id,
            "source_mapper_submasters": [m.get("sm_id") for m in self.mapper_docs],
            "status": "completed",
            "timestamp": time.time(),
            "output": aggregated_output,
        }

        if self.mongo_ok:
            try:
                self.mongo_sm_coll.insert_one(aggregated_doc)
            except Exception as e:
                logger.error(f"[{self.rsm_id}] Failed to save reducer submaster aggregate: {e}")

        elapsed = time.time() - t0
        aggregated_doc["elapsed"] = elapsed
        self.status = "completed"
        return aggregated_doc

    def _merge_worker_outputs(self, outputs: List[Dict[str, Any]]) -> Dict[str, Any]:
        combined = {
            "enhanced_summary": "",
            "entities": {},
            "keywords": {},
            "technical_terms": {},
            "key_points": [],
            "insights": []
        }

        for out in outputs:
            combined["enhanced_summary"] += " " + (out.get("summary") or out.get("enhanced_summary", ""))
            for k, v in (out.get("entities") or {}).items():
                combined["entities"][k] = combined["entities"].get(k, 0) + int(v)
            for k, v in (out.get("keywords") or {}).items():
                combined["keywords"][k] = combined["keywords"].get(k, 0) + int(v)
            for k, v in (out.get("technical_terms") or {}).items():
                combined["technical_terms"][k] = combined["technical_terms"].get(k, 0) + int(v)
            combined["key_points"].extend(out.get("key_points", []))
            combined["insights"].extend(out.get("insights", []))

        combined["enhanced_summary"] = combined["enhanced_summary"].strip()
        combined["key_points"] = list(dict.fromkeys(combined["key_points"]))
        combined["insights"] = list(dict.fromkeys(combined["insights"]))
        
        # Generate brief summary for residual agent
        combined["brief_summary"] = self._generate_brief_summary(combined)
        
        return combined


def group_mapper_docs(mapper_docs: List[Dict[str, Any]], group_size: int = 2) -> List[List[Dict[str, Any]]]:
    """
    Group mapper submaster docs into groups of `group_size`.
    """
    groups = [mapper_docs[i:i + group_size] for i in range(0, len(mapper_docs), group_size)]
    return groups


def _generate_overall_brief_summary(brief_summaries: List[Dict[str, Any]], aggregated: Dict[str, Any]) -> str:
    """
    Generate an overall brief summary from all RSM brief summaries.
    This provides the residual agent with a high-level document overview.
    """
    if not brief_summaries:
        return "No summary available."
    
    # Combine all brief summaries
    combined_briefs = "\n\n".join([
        f"Section {bs.get('rsm_id')}: {bs.get('brief_summary', '')}"
        for bs in brief_summaries
    ])
    
    # Get top 10 entities and keywords for context
    top_entities = sorted(
        aggregated.get("entities", {}).items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    top_keywords = sorted(
        aggregated.get("keywords", {}).items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    # Create a concise overall summary
    summary_parts = [
        f"Document Overview: This document covers {len(brief_summaries)} major sections.",
        f"Key entities: {', '.join([e[0] for e in top_entities[:5]])}.",
        f"Main topics: {', '.join([k[0] for k in top_keywords[:5]])}.",
        f"Total insights: {len(aggregated.get('insights', []))} key points identified."
    ]
    
    return " ".join(summary_parts)


def run_reducer_global(metadata: Dict[str, Any] = None, mapper_workers_override: int = None):
    """
    Orchestrator running in global mode: loads all mapper submaster docs from Mongo,
    sorts by sm_id, groups them, spawns ReducerSubMaster actors, collects and merges outputs.
    """
    metadata = metadata or {}

    uri = os.getenv("MONGO_URI")
    dbname = os.getenv("MONGO_DB")
    if not uri or not dbname:
        raise RuntimeError("MONGO_URI and MONGO_DB environment variables must be set")

    client = MongoClient(uri)
    db = client[dbname]
    mapper_coll_name = os.getenv("MONGO_SUBMASTER_COLLECTION", "submaster_results")

    # Load all mapper submasters (exclude _id to avoid ObjectId serialization issues)
    mapper_docs = list(db[mapper_coll_name].find({}, {"_id": 0}))
    if not mapper_docs:
        logger.error("No mapper submaster docs found in collection")
        return None

    # Sort by sm_id
    def sm_sort_key(doc):
        sm = doc.get("sm_id") or ""
        return sm

    mapper_docs.sort(key=sm_sort_key)
    N = len(mapper_docs)
    logger.info(f"Loaded {N} mapper submaster docs from {mapper_coll_name}")

    # Group into pairs
    groups = group_mapper_docs(mapper_docs, group_size=2)
    R = len(groups)
    logger.info(f"Creating {R} reducer submasters (groups of up to 2)")

    # Determine mapper workers per submaster by inspecting first mapper doc or fallback
    first_mapper = mapper_docs[0]
    mapper_workers = (
        mapper_workers_override
        or first_mapper.get("metadata", {}).get("num_workers_per_submaster")
        or first_mapper.get("num_workers_per_submaster")
        or int(os.getenv("DEFAULT_MAPPER_WORKERS", 3))
    )
    logger.info(f"Assumed mapper workers per submaster = {mapper_workers}")

    # reducer workers per rsm is ceil(mapper_workers / 2)
    reducer_workers_per_rsm = math.ceil(int(mapper_workers) / 2)
    logger.info(f"Reducer workers per submaster = {reducer_workers_per_rsm}")

    # Prepare global context from latest residual doc if available
    residual_coll = db[os.getenv("MONGO_RESIDUAL_COLLECTION", "residual_memory")]
    res_doc = residual_coll.find_one(sort=[("_id", -1)])
    global_context = res_doc.get("global_context", {}) if res_doc else metadata.get("global_context", {})

    # Ensure Ray is initialized with memory limits
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            num_cpus=4,
            object_store_memory=500 * 1024 * 1024,  # 500MB
        )

    rsm_refs = []
    for i, grp in enumerate(groups):
        rsm_id = f"RSM-{str(i+1).zfill(3)}"
        rsm = ReducerSubMasterActor.remote(
            rsm_id,
            grp,
            num_workers=reducer_workers_per_rsm,
            metadata=metadata,
            global_context=global_context,
        )
        rsm_refs.append(rsm)

    # initialize + run rsm actors
    init_futures = [r.initialize.remote() for r in rsm_refs]
    try:
        init_statuses = ray.get(init_futures)
        logger.info("Reducer submasters initialized")
    except Exception as e:
        logger.warning(f"Initialization warnings: {e}")

    proc_futures = [r.process.remote() for r in rsm_refs]
    try:
        results = ray.get(proc_futures)
    except Exception as e:
        logger.error(f"Error running reducer submasters: {e}")
        results = []

    # Merge per-rsm outputs into final reducer-level aggregated output
    aggregated = {
        "timestamp": time.time(),
        "rsm_results": results,
        "final_summary": "",
        "entities": {},
        "keywords": {},
        "technical_terms": {},
        "key_points": [],
        "insights": [],
        "brief_summaries": []  # Collect brief summaries from each RSM
    }

    for rdoc in results:
        out = rdoc.get("output", {})
        aggregated["final_summary"] += " " + (out.get("enhanced_summary") or "")
        
        # Collect brief summary from each submaster for residual agent
        brief_sum = out.get("brief_summary", "")
        if brief_sum:
            aggregated["brief_summaries"].append({
                "rsm_id": rdoc.get("rsm_id"),
                "brief_summary": brief_sum
            })
        
        for k, v in (out.get("entities") or {}).items():
            aggregated["entities"][k] = aggregated["entities"].get(k, 0) + int(v)
        for k, v in (out.get("keywords") or {}).items():
            aggregated["keywords"][k] = aggregated["keywords"].get(k, 0) + int(v)
        for k, v in (out.get("technical_terms") or {}).items():
            aggregated["technical_terms"][k] = aggregated["technical_terms"].get(k, 0) + int(v)
        aggregated["key_points"].extend(out.get("key_points", []))
        aggregated["insights"].extend(out.get("insights", []))

    aggregated["final_summary"] = aggregated["final_summary"].strip()
    aggregated["key_points"] = list(dict.fromkeys(aggregated["key_points"]))
    aggregated["insights"] = list(dict.fromkeys(aggregated["insights"]))
    
    # Generate overall brief summary for residual agent from all RSM brief summaries
    aggregated["overall_brief_summary"] = _generate_overall_brief_summary(
        aggregated["brief_summaries"], 
        aggregated
    )

    # Save final reducer result
    try:
        client = MongoClient(uri)
        db = client[dbname]
        db[os.getenv("MONGO_REDUCER_RESULTS_COLLECTION", "reducer_results")].insert_one(aggregated)
        logger.info("Persisted final reducer result to Mongo")
    except Exception as e:
        logger.error(f"Failed to persist final reducer result: {e}")

    return aggregated


def main():
    """
    Standalone entry. Running the module without args runs global mode.
    """
    # Load optional metadata file from env path or default location
    meta = {}
    meta_path = os.getenv("METADATA_PATH", "")
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata from {meta_path}: {e}")

    try:
        agg = run_reducer_global(metadata=meta)
        
        # Sanitize MongoDB ObjectIds before JSON serialization
        agg_clean = sanitize_mongo_doc(agg)
        
        print("\n=== FINAL REDUCER AGGREGATED OUTPUT ===\n")
        print(json.dumps(agg_clean, indent=2, ensure_ascii=False))
    finally:
        if ray.is_initialized():
            ray.shutdown()


if __name__ == "__main__":
    main()