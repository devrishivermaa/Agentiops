# agents/master_agent.py
"""
MasterAgent with integrated rate limiting and retry logic.
"""

import os
import json
import uuid
import re
from typing import Dict, Any
from dotenv import load_dotenv
from utils.logger import get_logger
from utils.llm_helper import LLMProcessor 
from pymongo import MongoClient
import time

load_dotenv()
logger = get_logger("MasterAgent")

class MasterAgent:
    """MasterAgent generates SubMaster execution plans with user feedback."""
    
    def __init__(self, model=None, temperature=0.3):
        self.id = f"MA-{uuid.uuid4().hex[:6].upper()}"
        self.logger = logger
        
        model = model or os.getenv("LLM_MODEL", "mistral-small-latest")
        
        self.llm = LLMProcessor(
            model=model,
            temperature=temperature,
            max_retries=3,
            caller_id=self.id
        )
        
        self.mongo_client = None
        self.mongo_db = None
        self.mongo_coll = None
        self.mongo_metadata_coll = None  

        try:
            uri = os.getenv("MONGO_URI")
            db_name = os.getenv("MONGO_DB")
            coll_name = os.getenv("MONGO_MASTER_COLLECTION", "master_agent")
            metadata_coll_name = os.getenv("MONGO_METADATA_COLLECTION", "metadata")

            if uri:
                self.mongo_client = MongoClient(uri)
                # if db_name is None this will raise; leaving behavior unchanged
                self.mongo_db = self.mongo_client[db_name]
                self.mongo_coll = self.mongo_db[coll_name]
                self.mongo_metadata_coll = self.mongo_db[metadata_coll_name]
                
                self.logger.info(f"[INIT] Connected to MongoDB collection {db_name}.{coll_name}")
                self.logger.info(f"[INIT] Connected extra metadata collection {db_name}.{metadata_coll_name}")
            else:
                self.logger.warning("[INIT] MONGO_URI is missing. MongoDB disabled.")

        except Exception as e:
            self.logger.error(f"[INIT] Failed to connect MongoDB: {e}")
        
        self.logger.info(f"[INIT] Master Agent {self.id} initialized with model {model}.")
    
    def extract_json(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            if text.startswith("json"):
                text = text[4:].strip()
        
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found in LLM response")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            raise
    
    def estimate_submasters_needed(self, metadata: dict) -> int:
        pages = metadata.get("num_pages", 10)
        complexity = metadata.get("complexity_level", "medium").lower()
        
        base = metadata.get("max_parallel_submasters", 2)
        
        if pages > 50:
            base = min(base + 1, 4)
        
        if complexity == "high":
            base = min(base + 1, 4)
        
        return base
    
    def ask_llm_for_plan(self, user_request: str, metadata: dict):
        num_submasters = self.estimate_submasters_needed(metadata)
        num_pages = metadata.get("num_pages", 100)
        sections = metadata.get("sections", {})
        
        section_summary = "\n".join([
            f"  - {name}: pages {info['page_start']}-{info['page_end']} ({info.get('description', 'N/A')})"
            for name, info in sections.items()
        ])
        
        prompt = f"""
You are the MasterAgent creating a document processing plan.

DOCUMENT: {metadata.get('file_name')}
- Total Pages: {num_pages}
- Document Type: {metadata.get('document_type')}
- Complexity: {metadata.get('complexity_level')}

SECTIONS:
{section_summary}

USER GOAL:
{user_request}

CRITICAL CONSTRAINTS:
- The PDF has EXACTLY {num_pages} pages
- ALL page ranges MUST be within [1, {num_pages}]
- Divide the document among {num_submasters} SubMasters
- Each SubMaster should handle 1+ sections logically
- Each SubMaster needs a distinct role
- page_range format: [start, end] or multiple pairs

Respond ONLY in valid JSON:
{{
  "num_submasters": {num_submasters},
  "distribution_strategy": "Brief explanation",
  "submasters": [
    {{
      "submaster_id": "SM-001",
      "role": "Summarize Abstract and Introduction",
      "assigned_sections": ["Abstract", "Introduction"],
      "page_range": [1, 2],
      "estimated_workload": "medium"
    }}
  ]
}}
"""
        
        try:
            response_text = self.llm.call_with_retry(prompt, parse_json=False)
            plan = self.extract_json(response_text)
        except RuntimeError as e:
            self.logger.error(f"Daily quota exceeded: {e}")
            raise RuntimeError(
                "Daily API quota exceeded. Please wait 24 hours or upgrade to paid tier."
            ) from e
        
        for sm in plan.get("submasters", []):
            if "submaster_id" not in sm:
                sm["submaster_id"] = f"SM-{uuid.uuid4().hex[:6].upper()}"
        
        return plan
    
    def validate_plan(self, plan: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        num_pages = metadata.get("num_pages", 100)
        errors = []
        
        submasters = plan.get("submasters", [])
        if not submasters:
            errors.append("No submasters defined in plan")
            return {"valid": False, "errors": errors}
        
        for sm in submasters:
            sm_id = sm.get("submaster_id", "UNKNOWN")
            page_range = sm.get("page_range", [])
            
            if not page_range or len(page_range) % 2 != 0:
                errors.append(f"{sm_id}: Invalid page_range format: {page_range}")
                continue
            
            for i in range(0, len(page_range), 2):
                start = page_range[i]
                end = page_range[i + 1]
                
                if start < 1:
                    errors.append(f"{sm_id}: Page start {start} < 1")
                if end > num_pages:
                    errors.append(f"{sm_id}: Page end {end} > {num_pages}")
                if start > end:
                    errors.append(f"{sm_id}: Page start {start} > end {end}")
        
        if errors:
            return {"valid": False, "errors": errors}
        
        return {"valid": True, "errors": []}
    
    def feedback_loop(self, metadata: dict):
        self.logger.info("[START] Entering feedback loop...")
        
        goal = metadata.get("user_notes", "Process the document according to standard workflow.")
        
        print("\nGenerating SubMaster execution plan...")
        
        try:
            plan = self.ask_llm_for_plan(goal, metadata)
        except RuntimeError as e:
            print(f"\n{str(e)}")
            return {"status": "quota_exceeded", "error": str(e)}
        
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            validation = self.validate_plan(plan, metadata)
            if not validation["valid"]:
                print(f"\nPlan validation errors (attempt {attempts}):")
                for error in validation["errors"]:
                    print(f"   {error}")
                
                if attempts >= max_attempts:
                    print("\nMax validation attempts reached. Please check metadata.")
                    return {"status": "validation_failed", "errors": validation["errors"]}
                
                print("\nRegenerating plan...")
                fix_prompt = f"""
Previous plan had errors:
{chr(10).join(f"- {e}" for e in validation["errors"])}

PDF has {metadata.get('num_pages')} pages. All ranges must be valid.
Generate corrected plan.
"""
                try:
                    plan = self.ask_llm_for_plan(goal + "\n\n" + fix_prompt, metadata)
                except RuntimeError as e:
                    print(f"\n{str(e)}")
                    return {"status": "quota_exceeded", "error": str(e)}
                continue
            
            print("\nProposed SubMaster Plan:\n")
            print(json.dumps(plan, indent=2))
            
            if not metadata.get("feedback_required", True):
                print("\nAuto approving plan")
                plan["status"] = "approved"
                return plan
            
            approval = input("\nApprove this plan? (yes/no): ").strip().lower()
            if approval in ["yes", "y"]:
                self.logger.info("User approved the plan.")
                plan["status"] = "approved"
                return plan
            
            feedback = input("\nWhat would you like to change?\n> ").strip()
            if not feedback:
                print("\nNo feedback provided. Please specify changes.")
                continue
            
            self.logger.info(f"User feedback: {feedback}")
            revised_prompt = f"""
USER GOAL:
{goal}

CURRENT PLAN:
{json.dumps(plan, indent=2)}

USER FEEDBACK:
{feedback}

Revise the plan. Keep JSON structure unchanged.
"""
            try:
                plan = self.ask_llm_for_plan(revised_prompt, metadata)
            except RuntimeError as e:
                print(f"\n{str(e)}")
                return {"status": "quota_exceeded", "error": str(e)}
        
        print("\nMax attempts reached without approval.")
        return None

    # ----------------------------------------------------
    # FIXED FUNCTION: use explicit None check and JSON safe
    # ----------------------------------------------------
    def save_plan_to_mongo(self, plan: dict, metadata: dict):
        if self.mongo_coll is None:
            self.logger.warning("MongoDB not configured. Cannot save plan.")
            return

        try:
            safe_plan = json.loads(json.dumps(plan))
            safe_metadata = json.loads(json.dumps(metadata))

            doc = {
                "master_id": self.id,
                "timestamp": time.time(),
                "plan": safe_plan,
                "metadata": safe_metadata
            }
            self.mongo_coll.insert_one(doc)
            self.logger.info("[MONGO] Plan saved to MongoDB")
        except Exception as e:
            self.logger.error(f"[MONGO] Failed to save plan: {e}")

    # ----------------------------------------------------
    # FIXED FUNCTION: explicit None check and JSON safe
    # ----------------------------------------------------
    def save_metadata_to_mongo(self, metadata: dict):
        if self.mongo_metadata_coll is None:
            self.logger.warning("MongoDB metadata collection not configured.")
            return

        try:
            safe_metadata = json.loads(json.dumps(metadata))
            doc = {
                "master_id": self.id,
                "timestamp": time.time(),
                "metadata": safe_metadata
            }
            self.mongo_metadata_coll.insert_one(doc)
            self.logger.info("[MONGO] Metadata saved to MongoDB")
        except Exception as e:
            self.logger.error(f"[MONGO] Failed to save metadata: {e}")

    # ----------------------------------------------------
    # FIXED FUNCTION: always saves plan (except None)
    # ----------------------------------------------------
    def execute(self, metadata_path: str, save_path: str = None):
        try:
            with open(metadata_path, "r", encoding="utf8") as f:
                metadata = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None

        # Prompt user for high level intent and document context and merge into metadata
        try:
            print("\nPlease provide a brief high level intent for this document (example: 'Summarize for a presentation'):")
            high_level_intent = input("> ").strip()
            if high_level_intent:
                metadata["high_level_intent"] = high_level_intent

            print("\nProvide any document context the user wants to add (example: 'this paper is about how ai is as good at reasoning as humans'):")
            user_doc_context = input("> ").strip()
            if user_doc_context:
                metadata["user_document_context"] = user_doc_context

            # Add a metadata update timestamp
            metadata["metadata_updated_at"] = time.time()

            # Save enriched metadata back to disk and to Mongo
            try:
                with open(metadata_path, "w", encoding="utf8") as f:
                    json.dump(metadata, f, indent=2)
                self.logger.info(f"Enriched metadata written to {metadata_path}")
            except Exception as e:
                self.logger.error(f"Failed to write enriched metadata to disk: {e}")

            # Save metadata to mongo
            try:
                self.save_metadata_to_mongo(metadata)
            except Exception as e:
                self.logger.error(f"Failed to save metadata to MongoDB: {e}")

        except Exception as e:
            self.logger.error(f"Failed to enrich metadata with user inputs: {e}")

        self.logger.info(f"[EXEC] Processing: {metadata.get('file_name')}")
        plan = self.feedback_loop(metadata)

        if plan is None:
            self.logger.error("No plan returned from feedback loop")
            return None

        # attach master summary into metadata and persist again
        try:
            master_summary = plan.get("distribution_strategy", "")
            if master_summary:
                metadata["master_summary"] = master_summary
                metadata["metadata_updated_at"] = time.time()
                # save updated metadata to disk
                try:
                    with open(metadata_path, "w", encoding="utf8") as f:
                        json.dump(metadata, f, indent=2)
                    self.logger.info(f"Updated metadata with master_summary written to {metadata_path}")
                except Exception as e:
                    self.logger.error(f"Failed to write updated metadata to disk: {e}")

                # save updated metadata to mongo
                try:
                    self.save_metadata_to_mongo(metadata)
                except Exception as e:
                    self.logger.error(f"Failed to save updated metadata to MongoDB: {e}")
        except Exception as e:
            self.logger.error(f"Failed to attach master_summary into metadata: {e}")

        if save_path is None:
            save_path = os.path.join(os.path.dirname(metadata_path), "submasters_plan.json")

        try:
            with open(save_path, "w", encoding="utf8") as f:
                json.dump(plan, f, indent=2)
            self.logger.info(f"Final plan saved locally to: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save plan locally: {e}")

        try:
            self.save_plan_to_mongo(plan, metadata)
        except Exception as e:
            self.logger.error(f"MongoDB insert failed: {e}")

        print("\nCompleted. Final plan stored locally and in MongoDB")
        return plan


if __name__ == "__main__":
    import sys
    
    agent = MasterAgent()
    print("\nMASTER AGENT ONLINE\n")
    
    if len(sys.argv) < 2:
        print("Usage: python -m agents.master_agent <metadata_file_path>")
        sys.exit(1)
    
    metadata_path = sys.argv[1]
    agent.execute(metadata_path)
