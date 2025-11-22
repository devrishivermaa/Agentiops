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
        
        model = model or "gemini-2.0-flash-exp"
        
       
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
        """Extract JSON from LLM response."""
        # Remove markdown if present
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
        """Estimate number of SubMasters based on document characteristics."""
        pages = metadata.get("num_pages", 10)
        complexity = metadata.get("complexity_level", "medium").lower()
        
        base = metadata.get("max_parallel_submasters", 2)
        
        # Adjust based on page count
        if pages > 50:
            base = min(base + 1, 4)
        
        # Adjust based on complexity
        if complexity == "high":
            base = min(base + 1, 4)
        
        return base
    
    def ask_llm_for_plan(self, user_request: str, metadata: dict):
        """Generate SubMaster plan based on sections."""
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
- Each SubMaster needs a distinct role (e.g., "Summarize Abstract and Introduction", "Extract entities from Methodology")
- page_range format: [start1, end1, start2, end2, ...] for multiple ranges OR [start, end] for single range

Respond ONLY in valid JSON:
{{
  "num_submasters": {num_submasters},
  "distribution_strategy": "Brief explanation of how sections are distributed and why",
  "submasters": [
    {{
      "submaster_id": "SM-001",
      "role": "Summarize Abstract and Introduction sections for overview",
      "assigned_sections": ["Abstract", "Introduction"],
      "page_range": [1, 2],
      "estimated_workload": "medium"
    }}
  ]
}}

ENSURE all page numbers are valid before responding.
"""
        
        # FIXED: Use rate-limited LLMProcessor instead of direct call
        try:
            response_text = self.llm.call_with_retry(prompt, parse_json=False)
            plan = self.extract_json(response_text)
        except RuntimeError as e:
            # Daily quota exceeded
            self.logger.error(f"Daily quota exceeded: {e}")
            raise RuntimeError(
                "Daily API quota exceeded. Please wait 24 hours or upgrade to paid tier."
            ) from e
        
        # Add unique IDs if missing
        for sm in plan.get("submasters", []):
            if "submaster_id" not in sm:
                sm["submaster_id"] = f"SM-{uuid.uuid4().hex[:6].upper()}"
        
        return plan
    
    def validate_plan(self, plan: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate plan has valid page ranges."""
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
            
            # Check each (start, end) pair
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
        """Interactive loop until user approves plan."""
        self.logger.info("[START] Entering feedback loop...")
        
        goal = metadata.get("user_notes", "Process the document according to standard workflow.")
        
        # Generate initial plan
        print("\n📋 Generating SubMaster execution plan...")
        
        try:
            plan = self.ask_llm_for_plan(goal, metadata)
        except RuntimeError as e:
            # Daily quota exceeded
            print(f"\n❌ {str(e)}")
            return {"status": "quota_exceeded", "error": str(e)}
        
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            attempts += 1
            
            # Validate plan
            validation = self.validate_plan(plan, metadata)
            if not validation["valid"]:
                print(f"\n⚠️ Plan validation errors (attempt {attempts}/{max_attempts}):")
                for error in validation["errors"]:
                    print(f"   ❌ {error}")
                
                if attempts >= max_attempts:
                    print("\n❌ Max validation attempts reached. Please check metadata.")
                    return {"status": "validation_failed", "errors": validation["errors"]}
                
                print("\n🔄 Regenerating plan...")
                fix_prompt = f"""
Previous plan had errors:
{chr(10).join(f"- {e}" for e in validation["errors"])}

CRITICAL: PDF has {metadata.get('num_pages')} pages. All ranges MUST be [1, {metadata.get('num_pages')}].
Generate corrected plan.
"""
                try:
                    plan = self.ask_llm_for_plan(goal + "\n\n" + fix_prompt, metadata)
                except RuntimeError as e:
                    print(f"\n❌ {str(e)}")
                    return {"status": "quota_exceeded", "error": str(e)}
                continue
            
            # Display plan
            print("\n📋 Proposed SubMaster Plan:\n")
            print(json.dumps(plan, indent=2))
            
            # Check if feedback is required
            if not metadata.get("feedback_required", True):
                print("\n✅ Auto-approving plan (feedback_required=False)")
                plan["status"] = "approved"
                return plan
            
            approval = input("\n✅ Approve this plan? (yes/no): ").strip().lower()
            if approval in ["yes", "y"]:
                self.logger.info("User approved the plan.")
                plan["status"] = "approved"
                return plan
            
            feedback = input("\n📝 What would you like to change?\n> ").strip()
            if not feedback:
                print("\n⚠️ No feedback provided. Please specify changes.")
                continue
            
            self.logger.info(f"User feedback: {feedback}")
            revised_prompt = f"""
USER GOAL:
{goal}

CURRENT PLAN:
{json.dumps(plan, indent=2)}

USER FEEDBACK:
{feedback}

Revise the plan accordingly. Keep the same JSON structure.
"""
            try:
                plan = self.ask_llm_for_plan(revised_prompt, metadata)
            except RuntimeError as e:
                print(f"\n❌ {str(e)}")
                return {"status": "quota_exceeded", "error": str(e)}
        
        print("\n⚠️ Max attempts reached without approval.")
        return None
    
    def save_plan_to_mongo(self, plan: dict, metadata: dict):
        """Save plan to MongoDB."""
        if not self.mongo_coll:
            self.logger.warning("MongoDB not configured. Cannot save plan.")
            return

        try:
            doc = {
                "master_id": self.id,
                "timestamp": time.time(),
                "plan": plan,
                "metadata": metadata
            }
            self.mongo_coll.insert_one(doc)
            self.logger.info("[MONGO] Plan saved to MongoDB")
        except Exception as e:
            self.logger.error(f"[MONGO] Failed to save plan: {e}")
    
    def save_metadata_to_mongo(self, metadata: dict):
        """Save metadata to MongoDB."""
        if self.mongo_metadata_coll is None:
            self.logger.warning("MongoDB metadata collection not configured. Skipping save.")
            return

        try:
            doc = {
                "master_id": self.id,
                "timestamp": time.time(),
                "metadata": metadata
            }
            self.mongo_metadata_coll.insert_one(doc)
            self.logger.info("[MONGO] Metadata saved to MongoDB")
        except Exception as e:
            self.logger.error(f"[MONGO] Failed to save metadata: {e}")
    
    def execute(self, metadata_path: str, save_path: str = None):
        """Run orchestration and save final plan."""
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
                self.save_metadata_to_mongo(metadata)
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None

        self.logger.info(f"[EXEC] Processing: {metadata.get('file_name')}")
        plan = self.feedback_loop(metadata)

        if plan is None or plan.get("status") not in ["approved", "quota_exceeded"]:
            return plan

        if plan.get("status") == "quota_exceeded":
            return plan

        # Determine save path
        if save_path is None:
            save_path = os.path.join(os.path.dirname(metadata_path), "submasters_plan.json")

        # Save plan to local JSON
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2)
            self.logger.info(f"Final plan saved: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save plan locally: {e}")
            return None

        # Save plan to MongoDB
        try:
            self.save_plan_to_mongo(plan, metadata)
        except Exception as e:
            self.logger.error(f"Failed to save plan to MongoDB: {e}")

        print("\n✅ Final Approved Plan:")
        print(json.dumps(plan, indent=2))
        print(f"\n📁 Plan saved: {save_path}")

        return plan


if __name__ == "__main__":
    import sys
    
    agent = MasterAgent()
    print("\n--- MASTER AGENT ONLINE ---\n")
    
    # Parse command-line argument
    if len(sys.argv) < 2:
        print("Usage: python -m agents.master_agent <metadata_file_path>")
        sys.exit(1)
    
    metadata_path = sys.argv[1]
    agent.execute(metadata_path)
