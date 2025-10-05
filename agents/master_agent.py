# agents/master_agent.py
import os
import json
import uuid
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from utils.logger import get_logger

# ----------------------------- Setup -----------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment!")

# ----------------------------- MasterAgent -----------------------------
class MasterAgent:
    """
    MasterAgent orchestrates document processing by:
    1. Validating metadata via LLM.
    2. Asking an LLM to generate a section-aware SubMaster plan.
    3. Running a persistent feedback loop until the user approves the plan.
    4. Saving the final plan to JSON.
    """

    def __init__(self, model=None, temperature=0.3):
        self.id = f"MA-{uuid.uuid4().hex[:6].upper()}"
        self.logger = get_logger("MasterAgent")

        model = model or "gemini-2.5-flash-lite-preview-06-17"
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.logger.info(f"[INIT] Master Agent {self.id} initialized with model {model}.")

    # ----------------------------- Utility -----------------------------
    def extract_json(self, text: str) -> dict:
        """Extract valid JSON from LLM response."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            self.logger.error("No valid JSON found in LLM response.")
            raise ValueError("No valid JSON found in LLM response")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            raise

    def estimate_submasters_needed(self, metadata: dict) -> int:
        """Estimate number of SubMasters based on complexity, priority, and page count."""
        pages = metadata.get("num_pages", 10)
        complexity = metadata.get("complexity_level", "medium").lower()
        priority = metadata.get("priority", "medium").lower()

        base = 1
        if pages > 50:
            base += 1
        if complexity in ["high", "complex"]:
            base += 1
        if priority == "high":
            base += 1

        return min(base, metadata.get("max_parallel_submasters", 4))

    # ----------------------------- LLM Metadata Validation -----------------------------
    def validate_metadata_with_llm(self, metadata: dict) -> dict:
        """
        Ask LLM to validate metadata and detect discrepancies.
        Returns {"status": "ok"} or {"status": "issues", "issues": [...]}
        """
        prompt = f"""
You are a MasterAgent tasked with validating document metadata.

DOCUMENT METADATA:
{json.dumps(metadata, indent=2)}

Check for:
1. Missing critical fields like file_name, num_pages, document_type, sections.
2. Sections with invalid page ranges (page_start > page_end, start < 1, end > num_pages).
3. Overlapping sections.
4. Sections that do not cover the full document.
5. Any other inconsistencies you notice.

Respond ONLY in JSON format:

{{
  "status": "ok" | "issues",
  "issues": ["description of each problem found"]
}}
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self.extract_json(response.content)

    # ----------------------------- LLM SubMaster Planning -----------------------------
    def ask_llm_for_plan(self, user_request: str, metadata: dict):
        """Ask the LLM to generate a structured SubMaster plan based on sections."""
        num_submasters = self.estimate_submasters_needed(metadata)
        sections = metadata.get("sections", {})
        section_summary = json.dumps(sections, indent=2)

        prompt = f"""
You are the MasterAgent, responsible for delegating document processing tasks.

GOAL:
{user_request}

DOCUMENT METADATA (excluding sections):
{json.dumps({k: v for k, v in metadata.items() if k != "sections"}, indent=2)}

DOCUMENT SECTIONS (with page ranges):
{section_summary}

TASK:
- Divide this document among {num_submasters} SubMasters.
- Each SubMaster should handle one or more sections logically.
- Each must have a distinct role (summarization, entity extraction, keyword extraction, etc.).
- Respect section boundaries; avoid splitting a section unless necessary.
- Include reasoning under "distribution_strategy".

Respond ONLY in JSON format like this:

{{
  "num_submasters": {num_submasters},
  "distribution_strategy": "Explain how sections are split among SubMasters and why.",
  "submasters": [
    {{
      "role": "Handle Abstract + Introduction for summarization.",
      "assigned_sections": ["Abstract", "Introduction"],
      "page_range": [1, 8],
      "estimated_workload": "medium"
    }}
  ]
}}
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self.extract_json(response.content)

    # ----------------------------- Feedback Loop -----------------------------
    def feedback_loop(self, metadata: dict):
        """Persistent interactive loop — continues until user approves the plan."""
        self.logger.info("[START] Entering feedback loop...")

        # Validate metadata first
        validation = self.validate_metadata_with_llm(metadata)
        if validation["status"] == "issues":
            print("\n⚠ Metadata discrepancies detected by LLM:")
            for i, issue in enumerate(validation["issues"], 1):
                print(f" {i}. {issue}")
            print("\nPlease fix the metadata before proceeding.")
            return {"status": "needs_fix", "issues": validation["issues"]}

        # Use user_notes as processing goal
        goal = metadata.get("user_notes", "Process the document according to standard workflow.")

        # Generate initial plan
        plan = self.ask_llm_for_plan(goal, metadata)
        for sm in plan.get("submasters", []):
            sm["submaster_id"] = f"SM-{uuid.uuid4().hex[:6].upper()}"

        # Persistent feedback loop
        while True:
            print("\n📋 Proposed SubMaster Plan:\n")
            print(json.dumps(plan, indent=2))

            approval = input("\n✅ Approve this plan? (yes/no): ").strip().lower()
            if approval in ["yes", "y"]:
                self.logger.info("User approved the plan.")
                plan["status"] = "approved"
                return plan

            feedback = input("\n📝 What would you like to change? Specify adjustments (merge sections, change roles, adjust workloads):\n> ").strip()
            if not feedback:
                print("\n⚠ No feedback provided. Please specify a change.")
                continue

            self.logger.info(f"User feedback: {feedback}")
            revised_prompt = f"""
The user has requested revisions to your previous plan.

USER GOAL:
{goal}

CURRENT PLAN:
{json.dumps(plan, indent=2)}

USER FEEDBACK:
{feedback}

Revise the plan accordingly. Keep JSON structure identical.
"""
            response = self.llm.invoke([HumanMessage(content=revised_prompt)])
            plan = self.extract_json(response.content)
            for sm in plan.get("submasters", []):
                sm["submaster_id"] = f"SM-{uuid.uuid4().hex[:6].upper()}"

    # ----------------------------- Main Execution -----------------------------
    def execute(self, metadata_path: str, save_path: str = None):
        """Run the full orchestration and save final SubMaster plan."""
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None

        self.logger.info(f"[EXEC] Processing document: {metadata.get('file_name')}")
        plan = self.feedback_loop(metadata)

        if plan.get("status") == "needs_fix":
            return None

        # Determine save path
        if save_path is None:
            save_path = os.path.join(os.path.dirname(metadata_path), "submasters_plan.json")

        # Save final plan
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(plan, f, indent=2)
            self.logger.info(f"Final SubMaster plan saved to: {save_path}")
        except Exception as e:
            self.logger.error(f"Failed to save SubMaster plan: {e}")
            return None

        print("\n✅ Final Approved Plan:\n")
        print(json.dumps(plan, indent=2))
        print(f"\n📁 SubMaster plan saved to: {save_path}")
        return plan

# ----------------------------- CLI Entry -----------------------------
if __name__ == "__main__":
    agent = MasterAgent()
    metadata_path = r"C:\Users\devri\OneDrive\Desktop\Agentiops\metadata.json"
    print("\n--- MASTER AGENT ONLINE ---\n")
    agent.execute(metadata_path)
