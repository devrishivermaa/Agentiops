# agents/master_agent.py
import os
import json
import uuid
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from utils.logger import get_logger

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment!")

class MasterAgent:
    def __init__(self, model="gemini-2.5-flash-lite-preview-06-17", temperature=0.3):
        self.id = str(uuid.uuid4())[:8]
        self.logger = get_logger("MasterAgent")
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=temperature)
        self.logger.info(f"[INIT] Master Agent {self.id} initialized.")

    def extract_json(self, text: str) -> dict:
        """Extract JSON object from LLM text output."""
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("No valid JSON found in response")

    def ask_llm_for_plan(self, user_request: str, metadata: dict):
        """Ask LLM to decide number of submasters and page splits."""
        prompt = f"""
You are a Master Agent organizing document processing.

USER REQUEST:
{user_request}

DOCUMENT METADATA:
{json.dumps(metadata, indent=2)}

Decide:
1. How many SubMasters are needed (1–4) based on total pages and complexity.
2. Assign page ranges to each SubMaster (e.g., pages 0–10, 11–20, etc.)
3. Each SubMaster must get:
   - A unique ID (generated in Python)
   - Assigned page range
   - Workload level (low/medium/high)

Respond **only** in JSON format:

{{
  "num_submasters": <int>,
  "distribution_strategy": "explain how pages are divided logically",
  "submasters": [
    {{
      "role": "Describe this submaster’s focus",
      "page_range": [start_page, end_page],
      "estimated_workload": "low/medium/high"
    }}
  ]
}}
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return self.extract_json(response.content)

    def feedback_loop(self, user_request: str, metadata: dict):
        """
        Master feedback loop:
        1. Checks if metadata has enough info.
        2. If not, asks the user up to 2 clarification questions.
        3. Once clarified, generates plan.
        """
        self.logger.info("[START] Master feedback loop...")

        required_fields = ["num_pages", "file_name"]
        missing = [f for f in required_fields if f not in metadata]
        clarifications = []

        if missing:
            for field in missing:
                clarifications.append(f"Please provide {field} for the document.")
            self.logger.warning("Clarifications needed: " + ", ".join(clarifications))
            return {"status": "needs_clarification", "clarifications": clarifications}

        self.logger.info("Metadata sufficient. Generating submaster plan...")
        plan = self.ask_llm_for_plan(user_request, metadata)

        # Assign unique Python-generated IDs
        for i, sm in enumerate(plan["submasters"], 1):
            sm["submaster_id"] = f"SM-{uuid.uuid4().hex[:6].upper()}"

        plan["status"] = "approved"
        return plan

    def execute(self, user_request: str, metadata_path: str):
        """Main entry point for running MasterAgent."""
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None

        result = self.feedback_loop(user_request, metadata)

        if result["status"] == "needs_clarification":
            print("\n⚠ Clarifications needed:")
            for c in result["clarifications"]:
                print(f"- {c}")
            return None

        print("\n✅ Final SubMaster Plan:")
        print(json.dumps(result, indent=2))
        return result


if __name__ == "__main__":
    agent = MasterAgent()
    metadata_path = r"C:\Users\devri\OneDrive\Desktop\Agentiops\metadata.json"

    user_request = input("\nEnter your processing goal: ").strip() or "Split document for efficient processing"

    print("\n--- MASTER AGENT INITIALIZED ---\n")
    agent.execute(user_request, metadata_path)
