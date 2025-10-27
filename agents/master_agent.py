# agents/master_agent.py
import os
import json
import uuid
import re
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from utils.logger import get_logger
from utils.pdf_extractor import PDFExtractor


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

        model = model or "gemini-2.0-flash-exp"
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

    # ----------------------------- Plan Validation -----------------------------
    def validate_plan(self, plan: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the generated plan has valid page ranges.
        
        Returns:
            Dict with "valid": bool and "errors": list
        """
        num_pages = metadata.get("num_pages", 100)
        errors = []
        
        submasters = plan.get("submasters", [])
        if not submasters:
            errors.append("No submasters defined in plan")
            return {"valid": False, "errors": errors}
        
        for idx, sm in enumerate(submasters):
            sm_id = sm.get("submaster_id", f"SM-{idx}")
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
                    errors.append(f"{sm_id}: Page end {end} > {num_pages} (PDF has {num_pages} pages)")
                if start > end:
                    errors.append(f"{sm_id}: Page start {start} > end {end}")
        
        if errors:
            self.logger.warning(f"Plan validation found {len(errors)} errors")
            return {"valid": False, "errors": errors}
        
        self.logger.info("Plan validation passed")
        return {"valid": True, "errors": []}

    # ----------------------------- PDF Validation & Metadata Correction -----------------------------
    def validate_and_fix_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate metadata against actual PDF file and fix discrepancies.
        
        Returns:
            Updated metadata with corrections applied
        """
        pdf_path = metadata.get("file_path", "")
        
        if not pdf_path or not os.path.exists(pdf_path):
            self.logger.error(f"PDF file not found: {pdf_path}")
            return metadata
        
        try:
            # Extract actual PDF info
            extractor = PDFExtractor(pdf_path)
            actual_pages = extractor.num_pages
            pdf_metadata = extractor.get_metadata()
            
            self.logger.info(f"PDF has {actual_pages} pages (metadata says {metadata.get('num_pages')})")
            
            # Fix 1: Correct page count
            if metadata.get("num_pages") != actual_pages:
                self.logger.warning(f"Correcting num_pages: {metadata.get('num_pages')} -> {actual_pages}")
                metadata["num_pages"] = actual_pages
            
            # Fix 2: Ensure consistent file_size_mb
            actual_file_size = pdf_metadata.get("file_size_mb", 0)
            if "pdf_metadata" not in metadata:
                metadata["pdf_metadata"] = {}
            metadata["file_size_mb"] = actual_file_size
            metadata["pdf_metadata"]["file_size_mb"] = actual_file_size
            
            # Fix 3: Use absolute path consistently
            abs_path = os.path.abspath(pdf_path)
            metadata["file_path"] = abs_path
            metadata["pdf_metadata"]["file_path"] = abs_path
            
            # Fix 4: Validate and fix sections - ENSURE COMPLETE COVERAGE
            sections = metadata.get("sections", {})
            fixed_sections = self._fix_sections_coverage(sections, actual_pages)
            
            metadata["sections"] = fixed_sections
            
            # Add actual PDF metadata
            metadata["pdf_metadata"].update(pdf_metadata)
            metadata["validated_against_pdf"] = True
            
            self.logger.info(f"Metadata validated and corrected. {len(fixed_sections)} valid sections.")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Error validating PDF: {e}")
            return metadata

    def _fix_sections_coverage(self, sections: Dict[str, Any], num_pages: int) -> Dict[str, Any]:
        """
        Fix sections to ensure complete page coverage.
        
        Args:
            sections: Current section definitions
            num_pages: Total pages in PDF
            
        Returns:
            Fixed sections with complete coverage
        """
        fixed_sections = {}
        removed_sections = []
        
        # First pass: validate and fix existing sections
        for section_name, section_info in sections.items():
            start = section_info.get("page_start", 1)
            end = section_info.get("page_end", 1)
            
            # Check if section is within valid range
            if start > num_pages or end > num_pages:
                self.logger.warning(
                    f"Section '{section_name}' pages {start}-{end} exceed PDF length ({num_pages}). "
                    f"Removing section."
                )
                removed_sections.append(section_name)
                continue
            
            # Clamp to valid range
            if end > num_pages:
                self.logger.warning(f"Clamping section '{section_name}' end page: {end} -> {num_pages}")
                section_info["page_end"] = num_pages
            
            fixed_sections[section_name] = section_info
        
        # Second pass: check for gaps and add missing sections
        covered_pages = set()
        for section_info in fixed_sections.values():
            start = section_info.get("page_start", 1)
            end = section_info.get("page_end", 1)
            covered_pages.update(range(start, end + 1))
        
        # Find uncovered pages
        all_pages = set(range(1, num_pages + 1))
        uncovered_pages = sorted(all_pages - covered_pages)
        
        if uncovered_pages:
            self.logger.warning(f"Found {len(uncovered_pages)} uncovered pages: {uncovered_pages[:10]}...")
            
            # Group consecutive uncovered pages into sections
            groups = []
            current_group = [uncovered_pages[0]]
            
            for page in uncovered_pages[1:]:
                if page == current_group[-1] + 1:
                    current_group.append(page)
                else:
                    groups.append(current_group)
                    current_group = [page]
            groups.append(current_group)
            
            # Add sections for uncovered pages
            for idx, group in enumerate(groups):
                section_name = f"Additional_Section_{idx+1}"
                fixed_sections[section_name] = {
                    "page_start": group[0],
                    "page_end": group[-1],
                    "description": f"Auto-generated section for pages {group[0]}-{group[-1]}"
                }
                self.logger.info(f"Added section '{section_name}' for pages {group[0]}-{group[-1]}")
        
        if removed_sections:
            print(f"\n⚠️  Removed {len(removed_sections)} invalid sections: {', '.join(removed_sections)}")
        
        self.logger.info(f"Section coverage fixed. Total sections: {len(fixed_sections)}")
        return fixed_sections

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
3. ALL pages from 1 to num_pages must be covered by at least one section.
4. Any other critical inconsistencies that would prevent processing.

IGNORE these minor issues:
- Small differences in file_size_mb (< 0.1 MB difference is acceptable)
- Relative vs absolute paths (both are valid)
- Minor formatting differences

Respond ONLY in JSON format:

{{
  "status": "ok" | "issues",
  "issues": ["description of each CRITICAL problem found"]
}}

If sections cover all pages and page ranges are valid, status should be "ok".
"""
        response = self.llm.invoke([HumanMessage(content=prompt)])
        result = self.extract_json(response.content)
        
        # Filter out non-critical issues
        if result.get("status") == "issues":
            critical_issues = []
            issues = result.get("issues", [])
            
            for issue in issues:
                issue_lower = issue.lower()
                # Skip non-critical issues
                if any(keyword in issue_lower for keyword in ["file_size", "file_path", "file path", "close", "minor"]):
                    self.logger.info(f"Ignoring non-critical issue: {issue}")
                    continue
                critical_issues.append(issue)
            
            # If no critical issues remain, mark as ok
            if not critical_issues:
                result["status"] = "ok"
                result["issues"] = []
            else:
                result["issues"] = critical_issues
        
        return result

    # ----------------------------- LLM SubMaster Planning -----------------------------
    def ask_llm_for_plan(self, user_request: str, metadata: dict):
        """Ask the LLM to generate a structured SubMaster plan based on sections."""
        num_submasters = self.estimate_submasters_needed(metadata)
        num_pages = metadata.get("num_pages", 100)
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

⚠️ CRITICAL CONSTRAINTS:
- The PDF has EXACTLY {num_pages} pages. DO NOT assign page ranges beyond page {num_pages}.
- Every page_range MUST be within [1, {num_pages}].
- Verify ALL page numbers before responding.

TASK:
- Divide this document among {num_submasters} SubMasters.
- Each SubMaster should handle one or more sections logically.
- Each must have a distinct role (summarization, entity extraction, keyword extraction, etc.).
- Respect section boundaries; avoid splitting a section unless necessary.
- Include reasoning under "distribution_strategy".
- ENSURE all page ranges are valid (start >= 1, end <= {num_pages}).

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

        # STEP 1: Validate and fix metadata against actual PDF
        print("\n🔍 Validating metadata against actual PDF...")
        metadata = self.validate_and_fix_metadata(metadata)
        
        # STEP 2: Validate metadata with LLM
        print("\n🤖 Validating metadata with LLM...")
        validation = self.validate_metadata_with_llm(metadata)
        
        if validation["status"] == "issues":
            print("\n⚠️  Critical metadata issues detected:")
            for i, issue in enumerate(validation["issues"], 1):
                print(f" {i}. {issue}")
            
            # Ask user if they want to proceed anyway
            proceed = input("\n❓ Metadata has issues. Try to proceed anyway? (yes/no): ").strip().lower()
            if proceed not in ["yes", "y"]:
                print("\nPlease fix the metadata before proceeding.")
                return {"status": "needs_fix", "issues": validation["issues"]}
            else:
                print("\n⚠️  Proceeding despite metadata issues...")
        else:
            print("✅ Metadata validation passed")

        # Use user_notes as processing goal
        goal = metadata.get("user_notes", "Process the document according to standard workflow.")

        # Generate initial plan
        print("\n📋 Generating SubMaster execution plan...")
        plan = self.ask_llm_for_plan(goal, metadata)
        for sm in plan.get("submasters", []):
            sm["submaster_id"] = f"SM-{uuid.uuid4().hex[:6].upper()}"

        # Persistent feedback loop
        while True:
            # Validate plan
            plan_validation = self.validate_plan(plan, metadata)
            if not plan_validation["valid"]:
                print("\n⚠️ Plan validation errors detected:")
                for error in plan_validation["errors"]:
                    print(f"   ❌ {error}")
                print("\n🔄 Regenerating plan...")
                
                # Ask LLM to fix the plan
                fix_prompt = f"""
The previous plan had validation errors:
{chr(10).join(f"- {e}" for e in plan_validation["errors"])}

CRITICAL: The PDF has {metadata.get('num_pages')} pages.
All page ranges MUST be within [1, {metadata.get('num_pages')}].

Generate a corrected plan with valid page ranges.
"""
                plan = self.ask_llm_for_plan(goal + "\n\n" + fix_prompt, metadata)
                for sm in plan.get("submasters", []):
                    sm["submaster_id"] = f"SM-{uuid.uuid4().hex[:6].upper()}"
                continue
            
            print("\n📋 Proposed SubMaster Plan:\n")
            print(json.dumps(plan, indent=2))

            approval = input("\n✅ Approve this plan? (yes/no): ").strip().lower()
            if approval in ["yes", "y"]:
                self.logger.info("User approved the plan.")
                plan["status"] = "approved"
                return plan

            feedback = input("\n📝 What would you like to change? Specify adjustments:\n> ").strip()
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

        if plan is None or plan.get("status") == "needs_fix":
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
    metadata_path = "metadata.json"
    print("\n--- MASTER AGENT ONLINE ---\n")
    agent.execute(metadata_path)
