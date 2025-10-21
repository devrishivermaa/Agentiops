# utils/llm_helper.py
"""
LLM Helper with retry logic, rate limiting, and error handling.
Designed for parallel SubMaster execution with Gemini API.
"""

import os
import time
import json
from typing import Dict, Any, Optional, List
from functools import wraps
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from utils.logger import get_logger

load_dotenv()
logger = get_logger("LLMHelper")


class RateLimiter:
    """Simple rate limiter to prevent API quota exhaustion."""
    
    def __init__(self, max_requests_per_minute: int = 60):
        self.max_requests = max_requests_per_minute
        self.request_times: List[float] = []
        self.lock_acquired = False
    
    def wait_if_needed(self):
        """Block if rate limit would be exceeded."""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.max_requests:
            # Wait until oldest request is >1 minute old
            sleep_time = 60 - (now - self.request_times[0]) + 0.1
            if sleep_time > 0:
                logger.warning(f"Rate limit reached. Sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
                self.request_times = []
        
        self.request_times.append(now)


class LLMProcessor:
    """
    Wrapper for LLM API calls with:
    - Retry logic with exponential backoff
    - Rate limiting
    - Error handling
    - Response parsing
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.3,
        max_retries: int = 3,
        rate_limit: int = 60
    ):
        """
        Initialize LLM processor.
        
        Args:
            model: Gemini model name
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Max retry attempts on failure
            rate_limit: Max requests per minute
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.rate_limiter = RateLimiter(rate_limit)
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key
            )
            logger.info(f"LLM initialized: {model} (temp={temperature})")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise
    
    def call_with_retry(self, prompt: str, parse_json: bool = False) -> Any:
        """
        Call LLM with retry logic.
        
        Args:
            prompt: Text prompt to send
            parse_json: If True, parse response as JSON
            
        Returns:
            LLM response text or parsed JSON
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Rate limiting
                self.rate_limiter.wait_if_needed()
                
                # Make API call
                response = self.llm.invoke([HumanMessage(content=prompt)])
                
                # Extract content
                content = self._extract_content(response)
                
                if not content:
                    raise ValueError("Empty response from LLM")
                
                # Parse JSON if requested
                if parse_json:
                    return self._parse_json(content)
                
                return content
                
            except Exception as e:
                last_error = e
                wait_time = (2 ** attempt) + (time.time() % 1)  # Exponential backoff with jitter
                
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time:.2f}s..."
                )
                
                if attempt < self.max_retries - 1:
                    time.sleep(wait_time)
        
        # All retries failed
        logger.error(f"LLM call failed after {self.max_retries} attempts: {last_error}")
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_error
    
    def _extract_content(self, response: Any) -> str:
        """Extract text content from LLM response."""
        content = getattr(response, "content", None)
        
        # Handle list of content blocks (Gemini format)
        if isinstance(content, list):
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        
        if not content or not isinstance(content, str):
            raise ValueError("Invalid response format from LLM")
        
        return content.strip()
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (``` markers)
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            # Remove json language hint
            if text.startswith("json"):
                text = text[4:].strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            logger.debug(f"Problematic text: {text[:500]}")
            raise


def create_analysis_prompt(
    role: str,
    text: str,
    page_num: int,
    section_name: Optional[str] = None,
    processing_requirements: Optional[List[str]] = None
) -> str:
    """
    Create a role-specific prompt for document analysis.
    
    Args:
        role: SubMaster's role description
        text: Extracted text from PDF page
        page_num: Page number being processed
        section_name: Name of document section (if known)
        processing_requirements: List of required tasks (e.g., ["entity_extraction"])
    
    Returns:
        Formatted prompt string
    """
    requirements = processing_requirements or []
    
    prompt_parts = [
        f"You are analyzing page {page_num} of a research paper.",
        f"\nYour role: {role}",
    ]
    
    if section_name:
        prompt_parts.append(f"\nSection: {section_name}")
    
    prompt_parts.append(f"\n\n=== PAGE {page_num} TEXT ===\n{text}\n{'=' * 40}")
    
    prompt_parts.append("\n\nAnalyze this page and provide:")
    
    # Add specific requirements based on processing needs
    if "summary_generation" in requirements or "summarization" in role.lower():
        prompt_parts.append("\n1. SUMMARY: A concise 2-3 sentence summary of the key points")
    
    if "entity_extraction" in requirements or "entit" in role.lower():
        prompt_parts.append("\n2. ENTITIES: List key entities (people, organizations, methods, datasets, metrics)")
    
    if "keyword_indexing" in requirements or "keyword" in role.lower():
        prompt_parts.append("\n3. KEYWORDS: Extract 5-10 important keywords or key phrases")
    
    # Role-specific additions
    if "methodology" in role.lower() or "method" in role.lower():
        prompt_parts.append("\n4. METHODS: Identify specific algorithms, techniques, or approaches mentioned")
    
    if "result" in role.lower() or "finding" in role.lower():
        prompt_parts.append("\n4. FINDINGS: Extract key results, metrics, or observations")
    
    if "discussion" in role.lower() or "conclusion" in role.lower():
        prompt_parts.append("\n4. INSIGHTS: Main conclusions, implications, or takeaways")
    
    prompt_parts.append("""

Respond in JSON format:
{
  "summary": "Brief summary of the page",
  "entities": ["entity1", "entity2", ...],
  "keywords": ["keyword1", "keyword2", ...],
  "key_points": ["point1", "point2", ...],
  "technical_terms": ["term1", "term2", ...]
}

Be concise and factual. Extract actual content from the text.
""")
    
    return "".join(prompt_parts)


# Convenience function for SubMasters
def analyze_page(
    llm_processor: LLMProcessor,
    role: str,
    text: str,
    page_num: int,
    section_name: Optional[str] = None,
    processing_requirements: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze a single page using LLM.
    
    Args:
        llm_processor: Initialized LLMProcessor instance
        role: SubMaster's role
        text: Page text
        page_num: Page number
        section_name: Section name (optional)
        processing_requirements: Processing tasks (optional)
    
    Returns:
        Structured analysis results
    """
    prompt = create_analysis_prompt(
        role=role,
        text=text,
        page_num=page_num,
        section_name=section_name,
        processing_requirements=processing_requirements
    )
    
    try:
        result = llm_processor.call_with_retry(prompt, parse_json=True)
        result["page"] = page_num
        result["status"] = "success"
        return result
    except Exception as e:
        logger.error(f"Analysis failed for page {page_num}: {e}")
        return {
            "page": page_num,
            "status": "error",
            "error": str(e),
            "summary": f"[ERROR: Analysis failed for page {page_num}]"
        }
