# utils/llm_helper.py
"""
LLM Helper with enhanced retry logic and rate limiting.
Optimized for Gemini API free tier (15 RPM).
"""

import os
import time
import json
import threading
from typing import Dict, Any, Optional, List
from functools import wraps
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from utils.logger import get_logger

load_dotenv()
logger = get_logger("LLMHelper")


class GlobalRateLimiter:
    """
    Global rate limiter shared across all SubMasters.
    Prevents exceeding Gemini free tier limits (15 RPM).
    """
    
    def __init__(self, max_requests_per_minute: int = 12):  # Leave buffer of 3
        self.max_requests = max_requests_per_minute
        self.request_times: List[float] = []
        self.lock = threading.Lock()
        logger.info(f"GlobalRateLimiter initialized: {max_requests_per_minute} RPM")
    
    def wait_if_needed(self, caller_id: str = "unknown"):
        """Block if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if now - t < 60]
            
            if len(self.request_times) >= self.max_requests:
                # Calculate wait time
                oldest_request = self.request_times[0]
                wait_time = 60 - (now - oldest_request) + 1  # +1 for safety buffer
                
                if wait_time > 0:
                    logger.warning(
                        f"[{caller_id}] Global rate limit reached "
                        f"({len(self.request_times)}/{self.max_requests}). "
                        f"Sleeping for {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    # Clear old requests after waiting
                    now = time.time()
                    self.request_times = [t for t in self.request_times if now - t < 60]
            
            # Record this request
            self.request_times.append(time.time())
            logger.debug(
                f"[{caller_id}] Request allowed. "
                f"Current: {len(self.request_times)}/{self.max_requests}"
            )


# Global rate limiter instance shared across all SubMasters
_global_rate_limiter = GlobalRateLimiter(max_requests_per_minute=12)


class LLMProcessor:
    """
    Wrapper for LLM API calls with:
    - Retry logic with exponential backoff
    - Global rate limiting
    - Error handling
    - Response parsing
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.3,
        max_retries: int = 5,
        rate_limit: int = 60,  # DEPRECATED: kept for backward compatibility
        caller_id: str = "unknown"  # NEW PARAMETER
    ):
        """
        Initialize LLM processor.
        
        Args:
            model: Gemini model name
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Max retry attempts on failure
            rate_limit: DEPRECATED - global rate limiter is used instead
            caller_id: Identifier for logging (e.g., SubMaster ID or Worker ID)
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.caller_id = caller_id
        
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=api_key
            )
            logger.info(f"[{caller_id}] LLM initialized: {model} (temp={temperature})")
        except Exception as e:
            logger.error(f"[{caller_id}] Failed to initialize LLM: {e}")
            raise
    
    def call_with_retry(self, prompt: str, parse_json: bool = False) -> Any:
        """
        Call LLM with retry logic and global rate limiting.
        
        Args:
            prompt: Text prompt to send
            parse_json: If True, parse response as JSON
            
        Returns:
            LLM response text or parsed JSON
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                # Global rate limiting BEFORE API call
                _global_rate_limiter.wait_if_needed(self.caller_id)
                
                # Make API call
                response = self.llm.invoke([HumanMessage(content=prompt)])
                
                # Extract content
                content = self._extract_content(response)
                
                if not content:
                    raise ValueError("Empty response from LLM")
                
                # Parse JSON if requested
                if parse_json:
                    return self._parse_json(content)
                
                logger.debug(f"[{self.caller_id}] LLM call successful (attempt {attempt+1})")
                return content
                
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check if it's a quota error
                if "429" in error_str or "quota" in error_str.lower():
                    # Exponential backoff with longer waits for quota errors
                    wait_time = min(60, (2 ** attempt) * 10)  # 10s, 20s, 40s, 60s
                    logger.warning(
                        f"[{self.caller_id}] Rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                        f"Waiting {wait_time:.1f}s before retry..."
                    )
                    time.sleep(wait_time)
                else:
                    # Standard exponential backoff for other errors
                    wait_time = (2 ** attempt) + (time.time() % 1)
                    logger.warning(
                        f"[{self.caller_id}] LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time:.2f}s..."
                    )
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
        
        # All retries failed
        logger.error(f"[{self.caller_id}] LLM call failed after {self.max_retries} attempts: {last_error}")
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
            logger.error(f"[{self.caller_id}] Failed to parse JSON: {e}")
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
        processing_requirements: List of required tasks
    
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
    
    # Add specific requirements
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
