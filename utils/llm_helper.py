# utils/llm_helper.py
"""
LLM Helper with ULTRA-AGGRESSIVE rate limiting for Gemini free tier.
Ensures we NEVER exceed 10 RPM and 50 requests/day.
"""

import os
import time
import json
import threading
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from utils.logger import get_logger

load_dotenv()
logger = get_logger("LLMHelper")


class GlobalRateLimiter:
    """
    Ultra-aggressive global rate limiter for Gemini free tier.
    Ensures NEVER exceeding 10 RPM or 50 requests/day.
    """
    
    def __init__(self, max_requests_per_minute: int = 8, max_requests_per_day: int = 45):
        """
        Initialize rate limiter with conservative limits.
        
        Args:
            max_requests_per_minute: Max requests per minute (default: 8, limit is 10)
            max_requests_per_day: Max requests per day (default: 45, limit is 50)
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_day = max_requests_per_day
        
        self.request_times_minute: List[float] = []  # Track last minute
        self.request_times_day: List[float] = []  # Track last 24 hours
        
        self.lock = threading.Lock()
        self.min_delay_between_requests = 8.0  # Minimum 8 seconds between ANY requests
        self.last_request_time = 0.0
        
        logger.info(
            f"GlobalRateLimiter initialized: {max_requests_per_minute} RPM, "
            f"{max_requests_per_day} requests/day (ULTRA-CONSERVATIVE)"
        )
    
    def wait_if_needed(self, caller_id: str = "unknown"):
        """
        Block if rate limit would be exceeded.
        Enforces BOTH per-minute AND per-day limits.
        """
        with self.lock:
            now = time.time()
            
            # RULE 1: Enforce absolute minimum delay between ANY requests
            time_since_last = now - self.last_request_time
            if self.last_request_time > 0 and time_since_last < self.min_delay_between_requests:
                wait_time = self.min_delay_between_requests - time_since_last
                logger.warning(
                    f"[{caller_id}] Enforcing {self.min_delay_between_requests}s minimum delay. "
                    f"Waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time)
                now = time.time()
            
            # RULE 2: Check per-minute limit (last 60 seconds)
            self.request_times_minute = [t for t in self.request_times_minute if now - t < 60]
            
            if len(self.request_times_minute) >= self.max_requests_per_minute:
                oldest_request = self.request_times_minute[0]
                wait_time = 60 - (now - oldest_request) + 2  # +2s safety buffer
                
                if wait_time > 0:
                    logger.warning(
                        f"[{caller_id}] Per-minute rate limit reached "
                        f"({len(self.request_times_minute)}/{self.max_requests_per_minute}). "
                        f"Waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    now = time.time()
                    self.request_times_minute = [t for t in self.request_times_minute if now - t < 60]
            
            # RULE 3: Check per-day limit (last 24 hours)
            self.request_times_day = [t for t in self.request_times_day if now - t < 86400]
            
            if len(self.request_times_day) >= self.max_requests_per_day:
                oldest_request_day = self.request_times_day[0]
                wait_time = 86400 - (now - oldest_request_day) + 60  # +1min safety buffer
                
                logger.error(
                    f"[{caller_id}] DAILY QUOTA REACHED! "
                    f"({len(self.request_times_day)}/{self.max_requests_per_day}). "
                    f"Must wait {wait_time/3600:.1f} hours"
                )
                
                # Don't actually wait 24 hours, just raise an error
                raise RuntimeError(
                    f"Daily quota exceeded ({len(self.request_times_day)}/{self.max_requests_per_day}). "
                    f"Please wait {wait_time/3600:.1f} hours or upgrade to paid tier."
                )
            
            # Record this request
            self.request_times_minute.append(now)
            self.request_times_day.append(now)
            self.last_request_time = now
            
            logger.debug(
                f"[{caller_id}] Request allowed. "
                f"Minute: {len(self.request_times_minute)}/{self.max_requests_per_minute}, "
                f"Day: {len(self.request_times_day)}/{self.max_requests_per_day}"
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current rate limiter statistics."""
        now = time.time()
        self.request_times_minute = [t for t in self.request_times_minute if now - t < 60]
        self.request_times_day = [t for t in self.request_times_day if now - t < 86400]
        
        return {
            "requests_last_minute": len(self.request_times_minute),
            "max_per_minute": self.max_requests_per_minute,
            "requests_last_day": len(self.request_times_day),
            "max_per_day": self.max_requests_per_day,
            "percent_daily_quota_used": (len(self.request_times_day) / self.max_requests_per_day) * 100
        }


# Global rate limiter instance (ULTRA-CONSERVATIVE: 8 RPM, 45/day)
_global_rate_limiter = GlobalRateLimiter(
    max_requests_per_minute=8,  # Conservative: 8 RPM (limit is 10)
    max_requests_per_day=45     # Conservative: 45/day (limit is 50)
)


class LLMProcessor:
    """
    LLM wrapper with retry logic and global rate limiting.
    """
    
    def __init__(
        self,
        model: str = "gemini-2.0-flash-exp",
        temperature: float = 0.3,
        max_retries: int = 5,
        caller_id: str = "unknown"
    ):
        """
        Initialize LLM processor.
        
        Args:
            model: Gemini model name
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Max retry attempts on failure
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
                # CRITICAL: Global rate limiting BEFORE API call
                try:
                    _global_rate_limiter.wait_if_needed(self.caller_id)
                except RuntimeError as e:
                    # Daily quota exceeded - propagate immediately
                    logger.error(f"[{self.caller_id}] Daily quota exceeded: {e}")
                    raise
                
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
                
            except RuntimeError as e:
                # Daily quota exceeded - don't retry
                raise
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                
                # Check if it's a quota error
                if "429" in str(e) or "quota" in error_str or "resource exhausted" in error_str:
                    # Exponential backoff with MUCH longer waits for quota errors
                    wait_time = min(120, (2 ** attempt) * 20)  # 20s, 40s, 80s, 120s, 120s
                    
                    logger.warning(
                        f"[{self.caller_id}] Quota/rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                        f"Waiting {wait_time:.1f}s before retry..."
                    )
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"[{self.caller_id}] Failed after {self.max_retries} attempts. "
                            f"You may have exceeded daily quota."
                        )
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
    Create detailed, role-specific analysis prompt.
    
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
    
    # Enhanced prompt for better summaries (limit text to avoid token issues)
    text_preview = text[:4000] if len(text) > 4000 else text
    truncated = " ...[truncated for length]" if len(text) > 4000 else ""
    
    prompt = f"""You are a research paper analyst processing page {page_num} from section: {section_name or 'Unknown'}.

Your role: {role}

=== PAGE {page_num} CONTENT ===
{text_preview}{truncated}
========================================

ANALYSIS TASKS:
"""
    
    # Add specific instructions based on requirements
    if "summary_generation" in requirements or "summar" in role.lower():
        prompt += """
1. SUMMARY (3-5 sentences):
   - What is the main topic/finding on this page?
   - What methods/approaches are mentioned?
   - What results/conclusions are presented?
   - Be specific and factual.
"""
    
    if "entity_extraction" in requirements or "entit" in role.lower():
        prompt += """
2. ENTITIES (extract ALL relevant):
   - Methods/algorithms (e.g., "Vision Transformer", "BERT", "ResNet")
   - Datasets (e.g., "ImageNet", "COCO", "ConceptARC")
   - Metrics (e.g., "accuracy", "F1-score", "BLEU")
   - People/authors (e.g., "Vaswani et al.", "Dosovitskiy")
   - Organizations (e.g., "Google", "OpenAI")
"""
    
    if "keyword_indexing" in requirements or "keyword" in role.lower():
        prompt += """
3. KEYWORDS (7-12 important terms):
   - Technical terms specific to this page
   - Key concepts mentioned
   - Domain-specific vocabulary
"""
    
    # Section-specific instructions
    if section_name:
        if "abstract" in section_name.lower():
            prompt += "\n4. FOCUS: Extract paper's main contribution, methods, and key results."
        elif "introduction" in section_name.lower():
            prompt += "\n4. FOCUS: Identify problem statement, motivation, and paper contributions."
        elif "method" in section_name.lower() or "body" in section_name.lower():
            prompt += "\n4. FOCUS: Detail algorithms, architectures, and technical approaches."
        elif "result" in section_name.lower():
            prompt += "\n4. FOCUS: Extract experimental results, metrics, and comparisons."
        elif "conclusion" in section_name.lower():
            prompt += "\n4. FOCUS: Summarize findings, limitations, and future work."
    
    prompt += """

RESPONSE FORMAT (valid JSON only):
{
  "summary": "Detailed 3-5 sentence summary capturing key information",
  "entities": ["Entity1", "Entity2", "Entity3", ...],
  "keywords": ["keyword1", "keyword2", "keyword3", ...],
  "key_points": ["Important point 1", "Important point 2", ...],
  "technical_terms": ["term1", "term2", ...]
}

Be thorough and extract ALL relevant information. This is critical for research paper analysis.
"""
    
    return prompt


def analyze_page(
    llm_processor: LLMProcessor,
    role: str,
    text: str,
    page_num: int,
    section_name: Optional[str] = None,
    processing_requirements: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Analyze a single page using LLM with improved prompts.
    
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
    # Skip if text is too short
    if len(text.strip()) < 100:
        return {
            "page": page_num,
            "section": section_name,
            "status": "skipped",
            "summary": "[Page has insufficient text for analysis]",
            "entities": [],
            "keywords": [],
            "key_points": [],
            "technical_terms": []
        }
    
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
        result["section"] = section_name
        result["status"] = "success"
        return result
    except RuntimeError as e:
        # Daily quota exceeded
        logger.error(f"Daily quota exceeded for page {page_num}: {e}")
        return {
            "page": page_num,
            "section": section_name,
            "status": "error",
            "error": "DAILY_QUOTA_EXCEEDED",
            "summary": f"[ERROR: Daily quota exceeded. Please wait or upgrade to paid tier.]",
            "entities": [],
            "keywords": [],
            "key_points": [],
            "technical_terms": []
        }
    except Exception as e:
        logger.error(f"Analysis failed for page {page_num}: {e}")
        return {
            "page": page_num,
            "section": section_name,
            "status": "error",
            "error": str(e),
            "summary": f"[ERROR: Analysis failed - {str(e)[:100]}]",
            "entities": [],
            "keywords": [],
            "key_points": [],
            "technical_terms": []
        }


def get_rate_limiter_stats() -> Dict[str, Any]:
    """Get current rate limiter statistics."""
    return _global_rate_limiter.get_stats()
