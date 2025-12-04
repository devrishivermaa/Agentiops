"""
LLM Helper with Mistral AI direct SDK and rate limiting.
"""

import os
import time
import json
import threading
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from mistralai import Mistral
from utils.logger import get_logger

load_dotenv()
logger = get_logger("LLMHelper")


class GlobalRateLimiter:
    """
    Global rate limiter for Mistral AI API.
    Adjust limits based on your Mistral tier.
    """
    
    def __init__(self, max_requests_per_minute: int = 50, max_requests_per_day: int = 5000):
        """
        Initialize rate limiter.
        
        Args:
            max_requests_per_minute: Max requests per minute (default: 50)
            max_requests_per_day: Max requests per day (default: 5000)
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_day = max_requests_per_day
        
        self.request_times_minute: List[float] = []
        self.request_times_day: List[float] = []
        
        self.lock = threading.Lock()
        self.min_delay_between_requests = 1.2  # ~50 RPM
        self.last_request_time = 0.0
        
        logger.info(
            f"GlobalRateLimiter initialized: {max_requests_per_minute} RPM, "
            f"{max_requests_per_day} requests/day (Mistral AI)"
        )
    
    def wait_if_needed(self, caller_id: str = "unknown"):
        """Block if rate limit would be exceeded."""
        with self.lock:
            now = time.time()
            
            # RULE 1: Enforce minimum delay
            time_since_last = now - self.last_request_time
            if self.last_request_time > 0 and time_since_last < self.min_delay_between_requests:
                wait_time = self.min_delay_between_requests - time_since_last
                logger.debug(
                    f"[{caller_id}] Enforcing {self.min_delay_between_requests}s minimum delay. "
                    f"Waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time)
                now = time.time()
            
            # RULE 2: Check per-minute limit
            self.request_times_minute = [t for t in self.request_times_minute if now - t < 60]
            
            if len(self.request_times_minute) >= self.max_requests_per_minute:
                oldest_request = self.request_times_minute[0]
                wait_time = 60 - (now - oldest_request) + 2
                
                if wait_time > 0:
                    logger.warning(
                        f"[{caller_id}] Per-minute rate limit reached "
                        f"({len(self.request_times_minute)}/{self.max_requests_per_minute}). "
                        f"Waiting {wait_time:.1f}s"
                    )
                    time.sleep(wait_time)
                    now = time.time()
                    self.request_times_minute = [t for t in self.request_times_minute if now - t < 60]
            
            # RULE 3: Check per-day limit
            self.request_times_day = [t for t in self.request_times_day if now - t < 86400]
            
            if len(self.request_times_day) >= self.max_requests_per_day:
                oldest_request_day = self.request_times_day[0]
                wait_time = 86400 - (now - oldest_request_day) + 60
                
                logger.error(
                    f"[{caller_id}] DAILY QUOTA REACHED! "
                    f"({len(self.request_times_day)}/{self.max_requests_per_day}). "
                    f"Must wait {wait_time/3600:.1f} hours"
                )
                
                raise RuntimeError(
                    f"Daily quota exceeded ({len(self.request_times_day)}/{self.max_requests_per_day}). "
                    f"Please wait {wait_time/3600:.1f} hours."
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


# Global rate limiter instance
_global_rate_limiter = GlobalRateLimiter(
    max_requests_per_minute=50,
    max_requests_per_day=5000
)


class LLMProcessor:
    """
    LLM wrapper with Mistral AI SDK and retry logic.
    """
    
    def __init__(
        self,
        model: str = "mistral-small-latest",
        temperature: float = 0.3,
        max_retries: int = 5,
        caller_id: str = "unknown"
    ):
        """
        Initialize LLM processor with Mistral AI.
        
        Args:
            model: Mistral model name (e.g., "mistral-small-latest", "mistral-large-latest")
            temperature: Sampling temperature (0.0-1.0)
            max_retries: Max retry attempts on failure
            caller_id: Identifier for logging
        """
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY not found in environment variables. "
                "Add it to your .env file."
            )
        
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.caller_id = caller_id
        
        try:
            # Initialize Mistral client (no context manager needed for long-lived usage)
            self.client = Mistral(api_key=api_key)
            logger.info(f"[{caller_id}] Mistral AI client initialized: {model} (temp={temperature})")
        except Exception as e:
            logger.error(f"[{caller_id}] Failed to initialize Mistral client: {e}")
            raise
    
    def call_with_retry(self, prompt: str, parse_json: bool = False, max_tokens: int = 2048) -> Any:
        """
        Call Mistral AI with retry logic and global rate limiting.
        
        Args:
            prompt: Text prompt to send
            parse_json: If True, parse response as JSON
            max_tokens: Maximum tokens in response
            
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
                
                # Make API call using Mistral SDK
                response = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )
                
                # Extract content
                content = response.choices[0].message.content
                
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
                
                # Check if it's a rate limit error
                if "429" in str(e) or "rate" in error_str or "limit" in error_str:
                    wait_time = min(120, (2 ** attempt) * 10)  # 10s, 20s, 40s, 80s, 120s
                    
                    logger.warning(
                        f"[{self.caller_id}] Rate limit hit (attempt {attempt + 1}/{self.max_retries}). "
                        f"Waiting {wait_time:.1f}s before retry..."
                    )
                    
                    if attempt < self.max_retries - 1:
                        time.sleep(wait_time)
                    else:
                        logger.error(
                            f"[{self.caller_id}] Failed after {self.max_retries} attempts. "
                            f"Rate limit still active."
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
    
    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response, handling markdown code blocks."""
        import re
        
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
            if text.startswith("json"):
                text = text[4:].strip()
        
        # Find JSON object
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No valid JSON found in LLM response")
        
        try:
            return json.loads(match.group(0))
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
    
    # Limit text to avoid token issues
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
            "summary": f"[ERROR: Daily quota exceeded. Please wait or check your Mistral account.]",
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
