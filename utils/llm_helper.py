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


# ============================================================
# Global Rate Limiter
# ============================================================

class GlobalRateLimiter:
    def __init__(self, max_requests_per_minute: int = 30, max_requests_per_day: int = 5000):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_day = max_requests_per_day
        
        self.request_times_minute: List[float] = []
        self.request_times_day: List[float] = []
        
        self.lock = threading.Lock()
        # Balance between rate limiting and performance
        self.min_delay_between_requests = 2.5  # 2.5 seconds between requests per worker
        self.last_request_time = 0.0
        
        logger.info(
            f"GlobalRateLimiter initialized: {max_requests_per_minute} RPM, "
            f"{max_requests_per_day} requests per day"
        )

    def wait_if_needed(self, caller_id: str = "unknown"):
        with self.lock:
            now = time.time()

            # enforce spacing
            delta = now - self.last_request_time
            if delta < self.min_delay_between_requests:
                wait = self.min_delay_between_requests - delta
                time.sleep(wait)
                now = time.time()

            # check last minute window
            self.request_times_minute = [t for t in self.request_times_minute if now - t < 60]
            if len(self.request_times_minute) >= self.max_requests_per_minute:
                oldest = self.request_times_minute[0]
                wait = 60 - (now - oldest) + 2
                time.sleep(wait)
                now = time.time()
                self.request_times_minute = [t for t in self.request_times_minute if now - t < 60]

            # check daily window
            self.request_times_day = [t for t in self.request_times_day if now - t < 86400]
            if len(self.request_times_day) >= self.max_requests_per_day:
                raise RuntimeError("Daily quota exceeded")

            # record
            self.request_times_minute.append(now)
            self.request_times_day.append(now)
            self.last_request_time = now


_global_rate_limiter = GlobalRateLimiter()


# ============================================================
# LLM Processor
# ============================================================

class LLMProcessor:
    def __init__(
        self,
        model: str = "mistral-small-latest",
        temperature: float = 0.3,
        max_retries: int = 5,
        caller_id: str = "unknown"
    ):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("Missing MISTRAL_API_KEY")

        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.caller_id = caller_id

        try:
            self.client = Mistral(api_key=api_key)
            logger.info(f"[{caller_id}] Mistral client initialized {model}")
        except Exception as e:
            raise RuntimeError(f"Failed to init client {e}")

    def call_with_retry(self, prompt: str, parse_json: bool = False, max_tokens: int = 4096) -> Any:
        last_error = None

        for attempt in range(self.max_retries):
            try:
                _global_rate_limiter.wait_if_needed(self.caller_id)

                response = self.client.chat.complete(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=max_tokens
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response")

                if parse_json:
                    return self._parse_json(content)

                return content

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # daily quota needs immediate fail
                if isinstance(e, RuntimeError) and "quota" in error_str:
                    raise

                # Rate limit - wait longer
                if "429" in str(e) or "rate" in error_str:
                    wait = min(180, (2 ** attempt) * 15)  # Longer waits for rate limits
                    logger.warning(f"[{self.caller_id}] Rate limited, waiting {wait}s before retry")
                else:
                    wait = min(120, (2 ** attempt) * 5)
                    
                time.sleep(wait)

        raise RuntimeError(f"LLM failed after retries: {last_error}")

    def _parse_json(self, text: str) -> Dict[str, Any]:
        import re
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])
            if text.startswith("json"):
                text = text[4:].strip()

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found")

        return json.loads(match.group(0))


# ============================================================
# GLOBAL CONTEXT AWARE PROMPT CONSTRUCTION
# ============================================================

def create_analysis_prompt(
    role: str,
    text: str,
    page_num: int,
    section_name: Optional[str] = None,
    processing_requirements: Optional[List[str]] = None,
    global_context: Optional[Any] = None
) -> str:
    """
    Extended prompt builder that embeds ResidualAgent global context.
    """

    requirements = processing_requirements or []

    # text trimming
    text_preview = text[:4000]
    truncated_text = " ...[truncated]" if len(text) > 4000 else ""

    # global context trimming
    gc_json = ""
    if global_context:
        try:
            gc_str = json.dumps(global_context, indent=2)
            if len(gc_str) > 3000:
                gc_str = gc_str[:3000] + " ...[global context truncated]"
            gc_json = gc_str
        except Exception:
            gc_json = str(global_context)[:3000]

    # build prompt
    prompt = f"""
You are a research analysis worker. You must ensure your output is globally consistent with the entire system.

=== GLOBAL CONTEXT (from ResidualAgent) ===
{gc_json}

=== YOUR ROLE ===
{role}

=== PAGE BEING ANALYZED ===
Page number: {page_num}
Section: {section_name or "Unknown"}

=== PAGE CONTENT ===
{text_preview}{truncated_text}

=== REQUIRED TASKS ===
"""

    if "summary_generation" in requirements:
        prompt += """
1. Write a clear summary in 3 to 5 sentences.
"""

    if "entity_extraction" in requirements:
        prompt += """
2. Extract all entities mentioned on this page.
"""

    if "keyword_indexing" in requirements:
        prompt += """
3. Extract 7 to 12 important keywords.
"""

    prompt += """

=== OUTPUT FORMAT (strict JSON) ===
{
  "summary": "...",
  "entities": ["e1", "e2"],
  "keywords": ["k1", "k2"],
  "key_points": ["point1", "point2"],
  "technical_terms": ["t1", "t2"]
}

Do not include commentary. Return only JSON.
"""

    return prompt


# ============================================================
# PAGE ANALYSIS (worker level)
# ============================================================

def analyze_page(
    llm_processor: LLMProcessor,
    role: str,
    text: str,
    page_num: int,
    section_name: Optional[str] = None,
    processing_requirements: Optional[List[str]] = None,
    global_context=None
) -> Dict[str, Any]:

    if len(text.strip()) < 80:
        return {
            "page": page_num,
            "section": section_name,
            "status": "skipped",
            "summary": "[Page too short]",
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
        processing_requirements=processing_requirements,
        global_context=global_context         # crucial fix
    )

    try:
        result = llm_processor.call_with_retry(prompt, parse_json=True)
        result["page"] = page_num
        result["section"] = section_name
        result["status"] = "success"
        return result

    except Exception as e:
        return {
            "page": page_num,
            "section": section_name,
            "status": "error",
            "error": str(e),
            "summary": "",
            "entities": [],
            "keywords": [],
            "key_points": [],
            "technical_terms": []
        }


def get_rate_limiter_stats() -> Dict[str, Any]:
    return _global_rate_limiter.get_stats()
