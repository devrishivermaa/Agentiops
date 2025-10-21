# agents/sub_master.py
import time, random, uuid
import ray
from utils.logger import get_logger
from utils.pdf_extractor import PDFExtractor
from utils.llm_helper import LLMProcessor, analyze_page

logger = get_logger("SubMaster")


@ray.remote
class SubMaster:
    def __init__(self, plan_piece, metadata):
        self.sm_id = plan_piece.get("submaster_id", f"SM-{uuid.uuid4().hex[:6].upper()}")
        self.role = plan_piece.get("role", "generic")
        self.sections = plan_piece.get("assigned_sections", [])
        self.pages = plan_piece.get("page_range", [1, 1])
        self.meta = metadata
        self.status = "initialized"
        
        # Get PDF path from metadata
        self.pdf_path = metadata.get("file_path", "")
        if not self.pdf_path:
            logger.error(f"[{self.sm_id}] No file_path in metadata!")
        
        # Get processing requirements
        self.processing_requirements = metadata.get("processing_requirements", [])
        
        # Get LLM config from metadata
        self.llm_model = metadata.get("preferred_model", "gemini-2.0-flash-exp")
        
        # Initialize PDF extractor and LLM processor (will be done in initialize())
        self.pdf_extractor = None
        self.llm_processor = None
        
        logger.info(
            f"[{self.sm_id}] Init: role={self.role}, pages={self.pages}, "
            f"model={self.llm_model}, pdf={self.pdf_path}"
        )

    def initialize(self):
        """Initialize the PDF extractor and LLM processor for this SubMaster."""
        self.status = "ready"
        
        # Initialize PDF extractor
        try:
            if self.pdf_path:
                self.pdf_extractor = PDFExtractor(self.pdf_path)
                logger.info(f"[{self.sm_id}] PDF extractor initialized for {self.pdf_path}")
            else:
                logger.warning(f"[{self.sm_id}] No PDF path provided, will use mock data")
        except Exception as e:
            logger.error(f"[{self.sm_id}] Failed to initialize PDF extractor: {e}")
            self.pdf_extractor = None
        
        # Initialize LLM processor
        try:
            self.llm_processor = LLMProcessor(
                model=self.llm_model,
                temperature=0.3,
                max_retries=3,
                rate_limit=60  # Max 60 requests per minute per SubMaster
            )
            logger.info(f"[{self.sm_id}] LLM processor initialized with model {self.llm_model}")
        except Exception as e:
            logger.error(f"[{self.sm_id}] Failed to initialize LLM processor: {e}")
            self.llm_processor = None
        
        return {"sm_id": self.sm_id, "status": "ready"}

    def process(self):
        """Process assigned pages: extract text from PDF and analyze with LLM."""
        self.status = "running"
        output = []
        
        # Track statistics
        total_chars_extracted = 0
        total_entities = 0
        total_keywords = 0
        llm_successes = 0
        llm_failures = 0

        # Handle multiple page ranges
        if len(self.pages) % 2 != 0:
            raise ValueError(f"Invalid page_range for {self.sm_id}: {self.pages}")

        # Iterate over each (start, end) pair
        for i in range(0, len(self.pages), 2):
            start, end = self.pages[i], self.pages[i + 1]
            
            if self.pdf_extractor:
                # STEP 1: Extract text from PDF
                try:
                    logger.info(f"[{self.sm_id}] Extracting pages {start}-{end} from PDF...")
                    extracted_pages = self.pdf_extractor.extract_page_range(start, end)
                    
                    # STEP 2: Process each page with LLM
                    for page_num, text in extracted_pages.items():
                        total_chars_extracted += len(text)
                        
                        # Determine section name for this page
                        section_name = self._get_section_for_page(page_num)
                        
                        page_result = {
                            "page": page_num,
                            "section": section_name,
                            "char_count": len(text),
                            "text_preview": text[:300] + "..." if len(text) > 300 else text
                        }
                        
                        # Use LLM to analyze if available
                        if self.llm_processor and len(text.strip()) > 50:
                            try:
                                logger.info(f"[{self.sm_id}] Analyzing page {page_num} with LLM...")
                                
                                analysis = analyze_page(
                                    llm_processor=self.llm_processor,
                                    role=self.role,
                                    text=text,
                                    page_num=page_num,
                                    section_name=section_name,
                                    processing_requirements=self.processing_requirements
                                )
                                
                                # Merge analysis results
                                page_result.update(analysis)
                                
                                # Update statistics
                                if analysis.get("status") == "success":
                                    llm_successes += 1
                                    total_entities += len(analysis.get("entities", []))
                                    total_keywords += len(analysis.get("keywords", []))
                                else:
                                    llm_failures += 1
                                
                                logger.info(
                                    f"[{self.sm_id}] Page {page_num} analyzed: "
                                    f"{len(analysis.get('entities', []))} entities, "
                                    f"{len(analysis.get('keywords', []))} keywords"
                                )
                                
                            except Exception as e:
                                logger.error(f"[{self.sm_id}] LLM analysis failed for page {page_num}: {e}")
                                page_result["llm_error"] = str(e)
                                page_result["summary"] = "[LLM analysis failed - text extracted only]"
                                llm_failures += 1
                        else:
                            # No LLM processing - just store extracted text
                            page_result["summary"] = "[No LLM analysis - text extracted only]"
                            if not self.llm_processor:
                                logger.warning(f"[{self.sm_id}] No LLM processor available for page {page_num}")
                        
                        output.append(page_result)
                        logger.debug(f"[{self.sm_id}] Processed page {page_num}: {len(text)} chars")
                    
                except Exception as e:
                    logger.error(f"[{self.sm_id}] Error processing pages {start}-{end}: {e}")
                    # Fallback to error message
                    for page in range(start, end + 1):
                        output.append({
                            "page": page,
                            "error": str(e),
                            "summary": f"[ERROR: Could not process page {page}]"
                        })
            else:
                # Fallback to mock data if no PDF extractor
                logger.warning(f"[{self.sm_id}] No PDF extractor, using mock data for pages {start}-{end}")
                for page in range(start, end + 1):
                    time.sleep(random.uniform(0.05, 0.15))  # simulate processing
                    output.append({
                        "page": page,
                        "summary": f"[MOCK] Processed page {page}",
                        "char_count": 0
                    })

        self.status = "completed"
        
        # Generate aggregate summary if we have LLM results
        aggregate_summary = self._generate_aggregate_summary(output)
        
        logger.info(
            f"[{self.sm_id}] Completed: {len(output)} pages, "
            f"{total_chars_extracted:,} chars, "
            f"{llm_successes} LLM successes, {llm_failures} failures"
        )

        return {
            "sm_id": self.sm_id,
            "role": self.role,
            "assigned_sections": self.sections,
            "page_range": self.pages,
            "results": output,
            "total_pages": len(output),
            "total_chars": total_chars_extracted,
            "total_entities": total_entities,
            "total_keywords": total_keywords,
            "llm_successes": llm_successes,
            "llm_failures": llm_failures,
            "aggregate_summary": aggregate_summary
        }
    
    def _get_section_for_page(self, page_num: int) -> str:
        """Determine which section a page belongs to."""
        sections = self.meta.get("sections", {})
        
        for section_name, page_info in sections.items():
            start = page_info.get("page_start", 0)
            end = page_info.get("page_end", 0)
            
            if start <= page_num <= end:
                return section_name
        
        return "Unknown"
    
    def _generate_aggregate_summary(self, results: list) -> str:
        """Generate an overall summary from all page results."""
        summaries = [
            r.get("summary", "") 
            for r in results 
            if r.get("summary") and not r["summary"].startswith("[")
        ]
        
        if not summaries:
            return "No analysis results available."
        
        # Combine first few summaries
        combined = " ".join(summaries[:3])
        
        if len(combined) > 500:
            combined = combined[:500] + "..."
        
        return combined
