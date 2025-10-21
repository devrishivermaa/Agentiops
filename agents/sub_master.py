# agents/sub_master.py
import time, random, uuid
import ray
from utils.logger import get_logger
from utils.pdf_extractor import PDFExtractor

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
        
        # Initialize PDF extractor (will be used in process())
        self.pdf_extractor = None
        
        logger.info(f"[{self.sm_id}] Init: role={self.role}, pages={self.pages}, pdf={self.pdf_path}")

    def initialize(self):
        """Initialize the PDF extractor for this SubMaster."""
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
        
        return {"sm_id": self.sm_id, "status": "ready"}

    def process(self):
        """Process assigned pages by extracting text from PDF."""
        self.status = "running"
        output = []

        # Handle multiple page ranges
        if len(self.pages) % 2 != 0:
            raise ValueError(f"Invalid page_range for {self.sm_id}: {self.pages}")

        # Iterate over each (start, end) pair
        for i in range(0, len(self.pages), 2):
            start, end = self.pages[i], self.pages[i + 1]
            
            if self.pdf_extractor:
                # Extract real text from PDF
                try:
                    logger.info(f"[{self.sm_id}] Extracting pages {start}-{end} from PDF...")
                    extracted_pages = self.pdf_extractor.extract_page_range(start, end)
                    
                    for page_num, text in extracted_pages.items():
                        output.append({
                            "page": page_num,
                            "text": text,
                            "char_count": len(text),
                            "preview": text[:200] + "..." if len(text) > 200 else text
                        })
                        logger.debug(f"[{self.sm_id}] Extracted page {page_num}: {len(text)} chars")
                    
                except Exception as e:
                    logger.error(f"[{self.sm_id}] Error extracting pages {start}-{end}: {e}")
                    # Fallback to error message
                    for page in range(start, end + 1):
                        output.append({
                            "page": page,
                            "text": f"[ERROR: Could not extract page {page}]",
                            "error": str(e)
                        })
            else:
                # Fallback to mock data if no PDF extractor
                logger.warning(f"[{self.sm_id}] No PDF extractor, using mock data for pages {start}-{end}")
                for page in range(start, end + 1):
                    time.sleep(random.uniform(0.05, 0.15))  # simulate processing
                    output.append({
                        "page": page,
                        "text": f"[MOCK] Processed text of page {page}",
                        "char_count": 0
                    })

        self.status = "completed"
        logger.info(f"[{self.sm_id}] Completed extraction of {len(output)} pages.")

        return {
            "sm_id": self.sm_id,
            "role": self.role,
            "assigned_sections": self.sections,
            "page_range": self.pages,
            "results": output,
            "total_pages": len(output),
            "total_chars": sum(r.get("char_count", 0) for r in output)
        }
