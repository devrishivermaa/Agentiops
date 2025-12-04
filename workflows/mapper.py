# workflows/mapper.py
"""
Complete Mapper workflow: Entry point for document processing pipeline.
"""

import os
import json
import time
from typing import Dict, Any
from datetime import datetime
from utils.logger import get_logger
from utils.pdf_extractor import PDFExtractor
from utils.monitor import SystemMonitor

logger = get_logger("Mapper")

class Mapper:
    """Mapper validates input and prepares metadata."""
    
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.monitor = SystemMonitor()
        logger.info(f"Mapper initialized: {output_dir}")
    
    def validate_input(self, file_path: str, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate PDF and configuration."""
        logger.info("Validating input...")
        errors = []
        warnings = []
        
        # Check file
        if not file_path:
            errors.append("No file path provided")
            return {"valid": False, "errors": errors}
        
        abs_path = os.path.abspath(file_path)
        
        if not os.path.exists(abs_path):
            errors.append(f"File not found: {abs_path}")
            return {"valid": False, "errors": errors}
        
        if not abs_path.lower().endswith('.pdf'):
            errors.append("File must be a PDF")
        
        # Validate PDF
        try:
            extractor = PDFExtractor(abs_path)
            num_pages = extractor.num_pages
            
            if num_pages == 0:
                errors.append("PDF has no pages")
            elif num_pages > 500:
                warnings.append(f"Large PDF: {num_pages} pages")
            
            file_size_mb = os.path.getsize(abs_path) / (1024 * 1024)
            if file_size_mb > 100:
                warnings.append(f"Large file: {file_size_mb:.1f}MB")
            
            logger.info(f"PDF validated: {num_pages} pages, {file_size_mb:.1f}MB")
            
        except Exception as e:
            errors.append(f"Failed to read PDF: {e}")
            return {"valid": False, "errors": errors}
        
        # Validate config
        if "document_type" not in user_config:
            errors.append("Missing 'document_type'")
        
        if "processing_requirements" not in user_config:
            errors.append("Missing 'processing_requirements'")
        elif not user_config["processing_requirements"]:
            errors.append("At least one processing requirement needed")
        
        if errors:
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        logger.info("✅ Validation passed")
        return {"valid": True, "num_pages": num_pages, "warnings": warnings}
    
    def extract_metadata(self, file_path: str, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive metadata."""
        logger.info(f"Extracting metadata: {file_path}")
        
        abs_path = os.path.abspath(file_path)
        extractor = PDFExtractor(abs_path)
        pdf_metadata = extractor.get_metadata()
        
        metadata = {
            "file_path": abs_path,
            "file_name": os.path.basename(abs_path),
            "file_size_mb": pdf_metadata.get("file_size_mb", 0),
            "num_pages": extractor.num_pages,
            "pdf_metadata": pdf_metadata,
            
            # User config
            "document_type": user_config.get("document_type", "research_paper"),
            "processing_requirements": user_config.get("processing_requirements", []),
            "user_notes": user_config.get("user_notes", ""),
            "brief_info": user_config.get("brief_info", ""),
            
            # Processing config
            "preferred_model": user_config.get("preferred_model", "mistral-small-latest"),
            "complexity_level": user_config.get("complexity_level", "medium"),
            "priority": user_config.get("priority", "medium"),
            "max_parallel_submasters": user_config.get("max_parallel_submasters", 2),
            "num_workers_per_submaster": user_config.get("num_workers_per_submaster", 3),
            
            # Advanced
            "has_ocr": user_config.get("has_ocr", False),
            "feedback_required": user_config.get("feedback_required", True),
            "output_format": user_config.get("output_format", "structured_json"),
            
            # Sections
            "sections": user_config.get("sections") or self._generate_sections(
                extractor, 
                user_config.get("document_type", "research_paper")
            ),
            
            # Timestamps
            "created_at": datetime.now().isoformat(),
            "status": "validated",
            "validated_against_pdf": True
        }
        
        logger.info(f"Metadata extracted: {metadata['num_pages']} pages, {len(metadata['sections'])} sections")
        return metadata
    
    def _generate_sections(self, extractor: PDFExtractor, doc_type: str) -> Dict[str, Dict]:
        """Auto-generate sections based on document type."""
        num_pages = extractor.num_pages
        
        if doc_type == "research_paper":
            return {
                "Abstract": {"page_start": 1, "page_end": 1},
                "Introduction": {"page_start": 2, "page_end": min(int(num_pages * 0.3), num_pages)},
                "Body": {"page_start": min(int(num_pages * 0.3) + 1, num_pages), 
                        "page_end": min(int(num_pages * 0.8), num_pages)},
                "Conclusion": {"page_start": min(int(num_pages * 0.8) + 1, num_pages), 
                              "page_end": num_pages}
            }
        else:
            return {"Full_Document": {"page_start": 1, "page_end": num_pages}}
    
    def save_metadata(self, metadata: Dict[str, Any]) -> str:
        """Save metadata to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_name = os.path.splitext(metadata["file_name"])
        file_name = f"metadata_{doc_name}_{timestamp}.json"
        path = os.path.join(self.output_dir, file_name)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Metadata saved: {path}")
        return path
    
    def execute(self, file_path: str, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute complete Mapper workflow."""
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("MAPPER WORKFLOW STARTED")
        logger.info(f"Input: {file_path}")
        logger.info("=" * 80)
        
        # Validate
        validation = self.validate_input(file_path, user_config)
        if not validation["valid"]:
            logger.error("Validation failed")
            return {"status": "failed", "stage": "validation", "errors": validation["errors"]}
        
        # Extract metadata
        try:
            metadata = self.extract_metadata(file_path, user_config)
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            return {"status": "failed", "stage": "metadata_extraction", "error": str(e)}
        
        # Save metadata
        try:
            metadata_path = self.save_metadata(metadata)
        except Exception as e:
            logger.error(f"Metadata save failed: {e}")
            return {"status": "failed", "stage": "metadata_save", "error": str(e)}
        
        elapsed = time.time() - start_time
        self.monitor.log_stats()
        
        logger.info(f"✅ Mapper completed in {elapsed:.2f}s")
        logger.info(f"📁 Metadata: {metadata_path}")
        
        return {
            "status": "success",
            "metadata_path": metadata_path,
            "num_pages": metadata["num_pages"],
            "num_sections": len(metadata["sections"]),
            "elapsed_time": elapsed,
            "warnings": validation.get("warnings", [])
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mapper.py <pdf_path>")
        sys.exit(1)
    
    config = {
    "document_type": "research_paper",
    "processing_requirements": ["summary_generation", "entity_extraction", "keyword_indexing"],
    "complexity_level": "high",
    "priority": "high",
    "preferred_model": "mistral-small-latest"  # Updated
}
    
    mapper = Mapper()
    result = mapper.execute(sys.argv[1], config)
    print(json.dumps(result, indent=2))
