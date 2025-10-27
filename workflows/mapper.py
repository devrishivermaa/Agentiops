# workflows/mapper.py
"""
Mapper workflow: Validates user input and prepares metadata for processing.
Serves as the entry point for the document processing pipeline.
"""

import os
import json
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from utils.logger import get_logger
from utils.pdf_extractor import PDFExtractor
from utils.monitor import SystemMonitor

logger = get_logger("Mapper")


class Mapper:
    """
    Mapper validates and prepares document metadata before processing.
    Acts as the entry point for the AgentOps pipeline.
    """
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize Mapper.
        
        Args:
            output_dir: Directory to store metadata and plans
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.monitor = SystemMonitor()
        logger.info(f"Mapper initialized with output dir: {output_dir}")
    
    def validate_input(
        self, 
        file_path: str, 
        user_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate user input and PDF file.
        
        Args:
            file_path: Path to PDF file
            user_config: User-provided configuration
            
        Returns:
            Validation result with status and errors
        """
        logger.info("Validating input...")
        errors = []
        warnings = []
        
        # Check file exists
        if not file_path:
            errors.append("No file path provided")
            return {"valid": False, "errors": errors}
        
        abs_path = os.path.abspath(file_path)
        
        if not os.path.exists(abs_path):
            errors.append(f"PDF file not found: {abs_path}")
            return {"valid": False, "errors": errors}
        
        # Check file extension
        if not abs_path.lower().endswith('.pdf'):
            errors.append(f"File must be a PDF, got: {os.path.splitext(abs_path)[1]}")
        
        # Check file size
        file_size_mb = os.path.getsize(abs_path) / (1024 * 1024)
        if file_size_mb > 100:
            warnings.append(f"Large PDF detected: {file_size_mb:.1f}MB - processing may be slow")
        
        # Validate PDF readability
        try:
            extractor = PDFExtractor(abs_path)
            num_pages = extractor.num_pages
            
            if num_pages == 0:
                errors.append("PDF has no pages")
            elif num_pages > 500:
                warnings.append(f"Large PDF: {num_pages} pages - consider splitting")
            
            logger.info(f"PDF validated: {num_pages} pages, {file_size_mb:.1f}MB")
            
        except Exception as e:
            errors.append(f"Failed to read PDF: {str(e)}")
            return {"valid": False, "errors": errors}
        
        # Validate required config fields
        required_fields = {
            "document_type": str,
            "processing_requirements": list
        }
        
        for field, expected_type in required_fields.items():
            if field not in user_config:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(user_config[field], expected_type):
                errors.append(
                    f"Field '{field}' must be {expected_type.__name__}, "
                    f"got {type(user_config[field]).__name__}"
                )
        
        # Validate processing requirements
        valid_requirements = [
            "summary_generation",
            "entity_extraction",
            "keyword_indexing",
            "question_answering"
        ]
        
        user_reqs = user_config.get("processing_requirements", [])
        if not user_reqs:
            errors.append("At least one processing requirement must be specified")
        else:
            invalid_reqs = [r for r in user_reqs if r not in valid_requirements]
            if invalid_reqs:
                errors.append(
                    f"Invalid processing requirements: {invalid_reqs}. "
                    f"Valid options: {valid_requirements}"
                )
        
        # Validate document type
        valid_doc_types = ["research_paper", "book", "report", "article", "manual", "thesis"]
        doc_type = user_config.get("document_type", "")
        if doc_type and doc_type not in valid_doc_types:
            warnings.append(
                f"Document type '{doc_type}' not in standard types: {valid_doc_types}"
            )
        
        # Validate sections if provided
        if "sections" in user_config:
            section_errors = self._validate_sections(user_config["sections"], num_pages)
            errors.extend(section_errors)
        
        # Log warnings
        for warning in warnings:
            logger.warning(warning)
        
        if errors:
            logger.error(f"Validation failed with {len(errors)} errors")
            return {"valid": False, "errors": errors, "warnings": warnings}
        
        logger.info("✅ Input validation passed")
        return {"valid": True, "num_pages": num_pages, "warnings": warnings}
    
    def _validate_sections(
        self, 
        sections: Dict[str, Dict[str, int]], 
        num_pages: int
    ) -> List[str]:
        """
        Validate section definitions.
        
        Args:
            sections: Section definitions
            num_pages: Total pages in PDF
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        
        for section_name, section_info in sections.items():
            if not isinstance(section_info, dict):
                errors.append(f"Section '{section_name}' must be a dictionary")
                continue
            
            start = section_info.get("page_start")
            end = section_info.get("page_end")
            
            if start is None:
                errors.append(f"Section '{section_name}' missing 'page_start'")
                continue
            
            if end is None:
                errors.append(f"Section '{section_name}' missing 'page_end'")
                continue
            
            # Validate page numbers
            if not isinstance(start, int) or not isinstance(end, int):
                errors.append(
                    f"Section '{section_name}' page numbers must be integers"
                )
                continue
            
            if start < 1:
                errors.append(
                    f"Section '{section_name}' start page {start} < 1"
                )
            
            if end > num_pages:
                errors.append(
                    f"Section '{section_name}' end page {end} > {num_pages}"
                )
            
            if start > end:
                errors.append(
                    f"Section '{section_name}' start page {start} > end page {end}"
                )
        
        return errors
    
    def extract_metadata(
        self, 
        file_path: str, 
        user_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract comprehensive metadata from PDF and user config.
        
        Args:
            file_path: Path to PDF file
            user_config: User configuration
            
        Returns:
            Complete metadata dictionary
        """
        logger.info(f"Extracting metadata from: {file_path}")
        
        # Initialize PDF extractor
        abs_path = os.path.abspath(file_path)
        extractor = PDFExtractor(abs_path)
        pdf_metadata = extractor.get_metadata()
        
        # Build comprehensive metadata
        metadata = {
            # File information
            "file_path": abs_path,
            "file_name": os.path.basename(abs_path),
            "file_size_mb": pdf_metadata.get("file_size_mb", 0),
            
            # PDF information
            "num_pages": extractor.num_pages,
            "pdf_metadata": pdf_metadata,
            
            # User configuration
            "document_type": user_config.get("document_type", "research_paper"),
            "processing_requirements": user_config.get("processing_requirements", []),
            "user_notes": user_config.get("user_notes", ""),
            "brief_info": user_config.get("brief_info", ""),
            
            # Processing configuration
            "preferred_model": user_config.get(
                "preferred_model", 
                "gemini-2.0-flash-exp"
            ),
            "complexity_level": user_config.get("complexity_level", "medium"),
            "priority": user_config.get("priority", "medium"),
            "max_parallel_submasters": user_config.get("max_parallel_submasters", 4),
            
            # Advanced options
            "has_ocr": user_config.get("has_ocr", False),
            "feedback_required": user_config.get("feedback_required", True),
            "output_format": user_config.get("output_format", "structured_json"),
            "expected_completion_time": user_config.get("expected_completion_time", "short"),
            "estimated_cost_per_agent": user_config.get("estimated_cost_per_agent", 0.02),
            "max_context_per_agent": user_config.get("max_context_per_agent", 20000),
            
            # Sections (user-defined or auto-detected)
            "sections": self._extract_sections(extractor, user_config),
            
            # Timestamps
            "created_at": datetime.now().isoformat(),
            "status": "validated",
            "validated_against_pdf": True
        }
        
        logger.info(
            f"Metadata extracted: {metadata['num_pages']} pages, "
            f"{len(metadata['sections'])} sections"
        )
        
        return metadata
    
    def _extract_sections(
        self, 
        extractor: PDFExtractor, 
        user_config: Dict[str, Any]
    ) -> Dict[str, Dict[str, int]]:
        """
        Extract or infer document sections.
        
        Args:
            extractor: PDFExtractor instance
            user_config: User configuration
            
        Returns:
            Dictionary of sections with page ranges
        """
        # Use user-provided sections if available
        if "sections" in user_config and user_config["sections"]:
            logger.info("Using user-provided sections")
            return user_config["sections"]
        
        # Otherwise, create default sections based on document type
        doc_type = user_config.get("document_type", "research_paper")
        num_pages = extractor.num_pages
        
        logger.info(f"Generating default sections for {doc_type}")
        
        if doc_type == "research_paper":
            # Standard research paper structure
            sections = {
                "Abstract": {
                    "page_start": 1,
                    "page_end": 1,
                    "description": "Paper abstract and overview"
                },
                "Introduction": {
                    "page_start": 2,
                    "page_end": min(3, num_pages),
                    "description": "Introduction and background"
                },
                "Methodology": {
                    "page_start": max(4, min(4, num_pages)),
                    "page_end": min(int(num_pages * 0.5), num_pages),
                    "description": "Methods and approach"
                },
                "Results": {
                    "page_start": min(int(num_pages * 0.5) + 1, num_pages),
                    "page_end": min(int(num_pages * 0.75), num_pages),
                    "description": "Experimental results"
                },
                "Discussion": {
                    "page_start": min(int(num_pages * 0.75) + 1, num_pages),
                    "page_end": min(int(num_pages * 0.9), num_pages),
                    "description": "Discussion and analysis"
                },
                "Conclusion": {
                    "page_start": min(int(num_pages * 0.9) + 1, num_pages),
                    "page_end": num_pages,
                    "description": "Conclusions and future work"
                }
            }
            
            # Remove sections with invalid ranges
            sections = {
                name: info for name, info in sections.items()
                if info["page_start"] <= info["page_end"]
            }
            
        elif doc_type == "book":
            # Simple chapter-based structure
            chapter_size = max(num_pages // 10, 5)
            sections = {}
            
            for i in range((num_pages // chapter_size) + 1):
                start = i * chapter_size + 1
                end = min((i + 1) * chapter_size, num_pages)
                
                if start <= num_pages:
                    sections[f"Chapter_{i+1}"] = {
                        "page_start": start,
                        "page_end": end,
                        "description": f"Chapter {i+1}"
                    }
        else:
            # Generic full-document section
            sections = {
                "Full_Document": {
                    "page_start": 1,
                    "page_end": num_pages,
                    "description": "Complete document"
                }
            }
        
        logger.info(f"Generated {len(sections)} default sections")
        return sections
    
    def save_metadata(self, metadata: Dict[str, Any]) -> str:
        """
        Save metadata to JSON file.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Path to saved metadata file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_name = os.path.splitext(metadata.get("file_name", "document"))[0]
        file_name = f"metadata_{doc_name}_{timestamp}.json"
        metadata_path = os.path.join(self.output_dir, file_name)
        
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Metadata saved to: {metadata_path}")
            return metadata_path
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def execute(
        self, 
        file_path: str, 
        user_config: Dict[str, Any],
        run_master_agent: bool = False
    ) -> Dict[str, Any]:
        """
        Execute complete Mapper workflow.
        
        Args:
            file_path: Path to PDF file
            user_config: User configuration
            run_master_agent: Whether to automatically invoke MasterAgent
            
        Returns:
            Result dictionary with metadata and plan paths
        """
        start_time = time.time()
        
        logger.info("=" * 80)
        logger.info("MAPPER WORKFLOW STARTED")
        logger.info("=" * 80)
        logger.info(f"Input File: {file_path}")
        logger.info(f"Output Dir: {self.output_dir}")
        logger.info("=" * 80)
        
        # Step 1: Validate input
        logger.info("\n[STEP 1/3] Validating input...")
        validation = self.validate_input(file_path, user_config)
        
        if not validation["valid"]:
            logger.error("❌ Validation failed:")
            for error in validation["errors"]:
                logger.error(f"  - {error}")
            
            return {
                "status": "failed",
                "stage": "validation",
                "errors": validation["errors"],
                "warnings": validation.get("warnings", [])
            }
        
        # Log warnings if any
        for warning in validation.get("warnings", []):
            print(f"⚠️  {warning}")
        
        logger.info("✅ Input validation passed")
        
        # Step 2: Extract metadata
        logger.info("\n[STEP 2/3] Extracting metadata...")
        try:
            metadata = self.extract_metadata(file_path, user_config)
        except Exception as e:
            logger.error(f"Failed to extract metadata: {e}")
            return {
                "status": "failed",
                "stage": "metadata_extraction",
                "error": str(e)
            }
        
        # Step 3: Save metadata
        logger.info("\n[STEP 3/3] Saving metadata...")
        try:
            metadata_path = self.save_metadata(metadata)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return {
                "status": "failed",
                "stage": "metadata_save",
                "error": str(e)
            }
        
        elapsed = time.time() - start_time
        self.monitor.log_stats()
        
        logger.info(f"\n✅ Mapper workflow completed in {elapsed:.2f}s")
        logger.info(f"📁 Metadata: {metadata_path}")
        logger.info("=" * 80)
        
        result = {
            "status": "success",
            "stage": "mapper_complete",
            "metadata_path": metadata_path,
            "num_pages": metadata["num_pages"],
            "num_sections": len(metadata["sections"]),
            "elapsed_time": elapsed,
            "warnings": validation.get("warnings", [])
        }
        
        # Step 4: Optionally run MasterAgent
        if run_master_agent:
            logger.info("\n[OPTIONAL] Invoking Master Agent...")
            try:
                from agents.master_agent import MasterAgent
                
                master = MasterAgent()
                plan = master.execute(
                    metadata_path=metadata_path,
                    save_path=os.path.join(self.output_dir, "submasters_plan.json")
                )
                
                if plan and plan.get("status") == "approved":
                    result["plan_status"] = "approved"
                    result["plan_path"] = os.path.join(
                        self.output_dir, 
                        "submasters_plan.json"
                    )
                    result["num_submasters"] = plan.get("num_submasters", 0)
                    logger.info("✅ Master Agent plan approved")
                else:
                    result["plan_status"] = "needs_revision"
                    logger.warning("⚠️ Master Agent plan needs revision")
                    
            except Exception as e:
                logger.error(f"Master Agent failed: {e}")
                result["plan_status"] = "error"
                result["plan_error"] = str(e)
        
        return result


# CLI entry point
if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="AgentOps Mapper - Document metadata extraction and validation"
    )
    parser.add_argument(
        "pdf_path",
        help="Path to PDF file to process"
    )
    parser.add_argument(
        "--config",
        help="Path to configuration JSON file",
        default=None
    )
    parser.add_argument(
        "--output",
        help="Output directory for metadata",
        default="./output"
    )
    parser.add_argument(
        "--run-master",
        help="Automatically run Master Agent after validation",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    # Load config from file or use defaults
    if args.config:
        try:
            with open(args.config, "r") as f:
                config = json.load(f)
            print(f"✅ Loaded config from: {args.config}")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            sys.exit(1)
    else:
        # Default configuration
        config = {
            "document_type": "research_paper",
            "processing_requirements": [
                "summary_generation",
                "entity_extraction",
                "keyword_indexing"
            ],
            "user_notes": "Process the document and extract key information",
            "complexity_level": "medium",
            "priority": "medium",
            "preferred_model": "gemini-2.0-flash-exp"
        }
        print("ℹ️  Using default configuration")
    
    # Run mapper
    mapper = Mapper(output_dir=args.output)
    
    try:
        result = mapper.execute(
            file_path=args.pdf_path,
            user_config=config,
            run_master_agent=args.run_master
        )
        
        print("\n" + "=" * 80)
        print("MAPPER RESULT:")
        print("=" * 80)
        print(json.dumps(result, indent=2))
        print("=" * 80)
        
        if result["status"] == "success":
            print(f"\n✅ Success! Metadata saved to: {result['metadata_path']}")
            sys.exit(0)
        else:
            print(f"\n❌ Failed at stage: {result.get('stage')}")
            if "errors" in result:
                for error in result["errors"]:
                    print(f"   - {error}")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
