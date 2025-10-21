# utils/pdf_extractor.py
import os
from typing import Dict, List, Optional
from pypdf import PdfReader
from utils.logger import get_logger

logger = get_logger("PDFExtractor")


class PDFExtractor:
    """
    Utility class for extracting text from PDF files.
    Supports page-range extraction for SubMaster processing.
    """

    def __init__(self, pdf_path: str):
        """
        Initialize PDF extractor with file path.
        
        Args:
            pdf_path: Absolute path to the PDF file
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.pdf_path = pdf_path
        try:
            self.reader = PdfReader(pdf_path)
            self.num_pages = len(self.reader.pages)
            logger.info(f"Loaded PDF: {pdf_path} ({self.num_pages} pages)")
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            raise

    def extract_page(self, page_number: int) -> str:
        """
        Extract text from a single page.
        
        Args:
            page_number: Page number (1-indexed)
            
        Returns:
            Extracted text from the page
        """
        if page_number < 1 or page_number > self.num_pages:
            raise ValueError(f"Invalid page number: {page_number}. PDF has {self.num_pages} pages.")
        
        try:
            # pypdf uses 0-indexed pages
            page = self.reader.pages[page_number - 1]
            text = page.extract_text()
            
            if not text or not text.strip():
                logger.warning(f"Page {page_number} appears to be empty or contains no extractable text")
                return ""
            
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting page {page_number}: {e}")
            raise

    def extract_page_range(self, start_page: int, end_page: int) -> Dict[int, str]:
        """
        Extract text from a range of pages.
        
        Args:
            start_page: Starting page number (1-indexed, inclusive)
            end_page: Ending page number (1-indexed, inclusive)
            
        Returns:
            Dictionary mapping page numbers to extracted text
        """
        if start_page < 1 or end_page > self.num_pages:
            raise ValueError(
                f"Invalid page range: {start_page}-{end_page}. "
                f"PDF has {self.num_pages} pages."
            )
        
        if start_page > end_page:
            raise ValueError(f"Start page ({start_page}) cannot be greater than end page ({end_page})")
        
        extracted = {}
        for page_num in range(start_page, end_page + 1):
            try:
                text = self.extract_page(page_num)
                extracted[page_num] = text
                logger.debug(f"Extracted page {page_num}: {len(text)} characters")
            except Exception as e:
                logger.error(f"Failed to extract page {page_num}: {e}")
                extracted[page_num] = f"[ERROR: Could not extract page {page_num}]"
        
        logger.info(f"Extracted {len(extracted)} pages from range {start_page}-{end_page}")
        return extracted

    def extract_sections(self, sections: Dict[str, Dict[str, int]]) -> Dict[str, Dict[int, str]]:
        """
        Extract text from multiple sections defined by page ranges.
        
        Args:
            sections: Dict mapping section names to page ranges
                     e.g., {"Abstract": {"page_start": 1, "page_end": 1}}
        
        Returns:
            Dict mapping section names to page-text dictionaries
        """
        results = {}
        
        for section_name, page_info in sections.items():
            start = page_info.get("page_start")
            end = page_info.get("page_end")
            
            if start is None or end is None:
                logger.warning(f"Section '{section_name}' missing page range info, skipping")
                continue
            
            try:
                section_text = self.extract_page_range(start, end)
                results[section_name] = section_text
                logger.info(f"Extracted section '{section_name}': pages {start}-{end}")
            except Exception as e:
                logger.error(f"Failed to extract section '{section_name}': {e}")
                results[section_name] = {}
        
        return results

    def get_metadata(self) -> Dict:
        """
        Extract PDF metadata.
        
        Returns:
            Dictionary with PDF metadata
        """
        metadata = {
            "num_pages": self.num_pages,
            "file_path": self.pdf_path,
            "file_name": os.path.basename(self.pdf_path)
        }
        
        # Try to get PDF metadata if available
        if self.reader.metadata:
            try:
                metadata.update({
                    "title": self.reader.metadata.get("/Title", ""),
                    "author": self.reader.metadata.get("/Author", ""),
                    "subject": self.reader.metadata.get("/Subject", ""),
                    "creator": self.reader.metadata.get("/Creator", "")
                })
            except Exception as e:
                logger.warning(f"Could not extract PDF metadata: {e}")
        
        return metadata


def extract_text_from_pdf(pdf_path: str, start_page: int, end_page: int) -> Dict[int, str]:
    """
    Convenience function to extract text from a PDF page range.
    
    Args:
        pdf_path: Path to PDF file
        start_page: Starting page (1-indexed)
        end_page: Ending page (1-indexed)
        
    Returns:
        Dictionary mapping page numbers to text
    """
    extractor = PDFExtractor(pdf_path)
    return extractor.extract_page_range(start_page, end_page)


# Quick test when run directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf> [start_page] [end_page]")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    try:
        extractor = PDFExtractor(pdf_file)
        print(f"\n📄 PDF Metadata:")
        print(extractor.get_metadata())
        
        print(f"\n📖 Extracting pages {start}-{end}...\n")
        pages = extractor.extract_page_range(start, end)
        
        for page_num, text in pages.items():
            print(f"{'='*60}")
            print(f"PAGE {page_num}")
            print(f"{'='*60}")
            print(text[:500] + "..." if len(text) > 500 else text)
            print()
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
