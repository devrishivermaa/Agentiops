# utils/pdf_extractor.py
"""
Enhanced PDF extraction utility with robust error handling.
Supports text extraction, metadata retrieval, and section processing.
"""

import os
from typing import Dict, List, Optional, Tuple
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
            pdf_path: Absolute or relative path to the PDF file
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If file is not a valid PDF
        """
        # Convert to absolute path
        self.pdf_path = os.path.abspath(pdf_path)
        
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        
        if not self.pdf_path.lower().endswith('.pdf'):
            raise ValueError(f"File must be a PDF: {self.pdf_path}")
        
        try:
            self.reader = PdfReader(self.pdf_path)
            self.num_pages = len(self.reader.pages)
            
            # Check if PDF is encrypted
            if self.reader.is_encrypted:
                logger.warning(f"PDF is encrypted: {self.pdf_path}")
                # Try to decrypt with empty password
                if not self.reader.decrypt(""):
                    raise ValueError("PDF is encrypted and requires a password")
            
            logger.info(f"Loaded PDF: {os.path.basename(self.pdf_path)} ({self.num_pages} pages)")
            
        except Exception as e:
            logger.error(f"Failed to load PDF: {e}")
            raise ValueError(f"Invalid or corrupted PDF file: {e}")

    def extract_page(self, page_number: int) -> str:
        """
        Extract text from a single page with encoding safety.
        
        Args:
            page_number: Page number (1-indexed)
            
        Returns:
            Extracted text from the page (cleaned)
            
        Raises:
            ValueError: If page number is invalid
        """
        if page_number < 1 or page_number > self.num_pages:
            raise ValueError(
                f"Invalid page number: {page_number}. "
                f"PDF has {self.num_pages} pages (1-indexed)."
            )
        
        try:
            # pypdf uses 0-indexed pages
            page = self.reader.pages[page_number - 1]
            text = page.extract_text()
            
            if not text or not text.strip():
                logger.warning(
                    f"Page {page_number} appears to be empty or contains no extractable text"
                )
                return ""
            
            # Clean text: handle encoding issues
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
            text = text.replace('\x00', '')  # Remove null bytes
            text = text.strip()
            
            logger.debug(f"Extracted page {page_number}: {len(text)} characters")
            return text
            
        except Exception as e:
            logger.error(f"Error extracting page {page_number}: {e}")
            return f"[ERROR: Could not extract page {page_number}: {str(e)}]"

    def extract_page_range(self, start_page: int, end_page: int) -> Dict[int, str]:
        """
        Extract text from a range of pages.
        
        Args:
            start_page: Starting page number (1-indexed, inclusive)
            end_page: Ending page number (1-indexed, inclusive)
            
        Returns:
            Dictionary mapping page numbers to extracted text
            
        Raises:
            ValueError: If page range is invalid
        """
        # Validate page range
        if start_page < 1:
            raise ValueError(f"Start page must be >= 1, got {start_page}")
        
        if end_page > self.num_pages:
            raise ValueError(
                f"End page {end_page} exceeds PDF length ({self.num_pages} pages)"
            )
        
        if start_page > end_page:
            raise ValueError(
                f"Start page ({start_page}) cannot be greater than end page ({end_page})"
            )
        
        extracted = {}
        failed_pages = []
        
        for page_num in range(start_page, end_page + 1):
            try:
                text = self.extract_page(page_num)
                extracted[page_num] = text
            except Exception as e:
                logger.error(f"Failed to extract page {page_num}: {e}")
                extracted[page_num] = f"[ERROR: Could not extract page {page_num}]"
                failed_pages.append(page_num)
        
        if failed_pages:
            logger.warning(
                f"Failed to extract {len(failed_pages)} pages: {failed_pages}"
            )
        
        logger.info(
            f"Extracted {len(extracted) - len(failed_pages)}/{len(extracted)} pages "
            f"from range {start_page}-{end_page}"
        )
        
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
                logger.warning(
                    f"Section '{section_name}' missing page range info, skipping"
                )
                continue
            
            # Validate section range
            if start < 1 or end > self.num_pages:
                logger.error(
                    f"Section '{section_name}' has invalid page range: {start}-{end}. "
                    f"Valid range: 1-{self.num_pages}"
                )
                continue
            
            try:
                section_text = self.extract_page_range(start, end)
                results[section_name] = section_text
                logger.info(
                    f"Extracted section '{section_name}': "
                    f"pages {start}-{end} ({len(section_text)} pages)"
                )
            except Exception as e:
                logger.error(f"Failed to extract section '{section_name}': {e}")
                results[section_name] = {}
        
        return results

    def get_metadata(self) -> Dict:
        """
        Extract PDF metadata safely.
        
        Returns:
            Dictionary with PDF metadata
        """
        metadata = {
            "num_pages": self.num_pages,
            "file_path": self.pdf_path,
            "file_name": os.path.basename(self.pdf_path),
            "file_size_mb": round(os.path.getsize(self.pdf_path) / (1024 * 1024), 2)
        }
        
        # Try to get PDF metadata if available
        if self.reader.metadata:
            try:
                pdf_meta = self.reader.metadata
                
                # Safely extract metadata fields
                for key, attr in [
                    ('title', '/Title'),
                    ('author', '/Author'),
                    ('subject', '/Subject'),
                    ('creator', '/Creator'),
                    ('producer', '/Producer'),
                    ('creation_date', '/CreationDate')
                ]:
                    value = pdf_meta.get(attr)
                    if value:
                        # Clean up value
                        if isinstance(value, bytes):
                            value = value.decode('utf-8', errors='ignore')
                        metadata[key] = str(value).strip()
                    else:
                        metadata[key] = ""
                        
            except Exception as e:
                logger.warning(f"Could not extract PDF metadata: {e}")
        
        return metadata

    def validate_page_ranges(self, page_ranges: List[Tuple[int, int]]) -> bool:
        """
        Validate multiple page ranges.
        
        Args:
            page_ranges: List of (start, end) tuples
            
        Returns:
            True if all ranges are valid
        """
        for start, end in page_ranges:
            if start < 1 or end > self.num_pages or start > end:
                logger.error(f"Invalid page range: {start}-{end}")
                return False
        return True


def extract_text_from_pdf(
    pdf_path: str, 
    start_page: int, 
    end_page: int
) -> Dict[int, str]:
    """
    Convenience function to extract text from a PDF page range.
    
    Args:
        pdf_path: Path to PDF file
        start_page: Starting page (1-indexed)
        end_page: Ending page (1-indexed)
        
    Returns:
        Dictionary mapping page numbers to text
        
    Raises:
        FileNotFoundError: If PDF doesn't exist
        ValueError: If page range is invalid
    """
    extractor = PDFExtractor(pdf_path)
    return extractor.extract_page_range(start_page, end_page)


# Quick test when run directly
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf> [start_page] [end_page]")
        print("\nExample: python pdf_extractor.py document.pdf 1 3")
        sys.exit(1)
    
    pdf_file = sys.argv[1]
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    end = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    try:
        print(f"\n{'='*60}")
        print("PDF EXTRACTOR TEST")
        print(f"{'='*60}\n")
        
        extractor = PDFExtractor(pdf_file)
        
        print("📄 PDF Metadata:")
        metadata = extractor.get_metadata()
        for key, value in metadata.items():
            print(f"  {key}: {value}")
        
        print(f"\n📖 Extracting pages {start}-{end}...\n")
        pages = extractor.extract_page_range(start, end)
        
        for page_num, text in pages.items():
            print(f"{'='*60}")
            print(f"PAGE {page_num} ({len(text)} characters)")
            print(f"{'='*60}")
            preview = text[:500] + "..." if len(text) > 500 else text
            print(preview)
            print()
        
        print(f"\n✅ Successfully extracted {len(pages)} pages")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
