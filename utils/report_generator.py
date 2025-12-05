"""
Generate PDF reports from SubMaster analysis results.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.colors import HexColor
from utils.logger import get_logger

logger = get_logger("ReportGenerator")


class PDFReportGenerator:
    """Generate structured PDF reports from document analysis results."""
    
    def __init__(self, output_path: str, metadata: Dict[str, Any]):
        """Initialize PDF report generator."""
        self.output_path = output_path
        self.metadata = metadata
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=0.75*inch,
            bottomMargin=0.75*inch
        )
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        self.story = []
        
        logger.info(f"PDF report generator initialized: {output_path}")

    # ---------------------------------------------------------------------
    # FIXED: Prevent duplicate style definition crashes
    # ---------------------------------------------------------------------
    def _safe_add_style(self, style):
        """Add a paragraph style only if not already defined."""
        if style.name in self.styles.byName:
            logger.debug(f"Style '{style.name}' already exists. Replacing safely.")
            self.styles.byName[style.name] = style
        else:
            self.styles.add(style)

    # ---------------------------------------------------------------------
    # Custom Styles
    # ---------------------------------------------------------------------
    def _create_custom_styles(self):
        
        # Title
        self._safe_add_style(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Section header
        self._safe_add_style(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection
        self._safe_add_style(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=HexColor('#34495e'),
            spaceAfter=8,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        ))
        
        # FIXED: BodyText safe override
        self._safe_add_style(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
            leading=14
        ))
        
        # List item
        self._safe_add_style(ParagraphStyle(
            name='ListItem',
            parent=self.styles['Normal'],
            fontSize=10,
            leftIndent=20,
            spaceAfter=4
        ))

    # ---------------------------------------------------------------------
    # Cover Page
    # ---------------------------------------------------------------------
    def add_cover_page(self):
        title = f"Analysis Report: {self.metadata.get('file_name', 'Document')}"
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(title, self.styles['CustomTitle']))
        self.story.append(Spacer(1, 0.5*inch))
        
        info_data = [
            ['Document Type:', self.metadata.get('document_type', 'N/A').upper()],
            ['Total Pages:', str(self.metadata.get('num_pages', 'N/A'))],
            ['File Size:', f"{self.metadata.get('file_size_mb', 0):.2f} MB"],
            ['Analysis Date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Model Used:', self.metadata.get('preferred_model', 'N/A')]
        ]
        
        info_table = Table(info_data, colWidths=[2*inch, 4*inch])
        info_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        self.story.append(info_table)
        self.story.append(PageBreak())

    # ---------------------------------------------------------------------
    # Executive Summary
    # ---------------------------------------------------------------------
    def add_executive_summary(self, results: Dict[str, Any]):
        self.story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        total_submasters = len(results)
        total_pages = 0
        total_entities = 0
        total_keywords = 0
        total_successes = 0
        total_failures = 0
        
        for r in results.values():
            if r.get('status') != 'ok':
                continue
            # Navigate nested structure: result['output']['output']
            outer = r.get('output', {})
            inner = outer.get('output', {})
            
            total_pages += inner.get('total_pages', 0)
            
            # Count entities and keywords from page results
            for page_result in inner.get('results', []):
                total_entities += len(page_result.get('entities', []))
                total_keywords += len(page_result.get('keywords', []))
                if page_result.get('status') == 'success':
                    total_successes += 1
                else:
                    total_failures += 1
        
        success_rate = (total_successes/(total_successes+total_failures)*100) if (total_successes+total_failures) else 0
        
        summary_text = f"""
        This report presents the automated analysis of the document using {total_submasters} parallel 
        processing units. A total of {total_pages} pages were analyzed, extracting 
        {total_entities} entities and {total_keywords} keywords. 
        The analysis achieved a {success_rate:.1f}% success rate.
        """
        
        self.story.append(Paragraph(summary_text, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.3*inch))
        
        stats_data = [
            ['Metric', 'Value'],
            ['Total SubMasters', str(total_submasters)],
            ['Pages Analyzed', str(total_pages)],
            ['Entities Extracted', str(total_entities)],
            ['Keywords Extracted', str(total_keywords)],
            ['LLM Successes', str(total_successes)],
            ['LLM Failures', str(total_failures)],
            ['Success Rate', f"{success_rate:.1f}%"]
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
        ]))
        
        self.story.append(stats_table)
        self.story.append(PageBreak())

    # ---------------------------------------------------------------------
    # Detailed SubMaster Results
    # ---------------------------------------------------------------------
    def add_submaster_results(self, results: Dict[str, Any]):
        self.story.append(Paragraph("Detailed Analysis by Section", self.styles['SectionHeader']))
        
        for sm_id, result in results.items():
            if result.get('status') != 'ok':
                continue
            
            # Navigate nested structure: result['output']['output']
            outer = result.get('output', {})
            inner = outer.get('output', {})
            
            self.story.append(Paragraph(f"SubMaster: {sm_id}", self.styles['SubsectionHeader']))
            
            role = inner.get('role', 'N/A')
            sections = ', '.join(inner.get('assigned_sections', []))
            page_range = inner.get('page_range', [])
            total_pages = inner.get('total_pages', 0)
            
            # Calculate totals from page results
            page_results = inner.get('results', [])
            total_chars = sum(p.get('char_count', 0) for p in page_results)
            total_entities = sum(len(p.get('entities', [])) for p in page_results)
            total_keywords = sum(len(p.get('keywords', [])) for p in page_results)
            
            info_text = f"""
            <b>Role:</b> {role}<br/>
            <b>Sections:</b> {sections}<br/>
            <b>Pages:</b> {page_range}<br/>
            <b>Total Pages:</b> {total_pages}<br/>
            <b>Characters:</b> {total_chars}<br/>
            <b>Entities:</b> {total_entities}<br/>
            <b>Keywords:</b> {total_keywords}
            """
            
            self.story.append(Paragraph(info_text, self.styles['BodyText']))
            self.story.append(Spacer(1, 0.2*inch))
            
            # Add page summaries
            for page_result in page_results[:3]:  # Show first 3 pages
                page_num = page_result.get('page', 'N/A')
                section = page_result.get('section', 'N/A')
                summary = page_result.get('summary', '')
                if summary and not summary.startswith('['):
                    self.story.append(Paragraph(f"<b>Page {page_num} ({section}):</b>", self.styles['BodyText']))
                    # Truncate long summaries
                    if len(summary) > 500:
                        summary = summary[:500] + "..."
                    self.story.append(Paragraph(summary, self.styles['BodyText']))
                    self.story.append(Spacer(1, 0.1*inch))
            
            if len(page_results) > 3:
                self.story.append(Paragraph(f"<i>... and {len(page_results) - 3} more pages</i>", self.styles['BodyText']))
            
            self.story.append(Spacer(1, 0.2*inch))

    # ---------------------------------------------------------------------
    # Appendix
    # ---------------------------------------------------------------------
    def add_appendix(self, results: Dict[str, Any]):
        self.story.append(PageBreak())
        self.story.append(Paragraph("Appendix: Complete Entity and Keyword List", self.styles['SectionHeader']))
        
        all_entities = set()
        all_keywords = set()
        
        for r in results.values():
            if r.get('status') != 'ok': 
                continue
            
            # Navigate nested structure: result['output']['output']['results']
            outer = r.get('output', {})
            inner = outer.get('output', {})
            
            for p in inner.get('results', []):
                all_entities.update(p.get('entities', []))
                all_keywords.update(p.get('keywords', []))
        
        if all_entities:
            self.story.append(Paragraph("<b>All Entities:</b>", self.styles['SubsectionHeader']))
            self.story.append(Paragraph(", ".join(sorted(all_entities)), self.styles['BodyText']))
            self.story.append(Spacer(1, 0.2*inch))
        
        if all_keywords:
            self.story.append(Paragraph("<b>All Keywords:</b>", self.styles['SubsectionHeader']))
            self.story.append(Paragraph(", ".join(sorted(all_keywords)), self.styles['BodyText']))

    # ---------------------------------------------------------------------
    # Generate PDF
    # ---------------------------------------------------------------------
    def generate(self, results: Dict[str, Any]) -> str:
        logger.info("Generating PDF report...")
        try:
            self.add_cover_page()
            self.add_executive_summary(results)
            self.add_submaster_results(results)
            self.add_appendix(results)
            
            self.doc.build(self.story)
            logger.info(f"PDF generated at: {self.output_path}")
            return self.output_path

        except Exception as e:
            logger.error(f"Failed to generate PDF: {e}")
            raise


# ---------------------------------------------------------------------
# JSON REPORT
# ---------------------------------------------------------------------

def save_results_as_json(results: Dict[str, Any], path: str) -> str:
    with open(path, "w", encoding="utf8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Saved JSON: {path}")
    return path


def generate_analysis_report(results: Dict[str, Any], metadata: Dict[str, Any], output_dir="output") -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    doc_name = os.path.splitext(metadata.get("file_name", "document"))[0]
    
    paths = {}
    
    json_path = os.path.join(output_dir, f"{doc_name}_results_{timestamp}.json")
    paths["json"] = save_results_as_json(results, json_path)

    try:
        pdf_path = os.path.join(output_dir, f"{doc_name}_analysis_{timestamp}.pdf")
        gen = PDFReportGenerator(pdf_path, metadata)
        paths["pdf"] = gen.generate(results)
    except Exception as e:
        paths["pdf_error"] = str(e)
    
    return paths
