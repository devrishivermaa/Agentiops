# utils/report_generator.py
"""
Generate PDF reports from SubMaster analysis results.
Creates professional, structured PDF documents with analysis findings.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, KeepTogether
)
from reportlab.lib.colors import HexColor
from utils.logger import get_logger

logger = get_logger("ReportGenerator")


class PDFReportGenerator:
    """Generate structured PDF reports from document analysis results."""
    
    def __init__(self, output_path: str, metadata: Dict[str, Any]):
        """
        Initialize PDF report generator.
        
        Args:
            output_path: Path where PDF will be saved
            metadata: Document metadata
        """
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
    
    def _create_custom_styles(self):
        """Create custom paragraph styles for the report."""
        # Only add styles if they don't exist
        if 'CustomTitle' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                textColor=HexColor('#1a1a1a'),
                spaceAfter=30,
                alignment=TA_CENTER,
                fontName='Helvetica-Bold'
            ))
        
        if 'SectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=16,
                textColor=HexColor('#2c3e50'),
                spaceAfter=12,
                spaceBefore=20,
                fontName='Helvetica-Bold'
            ))
        
        if 'SubsectionHeader' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='SubsectionHeader',
                parent=self.styles['Heading3'],
                fontSize=13,
                textColor=HexColor('#34495e'),
                spaceAfter=8,
                spaceBefore=12,
                fontName='Helvetica-Bold'
            ))
        
        if 'BodyText' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='BodyText',
                parent=self.styles['Normal'],
                fontSize=10,
                alignment=TA_JUSTIFY,
                spaceAfter=8,
                leading=14
            ))
        
        if 'ListItem' not in self.styles:
            self.styles.add(ParagraphStyle(
                name='ListItem',
                parent=self.styles['Normal'],
                fontSize=10,
                leftIndent=20,
                spaceAfter=4
            ))
    
    def add_cover_page(self):
        """Add cover page with document information."""
        # Title
        title = f"Analysis Report: {self.metadata.get('file_name', 'Document')}"
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(title, self.styles['CustomTitle']))
        self.story.append(Spacer(1, 0.5*inch))
        
        # Document info
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
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]))
        
        self.story.append(info_table)
        self.story.append(PageBreak())
    
    def add_executive_summary(self, results: Dict[str, Any]):
        """Add executive summary section."""
        self.story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Statistics
        total_submasters = len(results)
        total_pages = sum(r.get('output', {}).get('total_pages', 0) for r in results.values())
        total_entities = sum(r.get('output', {}).get('total_entities', 0) for r in results.values())
        total_keywords = sum(r.get('output', {}).get('total_keywords', 0) for r in results.values())
        total_successes = sum(r.get('output', {}).get('llm_successes', 0) for r in results.values())
        total_failures = sum(r.get('output', {}).get('llm_failures', 0) for r in results.values())
        
        summary_text = f"""
        This report presents the automated analysis of the document using {total_submasters} parallel 
        processing units (SubMasters). A total of {total_pages} pages were analyzed, extracting 
        {total_entities} entities and {total_keywords} keywords. The analysis achieved a 
        {(total_successes/(total_successes+total_failures)*100):.1f}% success rate with 
        {total_successes} successful analyses and {total_failures} failures.
        """
        
        self.story.append(Paragraph(summary_text, self.styles['BodyText']))
        self.story.append(Spacer(1, 0.3*inch))
        
        # Summary statistics table
        stats_data = [
            ['Metric', 'Value'],
            ['Total SubMasters', str(total_submasters)],
            ['Pages Analyzed', str(total_pages)],
            ['Entities Extracted', str(total_entities)],
            ['Keywords Extracted', str(total_keywords)],
            ['LLM Successes', str(total_successes)],
            ['LLM Failures', str(total_failures)],
            ['Success Rate', f"{(total_successes/(total_successes+total_failures)*100):.1f}%"]
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 11),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ecf0f1')),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f8f9fa')])
        ]))
        
        self.story.append(stats_table)
        self.story.append(PageBreak())
    
    def add_submaster_results(self, results: Dict[str, Any]):
        """Add detailed results from each SubMaster."""
        self.story.append(Paragraph("Detailed Analysis by Section", self.styles['SectionHeader']))
        
        for sm_id, result in results.items():
            if result.get('status') != 'ok':
                continue
            
            output = result.get('output', {})
            
            # SubMaster header
            self.story.append(Paragraph(
                f"SubMaster: {sm_id}",
                self.styles['SubsectionHeader']
            ))
            
            # Role and sections
            role = output.get('role', 'N/A')
            sections = ', '.join(output.get('assigned_sections', []))
            page_range = output.get('page_range', [])
            
            info_text = f"""
            <b>Role:</b> {role}<br/>
            <b>Sections:</b> {sections if sections else 'N/A'}<br/>
            <b>Pages:</b> {page_range}<br/>
            <b>Total Pages Processed:</b> {output.get('total_pages', 0)}<br/>
            <b>Characters Extracted:</b> {output.get('total_chars', 0):,}<br/>
            <b>Entities Found:</b> {output.get('total_entities', 0)}<br/>
            <b>Keywords Found:</b> {output.get('total_keywords', 0)}
            """
            
            self.story.append(Paragraph(info_text, self.styles['BodyText']))
            self.story.append(Spacer(1, 0.15*inch))
            
            # Aggregate summary
            agg_summary = output.get('aggregate_summary', '')
            if agg_summary and not agg_summary.startswith('No analysis'):
                self.story.append(Paragraph("<b>Summary:</b>", self.styles['BodyText']))
                self.story.append(Paragraph(agg_summary, self.styles['BodyText']))
                self.story.append(Spacer(1, 0.15*inch))
            
            # Page-by-page results (top 3 pages)
            page_results = output.get('results', [])
            if page_results:
                self.story.append(Paragraph(
                    "<b>Key Findings (Sample Pages):</b>",
                    self.styles['BodyText']
                ))
                
                for page_result in page_results[:3]:  # Show first 3 pages
                    page_num = page_result.get('page', '?')
                    
                    # Create page findings box
                    page_items = []
                    page_items.append(Paragraph(
                        f"<b>Page {page_num}</b>",
                        self.styles['ListItem']
                    ))
                    
                    if 'summary' in page_result and not page_result['summary'].startswith('['):
                        page_items.append(Paragraph(
                            f"Summary: {page_result['summary'][:200]}...",
                            self.styles['ListItem']
                        ))
                    
                    if page_result.get('entities'):
                        entities = ', '.join(page_result['entities'][:5])
                        page_items.append(Paragraph(
                            f"Entities: {entities}",
                            self.styles['ListItem']
                        ))
                    
                    if page_result.get('keywords'):
                        keywords = ', '.join(page_result['keywords'][:5])
                        page_items.append(Paragraph(
                            f"Keywords: {keywords}",
                            self.styles['ListItem']
                        ))
                    
                    # Add as keep-together unit
                    self.story.extend(page_items)
                    self.story.append(Spacer(1, 0.1*inch))
            
            self.story.append(Spacer(1, 0.2*inch))
    
    def add_appendix(self, results: Dict[str, Any]):
        """Add appendix with all extracted entities and keywords."""
        self.story.append(PageBreak())
        self.story.append(Paragraph("Appendix: Complete Entity and Keyword List", self.styles['SectionHeader']))
        
        # Collect all entities and keywords
        all_entities = set()
        all_keywords = set()
        
        for result in results.values():
            if result.get('status') != 'ok':
                continue
            
            output = result.get('output', {})
            for page_result in output.get('results', []):
                all_entities.update(page_result.get('entities', []))
                all_keywords.update(page_result.get('keywords', []))
        
        # Entities section
        if all_entities:
            self.story.append(Paragraph("<b>All Extracted Entities:</b>", self.styles['SubsectionHeader']))
            entities_text = ', '.join(sorted(all_entities))
            self.story.append(Paragraph(entities_text, self.styles['BodyText']))
            self.story.append(Spacer(1, 0.2*inch))
        
        # Keywords section
        if all_keywords:
            self.story.append(Paragraph("<b>All Extracted Keywords:</b>", self.styles['SubsectionHeader']))
            keywords_text = ', '.join(sorted(all_keywords))
            self.story.append(Paragraph(keywords_text, self.styles['BodyText']))
    
    def generate(self, results: Dict[str, Any]) -> str:
        """
        Generate the complete PDF report.
        
        Args:
            results: SubMaster analysis results
            
        Returns:
            Path to generated PDF file
        """
        logger.info("Generating PDF report...")
        
        try:
            # Build report sections
            self.add_cover_page()
            self.add_executive_summary(results)
            self.add_submaster_results(results)
            self.add_appendix(results)
            
            # Build PDF
            self.doc.build(self.story)
            
            logger.info(f"PDF report generated successfully: {self.output_path}")
            return self.output_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {e}")
            raise


def save_results_as_json(results: Dict[str, Any], output_path: str) -> str:
    """
    Save results as JSON for programmatic access.
    
    Args:
        results: SubMaster analysis results
        output_path: Path to save JSON file
        
    Returns:
        Path to saved JSON file
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved as JSON: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Failed to save JSON results: {e}")
        raise


def generate_analysis_report(
    results: Dict[str, Any],
    metadata: Dict[str, Any],
    output_dir: str = "output"
) -> Dict[str, str]:
    """
    Generate both PDF and JSON reports from analysis results.
    
    Args:
        results: SubMaster analysis results
        metadata: Document metadata
        output_dir: Directory to save reports
        
    Returns:
        Dictionary with paths to generated files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    doc_name = os.path.splitext(metadata.get('file_name', 'document'))[0]
    
    generated_files = {}
    
    # Save JSON results first (always works)
    try:
        json_path = os.path.join(output_dir, f"{doc_name}_results_{timestamp}.json")
        json_file = save_results_as_json(results, json_path)
        generated_files['json'] = json_file
        logger.info(f"JSON report saved: {json_file}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")
        raise
    
    # Generate PDF report
    try:
        pdf_path = os.path.join(output_dir, f"{doc_name}_analysis_{timestamp}.pdf")
        pdf_generator = PDFReportGenerator(pdf_path, metadata)
        pdf_file = pdf_generator.generate(results)
        generated_files['pdf'] = pdf_file
        logger.info(f"PDF report saved: {pdf_file}")
    except Exception as e:
        logger.error(f"Failed to generate PDF report: {e}")
        logger.warning("JSON report is still available")
        # Don't fail completely, just skip PDF
        generated_files['pdf_error'] = str(e)
    
    logger.info("Analysis reports generation completed")
    
    return generated_files
