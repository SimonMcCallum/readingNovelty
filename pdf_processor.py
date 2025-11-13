"""
PDF Processing Module

Handles PDF text extraction, chunking, and annotation.
"""

import re
import logging
from typing import List, Dict
from PyPDF2 import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.lib.colors import Color, red, orange, yellow, green
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import io

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF text extraction and annotation."""
    
    def __init__(self, chunk_size_words=(100, 200)):
        """
        Initialize PDF processor.
        
        Args:
            chunk_size_words: Tuple of (min_words, max_words) for chunking
        """
        self.min_chunk_words = chunk_size_words[0]
        self.max_chunk_words = chunk_size_words[1]
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as a string
        """
        try:
            reader = PdfReader(pdf_path)
            text = ""
            
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def chunk_text(self, text: str) -> List[Dict]:
        """
        Split text into chunks of approximately 100-200 words.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of dictionaries with chunk information
        """
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            words = para.split()
            word_count = len(words)
            
            # If adding this paragraph keeps us under max, add it
            if current_word_count + word_count <= self.max_chunk_words:
                current_chunk.append(para)
                current_word_count += word_count
            else:
                # If we have accumulated enough words, save current chunk
                if current_word_count >= self.min_chunk_words:
                    chunks.append({
                        'text': ' '.join(current_chunk),
                        'word_count': current_word_count
                    })
                    current_chunk = [para]
                    current_word_count = word_count
                else:
                    # If current chunk is too small, add this paragraph anyway
                    current_chunk.append(para)
                    current_word_count += word_count
                    
                    # If now too large, save and reset
                    if current_word_count >= self.min_chunk_words:
                        chunks.append({
                            'text': ' '.join(current_chunk),
                            'word_count': current_word_count
                        })
                        current_chunk = []
                        current_word_count = 0
        
        # Add final chunk if exists
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'word_count': current_word_count
            })
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    
    def extract_and_chunk_text(self, pdf_path: str) -> List[Dict]:
        """
        Extract text from PDF and chunk it.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text chunks
        """
        text = self.extract_text_from_pdf(pdf_path)
        return self.chunk_text(text)
    
    def get_color_for_score(self, score: float) -> Color:
        """
        Get color based on novelty score.
        
        Args:
            score: Novelty score (0-1, where higher = more novel)
            
        Returns:
            ReportLab Color object
        """
        # High novelty (>0.7) = green
        # Medium novelty (0.4-0.7) = yellow
        # Low novelty (0.2-0.4) = orange
        # Very low novelty (<0.2) = red
        
        if score >= 0.7:
            return Color(0, 0.8, 0, alpha=0.3)  # Green
        elif score >= 0.4:
            return Color(1, 1, 0, alpha=0.3)  # Yellow
        elif score >= 0.2:
            return Color(1, 0.6, 0, alpha=0.3)  # Orange
        else:
            return Color(1, 0, 0, alpha=0.3)  # Red
    
    def create_annotated_pdf(self, original_pdf_path: str, chunks: List[Dict], 
                           novelty_scores: List[float], output_path: str):
        """
        Create an annotated PDF with novelty scores.
        
        Args:
            original_pdf_path: Path to original PDF
            chunks: List of text chunks
            novelty_scores: List of novelty scores for each chunk
            output_path: Path for output annotated PDF
        """
        try:
            # Read original PDF
            reader = PdfReader(original_pdf_path)
            writer = PdfWriter()
            
            # Create a summary page
            packet = io.BytesIO()
            can = canvas.Canvas(packet, pagesize=letter)
            width, height = letter
            
            # Title
            can.setFont("Helvetica-Bold", 16)
            can.drawString(50, height - 50, "PDF Novelty Analysis Report")
            
            # Legend
            can.setFont("Helvetica-Bold", 12)
            can.drawString(50, height - 100, "Novelty Legend:")
            
            can.setFont("Helvetica", 10)
            y_pos = height - 120
            
            legend_items = [
                ("High Novelty (>0.7)", Color(0, 0.8, 0, alpha=0.5)),
                ("Medium Novelty (0.4-0.7)", Color(1, 1, 0, alpha=0.5)),
                ("Low Novelty (0.2-0.4)", Color(1, 0.6, 0, alpha=0.5)),
                ("Very Low Novelty (<0.2)", Color(1, 0, 0, alpha=0.5))
            ]
            
            for label, color in legend_items:
                can.setFillColor(color)
                can.rect(60, y_pos - 5, 20, 10, fill=1)
                can.setFillColorRGB(0, 0, 0)
                can.drawString(90, y_pos, label)
                y_pos -= 20
            
            # Summary statistics
            can.setFont("Helvetica-Bold", 12)
            can.drawString(50, y_pos - 20, "Summary Statistics:")
            
            can.setFont("Helvetica", 10)
            y_pos -= 40
            
            avg_novelty = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0
            high_novelty_chunks = sum(1 for s in novelty_scores if s >= 0.7)
            low_novelty_chunks = sum(1 for s in novelty_scores if s < 0.2)
            
            can.drawString(60, y_pos, f"Total chunks analyzed: {len(chunks)}")
            y_pos -= 20
            can.drawString(60, y_pos, f"Average novelty score: {avg_novelty:.2f}")
            y_pos -= 20
            can.drawString(60, y_pos, f"High novelty chunks: {high_novelty_chunks}")
            y_pos -= 20
            can.drawString(60, y_pos, f"Low novelty chunks: {low_novelty_chunks}")
            y_pos -= 40
            
            # Chunk details
            can.setFont("Helvetica-Bold", 12)
            can.drawString(50, y_pos, "Chunk Details:")
            y_pos -= 20
            
            can.setFont("Helvetica", 9)
            for i, (chunk, score) in enumerate(zip(chunks[:15], novelty_scores[:15])):  # Show first 15
                if y_pos < 100:
                    can.showPage()
                    y_pos = height - 50
                    can.setFont("Helvetica", 9)
                
                preview = chunk['text'][:60] + "..." if len(chunk['text']) > 60 else chunk['text']
                color = self.get_color_for_score(score)
                can.setFillColor(color)
                can.rect(55, y_pos - 5, 10, 10, fill=1)
                can.setFillColorRGB(0, 0, 0)
                can.drawString(70, y_pos, f"Chunk {i+1} (score: {score:.2f}): {preview}")
                y_pos -= 15
            
            if len(chunks) > 15:
                y_pos -= 10
                can.drawString(60, y_pos, f"... and {len(chunks) - 15} more chunks")
            
            can.save()
            
            # Add summary page
            packet.seek(0)
            from PyPDF2 import PdfReader as PdfReaderForSummary
            summary_pdf = PdfReaderForSummary(packet)
            
            # Add summary pages first
            for page in summary_pdf.pages:
                writer.add_page(page)
            
            # Add original pages
            for page in reader.pages:
                writer.add_page(page)
            
            # Write output
            with open(output_path, 'wb') as output_file:
                writer.write(output_file)
            
            logger.info(f"Created annotated PDF: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating annotated PDF: {str(e)}")
            raise
