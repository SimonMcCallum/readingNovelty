"""
PDF Processing Module

Handles PDF text extraction, chunking, and annotation.

Annotation uses PyMuPDF to add real inline highlights to the original PDF
(coloured by per-chunk novelty score) plus a prepended summary page. The
chunk-to-page mapping is best-effort: for each chunk, the first ~8 words
are searched on every page; matches get a highlight. Chunks whose opening
words are not found in any page are skipped silently (and still appear in
the summary).
"""

import re
import logging
from typing import List, Dict, Tuple

import fitz  # PyMuPDF

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
            doc = fitz.open(pdf_path)
            try:
                pages = [page.get_text() for page in doc]
            finally:
                doc.close()
            return "\n\n".join(p for p in pages if p).strip()
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
    
    @staticmethod
    def get_color_for_score(score: float) -> Tuple[float, float, float]:
        """RGB triple (0-1) for a novelty score. Used by PyMuPDF highlight annots."""
        if score >= 0.7:
            return (0.0, 0.8, 0.0)   # Green: high novelty
        if score >= 0.4:
            return (1.0, 1.0, 0.0)   # Yellow: medium
        if score >= 0.2:
            return (1.0, 0.6, 0.0)   # Orange: low
        return (1.0, 0.0, 0.0)       # Red: very low

    @staticmethod
    def _chunk_search_key(text: str, word_count: int = 8) -> str:
        """First N words of a chunk, normalised for PyMuPDF search."""
        words = text.split()[:word_count]
        return ' '.join(words).strip()

    def _highlight_chunk_in_doc(self, doc, chunk_text: str, score: float) -> int:
        """
        Highlight the first occurrence of chunk's opening words on each page
        where they appear. Returns the number of pages where a highlight was
        added (0 if the chunk's opening was not found anywhere).
        """
        search_str = self._chunk_search_key(chunk_text)
        if len(search_str) < 8:
            return 0
        color = self.get_color_for_score(score)
        hits = 0
        for page in doc:
            matches = page.search_for(search_str, quads=False)
            if not matches:
                continue
            annot = page.add_highlight_annot(matches[0])
            annot.set_colors(stroke=color)
            annot.set_opacity(0.35)
            annot.update()
            hits += 1
            # First occurrence per chunk is enough; novelty is per-chunk, not per-page.
            break
        return hits

    def _build_summary_page(self, doc, chunks: List[Dict], novelty_scores: List[float]):
        """Prepend a single summary page to `doc`."""
        # Standard A4-ish portrait. PyMuPDF coordinate origin is top-left.
        page = doc.new_page(pno=0, width=595, height=842)
        margin = 50
        y = margin

        page.insert_text((margin, y), "PDF Novelty Analysis Report", fontsize=16, fontname="helv")
        y += 30
        page.insert_text((margin, y), "Novelty legend:", fontsize=12, fontname="helv")
        y += 18

        legend = [
            ("High novelty (>= 0.7)", (0.0, 0.8, 0.0)),
            ("Medium novelty (0.4-0.7)", (1.0, 1.0, 0.0)),
            ("Low novelty (0.2-0.4)", (1.0, 0.6, 0.0)),
            ("Very low novelty (< 0.2)", (1.0, 0.0, 0.0)),
        ]
        for label, rgb in legend:
            swatch = fitz.Rect(margin, y, margin + 16, y + 10)
            page.draw_rect(swatch, color=rgb, fill=rgb)
            page.insert_text((margin + 24, y + 9), label, fontsize=10, fontname="helv")
            y += 16

        y += 12
        page.insert_text((margin, y), "Summary statistics:", fontsize=12, fontname="helv")
        y += 18
        avg = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0
        high = sum(1 for s in novelty_scores if s >= 0.7)
        low = sum(1 for s in novelty_scores if s < 0.2)
        for line in [
            f"Total chunks analysed: {len(chunks)}",
            f"Average novelty score: {avg:.2f}",
            f"High novelty chunks: {high}",
            f"Low novelty chunks: {low}",
        ]:
            page.insert_text((margin, y), line, fontsize=10, fontname="helv")
            y += 14

        y += 12
        page.insert_text((margin, y), "Top chunks (first 15):", fontsize=12, fontname="helv")
        y += 18
        for i, (chunk, score) in enumerate(zip(chunks[:15], novelty_scores[:15])):
            preview = chunk['text'][:60].replace('\n', ' ')
            if len(chunk['text']) > 60:
                preview += '...'
            rgb = self.get_color_for_score(score)
            page.draw_rect(fitz.Rect(margin, y, margin + 8, y + 8), color=rgb, fill=rgb)
            page.insert_text((margin + 14, y + 7), f"Chunk {i+1} ({score:.2f}): {preview}",
                             fontsize=9, fontname="helv")
            y += 12
            if y > 800:
                break
        if len(chunks) > 15:
            page.insert_text((margin, y + 4), f"... and {len(chunks) - 15} more chunks",
                             fontsize=9, fontname="helv")

    def create_annotated_pdf(self, original_pdf_path: str, chunks: List[Dict],
                             novelty_scores: List[float], output_path: str) -> Dict:
        """
        Build an annotated PDF: inline highlights coloured by novelty score on the
        original pages, plus a prepended summary page.

        Returns a small report dict with counts so callers can surface how many
        chunks were successfully located on the page surface.
        """
        try:
            doc = fitz.open(original_pdf_path)
            try:
                highlighted_chunks = 0
                for chunk, score in zip(chunks, novelty_scores):
                    if self._highlight_chunk_in_doc(doc, chunk['text'], score) > 0:
                        highlighted_chunks += 1

                self._build_summary_page(doc, chunks, novelty_scores)
                doc.save(output_path, deflate=True, garbage=3)
            finally:
                doc.close()

            report = {
                'output_path': output_path,
                'chunks_total': len(chunks),
                'chunks_highlighted': highlighted_chunks,
                'chunks_unmatched': len(chunks) - highlighted_chunks,
            }
            logger.info(
                "Annotated PDF written to %s (%d/%d chunks highlighted)",
                output_path, highlighted_chunks, len(chunks),
            )
            return report

        except Exception as e:
            logger.error(f"Error creating annotated PDF: {str(e)}")
            raise
