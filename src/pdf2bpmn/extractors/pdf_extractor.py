"""PDF text and structure extraction."""
import hashlib
import re
from pathlib import Path
from typing import Generator

import pdfplumber

from ..models.entities import Document, Section, ReferenceChunk, generate_id
from ..config import Config


class PDFExtractor:
    """Extract text and structure from PDF files."""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None, chunking_strategy: str = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.chunking_strategy = chunking_strategy or Config.CHUNKING_STRATEGY
    
    def extract_document(self, pdf_path: str) -> tuple[Document, list[Section], list[ReferenceChunk]]:
        """Extract document structure and content from PDF."""
        path = Path(pdf_path)
        
        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)
            
            # Create document
            doc = Document(
                doc_id=generate_id(),
                title=path.stem,
                source=str(path),
                page_count=page_count
            )
            
            # Extract all text with page info
            all_text = []
            page_texts = {}
            
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                page_texts[i + 1] = text
                all_text.append((i + 1, text))
            
            # Extract sections (heading detection)
            sections = self._extract_sections(doc.doc_id, all_text)
            
            # Create reference chunks
            if self.chunking_strategy == "semantic":
                chunks = self._create_semantic_chunks(doc.doc_id, page_texts)
            else:
                chunks = self._create_chunks(doc.doc_id, page_texts)
            
            return doc, sections, chunks
    
    def _extract_sections(
        self, 
        doc_id: str, 
        page_texts: list[tuple[int, str]]
    ) -> list[Section]:
        """Extract section hierarchy from document."""
        sections = []
        
        # Patterns for detecting headings
        heading_patterns = [
            (1, r'^#{1}\s+(.+)$'),  # Markdown style
            (1, r'^제\s*\d+\s*장\s*(.+)$'),  # Korean chapter
            (2, r'^제\s*\d+\s*절\s*(.+)$'),  # Korean section
            (2, r'^#{2}\s+(.+)$'),
            (1, r'^\d+\.\s+([A-Z가-힣].+)$'),  # Numbered heading
            (2, r'^\d+\.\d+\s+(.+)$'),
            (3, r'^\d+\.\d+\.\d+\s+(.+)$'),
            (1, r'^[IVX]+\.\s+(.+)$'),  # Roman numerals
            (2, r'^[A-Z]\.\s+(.+)$'),  # Letter headings
        ]
        
        current_section = None
        section_start_page = 1
        
        for page_num, text in page_texts:
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for heading patterns
                for level, pattern in heading_patterns:
                    match = re.match(pattern, line, re.MULTILINE)
                    if match:
                        # Save previous section
                        if current_section:
                            current_section.page_to = page_num - 1
                            sections.append(current_section)
                        
                        # Create new section
                        current_section = Section(
                            section_id=generate_id(),
                            doc_id=doc_id,
                            heading=line,
                            level=level,
                            page_from=page_num,
                            page_to=page_num,  # Will be updated
                            content=""
                        )
                        section_start_page = page_num
                        break
                
                # Add content to current section
                if current_section:
                    current_section.content += line + "\n"
        
        # Close last section
        if current_section:
            current_section.page_to = page_texts[-1][0] if page_texts else 1
            sections.append(current_section)
        
        # If no sections detected, create one for whole document
        if not sections:
            full_text = "\n".join(text for _, text in page_texts)
            sections.append(Section(
                section_id=generate_id(),
                doc_id=doc_id,
                heading="Document Content",
                level=1,
                page_from=1,
                page_to=len(page_texts),
                content=full_text
            ))
        
        return sections
    
    def _create_chunks(
        self, 
        doc_id: str, 
        page_texts: dict[int, str]
    ) -> list[ReferenceChunk]:
        """Create overlapping text chunks for embedding."""
        chunks = []
        
        for page_num, text in page_texts.items():
            if not text.strip():
                continue
            
            # Split into sentences/paragraphs
            paragraphs = re.split(r'\n\s*\n', text)
            
            current_chunk = ""
            chunk_start = 0
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                if len(current_chunk) + len(para) > self.chunk_size:
                    # Save current chunk
                    if current_chunk:
                        chunks.append(self._create_chunk(
                            doc_id, page_num, chunk_start, current_chunk
                        ))
                    
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + " " + para
                    chunk_start = chunk_start + len(current_chunk) - len(overlap_text) - len(para) - 1
                else:
                    current_chunk += ("\n\n" if current_chunk else "") + para
            
            # Save remaining chunk
            if current_chunk:
                chunks.append(self._create_chunk(
                    doc_id, page_num, chunk_start, current_chunk
                ))
        
        return chunks
    
    def _create_semantic_chunks(
        self, 
        doc_id: str, 
        page_texts: dict[int, str]
    ) -> list[ReferenceChunk]:
        """Create semantic chunks based on sections/paragraphs to minimize overlap."""
        chunks = []
        
        # Combine all pages
        full_text = []
        for page_num in sorted(page_texts.keys()):
            text = page_texts[page_num]
            if text.strip():
                full_text.append((page_num, text))
        
        if not full_text:
            return chunks
        
        # Split by major sections (double newlines, headings)
        sections = []
        current_section = []
        current_page = full_text[0][0]
        
        for page_num, text in full_text:
            # Split by paragraphs (double newlines)
            paragraphs = re.split(r'\n\s*\n+', text)
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Check if this is a heading (potential section break)
                is_heading = any(re.match(pattern, para.split('\n')[0], re.MULTILINE) 
                                for pattern in [
                                    r'^제\s*\d+\s*[장절]',
                                    r'^\d+\.\s+[A-Z가-힣]',
                                    r'^#{1,3}\s+',
                                ])
                
                # If heading and current section is large enough, start new section
                if is_heading and current_section and len('\n\n'.join(current_section)) > self.chunk_size * 0.5:
                    sections.append((current_page, '\n\n'.join(current_section)))
                    current_section = [para]
                    current_page = page_num
                else:
                    current_section.append(para)
                    current_page = page_num
        
        # Add last section
        if current_section:
            sections.append((current_page, '\n\n'.join(current_section)))
        
        # Create chunks from sections (minimal overlap)
        for i, (page_num, section_text) in enumerate(sections):
            # If section is too large, split it
            if len(section_text) > self.chunk_size * 1.5:
                # Split large section
                paragraphs = re.split(r'\n\s*\n+', section_text)
                current_chunk = ""
                chunk_start = 0
                
                for para in paragraphs:
                    para = para.strip()
                    if not para:
                        continue
                    
                    if len(current_chunk) + len(para) > self.chunk_size:
                        if current_chunk:
                            chunks.append(self._create_chunk(
                                doc_id, page_num, chunk_start, current_chunk
                            ))
                        current_chunk = para
                        chunk_start = 0
                    else:
                        current_chunk += ("\n\n" if current_chunk else "") + para
                
                if current_chunk:
                    chunks.append(self._create_chunk(
                        doc_id, page_num, chunk_start, current_chunk
                    ))
            else:
                # Use entire section as chunk (minimal overlap with previous)
                if i > 0 and self.chunk_overlap > 0:
                    # Add minimal overlap from previous chunk
                    prev_chunk = chunks[-1].text if chunks else ""
                    overlap = prev_chunk[-min(self.chunk_overlap, len(prev_chunk)):] if prev_chunk else ""
                    if overlap:
                        section_text = overlap + "\n\n" + section_text
                
                chunks.append(self._create_chunk(
                    doc_id, page_num, 0, section_text
                ))
        
        return chunks
    
    def _create_chunk(
        self, 
        doc_id: str, 
        page: int, 
        start: int, 
        text: str
    ) -> ReferenceChunk:
        """Create a single reference chunk."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        return ReferenceChunk(
            chunk_id=generate_id(),
            doc_id=doc_id,
            page=page,
            span=f"{start}:{start + len(text)}",
            text=text,
            hash=text_hash
        )
    
    def iter_chunks(self, pdf_path: str) -> Generator[ReferenceChunk, None, None]:
        """Stream chunks from PDF for incremental processing."""
        _, _, chunks = self.extract_document(pdf_path)
        for chunk in chunks:
            yield chunk




