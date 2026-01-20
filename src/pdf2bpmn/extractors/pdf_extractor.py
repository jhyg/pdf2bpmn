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

                # If the page contains images, always attempt OCR/Vision extraction (configurable).
                # This is NOT a fallback only when text is empty; images can contain critical tables/steps.
                if Config.ENABLE_OCR and i < Config.OCR_MAX_PAGES:
                    try:
                        has_images = bool(getattr(page, "images", None)) and len(page.images) > 0
                    except Exception:
                        has_images = False

                    if has_images and Config.OCR_ALWAYS_IF_IMAGES:
                        ocr_text = self._extract_text_from_page_image(path, page_index=i)
                        if ocr_text:
                            # Merge without exploding duplicates: append unique lines only.
                            merged = _merge_text_lines(text, ocr_text)
                            text = merged
                page_texts[i + 1] = text
                all_text.append((i + 1, text))

            # Extract sections
            # 1) SOP boundary detection (optional, more robust for manuals/SOP documents)
            # 2) Fallback to heading detection
            sections = []
            if Config.ENABLE_SOP_SEGMENTATION and Config.OPENAI_API_KEY:
                try:
                    sections = self._extract_sop_sections(doc.doc_id, page_texts)
                except Exception:
                    sections = []
            if not sections:
                sections = self._extract_sections(doc.doc_id, all_text)
            
            # Create reference chunks
            if self.chunking_strategy == "semantic":
                chunks = self._create_semantic_chunks(doc.doc_id, page_texts)
            else:
                chunks = self._create_chunks(doc.doc_id, page_texts)
            
            return doc, sections, chunks

    def _extract_sop_sections(self, doc_id: str, page_texts: dict[int, str]) -> list[Section]:
        """
        문서 전체에서 SOP(독립 프로세스) 경계를 LLM으로 식별하고,
        SOP 단위로 Section을 생성합니다.
        """
        # Limit pages for boundary detection
        pages = sorted(page_texts.keys())[: Config.SOP_MAX_PAGES_FOR_BOUNDARY]
        joined = "\n\n".join([f"[PAGE {p}]\n{page_texts.get(p, '')}" for p in pages])
        if not joined.strip():
            return []

        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            from langchain_core.prompts import ChatPromptTemplate  # type: ignore
            from langchain_core.output_parsers import JsonOutputParser  # type: ignore
            from pydantic import BaseModel, Field  # type: ignore
        except Exception:
            return []

        class _SOPBoundary(BaseModel):
            title: str = Field(...)
            page_from: int = Field(...)
            page_to: int = Field(...)

        class _SOPBoundaries(BaseModel):
            sops: list[_SOPBoundary] = Field(default_factory=list)

        prompt = ChatPromptTemplate.from_template(
            """당신은 문서 구조 분석 전문가입니다.
주어진 문서에서 SOP(표준작업절차) 또는 독립적인 업무 프로세스의 경계를 식별해주세요.

규칙:
- 문서 제목 전체는 SOP가 아닐 수 있습니다. 문서 내부의 구체적인 업무 절차(요청관리, 변경관리, 장애관리 등)를 찾으세요.
- SOP는 독립적인 시작/끝을 가진 절차 단위입니다.
- 반환은 최대한 포괄적으로 하되, 과도하게 잘게 쪼개지 마세요.

응답 형식(JSON):
{{"sops":[{{"title":"SOP 제목","page_from":1,"page_to":3}}]}}

문서 내용:
{text}
"""
        )
        llm = ChatOpenAI(model=Config.OPENAI_MODEL, api_key=Config.OPENAI_API_KEY, temperature=0)
        parser = JsonOutputParser(pydantic_object=_SOPBoundaries)
        chain = prompt | llm | parser
        data = chain.invoke({"text": joined})
        boundaries = _SOPBoundaries(**data)
        if not boundaries.sops:
            return []

        sections: list[Section] = []
        for sop in boundaries.sops:
            pf = max(1, int(sop.page_from))
            pt = max(pf, int(sop.page_to))
            content = "\n\n".join([page_texts.get(p, "") for p in range(pf, pt + 1)]).strip()
            if not content:
                continue
            sections.append(
                Section(
                    section_id=generate_id(),
                    doc_id=doc_id,
                    heading=sop.title.strip() or "SOP",
                    level=1,
                    page_from=pf,
                    page_to=pt,
                    content=content,
                )
            )

        return sections

    def _extract_text_from_page_image(self, pdf_path: Path, page_index: int) -> str:
        """
        Render a PDF page to an image and extract text via OCR/Vision.
        Tries multiple render/OCR backends; always fails gracefully (returns "").
        """
        image = None

        # 1) Render via PyMuPDF if available (no external poppler dependency)
        try:
            import fitz  # type: ignore

            with fitz.open(str(pdf_path)) as doc:
                if page_index < 0 or page_index >= doc.page_count:
                    return ""
                page = doc.load_page(page_index)
                zoom = max(1.0, Config.OCR_DPI / 72.0)
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                if pix.width * pix.height > Config.OCR_MAX_IMAGE_PIXELS:
                    # Downscale if too large
                    scale = (Config.OCR_MAX_IMAGE_PIXELS / float(pix.width * pix.height)) ** 0.5
                    mat = fitz.Matrix(zoom * scale, zoom * scale)
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                from PIL import Image  # type: ignore

                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception:
            image = None

        # 2) Render via pdfplumber (best-effort)
        if image is None:
            try:
                from PIL import Image  # type: ignore
                import io

                with pdfplumber.open(pdf_path) as pdf:
                    if page_index < 0 or page_index >= len(pdf.pages):
                        return ""
                    page = pdf.pages[page_index]
                    page_image = page.to_image(resolution=Config.OCR_DPI)
                    # page_image.original is a PIL Image in many environments
                    image = getattr(page_image, "original", None)
                    if image is None:
                        # Fallback to bytes if possible
                        bio = io.BytesIO()
                        page_image.save(bio, format="PNG")
                        bio.seek(0)
                        image = Image.open(bio)
            except Exception:
                image = None

        if image is None:
            return ""

        # OCR engine selection
        engine = (Config.OCR_ENGINE or "tesseract").lower()

        if engine == "openai_vision":
            return self._ocr_with_openai_vision(image)

        # Default: tesseract
        return self._ocr_with_tesseract(image)

    def _ocr_with_tesseract(self, image) -> str:
        try:
            import pytesseract  # type: ignore
        except Exception:
            return ""

        try:
            # Korean+English is common for business docs
            text = pytesseract.image_to_string(image, lang="kor+eng")
            return (text or "").strip()
        except Exception:
            return ""

    def _ocr_with_openai_vision(self, image) -> str:
        """
        OCR via OpenAI Vision (multimodal). This is used when Tesseract isn't available or quality is insufficient.
        """
        try:
            import base64
            import io
            from langchain_openai import ChatOpenAI  # type: ignore
            from langchain_core.messages import HumanMessage  # type: ignore
        except Exception:
            return ""

        try:
            buf = io.BytesIO()
            # Keep PNG for better text clarity
            image.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            llm = ChatOpenAI(model=Config.OPENAI_MODEL, api_key=Config.OPENAI_API_KEY, temperature=0)
            prompt = (
                "You are an OCR engine. Extract ALL readable Korean/English text from the image. "
                "Preserve line breaks. Do not add commentary. Output plain text only."
            )
            msg = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]
            )
            resp = llm.invoke([msg])
            text = getattr(resp, "content", "") or ""
            return str(text).strip()
        except Exception:
            return ""


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


def _merge_text_lines(base_text: str, extra_text: str) -> str:
    """
    Merge OCR text into extracted text while minimizing obvious duplication.
    Keeps ordering: base_text first, then any new lines from extra_text.
    """
    base_lines = [ln.strip() for ln in (base_text or "").splitlines() if ln.strip()]
    extra_lines = [ln.strip() for ln in (extra_text or "").splitlines() if ln.strip()]
    if not extra_lines:
        return base_text or ""

    seen = set(base_lines)
    merged = list(base_lines)
    for ln in extra_lines:
        if ln not in seen:
            merged.append(ln)
            seen.add(ln)
    return "\n".join(merged).strip()



