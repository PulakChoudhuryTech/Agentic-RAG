"""
Loads raw documents from data/raw_docs/** -- both markdown (data/raw_docs/{hr,it,travel}/*.md)
and PDF (data/raw_docs/personal/*.pdf) -- and extracts the lightweight metadata
used for metadata-based routing (rag/metadata_router.py): category, title,
effective date, and per-section country tags.

File layout convention:
  - category comes from the immediate parent folder name (hr/it/travel/personal),
    which is authoritative -- not guessed from content.
  - for markdown: title is the first `# Heading` line; effective date is
    parsed from a `**Effective date:** YYYY-MM-DD` line.
  - for PDFs: there's no such convention (these are real-world documents --
    bills, insurance policies, e-tickets -- not authored for this project),
    so title falls back to the filename and there's no effective_date.
  - country is NOT a whole-document property here: several HR documents
    (e.g. parental_leave_policy.md) cover multiple countries in separate
    `### <Country Name>` subsections. Instead, `tag_country()` below is
    applied per PARENT CHUNK during ingestion (ingest.py), scanning that
    chunk's text for a level-3 heading matching a known country name, and
    tagging the resulting parent/child chunk metadata with that country.
    This is what lets "the India section of the parental leave policy" be
    found by a country-filtered search even though the document as a whole
    isn't India-specific. PDFs never produce a country tag this way (no
    markdown headings), which is fine -- country/metadata filtering for
    personal docs works via `category=personal` alone.

PDF text extraction, two-tier:
  1. Try pypdf's text layer extraction first (fast, no external tools).
  2. If that yields near-nothing per page (a common symptom of a scanned/
     image-based PDF, or a PDF whose font encoding has no ToUnicode map --
     both are common with e-tickets/boarding passes), fall back to OCR via
     pytesseract + pdf2image, which renders each page to an image and reads
     it visually. This needs the `tesseract` and `poppler` system binaries
     installed (`brew install tesseract poppler` on macOS) -- if they're
     missing, `_ocr_pdf()` raises and `load_documents()` skips that file
     with a printed warning rather than silently ingesting empty content.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

RAW_DOCS_DIR = Path(__file__).parent.parent.parent.parent / "data" / "raw_docs"

VALID_CATEGORIES = {"hr", "it", "travel", "personal"}

# Below this many extracted characters per page, we treat pypdf's text-layer
# extraction as having failed and fall back to OCR.
MIN_CHARS_PER_PAGE_BEFORE_OCR_FALLBACK = 20

COUNTRY_HEADING_MAP: dict[str, str] = {
    "india": "IN",
    "united states": "US",
    "united kingdom": "GB",
    "germany": "DE",
}

_TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_EFFECTIVE_DATE_RE = re.compile(r"\*\*Effective date:\*\*\s*(\d{4}-\d{2}-\d{2})")
_H3_HEADING_RE = re.compile(r"^###\s+(.+)$", re.MULTILINE)


@dataclass
class RawDocument:
    source_path: str  # relative to data/raw_docs/, e.g. "hr/parental_leave_policy.md"
    title: str
    category: str
    effective_date: str | None
    raw_text: str


def _load_markdown(path: Path, category: str) -> RawDocument:
    text = path.read_text()
    title_match = _TITLE_RE.search(text)
    date_match = _EFFECTIVE_DATE_RE.search(text)

    return RawDocument(
        source_path=str(path.relative_to(RAW_DOCS_DIR)),
        title=title_match.group(1).strip() if title_match else path.stem,
        category=category,
        effective_date=date_match.group(1) if date_match else None,
        raw_text=text,
    )


def _ocr_pdf(path: Path) -> str:
    from pdf2image import convert_from_path
    from pytesseract import image_to_string

    images = convert_from_path(str(path), dpi=300)
    pages = [image_to_string(image) for image in images]
    return "\n\n".join(pages)


def _load_pdf(path: Path, category: str) -> RawDocument | None:
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n\n".join(pages)

    avg_chars_per_page = len(text) / max(len(pages), 1)
    if avg_chars_per_page < MIN_CHARS_PER_PAGE_BEFORE_OCR_FALLBACK:
        print(f"  {path.name}: text layer nearly empty ({len(text)} chars over {len(pages)} pages) -- trying OCR")
        try:
            text = _ocr_pdf(path)
        except Exception as exc:  # noqa: BLE001 - OCR is a best-effort fallback, not a hard requirement
            print(
                f"  SKIPPING {path.name}: no usable text layer and OCR failed ({exc}). "
                "Install OCR support with `brew install tesseract poppler` (macOS) and "
                "`pip install -e .` to pick up pytesseract/pdf2image, then re-run ingestion."
            )
            return None

    if not text.strip():
        print(f"  SKIPPING {path.name}: no text could be extracted (text layer and OCR both empty)")
        return None

    return RawDocument(
        source_path=str(path.relative_to(RAW_DOCS_DIR)),
        title=path.stem,
        category=category,
        effective_date=None,
        raw_text=text,
    )


def load_documents() -> list[RawDocument]:
    documents: list[RawDocument] = []

    for path in sorted(RAW_DOCS_DIR.glob("**/*")):
        if not path.is_file() or path.suffix.lower() not in (".md", ".pdf"):
            continue
        if path.name == "README.md":  # folder-level documentation, not ingestible content
            continue

        category = path.parent.name
        if category not in VALID_CATEGORIES:
            continue

        if path.suffix.lower() == ".md":
            documents.append(_load_markdown(path, category))
        else:
            doc = _load_pdf(path, category)
            if doc is not None:
                documents.append(doc)

    return documents


def tag_country(chunk_text: str) -> str | None:
    """Scan a chunk of text for a level-3 heading matching a known country
    name (e.g. "### India") and return its ISO code, or None if no such
    heading appears in this chunk."""
    for heading in _H3_HEADING_RE.findall(chunk_text):
        code = COUNTRY_HEADING_MAP.get(heading.strip().lower())
        if code:
            return code
    return None
