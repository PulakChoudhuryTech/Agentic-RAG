# data/raw_docs/personal/

Drop your own real personal PDFs here (bills, insurance policies, e-tickets,
etc.) to test the RAG pipeline against genuinely unstructured, real-world
documents instead of only the synthetic HR/IT/travel sample docs.

- Ingested into the `personal` document category (`ingestion/loader.py`).
- Text extraction: `pypdf` first, falling back to OCR (`pytesseract` +
  `pdf2image`) for scanned/image-based PDFs with no text layer -- see
  `ingestion/loader.py`'s module docstring for details, and the OCR setup
  note in the main README if `make ingest` reports a page it couldn't read.
- **The actual PDF files are gitignored** (`.gitignore`'s
  `data/raw_docs/personal/*.pdf` rule) -- they're your real personal data
  and should never be committed. This README is tracked; your PDFs are not.
