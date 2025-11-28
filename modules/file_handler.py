# modules/file_handler.py
"""
Robust PDF text extraction helper using PyPDF2.

Public:
    extract_pdf_data(file, max_pages=500) -> str

Returns:
    - concatenated text if extraction succeeds
    - a helpful short string (e.g., "No text found...") when PDF contains no extractable text
    - empty string ("") on failure (so callers can handle UI messaging)
Notes:
    - This function avoids importing Streamlit so it can be used from non-UI code.
    - If the PDF is image-only (scanned), consider using OCR (e.g., pytesseract) upstream.
"""
from typing import Any
import warnings

try:
    from PyPDF2 import PdfReader
except Exception as e:
    raise ImportError("PyPDF2 is required for PDF extraction. Install with: pip install PyPDF2") from e


def extract_pdf_data(file: Any, max_pages: int = 500) -> str:
    """
    Extract text from a PDF file-like or path.

    Args:
        file: file-like object (e.g., Streamlit UploadedFile) or filesystem path (str / Path)
        max_pages: safety cap to avoid processing extremely large PDFs (set to 0 for no limit)

    Returns:
        Cleaned text (str). Returns "" on failure so the caller can show UI errors.
    """
    text_parts = []
    try:
        # If file-like, ensure pointer at start
        try:
            if hasattr(file, "seek"):
                file.seek(0)
        except Exception:
            # best-effort; continue even if seek fails
            pass

        reader = PdfReader(file)
        total_pages = len(reader.pages or [])
        if total_pages == 0:
            return "PDF has no readable pages."

        # Enforce max_pages cap (if > 0)
        pages_to_read = total_pages if (max_pages <= 0) else min(total_pages, max_pages)
        if pages_to_read < total_pages:
            warnings.warn(f"PDF has {total_pages} pages; limiting extraction to first {pages_to_read} pages.")

        for i in range(pages_to_read):
            try:
                page = reader.pages[i]
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text.strip())
                else:
                    text_parts.append(f"[Page {i+1} contains no readable text]")
            except IndexError:
                # defensive: if pages change, break
                break
            except Exception as inner_err:
                warnings.warn(f"Error extracting page {i+1}: {inner_err}")
                text_parts.append(f"[Error extracting text from page {i+1}]")
                continue

    except Exception as e:
        # Do not raise; return empty string so caller (UI) can display a friendly message
        warnings.warn(f"Could not read PDF: {e}")
        return ""

    # Join and clean
    full_text = "\n\n".join(text_parts).strip()

    if not full_text:
        return "No text found. This PDF may be scanned or image-only."

    return full_text