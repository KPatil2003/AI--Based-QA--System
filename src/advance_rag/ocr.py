import os
from typing import List

from langchain_core.documents import Document
from src.common.logger import get_logger
from src.common.custom_exception import CustomException

logger = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

# If a page has fewer than this many non-whitespace chars we consider it "blank"
MIN_TEXT_CHARS    = 50
# DPI for PDF→image conversion (higher = better OCR but slower; 200 is good default)
OCR_DPI           = 200
# Supported image-only formats (no PDF here — handled separately)
IMAGE_EXTENSIONS  = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}


# ── Lazy import guards ────────────────────────────────────────────────────────

def _import_pytesseract():
    try:
        import pytesseract
        return pytesseract
    except ImportError:
        raise CustomException(
            "pytesseract is not installed. Run: pip install pytesseract\n"
            "Also install Tesseract: https://tesseract-ocr.github.io/tessdoc/Installation.html"
        )

def _import_pdf2image():
    try:
        from pdf2image import convert_from_path
        return convert_from_path
    except ImportError:
        raise CustomException(
            "pdf2image is not installed. Run: pip install pdf2image\n"
            "Also install poppler: apt install poppler-utils (Ubuntu) or brew install poppler (macOS)"
        )

def _import_pil():
    try:
        from PIL import Image
        return Image
    except ImportError:
        raise CustomException("Pillow is not installed. Run: pip install Pillow")


# ── Core OCR helpers ──────────────────────────────────────────────────────────

def _ocr_image_to_text(image, lang: str = "eng") -> str:
    """Run pytesseract on a single PIL Image and return extracted text."""
    pytesseract = _import_pytesseract()
    try:
        text = pytesseract.image_to_string(image, lang=lang)
        return text.strip()
    except Exception as e:
        logger.warning("pytesseract failed on image: %s", e)
        return ""


def _is_text_sufficient(text: str) -> bool:
    return len(text.replace(" ", "").replace("\n", "")) >= MIN_TEXT_CHARS


# ── PDF handling ──────────────────────────────────────────────────────────────

def _extract_pdf_text_native(filepath: str) -> List[Document]:
    """
    Try fast native text extraction first using PyPDFLoader.
    Returns list of Document objects (one per page).
    """
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(filepath)
        docs = loader.load()
        logger.debug("Native PDF extraction: %d pages from %s", len(docs), filepath)
        return docs
    except Exception as e:
        logger.warning("Native PDF extraction failed: %s", e)
        return []


def _extract_pdf_text_ocr(filepath: str, lang: str = "eng") -> List[Document]:
    """
    Convert each PDF page to an image and OCR it.
    Used when native extraction yields blank pages (scanned documents).
    """
    convert_from_path = _import_pdf2image()
    docs = []

    try:
        logger.info("Running OCR on PDF: %s (DPI=%d)", filepath, OCR_DPI)
        pages = convert_from_path(filepath, dpi=OCR_DPI)

        for page_num, page_image in enumerate(pages, start=1):
            text = _ocr_image_to_text(page_image, lang=lang)
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": filepath,
                        "page":   page_num,
                        "method": "ocr"
                    }
                ))
                logger.debug("OCR page %d: %d chars", page_num, len(text))
            else:
                logger.warning("OCR returned empty text for page %d of %s", page_num, filepath)

        logger.info("OCR complete: %d pages extracted from %s", len(docs), filepath)
        return docs

    except Exception as e:
        logger.error("PDF OCR failed: %s", e)
        raise CustomException(f"OCR failed for {os.path.basename(filepath)}: {str(e)}") from e


# ── Image handling ────────────────────────────────────────────────────────────

def _extract_image_text_ocr(filepath: str, lang: str = "eng") -> List[Document]:
    """OCR a standalone image file (PNG/JPG etc.)."""
    Image = _import_pil()
    try:
        logger.info("Running OCR on image: %s", filepath)
        img  = Image.open(filepath)
        text = _ocr_image_to_text(img, lang=lang)

        if not text:
            raise CustomException(f"OCR returned no text from image: {os.path.basename(filepath)}")

        return [Document(
            page_content=text,
            metadata={
                "source": filepath,
                "page":   1,
                "method": "ocr"
            }
        )]
    except CustomException:
        raise
    except Exception as e:
        raise CustomException(f"Image OCR failed: {str(e)}") from e


# ── Text-based format loaders ─────────────────────────────────────────────────

TEXT_EXTENSIONS = {".docx", ".doc", ".txt", ".md"}

def _extract_text_document(filepath: str) -> List[Document]:
    """
    Load DOCX / DOC / TXT / MD using the appropriate LangChain loader.
    No OCR needed — these are native text formats.
    """
    from langchain_community.document_loaders import (
        Docx2txtLoader,
        TextLoader,
        UnstructuredMarkdownLoader,
    )

    ext = os.path.splitext(filepath)[1].lower()

    LOADER_MAP = {
        ".docx": Docx2txtLoader,
        ".doc":  Docx2txtLoader,
        ".txt":  TextLoader,
        ".md":   UnstructuredMarkdownLoader,
    }

    loader_cls = LOADER_MAP.get(ext)
    if not loader_cls:
        raise CustomException(f"No loader available for '{ext}'")

    try:
        logger.info("Loading text document (%s): %s", ext, filepath)
        docs = loader_cls(filepath).load()
        logger.info("Loaded %d section(s) from %s", len(docs), filepath)
        return docs
    except Exception as e:
        raise CustomException(f"Failed to load {ext} file: {str(e)}") from e


#  Public API 

ALL_SUPPORTED_EXTENSIONS = {".pdf"} | IMAGE_EXTENSIONS | TEXT_EXTENSIONS

def load_document_with_ocr(filepath: str, lang: str = "eng") -> List[Document]:
  
    if not os.path.exists(filepath):
        raise CustomException(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    # ── Scanned image files → OCR ──
    if ext in IMAGE_EXTENSIONS:
        logger.info("Image file → OCR: %s", filepath)
        return _extract_image_text_ocr(filepath, lang=lang)

    # ── PDF → native first, OCR fallback for scanned pages ──
    if ext == ".pdf":
        native_docs = _extract_pdf_text_native(filepath)

        sufficient_pages = [
            d for d in native_docs
            if _is_text_sufficient(d.page_content)
        ]

        if sufficient_pages:
            pct = len(sufficient_pages) / max(len(native_docs), 1) * 100
            logger.info(
                "Digital PDF — native text used (%.0f%% pages have text): %s",
                pct, filepath
            )
            return native_docs

        logger.info(
            "Scanned PDF detected (%d/%d pages empty) — falling back to OCR: %s",
            len(native_docs) - len(sufficient_pages), len(native_docs), filepath
        )
        return _extract_pdf_text_ocr(filepath, lang=lang)

    # ── DOCX / TXT / MD → native text loaders ──
    if ext in TEXT_EXTENSIONS:
        logger.info("Text document (%s) → native loader: %s", ext, filepath)
        return _extract_text_document(filepath)

    raise CustomException(
        f"Unsupported file type '{ext}'. "
        f"Supported: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}"
    )


def is_scanned_pdf(filepath: str) -> bool:
 
    try:
        docs = _extract_pdf_text_native(filepath)
        total_chars = sum(
            len(d.page_content.replace(" ", "").replace("\n", ""))
            for d in docs
        )
        return total_chars < MIN_TEXT_CHARS * max(len(docs), 1)
    except Exception:
        return False