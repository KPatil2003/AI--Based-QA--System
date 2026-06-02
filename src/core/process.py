import os

from src.core.pdf_loader import split_documents
from src.advance_rag.ocr import load_document_with_ocr, ALL_SUPPORTED_EXTENSIONS
from src.core.vectorestore_load import save_vectorstore
from src.common.logger import get_logger
from src.common.custom_exception import CustomException

logger = get_logger(__name__)


def process_uploaded_document(filename: str, user_id: int):

    try:
        logger.info("Starting pipeline for user=%d file=%s", user_id, filename)

        filepath = os.path.join("data/uploads", filename)

        if not os.path.exists(filepath):
            raise CustomException(f"File not found: {filepath}")

        ext = os.path.splitext(filename)[1].lower()

        if ext not in ALL_SUPPORTED_EXTENSIONS:
            raise CustomException(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}"
            )

        # ── Load (with automatic OCR fallback for scanned PDFs / images) 
        documents = load_document_with_ocr(filepath)

        if not documents:
            raise CustomException(
                f"No text could be extracted from '{filename}'. "
                "The file may be encrypted, corrupted, or contain only images without OCR support."
            )

        #  Split into chunks 
        text_chunks = split_documents(documents)

        #  Save to per-user FAISS vectorstore 
        save_vectorstore(text_chunks, user_id=user_id)

        logger.info(
            "Pipeline complete for user=%d file=%s → %d chunks",
            user_id, filename, len(text_chunks)
        )

        return {"status": "success", "chunks": len(text_chunks)}

    except CustomException:
        raise
    except Exception as e:
        logger.error("Pipeline failed: %s", str(e))
        raise CustomException(f"Pipeline failed: {str(e)}") from e


# Backwards compat alias
def process_uploaded_pdf(filename: str, user_id: int):
    return process_uploaded_document(filename, user_id)