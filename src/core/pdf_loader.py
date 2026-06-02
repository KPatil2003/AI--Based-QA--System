import os
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.common.logger import get_logger
from src.common.custom_exception import CustomException

logger    = get_logger(__name__)
data_path = "data/uploads"

# Maps file extension → loader class
LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".docx": Docx2txtLoader,
    ".doc":  Docx2txtLoader,
    ".txt":  TextLoader,
    ".md":   UnstructuredMarkdownLoader,
}

SUPPORTED_EXTENSIONS = set(LOADER_MAP.keys())


def load_single_document(filepath: str):
    """
    Load ONE specific document (PDF / DOCX / TXT / MD).
    Called by process_uploaded_document() with the just-uploaded file path.
    """
    try:
        ext = os.path.splitext(filepath)[1].lower()
        loader_cls = LOADER_MAP.get(ext)
        if not loader_cls:
            raise CustomException(
                f"Unsupported file type '{ext}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        logger.info("Loading document (%s): %s", ext, filepath)
        loader    = loader_cls(filepath)
        documents = loader.load()
        logger.info("Loaded %d page(s)/section(s) from %s", len(documents), filepath)
        return documents
    except CustomException:
        raise
    except Exception as e:
        logger.error("Error loading document %s: %s", filepath, str(e))
        raise CustomException(f"Error loading document: {str(e)}") from e


def load_single_pdf(filepath: str):
    return load_single_document(filepath)


def load_all_documents():
    """
    Loads every supported document in the uploads directory.
    Use only when reprocessing everything from scratch.
    """
    try:
        logger.info("Loading all documents from: %s", data_path)
        all_docs = []
        for fname in os.listdir(data_path):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                fpath = os.path.join(data_path, fname)
                try:
                    docs = load_single_document(fpath)
                    all_docs.extend(docs)
                except Exception as e:
                    logger.warning("Skipping %s: %s", fname, e)
        logger.info("Loaded %d total document sections", len(all_docs))
        return all_docs
    except Exception as e:
        logger.error("Error loading documents: %s", str(e))
        raise CustomException(f"Error loading documents: {str(e)}") from e


def load_all_pdfs():
    return load_all_documents()


def split_documents(documents):
    try:
        logger.info("Splitting %d document section(s) into chunks", len(documents))
        splitter    = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
        text_chunks = splitter.split_documents(documents)
        logger.info("Created %d chunks", len(text_chunks))
        return text_chunks
    except Exception as e:
        logger.error("Error splitting documents: %s", str(e))
        raise CustomException(f"Error splitting documents: {str(e)}") from e