import threading
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from src.common.logger import get_logger
from src.common.custom_exception import CustomException
from src.config.config import groq_api_key as _groq_api_key

logger = get_logger(__name__)

_llm_instance:       ChatGroq             | None = None
_embedding_instance: HuggingFaceEmbeddings | None = None

_llm_lock       = threading.Lock()
_embedding_lock = threading.Lock()


#  LLM singleton 

def get_llm(api_key: str = _groq_api_key) -> ChatGroq:
    """
    Return the shared ChatGroq instance, creating it on the first call.
    Thread-safe via a lock — only one thread runs the constructor.
    """
    global _llm_instance

    if _llm_instance is not None:
        return _llm_instance

    with _llm_lock:
        if _llm_instance is not None:
            return _llm_instance

        try:
            logger.info("Initialising LLM (first call — will cache for process lifetime)")
            _llm_instance = ChatGroq(
                api_key=api_key,
                model="openai/gpt-oss-120b",
                temperature=0.3,
                max_tokens=2048
            )
            logger.info("LLM ready and cached ✓")
            return _llm_instance
        except Exception as e:
            logger.error("Error initialising LLM: %s", str(e))
            raise CustomException("Error initialising LLM.") from e


#  Embedding model singleton 

def get_embedding_model() -> HuggingFaceEmbeddings:

    global _embedding_instance

    if _embedding_instance is not None:
        return _embedding_instance

    with _embedding_lock:
        if _embedding_instance is not None:
            return _embedding_instance

        try:
            logger.info("Initialising embedding model (first call — will cache for process lifetime)")
            _embedding_instance = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Embedding model ready and cached ✓")
            return _embedding_instance
        except Exception as e:
            logger.error("Error initialising embedding model: %s", str(e))
            raise CustomException("Error initialising embedding model.") from e


#  Warm-up helper (optional) 

def warmup():

    logger.info("Running model warm-up...")
    get_embedding_model()  
    get_llm()               
    logger.info("Warm-up complete — models are hot and ready")