import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from src.core.llm import get_embedding_model
from src.common.logger import get_logger
from src.common.custom_exception import CustomException

logger = get_logger(__name__)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,                  # RRF constant — 60 is the standard default
    top_n: int = 5
) -> List[Document]:
    """
    Merge multiple ranked document lists into one using RRF.

    RRF score for a doc d = Σ  1 / (k + rank(d, list_i))
    Documents appearing in multiple lists get boosted naturally.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            # Use page_content as the dedup key (good enough for exam docs)
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            doc_map[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys[:top_n]]


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """
    Retriever that queries both FAISS (dense) and BM25 (sparse),
    then merges results with Reciprocal Rank Fusion.
    """

    faiss_store: object = Field(default=None, exclude=True)
    bm25_retriever: object = Field(default=None, exclude=True)
    k: int = Field(default=5)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:

        # ── Dense retrieval (FAISS) ──
        try:
            dense_docs = self.faiss_store.similarity_search(query, k=self.k * 2)
            logger.debug("Dense retrieval returned %d docs", len(dense_docs))
        except Exception as e:
            logger.warning("Dense retrieval failed: %s", e)
            dense_docs = []

        # ── Sparse retrieval (BM25) ──
        try:
            sparse_docs = self.bm25_retriever.invoke(query)
            logger.debug("Sparse (BM25) retrieval returned %d docs", len(sparse_docs))
        except Exception as e:
            logger.warning("Sparse retrieval failed: %s", e)
            sparse_docs = []

        # ── Fallback: if one side is empty just return the other ──
        if not dense_docs and not sparse_docs:
            logger.warning("Both retrievers returned nothing for query: %s", query[:60])
            return []
        if not dense_docs:
            return sparse_docs[:self.k]
        if not sparse_docs:
            return dense_docs[:self.k]

        # ── Fuse rankings ──
        fused = reciprocal_rank_fusion(
            [dense_docs, sparse_docs],
            top_n=self.k
        )
        logger.info("Hybrid RRF returned %d docs", len(fused))
        return fused


# ── Public factory ────────────────────────────────────────────────────────────

def get_hybrid_retriever(user_id: int, k: int = 5) -> HybridRetriever:

    try:
        path = f"vectorstore/user_{user_id}/faiss_db"

        if not os.path.exists(os.path.join(path, "index.faiss")):
            raise CustomException(
                "No documents found. Please upload a document first."
            )

        embedding_model = get_embedding_model()

        # ── Load FAISS ──
        faiss_store = FAISS.load_local(
            path,
            embedding_model,
            allow_dangerous_deserialization=True
        )
        logger.info("FAISS loaded for user %d (%s)", user_id, path)

        # ── Extract all stored docs for BM25 ──
        # FAISS stores a docstore dict under faiss_store.docstore._dict
        all_docs = list(faiss_store.docstore._dict.values())

        if not all_docs:
            raise CustomException("Vectorstore is empty. Please re-upload your document.")

        bm25 = BM25Retriever.from_documents(all_docs, k=k * 2)
        logger.info("BM25 index built with %d documents for user %d", len(all_docs), user_id)

        return HybridRetriever(
            faiss_store=faiss_store,
            bm25_retriever=bm25,
            k=k
        )

    except CustomException:
        raise
    except Exception as e:
        logger.error("Failed to build hybrid retriever: %s", e)
        raise CustomException(f"Retriever initialisation failed: {str(e)}") from e