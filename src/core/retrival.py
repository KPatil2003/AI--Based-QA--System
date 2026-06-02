import os

from langchain_community.vectorstores import FAISS
from src.core.llm import get_embedding_model
from src.common.logger import get_logger
from src.common.custom_exception import CustomException

logger = get_logger(__name__)


def get_retriever(
    user_id: int,
    k=5
):

    try:

        path = (
            f"vectorstore/user_{user_id}/faiss_db"
        )

        if not os.path.exists(path):

            raise CustomException(
                "No documents found. "
                "Please upload document first."
            )

        embedding_model = get_embedding_model()

        vectorstore = FAISS.load_local(
            path,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        return vectorstore.as_retriever(
            search_kwargs={"k": k}
        )

    except Exception as e:

        raise CustomException(
            f"Error loading retriever: {str(e)}"
        )