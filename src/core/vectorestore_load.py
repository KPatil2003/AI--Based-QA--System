import os
from langchain_community.vectorstores import FAISS
from src.core.llm import get_embedding_model
from src.common.logger import get_logger
from src.common.custom_exception import CustomException

logger = get_logger(__name__)


def get_vectorstore_path(user_id: int) -> str:
    return f"vectorstore/user_{user_id}/faiss_db"


def save_vectorstore(text_chunks, user_id: int):

    try:
        path = get_vectorstore_path(user_id)

        os.makedirs(path, exist_ok=True)

        embedding_model = get_embedding_model()

        new_index = FAISS.from_documents(
            text_chunks,
            embedding_model
        )

        index_file = os.path.join(path, "index.faiss")

        if os.path.exists(index_file):

            logger.info(
                "Existing vectorstore found for user %d — merging",
                user_id
            )

            existing = FAISS.load_local(
                path,
                embedding_model,
                allow_dangerous_deserialization=True
            )

            existing.merge_from(new_index)
            existing.save_local(path)

            logger.info(
                "Merged vectorstore for user %d",
                user_id
            )

        else:

            logger.info(
                "Creating new vectorstore for user %d",
                user_id
            )

            new_index.save_local(path)

            logger.info(
                "Saved new vectorstore for user %d",
                user_id
            )

    except Exception as e:

        logger.error(
            "Error saving vectorstore: %s",
            str(e)
        )

        raise CustomException(
            f"Error saving vectorstore: {str(e)}"
        ) from e