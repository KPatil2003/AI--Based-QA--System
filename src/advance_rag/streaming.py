

import json
import os

from flask import Blueprint, Response, request, g, stream_with_context
from langchain_core.prompts import PromptTemplate

from src.core.llm import get_llm
from src.advance_rag.hybrid_search import get_hybrid_retriever
from src.advance_rag.query_rewriting import rewrite_query
from src.advance_rag.semantic_cache import SemanticCache
from src.core.auth import decode_token
from src.models.database import execute, fetch_one
from src.common.logger import get_logger

import jwt

logger = get_logger(__name__)

streaming_ask = Blueprint("streaming_ask", __name__)


#  SSE helpers 

def _sse(data: str) -> str:
    """Format a string as an SSE data event."""
    return f"data: {data}\n\n"

def _sse_error(message: str) -> str:
    return f"data: [ERROR] {message}\n\n"

def _sse_done() -> str:
    return "data: [DONE]\n\n"

def _sse_meta(payload: dict) -> str:
    """Send metadata (marks, cached flag, etc.) as a special event."""
    return f"event: meta\ndata: {json.dumps(payload)}\n\n"


#  Auth helper (JWT from query param for SSE) 

def _resolve_user_from_request():
    """
    SSE connections can't easily send Authorization headers via EventSource.
    We accept the token as either:
      - Authorization: Bearer <token>  header  (fetch-based SSE)
      - ?token=<token>                 query   (EventSource-based SSE)
    Returns (user_id, user_email) or raises ValueError.
    """
    auth_header = request.headers.get("Authorization", "")
    token = None

    if auth_header.startswith("Bearer "):
        token = auth_header.split(" ", 1)[1]
    elif request.args.get("token"):
        token = request.args.get("token")
    else:
        raise ValueError("No authentication token provided.")

    try:
        payload = decode_token(token)
        return payload["sub"], payload["email"]
    except jwt.ExpiredSignatureError:
        raise ValueError("Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise ValueError("Invalid token. Please log in again.")


#  Prompt (reuse logic from app.py) 

def _build_stream_prompt(marks: int, language: str) -> PromptTemplate:
    if marks <= 5:
        structure = "Concise 5-mark answer: Introduction, Key Points (3 bullets), Example, Conclusion."
    elif marks <= 10:
        structure = "Detailed 10-mark answer: Introduction, Main Explanation, 5 Key Points, Advantages/Applications, Conclusion."
    else:
        structure = "Comprehensive 15-mark answer: Introduction, Deep Explanation, 6 Key Points, Types/Categories, Advantages, Disadvantages, Real-World Applications, Conclusion."

    template = f"""You are ScholAI, an expert academic assistant.
Answer the student's question using ONLY the context below.
Write a well-structured {marks}-mark exam answer in {language}.
{structure}
Use **bold** for technical terms. Use "- " for bullet points.
If the context lacks the information, say: "The uploaded document does not contain sufficient information on this topic."

Context:
{{context}}

Question: {{question}}

Answer:"""
    return PromptTemplate(input_variables=["context", "question"], template=template)


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(d.page_content for d in docs)


#  Streaming route 

@streaming_ask.route("/ask/stream", methods=["POST", "OPTIONS"])
def ask_stream():
    """
    POST /ask/stream
    Body: {"question": "...", "marks": 10, "language": "English"}
    Returns: SSE stream

    The frontend should open this with fetch() + ReadableStream.
    """
    if request.method == "OPTIONS":
        response = Response("", status=200)
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        return response

    # ── Auth ──
    try:
        user_id, user_email = _resolve_user_from_request()
    except ValueError as e:
        def _auth_error():
            yield _sse_error(str(e))
            yield _sse_done()
        return Response(
            stream_with_context(_auth_error()),
            mimetype="text/event-stream"
        )

    #  Parse body 
    data     = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    marks    = int(data.get("marks", 10))
    language = data.get("language", "English").strip()

    if marks not in (5, 10, 15):
        marks = 10

    def _generate():
        if not question:
            yield _sse_error("Question cannot be empty.")
            yield _sse_done()
            return

        # ── Semantic cache check ──
        cache = SemanticCache(user_id=user_id, marks=marks)
        cached_answer = cache.get(question)
        if cached_answer:
            yield _sse_meta({"marks": marks, "cached": True})
            # Stream cached answer char-by-char (feels live, not jarring)
            chunk_size = 4
            for i in range(0, len(cached_answer), chunk_size):
                yield _sse(cached_answer[i:i + chunk_size])
            yield _sse_done()
            return

        # ── Query rewriting ──
        try:
            llm = get_llm()
            search_query = rewrite_query(question, llm)
        except Exception as e:
            yield _sse_error(f"LLM initialisation failed: {e}")
            yield _sse_done()
            return

        # ── Retrieval ──
        try:
            retriever = get_hybrid_retriever(user_id=user_id, k=5)
            docs      = retriever.invoke(search_query)
        except Exception as e:
            yield _sse_error(str(e))
            yield _sse_done()
            return

        if not docs:
            yield _sse("No relevant content found. Please upload a relevant document first.")
            yield _sse_done()
            return

        # ── Stream LLM response ──
        context = _format_docs(docs)
        prompt  = _build_stream_prompt(marks=marks, language=language)
        chain   = prompt | llm

        full_answer = []
        yield _sse_meta({"marks": marks, "cached": False})

        try:
            for chunk in chain.stream({"context": context, "question": question}):
                # LangChain streaming chunks have .content for chat models
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_answer.append(token)
                    yield _sse(token)

        except Exception as e:
            logger.error("Streaming LLM error for user %d: %s", user_id, e)
            yield _sse_error(f"Generation interrupted: {e}")
            yield _sse_done()
            return

        # ── Persist to history + cache ──
        answer_text = "".join(full_answer)
        try:
            execute(
                """INSERT INTO query_history (user_id, question, answer, marks, language)
                   VALUES (?, ?, ?, ?, ?)""",
                (user_id, question, answer_text, marks, language)
            )
            cache.set(question, answer_text)
        except Exception as e:
            logger.warning("Failed to persist streamed answer: %s", e)

        yield _sse_done()
        logger.info("Stream complete for user %d: %s", user_id, question[:60])

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control":                "no-cache",
            "X-Accel-Buffering":            "no",   
            "Access-Control-Allow-Origin":  "*",
            "Access-Control-Allow-Headers": "Content-Type, Authorization",
        }
    )

