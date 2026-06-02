import re
from typing import Optional

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.common.logger import get_logger
from src.common.custom_exception import CustomException

logger = get_logger(__name__)


# ── Prompt templates ──────────────────────────────────────────────────────────

REWRITE_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are a search query optimiser for an academic document retrieval system.

Given a student's exam question, rewrite it as a SHORT, keyword-rich search query (max 20 words).

Rules:
- Remove phrases like "explain", "describe", "with example", "in detail", "10 marks", "15 marks"
- Keep all technical terms, subject names, and key concepts
- Add 2-3 closely related synonyms or related terms in parentheses if helpful
- Output ONLY the rewritten query — no explanation, no numbering, no quotation marks

Student's question: {question}

Rewritten search query:"""
)

HYDE_PROMPT = PromptTemplate(
    input_variables=["question", "marks"],
    template="""You are an expert academic assistant.

Write a SHORT ({marks}-mark level) answer snippet (4-6 sentences) for the question below.
This snippet will be used as a search query to find relevant document sections.

Rules:
- Use precise technical language and key terms
- Cover the main concepts the question asks about
- Do NOT write a full answer — just enough to surface the right document sections
- Output ONLY the snippet, no preamble

Question: {question}

Snippet:"""
)

MULTI_QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an academic search expert.

Generate 3 different search queries that would help retrieve relevant content for this exam question.
Each query should approach the topic from a slightly different angle.

Rules:
- One query per line
- No numbering, no bullet points, no extra text
- Keep each query under 15 words

Question: {question}

Three search queries:"""
)


# ── Cleaning helper ───────────────────────────────────────────────────────────

# Patterns to strip from student questions before rewriting
_NOISE_PATTERNS = [
    r"\b(explain|describe|discuss|define|elaborate|write about|write a note on)\b",
    r"\b(in detail|with (an )?example[s]?|briefly|clearly|concisely)\b",
    r"\b(\d+[\s-]*marks?)\b",
    r"\b(short|long|detailed)\s+answer\b",
    r"\b(q\d+|question\s*\d+)\b",
    r"[^\w\s]",          # punctuation (except what we add back)
]

def _clean_question(question: str) -> str:
    """Strip exam-specific noise from a question for direct search use."""
    q = question.lower()
    for pattern in _NOISE_PATTERNS:
        q = re.sub(pattern, " ", q, flags=re.IGNORECASE)
    # Collapse whitespace
    return " ".join(q.split()).strip()


# ── Public API ────────────────────────────────────────────────────────────────

def rewrite_query(question: str, llm) -> str:

    try:
        chain = REWRITE_PROMPT | llm | StrOutputParser()
        rewritten = chain.invoke({"question": question}).strip()

        # Sanity check: reject if output is too long (LLM went off-rails)
        # Raised from 30 → 40 words to allow richer rewrites
        if len(rewritten.split()) > 40:
            logger.warning(
                "Rewrite output too long (%d words) — using cleaned original. Output was: %r",
                len(rewritten.split()), rewritten[:120]
            )
            return _clean_question(question)

        # Reject if output looks like the raw question came back unchanged
        if rewritten.lower() == question.lower().strip():
            logger.warning("LLM returned input unchanged — falling back to cleaned query")
            return _clean_question(question)

        logger.info("Query rewritten: '%s' → '%s'", question[:60], rewritten[:60])
        return rewritten

    except Exception as e:
        logger.error(          # upgraded from warning → error so it's visible in logs
            "Query rewrite FAILED: %s | type=%s | question=%r",
            e, type(e).__name__, question[:80]
        )
        return _clean_question(question)


def hyde_query(question: str, marks: int, llm) -> str:

    try:
        chain = HYDE_PROMPT | llm | StrOutputParser()
        snippet = chain.invoke({"question": question, "marks": marks}).strip()

        logger.info(
            "HyDE snippet generated for: '%s' (%d words)",
            question[:60], len(snippet.split())
        )
        return snippet

    except Exception as e:
        logger.warning("HyDE generation failed (non-fatal): %s — falling back to rewrite", e)
        return rewrite_query(question, llm)


def get_multi_queries(question: str, llm) -> list[str]:

    try:
        chain = MULTI_QUERY_PROMPT | llm | StrOutputParser()
        output = chain.invoke({"question": question}).strip()

        queries = [
            line.strip()
            for line in output.splitlines()
            if line.strip() and len(line.strip()) > 5
        ][:3]   # cap at 3

        if not queries:
            raise ValueError("LLM returned no usable queries")

        logger.info("Multi-query expansion: %d variants for: %s", len(queries), question[:50])
        return queries

    except Exception as e:
        logger.warning("Multi-query generation failed: %s", e)
        return [_clean_question(question)]


def rerank_by_relevance(question: str, docs: list, llm, top_k: int = 5) -> list:

    if not docs:
        return docs

    if len(docs) <= top_k:
        return docs   # No need to rerank if we have fewer docs than top_k

    try:
        numbered_passages = "\n\n".join(
            f"[{i+1}] {doc.page_content[:300]}"
            for i, doc in enumerate(docs)
        )

        rerank_prompt = PromptTemplate(
            input_variables=["question", "passages"],
            template="""You are a relevance judge for academic content retrieval.

Question: {question}

Below are {n} retrieved passages numbered [1] to [{n}].
Return ONLY the numbers of the top {k} most relevant passages, comma-separated.
Example output: 2, 5, 1

Passages:
{passages}

Top {k} passage numbers:"""
        )

        chain = rerank_prompt | llm | StrOutputParser()
        result = chain.invoke({
            "question": question,
            "passages": numbered_passages,
            "n": len(docs),
            "k": top_k
        }).strip()

        # Parse "2, 5, 1" → [1, 4, 0] (0-indexed)
        indices = []
        for part in result.replace(" ", "").split(","):
            try:
                idx = int(part) - 1   # convert to 0-indexed
                if 0 <= idx < len(docs):
                    indices.append(idx)
            except ValueError:
                continue

        if not indices:
            logger.warning("Reranker returned unparseable output: %s", result)
            return docs[:top_k]

        reranked = [docs[i] for i in indices[:top_k]]
        logger.info("Reranked %d → %d docs for: %s", len(docs), len(reranked), question[:50])
        return reranked

    except Exception as e:
        logger.warning("Reranking failed (non-fatal): %s", e)
        return docs[:top_k]