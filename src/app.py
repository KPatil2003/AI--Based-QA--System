import os
import functools
from flask import Flask, request, jsonify, render_template, g
from werkzeug.utils import secure_filename
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import jwt

from src.core.process        import process_uploaded_document, process_uploaded_pdf
from src.advance_rag.hybrid_search  import get_hybrid_retriever         
from src.core.llm            import get_llm, warmup
from src.core.auth           import register_user, login_user, decode_token
from src.advance_rag.semantic_cache import SemanticCache                  
from src.advance_rag.query_rewriting import rewrite_query, hyde_query    
from src.advance_rag.streaming      import streaming_ask                 
from src.models.database     import init_db, execute, fetch_all, fetch_one
from src.common.logger       import get_logger
from src.common.custom_exception import CustomException

app    = Flask(__name__)
logger = get_logger(__name__)

UPLOAD_FOLDER      = "data/uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "doc", "txt", "md", "png", "jpg", "jpeg", "tiff"}  
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"]      = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   

init_db()
warmup()   

#  Register streaming blueprint 
app.register_blueprint(streaming_ask)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,DELETE,OPTIONS"
    return response

@app.route("/upload",                  methods=["OPTIONS"])
@app.route("/ask",                     methods=["OPTIONS"])
@app.route("/ask/stream",              methods=["OPTIONS"])  # ← NEW
@app.route("/status",                  methods=["OPTIONS"])
@app.route("/auth/register",           methods=["OPTIONS"])
@app.route("/auth/login",              methods=["OPTIONS"])
@app.route("/auth/me",                 methods=["OPTIONS"])
@app.route("/history/queries",         methods=["OPTIONS"])
@app.route("/history/uploads",         methods=["OPTIONS"])
@app.route("/history/queries/<int:qid>", methods=["OPTIONS"])
def handle_options(**kwargs):
    return jsonify({}), 200


def require_auth(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authentication required. Please log in."}), 401
        token = auth_header.split(" ", 1)[1]
        try:
            payload      = decode_token(token)
            g.user_id    = payload["sub"]
            g.user_email = payload["email"]
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Session expired. Please log in again."}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token. Please log in again."}), 401
        return f(*args, **kwargs)
    return decorated


#  Auth routes (unchanged) 

@app.route("/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(silent=True) or {}
    result = register_user(
        name=data.get("name", ""), email=data.get("email", ""), password=data.get("password", "")
    )
    if result["ok"]:
        return jsonify(result), 201
    return jsonify({"error": result["error"]}), 400


@app.route("/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or {}
    result = login_user(email=data.get("email", ""), password=data.get("password", ""))
    if result["ok"]:
        return jsonify(result), 200
    return jsonify({"error": result["error"]}), 401


@app.route("/auth/me", methods=["GET"])
@require_auth
def auth_me():
    row = fetch_one("SELECT id, name, email, created_at FROM users WHERE id = ?", (g.user_id,))
    if not row:
        return jsonify({"error": "User not found."}), 404
    upload_count  = fetch_one("SELECT COUNT(*) AS c FROM upload_history WHERE user_id = ?", (g.user_id,))
    query_count   = fetch_one("SELECT COUNT(*) AS c FROM query_history  WHERE user_id = ?", (g.user_id,))
    total_bytes   = fetch_one("SELECT SUM(file_size) AS b FROM upload_history WHERE user_id = ?", (g.user_id,))
    last_query    = fetch_one("SELECT asked_at FROM query_history WHERE user_id = ? ORDER BY asked_at DESC LIMIT 1", (g.user_id,))
    recent_uploads = fetch_all(
        "SELECT filename, file_size, chunks, uploaded_at FROM upload_history WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 5",
        (g.user_id,)
    )
    return jsonify({
        "id":             row["id"],
        "name":           row["name"],
        "email":          row["email"],
        "created_at":     row["created_at"],
        "upload_count":   upload_count["c"]   if upload_count  else 0,
        "query_count":    query_count["c"]    if query_count   else 0,
        "total_bytes":    total_bytes["b"]    if total_bytes and total_bytes["b"] else 0,
        "last_active":    last_query["asked_at"] if last_query else None,
        "recent_uploads": [dict(r) for r in recent_uploads],
    }), 200


@app.route("/history/uploads", methods=["GET"])
@require_auth
def history_uploads():
    rows = fetch_all(
        "SELECT id, filename, file_size, chunks, uploaded_at FROM upload_history WHERE user_id = ? ORDER BY uploaded_at DESC LIMIT 50",
        (g.user_id,)
    )
    return jsonify([dict(r) for r in rows]), 200


@app.route("/history/queries", methods=["GET"])
@require_auth
def history_queries():
    rows = fetch_all(
        "SELECT id, question, answer, marks, language, asked_at FROM query_history WHERE user_id = ? ORDER BY asked_at DESC LIMIT 100",
        (g.user_id,)
    )
    return jsonify([dict(r) for r in rows]), 200


@app.route("/history/queries/<int:qid>", methods=["DELETE"])
@require_auth
def delete_query(qid: int):
    execute("DELETE FROM query_history WHERE id = ? AND user_id = ?", (qid, g.user_id))
    return jsonify({"deleted": qid}), 200


#  Casual question detection (unchanged) 

CASUAL_PATTERNS = [
    r"^(hi|hello|hey|hiya|howdy|yo)\b",
    r"^good\s+(morning|evening|afternoon|night)\b",
    r"^how are you",
    r"^what('s| is) up",
    r"^who are you",
    r"^what can you do",
    r"^thank(s| you)",
    r"^(bye|goodbye|see you)",
    r"^help$",
    r"^\?+$",
]

def is_casual_question(question: str) -> bool:
    import re
    q = question.strip().lower()
    return any(re.match(p, q) for p in CASUAL_PATTERNS)

CASUAL_PROMPT_TEMPLATE = """You are ScholAI, a friendly and helpful academic assistant.
The student has sent you a casual or conversational message — NOT an academic question.

Respond warmly and helpfully in {language}. Keep your reply short (2-4 sentences).
- Introduce yourself briefly if it's a greeting.
- Remind them you're here to help with exam questions from their uploaded documents.
- Be encouraging and friendly.

Student message: {{question}}

Your friendly response:
"""

def build_casual_prompt(language: str = "English") -> PromptTemplate:
    template = CASUAL_PROMPT_TEMPLATE.format(language=language)
    return PromptTemplate(input_variables=["question"], template=template)


#  Main exam prompt (unchanged) 

def build_prompt(marks: int, language: str = "English") -> PromptTemplate:
    if marks <= 5:
        structure = """
Structure your answer as a 5-mark response with these clearly labelled sections:
1. Introduction:        Brief definition (1-2 lines)
2. Key Explanation:     Core concept explained clearly (3-4 lines)
3. Key Points:
   - Point 1
   - Point 2
   - Point 3
4. Example:             One concrete real-world example (1-2 lines)
5. Conclusion:          One-line summary
"""
    elif marks <= 10:
        structure = """
Structure your answer as a 10-mark response with these clearly labelled sections:
1. Introduction:        Define the topic clearly (2-3 lines)
2. Main Explanation:    Explain in depth with concepts (4-5 lines)
3. Key Points:
   - Point 1
   - Point 2
   - Point 3
   - Point 4
   - Point 5
4. Advantages/Applications:
   - Advantage/Use 1
   - Advantage/Use 2
   - Advantage/Use 3
5. Conclusion:          Summarise in 2-3 lines
"""
    else:
        structure = """
Structure your answer as a 15-mark response with these clearly labelled sections:
1. Introduction:        Thorough definition and context (3-4 lines)
2. Main Explanation:    Deep explanation with theory (5-6 lines)
3. Key Points:
   - Point 1 - Point 2 - Point 3 - Point 4 - Point 5 - Point 6
4. Types/Categories:    List each type with 1-line description (if applicable)
5. Advantages / Disadvantages
6. Real-World Applications
7. Conclusion:          Comprehensive summary (3-4 lines)
"""

    template = f"""You are an expert academic assistant helping students prepare for university exams.
Answer the question below using ONLY the context extracted from the student's uploaded document.
Write a detailed, well-structured {marks}-mark exam answer in {language}.

{structure}

Important rules:
- Use ONLY the information provided in the context below.
- If the context does not cover the topic, reply: "The uploaded document does not contain sufficient information on this topic."
- Write in clear, formal academic language.
- Do NOT invent or assume any facts not present in the context.
- Label EVERY section heading clearly followed by a colon.
- Use "- " prefix for ALL bullet points consistently.
- Bold any technical terms by wrapping them like **term**.

Context extracted from document:
{{context}}

Student's Question: {{question}}

Your {marks}-mark answer:
"""
    return PromptTemplate(input_variables=["context", "question"], template=template)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def format_docs(docs) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


#  Routes 

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
@require_auth
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(file.filename):
        exts = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return jsonify({"error": f"Unsupported file type. Allowed: {exts}"}), 400

    try:
        filename  = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)
        logger.info("File saved: %s (user_id=%d)", save_path, g.user_id)

        result = process_uploaded_document(filename=filename, user_id=g.user_id)
        logger.info("Pipeline result: %s", result)

        execute(
            "INSERT INTO upload_history (user_id, filename, file_size, chunks) VALUES (?, ?, ?, ?)",
            (g.user_id, filename, os.path.getsize(save_path), result.get("chunks", 0))
        )

        #  Invalidate semantic cache when new doc is uploaded 
        SemanticCache(user_id=g.user_id).invalidate()

        return jsonify({
            "status":   "success",
            "message":  f"'{filename}' processed successfully.",
            "chunks":   result.get("chunks", 0),
            "filename": filename
        }), 200

    except CustomException as ce:
        logger.error("Upload pipeline error: %s", str(ce))
        return jsonify({"error": str(ce)}), 500
    except Exception as e:
        logger.error("Unexpected upload error: %s", str(e))
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500


@app.route("/ask", methods=["POST"])
@require_auth
def ask():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    question = data.get("question", "").strip()
    marks    = int(data.get("marks", 10))
    language = data.get("language", "English").strip()

    if not question:
        return jsonify({"error": "Question cannot be empty"}), 400
    if marks not in (5, 10, 15):
        marks = 10

    try:
        llm = get_llm()

        #  Casual question 
        if is_casual_question(question):
            casual_prompt = build_casual_prompt(language=language)
            answer = (casual_prompt | llm | StrOutputParser()).invoke({"question": question})
            execute(
                "INSERT INTO query_history (user_id, question, answer, marks, language) VALUES (?, ?, ?, ?, ?)",
                (g.user_id, question, answer, marks, language)
            )
            return jsonify({"answer": answer, "marks": marks, "casual": True}), 200

        #  Semantic cache check 
        cache = SemanticCache(user_id=g.user_id, marks=marks)
        cached_answer = cache.get(question)
        if cached_answer:
            logger.info("Serving cached answer for user %d: %s", g.user_id, question[:60])
            return jsonify({"answer": cached_answer, "marks": marks, "cached": True}), 200

        # ── Query rewriting ──  ← NEW
        search_query = rewrite_query(question, llm)
        logger.info("Rewritten query: %s → %s", question[:50], search_query[:50])

        # ── Hybrid retrieval ──  ← NEW (was: get_retriever)
        retriever = get_hybrid_retriever(user_id=g.user_id, k=5)
        relevant_docs = retriever.invoke(search_query)

        if not relevant_docs:
            return jsonify({"answer": "No relevant content found. Please upload a relevant document first."}), 200

        context = format_docs(relevant_docs)
        prompt  = build_prompt(marks=marks, language=language)
        answer  = (prompt | llm | StrOutputParser()).invoke({
            "context":  context,
            "question": question
        })

        execute(
            "INSERT INTO query_history (user_id, question, answer, marks, language) VALUES (?, ?, ?, ?, ?)",
            (g.user_id, question, answer, marks, language)
        )

        # ── Store in semantic cache ──  ← NEW
        cache.set(question, answer)

        logger.info("Answer generated for user %d: %s", g.user_id, question[:60])
        return jsonify({"answer": answer, "marks": marks, "rewritten_query": search_query}), 200

    except CustomException as ce:
        logger.error("Ask error: %s", str(ce))
        return jsonify({"error": str(ce)}), 500
    except Exception as e:
        logger.error("Unexpected ask error: %s", str(e))
        return jsonify({"error": f"Answer generation failed: {str(e)}"}), 500


@app.route("/status", methods=["GET"])
@require_auth
def status():
    path  = f"vectorstore/user_{g.user_id}/faiss_db"
    ready = (
        os.path.exists(f"{path}/index.faiss") and
        os.path.exists(f"{path}/index.pkl")
    )
    return jsonify({"status": "running", "vectorstore_ready": ready}), 200


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)