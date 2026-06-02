import sqlite3
import os

DB_PATH = os.environ.get("SCHOLAI_DB", "data/scholai.db")

def run():
    if not os.path.exists(DB_PATH):
        print(f"[!] Database not found at {DB_PATH}")
        print("    Start Flask once to auto-create it.")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    print("\n" + "═"*60)
    print("  ScholAI Database Inspector")
    print("═"*60)

    # Users
    users = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    print(f"\n👤 USERS ({len(users)} total)")
    print("-"*60)
    for u in users:
        print(f"  ID {u['id']:>3} | {u['name']:<20} | {u['email']:<30} | joined {u['created_at'][:10]}")

    # Uploads
    uploads = conn.execute(
        "SELECT uh.*, u.name FROM upload_history uh JOIN users u ON u.id=uh.user_id ORDER BY uploaded_at DESC LIMIT 20"
    ).fetchall()
    print(f"\n📄 UPLOADS — last {len(uploads)}")
    print("-"*60)
    for r in uploads:
        size_kb = f"{r['file_size']/1024:.1f} KB" if r['file_size'] else "—"
        print(f"  [{r['id']:>3}] {r['name']:<15} | {r['filename']:<30} | {r['chunks'] or '—'} chunks | {size_kb} | {r['uploaded_at'][:16]}")

    # Queries
    queries = conn.execute(
        "SELECT qh.id, qh.question, qh.marks, qh.language, qh.asked_at, u.name FROM query_history qh JOIN users u ON u.id=qh.user_id ORDER BY asked_at DESC LIMIT 20"
    ).fetchall()
    print(f"\n🔍 QUERIES — last {len(queries)}")
    print("-"*60)
    for q in queries:
        question_short = q['question'][:45] + '…' if len(q['question']) > 45 else q['question']
        print(f"  [{q['id']:>3}] {q['name']:<15} | {q['marks']}M {q['language']:<8} | {question_short}")

    print("\n" + "═"*60 + "\n")
    conn.close()

if __name__ == "__main__":
    run()