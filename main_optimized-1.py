import os
import time
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from jose import jwt, JWTError
from passlib.context import CryptContext
from openai import OpenAI

# ============================================================
# JUVENS AI — optimized backend
# - Gemini 3.8 Flash
# - Low thinking for faster chat
# - Shorter conversation context
# - Safe plain-string message format
# - Timing information in /chat response
# ============================================================

APP_TITLE = "Juvens AI API"
VERSION = "3.1.0"

DATABASE_PATH = os.getenv("DATABASE_PATH", "./juvens_ai.db")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-secret")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")

# Use the current stable Gemini 3.8 Flash model unless explicitly overridden.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"

# low = lower latency for normal chat
THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
if THINKING_LEVEL not in {"low", "medium", "high"}:
    THINKING_LEVEL = "low"

# Keep the prompt/history small to reduce latency and token use.
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "12"))
MAX_REPLY_CHARS = int(os.getenv("MAX_REPLY_CHARS", "8000"))

app = FastAPI(title=APP_TITLE, version=VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if FRONTEND_ORIGIN == "*" else [FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

client: Optional[OpenAI] = None
if GEMINI_API_KEY:
    client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

SYSTEM_PROMPT = """
Tu es Juvens AI, un assistant IA créé par Juvens.
Réponds de façon claire, utile et naturelle.
Tu peux répondre en français, créole haïtien ou anglais selon la langue de l'utilisateur.
Pour les questions liées au commerce, à la comptabilité et aux services en Haïti,
adapte les explications au contexte haïtien lorsque c'est pertinent.
Pour une question simple, donne une réponse directe et concise.
""".strip()


# ============================================================
# DATABASE
# ============================================================

def db():
    conn = sqlite3.connect(DATABASE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            title TEXT NOT NULL DEFAULT 'Nouvelle conversation',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            filename TEXT NOT NULL,
            content_type TEXT,
            size INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            type TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS invoices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            client_name TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS expenses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            description TEXT NOT NULL,
            amount REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# MODELS
# ============================================================

class RegisterIn(BaseModel):
    email: str
    name: str
    password: str = Field(min_length=6, max_length=200)


class LoginIn(BaseModel):
    email: str
    password: str


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=30000)
    conversation_id: Optional[int] = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class ConversationIn(BaseModel):
    title: str = Field(default="Nouvelle conversation", min_length=1, max_length=200)


class TransactionIn(BaseModel):
    description: str
    amount: float
    type: str = "expense"


class InvoiceIn(BaseModel):
    client_name: str
    amount: float
    status: str = "pending"


class ExpenseIn(BaseModel):
    description: str
    amount: float


# ============================================================
# AUTH
# ============================================================

def make_token(user_id: int) -> str:
    payload = {
        "sub": str(user_id),
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def get_optional_user(request: Request) -> Optional[int]:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None

    token = auth[7:].strip()
    if not token:
        return None

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return int(payload["sub"])
    except (JWTError, ValueError, KeyError, TypeError):
        return None


def require_user(request: Request) -> int:
    user_id = get_optional_user(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentification requise")
    return user_id


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Juvens AI",
        "status": "ok",
        "version": VERSION,
        "model": MODEL,
        "thinking_level": THINKING_LEVEL,
        "ai_active": bool(client),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai_active": bool(client),
        "gemini_key_detected": bool(GEMINI_API_KEY),
        "model": MODEL,
        "thinking_level": THINKING_LEVEL,
    }


# ============================================================
# AUTH ROUTES
# ============================================================

@app.post("/auth/register")
def register(data: RegisterIn):
    conn = db()
    try:
        password_hash = pwd_context.hash(data.password)
        now = datetime.now(timezone.utc).isoformat()

        cur = conn.execute(
            "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (data.email.lower().strip(), data.name.strip(), password_hash, now),
        )
        conn.commit()

        user_id = cur.lastrowid
        return {"token": make_token(user_id), "user": {"id": user_id, "email": data.email, "name": data.name}}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Cet email existe déjà")
    finally:
        conn.close()


@app.post("/auth/login")
def login(data: LoginIn):
    conn = db()
    row = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (data.email.lower().strip(),),
    ).fetchone()
    conn.close()

    if not row or not pwd_context.verify(data.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

    return {
        "token": make_token(row["id"]),
        "user": {"id": row["id"], "email": row["email"], "name": row["name"]},
    }


# ============================================================
# CONVERSATIONS
# ============================================================

@app.post("/conversations")
def create_conversation(data: ConversationIn, request: Request):
    user_id = get_optional_user(request)
    now = datetime.now(timezone.utc).isoformat()

    conn = db()
    cur = conn.execute(
        "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (user_id, data.title, now, now),
    )
    conn.commit()
    cid = cur.lastrowid
    conn.close()

    return {"id": cid, "title": data.title}


@app.get("/conversations")
def list_conversations(request: Request):
    user_id = require_user(request)
    conn = db()
    rows = conn.execute(
        "SELECT id, title, created_at, updated_at FROM conversations "
        "WHERE user_id = ? ORDER BY updated_at DESC",
        (user_id,),
    ).fetchall()
    conn.close()

    return [dict(row) for row in rows]


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: int, request: Request):
    user_id = require_user(request)
    conn = db()

    row = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
        (conversation_id, user_id),
    ).fetchone()

    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Conversation introuvable")

    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()
    conn.close()

    return {"status": "deleted"}


# ============================================================
# CHAT
# ============================================================

def load_history(conversation_id: int, user_id: Optional[int]):
    conn = db()

    row = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
        (conversation_id, user_id),
    ).fetchone()

    if not row:
        conn.close()
        return []

    rows = conn.execute(
        "SELECT role, content FROM messages "
        "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?",
        (conversation_id, MAX_HISTORY_MESSAGES),
    ).fetchall()
    conn.close()

    # Reverse so oldest selected message comes first.
    return [{"role": r["role"], "content": str(r["content"])} for r in reversed(rows)]


@app.post("/chat")
def chat(data: ChatIn, request: Request):
    started = time.perf_counter()

    if not client:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY n'est pas configurée sur Render."
        )

    user_id = get_optional_user(request)
    message = data.message.strip()

    if not message:
        raise HTTPException(status_code=422, detail="Le message est vide.")

    # Create conversation if needed.
    cid = data.conversation_id

    if cid is None:
        now = datetime.now(timezone.utc).isoformat()
        conn = db()
        cur = conn.execute(
            "INSERT INTO conversations (user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (user_id, message[:80], now, now),
        )
        conn.commit()
        cid = cur.lastrowid
        conn.close()
    else:
        # Verify conversation exists and belongs to this user (or is public/anonymous).
        conn = db()
        row = conn.execute(
            "SELECT id FROM conversations WHERE id = ? AND (user_id = ? OR user_id IS NULL)",
            (cid, user_id),
        ).fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Conversation introuvable.")

    history = load_history(cid, user_id)

    # IMPORTANT: all message content is plain strings.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    try:
        # OpenAI-compatible Gemini endpoint.
        # thinking_config is passed through extra_body so the Gemini
        # OpenAI-compatible API can use a lower reasoning level for latency.
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            extra_body={
                "thinking_config": {
                    "thinking_level": THINKING_LEVEL
                }
            },
        )

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            answer = "Je n'ai pas pu générer une réponse."

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur du service IA : {str(exc)}"
        )

    if len(answer) > MAX_REPLY_CHARS:
        answer = answer[:MAX_REPLY_CHARS].rstrip() + "…"

    now = datetime.now(timezone.utc).isoformat()
    conn = db()

    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (cid, "user", message, now),
    )
    conn.execute(
        "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
        (cid, "assistant", answer, now),
    )
    conn.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, cid),
    )

    conn.commit()
    conn.close()

    latency_ms = round((time.perf_counter() - started) * 1000)

    return {
        "conversation_id": cid,
        "reply": answer,
        "model": MODEL,
        "thinking_level": THINKING_LEVEL,
        "latency_ms": latency_ms,
    }


# ============================================================
# FILE UPLOAD
# ============================================================

@app.post("/files/upload")
async def upload_file(request: Request, file: UploadFile = File(...)):
    user_id = get_optional_user(request)

    content = await file.read()
    max_size = 10 * 1024 * 1024

    if len(content) > max_size:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux. Maximum: 10 MB.")

    now = datetime.now(timezone.utc).isoformat()
    conn = db()

    cur = conn.execute(
        "INSERT INTO files (user_id, filename, content_type, size, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, file.filename or "file", file.content_type, len(content), now),
    )
    conn.commit()
    file_id = cur.lastrowid
    conn.close()

    return {
        "id": file_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "size": len(content),
    }


# ============================================================
# ACCOUNTING
# ============================================================

@app.post("/accounting/transactions")
def add_transaction(data: TransactionIn, request: Request):
    user_id = require_user(request)
    now = datetime.now(timezone.utc).isoformat()

    conn = db()
    cur = conn.execute(
        "INSERT INTO transactions (user_id, description, amount, type, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, data.description, data.amount, data.type, now),
    )
    conn.commit()
    tid = cur.lastrowid
    conn.close()

    return {"id": tid, "status": "created"}


@app.post("/accounting/invoices")
def add_invoice(data: InvoiceIn, request: Request):
    user_id = require_user(request)
    now = datetime.now(timezone.utc).isoformat()

    conn = db()
    cur = conn.execute(
        "INSERT INTO invoices (user_id, client_name, amount, status, created_at) VALUES (?, ?, ?, ?, ?)",
        (user_id, data.client_name, data.amount, data.status, now),
    )
    conn.commit()
    iid = cur.lastrowid
    conn.close()

    return {"id": iid, "status": "created"}


@app.post("/accounting/expenses")
def add_expense(data: ExpenseIn, request: Request):
    user_id = require_user(request)
    now = datetime.now(timezone.utc).isoformat()

    conn = db()
    cur = conn.execute(
        "INSERT INTO expenses (user_id, description, amount, created_at) VALUES (?, ?, ?, ?)",
        (user_id, data.description, data.amount, now),
    )
    conn.commit()
    eid = cur.lastrowid
    conn.close()

    return {"id": eid, "status": "created"}


@app.get("/accounting/summary")
def accounting_summary(request: Request):
    user_id = require_user(request)
    conn = db()

    income = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS total FROM transactions WHERE user_id = ? AND type = 'income'",
        (user_id,),
    ).fetchone()["total"]

    expenses = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS total FROM expenses WHERE user_id = ?",
        (user_id,),
    ).fetchone()["total"]

    invoices = conn.execute(
        "SELECT COALESCE(SUM(amount), 0) AS total FROM invoices WHERE user_id = ?",
        (user_id,),
    ).fetchone()["total"]

    conn.close()

    return {
        "income": income,
        "expenses": expenses,
        "invoices": invoices,
        "balance": income - expenses,
    }


@app.get("/accounting/export.csv")
def accounting_export(request: Request):
    user_id = require_user(request)
    conn = db()

    rows = conn.execute(
        "SELECT id, description, amount, type, created_at "
        "FROM transactions WHERE user_id = ? ORDER BY id DESC",
        (user_id,),
    ).fetchall()
    conn.close()

    def generate():
        yield "id,description,amount,type,created_at\n"
        for r in rows:
            description = str(r["description"]).replace('"', '""')
            yield f'{r["id"]},"{description}",{r["amount"]},{r["type"]},{r["created_at"]}\n'

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=juvens-ai-accounting.csv"},
    )
