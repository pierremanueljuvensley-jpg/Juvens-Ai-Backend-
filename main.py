import os
import time
import sqlite3
import hashlib
import hmac
import base64
import json
from datetime import datetime, timedelta, timezone
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import OpenAI

# ============================================================
# JUVENS AI - MAIN BACKEND
# ============================================================
# Designed to start with:
#   uvicorn main:app --host 0.0.0.0 --port $PORT
#
# Required Render environment variable:
#   GEMINI_API_KEY = your Gemini API key
#
# Optional:
#   GEMINI_MODEL=gemini-3.8-flash
#   GEMINI_THINKING_LEVEL=low
#   JWT_SECRET=change-this
#   FRONTEND_ORIGIN=*
#   DATABASE_PATH=./juvens_ai.db
# ============================================================

APP_VERSION = "4.0.0"
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
THINKING_LEVEL = os.getenv("GEMINI_THINKING_LEVEL", "low").strip().lower()
if THINKING_LEVEL not in ("low", "medium", "high"):
    THINKING_LEVEL = "low"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
JWT_SECRET = os.getenv("JWT_SECRET", "juvens-ai-change-this-secret")
DATABASE_PATH = os.getenv("DATABASE_PATH", "./juvens_ai.db")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")

MAX_HISTORY = 10
MAX_UPLOAD = 10 * 1024 * 1024

SYSTEM_PROMPT = """
Tu es Juvens AI, un assistant IA créé par Juvens.
Réponds dans la langue utilisée par l'utilisateur: français, créole haïtien ou anglais.
Sois clair, utile, naturel et précis.
Pour les questions simples, réponds directement.
Pour la comptabilité, le commerce et les services en Haïti, adapte tes explications au contexte haïtien lorsque c'est pertinent.
N'invente pas de faits lorsque l'information est incertaine.
""".strip()

app = FastAPI(
    title="Juvens AI API",
    description="Backend officiel de Juvens AI",
    version=APP_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if FRONTEND_ORIGIN == "*" else [FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini OpenAI-compatible client.
ai_client = None
if GEMINI_API_KEY:
    ai_client = OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(DATABASE_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = get_db()
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
            status TEXT NOT NULL,
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


init_database()


# ============================================================
# SIMPLE AUTH - NO jose/passlib DEPENDENCIES
# ============================================================

def password_hash(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        120000,
    )
    return base64.b64encode(salt + digest).decode("ascii")


def password_verify(password: str, stored: str) -> bool:
    try:
        raw = base64.b64decode(stored.encode("ascii"))
        salt = raw[:16]
        expected = raw[16:]
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            120000,
        )
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def make_token(user_id: int) -> str:
    payload = {
        "sub": int(user_id),
        "exp": int(time.time()) + 7 * 24 * 60 * 60,
    }
    body = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode()
    ).decode().rstrip("=")

    signature = hmac.new(
        JWT_SECRET.encode(),
        body.encode(),
        hashlib.sha256,
    ).hexdigest()

    return body + "." + signature


def get_user_id(request: Request) -> Optional[int]:
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None

    token = header[7:].strip()
    parts = token.split(".")
    if len(parts) != 2:
        return None

    body, signature = parts

    expected = hmac.new(
        JWT_SECRET.encode(),
        body.encode(),
        hashlib.sha256,
    ).hexdigest()

    if not hmac.compare_digest(signature, expected):
        return None

    try:
        padded = body + "=" * (-len(body) % 4)
        payload = json.loads(
            base64.urlsafe_b64decode(padded.encode()).decode()
        )

        if int(payload["exp"]) < int(time.time()):
            return None

        return int(payload["sub"])
    except Exception:
        return None


def require_user(request: Request) -> int:
    user_id = get_user_id(request)
    if user_id is None:
        raise HTTPException(status_code=401, detail="Authentification requise.")
    return user_id


# ============================================================
# PYDANTIC MODELS
# ============================================================

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str = Field(min_length=6, max_length=200)


class LoginRequest(BaseModel):
    email: str
    password: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=30000)
    conversation_id: Optional[int] = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class ConversationRequest(BaseModel):
    title: str = Field(
        default="Nouvelle conversation",
        min_length=1,
        max_length=200,
    )


class TransactionRequest(BaseModel):
    description: str
    amount: float
    type: str = "expense"


class InvoiceRequest(BaseModel):
    client_name: str
    amount: float
    status: str = "pending"


class ExpenseRequest(BaseModel):
    description: str
    amount: float


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Juvens AI",
        "status": "ok",
        "version": APP_VERSION,
        "ai_active": bool(ai_client),
        "gemini_key_detected": bool(GEMINI_API_KEY),
        "model": MODEL,
        "thinking_level": THINKING_LEVEL,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai_active": bool(ai_client),
        "gemini_key_detected": bool(GEMINI_API_KEY),
        "model": MODEL,
        "thinking_level": THINKING_LEVEL,
    }


# ============================================================
# AUTH
# ============================================================

@app.post("/auth/register")
def register(data: RegisterRequest):
    email = data.email.strip().lower()
    name = data.name.strip()

    if not email or not name:
        raise HTTPException(status_code=400, detail="Nom et email requis.")

    conn = get_db()

    try:
        cur = conn.execute(
            """
            INSERT INTO users
            (email, name, password_hash, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                email,
                name,
                password_hash(data.password),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        raise HTTPException(
            status_code=409,
            detail="Cet email existe déjà.",
        )

    conn.close()

    return {
        "token": make_token(user_id),
        "user": {
            "id": user_id,
            "email": email,
            "name": name,
        },
    }


@app.post("/auth/login")
def login(data: LoginRequest):
    email = data.email.strip().lower()

    conn = get_db()
    user = conn.execute(
        "SELECT * FROM users WHERE email = ?",
        (email,),
    ).fetchone()
    conn.close()

    if not user or not password_verify(
        data.password,
        user["password_hash"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Email ou mot de passe incorrect.",
        )

    return {
        "token": make_token(user["id"]),
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
        },
    }


# ============================================================
# CONVERSATIONS
# ============================================================

@app.post("/conversations")
def create_conversation(
    data: ConversationRequest,
    request: Request,
):
    user_id = get_user_id(request)
    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()
    cur = conn.execute(
        """
        INSERT INTO conversations
        (user_id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, data.title.strip(), now, now),
    )
    conn.commit()
    conversation_id = cur.lastrowid
    conn.close()

    return {
        "id": conversation_id,
        "title": data.title.strip(),
    }


@app.get("/conversations")
def conversations(request: Request):
    user_id = require_user(request)

    conn = get_db()
    rows = conn.execute(
        """
        SELECT id, title, created_at, updated_at
        FROM conversations
        WHERE user_id = ?
        ORDER BY updated_at DESC
        """,
        (user_id,),
    ).fetchall()
    conn.close()

    return [dict(row) for row in rows]


@app.get("/conversations/{conversation_id}")
def conversation(
    conversation_id: int,
    request: Request,
):
    user_id = require_user(request)

    conn = get_db()

    conv = conn.execute(
        """
        SELECT id, title, created_at, updated_at
        FROM conversations
        WHERE id = ? AND user_id = ?
        """,
        (conversation_id, user_id),
    ).fetchone()

    if not conv:
        conn.close()
        raise HTTPException(
            status_code=404,
            detail="Conversation introuvable.",
        )

    rows = conn.execute(
        """
        SELECT id, role, content, created_at
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id ASC
        """,
        (conversation_id,),
    ).fetchall()

    conn.close()

    return {
        "conversation": dict(conv),
        "messages": [dict(row) for row in rows],
    }


@app.delete("/conversations/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    request: Request,
):
    user_id = require_user(request)

    conn = get_db()

    conv = conn.execute(
        """
        SELECT id FROM conversations
        WHERE id = ? AND user_id = ?
        """,
        (conversation_id, user_id),
    ).fetchone()

    if not conv:
        conn.close()
        raise HTTPException(
            status_code=404,
            detail="Conversation introuvable.",
        )

    conn.execute(
        "DELETE FROM messages WHERE conversation_id = ?",
        (conversation_id,),
    )
    conn.execute(
        "DELETE FROM conversations WHERE id = ?",
        (conversation_id,),
    )

    conn.commit()
    conn.close()

    return {"status": "deleted"}


# ============================================================
# CHAT
# ============================================================

def get_history(
    conversation_id: int,
    user_id: Optional[int],
):
    conn = get_db()

    rows = conn.execute(
        """
        SELECT role, content
        FROM messages
        WHERE conversation_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (conversation_id, MAX_HISTORY),
    ).fetchall()

    conn.close()

    return [
        {
            "role": str(row["role"]),
            "content": str(row["content"]),
        }
        for row in reversed(rows)
    ]


def conversation_belongs_to(
    conversation_id: int,
    user_id: Optional[int],
) -> bool:
    conn = get_db()

    row = conn.execute(
        """
        SELECT id FROM conversations
        WHERE id = ?
        AND (user_id = ? OR user_id IS NULL)
        """,
        (conversation_id, user_id),
    ).fetchone()

    conn.close()
    return row is not None


@app.post("/chat")
def chat(
    data: ChatRequest,
    request: Request,
):
    start = time.perf_counter()

    if not GEMINI_API_KEY or ai_client is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "GEMINI_API_KEY n'est pas configurée. "
                "Ajoute GEMINI_API_KEY dans Render > Environment."
            ),
        )

    message = data.message.strip()

    if not message:
        raise HTTPException(
            status_code=422,
            detail="Le message est vide.",
        )

    user_id = get_user_id(request)
    conversation_id = data.conversation_id

    # Create a conversation automatically for Swagger/frontend testing.
    if conversation_id is None:
        now = datetime.now(timezone.utc).isoformat()

        conn = get_db()
        cur = conn.execute(
            """
            INSERT INTO conversations
            (user_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, message[:80], now, now),
        )
        conn.commit()
        conversation_id = cur.lastrowid
        conn.close()

    elif not conversation_belongs_to(
        conversation_id,
        user_id,
    ):
        raise HTTPException(
            status_code=404,
            detail="Conversation introuvable.",
        )

    history = get_history(
        conversation_id,
        user_id,
    )

    # IMPORTANT:
    # content is always a plain string.
    # This avoids the previous Gemini 400 error.
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(history)

    messages.append(
        {
            "role": "user",
            "content": message,
        }
    )

    try:
        # Google officially supports OpenAI-compatible Chat Completions.
        # reasoning_effort="low" reduces reasoning latency.
        response = ai_client.chat.completions.create(
            model=MODEL,
            messages=messages,
            reasoning_effort=THINKING_LEVEL,
        )

        answer = response.choices[0].message.content

        if not isinstance(answer, str):
            answer = str(answer)

        answer = answer.strip()

        if not answer:
            answer = "Je n'ai pas reçu de réponse du modèle."

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Erreur du service IA : {str(exc)}",
        )

    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()

    conn.execute(
        """
        INSERT INTO messages
        (conversation_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            conversation_id,
            "user",
            message,
            now,
        ),
    )

    conn.execute(
        """
        INSERT INTO messages
        (conversation_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            conversation_id,
            "assistant",
            answer,
            now,
        ),
    )

    conn.execute(
        """
        UPDATE conversations
        SET updated_at = ?
        WHERE id = ?
        """,
        (now, conversation_id),
    )

    conn.commit()
    conn.close()

    latency_ms = round(
        (time.perf_counter() - start) * 1000
    )

    return {
        "conversation_id": conversation_id,
        "reply": answer,
        "model": MODEL,
        "thinking_level": THINKING_LEVEL,
        "latency_ms": latency_ms,
    }


# ============================================================
# FILE UPLOAD
# ============================================================

@app.post("/files/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
):
    user_id = get_user_id(request)

    content = await file.read()

    if len(content) > MAX_UPLOAD:
        raise HTTPException(
            status_code=413,
            detail="Fichier trop volumineux. Maximum: 10 MB.",
        )

    now = datetime.now(timezone.utc).isoformat()

    conn = get_db()

    cur = conn.execute(
        """
        INSERT INTO files
        (user_id, filename, content_type, size, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            file.filename or "file",
            file.content_type,
            len(content),
            now,
        ),
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
def add_transaction(
    data: TransactionRequest,
    request: Request,
):
    user_id = require_user(request)

    conn = get_db()

    cur = conn.execute(
        """
        INSERT INTO transactions
        (user_id, description, amount, type, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            data.description,
            data.amount,
            data.type,
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    conn.commit()
    transaction_id = cur.lastrowid
    conn.close()

    return {
        "id": transaction_id,
        "status": "created",
    }


@app.post("/accounting/invoices")
def add_invoice(
    data: InvoiceRequest,
    request: Request,
):
    user_id = require_user(request)

    conn = get_db()

    cur = conn.execute(
        """
        INSERT INTO invoices
        (user_id, client_name, amount, status, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            user_id,
            data.client_name,
            data.amount,
            data.status,
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    conn.commit()
    invoice_id = cur.lastrowid
    conn.close()

    return {
        "id": invoice_id,
        "status": "created",
    }


@app.post("/accounting/expenses")
def add_expense(
    data: ExpenseRequest,
    request: Request,
):
    user_id = require_user(request)

    conn = get_db()

    cur = conn.execute(
        """
        INSERT INTO expenses
        (user_id, description, amount, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            user_id,
            data.description,
            data.amount,
            datetime.now(timezone.utc).isoformat(),
        ),
    )

    conn.commit()
    expense_id = cur.lastrowid
    conn.close()

    return {
        "id": expense_id,
        "status": "created",
    }


@app.get("/accounting/summary")
def accounting_summary(request: Request):
    user_id = require_user(request)

    conn = get_db()

    income = conn.execute(
        """
        SELECT COALESCE(SUM(amount), 0) AS total
        FROM transactions
        WHERE user_id = ? AND type = 'income'
        """,
        (user_id,),
    ).fetchone()["total"]

    transaction_expenses = conn.execute(
        """
        SELECT COALESCE(SUM(amount), 0) AS total
        FROM transactions
        WHERE user_id = ? AND type = 'expense'
        """,
        (user_id,),
    ).fetchone()["total"]

    expenses = conn.execute(
        """
        SELECT COALESCE(SUM(amount), 0) AS total
        FROM expenses
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()["total"]

    invoices = conn.execute(
        """
        SELECT COALESCE(SUM(amount), 0) AS total
        FROM invoices
        WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()["total"]

    conn.close()

    total_expenses = float(transaction_expenses) + float(expenses)

    return {
        "income": float(income),
        "expenses": total_expenses,
        "invoices": float(invoices),
        "balance": float(income) - total_expenses,
    }


@app.get("/accounting/export.csv")
def export_accounting(request: Request):
    user_id = require_user(request)

    conn = get_db()

    rows = conn.execute(
        """
        SELECT id, description, amount, type, created_at
        FROM transactions
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user_id,),
    ).fetchall()

    conn.close()

    def generate():
        yield "id,description,amount,type,created_at\n"

        for row in rows:
            description = (
                str(row["description"])
                .replace('"', '""')
                .replace("\n", " ")
            )

            yield (
                f'{row["id"]},"{description}",'
                f'{row["amount"]},{row["type"]},'
                f'{row["created_at"]}\n'
            )

    return StreamingResponse(
        generate(),
        media_type="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=juvens-ai-accounting.csv"
        },
    )
