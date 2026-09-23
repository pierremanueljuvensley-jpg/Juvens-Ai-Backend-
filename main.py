import os
import csv
import io
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, Any

import jwt
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from passlib.context import CryptContext
from pydantic import BaseModel, Field
from openai import OpenAI
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# ============================================================
# JUVENS AI — BACKEND
# Corrected Gemini/OpenAI-compatible chat integration.
# ============================================================

DB = os.getenv("DATABASE_PATH", "./juvens_ai.db")
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_IN_PRODUCTION_PLEASE")
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@example.com")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Current production default. If Render defines GEMINI_MODEL, it is used,
# except for old model values that previously caused availability errors.
_requested_model = os.getenv("GEMINI_MODEL", "").strip()
if _requested_model in {"", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-001"}:
    MODEL = "gemini-3.8-flash"
else:
    MODEL = _requested_model

client = (
    OpenAI(
        api_key=GEMINI_API_KEY,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    if GEMINI_API_KEY
    else None
)

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Juvens AI API", version="3.1.0")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Trop de requêtes. Réessayez dans une minute."},
    )


origins = [x.strip() for x in FRONTEND_ORIGIN.split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE
# ============================================================

def db():
    con = sqlite3.connect(DB)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    con = db()
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS conversations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id)
        );

        CREATE TABLE IF NOT EXISTS files(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            file_path TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS transactions(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            description TEXT NOT NULL,
            debit_account TEXT NOT NULL,
            credit_account TEXT NOT NULL,
            amount REAL NOT NULL,
            reference TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS invoices(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            number TEXT NOT NULL,
            customer TEXT NOT NULL,
            amount REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'unpaid',
            due_date TEXT DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS expenses(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            amount REAL NOT NULL
        );
        """
    )
    con.commit()
    con.close()


init_db()


def now():
    return datetime.now(timezone.utc).isoformat()


# ============================================================
# AUTH
# ============================================================

pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


def make_token(user):
    return jwt.encode(
        {
            "sub": str(user["id"]),
            "email": user["email"],
            "exp": datetime.now(timezone.utc) + timedelta(days=7),
        },
        JWT_SECRET,
        algorithm="HS256",
    )


def current_user(request: Request):
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Authentification requise")

    try:
        payload = jwt.decode(auth[7:], JWT_SECRET, algorithms=["HS256"])
        uid = int(payload["sub"])
    except Exception:
        raise HTTPException(401, "Session invalide ou expirée")

    con = db()
    user = con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    con.close()

    if not user:
        raise HTTPException(401, "Utilisateur introuvable")

    return user


def optional_user(request: Request):
    """Return the authenticated user when a valid Bearer token exists.
    Chat can also be tested directly from Swagger without authentication.
    """
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None

    try:
        payload = jwt.decode(auth[7:], JWT_SECRET, algorithms=["HS256"])
        uid = int(payload["sub"])
    except Exception:
        return None

    con = db()
    user = con.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    con.close()
    return user


# ============================================================
# PYDANTIC MODELS
# ============================================================

class RegisterIn(BaseModel):
    name: str = Field(min_length=2, max_length=80)
    email: str
    password: str = Field(min_length=8, max_length=200)


class LoginIn(BaseModel):
    email: str
    password: str


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=30000)
    conversation_id: Optional[int] = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)


class TitleIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class TransactionIn(BaseModel):
    date: str
    description: str
    debit_account: str
    credit_account: str
    amount: float = Field(gt=0)
    reference: str = ""


class InvoiceIn(BaseModel):
    number: str
    customer: str
    amount: float = Field(gt=0)
    status: str = "unpaid"
    due_date: str = ""


class ExpenseIn(BaseModel):
    date: str
    category: str
    description: str
    amount: float = Field(gt=0)


# ============================================================
# BASE ROUTES
# ============================================================

@app.get("/")
def root():
    return {
        "name": "Juvens AI",
        "status": "ok",
        "version": "3.1.0",
        "ai_active": client is not None,
        "model": MODEL,
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ai_active": client is not None,
        "gemini_key_detected": bool(GEMINI_API_KEY),
        "model": MODEL,
    }


# ============================================================
# AUTHENTICATION
# ============================================================

@app.post("/auth/register")
def register(data: RegisterIn):
    con = db()
    try:
        is_admin = 1 if data.email.lower() == ADMIN_EMAIL.lower() else 0
        cur = con.execute(
            "INSERT INTO users(email,name,password_hash,created_at,is_admin) VALUES(?,?,?,?,?)",
            (
                data.email.lower(),
                data.name,
                pwd_ctx.hash(data.password),
                now(),
                is_admin,
            ),
        )
        con.commit()
        user = con.execute(
            "SELECT * FROM users WHERE id=?", (cur.lastrowid,)
        ).fetchone()

        return {
            "token": make_token(user),
            "user": {
                "id": user["id"],
                "name": user["name"],
                "email": user["email"],
                "is_admin": bool(user["is_admin"]),
            },
        }
    except sqlite3.IntegrityError:
        raise HTTPException(409, "Cet e-mail est déjà utilisé")
    finally:
        con.close()


@app.post("/auth/login")
def login(data: LoginIn):
    con = db()
    user = con.execute(
        "SELECT * FROM users WHERE email=?", (data.email.lower(),)
    ).fetchone()
    con.close()

    if not user or not pwd_ctx.verify(data.password, user["password_hash"]):
        raise HTTPException(401, "E-mail ou mot de passe incorrect")

    return {
        "token": make_token(user),
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
            "is_admin": bool(user["is_admin"]),
        },
    }


@app.get("/me")
def me(user=Depends(current_user)):
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "is_admin": bool(user["is_admin"]),
    }


# ============================================================
# CONVERSATIONS
# ============================================================

@app.get("/conversations")
def conversations(user=Depends(current_user)):
    con = db()
    rows = con.execute(
        "SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC",
        (user["id"],),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.post("/conversations")
def new_conversation(user=Depends(current_user)):
    con = db()
    t = now()
    cur = con.execute(
        "INSERT INTO conversations(user_id,title,created_at,updated_at) VALUES(?,?,?,?)",
        (user["id"], "Nouvelle conversation", t, t),
    )
    con.commit()
    row = con.execute(
        "SELECT * FROM conversations WHERE id=?", (cur.lastrowid,)
    ).fetchone()
    con.close()
    return dict(row)


@app.get("/conversations/{cid}")
def get_conversation(cid: int, user=Depends(current_user)):
    con = db()
    c = con.execute(
        "SELECT * FROM conversations WHERE id=? AND user_id=?",
        (cid, user["id"]),
    ).fetchone()

    if not c:
        con.close()
        raise HTTPException(404, "Conversation introuvable")

    msgs = con.execute(
        "SELECT role,content,created_at FROM messages WHERE conversation_id=? ORDER BY id",
        (cid,),
    ).fetchall()
    con.close()

    return {
        "conversation": dict(c),
        "messages": [dict(m) for m in msgs],
    }


@app.patch("/conversations/{cid}")
def rename_conversation(
    cid: int, data: TitleIn, user=Depends(current_user)
):
    con = db()
    cur = con.execute(
        "UPDATE conversations SET title=?,updated_at=? WHERE id=? AND user_id=?",
        (data.title, now(), cid, user["id"]),
    )
    con.commit()
    con.close()

    if not cur.rowcount:
        raise HTTPException(404, "Conversation introuvable")

    return {"ok": True}


@app.delete("/conversations/{cid}")
def delete_conversation(cid: int, user=Depends(current_user)):
    con = db()
    con.execute("DELETE FROM messages WHERE conversation_id=?", (cid,))
    cur = con.execute(
        "DELETE FROM conversations WHERE id=? AND user_id=?",
        (cid, user["id"]),
    )
    con.commit()
    con.close()

    if not cur.rowcount:
        raise HTTPException(404, "Conversation introuvable")

    return {"ok": True}


# ============================================================
# CHAT IA — CORRECTION PRINCIPALE
# ============================================================

SYSTEM_PROMPT = """Tu es Juvens AI, un assistant intelligent et professionnel créé par Juvens.
Tu réponds dans la langue de l'utilisateur : français, créole haïtien ou anglais.
Sois clair, précis, honnête et utile.
Quand une information est incertaine, dis-le clairement.
Pour les questions liées aux affaires, à la comptabilité et au commerce en Haïti,
adapte tes réponses au contexte haïtien.
"""


@app.post("/chat")
@limiter.limit("30/minute")
def chat(
    request: Request,
    data: ChatIn,
    user=Depends(optional_user),
):
    if not client:
        raise HTTPException(
            503,
            "GEMINI_API_KEY non configurée. Veuillez contacter l'administrateur.",
        )

    cid = data.conversation_id

    if user:
        con = db()

        if cid is not None:
            c = con.execute(
                "SELECT * FROM conversations WHERE id=? AND user_id=?",
                (cid, user["id"]),
            ).fetchone()

            if not c:
                con.close()
                raise HTTPException(404, "Conversation introuvable")
        else:
            t = now()
            title = data.message[:60] + (
                "..." if len(data.message) > 60 else ""
            )
            cur = con.execute(
                "INSERT INTO conversations(user_id,title,created_at,updated_at) VALUES(?,?,?,?)",
                (user["id"], title, t, t),
            )
            cid = cur.lastrowid

        history = con.execute(
            "SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 40",
            (cid,),
        ).fetchall()
        history = list(reversed(history))

        con.execute(
            "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
            (cid, "user", data.message, now()),
        )
        con.commit()
        con.close()
    else:
        history = []

    # IMPORTANT:
    # Gemini OpenAI compatibility expects message content to be a STRING
    # (or a supported list of content parts). Do NOT send:
    # {"text": "...", "type": "text"}
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    for item in history:
        role = item["role"]
        if role not in {"user", "assistant"}:
            continue

        content = item["content"]
        if not isinstance(content, str):
            content = str(content)

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": data.message,
        }
    )

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

        answer = response.choices[0].message.content

        if not answer:
            answer = "Je n'ai pas pu générer une réponse."

    except Exception as exc:
        raise HTTPException(
            502,
            f"Erreur du service IA : {str(exc)}",
        )

    if user and cid is not None:
        con = db()
        con.execute(
            "INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",
            (cid, "assistant", answer, now()),
        )
        con.execute(
            "UPDATE conversations SET updated_at=? WHERE id=?",
            (now(), cid),
        )
        con.commit()
        con.close()

    return {
        "conversation_id": cid,
        "reply": answer,
        "model": MODEL,
    }


# ============================================================
# FILES
# ============================================================

@app.post("/files/upload")
@limiter.limit("10/minute")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    user=Depends(current_user),
):
    content = await file.read()

    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "Fichier trop volumineux (max 10 MB)")

    os.makedirs("./uploads", exist_ok=True)

    original_name = file.filename or "file"
    safe_name = f"user{user['id']}_{os.path.basename(original_name)}"
    file_path = f"./uploads/{safe_name}"

    with open(file_path, "wb") as f:
        f.write(content)

    con = db()
    cur = con.execute(
        "INSERT INTO files(user_id,name,file_path,created_at) VALUES(?,?,?,?)",
        (user["id"], original_name, file_path, now()),
    )
    con.commit()
    con.close()

    return {
        "id": cur.lastrowid,
        "name": original_name,
        "ok": True,
    }


@app.get("/files")
def list_files(user=Depends(current_user)):
    con = db()
    rows = con.execute(
        "SELECT id,name,created_at FROM files WHERE user_id=? ORDER BY id DESC",
        (user["id"],),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ============================================================
# ACCOUNTING
# ============================================================

@app.get("/accounting/summary")
def accounting_summary(user=Depends(current_user)):
    con = db()

    revenue = con.execute(
        "SELECT COALESCE(SUM(amount),0) x FROM invoices WHERE user_id=? AND status='paid'",
        (user["id"],),
    ).fetchone()["x"]

    expenses = con.execute(
        "SELECT COALESCE(SUM(amount),0) x FROM expenses WHERE user_id=?",
        (user["id"],),
    ).fetchone()["x"]

    receivable = con.execute(
        "SELECT COALESCE(SUM(amount),0) x FROM invoices WHERE user_id=? AND status='unpaid'",
        (user["id"],),
    ).fetchone()["x"]

    con.close()

    return {
        "revenue": revenue,
        "expenses": expenses,
        "profit": revenue - expenses,
        "receivables": receivable,
    }


@app.get("/accounting/transactions")
def list_transactions(user=Depends(current_user)):
    con = db()
    rows = con.execute(
        "SELECT * FROM transactions WHERE user_id=? ORDER BY date DESC, id DESC",
        (user["id"],),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.post("/accounting/transactions")
def add_transaction(data: TransactionIn, user=Depends(current_user)):
    con = db()
    cur = con.execute(
        "INSERT INTO transactions(user_id,date,description,debit_account,credit_account,amount,reference) VALUES(?,?,?,?,?,?,?)",
        (
            user["id"],
            data.date,
            data.description,
            data.debit_account,
            data.credit_account,
            data.amount,
            data.reference,
        ),
    )
    con.commit()
    con.close()
    return {"id": cur.lastrowid, "ok": True}


@app.get("/accounting/invoices")
def list_invoices(user=Depends(current_user)):
    con = db()
    rows = con.execute(
        "SELECT * FROM invoices WHERE user_id=? ORDER BY id DESC",
        (user["id"],),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.post("/accounting/invoices")
def add_invoice(data: InvoiceIn, user=Depends(current_user)):
    con = db()
    cur = con.execute(
        "INSERT INTO invoices(user_id,number,customer,amount,status,due_date,created_at) VALUES(?,?,?,?,?,?,?)",
        (
            user["id"],
            data.number,
            data.customer,
            data.amount,
            data.status,
            data.due_date,
            now(),
        ),
    )
    con.commit()
    con.close()
    return {"id": cur.lastrowid, "ok": True}


@app.get("/accounting/expenses")
def list_expenses(user=Depends(current_user)):
    con = db()
    rows = con.execute(
        "SELECT * FROM expenses WHERE user_id=? ORDER BY date DESC, id DESC",
        (user["id"],),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


@app.post("/accounting/expenses")
def add_expense(data: ExpenseIn, user=Depends(current_user)):
    con = db()
    cur = con.execute(
        "INSERT INTO expenses(user_id,date,category,description,amount) VALUES(?,?,?,?,?)",
        (
            user["id"],
            data.date,
            data.category,
            data.description,
            data.amount,
        ),
    )
    con.commit()
    con.close()
    return {"id": cur.lastrowid, "ok": True}


@app.get("/accounting/export.csv")
def export_csv(user=Depends(current_user)):
    con = db()
    rows = con.execute(
        "SELECT date,description,debit_account,credit_account,amount,reference "
        "FROM transactions WHERE user_id=? ORDER BY date,id",
        (user["id"],),
    ).fetchall()
    con.close()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        ["Date", "Description", "Débit", "Crédit", "Montant (HTG)", "Référence"]
    )

    for r in rows:
        writer.writerow(list(r))

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=juvens-ai-comptabilite.csv"
        },
    )


# ============================================================
# ADMINISTRATION
# ============================================================

@app.get("/admin/stats")
def admin_stats(user=Depends(current_user)):
    if not user["is_admin"]:
        raise HTTPException(403, "Accès réservé à l'administrateur")

    con = db()

    stats = {
        "users": con.execute(
            "SELECT COUNT(*) x FROM users"
        ).fetchone()["x"],
        "conversations": con.execute(
            "SELECT COUNT(*) x FROM conversations"
        ).fetchone()["x"],
        "messages": con.execute(
            "SELECT COUNT(*) x FROM messages"
        ).fetchone()["x"],
        "files": con.execute(
            "SELECT COUNT(*) x FROM files"
        ).fetchone()["x"],
    }

    con.close()
    return stats
