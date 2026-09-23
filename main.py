import os, io, csv, base64, mimetypes, sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, List
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
DB=os.getenv("DATABASE_PATH","./juvens_ai.db")
MODEL=os.getenv("GEMINI_MODEL","gemini-3-flash")
JWT_SECRET=os.getenv("JWT_SECRET","CHANGE_ME_IN_PRODUCTION")
ADMIN_EMAIL=os.getenv("ADMIN_EMAIL","admin@example.com")
KEY=os.getenv("GEMINI_API_KEY","").strip()
client=OpenAI(api_key=KEY,base_url="https://generativelanguage.googleapis.com/v1beta/openai/") if KEY else None
pwd=CryptContext(schemes=["bcrypt"],deprecated="auto")
limiter=Limiter(key_func=get_remote_address)
app=FastAPI(title="Juvens AI API",version="4.0.0")
app.state.limiter=limiter
@app.exception_handler(RateLimitExceeded)
async def rl(request,exc): return JSONResponse(status_code=429,content={"detail":"Trop de requêtes. Réessayez dans une minute."})

app.add_middleware(CORSMiddleware,allow_origins=["https://aijuvensley1.pages.dev","https://aijuvensley.pages.dev"],allow_origin_regex=r"https://[a-zA-Z0-9-]+\.pages\.dev",allow_credentials=False,allow_methods=["*"],allow_headers=["*"])

def db():
 c=sqlite3.connect(DB);c.row_factory=sqlite3.Row;return c
def now():return datetime.now(timezone.utc).isoformat()
def init():
 c=db();c.executescript("""CREATE TABLE IF NOT EXISTS users(id INTEGER PRIMARY KEY AUTOINCREMENT,email TEXT UNIQUE NOT NULL,name TEXT NOT NULL,password_hash TEXT NOT NULL,created_at TEXT NOT NULL,is_admin INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS conversations(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,title TEXT NOT NULL,created_at TEXT NOT NULL,updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,conversation_id INTEGER NOT NULL,role TEXT NOT NULL,content TEXT NOT NULL,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS files(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,name TEXT NOT NULL,file_path TEXT,created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS transactions(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,date TEXT NOT NULL,description TEXT NOT NULL,debit_account TEXT NOT NULL,credit_account TEXT NOT NULL,amount REAL NOT NULL,reference TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS invoices(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,number TEXT NOT NULL,customer TEXT NOT NULL,amount REAL NOT NULL,status TEXT NOT NULL DEFAULT 'unpaid',due_date TEXT DEFAULT '',created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS expenses(id INTEGER PRIMARY KEY AUTOINCREMENT,user_id INTEGER NOT NULL,date TEXT NOT NULL,category TEXT NOT NULL,description TEXT NOT NULL,amount REAL NOT NULL);""");c.commit();c.close()
init()
def token(u):return jwt.encode({"sub":str(u["id"]),"email":u["email"],"exp":datetime.now(timezone.utc)+timedelta(days=7)},JWT_SECRET,algorithm="HS256")
def current_user(request:Request):
 a=request.headers.get("Authorization","")
 if not a.startswith("Bearer "):raise HTTPException(401,"Authentification requise")
 try:uid=int(jwt.decode(a[7:],JWT_SECRET,algorithms=["HS256"])["sub"])
 except:raise HTTPException(401,"Session invalide ou expirée")
 c=db();u=c.execute("SELECT * FROM users WHERE id=?",(uid,)).fetchone();c.close()
 if not u:raise HTTPException(401,"Utilisateur introuvable")
 return u

class Register(BaseModel):name:str=Field(min_length=2,max_length=80);email:str;password:str=Field(min_length=8,max_length=200)
class Login(BaseModel):email:str;password:str
class Attach(BaseModel):name:str;mime_type:str="application/octet-stream";data:str
class Chat(BaseModel):message:str=Field(min_length=1,max_length=30000);conversation_id:Optional[int]=None;attachments:List[Attach]=[]
class Title(BaseModel):title:str=Field(min_length=1,max_length=120)

SYSTEM="""Tu es Juvens AI, un assistant intelligent et professionnel créé par Pierre Manuel Juvensley.
Si on te demande qui t'a créé, ton créateur, qui a développé Juvens AI ou qui est derrière Juvens AI,
réponds clairement : « J'ai été créé par Pierre Manuel Juvensley. »
Réponds dans la langue de l'utilisateur (français, créole haïtien ou anglais).
Sois clair, précis, honnête et utile. Adapte les réponses business, comptabilité et commerce au contexte haïtien."""

@app.get("/")
def root():return {"name":"Juvens AI","status":"ok","version":"4.0.0"}
@app.get("/health")
def health():return {"status":"ok","ai_active":client is not None,"gemini_key_detected":bool(KEY),"model":MODEL,"guest_chat":True,"attachments":True}

@app.post("/auth/register")
def register(x:Register):
 c=db()
 try:
  cur=c.execute("INSERT INTO users(email,name,password_hash,created_at,is_admin) VALUES(?,?,?,?,?)",(x.email.lower(),x.name,pwd.hash(x.password),now(),1 if x.email.lower()==ADMIN_EMAIL.lower() else 0));c.commit();u=c.execute("SELECT * FROM users WHERE id=?",(cur.lastrowid,)).fetchone();return {"token":token(u),"user":{"id":u["id"],"name":u["name"],"email":u["email"],"is_admin":bool(u["is_admin"])}}
 except sqlite3.IntegrityError:raise HTTPException(409,"Cet e-mail est déjà utilisé")
 finally:c.close()
@app.post("/auth/login")
def login(x:Login):
 c=db();u=c.execute("SELECT * FROM users WHERE email=?",(x.email.lower(),)).fetchone();c.close()
 if not u or not pwd.verify(x.password,u["password_hash"]):raise HTTPException(401,"E-mail ou mot de passe incorrect")
 return {"token":token(u),"user":{"id":u["id"],"name":u["name"],"email":u["email"],"is_admin":bool(u["is_admin"])}}

def extract_text(a:Attach):
 try:
  raw=base64.b64decode(a.data.split(",",1)[1] if "," in a.data else a.data)
  if a.mime_type.startswith("text/") or a.mime_type in ("application/json","text/csv"):return raw.decode("utf-8","ignore")[:60000]
  if a.mime_type=="application/pdf":
   from pypdf import PdfReader
   return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(raw)).pages)[:60000]
 except:pass
 return None

def model_messages(chat:Chat, history=None):
 parts=[]
 for a in chat.attachments:
  if a.mime_type.startswith("image/"):
   parts.append({"type":"image_url","image_url":{"url":a.data}})
  else:
   t=extract_text(a)
   if t: parts.append({"type":"text","text":f"Contenu du fichier {a.name}:\n{t}"})
 parts.append({"type":"text","text":chat.message})
 user_content=parts[0] if len(parts)==1 and parts[0]["type"]=="text" else parts
 msgs=[{"role":"system","content":SYSTEM}]
 if history:msgs+=history
 msgs.append({"role":"user","content":user_content})
 return msgs

@app.post("/chat")
@limiter.limit("30/minute")
def chat(request:Request,x:Chat):
 if not client:raise HTTPException(503,"GEMINI_API_KEY non configurée.")
 # Guest mode: no account required. Authenticated users get persistent history.
 user=None
 try:user=current_user(request)
 except HTTPException as e:
  if e.status_code!=401:raise
 if user and x.conversation_id:
  c=db();row=c.execute("SELECT id FROM conversations WHERE id=? AND user_id=?",(x.conversation_id,user["id"])).fetchone()
  if not row:c.close();raise HTTPException(404,"Conversation introuvable")
  hist=c.execute("SELECT role,content FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT 40",(x.conversation_id,)).fetchall();hist=list(reversed(hist))
  c.close()
 else: hist=[]
 msgs=[{"role":m["role"],"content":m["content"]} for m in hist]
 try:
  resp=client.chat.completions.create(model=MODEL,messages=model_messages(x,msgs))
  answer=resp.choices[0].message.content or "Je n'ai pas pu générer une réponse."
 except Exception as e:raise HTTPException(502,f"Erreur du service IA : {str(e)}")
 cid=None
 if user:
  c=db()
  if x.conversation_id:cid=x.conversation_id
  else:
   cur=c.execute("INSERT INTO conversations(user_id,title,created_at,updated_at) VALUES(?,?,?,?)",(user["id"],x.message[:60],now(),now()));cid=cur.lastrowid
  c.execute("INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",(cid,"user",x.message,now()))
  c.execute("INSERT INTO messages(conversation_id,role,content,created_at) VALUES(?,?,?,?)",(cid,"assistant",answer,now()));c.execute("UPDATE conversations SET updated_at=? WHERE id=?",(now(),cid));c.commit();c.close()
 return {"reply":answer,"conversation_id":cid,"model":MODEL,"guest":user is None}

@app.post("/files/upload")
@limiter.limit("10/minute")
async def upload(request:Request,file:UploadFile=File(...),user=Depends(current_user)):
 b=await file.read()
 if len(b)>10*1024*1024:raise HTTPException(413,"Fichier trop volumineux (max 10 MB)")
 os.makedirs("uploads",exist_ok=True);name=os.path.basename(file.filename or "file");path=f"uploads/user{user['id']}_{name}"
 open(path,"wb").write(b);c=db();cur=c.execute("INSERT INTO files(user_id,name,file_path,created_at) VALUES(?,?,?,?)",(user["id"],name,path,now()));c.commit();c.close()
 return {"id":cur.lastrowid,"name":name,"ok":True}

@app.get("/conversations")
def conversations(user=Depends(current_user)):
 c=db();r=c.execute("SELECT * FROM conversations WHERE user_id=? ORDER BY updated_at DESC",(user["id"],)).fetchall();c.close();return [dict(x) for x in r]
@app.get("/conversations/{cid}")
def conversation(cid:int,user=Depends(current_user)):
 c=db();r=c.execute("SELECT * FROM conversations WHERE id=? AND user_id=?",(cid,user["id"])).fetchone()
 if not r:c.close();raise HTTPException(404,"Conversation introuvable")
 m=c.execute("SELECT role,content,created_at FROM messages WHERE conversation_id=? ORDER BY id",(cid,)).fetchall();c.close();return {"conversation":dict(r),"messages":[dict(x) for x in m]}
@app.delete("/conversations/{cid}")
def delete_conversation(cid:int,user=Depends(current_user)):
 c=db();c.execute("DELETE FROM messages WHERE conversation_id=?",(cid,));r=c.execute("DELETE FROM conversations WHERE id=? AND user_id=?",(cid,user["id"]));c.commit();c.close()
 if not r.rowcount:raise HTTPException(404,"Conversation introuvable")
 return {"ok":True}
