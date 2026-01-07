# -*- coding: utf-8 -*-
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from openai import OpenAI
from rag import ingest_document, query_document
from web_memory import init_db, save_message, load_messages

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Put it in .env like: OPENAI_API_KEY=sk-...")

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "أنت موظف خدمة عملاء محترف لمتجر إلكتروني للأجهزة الكهربائية. "
        "ترد باللغة العربية بشكل مهذب وواضح. "
        "قواعد صارمة: لا تخمّن ولا تخترع معلومات. "
        "اعتمد فقط على المعلومات التي سأزوّدك بها في الرسائل. "
        "إذا لم تجد الإجابة في المعلومات، قل حرفيًا: "
        "\"المعلومة غير متوفرة في بيانات المتجر.\" "
        "إجاباتك مختصرة (1-3 جمل) وتسأل سؤال توضيحي عند الحاجة."
    )
}

app = FastAPI(title="AI Store Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # بعدين نحدد دومين Wuilt فقط
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# نخدم صفحة HTML
@app.get("/", response_class=HTMLResponse)
def home():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.on_event("startup")
def startup():
    init_db()
    ingest_document()  # يبني/يحدّث الفهرس عند تشغيل السيرفر

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    user_input = (req.message or "").strip()
    session_id = (req.session_id or "").strip()

    if not session_id:
        return ChatResponse(reply="من فضلك حدّث الصفحة وأعد المحاولة.")

    if not user_input:
        return ChatResponse(reply="اكتب سؤالك من فضلك.")

    # خزّن رسالة المستخدم
    save_message(session_id, "user", user_input)

    # RAG: استرجاع أفضل أجزاء من المستند
    rag_chunks = query_document(user_input, k=3)
    rag_context = "\n\n---\n\n".join(rag_chunks)

    rag_prompt = {
        "role": "system",
        "content": (
            "المصدر الوحيد للإجابة هو المعلومات التالية. "
            "ممنوع الإجابة من خارجها. "
            "إذا لم تجد الإجابة، قل: \"المعلومة غير متوفرة في بيانات المتجر.\""
            "\n\nالمعلومات:\n"
            f"{rag_context}"
        )
    }

    # آخر رسائل المحادثة لهذه الجلسة فقط
    history = load_messages(session_id, limit=10)

    messages = [SYSTEM_PROMPT, rag_prompt] + history

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    reply = resp.choices[0].message.content
    save_message(session_id, "assistant", reply)

    return ChatResponse(reply=reply)
