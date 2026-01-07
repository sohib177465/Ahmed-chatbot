# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv
from openai import OpenAI

from memory import init_db, save_message, load_messages
from rag import ingest_document, query_document

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


def chatbot():
    init_db()

    # فهرسة المستند مرة عند التشغيل (ولو عدّلت الملف هيتحدث)
    ingest_document()

    print("ابدأ الكتابة الآن (اكتب exit للخروج)\n")

    while True:
        user_input = input("أنت: ").strip()

        if user_input.lower() == "exit":
            print("Bot: مع السلامة 👋")
            break

        # حفظ رسالة المستخدم في الذاكرة
        save_message("user", user_input)

        # RAG: استرجاع أفضل أجزاء من المستند حسب سؤال المستخدم
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

        # الرسائل: System + RAG + آخر جزء من المحادثة (قصير لتقليل التكلفة)
        messages = [SYSTEM_PROMPT, rag_prompt] + load_messages(limit=10)

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )

        bot_reply = resp.choices[0].message.content
        print("Bot:", bot_reply)

        # حفظ رد البوت
        save_message("assistant", bot_reply)


if __name__ == "__main__":
    chatbot()
