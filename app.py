import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from rag_engine import build_system, retrieve, build_context

# ── Init ────────────────────────────────────────────────────────────
app = FastAPI()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Replace with your uncle's actual domain once live
ALLOWED_ORIGINS = [
    "http://localhost",
    "https://yourdomain.com",       # ← change this
    "https://www.yourdomain.com",   # ← change this
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Load RAG system once at startup ────────────────────────────────
print("Loading RAG system...")
vectorstore, bm25, chunks, reranker, embedding_fn = build_system()
print("RAG system ready.")

# ── Conversation memory (in-memory, per session) ───────────────────
sessions: dict[str, list] = {}

SYSTEM_PROMPT = """You are a helpful assistant for Office Max, a furniture and interior design company based in Islamabad, Pakistan.

CONTACT INFORMATION (always available):
- Address: Office Max Plaza, Opposite CMT and SD Gate, Gollra Road, Islamabad
- Phone: 0515495779
- Email: sales@officemaxpk.com
- WhatsApp (Faisal Qureshi): 03165280200

Rules:
1. Answer ONLY from the provided context. Never make up prices, specs, or availability.
2. Be concise and friendly (2-3 sentences max unless listing items).
3. If asked about pricing, delivery, stock, discounts, or returns say:
   "I don't have that information right now. Please contact Faisal Qureshi directly on WhatsApp: 03165280200"
4. If the question is not covered in the context at all, say:
   "I'm not sure about that. For more help, please reach out to Faisal Qureshi on WhatsApp: 03165280200 — he'll be happy to assist!"
5. Never make up specifications, prices, or availability.
6. If asked about contact details, always provide the address, phone, and email above.
7. Escalation message MUST be exactly:
   "I don't have that information right now. Please contact Faisal Qureshi directly on WhatsApp for immediate help: 03165280200"
8.If asked about ANY of these topics, ALWAYS escalate to Faisal Qureshi,Please contact Faisal Qureshi directly on WhatsApp for immediate help: 03165280200" — do NOT try to answer:
   - Pricing or quotes
   - Delivery time or shipping
   - Stock or availability
   - Discounts or offers
   - Returns or complaints
   - Custom orders
   - Anything not clearly covered in the context

"""

# ── Request model ───────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

# ── Health check ────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "message": "RAG chatbot is running"}

# ── Main chat endpoint ──────────────────────────────────────────────
@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Empty message")

    # Get or create session history
    history = sessions.get(req.session_id, [])

    # Retrieve relevant chunks
    docs = retrieve(req.message, vectorstore, bm25, chunks, reranker)
    context = build_context(docs) if docs else "No relevant information found."

    # Build messages for Groq
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + f"\n\nCONTEXT:\n{context}"
        }
    ]
    # Attach last 4 turns of history
    messages += history[-8:]  # 4 user + 4 assistant = 8 messages
    messages.append({"role": "user", "content": req.message})

    # Call Groq — LLaMA 3.3 70B
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3,
        max_tokens=400,
    )
    answer = completion.choices[0].message.content.strip()

    # Save to session history
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})
    sessions[req.session_id] = history[-8:]  # keep last 4 turns only

    return {
        "answer": answer,
        "sources": list({doc.metadata.get("source", "") for doc in docs})
    }