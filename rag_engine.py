import os
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from rank_bm25 import BM25Okapi

# ── FIXED PATHS (IMPORTANT) ─────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent  # project root

KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
FAISS_PATH = BASE_DIR / "faiss_index"
BM25_CACHE = BASE_DIR / "faiss_index" / "bm25_cache.pkl"


# ── Embeddings wrapper ──────────────────────────────────────────────
class BGEEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.model.encode(
            f"Represent this sentence for searching relevant passages: {text}",
            normalize_embeddings=True
        ).tolist()


# ── Load knowledge base ─────────────────────────────────────────────
def load_knowledge_base():
    docs = []
    kb_path = Path(KNOWLEDGE_DIR)

    print("📂 Loading from:", kb_path)
    print("📂 Exists:", kb_path.exists())

    # TXT files
    for file in kb_path.glob("*.txt"):
        text = file.read_text(encoding="utf-8")
        docs.append(Document(
            page_content=text,
            metadata={"source": file.stem, "type": "txt"}
        ))

    # PDF files (optional)
    try:
        from langchain_community.document_loaders import PyPDFLoader
        for file in kb_path.glob("*.pdf"):
            loader = PyPDFLoader(str(file))
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata["source"] = file.stem
            docs.extend(pdf_docs)
    except Exception as e:
        print("PDF loader skipped:", e)

    return docs


# ── Build system ────────────────────────────────────────────────────
def build_system():
    embed_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    embedding_fn = BGEEmbeddings(embed_model)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    raw_docs = load_knowledge_base()

    if not raw_docs:
        raise ValueError("❌ No documents found in knowledge_base folder!")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", "?", "!", " "]
    )

    chunks = splitter.split_documents(raw_docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print("---- DEBUG START ----")
    print("Chunks count:", len(chunks))

    for i, c in enumerate(chunks[:3]):
        print(f"Chunk {i}:", repr(c.page_content))

    print("---- DEBUG END ----")

    if len(chunks) == 0:
        raise ValueError("❌ Chunking failed: 0 chunks produced")

    # ── FAISS ───────────────────────────────────────────────────────
    if FAISS_PATH.exists() and any(FAISS_PATH.iterdir()):
        vectorstore = FAISS.load_local(
            str(FAISS_PATH),
            embedding_fn,
            allow_dangerous_deserialization=True
        )
    else:
        vectorstore = FAISS.from_documents(chunks, embedding_fn)
        FAISS_PATH.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(FAISS_PATH))

    # ── BM25 ────────────────────────────────────────────────────────
    if BM25_CACHE.exists():
        with open(BM25_CACHE, "rb") as f:
            bm25 = pickle.load(f)
    else:
        tokenized = [c.page_content.lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized)

        FAISS_PATH.mkdir(parents=True, exist_ok=True)
        with open(BM25_CACHE, "wb") as f:
            pickle.dump(bm25, f)

    return vectorstore, bm25, chunks, reranker, embedding_fn


# ── Retrieval ───────────────────────────────────────────────────────
def retrieve(query, vectorstore, bm25, chunks, reranker, top_k=3):
    semantic_docs = vectorstore.similarity_search(query, k=8)

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:8]

    keyword_docs = [chunks[i] for i in top_bm25_idx]

    all_docs = {doc.page_content: doc for doc in semantic_docs + keyword_docs}
    unique_docs = list(all_docs.values())

    if not unique_docs:
        return []

    pairs = [[query, doc.page_content] for doc in unique_docs]
    scores = reranker.predict(pairs)

    reranked = sorted(zip(unique_docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in reranked[:top_k]]


# ── Context builder ────────────────────────────────────────────────
def build_context(docs):
    context = ""
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        context += f"\n[Source: {source}]\n{doc.page_content}\n"
    return context