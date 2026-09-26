import os
import requests
import numpy as np
from pypdf import PdfReader


# ============================================================
# SETTINGS
# ============================================================

DOCS_FOLDER = "docs"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"

CHUNK_SIZE = 100
OVERLAP = 20

TOP_K = 3


# ============================================================
# 1. READ PDF
# ============================================================

def load_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# ============================================================
# 1b. READ TXT
# ============================================================

def load_txt(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================
# 2. LOAD ALL DOCUMENTS (PDF + TXT)
# ============================================================

def load_documents():
    documents = []

    for filename in os.listdir(DOCS_FOLDER):

        path = os.path.join(DOCS_FOLDER, filename)

        if filename.lower().endswith(".pdf"):
            text = load_pdf(path)
        elif filename.lower().endswith(".txt"):
            text = load_txt(path)
        else:
            continue

        documents.append({
            "source": filename,
            "text": text
        })

    return documents


# ============================================================
# 3. FIXED WORD CHUNKING + OVERLAP
# ============================================================

def chunk_text(text, chunk_size=100, overlap=20):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        chunks.append(chunk)

        start = end - overlap

    return chunks


# ============================================================
# 4. CREATE CHUNKS FOR ALL DOCUMENTS
# ============================================================

def create_chunks(documents):

    chunks = []

    for document in documents:

        text_chunks = chunk_text(
            document["text"],
            CHUNK_SIZE,
            OVERLAP
        )

        for i, chunk in enumerate(text_chunks):

            chunks.append({
                "source": document["source"],
                "chunk_id": i,
                "text": chunk
            })

    return chunks


# ============================================================
# 5. CREATE EMBEDDING
# ============================================================

def embed_text(text):

    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={
            "model": EMBEDDING_MODEL,
            "prompt": text
        }
    )

    response.raise_for_status()

    return response.json()["embedding"]


# ============================================================
# 6. EMBED ALL CHUNKS
# ============================================================

def create_embeddings(chunks):

    for chunk in chunks:

        chunk["embedding"] = embed_text(
            chunk["text"]
        )

    return chunks


# ============================================================
# 7. COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# ============================================================
# 8. RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve(query, chunks, top_k=3):

    query_embedding = embed_text(query)

    results = []

    for chunk in chunks:

        score = cosine_similarity(
            query_embedding,
            chunk["embedding"]
        )

        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "chunk_id": chunk["chunk_id"],
            "score": score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# 9. SEND CONTEXT TO LLM
# ============================================================

def generate_answer(query, results):

    context = ""

    for result in results:

        context += (
            f"\nSource: {result['source']}\n"
            f"{result['text']}\n"
        )

    prompt = f"""
Answer the question using only the context below.

If the answer is not present in the context,
say that you could not find the answer.

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_predict": 300}
        }
    )

    response.raise_for_status()

    return response.json()["response"]


# ============================================================
# 10. MAIN RAG PIPELINE
# ============================================================

def main():

    print("Loading documents...")

    documents = load_documents()

    print("Creating chunks...")

    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings...")

    chunks = create_embeddings(chunks)

    print("RAG is ready.")

    while True:

        query = input("\nAsk a question (type 'exit' to stop): ")

        if query.lower() == "exit":
            break

        results = retrieve(
            query,
            chunks,
            TOP_K
        )

        answer = generate_answer(
            query,
            results
        )

        print("\nAnswer:")
        print(answer)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()
