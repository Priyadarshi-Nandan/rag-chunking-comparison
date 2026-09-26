import os
import re
import requests
import numpy as np
from pypdf import PdfReader


# ============================================================
# SETTINGS
# ============================================================

DOCS_FOLDER = "docs"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:3b"

CHUNK_SIZE = 500

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
# 2. LOAD DOCUMENTS (PDF + TXT)
# ============================================================

def load_documents():

    documents = []

    for filename in os.listdir(DOCS_FOLDER):

        path = os.path.join(
            DOCS_FOLDER,
            filename
        )

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
# 3. RECURSIVE CHUNKING
# ============================================================

def recursive_split(text, chunk_size, separators):

    text = text.strip()

    if len(text) <= chunk_size:

        return [text]

    if not separators:

        words = text.split()

        chunks = []

        current = ""

        for word in words:

            if len(current) + len(word) + 1 <= chunk_size:

                current += word + " "

            else:

                if current:
                    chunks.append(
                        current.strip()
                    )

                current = word + " "

        if current:
            chunks.append(
                current.strip()
            )

        return chunks

    separator = separators[0]

    if separator == "SENTENCE":

        parts = re.split(
            r'(?<=[.!?])\s+',
            text
        )

    else:

        parts = text.split(separator)

    if len(parts) == 1:

        return recursive_split(
            text,
            chunk_size,
            separators[1:]
        )

    chunks = []

    current = ""

    for part in parts:

        part = part.strip()

        if not part:
            continue

        if len(part) > chunk_size:

            if current:

                chunks.append(
                    current.strip()
                )

                current = ""

            smaller_chunks = recursive_split(
                part,
                chunk_size,
                separators[1:]
            )

            chunks.extend(smaller_chunks)

        elif len(current) + len(part) + 1 <= chunk_size:

            if current:
                current += separator if separator != "SENTENCE" else " "

            current += part

        else:

            if current:

                chunks.append(
                    current.strip()
                )

            current = part

    if current:

        chunks.append(
            current.strip()
        )

    return chunks


def chunk_text(text, chunk_size=500):

    separators = [
        "\n\n",
        "\n",
        "SENTENCE",
        " "
    ]

    return recursive_split(
        text,
        chunk_size,
        separators
    )


# ============================================================
# 4. CREATE CHUNKS
# ============================================================

def create_chunks(documents):

    chunks = []

    for document in documents:

        text_chunks = chunk_text(
            document["text"],
            CHUNK_SIZE
        )

        for i, chunk in enumerate(text_chunks):

            chunks.append({
                "source": document["source"],
                "chunk_id": i,
                "text": chunk
            })

    return chunks


# ============================================================
# 5. EMBEDDING
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
# 6. CREATE EMBEDDINGS
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
# 8. RETRIEVAL
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
# 9. GENERATE ANSWER
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
# 10. MAIN
# ============================================================

def main():

    print("Loading documents...")

    documents = load_documents()

    print("Creating recursive chunks...")

    chunks = create_chunks(documents)

    print(f"Created {len(chunks)} chunks.")

    print("Creating embeddings...")

    chunks = create_embeddings(chunks)

    print("RAG is ready.")

    while True:

        query = input(
            "\nAsk a question (type 'exit' to stop): "
        )

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


if __name__ == "__main__":
    main()
