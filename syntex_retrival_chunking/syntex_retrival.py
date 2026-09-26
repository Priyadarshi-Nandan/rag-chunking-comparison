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
# 3. SPLIT LARGE PARAGRAPH INTO SENTENCES
# ============================================================

def split_sentences(text):

    return re.split(
        r'(?<=[.!?])\s+',
        text
    )


# ============================================================
# 4. RECURSIVELY SPLIT A LARGE SECTION
# ============================================================

def recursive_split(text, chunk_size):

    text = text.strip()

    if len(text) <= chunk_size:

        return [text]

    # ----------------------------------------
    # Try paragraphs
    # ----------------------------------------

    paragraphs = text.split("\n\n")

    if len(paragraphs) > 1:

        chunks = []
        current = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            if len(paragraph) > chunk_size:

                if current:

                    chunks.append(
                        current.strip()
                    )

                    current = ""

                chunks.extend(
                    recursive_split(
                        paragraph,
                        chunk_size
                    )
                )

            elif len(current) + len(paragraph) + 2 <= chunk_size:

                if current:
                    current += "\n\n"

                current += paragraph

            else:

                if current:

                    chunks.append(
                        current.strip()
                    )

                current = paragraph

        if current:

            chunks.append(
                current.strip()
            )

        return chunks

    # ----------------------------------------
    # Try sentences
    # ----------------------------------------

    sentences = split_sentences(text)

    if len(sentences) > 1:

        chunks = []
        current = ""

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            if len(sentence) > chunk_size:

                if current:

                    chunks.append(
                        current.strip()
                    )

                    current = ""

                chunks.extend(
                    recursive_split(
                        sentence,
                        chunk_size
                    )
                )

            elif len(current) + len(sentence) + 1 <= chunk_size:

                if current:
                    current += " "

                current += sentence

            else:

                if current:

                    chunks.append(
                        current.strip()
                    )

                current = sentence

        if current:

            chunks.append(
                current.strip()
            )

        return chunks

    # ----------------------------------------
    # Last option: words
    # ----------------------------------------

    words = text.split()

    chunks = []

    current = ""

    for word in words:

        if len(current) + len(word) + 1 <= chunk_size:

            if current:
                current += " "

            current += word

        else:

            if current:

                chunks.append(
                    current.strip()
                )

            current = word

    if current:

        chunks.append(
            current.strip()
        )

    return chunks


# ============================================================
# 5. STRUCTURAL + RECURSIVE CHUNKING
# ============================================================
#
# Detects TWO heading styles:
#   - Markdown-style:  # Heading / ## Heading / ### Heading
#   - Chapter-style:   Chapter 1: Heading Name
#
# This matters because plain-text/PDF documents (like a book or
# report) usually use "Chapter N:" headings, not Markdown syntax.
# The original version only matched Markdown, so on a Chapter-style
# document it never found a heading and quietly behaved like plain
# recursive chunking. This version catches both.
# ============================================================

def chunk_text(text, chunk_size=500):

    # Split before a Markdown heading OR a "Chapter N:" heading
    sections = re.split(
        r'(?=^#{1,3}\s|^Chapter\s+\d+:)',
        text,
        flags=re.MULTILINE
    )

    chunks = []

    for section in sections:

        section = section.strip()

        if not section:
            continue

        # ----------------------------------------
        # Find heading (Markdown OR "Chapter N: ...")
        # ----------------------------------------

        heading_match = re.match(
            r'^(#{1,3}\s+.+|Chapter\s+\d+:.+)',
            section
        )

        if heading_match:

            heading = heading_match.group(1).strip()

            content = section[
                heading_match.end():
            ].strip()

        else:

            heading = ""

            content = section

        # ----------------------------------------
        # Recursively split content
        # ----------------------------------------

        content_chunks = recursive_split(
            content,
            chunk_size
        )

        # ----------------------------------------
        # Add heading to every chunk
        # ----------------------------------------

        for chunk in content_chunks:

            if heading:

                final_text = (
                    heading +
                    "\n\n" +
                    chunk
                )

            else:

                final_text = chunk

            chunks.append({
                "text": final_text,
                "section": heading
            })

    return chunks


# ============================================================
# 6. CREATE CHUNKS
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
                "text": chunk["text"],
                "section": chunk["section"]
            })

    return chunks


# ============================================================
# 7. EMBEDDING
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
# 8. CREATE EMBEDDINGS
# ============================================================

def create_embeddings(chunks):

    for chunk in chunks:

        chunk["embedding"] = embed_text(
            chunk["text"]
        )

    return chunks


# ============================================================
# 9. COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# ============================================================
# 10. RETRIEVAL
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
            "section": chunk["section"],
            "score": score
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# 11. GENERATE ANSWER
# ============================================================

def generate_answer(query, results):

    context = ""

    for result in results:

        context += (
            f"\nSource: {result['source']}\n"
            f"Section: {result['section']}\n"
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
# 12. MAIN RAG PIPELINE
# ============================================================

def main():

    print("Loading documents...")

    documents = load_documents()

    print("Creating structural + recursive chunks...")

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
