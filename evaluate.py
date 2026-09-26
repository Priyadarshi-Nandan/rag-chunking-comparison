import os
import sys
import time
import importlib.util


# ============================================================
# SETTINGS
# ============================================================

# Each entry: (folder name, script filename)
METHODS = [
    ("fixed_size_chunking", "fixed.py"),
    ("sentence_size_chunking", "sentence.py"),
    ("paragraph_size_chunking", "paragraph.py"),
    ("retrival_chunking", "retrival.py"),
    ("syntex_retrival_chunking", "syntex_retrival.py"),
]

# Test questions with a keyword that MUST appear in a correct answer.
# This is a simple, honest way to auto-check correctness without
# needing a human to read every single answer.
TEST_QUESTIONS = [
    {
        "question": "Which planet is the largest in our solar system?",
        "expected_keyword": "Jupiter"
    },
    {
        "question": "What is Olympus Mons?",
        "expected_keyword": "volcano"
    },
    {
        "question": "How long does Mercury take to orbit the Sun?",
        "expected_keyword": "87.97"
    },
    {
        "question": "What is the Great Red Spot?",
        "expected_keyword": "storm"
    },
    {
        "question": "Which planet rotates on its side?",
        "expected_keyword": "Uranus"
    },
    {
        "question": "What is Mount Everest?",
        "expected_keyword": "I don't"  # not in the docs -> should abstain
    },
]


# ============================================================
# LOAD A CHUNKING METHOD'S SCRIPT AS A MODULE
# ============================================================

def load_module(folder, filename):
    path = os.path.join(folder, filename)
    module_name = folder  # unique name so imports don't clash
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# RUN ONE METHOD AGAINST ALL TEST QUESTIONS
# ============================================================

def run_method(folder, filename):

    print(f"\n{'=' * 60}")
    print(f"METHOD: {folder}")
    print(f"{'=' * 60}")

    # Run from inside the method's own folder so its
    # DOCS_FOLDER = "docs" resolves correctly.
    original_dir = os.getcwd()
    os.chdir(folder)

    try:
        module = load_module(".", filename)

        print("Loading documents...")
        documents = module.load_documents()

        print("Creating chunks...")
        chunks = module.create_chunks(documents)
        print(f"Created {len(chunks)} chunks.")

        print("Creating embeddings...")
        start = time.time()
        chunks = module.create_embeddings(chunks)
        embed_time = time.time() - start

        results = []

        for item in TEST_QUESTIONS:
            question = item["question"]
            keyword = item["expected_keyword"]

            start = time.time()
            retrieved = module.retrieve(question, chunks, module.TOP_K)
            answer = module.generate_answer(question, retrieved)
            query_time = time.time() - start

            top_chunk_text = retrieved[0]["text"] if retrieved else ""

            retrieval_hit = keyword.lower() in top_chunk_text.lower()
            answer_hit = keyword.lower() in answer.lower()

            results.append({
                "question": question,
                "expected_keyword": keyword,
                "retrieval_hit": retrieval_hit,
                "answer_hit": answer_hit,
                "answer": answer.strip(),
                "query_time": round(query_time, 2)
            })

            print(f"\nQ: {question}")
            print(f"Retrieval hit: {retrieval_hit} | Answer hit: {answer_hit} | Time: {query_time:.2f}s")

        return {
            "method": folder,
            "chunk_count": len(chunks),
            "embed_time": round(embed_time, 2),
            "results": results
        }

    finally:
        os.chdir(original_dir)


# ============================================================
# BUILD MARKDOWN REPORT
# ============================================================

def build_report(all_results):

    lines = []
    lines.append("# Chunking Strategy Comparison Results\n")
    lines.append(f"Tested {len(TEST_QUESTIONS)} questions across {len(all_results)} chunking methods.\n")

    # Summary table: accuracy per method
    lines.append("## Summary\n")
    lines.append("| Method | Chunks Created | Embed Time (s) | Retrieval Hits | Answer Hits |")
    lines.append("|---|---|---|---|---|")

    for r in all_results:
        retrieval_hits = sum(1 for q in r["results"] if q["retrieval_hit"])
        answer_hits = sum(1 for q in r["results"] if q["answer_hit"])
        total = len(r["results"])
        lines.append(
            f"| {r['method']} | {r['chunk_count']} | {r['embed_time']} | "
            f"{retrieval_hits}/{total} | {answer_hits}/{total} |"
        )

    # Detailed breakdown per method
    lines.append("\n## Detailed Results\n")
    for r in all_results:
        lines.append(f"### {r['method']}\n")
        lines.append("| Question | Retrieval Hit | Answer Hit | Time (s) | Answer |")
        lines.append("|---|---|---|---|---|")
        for q in r["results"]:
            answer_preview = q["answer"].replace("\n", " ")[:100]
            lines.append(
                f"| {q['question']} | {q['retrieval_hit']} | {q['answer_hit']} | "
                f"{q['query_time']} | {answer_preview}... |"
            )
        lines.append("")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    all_results = []

    for folder, filename in METHODS:
        result = run_method(folder, filename)
        all_results.append(result)

    report = build_report(all_results)

    output_path = os.path.join("results", "comparison_results.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n\nDone. Report saved to {output_path}")


if __name__ == "__main__":
    main()
