# RAG Chunking Strategy Comparison

Five versions of the same basic RAG pipeline, identical in every way except the chunking method, tested against the same documents and the same questions. The goal is to see, with actual numbers rather than assumptions, how much chunking strategy really affects retrieval and answer quality.

## Folder structure

```
simple_rag/
├── fixed_size_chunking/
│   ├── fixed.py
│   └── docs/            (same test documents, copied into every folder)
├── sentence_size_chunking/
│   ├── sentence.py
│   └── docs/
├── paragraph_size_chunking/
│   ├── paragraph.py
│   └── docs/
├── retrival_chunking/
│   ├── retrival.py       (recursive: paragraph -> sentence -> word fallback)
│   └── docs/
├── syntex_retrival_chunking/
│   ├── syntex_retrival.py (heading-aware + recursive fallback)
│   └── docs/
├── evaluate.py            (runs all 5 methods against the same test questions)
└── results/
    └── comparison_results.md   (auto-generated report)
```

Each folder is a fully independent, runnable RAG script — same embedding model, same LLM, same retrieval/generation logic. The only thing that changes between folders is the `chunk_text()` function.

## How the evaluation works

```
evaluate.py
  -> for each of the 5 method folders:
       -> enter that folder (so its own "docs" path resolves correctly)
       -> load that folder's script as a module
       -> run its load_documents() -> create_chunks() -> create_embeddings()
       -> for each of 6 fixed test questions:
            -> run that method's retrieve() and generate_answer()
            -> check: did the correct keyword appear in the top retrieved chunk?
            -> check: did the correct keyword appear in the final answer?
       -> record chunk count, embedding time, and per-question results
  -> write everything to results/comparison_results.md
```

The point of checking retrieval and the final answer separately is to tell apart two different kinds of failure: the right evidence never being found (a retrieval problem) versus the right evidence being found but the model still answering wrong (a generation problem). These are different bugs with different fixes, and conflating them hides which part of the pipeline actually needs work.

## Stack and links

| Component | Tool | Link |
|---|---|---|
| Local LLM runtime | Ollama | https://ollama.com |
| Generation model | qwen2.5:3b | https://ollama.com/library/qwen2.5 |
| Embedding model | nomic-embed-text | https://ollama.com/library/nomic-embed-text |
| PDF parsing | pypdf | https://pypi.org/project/pypdf |

## Setup

If you want to run this on your own computer, follow these steps:

1. Clone this repository
2. Install Ollama: https://ollama.com/download
3. Pull models:
   ```
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text
   ```
4. Create and activate a virtual environment at the project root:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
5. Install dependencies:
   ```
   pip install requests numpy pypdf
   ```
6. Run any single method directly, e.g.:
   ```
   cd fixed_size_chunking
   python fixed.py
   ```
7. Or run the full comparison from the project root:
   ```
   python evaluate.py
   ```

## Results

Six test questions were run against all five methods, using the same solar system document. Two of the questions test straightforward factual lookup, one tests a fact buried in an indirect chunk, and one ("What is Mount Everest?") isn't answerable from the documents at all — it's there specifically to check whether each method correctly abstains instead of guessing.

| Method | Chunks Created | Embed Time (s) | Retrieval Hits (top-1) | Answer Hits |
|---|---|---|---|---|
| fixed_size_chunking | 81 | 174.92 | 4/6 | 5/6 |
| sentence_size_chunking | 65 | 139.87 | 4/6 | 5/6 |
| paragraph_size_chunking | 50 | 108.35 | 3/6 | 5/6 |
| retrival_chunking | 105 | 225.14 | 5/6 | 5/6 |
| syntex_retrival_chunking | 113 | 242.44 | 3/6 | 5/6 |

### A more accurate reading of these results

Two things about the evaluation script itself need to be accounted for before trusting the raw numbers above:

- **The Mount Everest question shows as an "answer miss" for every single method, but that's a flaw in the checker, not the RAG systems.** Every method correctly abstained (answers like "the context does not contain information about Mount Everest"), which is exactly the right behavior. The script was checking for the literal phrase "I don't," which none of the models happened to use verbatim. Once that's accounted for, every method actually got **6/6 answer accuracy**, not 5/6.

- **"Retrieval Hits" only checked whether the correct chunk was ranked in the number 1 position, not whether it appeared anywhere in the top 3 chunks that were actually sent to the model.** This understates methods where the right chunk was found but ranked second or third — which explains cases where a method shows a retrieval "miss" but still produced a fully correct answer (the model still had the right evidence, just not ranked first). A fairer retrieval metric would check the whole top-k list, not only the top slot.

**What the results honestly show once those two issues are kept in mind:** all five chunking methods produced correct final answers on every answerable question, and correctly abstained on the one unanswerable question. On this particular document, `retrival_chunking` (recursive, paragraph → sentence → word fallback) had the cleanest top-1 retrieval performance. Interestingly, the more sophisticated heading-aware method (`syntex_retrival_chunking`) did not come out ahead on top-1 retrieval here — likely because the chapters in this particular document are already short and self-contained, so structure-awareness had less to fix than it would on a messier or longer document. That's a useful, honest finding in itself: a more advanced chunking method isn't automatically better on every document, its advantage shows up specifically when documents have real structural boundaries that a naive chunker would otherwise cut through.

## Known limitations

- The evaluation only ran 6 questions on one document — not enough to draw strong general conclusions, just enough to compare these five methods directly under the same conditions.
- Retrieval-hit checking only looked at the top-ranked chunk, not the full top-k list (noted above).
- No reranking or hybrid search in any of these five versions — this is a pure chunking comparison, kept deliberately simple.

## Future plans

- Fix retrieval-hit checking to look at the full top-k list, not just position 1
- Test against a longer, messier document where structural chunking should show a clearer advantage
- Add hybrid search and reranking to the winning method
- Expand the test set beyond 6 questions

---

## My journey building this

### Why I built this at all

After getting the heading-aware chunking fix working in my main industry-style RAG project, I wanted to actually check whether all the chunking methods I'd read about mattered as much as people say, instead of just assuming the fancier one is always better. Rather than rebuilding one RAG system five times over, I set it up as five small, independent, side-by-side versions that only differ in one function.

### Setting up the comparison

Each folder is basically the same script copy-pasted, with only `chunk_text()` swapped out — fixed word count, sentence grouping, paragraph grouping, plain recursive fallback, and heading-aware recursive fallback. Getting them to run cleanly meant fixing a document-loader bug (originally PDF-only, needed to also accept `.txt`), and fixing a heading-detection bug in the structural version that only recognized Markdown headers, not the `Chapter 1:` style headings my actual test document uses.

### Building the actual evaluation

Writing `evaluate.py` was the more interesting part. It loads each folder's script as a separate Python module using `importlib`, temporarily switches into that folder so its own local `docs` path resolves correctly, runs the full pipeline, and checks two things separately for every test question: was the right chunk retrieved, and was the right answer generated. Keeping those two checks separate mattered — a method can retrieve the right evidence and still generate a wrong answer, or generate a right-sounding answer even when the "official" top chunk was technically something else.

### What the first real run actually showed

The first full run genuinely surprised me a bit. I expected the heading-aware method to clearly win, since that's the one that fixed a real bug earlier. Instead, all five methods answered almost everything correctly, and the simplest recursive method actually had the best top-1 retrieval numbers on this particular document. Digging into why, it came down to my own evaluation script only checking the top-ranked chunk instead of the full top-3 — a few "misses" were really just the right chunk sitting in second or third place, still used correctly by the model. That was a useful reminder that a benchmark is only as trustworthy as the metric it's built on, not just the system it's measuring.

### Where it's at now

All five methods work, are documented honestly including where the benchmark itself has known blind spots, and the whole comparison is reproducible with one command. Next step is pushing this to GitHub, then going back to the main industry-level RAG project to keep building forward.
