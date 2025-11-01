
import os
import re
import json
import asyncio
import hashlib
import numpy as np
import tiktoken
import spacy
from openai import AsyncOpenAI
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict

# ------------------------------
# SETUP
# ------------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("Please set your OPENAI_API_KEY environment variable.")

client = AsyncOpenAI(api_key=OPENAI_KEY)
nlp = spacy.load("en_core_web_sm")
enc = tiktoken.encoding_for_model("gpt-4-turbo")

# config knobs
TARGET_TOKENS = 800
MIN_TOKENS = 200
OVERLAP_SENTENCES = 2
DEDUP_THRESHOLD = 0.97
QUALITY_THRESHOLD = 0.5
BATCH_SIZE = 64
CONCURRENT_REQUESTS = 5


# ------------------------------
# TOKEN + TEXT HELPERS
# ------------------------------
def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def section_split(text: str) -> List[str]:
    """Split document into top-level sections by headings or numbering patterns."""
    pattern = r"(?:\n|^)(?:[A-Z][A-Z\s]{3,}|#+\s+.+|[0-9]+\.\s+.+|[IVX]+\.\s+.+)"
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p.strip()]


def token_density_function(text: str) -> float:
    """TDF = tokens per sentence, used for adaptive resizing."""
    doc = nlp(text)
    sents = [s.text for s in doc.sents]
    return count_tokens(text) / max(1, len(sents))


def clean_text_basic(text: str) -> str:
    """Remove boilerplate, ads, or policies."""
    t = re.sub(r"\s+", " ", text)
    for pat in [
        r"(?i)privacy policy", r"(?i)terms of use", r"(?i)subscribe",
        r"(?i)cookie policy", r"(?i)advertisement", r"(?i)© \d{4}"
    ]:
        t = re.sub(pat, "", t)
    return t.strip()


# ------------------------------
# QUALITY CHECK
# ------------------------------
def chunk_quality_score(text: str) -> float:
    tokens = max(1, count_tokens(text))
    doc = nlp(text)
    sents = list(doc.sents)
    n_sents = max(1, len(sents))
    alpha_ratio = sum(1 for c in text if c.isalpha()) / max(1, len(text))
    uniq_ratio = len(set(text.split())) / max(1, len(text.split()))
    completeness = 1.0 if text.strip().endswith(('.', '?', '!')) else 0.6

    # Weighted heuristic
    t_score = min(tokens / TARGET_TOKENS, 1.0)
    sent_score = min(n_sents / 4.0, 1.0)
    score = 0.25 * t_score + 0.25 * alpha_ratio + 0.20 * uniq_ratio + 0.20 * completeness + 0.10 * sent_score
    return float(max(0.0, min(1.0, score)))


# ------------------------------
# CHUNKING ENGINE
# ------------------------------
def hybrid_adaptive_chunking(text: str) -> List[Dict]:
    """Adaptive chunking with Token Density Function."""
    sections = section_split(text)
    chunks = []
    idx = 0

    for sec in sections:
        density = token_density_function(sec)
        # Adaptive resizing
        if density > 60:
            t_target = int(TARGET_TOKENS * 0.6)
        elif density < 25:
            t_target = int(TARGET_TOKENS * 1.3)
        else:
            t_target = TARGET_TOKENS

        doc = nlp(sec)
        sents = [s.text.strip() for s in doc.sents if s.text.strip()]
        cur, cur_toks = [], 0


        for i, s in enumerate(sents):
            t = count_tokens(s)
            if cur_toks + t > t_target and cur:
                chunk_text = " ".join(cur).strip()
                chunks.append({
                    "chunk_id": idx, "text": chunk_text, "tokens": cur_toks, "density": density
                })
                idx += 1
                overlap = sents[max(0, i - OVERLAP_SENTENCES):i]
                cur = overlap + [s]
                cur_toks = sum(count_tokens(x) for x in cur)
            else:
                cur.append(s)
                cur_toks += t

        if cur:
            chunk_text = " ".join(cur).strip()
            chunks.append({"chunk_id": idx, "text": chunk_text, "tokens": cur_toks, "density": density})
            idx += 1

    # Merge undersized chunks
    merged = []
    buf, buf_toks = "", 0
    for c in chunks:
        if c["tokens"] < MIN_TOKENS:
            buf += " " + c["text"]
            buf_toks += c["tokens"]
        else:
            if buf:
                merged.append({"chunk_id": None, "text": buf.strip(), "tokens": buf_toks})
                buf, buf_toks = "", 0
            merged.append(c)
    if buf:
        merged.append({"chunk_id": None, "text": buf.strip(), "tokens": buf_toks})

    # Assign final IDs
    for i, c in enumerate(merged):
        c["chunk_id"] = i

    return merged


# ------------------------------
# QUALITY REMEDIATION
# ------------------------------
def remediate_low_quality(chunks: List[Dict]) -> List[Dict]:
    """Improve chunks that score low by cleaning or re-chunking."""
    good, bad = [], []

    for c in chunks:
        score = chunk_quality_score(c["text"])
        c["quality"] = score
        if score >= QUALITY_THRESHOLD:
            good.append(c)
        else:
            bad.append(c)

    improved = []
    for c in bad:
        cleaned = clean_text_basic(c["text"])
        new_score = chunk_quality_score(cleaned)
        if new_score >= QUALITY_THRESHOLD:
            c["text"], c["quality"] = cleaned, new_score
            improved.append(c)
        else:
            # last resort: re-chunk into smaller pieces
            re_chunks = hybrid_adaptive_chunking(cleaned)
            for rc in re_chunks:
                rc_score = chunk_quality_score(rc["text"])
                if rc_score >= (QUALITY_THRESHOLD * 0.8):
                    rc["quality"] = rc_score
                    improved.append(rc)
    return good + improved


# ------------------------------
# OPENAI EMBEDDING DEDUP
# ------------------------------
async def embed_batches(texts: List[str], sem: asyncio.Semaphore) -> np.ndarray:
    """Async batched embeddings with retry/backoff."""
    results = [None] * len(texts)

    async def embed_batch(batch_texts: List[str], batch_idx: List[int]):
        backoff = 1
        for attempt in range(5):
            try:
                async with sem:
                    res = await client.embeddings.create(
                        model="text-embedding-3-small",
                        input=batch_texts
                    )
                for j, data in enumerate(res.data):
                    results[batch_idx[j]] = np.array(data.embedding)
                return
            except Exception as e:
                await asyncio.sleep(backoff)
                backoff *= 2

    batches = [texts[i:i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]
    idx_batches = [list(range(i, min(i + BATCH_SIZE, len(texts)))) for i in range(0, len(texts), BATCH_SIZE)]
    tasks = [embed_batch(b, ib) for b, ib in zip(batches, idx_batches)]
    sem_tasks = [asyncio.create_task(t) for t in tasks]
    await asyncio.gather(*sem_tasks)
    return np.vstack(results)

import math
from typing import Optional

# Merge threshold: if adjacent similarity >= this, they get merged
MERGE_THRESHOLD = 0.80
# Maximum tokens allowed for merged chunk (avoid too-large chunks)
MAX_MERGED_TOKENS = 1200

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    # safe cosine
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def semantic_coherence_merge(chunks: List[Dict],
                             merge_threshold: float = MERGE_THRESHOLD,
                             max_tokens: int = MAX_MERGED_TOKENS) -> List[Dict]:
    """
    Greedy adjacent merging of chunks based on embedding similarity.
    - chunks: list of dicts with keys 'text','tokens','embedding' (np.array) and optional metadata.
    - returns new list of chunks with merged texts and averaged embeddings.
    """
    if not chunks:
        return []

    # ensure embeddings are numpy arrays
    for c in chunks:
        if not isinstance(c.get("embedding"), np.ndarray):
            c["embedding"] = np.array(c.get("embedding", []))

    merged = []
    i = 0
    while i < len(chunks):
        current = chunks[i].copy()
        j = i + 1
        # try to greedily merge with subsequent adjacent chunks
        while j < len(chunks):
            next_chunk = chunks[j]
            # check size constraint
            combined_tokens = current["tokens"] + next_chunk["tokens"]
            if combined_tokens > max_tokens:
                break
            # similarity between current and next
            sim = _cosine(np.asarray(current["embedding"]), np.asarray(next_chunk["embedding"]))
            if sim >= merge_threshold:
                # merge: append text, sum tokens, average embeddings weighted by tokens
                combined_text = current["text"] + " " + next_chunk["text"]
                # weighted average embedding by token counts to bias longer chunk
                w1 = current["tokens"]
                w2 = next_chunk["tokens"]
                emb1 = np.asarray(current["embedding"])
                emb2 = np.asarray(next_chunk["embedding"])
                if emb1.size and emb2.size:
                    new_emb = (emb1 * w1 + emb2 * w2) / (w1 + w2)
                else:
                    new_emb = emb1 if emb1.size else emb2
                current["text"] = combined_text
                current["tokens"] = combined_tokens
                current["embedding"] = new_emb
                # merge other numeric fields conservatively (e.g., quality -> min)
                current["quality"] = min(current.get("quality", 1.0), next_chunk.get("quality", 1.0))
                # advance to next (coalesce further neighbors)
                j += 1
            else:
                break
        # finished merging chain starting at i
        merged.append(current)
        i = j
    # reassign chunk ids
    for idx, c in enumerate(merged):
        c["chunk_id"] = idx
    return merged

# Optional: re-embed merged chunks with OpenAI for higher accuracy
async def reembed_merged_chunks(merged_chunks: List[Dict], sem: asyncio.Semaphore) -> np.ndarray:
    """
    Recompute embeddings for merged_chunks using the same batch embedding helper.
    Expects merged_chunks to be list of dicts with 'text'.
    Updates chunk['embedding'] with new numpy arrays and returns embeddings array.
    """
    texts = [c["text"] for c in merged_chunks]
    if not texts:
        return np.array([])
    # reuse embed_batches function from earlier (async)
    embeddings = await embed_batches(texts, sem)  # embed_batches returns np.ndarray
    for i, c in enumerate(merged_chunks):
        c["embedding"] = embeddings[i]
    return embeddings



async def deduplicate_chunks(chunks: List[Dict], embeddings: np.ndarray) -> List[Dict]:
    """
    Remove near-duplicate chunks via cosine similarity.

    Strategy:
    1) Run one dedup pass using provided `embeddings`.
    2) Merge adjacent semantically-coherent chunks.
    3) If merges occurred, re-embed merged chunks and run a second dedup pass.
       (No recursion — controlled two-pass approach.)
    """
    def _dedup_pass(chunks_in: List[Dict], embs: np.ndarray) -> List[Dict]:
        """Synchronous helper: run single-pass dedup based on similarity matrix."""
        sims = cosine_similarity(embs)  # returns NxN
        used = set()
        out = []
        for i in range(len(chunks_in)):
            if i in used:
                continue
            dupes = np.where(sims[i] > DEDUP_THRESHOLD)[0]
            used.update(dupes.tolist())
            # attach embedding as python list for later JSON-compatibility / re-use
            chunks_in[i]["embedding"] = embs[i].tolist()
            out.append(chunks_in[i])
        return out

    # Defensive: if there are zero chunks, return immediately
    if len(chunks) == 0:
        return []

    # PASS 1: deduplicate using incoming embeddings
    final_chunks = _dedup_pass(chunks, embeddings)

    # 1) Merge adjacent similar chunks (this may reduce or change chunk boundaries)
    merged = semantic_coherence_merge(final_chunks, merge_threshold=0.80, max_tokens=1200)

    # If merging produced identical set (no change), we can return final_chunks already deduped.
    # But still safe to re-embed if you want higher accuracy — here we only re-embed when merged size differs
    if len(merged) == len(final_chunks):
        # no merges — final_chunks are already deduplicated
        return final_chunks

    # 2) Re-embed merged chunks for accuracy (async)
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)
    # Expectation: reembed_merged_chunks will set `chunk["embedding"]` for each merged chunk,
    # but we will also accept returning an ndarray if implemented that way.
    re_embs = await reembed_merged_chunks(merged, sem)

    # Normalize re_embs: support both in-place modification and ndarray returns
    if re_embs is None:
        # assume merged[i]["embedding"] was set by reembed_merged_chunks
        embs = np.vstack([np.array(c["embedding"]) for c in merged])
    else:
        # if reembed returned an ndarray, use it and also attach embeddings to chunks
        embs = np.asarray(re_embs)
        for i, c in enumerate(merged):
            c["embedding"] = embs[i].tolist()

    # PASS 2: a final dedup pass on merged+reembedded chunks
    final = _dedup_pass(merged, embs)
    return final



# ------------------------------
# MAIN PIPELINE
# ------------------------------
async def process_text(text: str) -> List[Dict]:
    """Core end-to-end pipeline."""
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

    # 1. Chunk
    chunks = hybrid_adaptive_chunking(text)
    # 2. Remediate low-quality chunks
    chunks = remediate_low_quality(chunks)
    # 3. Embed + deduplicate
    texts = [c["text"] for c in chunks]
    embeddings = await embed_batches(texts, sem)
    final = await deduplicate_chunks(chunks, embeddings)
    return final


# ------------------------------
# RUN EXAMPLE
# ------------------------------
async def main():
    sample = """
    1. Market Outlook
    The US economy continues to expand with strong employment growth.
    However, rising bond yields have begun to weigh on equities.

    2. Bank Performance
    JPMorgan and other large banks reported higher net interest income this quarter,
    although trading revenue softened. Analysts remain divided about the broader implications
    for the financial sector.
    """

    samples = [sample]
    # create a list of coroutines
    tasks = [process_text(s) for s in samples]
    # run them concurrently
    results_list = await asyncio.gather(*tasks)  # results_list is a list of lists
    return results_list

def main_():
    all_results = asyncio.run(main())
    return all_results


# ---------------------
# helper: flatten nested result
# ---------------------
def flatten_results(all_results):
    # sometimes you get [[{...}, {...}]] or [{...}, {...}]
    if len(all_results) == 0:
        return []
    # If outer element is a list of chunks already, flatten one level
    if isinstance(all_results[0], list):
        return [c for sub in all_results for c in sub]
    return all_results

# ---------------------
# helper: get query embedding (wrap your existing embed function)
# ---------------------
async def get_query_embedding(text: str, sem: asyncio.Semaphore):
    # adapt to your existing embed API. Example assumes embed_batches(texts, sem) exists and returns ndarray
    res = await embed_batches([text], sem)   # returns shape (1, dim)
    return np.asarray(res)[0]
import numpy as np

def cosine_sim(a, b):
    # assume 1-D numpy arrays
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))

def normalize_scores(arr):
    arr = np.array(arr, dtype=float)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    return (arr - arr.min()) / (arr.max() - arr.min())

def select_chunks_for_prompt(chunks, query_emb, top_k=5, token_budget=1500,
                             alpha=0.7, use_mmr=True, mmr_lambda=0.7):
    """
    chunks: list of dicts, each must have keys:
      - 'text', 'tokens' (int), 'quality' (0..1), 'embedding' (np.ndarray or list)
    query_emb: np.ndarray (query embedding)
    top_k: maximum number of chunks to consider before token budget trimming
    token_budget: available tokens for chunk text (after accounting for prompt & answer)
    alpha: weight for similarity vs quality in combined score
    use_mmr: whether to apply MMR for diversity
    mmr_lambda: higher -> prefer relevance, lower -> prefer diversity
    """
    # prepare arrays
    embs = np.vstack([np.array(c['embedding']) for c in chunks])
    sims = np.array([cosine_sim(query_emb, embs[i]) for i in range(len(chunks))])
    qualities = np.array([float(c.get('quality', 0.5)) for c in chunks])
    # normalize ranges
    sims_n = normalize_scores(sims)
    qual_n = normalize_scores(qualities)
    combined = alpha * sims_n + (1 - alpha) * qual_n

    idxs = np.argsort(-combined)  # descending by combined score

    # Option A: simple top-k then enforce token budget
    candidate_idxs = idxs[:top_k].tolist()

    # Option B: MMR greedy selection to reduce redundancy (applied on top_k candidates)
    if use_mmr and len(candidate_idxs) > 1:
        selected = []
        candidate_embs = embs[candidate_idxs]
        candidate_scores = combined[candidate_idxs]
        # start with highest combined score
        first = 0
        selected.append(candidate_idxs[first])
        remaining = candidate_idxs[:first] + candidate_idxs[first+1:]
        while remaining and len(selected) < top_k:
            mmr_scores = []
            for idx in remaining:
                sim_to_query = combined[idx]  # relevance (already normalized blended)
                # compute max similarity to already selected
                max_sim_to_selected = max(
                    cosine_sim(embs[idx], embs[s]) for s in selected
                )
                mmr_score = mmr_lambda * sim_to_query - (1 - mmr_lambda) * max_sim_to_selected
                mmr_scores.append((mmr_score, idx))
            mmr_scores.sort(reverse=True, key=lambda x: x[0])
            selected.append(mmr_scores[0][1])
            remaining.remove(mmr_scores[0][1])
        candidate_idxs = selected

    # respect token budget: select greedily until budget exhausted
    chosen = []
    used_tokens = 0
    for i in candidate_idxs:
        t = int(chunks[i].get('tokens', len(chunks[i]['text'].split())))
        if used_tokens + t > token_budget:
            # optionally: try to include a trimmed/summarized snippet; skip otherwise
            continue
        chosen.append(i)
        used_tokens += t

    # final output: ordered by combined score
    chosen = sorted(chosen, key=lambda x: -combined[x])
    return [chunks[i] for i in chosen]

# ---------------------
# build prompt & call LLM
# ---------------------

async def chat_complete(system_prompt: str, user_prompt: str, sem: asyncio.Semaphore):
    """
    Sends the system and user prompt to the LLM asynchronously.
    Returns the model's text reply.
    """
    async with sem:  # respect concurrency limit
        response = await client.chat.completions.create(
            model="gpt-4o-mini",     # or "gpt-4o" / "gpt-4-turbo" / any chat-capable model you prefer
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,         # low = factual
            max_tokens=500           # cap response length
        )

    # extract plain text
    return response.choices[0].message.content.strip()
# =====================
# EVALUATION PROMPT TEMPLATE
# =====================
EVALUATOR_SYSTEM_PROMPT = """
You are a meticulous evaluator for an AI RAG system.
Judge whether the assistant's answer is correct and grounded in the provided context.

Respond ONLY in JSON:
{
  "answer_quality": "High | Medium | Low",
  "hallucination_detected": true/false,
  "grounded_in_context": true/false,
  "coverage_complete": true/false,
  "citation_accuracy": "Good | Partial | Incorrect",
  "feedback": "short 2–3 sentence explanation"
}
"""

def make_evaluator_prompt(question, chunks, answer):
    context_text = "\n\n".join([f"[chunk_id={c['chunk_id']}]\n{c['text']}" for c in chunks])
    return f"""
Question:
{question}

Context:
{context_text}

Model Answer:
{answer}
"""


# =====================
# MAIN ANSWER PIPELINE
# =====================
async def answer_query(all_results, user_question: str,
                       top_k=6, token_budget=1200, alpha=0.7):
    sem = asyncio.Semaphore(CONCURRENT_REQUESTS)

    # 1️⃣ Flatten
    chunks = flatten_results(all_results)

    # 2️⃣ Compute query embedding
    query_emb = await get_query_embedding(user_question, sem)

    # 3️⃣ Select best chunks
    selected_chunks = select_chunks_for_prompt(
        chunks, query_emb, top_k=top_k, token_budget=token_budget, alpha=alpha
    )

    # 4️⃣ Build the final prompt
    context_parts = []
    for c in selected_chunks:
        if isinstance(c.get("embedding"), np.ndarray):
            c["embedding"] = c["embedding"].tolist()
        context_parts.append(f"[chunk_id={c['chunk_id']}] {c['text'].strip()}")
    context_block = "\n\n---\n\n".join(context_parts)

    system_instruction = (
        "You are an expert financial research assistant. Use only the provided context chunks "
        "to answer the user's question. Cite the chunk_id(s) you used in square brackets "
        "after any factual statement. If context lacks info, say 'Insufficient information in context.'"
    )

    user_prompt = (
        f"User question: {user_question}\n\n"
        f"Context (only use this):\n{context_block}\n\n"
        "Answer concisely (max 300 words). Provide any assumptions and cite chunk_ids."
    )

    # 5️⃣ Get LLM answer
    answer = await chat_complete(system_instruction, user_prompt, sem)

    # 6️⃣ Evaluate the answer
    eval_prompt = make_evaluator_prompt(user_question, selected_chunks, answer)
    evaluation = await chat_complete(EVALUATOR_SYSTEM_PROMPT, eval_prompt, sem)

    # Return both answer and evaluation
    return {
        "answer": answer,
        "evaluation": evaluation,
        "used_chunks": [c["chunk_id"] for c in selected_chunks]
    }


# =====================
# RUN EXAMPLE
# =====================
if __name__ == "__main__":
    # assume main_() produces your chunk list
    all_results = main_()

    print(f"✅ Final {len(all_results)} chunks produced:\n")
    print(all_results)

    question = "How are US banks performing this quarter and why?"
    results = asyncio.run(answer_query(all_results, question))

    print("\n=== LLM ANSWER ===\n")
    print(results["answer"])

    print("\n=== EVALUATION ===\n")
    print(results["evaluation"])

    print("\n=== USED CHUNKS ===\n")
    print(results["used_chunks"])


# Step1 is Raw Document Ingestion. This should be basically an etc pipeline in itself,
# but I kept it a simple script to explore more at the RAG side. I used playwright lib to scrape 
# document content. I am dealing with text data currently but the etl pipeline would differ if its 
# for multi modal context.iter

# Ideally these contents would the context which the llm would pull from the augmented prompt.
# Next step was segment split. This is done by separating the doc into major logical boundaries which 
# are majorly the headings and subheadings of a content.

# And then comes calculating the token_density_function TDF which lets us know if the content is
# of dense, sparse or normal, if its sparse like long prose then we need to chunk them down and if dense
# then if we need to merge them up to the previous token. This is very vital in preserving the 
# context.

# Then we need to do QA before turing the doc into vector which is called as doc embedding.
# Here are the checks we need to perform in the hueristic scoring - 

# Enough tokens?
# Mostly letters (not garbage)?
# Unique words (no repetition)?
# Ends cleanly?

# If score < threshold (e.g., 0.5):
# Clean text (remove noise)
# Re-chunk smaller
# Flag low-quality chunks

# Then comes the embedding part which is supposed to be stored in vector database. I used OpenAI text-embedding-3-small
# for this

# Now comes the query time. 

# When a user asks:
# "How did JPMorgan’s trading revenue perform in Q2?"
# Steps:
# Embed the query → q_vec
# Vector search → find top-k most similar chunks (cosine similarity)
# Retrieve those chunks' text
# Construct prompt context:

# Give all of these to the llm and get the answer and then implement the feedback loop
# where we could  evaluate the system using retrieval precision, recall, and factual accuracy.    



# 🔹 Step-by-Step Process
# 1️⃣ Section Segmentation

# The text is first split by macro sections using headings, enumerations, or formatting cues:

# 1. Introduction
# ## Financial Results
# III. Outlook


# Helps isolate large thematic blocks (like “MD&A”, “Risk Factors”, etc.).

# Function: section_split()

# 2️⃣ Token Density Function (TDF)

# Calculates average tokens per sentence for each section.

# TDF = total_tokens / sentence_count


# If TDF is high, the section is dense (tables, figures, equations).

# If TDF is low, it’s narrative (long sentences, paragraphs).

# ✅ Used to adapt target chunk size dynamically:

# Density	Meaning	Adjustment
# > 60	Very dense section	Shrink chunk size → 60% of normal
# < 25	Very sparse section	Expand chunk size → 130% of normal
# 25–60	Balanced	Keep default target
# 3️⃣ Sentence-Aware Adaptive Chunking

# Within each section, spaCy is used to identify sentence boundaries.

# Sentences are appended to a temporary chunk until the target token limit is reached (e.g., ~800 tokens).

# When the limit is exceeded:

# The current chunk is finalized.

# The last few sentences (1–2) are overlapped into the next chunk (for context continuity).

# This yields human-like chunk boundaries — no sentence cutoffs or abrupt splits.

# 4️⃣ Undersized Chunk Merging

# After the first pass, chunks smaller than a minimum threshold (e.g., <200 tokens) are merged with their neighbors.

# This ensures all chunks are “embedding-efficient” — you don’t waste API calls on trivial fragments.

# 5️⃣ Semantic Deduplication (Hybrid Part)

# After chunk creation, OpenAI embeddings (text-embedding-3-small) are computed for all chunks.

# Cosine similarity between embeddings identifies near-duplicates (similarity > 0.97).

# Duplicate chunks are discarded — keeping only the first representative.

# Removes repeated intros, disclaimers, or boilerplate language (common in financial reports).

# ✅ This is what makes it “hybrid”:

# Structural + Token-aware (rule-based)

# Meaning-aware (embedding-based)

# 6️⃣ Output: High-Quality Chunks

# Each output chunk includes metadata:

# {
#   "chunk_id": 0,
#   "text": "...",
#   "tokens": 742,
#   "density": 31.2
# }


# These chunks are:

# Semantic units (self-contained)

# Sized appropriately for embedding models

# Deduplicated and overlap-adjusted


