# Rerankers and Retrieval Ranking — Comprehensive Technical Guide

> Lecture-note style reference for ST5230 / Binance interview preparation.
> Assumes familiarity with transformer architectures, embeddings, and basic IR concepts.

---

## Table of Contents

1. Why Rerankers? The Two-Stage Retrieval Paradigm
2. Bi-Encoders (Retriever) — Limitations
3. Cross-Encoders (Reranker) — Deep Dive
4. ColBERT — Late Interaction
5. MonoT5 / RankT5
6. LLM-based Rerankers
7. Training Rerankers
8. Evaluation Metrics — Full Derivations
9. Reciprocal Rank Fusion (RRF)
10. Hybrid Search Architecture
11. Problems & Mitigations
12. Industry Practices
13. Full System Design: Production Retrieval + Reranking
14. Interview Q&A
15. Coding Problems

---

## 1. Why Rerankers? The Two-Stage Retrieval Paradigm

### 1.1 The Core Intuition

Information retrieval has always faced a fundamental tension: you want results that are **accurate** and you want them **fast**. These two goals conflict at scale.

Consider a corpus of 10 million documents. A user submits a query. Ideally you would compare the query against every document using the most powerful comparison function available. In practice, that is computationally infeasible within a 100ms latency budget.

The solution is **two-stage retrieval**:

1. **Stage 1 — Retrieval (fast, coarse):** Use a lightweight model to reduce 10M documents to ~1000 candidates. Speed is the priority here.
2. **Stage 2 — Reranking (slow, precise):** Use a powerful model to reorder the 1000 candidates and return the top-10. Quality is the priority here.

### 1.2 The Two-Stage Pipeline

```
                         Stage 1: Retrieval                    Stage 2: Reranking
                    (Bi-encoder or BM25)                     (Cross-encoder)

                ┌─────────────────────────────┐          ┌──────────────────────┐
                │                             │          │                      │
Query ─────────►│  Bi-encoder / BM25 Index   │─────────►│   Cross-Encoder      │─────► Top-10
                │  (pre-computed doc vecs)    │  Top-1000│   (joint encoding)   │       Final
                │                             │  candidates                    │       Results
                └─────────────────────────────┘          └──────────────────────┘

Latency:              ~5–20ms                                  ~50–200ms
Corpus coverage:      All 10M docs                             1000 candidates only
Comparison fn:        Dot product                              Full transformer forward pass
```

### 1.3 The Fundamental Tradeoff: Speed vs Accuracy

| Model Type      | Latency (10M docs) | Quality  | Pre-compute docs? |
|-----------------|--------------------|----------|-------------------|
| BM25            | ~5ms               | Medium   | Yes (inverted idx)|
| Bi-encoder      | ~10ms (ANN)        | Good     | Yes               |
| ColBERT         | ~50ms              | Very good| Yes (per-token)   |
| Cross-encoder   | O(N) forward passes| Excellent| No                |
| LLM reranker    | Seconds            | Best     | No                |

### 1.4 Why Not Use Cross-Encoder for Everything?

Complexity analysis: suppose your corpus has $N$ documents and the query+document concatenation has $L$ tokens on average. The transformer self-attention is $O(L^2)$ per layer and per document. For $N$ documents:

$$\text{Cost}_{CE} = O(N \cdot L^2 \cdot \text{layers})$$

For $N = 10^7$, $L = 512$, layers = 12, this is approximately $10^7 \times 2.6 \times 10^5 \times 12 \approx 3 \times 10^{13}$ operations per query. At modern GPU speeds (~$10^{13}$ FLOPs/sec) this takes ~3 seconds — 30x over budget.

The bi-encoder pre-computes document embeddings offline:

$$\text{Cost}_{BE} = O(L^2 \cdot \text{layers}) \text{ at query time} + O(N \cdot d) \text{ for ANN search}$$

where $d$ is embedding dimension. The ANN search is near-linear in $N$.

### 1.5 Analogy

Think of it like hiring: a recruiter (bi-encoder) quickly screens 10,000 resumes using keyword matching and produces 50 finalists. The hiring manager (cross-encoder) then carefully reads all 50 and ranks them. You do not have the hiring manager read all 10,000 — that would take months.

---

## 2. Bi-Encoders (Retriever) — Limitations

### 2.1 Architecture

A bi-encoder encodes query and document **independently** through separate (or shared) transformer encoders:

```
Query: "bitcoin price prediction"          Document: "BTC/USDT analysis..."
         │                                              │
    ┌────▼────┐                                   ┌────▼────┐
    │ Encoder │                                   │ Encoder │
    │  (BERT) │                                   │  (BERT) │
    └────┬────┘                                   └────┬────┘
         │ E_q ∈ R^d                                    │ E_d ∈ R^d
         └──────────────── dot product ─────────────────┘
                                 │
                            s(q,d) scalar
```

The scoring function is:

$$s(q, d) = \phi(q) \cdot \psi(d) = E_q^T E_d$$

where $\phi$ and $\psi$ are encoder functions mapping text to $\mathbb{R}^d$.

### 2.2 The Independence Assumption

The fundamental limitation is that query and document are encoded **independently** — there is no cross-attention between query tokens and document tokens. All information about relevance must be compressed into two fixed-size vectors before any comparison occurs.

This is the **representational bottleneck**: the entire semantic content of a 512-token document must be captured in a single 768-dimensional vector. By the time the query and document "meet" (at the dot product), it is too late for fine-grained interaction.

### 2.3 Failure Modes

**Exact match failures.** Query: "What is the ticker symbol for Bitcoin?" Document containing "BTC is Bitcoin's ticker on most exchanges." A bi-encoder may not capture the specific intent to retrieve the abbreviation.

**Semantic mismatch.** Two documents may have similar embeddings but very different relevance for a specific query depending on context. E.g., "bear market" and "bull market" may be close in embedding space but completely opposite in relevance for "how to profit in a bear market."

**Multi-hop reasoning.** Query requires connecting two facts: "CEO of the company that created Ethereum." A bi-encoder compresses "Ethereum founding CEO" into one vector, which may not match documents that discuss Vitalik Buterin separately from Ethereum's founding.

**No term-level matching.** If the exact phrase "stop-loss order" appears in a document but the query is about "how to limit losses in crypto trading," a bi-encoder might miss the term-level signal that BM25 would catch.

### 2.4 Quality Ceiling

The bi-encoder quality ceiling exists because:

$$I(q; d | E_q, E_d) = 0$$

Once you have the two embeddings, all mutual information between query and document has been discarded. The dot product is just a single number computed from two vectors — it cannot recover fine-grained signals that were lost during compression.

### 2.5 ColBERT as a Middle Ground

ColBERT addresses this by representing each query and document as a **set of per-token embeddings** rather than a single vector, enabling richer interaction at scoring time without joint encoding. (Detailed in Section 4.)

---

## 3. Cross-Encoders (Reranker) — Deep Dive

### 3.1 Architecture

A cross-encoder takes the query and document **concatenated** as a single input:

$$\text{input} = \texttt{[CLS]}\ q\ \texttt{[SEP]}\ d\ \texttt{[SEP]}$$

This is fed through a full transformer encoder. The $\texttt{[CLS]}$ token's final hidden state is projected to a scalar score:

$$s(q, d) = \text{BERT}([\texttt{CLS}]\ q\ [\texttt{SEP}]\ d\ [\texttt{SEP}])_{\texttt{CLS}} \cdot w^T$$

where $w \in \mathbb{R}^H$ is a learned linear projection ($H$ = hidden size).

```
Input: [CLS] "bitcoin price" [SEP] "BTC is the leading cryptocurrency..." [SEP]
         │        │             │              │                               │
    ┌────┴────────┴─────────────┴──────────────┴───────────────────────────────┴────┐
    │                                                                                │
    │                    12-layer BERT Transformer                                   │
    │                    (full cross-attention across all tokens)                    │
    │                                                                                │
    │   Every query token attends to every document token and vice versa            │
    │                                                                                │
    └────┬────────────────────────────────────────────────────────────────────────┘
         │
    h_CLS ∈ R^768
         │
    ┌────▼────┐
    │  Linear │  w ∈ R^768
    └────┬────┘
         │
    s(q,d) ∈ R  (relevance score)
```

### 3.2 Why Cross-Attention Makes It Powerful

In a cross-encoder, the attention mechanism computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q, K, V$ are derived from the **full concatenated sequence**. This means:

- Token "price" in the query can directly attend to "valuation" in the document
- Token "bitcoin" in the query can attend to "BTC", "crypto", "digital asset" in the document
- Context from the document can modulate how query tokens are interpreted
- Negation, qualification, and complex semantics are handled naturally

This is the key insight: **every query token can attend to every document token in every layer**. The model can learn arbitrarily complex relevance patterns.

### 3.3 Complexity Analysis

For a concatenated sequence of length $L = L_q + L_d$:

- Self-attention per layer: $O(L^2)$
- For $N$ documents, must run $N$ forward passes: $O(N \cdot L^2 \cdot \text{layers})$
- Cannot pre-compute document representations: the document embedding depends on the query

This makes cross-encoders **infeasible as first-stage retrievers** for large corpora but **ideal as rerankers** for small candidate sets.

### 3.4 Training

**Binary relevance classification:**

$$\mathcal{L}_{BCE} = -y \log(\sigma(s)) - (1-y)\log(1 - \sigma(s))$$

where $y \in \{0, 1\}$ is the binary relevance label.

**Margin MSE loss (with soft labels from teacher):**

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} (s_i - \hat{s}_i)^2$$

where $\hat{s}_i$ are soft relevance scores from a teacher model (e.g., human annotations on a 0-3 scale normalized to [0,1]).

**Listwise ranking loss:**

$$\mathcal{L}_{LW} = -\sum_{q} \sum_{d^+ \in R_q^+} \log \frac{\exp(s(q, d^+))}{\sum_{d \in C_q} \exp(s(q, d))}$$

where $R_q^+$ is the set of relevant documents and $C_q$ is the candidate set for query $q$.

### 3.5 Inference

At inference time, for $K$ candidate documents:

```python
scores = []
for doc in candidates:  # K iterations, no vectorization trick
    score = cross_encoder.predict(query, doc)
    scores.append(score)
ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
```

The lack of pre-computation means inference is $O(K)$ forward passes. With $K=100$ and a 12-layer BERT, this is ~200ms on GPU — acceptable for a reranker but not a retriever.

### 3.6 Problems & Mitigations

| Problem | Mitigation |
|---------|-----------|
| High latency (O(K) forward passes) | Model distillation to smaller model (DistilBERT); TensorRT quantization; limit K |
| Cannot scale to full corpus | Use as second stage only; pre-filter with bi-encoder or BM25 |
| Long documents exceed context window | Sliding window chunking; hierarchical reranking |
| Training requires labeled pairs | Hard negative mining; synthetic data generation |

---

## 4. ColBERT — Late Interaction

### 4.1 Motivation

ColBERT (Contextualized Late Interaction over BERT) sits between bi-encoders and cross-encoders on the speed-quality tradeoff curve:

```
Speed:    BM25 ──── Bi-encoder ──── ColBERT ──── Cross-encoder ──── LLM reranker
Quality:  Low         Medium        High           Very high           Best
Pre-comp: Yes          Yes           Yes (costly)   No                  No
```

### 4.2 Architecture

ColBERT encodes query and document independently but **retains per-token embeddings** rather than pooling to a single vector:

**Query encoding:**
$$E_q = [e_{q_1}, e_{q_2}, \ldots, e_{q_m}] \quad \text{where each } e_{q_i} \in \mathbb{R}^d$$

**Document encoding:**
$$E_d = [e_{d_1}, e_{d_2}, \ldots, e_{d_n}] \quad \text{where each } e_{d_j} \in \mathbb{R}^d$$

**MaxSim scoring:**
$$s(q, d) = \sum_{i=1}^{m} \max_{j=1}^{n}\, e_{q_i}^T e_{d_j}$$

```
Query: "bitcoin price"             Document: "BTC valuation rose sharply"
         │        │                            │       │       │     │
    ┌────▼──┐ ┌───▼──┐               ┌────────▼─┐ ┌───▼──┐ ┌─▼──┐ ┌▼──┐
    │e_q1   │ │e_q2  │               │e_d1      │ │e_d2  │ │e_d3│ │e_d4│
    │(768d) │ │(768d)│               │(768d)    │ │(768d)│ │    │ │    │
    └────┬──┘ └──┬───┘               └──────────┘ └──────┘ └────┘ └────┘
         │       │
         │       │    MaxSim: for each query token, find max dot product with any doc token
         │       │
         ▼       ▼
    max(e_q1·e_d1, e_q1·e_d2, ...) + max(e_q2·e_d1, e_q2·e_d2, ...)
                                 │
                            s(q,d) = sum of per-token max similarities
```

### 4.3 Why MaxSim Works

For each query token $e_{q_i}$, MaxSim finds the **most relevant document token** it can align with. This is a soft, differentiable version of term matching:

- "bitcoin" in query aligns with "BTC" in document (highest dot product)
- "price" in query aligns with "valuation" in document
- Each query term finds its best match; missing terms penalize the score

This captures **term-level alignment signals** that single-vector bi-encoders miss, without requiring joint encoding.

### 4.4 Pre-computation and Indexing

Document token embeddings $E_d$ can be **pre-computed and stored**:

- Storage: for 10M documents, average 100 tokens each, 128-dim embeddings, float32: $10^7 \times 100 \times 128 \times 4 = 512$ GB
- This is the main drawback: ColBERT indices are **very large**

At query time:
1. Encode query tokens: $O(m^2 \cdot \text{layers})$ — fast
2. For each query token, retrieve top-K nearest document tokens using ANN: $O(m \cdot K \cdot d)$
3. Aggregate by document and compute MaxSim: $O(m \cdot n)$ per candidate

### 4.5 PLAID: Approximate ColBERT

PLAID (Performance-optimized Late Interaction Driver) reduces ColBERT's index size via:

1. **Centroid-based compression**: cluster all document token embeddings into $C$ centroids. Store only (centroid_id, residual) per token.
2. **Candidate generation**: for each query token, retrieve nearest centroids, then only score documents whose tokens are near those centroids.
3. **Approximate MaxSim**: use centroid distances as a proxy before computing exact MaxSim.

This reduces index size by 5-10x with minimal quality loss.

### 4.6 Problems & Mitigations

| Problem | Mitigation |
|---------|-----------|
| Large index (100x larger than bi-encoder) | PLAID compression; quantization (int8) |
| Complex indexing pipeline | Use ColBERT-v2 library with built-in PLAID |
| Slow exact retrieval at query time | Two-stage: centroid pre-filtering then MaxSim |
| Not supported by standard ANN libraries | Custom FAISS wrapper or Vespa/Weaviate with ColBERT support |

---

## 5. MonoT5 / RankT5

### 5.1 Sequence-to-Sequence Reranking

MonoT5 reformulates reranking as a **text generation task**. The input is a natural language prompt containing the query and document:

```
Input:  "Query: What is Bitcoin? Document: Bitcoin is a decentralized
         digital currency. Relevant:"

Output: "true"  or  "false"
```

The relevance score is the **log-probability of generating "true"**:

$$s(q, d) = \log P(\texttt{"true"} \mid \text{prompt}(q, d))$$

This is computed from the decoder's output distribution over vocabulary tokens.

### 5.2 Architecture

```
    Input prompt (query + document)
             │
    ┌────────▼────────┐
    │   T5 Encoder    │
    └────────┬────────┘
             │ encoder hidden states
    ┌────────▼────────┐
    │   T5 Decoder    │  (generates one token)
    └────────┬────────┘
             │
    P(token | input)
             │
    score = log P("true") - log P("false")  [optional normalization]
```

### 5.3 Advantages

- **Natural language interface**: the scoring criterion is expressed in the prompt, making it interpretable
- **Handles long documents**: can chunk long documents and aggregate scores
- **Cross-lingual**: mT5 (multilingual T5) generalizes zero-shot to non-English languages without retraining
- **Soft scoring**: log-probability is a continuous score, not just binary

### 5.4 RankT5

RankT5 fine-tunes T5 directly for ranking using **listwise losses** on the sequence of scores:

$$\mathcal{L}_{RankT5} = \text{SoftmaxCrossEntropy}(\text{scores}, \text{relevance\_labels})$$

It processes all candidates jointly, allowing the model to reason about relative relevance within the candidate set.

### 5.5 Problems & Mitigations

| Problem | Mitigation |
|---------|-----------|
| Slower than BERT-based cross-encoders (seq2seq overhead) | Use encoder-only logits for "true"/"false" tokens without full decoding |
| Context length limits (T5 default 512 tokens) | Use T5-3B with 1024 tokens; chunk and aggregate |
| Score calibration issues | Normalize using log P("true") - log P("false") |

---

## 6. LLM-based Rerankers

### 6.1 RankGPT

RankGPT prompts an LLM (GPT-4, LLaMA, etc.) to rerank documents. The key insight is that LLMs have world knowledge and instruction-following ability that specialized rerankers lack.

**Prompt format:**

```
System: You are a helpful assistant that ranks passages based on relevance to a query.

User:
I will provide you with 5 passages. Rank them by relevance to the query.

Query: "How to set a stop-loss on Binance futures?"

[1] To set a stop-loss on Binance, navigate to the futures trading interface...
[2] Binance supports various order types including limit and market orders...
[3] Risk management in crypto trading requires setting appropriate...
[4] Stop-loss orders automatically sell your position when price drops...
[5] Leverage trading on Binance Futures requires understanding margin...

Rank the passages from most to least relevant. Output: [ranking]
```

**Output:** `[1, 4, 2, 5, 3]`

### 6.2 Sliding Window Strategy

For large candidate lists (e.g., 100 documents), LLMs have context window limits. RankGPT uses a **sliding window**:

```
Step 1: Rank docs 1-10, get partial ranking R1
Step 2: Take bottom-5 of R1, combine with docs 11-15, rank → R2
Step 3: Continue sliding until all docs processed
Step 4: Merge partial rankings
```

This is $O(N / \text{window\_size})$ LLM calls with $O(\text{window\_size} \cdot L)$ tokens each.

### 6.3 Pointwise, Pairwise, and Listwise

**Pointwise reranking:** Score each document independently with the LLM, then sort by score.
- Pros: embarrassingly parallel, simple
- Cons: no cross-document comparison, scores poorly calibrated across queries

$$s_i = \text{LLM}(\text{prompt}(q, d_i)) \quad \text{for each } i$$

**Pairwise reranking:** Compare pairs of documents, aggregate into a ranking.

$$p_{ij} = P(d_i \succ d_j | q) = \text{LLM}(\text{pairwise\_prompt}(q, d_i, d_j))$$

Aggregate using a sorting algorithm (merge sort) or Bradley-Terry model:

$$\text{score}(d_i) = \sum_{j \neq i} \mathbb{1}[d_i \succ d_j]$$

Requires $O(N^2)$ LLM calls for $N$ documents — expensive.

**Listwise reranking:** Rank all documents at once.
- Pros: captures relative ordering, single LLM call
- Cons: context window limits, inconsistency at long lists, position bias

### 6.4 Cost vs Quality Ordering

$$\text{GPT-4 listwise} \succ \text{GPT-3.5 listwise} \succ \text{ColBERT} \approx \text{Cross-encoder} \succ \text{Bi-encoder} \succ \text{BM25}$$

In practice the gap between GPT-4 reranker and a well-trained cross-encoder is smaller than expected for domain-specific tasks where the cross-encoder has been fine-tuned.

### 6.5 Problems & Mitigations

| Problem | Mitigation |
|---------|-----------|
| Position bias: LLM prefers first/last documents in list | Permutation calibration: run multiple orderings and aggregate |
| Context window limits listwise input size | Sliding window; two-stage LLM reranking |
| Inconsistency: same query gives different rankings on re-runs | Temperature=0; ensemble multiple runs |
| Hallucinated relevance: LLM invents relevance signals not in documents | Constrain prompt to only consider provided text; citation grounding |
| Cost: $1-10 per 1000 queries with GPT-4 | Use smaller open-source LLMs (LLaMA-3 8B); cache results for common queries |

---

## 7. Training Rerankers

### 7.1 Training Data

**MS MARCO** (Microsoft Machine Reading Comprehension): 8.8M passages, 500K queries with sparse relevance labels. The gold standard for passage retrieval training.

**Natural Questions (NQ)**: 307K training questions from Google search with Wikipedia passages.

**BEIR Benchmark** (Benchmarking Information Retrieval): 18 heterogeneous datasets covering biomedical, legal, financial, and general domains. Used for zero-shot evaluation.

### 7.2 Cross-Encoder Training

**Binary cross-entropy:**

$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\sigma(s_i)) + (1-y_i)\log(1-\sigma(s_i)) \right]$$

**In-batch negatives:** For a batch of $(q_i, d_i^+)$ positive pairs, use other documents in the batch as negatives:

$$\mathcal{L}_{IBN} = -\sum_{i} \log \frac{\exp(s(q_i, d_i^+))}{\sum_{j} \exp(s(q_i, d_j^+))}$$

**Margin MSE loss with soft teacher labels:**

$$\mathcal{L}_{MarginMSE} = \left( (s_+ - s_-) - (\hat{s}_+ - \hat{s}_-) \right)^2$$

where $s_+, s_-$ are student scores and $\hat{s}_+, \hat{s}_-$ are teacher scores for positive and negative documents.

### 7.3 Hard Negative Mining

Random negatives are too easy — the model quickly learns to distinguish them. Hard negatives are documents that are **superficially similar but not relevant**.

**BM25 negatives:** Retrieve top-K documents using BM25 that are NOT relevant. These contain query keywords but discuss different aspects.

**Bi-encoder negatives:** Retrieve top-K by dense retrieval. These are semantically similar but not relevant — harder than BM25 negatives.

**Cross-encoder scored negatives (ANCE):** Dynamically mine hard negatives during training using the model's current state.

```python
# Hard negative mining pipeline
def mine_hard_negatives(queries, corpus, retriever, k=50):
    negatives = {}
    for query in queries:
        candidates = retriever.retrieve(query, top_k=k)
        # Filter out known positives
        hard_negatives = [c for c in candidates if c not in positives[query]]
        negatives[query] = hard_negatives[:10]  # Keep top-10 hardest
    return negatives
```

### 7.4 Knowledge Distillation

The key insight: train a fast bi-encoder to **mimic the scores of a slow cross-encoder**. The cross-encoder is the teacher; the bi-encoder is the student.

$$\mathcal{L}_{KD} = \text{KL}\left(\text{softmax}\left(\frac{s_{CE}}{T}\right) \,\Big\|\, \text{softmax}\left(\frac{s_{BE}}{T}\right)\right)$$

where $T$ is the temperature parameter. Higher $T$ produces softer probability distributions that carry more information about relative similarities.

Expanded KL divergence:

$$\mathcal{L}_{KD} = \sum_{d \in C_q} p_{CE}(d|q) \log \frac{p_{CE}(d|q)}{p_{BE}(d|q)}$$

where:

$$p_{CE}(d|q) = \frac{\exp(s_{CE}(q,d)/T)}{\sum_{d'} \exp(s_{CE}(q,d')/T)}$$

The student learns to assign high similarity to documents that the teacher scores highly, even for documents without binary labels.

**Augmented SBERT** uses this approach:
1. Generate soft labels for all query-document pairs using a cross-encoder
2. Train a bi-encoder using these soft labels with margin MSE

This can bring bi-encoder quality close to cross-encoder quality on the training distribution.

---

## 8. Evaluation Metrics — Full Derivations

### 8.1 MRR (Mean Reciprocal Rank)

MRR measures how highly the **first** relevant document is ranked.

$$\text{MRR} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \frac{1}{\text{rank}_q}$$

where $\text{rank}_q$ is the rank position of the first relevant document for query $q$.

**Worked Example:**

| Query | Ranking | First relevant at rank |
|-------|---------|----------------------|
| Q1    | [R, N, N, R, N] | 1 |
| Q2    | [N, R, N, N, R] | 2 |
| Q3    | [N, N, N, R, N] | 4 |

(R = relevant, N = not relevant)

$$\text{MRR} = \frac{1}{3}\left(\frac{1}{1} + \frac{1}{2} + \frac{1}{4}\right) = \frac{1}{3}(1 + 0.5 + 0.25) = \frac{1.75}{3} \approx 0.583$$

**When to use:** FAQ lookup, question answering — you care about getting *one* good answer at the top, not comprehensive coverage.

### 8.2 MAP (Mean Average Precision)

MAP measures the quality of ranking across **all** relevant documents.

**Precision at rank $k$:**

$$P(k) = \frac{\text{# relevant in top-}k}{k}$$

**Average Precision for query $q$:**

$$\text{AP}(q) = \frac{\sum_{k=1}^{n} P(k) \cdot \text{rel}(k)}{\text{\# relevant documents for } q}$$

where $\text{rel}(k) \in \{0, 1\}$ indicates if position $k$ is relevant.

**MAP:**

$$\text{MAP} = \frac{1}{|Q|} \sum_{q=1}^{|Q|} \text{AP}(q)$$

**Worked Example:**

Query Q1 has 3 relevant documents. System returns:

| Rank | Doc | Relevant? | P(k) | P(k) * rel(k) |
|------|-----|-----------|------|----------------|
| 1    | D1  | Yes       | 1/1 = 1.0 | 1.0 |
| 2    | D2  | No        | 1/2 = 0.5 | 0.0 |
| 3    | D3  | Yes       | 2/3 = 0.667 | 0.667 |
| 4    | D4  | No        | 2/4 = 0.5 | 0.0 |
| 5    | D5  | Yes       | 3/5 = 0.6 | 0.6 |

$$\text{AP}(Q1) = \frac{1.0 + 0.0 + 0.667 + 0.0 + 0.6}{3} = \frac{2.267}{3} \approx 0.756$$

**When to use:** Document retrieval where all relevant documents matter (e.g., legal discovery, comprehensive search).

### 8.3 NDCG@K (Normalized Discounted Cumulative Gain)

NDCG handles **graded relevance** (e.g., 0 = not relevant, 1 = partially relevant, 2 = highly relevant, 3 = perfect).

**DCG@K:**

$$\text{DCG@K} = \sum_{i=1}^{K} \frac{2^{r_i} - 1}{\log_2(i+1)}$$

where $r_i \in \{0, 1, 2, 3\}$ is the relevance grade of the document at rank $i$.

The denominator $\log_2(i+1)$ is the **position discount** — documents ranked lower contribute less, logarithmically.

**IDCG@K (Ideal DCG):** The DCG achieved by the perfect ranking (all documents sorted by decreasing relevance):

$$\text{IDCG@K} = \sum_{i=1}^{K} \frac{2^{r_i^*} - 1}{\log_2(i+1)}$$

where $r_i^*$ is the relevance of the $i$-th document in the ideal ranking.

**NDCG@K:**

$$\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$$

**Worked Example:**

5 candidate documents with relevance grades [2, 3, 1, 0, 2]. System returns them in order [D1(2), D3(1), D2(3), D5(2), D4(0)].

System ranking relevance grades: $[2, 1, 3, 2, 0]$

$$\text{DCG@5} = \frac{2^2 - 1}{\log_2 2} + \frac{2^1 - 1}{\log_2 3} + \frac{2^3 - 1}{\log_2 4} + \frac{2^2 - 1}{\log_2 5} + \frac{2^0 - 1}{\log_2 6}$$

$$= \frac{3}{1} + \frac{1}{1.585} + \frac{7}{2} + \frac{3}{2.322} + \frac{0}{2.585}$$

$$= 3.0 + 0.631 + 3.5 + 1.292 + 0 = 8.423$$

Ideal ranking grades (sorted descending): $[3, 2, 2, 1, 0]$

$$\text{IDCG@5} = \frac{7}{1} + \frac{3}{1.585} + \frac{3}{2} + \frac{1}{2.322} + \frac{0}{2.585}$$

$$= 7.0 + 1.893 + 1.5 + 0.431 + 0 = 10.824$$

$$\text{NDCG@5} = \frac{8.423}{10.824} \approx 0.778$$

**When to use:** Graded relevance judgments, search engine evaluation, recommendation systems.

### 8.4 Recall@K

$$\text{Recall@K} = \frac{|\text{relevant documents in top-}K|}{|\text{total relevant documents}|}$$

**When to use:** RAG retrieval stage evaluation — you need to ensure relevant context is in the retrieved set before it can be used for generation. A low Recall@K means the generator will miss relevant information regardless of reranking quality.

### 8.5 Metric Selection Guide

| Use case | Metric | Reason |
|----------|--------|--------|
| Crypto FAQ chatbot (one answer) | MRR | Cares about first hit |
| Legal document search | MAP | All relevant docs matter |
| Search engine with click feedback | NDCG | Graded relevance from clicks |
| RAG context retrieval | Recall@K | Coverage matters for generation |
| Recommendation with ratings | NDCG@10 | Graded + position matters |

---

## 9. Reciprocal Rank Fusion (RRF)

### 9.1 Formula

Given multiple ranked lists $R$ (e.g., one from BM25, one from dense retrieval), the RRF score for document $d$ is:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

where $k = 60$ is a smoothing constant (empirically determined) and $r(d)$ is the rank of document $d$ in ranked list $r$ (1-indexed).

### 9.2 Worked Example

Three retrieval systems return the following rankings for documents A, B, C, D, E:

| System | Rank 1 | Rank 2 | Rank 3 | Rank 4 | Rank 5 |
|--------|--------|--------|--------|--------|--------|
| BM25   | A      | C      | B      | E      | D      |
| Dense  | C      | A      | D      | B      | E      |
| ColBERT| A      | B      | C      | D      | E      |

With $k = 60$:

$$\text{RRF}(A) = \frac{1}{60+1} + \frac{1}{60+2} + \frac{1}{60+1} = \frac{1}{61} + \frac{1}{62} + \frac{1}{61} \approx 0.0164 + 0.0161 + 0.0164 = 0.0489$$

$$\text{RRF}(C) = \frac{1}{60+2} + \frac{1}{60+1} + \frac{1}{60+3} = \frac{1}{62} + \frac{1}{61} + \frac{1}{63} \approx 0.0161 + 0.0164 + 0.0159 = 0.0484$$

$$\text{RRF}(B) = \frac{1}{60+3} + \frac{1}{60+4} + \frac{1}{60+2} = \frac{1}{63} + \frac{1}{64} + \frac{1}{62} \approx 0.0159 + 0.0156 + 0.0161 = 0.0476$$

Final RRF ranking: A (0.0489) > C (0.0484) > B (0.0476) > D > E

### 9.3 Why RRF Often Beats Score Combination

**Score normalization problem:** BM25 scores are in $[0, 20]$; dense scores are cosine similarities in $[-1, 1]$. Directly combining these requires careful normalization. Min-max normalization is sensitive to outliers.

**Rank saturation:** RRF's $1/(k+r)$ function saturates quickly. The difference between rank 1 and rank 2 is $\frac{1}{61} - \frac{1}{62} \approx 0.0003$, while between rank 1 and rank 100 is $\frac{1}{61} - \frac{1}{160} \approx 0.0101$. This means **rank 1 is much more valuable than rank 100**, but all top-ranked documents get similar scores — a smooth, robust aggregation.

**No hyperparameter tuning:** Linear combination requires tuning weights $\alpha, \beta$ per dataset. RRF with $k=60$ works well out of the box.

### 9.4 Problems & Mitigations

| Problem | Mitigation |
|---------|-----------|
| Documents not in all lists get penalized | Assign penalty rank (e.g., K+1) for missing documents |
| k=60 may not be optimal for all use cases | Grid search k on a dev set; typical range [30, 100] |
| RRF ignores score magnitude | For very high-confidence systems, consider weighted RRF |

---

## 10. Hybrid Search Architecture

### 10.1 Full System Architecture

```
                                    ┌────────────────────────────┐
                                    │     Query Processing       │
                                    │  - Spell correction        │
                                    │  - Query expansion         │
                                    │  - Query rewriting (LLM)   │
                                    └─────────────┬──────────────┘
                                                  │
                     ┌────────────────────────────▼──────────────────────────────┐
                     │                     Retrieval Layer                        │
                     │                                                             │
          ┌──────────▼──────────┐                          ┌──────────▼──────────┐
          │      BM25 / TF-IDF  │                          │    Dense Retrieval  │
          │  (Elasticsearch /   │                          │  (FAISS / Milvus)   │
          │   Opensearch)       │                          │  Bi-encoder encoded │
          │                     │                          │  document vectors   │
          │  Inverted Index      │                          │  ANN search         │
          └──────────┬──────────┘                          └──────────┬──────────┘
                     │ Sparse results                                  │ Dense results
                     │ (ranked by BM25 score)                          │ (ranked by cosine sim)
                     └───────────────────┐     ┌───────────────────────┘
                                         │     │
                                    ┌────▼─────▼────┐
                                    │  RRF Fusion   │
                                    │  (k=60)       │
                                    └───────┬───────┘
                                            │ Top-100 fused candidates
                                    ┌───────▼───────┐
                                    │   Reranker    │
                                    │ (Cross-encoder│
                                    │  or ColBERT)  │
                                    └───────┬───────┘
                                            │ Top-10 reranked results
                                    ┌───────▼───────────────────┐
                                    │   Post-Processing         │
                                    │ - Deduplication           │
                                    │ - MMR diversity           │
                                    │ - Metadata filtering      │
                                    │ - Highlight extraction    │
                                    └───────────────────────────┘
```

### 10.2 When BM25 Dominates vs Dense

**BM25 dominates when:**
- Query contains rare, specific terms (product codes, tickers like "BTCUSDT")
- Query is a precise technical phrase
- Short, keyword-style queries

**Dense retrieval dominates when:**
- Query is natural language / conversational
- Semantic similarity matters (synonyms, paraphrases)
- Cross-language retrieval
- Long queries with complex intent

**Hybrid is almost always better** than either alone because they capture complementary signals.

### 10.3 Score Normalization

Before linear combination (alternative to RRF):

**Min-max normalization:**

$$\hat{s} = \frac{s - s_{\min}}{s_{\max} - s_{\min}}$$

Sensitive to outliers. Use percentile normalization instead:

$$\hat{s} = \frac{s - P_5}{P_{95} - P_5}$$

where $P_5, P_{95}$ are 5th and 95th percentiles of scores in the candidate set.

### 10.4 ANN Indexes

**FAISS (Facebook AI Similarity Search):**
- `IndexFlatIP`: exact inner product search, $O(N \cdot d)$ — use for small corpora
- `IndexIVFFlat`: inverted file index, partitions space into $n_{list}$ Voronoi cells, searches only `n_probe` cells — $O(N/n_{list} \cdot d)$
- `IndexHNSW`: Hierarchical Navigable Small World graph, best recall/speed tradeoff for medium corpora

**HNSW complexity:**
- Build: $O(N \log N)$
- Query: $O(\log N)$ for approximate search

---

## 11. Problems & Mitigations (Comprehensive)

### 11.1 Bi-encoder Quality Ceiling

**Problem:** The independence assumption forces all document semantics into a fixed-size vector, losing fine-grained signals.

**Mitigations:**
- **ColBERT late interaction**: retain per-token embeddings, enabling term-level matching
- **Hard negative training**: expose model to difficult negatives during training to sharpen decision boundaries
- **Knowledge distillation**: use cross-encoder soft labels to train bi-encoder beyond binary supervision
- **Asymmetric encoders**: use different (larger) encoder for documents than queries
- **Matryoshka representation learning (MRL)**: train embeddings that are useful at multiple truncated dimensions

### 11.2 Cross-Encoder Latency

**Problem:** $O(K)$ forward passes at inference time. With $K=100$, $L=512$, this is ~200ms on a single GPU.

**Mitigations:**
- **Model distillation**: distill 12-layer BERT to 6-layer or 4-layer student. 2x speedup with ~5% quality drop.
- **Quantization (INT8)**: TensorRT or ONNX Runtime quantization. 2-4x speedup.
- **Early exit**: stop inference when confidence is high enough (adaptive computation).
- **Limit K**: reduce candidates to 50 instead of 100. Quality loss depends on retrieval quality.
- **Caching**: cache scores for (query, document) pairs seen previously. Effective for FAQ-style queries.
- **Batching**: process multiple query-document pairs in parallel on GPU.

### 11.3 Reranker as System Bottleneck

**Problem:** Reranker is the slowest component in the pipeline, blocking the entire response.

**Mitigations:**
- **Cascading K reduction**: 10M → 1000 (BM25) → 100 (bi-encoder) → 10 (cross-encoder). Never pass 1000 documents to cross-encoder.
- **Async reranking**: return top-5 bi-encoder results immediately, rerank asynchronously, update results when ready.
- **Query result caching**: for popular queries, cache the final reranked results.
- **Profile-based routing**: simple queries skip reranker entirely; only complex queries go through full pipeline.

### 11.4 Training Data Distribution Mismatch

**Problem:** Models trained on MS MARCO (web search) perform poorly on domain-specific corpora (crypto, legal, medical).

**Mitigations:**
- **Domain adaptation fine-tuning**: fine-tune on a small domain-specific dataset (1K labeled pairs is often sufficient).
- **Synthetic data generation**: use LLM to generate query-document pairs from your corpus. GPT-4 + your documents → synthetic training data.
- **BEIR-style zero-shot evaluation**: evaluate on BEIR before deciding whether to fine-tune.
- **Continued pre-training**: train base model on domain text (e.g., Binance help articles) before fine-tuning.

### 11.5 Long Document Reranking

**Problem:** Documents exceed the 512-token context window of BERT-based rerankers.

**Mitigations:**
- **Sliding window chunking**: split document into overlapping chunks, score each chunk, aggregate with max or mean pooling.
  $$s(q, d) = \max_{c \in \text{chunks}(d)} s(q, c)$$
- **First-$k$ tokens**: score only the first 512 tokens (works if documents have summary-style headers).
- **Hierarchical reranking**: first select relevant chunks, then rerank at chunk level.
- **Longformer/BigBird**: models with linear attention supporting up to 16K tokens.

### 11.6 Hallucinated Relevance by LLM Rerankers

**Problem:** LLM rerankers may judge documents as relevant based on world knowledge rather than document content.

**Mitigations:**
- **Grounded prompting**: "Rank documents ONLY based on whether the document text answers the query. Do not use outside knowledge."
- **Citation verification**: require LLM to quote the relevant passage from each document as justification.
- **Constrained decoding**: force LLM to output only the document IDs in a specified format, preventing it from reasoning beyond the documents.

### 11.7 False Negatives in Training Data

**Problem:** MS MARCO has very sparse labels — most relevant documents are not labeled as relevant, creating false negatives. Training treats them as negatives, leading to poor recall.

**Mitigations:**
- **Soft labels**: instead of 0/1 labels, use cross-encoder score as a continuous label — false negatives get a moderate score, not 0.
- **Label denoising**: filter hard negatives by checking if cross-encoder assigns high score (likely false negatives).
- **Closed-world assumption relaxation**: use in-batch negatives sparingly; prefer explicit hard negatives from BM25.
- **Human annotation of a small held-out set**: validate evaluation metrics are not corrupted by false negatives.

---

## 12. Industry Practices

### 12.1 Binance-Relevant Use Cases

**Crypto documentation search:** Users query "how to set leverage on Binance futures." BM25 catches keyword matches on "leverage", "futures". Dense retrieval catches semantic intent "position sizing" and "risk management."

**Whitepaper similarity:** Given a new DeFi protocol whitepaper, find similar whitepapers. Pure dense retrieval, no BM25 needed (technical vocabulary is well-handled by embeddings trained on crypto text).

**Regulatory text retrieval:** "Which regulations apply to crypto derivatives in Singapore?" Requires both keyword precision (jurisdiction) and semantic understanding (derivatives = futures = leveraged products). Hybrid search + cross-encoder reranker.

**Support ticket routing:** Classify incoming support tickets by routing to the relevant FAQ answer. MRR metric, bi-encoder retrieval is sufficient for most cases.

### 12.2 Latency Budget Allocation

For a 100ms total latency budget:

```
Query preprocessing:    5ms   (tokenization, spell check)
BM25 retrieval:        10ms   (Elasticsearch)
Dense ANN retrieval:   15ms   (FAISS HNSW)
RRF fusion:             2ms   (in-memory computation)
Cross-encoder rerank:  60ms   (GPU inference, K=50)
Post-processing:        5ms   (deduplication, formatting)
Network overhead:       3ms
                      ────
Total:                100ms
```

Key constraint: **limit K to 50 for reranking** to stay within 60ms GPU budget.

### 12.3 API Services vs Self-Hosted

**Cohere Rerank API:**
- Pros: no infrastructure, state-of-the-art model, easy integration
- Cons: ~$1/1000 queries, data leaves your infrastructure, latency adds network round-trip

**Jina Reranker:**
- Open-source model (jina-reranker-v2-base-multilingual), self-hostable
- 512-token limit; multilingual

**VoyageAI Rerank:**
- Strong on retrieval benchmarks, good for technical/legal text
- API-only

**Self-hosted cross-encoder (ms-marco-MiniLM-L-6-v2):**
- Free, fast (6-layer MiniLM), excellent latency
- Requires GPU infrastructure
- Best for regulated industries (crypto, finance) where data cannot leave the org

**Recommendation for Binance:** Self-hosted cross-encoder (data privacy) + fallback to Cohere Rerank for overflow.

### 12.4 Monitoring

**Offline metrics (computed on labeled test set):**
- NDCG@10: primary quality metric
- MRR@10: first-hit quality
- Recall@100: retrieval stage coverage

**Online metrics (production):**
- **Click-through rate (CTR)**: fraction of queries where user clicks a result
- **Reranker hit rate**: fraction of final results that the reranker moved up from the top-10 bi-encoder results
- **Latency p95/p99**: tail latency for reranker forward passes
- **NDCG drift**: monitor weekly NDCG on a sample of queries to detect distribution shift

**Alerting:** Alert if NDCG drops >5% week-over-week, or p99 latency exceeds 200ms.

### 12.5 A/B Testing

1. **Offline evaluation first**: compute NDCG@10 on BEIR subset + internal test set. Gate: new model must be ≥99% of baseline.
2. **Shadow mode**: run new reranker in parallel, log results but don't serve. Compare with baseline offline.
3. **Small traffic split (1-5%)**: expose small fraction of users to new reranker. Monitor CTR, session length.
4. **Ramp up**: if CTR improves and no latency regression, ramp to 50% then 100%.

---

## 13. Full System Design: Production Retrieval + Reranking

### 13.1 Complete Architecture

```
                             PRODUCTION RAG RETRIEVAL SYSTEM
                             ================================

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INDEXING PIPELINE (Offline)                         │
│                                                                                   │
│  Raw Documents                                                                    │
│       │                                                                           │
│  ┌────▼────────────────────────────────────┐                                     │
│  │ Document Processing                      │                                     │
│  │ - HTML/PDF parsing                       │                                     │
│  │ - Chunking (512-token with 50-token overlap)                                   │
│  │ - Metadata extraction (date, category)   │                                     │
│  └────┬──────────────────────┬─────────────┘                                     │
│       │                      │                                                    │
│  ┌────▼────────┐      ┌──────▼───────────┐                                      │
│  │ BM25 Index  │      │ Bi-encoder        │                                      │
│  │ (Elastic-   │      │ (text-embedding-  │                                      │
│  │  search)    │      │  ada-002 / BGE)   │                                      │
│  │ Inverted    │      │ → FAISS HNSW idx  │                                      │
│  │ index over  │      │   768-dim vectors │                                      │
│  │ tokens      │      │                   │                                      │
│  └─────────────┘      └───────────────────┘                                      │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│                              QUERY PIPELINE (Online)                              │
│                                                                                   │
│  User Query                                                                       │
│       │                                                                           │
│  ┌────▼─────────────────────────┐                                                 │
│  │ Query Processing              │                                                 │
│  │ - Spelling correction         │                                                 │
│  │ - Language detection          │                                                 │
│  │ - Query classification        │                                                 │
│  │   (navigational/informational)│                                                 │
│  │ - Query expansion (synonyms,  │                                                 │
│  │   abbreviation expansion)     │                                                 │
│  └────┬─────────────────────────┘                                                 │
│       │ processed query                                                            │
│       ├──────────────────────────┐                                                 │
│  ┌────▼─────────┐          ┌─────▼──────────┐                                    │
│  │ BM25 Retrieval│          │ ANN Retrieval  │                                    │
│  │ (ES query)   │          │ (FAISS HNSW)   │                                    │
│  │ Top-200      │          │ Top-200        │                                    │
│  └────┬─────────┘          └──────┬─────────┘                                    │
│       │                           │                                                │
│       └──────────┐ ┌──────────────┘                                               │
│              ┌───▼─▼──────┐                                                       │
│              │ RRF Fusion │  k=60, deduplicate, keep top-100                     │
│              └─────┬──────┘                                                       │
│                    │ 100 candidates                                                │
│              ┌─────▼──────────────────────────────┐                              │
│              │ Cross-Encoder Reranker              │                              │
│              │ (ms-marco-MiniLM-L-6-v2)            │                              │
│              │ Batch forward pass: 100 × 512 tok   │                              │
│              │ Output: relevance scores [0,1]      │                              │
│              └─────┬──────────────────────────────┘                              │
│                    │ 100 scored candidates                                         │
│              ┌─────▼──────────────────────────────┐                              │
│              │ Post-Processing                     │                              │
│              │ - Sort by reranker score            │                              │
│              │ - Deduplicate near-duplicate chunks │                              │
│              │   (cosine sim > 0.95 → keep top)    │                              │
│              │ - MMR diversity (λ=0.7)             │                              │
│              │ - Apply metadata filters            │                              │
│              │   (date range, category, lang)      │                              │
│              └─────┬──────────────────────────────┘                              │
│                    │ Top-5 to Top-10 results                                       │
│              ┌─────▼──────────────────────────────┐                              │
│              │ Response Generation (RAG)           │                              │
│              │ - Format context from top results   │                              │
│              │ - LLM generates answer with citations│                             │
│              └─────────────────────────────────────┘                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 13.2 Maximum Marginal Relevance (MMR) for Diversity

MMR selects results that are relevant to the query but dissimilar to already-selected results:

$$\text{MMR}(d_i) = \lambda \cdot s(q, d_i) - (1-\lambda) \cdot \max_{d_j \in S} \text{sim}(d_i, d_j)$$

where $S$ is the set of already-selected documents and $\lambda \in [0,1]$ controls relevance-diversity tradeoff.

### 13.3 Index Management

**FAISS index lifecycle:**
```
Nightly batch:
  1. Encode new/updated documents with bi-encoder
  2. Add vectors to FAISS index
  3. Rebuild HNSW graph (for consistency)
  4. Atomic swap: point live traffic to new index

Real-time updates:
  1. Queue new documents
  2. Add to FAISS IndexIDMap (allows deletion)
  3. Rebuild index during off-peak hours
```

**Elasticsearch management:**
- Use index aliases for zero-downtime updates
- Configure `number_of_shards` based on document count
- Enable `_source: false` for BM25-only index (save storage)

---

## 14. Interview Q&A

### Q1 (Basic): What is the difference between a bi-encoder and a cross-encoder?

**A:** A bi-encoder encodes the query and document **independently** using separate (or shared) transformer encoders, producing one embedding per text. The relevance score is the dot product $E_q^T E_d$. The key advantage is that document embeddings can be **pre-computed and indexed**, enabling fast retrieval over millions of documents using approximate nearest neighbor search.

A cross-encoder takes the query and document **concatenated** as a single input: `[CLS] query [SEP] document [SEP]`. The full transformer processes both texts jointly, allowing every query token to attend to every document token through cross-attention. The CLS token's representation is projected to a scalar score. Cross-encoders are significantly more accurate but cannot pre-compute document representations since the encoding depends on the query. This makes them suitable only as **rerankers** over small candidate sets (< 1000 documents).

---

### Q2 (Basic): What is Reciprocal Rank Fusion and why is it preferred over score combination?

**A:** RRF aggregates multiple ranked lists by summing the reciprocal of each document's rank across all lists:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

It is preferred over linear score combination for three reasons: (1) **no normalization needed** — raw BM25 and cosine similarity scores have incompatible scales; (2) **robust to outliers** — a single very high BM25 score for one document doesn't dominate the fusion; (3) **no tuning required** — $k=60$ works well empirically without dataset-specific weight optimization.

---

### Q3 (Intermediate): Derive NDCG@3 for the following ranking.

**Setup:** Query has 4 documents with relevance [3, 1, 2, 0]. System returns order: D1(3), D3(2), D2(1), D4(0).

**A:**

System DCG@3:

$$\text{DCG@3} = \frac{2^3-1}{\log_2(2)} + \frac{2^2-1}{\log_2(3)} + \frac{2^1-1}{\log_2(4)}$$
$$= \frac{7}{1} + \frac{3}{1.585} + \frac{1}{2} = 7.0 + 1.893 + 0.5 = 9.393$$

Ideal order (sorted desc by relevance): D1(3), D3(2), D2(1), D4(0)

$$\text{IDCG@3} = \frac{7}{1} + \frac{3}{1.585} + \frac{1}{2} = 7.0 + 1.893 + 0.5 = 9.393$$

$$\text{NDCG@3} = \frac{9.393}{9.393} = 1.0$$

The system achieved perfect NDCG@3 — it ranked the top-3 documents in the ideal order.

---

### Q4 (Intermediate): Why does MaxSim in ColBERT work better than single-vector similarity for term-level matching?

**A:** Consider query "stop loss order" and document "automatic sell order triggered by price drop." A bi-encoder must compress both into single vectors. The embedding of "stop loss order" and "automatic sell order triggered by price drop" might be reasonably close, but the single-vector comparison discards exactly which terms matched.

With MaxSim, each query token independently finds its best-matching document token:
- "stop" → aligns with "price drop" context or "triggered" (soft semantic match)
- "loss" → aligns with "sell" (in financial context, selling = loss prevention)
- "order" → aligns with "order" (exact match, high dot product)

The sum aggregates per-term evidence. If a document is missing any semantically relevant term, the corresponding MaxSim value for that query token will be low, penalizing the overall score. This is a **soft version of BM25-style term matching** in the dense embedding space, combining the flexibility of dense representations with the term-level precision of sparse retrieval.

---

### Q5 (Intermediate): How does knowledge distillation from cross-encoder to bi-encoder work?

**A:** The cross-encoder teacher generates **soft relevance scores** for query-document pairs. These are softmax probabilities over a candidate set, carrying rich information about relative similarity (not just binary relevant/not).

For a query $q$ with candidate set $\{d_1, \ldots, d_K\}$:

1. Compute teacher scores: $\hat{s}_i = s_{CE}(q, d_i)$ for all $i$
2. Convert to distribution: $p_i = \text{softmax}(\hat{s}_i / T)$
3. Train bi-encoder to minimize KL divergence:
$$\mathcal{L}_{KD} = \text{KL}(p_{CE} \| p_{BE}) = \sum_i p_{CE,i} \log \frac{p_{CE,i}}{p_{BE,i}}$$

Temperature $T > 1$ softens the distributions, preventing the student from just learning to copy the argmax. This is important because near-relevant documents (score 0.7) carry more signal than clear non-relevants (score 0.1), and soft labels preserve this distinction.

The result is a bi-encoder that better aligns its similarity scores with the cross-encoder's judgment, even for documents that lack binary labels. This has been shown to improve bi-encoder performance by 5-10 NDCG points on MS MARCO.

---

### Q6 (Advanced): How would you design a reranker for Binance's support system with a 10ms latency budget?

**A:** 10ms is an extremely tight budget — cross-encoder inference typically takes 50-200ms. The design must be aggressive:

**Option 1: Skip full reranking, use ColBERT-style late interaction**
- ColBERT with PLAID can complete retrieval + scoring in ~15ms, which is close
- Pre-compute document token embeddings (large storage cost ~10GB for 100K docs)
- At query time: encode query tokens (~1ms), ANN search for max token alignment (~5ms), MaxSim aggregation (~2ms)
- Total: ~8ms achievable

**Option 2: Distilled cross-encoder with extreme quantization**
- Start with ms-marco-MiniLM-L-6-v2 (6-layer, 22M params)
- Further distill to 4-layer, 128-hidden (MiniLM-v2)
- INT8 quantize with TensorRT
- Limit candidates to K=20 (not 100)
- With GPU batching, 20 forward passes through 4-layer INT8 model: ~8ms

**Option 3: Bi-encoder only with better training**
- Accept quality tradeoff; train with hard negatives and knowledge distillation
- FAISS HNSW search: ~2-3ms
- No reranker; return top-10 ANN results directly

**Practical recommendation for Binance support:** Use Option 3 for the primary path with Option 2 as an optional upgrade. Cache popular queries (top-1000 queries often cover 80% of traffic — serve from Redis in < 1ms). For the remaining 20% of queries, run Option 3. This gives average latency <<10ms.

**Monitoring:** Track p99 latency, not average. Ensure FAISS index fits in GPU memory (< 16GB) to avoid I/O bottlenecks.

---

### Q7 (Advanced): What are the tradeoffs between listwise, pairwise, and pointwise reranking?

**A:**

**Pointwise:**
- Each document scored independently: $s_i = f(q, d_i)$
- Pros: embarrassingly parallel, $O(N)$ LLM calls
- Cons: no cross-document comparison; scores are not calibrated relative to each other; LLM may give all documents 8/10

**Pairwise:**
- Compare all pairs: $p_{ij} = P(d_i \succ d_j | q)$
- Aggregate via merge sort or Bradley-Terry model
- Pros: directly optimizes for relative order, which is what ranking requires; reduces position bias
- Cons: $O(N^2)$ comparisons — infeasible for $N > 20$ with LLMs; position bias in which document is listed first

**Listwise:**
- All documents input at once, output is permutation
- Pros: global context; model can reason about relative relevance across the full set
- Cons: context window limits (can't fit 100 documents); inconsistency across runs; position bias (LLM tends to favor documents at start/end of list); harder to parallelize

**For production:** Cross-encoder-based pointwise reranking is standard (fast, reliable, no LLM cost). LLM listwise is used when quality justifies GPT-4 costs (e.g., enterprise search with high per-query value). Pairwise is used in offline evaluation/dataset creation (where cost is acceptable) but rarely in production.

---

### Q8 (Intermediate): How do you evaluate a reranker when your test set has sparse relevance labels?

**A:** Sparse labels (MS MARCO-style) are a fundamental problem. Most documents that *are* relevant are not labeled as such, meaning standard NDCG/MAP underestimates quality.

**Mitigations:**

1. **Inference-time ensembling**: treat unlabeled documents retrieved by multiple systems as "potentially relevant." If a document is in the top-5 of three different systems, assume it is relevant even if not labeled.

2. **Pooled evaluation (TREC-style)**: pool top-K results from all competing systems, annotate the pool. This ensures any relevant document found by any system is labeled.

3. **Recall-based metrics**: use Recall@100 to evaluate the retrieval stage separately from reranking. This sidesteps the label sparsity issue for the retrieval component.

4. **Using soft cross-encoder labels**: instead of binary labels, use cross-encoder scores as pseudo-labels. Compute NDCG with continuous relevance grades. This is less interpretable but correlates well with human judgments.

5. **Held-out human annotation**: for a random sample of 200-500 queries, conduct blind relevance assessment. This provides unbiased ground truth even if sparse.

---

### Q9 (Basic): When would you use MRR vs NDCG?

**A:** Use **MRR** when you care about the first relevant result only — e.g., a crypto FAQ bot where users want a single correct answer. MRR rewards systems that consistently put one relevant result at rank 1 regardless of the rest.

Use **NDCG** when relevance is graded and position matters throughout the list. For example, a Binance document search where users may want multiple related documents. NDCG accounts for the fact that a highly relevant document (grade 3) at rank 2 is better than a moderately relevant document (grade 2) at rank 1.

NDCG is generally preferred for evaluating modern rerankers because it captures graded relevance and the full ranking, not just the first hit.

---

### Q10 (Advanced): How would you adapt a reranker trained on MS MARCO to Binance's crypto documentation?

**A:**

**Step 1: Evaluate zero-shot performance.** Run the off-the-shelf model on a small held-out set of 50-100 labeled Binance queries. If NDCG@10 > 0.7, zero-shot may be sufficient.

**Step 2: Collect domain-specific training data.**
- Option A: Label 500-1000 query-document pairs manually (expensive, high quality)
- Option B: Generate synthetic queries from Binance documentation using an LLM: "Generate 5 questions that this paragraph answers." Then use BM25/dense retrieval to generate candidates, and a cross-encoder to provide soft labels.
- Option C: Mine clickthrough data from Binance search logs (if available) — queries where users clicked a document signal relevance.

**Step 3: Fine-tune.**
- Initialize from MS MARCO checkpoint
- Fine-tune with domain data using a low learning rate ($10^{-5}$) to avoid forgetting general knowledge
- Use continued training (not from scratch) — this requires 10x less data
- Train for 1-3 epochs on domain data; monitor for overfitting

**Step 4: Evaluate and iterate.**
- Compare fine-tuned vs zero-shot NDCG@10 on held-out Binance test set
- Use BEIR as a control to ensure fine-tuning didn't hurt generalization

**Expected improvement:** 5-15 NDCG points for a specialized crypto domain with domain-specific vocabulary (DeFi, perpetuals, margin trading, blockchain terminology).

---

### Q11 (Intermediate): Explain the position bias problem in LLM rerankers and how to mitigate it.

**A:** LLMs exhibit strong **position bias** — they tend to assign higher relevance to documents that appear at the beginning or end of the input list (primacy and recency effects), regardless of content. This is because LLMs are trained on text where important information often appears early or late.

**Evidence:** In RankGPT experiments, simply reversing the order of documents in the prompt changes the output ranking significantly (~15-20% change in top-1 result).

**Mitigations:**

1. **Permutation ensembling**: run the same query with $n$ random permutations of the document list. Average the resulting rank scores. This is the most effective mitigation but requires $n$ LLM calls.

2. **Calibration fine-tuning**: fine-tune the LLM reranker on examples where position-irrelevant ranking is rewarded. Include training examples with identical content in different positions.

3. **Pairwise comparison**: instead of listing all documents, compare pairs. Position bias is less severe with just 2 documents (though still present for first vs second position).

4. **Constrained prompt format**: explicitly instruct "The order of passages in this list does NOT reflect their relevance. Judge only based on content." This partially but not fully resolves the issue.

---

### Q12 (Advanced): How would you implement a hybrid search + reranking system for a corpus that is updated in real time?

**A:**

**Real-time index updates** require careful architecture:

**Sparse (BM25) index:**
- Elasticsearch supports real-time indexing via the `_index` API
- Documents are searchable within ~1 second of indexing
- No rebuilding required

**Dense (FAISS) index:**
- `IndexFlatL2` supports `add()` at runtime but no deletion
- Use `IndexIDMap` for deletion support
- HNSW does NOT support deletion natively; use soft deletion (mark as deleted, filter in post-processing)
- For high update rate: use **streaming indexing** — maintain a small buffer index (in-memory, exact search) and a large main index (HNSW, approximate). Merge periodically.

**Pipeline:**
```
New document arrives
         │
    ┌────▼────────────────────────────────────┐
    │ 1. Parse and chunk document             │
    │ 2. Compute bi-encoder embedding         │
    │ 3. Add to Elasticsearch (BM25 index)   │
    │ 4. Add to FAISS buffer index            │
    └────────────────────────────────────────┘
         │
    Nightly batch:
    ┌────▼────────────────────────────────────┐
    │ 1. Merge buffer index into main HNSW    │
    │ 2. Rebuild HNSW for optimal graph       │
    │ 3. Atomic swap to new index             │
    └────────────────────────────────────────┘
```

**Query time:** search both buffer and main index, take union, pass to reranker. This ensures newly added documents are searchable immediately.

---

### Q13 (Basic): What is hard negative mining and why is it important?

**A:** Hard negatives are documents that are **retrieved by the model but are NOT relevant**. They are "hard" because the model initially scores them highly — they are semantically similar to the query but differ in the key ways that make them non-relevant.

**Why it matters:** If you train with only random negatives, the model easily learns to distinguish them (e.g., "what is Bitcoin" vs a document about cooking recipes). The decision boundary is coarse. With hard negatives, the model must learn subtle distinctions: e.g., "Bitcoin price prediction" vs a document about "Bitcoin historical price" — both are about Bitcoin prices, but only one answers the query.

**Process:**
1. Train an initial model with random negatives
2. Use the trained model to retrieve top-K candidates for training queries
3. Mark candidates that are NOT relevant as hard negatives
4. Retrain with these hard negatives
5. Repeat (ANCE: Approximate Nearest Neighbor Negative Contrastive Estimation)

Hard negative mining can improve NDCG@10 by 5-8 points on MS MARCO compared to random negative training.

---

### Q14 (Advanced): Design an evaluation framework for a new reranker at Binance.

**A:**

**Offline evaluation:**

1. **Internal test set**: 200 queries manually labeled by domain experts (Binance support team). Labels: 0 (not relevant), 1 (partially relevant), 2 (relevant), 3 (directly answers query). Use NDCG@10 as primary metric.

2. **BEIR zero-shot**: evaluate on FiQA (financial QA) and NFCorpus (domain-adjacent) to check generalization.

3. **Latency profiling**: measure p50, p95, p99 latency on a representative query workload. Fail gate: p95 > 100ms.

4. **Regression test**: compare NDCG@10 on a frozen benchmark against the current production model. New model must not regress more than 1%.

**Online A/B test:**

1. **Traffic split**: 5% of queries go to new reranker (treatment), 95% to existing (control)
2. **Primary metric**: session CTR (fraction of sessions with at least one click)
3. **Secondary metrics**: time to first click (TFC), number of searches per session (lower = better)
4. **Guardrails**: p99 latency, error rate, coverage (fraction of queries that return results)
5. **Duration**: run for 2 weeks minimum to account for day-of-week effects

**Decision criteria:**
- Ship if: CTR improves ≥ 2%, no latency regression, NDCG@10 improves ≥ 1%
- Investigate if: CTR improves but NDCG regresses (online and offline metrics disagree — check label quality)

---

### Q15 (Intermediate): What is the difference between Recall@K and Precision@K, and when does each matter?

**A:**

**Precision@K:**

$$\text{Precision@K} = \frac{\text{# relevant in top-}K}{K}$$

Measures: of the K documents I returned, what fraction were relevant? Cares about **quality of results shown to user**.

**Recall@K:**

$$\text{Recall@K} = \frac{\text{# relevant in top-}K}{\text{total # relevant}}$$

Measures: of all relevant documents, what fraction did I find in my top-K? Cares about **coverage**.

**When each matters:**

- **Precision@K**: user-facing search (Google-style). User sees K results; each should be good. A precision of 0.3 means 3 out of 10 shown results are relevant — user is annoyed.

- **Recall@K**: RAG retrieval stage. If Recall@100 = 0.6, the LLM only has access to 60% of relevant information even before reranking. The generator cannot cite what it doesn't see. High recall at retrieval stage is a prerequisite for high-quality RAG answers.

- **For two-stage systems**: optimize Recall@1000 at the retrieval stage (ensure all relevant documents make it to the candidate set), then optimize NDCG@10 at the reranking stage (put the best ones at the top).

---

## 15. Coding Problems

### 15.1 Cross-Encoder Reranker with HuggingFace

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple
import numpy as np

class CrossEncoderReranker:
    """
    Cross-encoder reranker using a BERT-based sequence classification model.

    Model: cross-encoder/ms-marco-MiniLM-L-6-v2
    - 6-layer MiniLM, fast and strong on MS MARCO
    - Output: logit for relevance (higher = more relevant)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_length: int = 512,
        batch_size: int = 32,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    @torch.no_grad()
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, str, float]]:
        """
        Rerank documents given a query.

        Returns list of (original_index, document, score) sorted by score desc.
        """
        scores = []

        # Process in batches to avoid OOM
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]

            # Tokenize: [CLS] query [SEP] document [SEP]
            encodings = self.tokenizer(
                [query] * len(batch_docs),
                batch_docs,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            # Forward pass
            logits = self.model(**encodings).logits

            # For binary cross-encoder, logits[:, 0] is irrelevance, logits[:, 1] is relevance
            # For single-output models, logits[:, 0] is the score
            if logits.shape[1] == 1:
                batch_scores = logits[:, 0].cpu().numpy()
            else:
                # Apply softmax and take relevance probability
                probs = torch.softmax(logits, dim=1)
                batch_scores = probs[:, 1].cpu().numpy()

            scores.extend(batch_scores.tolist())

        # Sort by score descending
        ranked = sorted(
            enumerate(zip(documents, scores)),
            key=lambda x: x[1][1],
            reverse=True,
        )

        results = [
            (orig_idx, doc, score)
            for orig_idx, (doc, score) in ranked[:top_k]
        ]
        return results


# Example usage
if __name__ == "__main__":
    reranker = CrossEncoderReranker()

    query = "How do I set a stop-loss on Binance?"
    candidates = [
        "To set a stop-loss on Binance, go to the trading interface and select Stop-Limit order.",
        "Binance offers various cryptocurrencies including Bitcoin and Ethereum.",
        "Risk management in crypto trading involves setting stop-loss orders to limit downside.",
        "Leverage trading requires understanding liquidation prices and margin requirements.",
        "A stop-loss order automatically closes your position when price hits a threshold.",
    ]

    results = reranker.rerank(query, candidates, top_k=3)
    for rank, (idx, doc, score) in enumerate(results, 1):
        print(f"Rank {rank} (original idx {idx}, score {score:.4f}): {doc[:60]}...")
```

---

### 15.2 NDCG@K from Scratch

```python
import numpy as np
from typing import List


def dcg_at_k(relevances: List[float], k: int) -> float:
    """
    Compute Discounted Cumulative Gain at K.

    DCG@K = sum_{i=1}^{K} (2^r_i - 1) / log2(i + 1)

    Args:
        relevances: list of relevance grades in ranked order
        k: cutoff rank

    Returns:
        DCG@K score
    """
    k = min(k, len(relevances))
    gains = np.array(relevances[:k], dtype=np.float64)
    # Position discounts: log2(2), log2(3), ..., log2(k+1)
    discounts = np.log2(np.arange(2, k + 2))  # arange(2, k+2) gives [2, 3, ..., k+1]
    return np.sum((2 ** gains - 1) / discounts)


def ndcg_at_k(relevances: List[float], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain at K.

    NDCG@K = DCG@K / IDCG@K

    Args:
        relevances: list of relevance grades in system-ranked order
        k: cutoff rank

    Returns:
        NDCG@K in [0, 1]
    """
    dcg = dcg_at_k(relevances, k)

    # Ideal ranking: sort by relevance descending
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal_relevances, k)

    if idcg == 0:
        return 0.0  # No relevant documents

    return dcg / idcg


def mean_ndcg_at_k(
    all_relevances: List[List[float]], k: int
) -> float:
    """Compute mean NDCG@K over multiple queries."""
    return np.mean([ndcg_at_k(rel, k) for rel in all_relevances])


def reciprocal_rank(relevances: List[int]) -> float:
    """
    Compute Reciprocal Rank: 1 / rank of first relevant document.

    Args:
        relevances: binary relevance list (1=relevant, 0=not) in ranked order

    Returns:
        Reciprocal rank (0 if no relevant doc found)
    """
    for i, rel in enumerate(relevances, 1):  # 1-indexed rank
        if rel > 0:
            return 1.0 / i
    return 0.0


def mean_reciprocal_rank(all_relevances: List[List[int]]) -> float:
    """Compute MRR over multiple queries."""
    return np.mean([reciprocal_rank(rel) for rel in all_relevances])


def average_precision(relevances: List[int]) -> float:
    """
    Compute Average Precision for a single query.

    AP = sum_k [P(k) * rel(k)] / num_relevant
    """
    num_relevant = sum(relevances)
    if num_relevant == 0:
        return 0.0

    precision_sum = 0.0
    hits = 0
    for k, rel in enumerate(relevances, 1):
        if rel > 0:
            hits += 1
            precision_sum += hits / k  # P(k) = hits so far / k

    return precision_sum / num_relevant


def mean_average_precision(all_relevances: List[List[int]]) -> float:
    """Compute MAP over multiple queries."""
    return np.mean([average_precision(rel) for rel in all_relevances])


# ── Tests & examples ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Example from Section 8.3
    system_grades = [2, 1, 3, 2, 0]   # system ranking
    print(f"DCG@5  = {dcg_at_k(system_grades, 5):.3f}")   # Expected: ~8.423
    print(f"NDCG@5 = {ndcg_at_k(system_grades, 5):.3f}")  # Expected: ~0.778

    # Perfect ranking
    perfect_grades = [3, 2, 2, 1, 0]
    print(f"NDCG@5 (perfect) = {ndcg_at_k(perfect_grades, 5):.3f}")  # Expected: 1.0

    # MRR example
    queries_binary = [
        [1, 0, 0, 1, 0],  # first relevant at rank 1 → RR = 1.0
        [0, 1, 0, 0, 1],  # first relevant at rank 2 → RR = 0.5
        [0, 0, 0, 1, 0],  # first relevant at rank 4 → RR = 0.25
    ]
    print(f"MRR = {mean_reciprocal_rank(queries_binary):.3f}")  # Expected: 0.583

    # MAP example
    map_query = [1, 0, 1, 0, 1]  # relevant at ranks 1, 3, 5
    print(f"AP = {average_precision(map_query):.3f}")  # Expected: ~0.756
```

---

### 15.3 Reciprocal Rank Fusion from Scratch

```python
from typing import List, Dict, Optional
from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: List[List[str]],
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[str]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.

    RRF(d) = sum_{r in R} w_r / (k + rank_r(d))

    Args:
        ranked_lists: list of ranked document ID lists (first = highest ranked)
        k: smoothing constant (default 60)
        weights: optional per-list weights (default: uniform)

    Returns:
        Fused ranked list of document IDs (highest RRF score first)
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    assert len(weights) == len(ranked_lists), "weights must match number of lists"

    rrf_scores: Dict[str, float] = defaultdict(float)

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, doc_id in enumerate(ranked_list, 1):  # 1-indexed
            rrf_scores[doc_id] += weight / (k + rank)

    # Sort by score descending
    fused_ranking = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return [doc_id for doc_id, _ in fused_ranking]


def rrf_with_scores(
    ranked_lists: List[List[str]],
    k: int = 60,
) -> List[tuple]:
    """Return (doc_id, rrf_score) pairs for inspection."""
    rrf_scores: Dict[str, float] = defaultdict(float)

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, 1):
            rrf_scores[doc_id] += 1.0 / (k + rank)

    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


# Example usage
if __name__ == "__main__":
    bm25_results  = ["A", "C", "B", "E", "D"]
    dense_results = ["C", "A", "D", "B", "E"]
    colbert_results = ["A", "B", "C", "D", "E"]

    fused = reciprocal_rank_fusion(
        [bm25_results, dense_results, colbert_results], k=60
    )
    print("Fused ranking:", fused)

    # With scores
    scored = rrf_with_scores(
        [bm25_results, dense_results, colbert_results], k=60
    )
    for doc, score in scored:
        print(f"  {doc}: {score:.6f}")
```

---

### 15.4 Full Hybrid Search + Reranking Pipeline

```python
"""
Full hybrid search + reranking pipeline.

Components:
  - BM25 via rank_bm25
  - Dense retrieval via sentence-transformers + FAISS
  - RRF fusion
  - Cross-encoder reranking
"""

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Dict, Tuple
from collections import defaultdict


class HybridSearchPipeline:
    def __init__(
        self,
        bi_encoder_model: str = "BAAI/bge-small-en-v1.5",
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        rrf_k: int = 60,
    ):
        # BM25: initialized per corpus
        self.bm25 = None
        self.corpus = []

        # Dense retrieval
        self.bi_encoder = SentenceTransformer(bi_encoder_model, device=device)
        self.faiss_index = None

        # Reranker
        self.ce_tokenizer = AutoTokenizer.from_pretrained(cross_encoder_model)
        self.ce_model = AutoModelForSequenceClassification.from_pretrained(
            cross_encoder_model
        ).to(device)
        self.ce_model.eval()

        self.device = device
        self.rrf_k = rrf_k

    def index(self, documents: List[str]) -> None:
        """Build BM25 and FAISS indexes for the corpus."""
        self.corpus = documents

        # BM25 index
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)

        # Dense index
        print(f"Encoding {len(documents)} documents...")
        embeddings = self.bi_encoder.encode(
            documents,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  # for cosine via dot product
        )

        # FAISS HNSW index for fast ANN
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors per node
        self.faiss_index.add(embeddings.astype(np.float32))

        print(f"Indexed {len(documents)} documents.")

    def _bm25_retrieve(self, query: str, top_k: int) -> List[int]:
        """Retrieve top_k document indices via BM25."""
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        # argsort descending
        top_indices = np.argsort(scores)[::-1][:top_k].tolist()
        return top_indices

    def _dense_retrieve(self, query: str, top_k: int) -> List[int]:
        """Retrieve top_k document indices via dense ANN."""
        query_emb = self.bi_encoder.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)
        _, indices = self.faiss_index.search(query_emb, top_k)
        return indices[0].tolist()

    def _rrf_fuse(
        self,
        ranked_lists: List[List[int]],
    ) -> List[int]:
        """RRF fusion over lists of document indices."""
        scores: Dict[int, float] = defaultdict(float)
        for ranked_list in ranked_lists:
            for rank, idx in enumerate(ranked_list, 1):
                scores[idx] += 1.0 / (self.rrf_k + rank)
        return sorted(scores, key=lambda x: scores[x], reverse=True)

    @torch.no_grad()
    def _cross_encoder_score(
        self, query: str, doc_indices: List[int]
    ) -> List[float]:
        """Score query-document pairs with cross-encoder."""
        docs = [self.corpus[i] for i in doc_indices]
        encodings = self.ce_tokenizer(
            [query] * len(docs),
            docs,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        logits = self.ce_model(**encodings).logits
        if logits.shape[1] == 1:
            return logits[:, 0].cpu().tolist()
        return torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

    def search(
        self,
        query: str,
        retrieval_top_k: int = 100,
        rerank_top_k: int = 10,
    ) -> List[Dict]:
        """
        Full hybrid search + reranking.

        Returns list of dicts with keys: rank, doc_index, document, score
        """
        # Stage 1: Retrieve candidates
        bm25_indices = self._bm25_retrieve(query, retrieval_top_k)
        dense_indices = self._dense_retrieve(query, retrieval_top_k)

        # Stage 2: RRF fusion
        fused_indices = self._rrf_fuse([bm25_indices, dense_indices])
        fused_indices = fused_indices[:retrieval_top_k]  # cap at top_k

        # Stage 3: Cross-encoder reranking
        ce_scores = self._cross_encoder_score(query, fused_indices)

        # Sort by cross-encoder score
        scored = sorted(
            zip(fused_indices, ce_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:rerank_top_k]

        results = [
            {
                "rank": rank,
                "doc_index": idx,
                "document": self.corpus[idx],
                "ce_score": score,
            }
            for rank, (idx, score) in enumerate(scored, 1)
        ]
        return results


# Example usage
if __name__ == "__main__":
    # Small crypto support corpus
    corpus = [
        "To set a stop-loss on Binance, navigate to the futures trading interface and select Stop-Limit order type.",
        "Bitcoin (BTC) is the first and largest cryptocurrency by market capitalization.",
        "Leverage in futures trading amplifies both gains and losses. 10x leverage means a 10% move liquidates your position.",
        "A stop-loss order automatically exits your position when the price reaches your specified level.",
        "Binance supports spot trading, margin trading, and perpetual futures contracts.",
        "Liquidation occurs when your margin balance falls below the maintenance margin requirement.",
        "To reduce risk in futures, always set a stop-loss and never risk more than 2% of your account per trade.",
        "Binance fees for futures trading are 0.02% maker and 0.05% taker.",
        "DeFi protocols allow decentralized trading through smart contracts on blockchains like Ethereum.",
        "USDT is a stablecoin pegged to the US dollar, commonly used as collateral in crypto trading.",
    ]

    pipeline = HybridSearchPipeline()
    pipeline.index(corpus)

    results = pipeline.search(
        "how do I protect myself from big losses in futures trading",
        retrieval_top_k=10,
        rerank_top_k=3,
    )

    for r in results:
        print(f"Rank {r['rank']} (score={r['ce_score']:.4f}): {r['document'][:80]}...")
```

---

### 15.5 MaxSim Scoring (ColBERT-style)

```python
"""
ColBERT-style MaxSim scoring.

For each query token embedding, find the maximum similarity
with any document token embedding, then sum across query tokens.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple
import numpy as np


class ColBERTScorer:
    """
    Simplified ColBERT scorer for demonstration.

    In production ColBERT v2:
    - Separate linear projection layers for query/doc
    - L2 normalization per token embedding
    - Query augmentation with [MASK] tokens to fixed length
    - INT8 quantization of doc token embeddings
    """

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        dim: int = 128,  # projection dimension
        device: str = "cpu",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        self.encoder.eval()

        # Linear projection to reduce dimensionality
        hidden_size = self.encoder.config.hidden_size
        self.linear = torch.nn.Linear(hidden_size, dim, bias=False).to(device)

        self.device = device
        self.dim = dim

    @torch.no_grad()
    def encode(self, texts: List[str], max_length: int = 128) -> List[torch.Tensor]:
        """
        Encode texts to per-token embeddings.
        Returns list of (seq_len, dim) tensors (one per text).
        """
        encodings = self.tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.encoder(**encodings)
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # Project to lower dimension
        projected = self.linear(token_embeddings)  # (batch, seq_len, dim)

        # L2 normalize each token embedding
        projected = F.normalize(projected, p=2, dim=-1)

        # Get actual sequence lengths (excluding padding)
        attention_mask = encodings["attention_mask"]  # (batch, seq_len)

        # Return list of (actual_length, dim) tensors
        result = []
        for i, mask in enumerate(attention_mask):
            actual_len = mask.sum().item()
            result.append(projected[i, :actual_len, :])  # remove padding

        return result

    def maxsim_score(
        self,
        query_embs: torch.Tensor,
        doc_embs: torch.Tensor,
    ) -> float:
        """
        Compute MaxSim score between query and document token embeddings.

        s(q, d) = sum_{i=1}^{m} max_{j=1}^{n} e_{q_i}^T e_{d_j}

        Args:
            query_embs: (m, dim) tensor of query token embeddings
            doc_embs:   (n, dim) tensor of document token embeddings

        Returns:
            Scalar MaxSim score
        """
        # (m, dim) x (dim, n) = (m, n) similarity matrix
        sim_matrix = torch.matmul(query_embs, doc_embs.T)

        # For each query token, take max similarity with any doc token
        max_sims = sim_matrix.max(dim=1).values  # (m,)

        # Sum across query tokens
        return max_sims.sum().item()

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 10,
    ) -> List[Tuple[int, str, float]]:
        """Rerank documents using MaxSim scoring."""
        # Encode query (single)
        query_embs_list = self.encode([query])
        q_embs = query_embs_list[0]  # (m, dim)

        # Encode all documents
        doc_embs_list = self.encode(documents)

        # Compute MaxSim scores
        scores = []
        for doc_embs in doc_embs_list:
            score = self.maxsim_score(q_embs, doc_embs)
            scores.append(score)

        # Sort by score descending
        ranked = sorted(
            enumerate(zip(documents, scores)),
            key=lambda x: x[1][1],
            reverse=True,
        )

        return [
            (orig_idx, doc, score)
            for orig_idx, (doc, score) in ranked[:top_k]
        ]


def numpy_maxsim(
    query_embs: np.ndarray,
    doc_embs: np.ndarray,
) -> float:
    """
    Numpy implementation of MaxSim for educational purposes.

    query_embs: (m, d)
    doc_embs:   (n, d)
    """
    # Similarity matrix: (m, n)
    sim_matrix = query_embs @ doc_embs.T

    # Max over document tokens for each query token
    max_sims = sim_matrix.max(axis=1)  # (m,)

    return float(max_sims.sum())


# Example usage
if __name__ == "__main__":
    scorer = ColBERTScorer()

    query = "bitcoin stop loss order"
    documents = [
        "A stop-loss order automatically closes your Bitcoin position at a target price.",
        "Bitcoin price reached a new all-time high above $100,000.",
        "Order management in crypto trading includes limit, market, and stop orders.",
    ]

    results = scorer.rerank(query, documents, top_k=3)
    for rank, (idx, doc, score) in enumerate(results, 1):
        print(f"Rank {rank} (score={score:.4f}): {doc[:60]}...")

    # Manual MaxSim with numpy
    np.random.seed(42)
    q_embs = np.random.randn(5, 128)  # 5 query tokens
    d_embs = np.random.randn(20, 128)  # 20 document tokens
    # Normalize
    q_embs /= np.linalg.norm(q_embs, axis=1, keepdims=True)
    d_embs /= np.linalg.norm(d_embs, axis=1, keepdims=True)

    score = numpy_maxsim(q_embs, d_embs)
    print(f"\nNumpy MaxSim score: {score:.4f}")
```

---

## Summary: Key Equations Reference

| Concept | Formula |
|---------|---------|
| Bi-encoder score | $s(q,d) = E_q^T E_d$ |
| Cross-encoder score | $s(q,d) = \text{BERT}([\texttt{CLS}]\, q\, [\texttt{SEP}]\, d\, [\texttt{SEP}])_{\texttt{CLS}} \cdot w^T$ |
| ColBERT MaxSim | $s(q,d) = \sum_{i=1}^{m} \max_{j=1}^{n} e_{q_i}^T e_{d_j}$ |
| RRF | $\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$, $k=60$ |
| DCG@K | $\sum_{i=1}^{K} \frac{2^{r_i}-1}{\log_2(i+1)}$ |
| NDCG@K | $\text{DCG@K} / \text{IDCG@K}$ |
| MRR | $\frac{1}{|Q|}\sum_q \frac{1}{\text{rank}_q}$ |
| MAP | $\frac{1}{|Q|}\sum_q \text{AP}(q)$; $\text{AP} = \frac{\sum_k P(k)\cdot\text{rel}(k)}{\#\text{relevant}}$ |
| Recall@K | $\frac{\|\text{relevant in top-K}\|}{\|\text{total relevant}\|}$ |
| KD loss | $\mathcal{L}_{KD} = \text{KL}(\text{softmax}(s_{CE}/T) \| \text{softmax}(s_{BE}/T))$ |
| MMR | $\lambda \cdot s(q,d_i) - (1-\lambda)\cdot\max_{d_j \in S}\text{sim}(d_i,d_j)$ |

---

*End of guide — ST5230 / Binance Interview Preparation*
*Last updated: March 2026*
