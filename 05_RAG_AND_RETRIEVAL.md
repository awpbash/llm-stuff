# RAG (Retrieval-Augmented Generation) and Retrieval Systems
## Comprehensive Technical Guide — ST5230 Interview Preparation

---

## Table of Contents
1. Why RAG? Motivation from First Principles
2. The Naive RAG Pipeline (and its problems)
3. Chunking Strategies — In Depth
4. Retrieval Methods
5. Advanced RAG Architectures
6. Reranking in RAG
7. Context Processing
8. Generation
9. Agentic RAG
10. Evaluation of RAG Systems
11. Vector Databases in Production
12. Problems & Mitigations (Dedicated Section)
13. Industry Practices
14. Interview Q&A
15. Coding Problems

---

## 1. Why RAG? Motivation from First Principles

### 1.1 Fundamental LLM Limitations

Large language models are trained on a static corpus and then frozen. This creates three deeply structural problems:

**Knowledge cutoff.** A model trained with data up to, say, October 2023 will have no knowledge of events after that date. If you ask it about a new regulation, a recent exploit, or a product launched last month, it will either refuse to answer or confabulate.

**Hallucination.** LLMs generate text by predicting the next token based on patterns in their parametric memory. When a question touches on a domain that is underrepresented or ambiguous in training data, the model will still produce fluent, confident text — but the content will be fabricated. The model cannot distinguish between what it actually "knows" and what it is plausibly guessing.

**No source attribution.** Even when an LLM produces a correct answer, it cannot cite which document it derived that answer from. This makes it unusable in any compliance or audit context where traceability matters.

### 1.2 The RAG Intuition

The core insight of RAG: instead of expecting the model to memorize all knowledge in its weights, supply the relevant information as part of the input context at inference time. The model's job then shifts from "recall this fact from parametric memory" to "read this passage and extract the answer."

This is analogous to an open-book exam. You are not tested on what you have memorized; you are given a reference booklet and tested on your ability to reason over it.

Formally, an LLM with parameters $\theta$ normally computes:

$$P(y \mid x; \theta)$$

where $x$ is the query. With RAG, we instead compute:

$$P(y \mid x, D_{\text{retrieved}}; \theta)$$

where $D_{\text{retrieved}} = \text{Retrieve}(x, \mathcal{K})$ is a set of documents retrieved from a knowledge base $\mathcal{K}$.

### 1.3 Open-Domain QA as the Original Motivation

RAG was introduced by Lewis et al. (2020) specifically to address open-domain question answering (ODQA). In ODQA, the system must answer factoid questions over Wikipedia or similar large corpora without knowing in advance which document contains the answer. Previous approaches either used pipelines with explicit document retrieval (ORQA, REALM) or relied purely on the LLM's parametric knowledge. RAG unified both by jointly training a dense retriever and a sequence-to-sequence generator.

### 1.4 RAG vs Fine-Tuning: When to Use Each

This is one of the most important architectural decisions you will face:

| Dimension | RAG | Fine-Tuning |
|---|---|---|
| Knowledge type | Factual, document-grounded | Behavioral, stylistic |
| Update frequency | Real-time (just update index) | Requires retraining |
| Interpretability | High (can show retrieved docs) | Low (parametric knowledge) |
| Hallucination risk | Lower (context grounded) | Higher |
| Cost at inference | Higher (retrieval + generation) | Lower (generation only) |
| Cost to update | Low | High |
| Best use case | Current regulations, product docs | Specific tone, domain jargon |

**Rule of thumb:** Use RAG when the knowledge changes frequently, when you need source attribution, or when the document corpus is large and diverse. Use fine-tuning when you need consistent behavior, specific output format, or domain-specific reasoning patterns that the base model lacks.

You can, and often should, use both simultaneously: fine-tune for behavior and retrieval-augment for knowledge.

### 1.5 Conceptual Diagram

```
                         KNOWLEDGE BASE
                        ┌─────────────────────────────┐
                        │  Document 1                  │
                        │  Document 2   ──► Embed ──►  │ Vector Index
                        │  Document N                  │
                        └─────────────────────────────┘
                                            │
                                            │  Top-K retrieval
                                            ▼
User Query ──► Embed Query ──► [Retrieve] ──► Retrieved Chunks
                                            │
                                            ▼
                                     [Augment Prompt]
                                     ┌──────────────────┐
                                     │ System: You are  │
                                     │ [Retrieved docs] │
                                     │ User: [Query]    │
                                     └──────────────────┘
                                            │
                                            ▼
                                        LLM Generate
                                            │
                                            ▼
                                        Response
```

---

## 2. The Naive RAG Pipeline (and Its Problems)

### 2.1 Full Pipeline

```
OFFLINE (INDEXING PHASE)
=======================

Raw Documents
     │
     ▼
┌─────────────┐
│   Chunking  │  Split docs into smaller passages
└─────────────┘
     │
     ▼
┌─────────────┐
│  Embedding  │  Encode each chunk with embedding model
└─────────────┘
     │
     ▼
┌─────────────┐
│   Index     │  Store vectors in FAISS / vector DB
└─────────────┘


ONLINE (QUERY PHASE)
====================

User Query
     │
     ▼
┌──────────────┐
│  Embed Query │  Same embedding model as indexing
└──────────────┘
     │
     ▼
┌──────────────┐
│ Retrieve K   │  ANN search: cosine/dot product similarity
│ Chunks       │
└──────────────┘
     │
     ▼
┌──────────────────────────────────────────┐
│  Prompt Assembly                          │
│  "Context:\n[chunk1]\n[chunk2]\n..."      │
│  "Question: [user query]\nAnswer:"        │
└──────────────────────────────────────────┘
     │
     ▼
┌──────────────┐
│   LLM Call   │
└──────────────┘
     │
     ▼
  Response
```

### 2.2 Each Step in Detail

**Chunking.** Documents are split into passages. The choice of chunk size is a fundamental tradeoff: large chunks have more context but reduce retrieval precision (you may retrieve an entire chapter when you only need one sentence); small chunks are more precise but may lose necessary context.

**Embedding.** Each chunk is encoded into a dense vector $\mathbf{e} \in \mathbb{R}^d$ using a bi-encoder model (e.g., `text-embedding-3-small`, `bge-large-en`, `e5-mistral-7b`). The key property of a good embedder: semantically similar texts should be close in the embedding space, measured by cosine similarity:

$$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$

**Indexing.** Vectors are stored in a data structure that supports approximate nearest neighbor (ANN) search. FAISS with IVF (Inverted File Index) or HNSW (Hierarchical Navigable Small World) are typical choices.

**Retrieval.** At query time, embed the query with the same model, then find the top-$K$ chunks with highest similarity to the query vector. Typically $K \in [3, 20]$.

**Generation.** Concatenate retrieved chunks into a context block, prepend a system prompt, append the user query, and call the LLM.

### 2.3 Problems with Naive RAG

**Chunking artifacts.** Fixed-size splitting can cut a table in half, split a code block in the middle, or separate a claim from its supporting evidence. The retrieved chunk is then incomplete and potentially misleading.

**Retrieval recall.** The relevant document may not be in the top-$K$ if the embedding model represents the query and document differently. For example, a query about "how to buy BTC" may not retrieve a chunk titled "Bitcoin purchase flow" if the embedder has not learned that "buy" = "purchase."

**Context relevance.** Topically related chunks may not actually answer the question. A query about "Ethereum gas fees" might retrieve chunks about "Ethereum history" or "Ethereum 2.0 staking" that are on-topic but not answering the specific question.

**Lost in the middle.** Empirical work by Liu et al. (2023) shows that LLMs exhibit strong positional bias: they tend to use information from the beginning and end of the context window, and largely ignore content placed in the middle. If the truly relevant chunk lands in the middle of a long context, performance degrades substantially.

**Faithfulness failure.** Even with correct retrieved context, the LLM may use its parametric memory rather than the provided context. This is particularly common when the parametric knowledge contradicts the retrieved information.

**Answer relevance.** The LLM may produce an answer that is technically supported by the retrieved docs but does not actually address what the user asked.

---

## 3. Chunking Strategies — In Depth

### 3.1 Fixed-Size Chunking

Split the text into windows of $N$ tokens with an overlap of $M$ tokens to preserve boundary context.

```python
def fixed_size_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    tokens = text.split()  # approximate tokenization
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunks.append(" ".join(tokens[start:end]))
        start += chunk_size - overlap
    return chunks
```

**Problems:** Completely ignores semantic structure. A chunk boundary may split:
- A table row from its header
- A code block in the middle of a function
- A numbered argument mid-sentence

### 3.2 Sentence-Based Chunking

Use NLP sentence boundary detection (NLTK's `sent_tokenize`, spaCy's sentence segmenter) to create chunks that always begin and end at sentence boundaries.

**Problems:** Sentences in technical documentation can be extremely long. A single sentence with a complex formula may exceed the embedding model's token limit. Adjacent sentences within a chunk may be thematically disconnected if the document jumps between topics.

### 3.3 Semantic Chunking

Embed each sentence individually. Compute the cosine similarity between adjacent sentence pairs. Where similarity drops sharply, insert a chunk boundary — this indicates a topic transition.

Let $\mathbf{s}_i$ be the embedding of sentence $i$. Compute:

$$\delta_i = 1 - \text{sim}(\mathbf{s}_i, \mathbf{s}_{i+1})$$

Insert a boundary at index $i$ when $\delta_i > \text{threshold}$, where the threshold is typically set at the 95th percentile of all $\delta$ values.

**Problems:** Computationally expensive at indexing time. Requires calling the embedding model $O(N)$ times per document. Threshold selection is heuristic and may fail on highly technical text where every sentence is topically distinct.

### 3.4 Recursive Character Splitting (LangChain Default)

Try splitting on progressively finer separators in order: `["\n\n", "\n", " ", ""]`. This respects natural structure (paragraphs before lines before words) and falls back gracefully.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_text(document_text)
```

This is a practical default that handles most document types well.

### 3.5 Document-Structure-Aware Chunking

Parse the document's native structure. For Markdown, split on `##` headers. For HTML, split on `<section>` or `<article>` tags. For code, split on function/class definitions using AST parsing.

```python
import ast

def chunk_python_file(source_code: str) -> list[dict]:
    tree = ast.parse(source_code)
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            start = node.lineno - 1
            end = node.end_lineno
            lines = source_code.split('\n')[start:end]
            chunks.append({
                "content": '\n'.join(lines),
                "type": type(node).__name__,
                "name": node.name
            })
    return chunks
```

### 3.6 Small-to-Big (Parent-Child) Chunking

Index small chunks (e.g., 100 tokens) for precise retrieval, but when a small chunk is retrieved, return its parent document (e.g., the full section or page) to the LLM.

```
Index:   [small chunk 1] [small chunk 2] [small chunk 3]  ← used for retrieval
                \              |              /
                 \             |             /
Retrieve:    [         Parent Document          ]  ← used for generation
```

This solves the precision-context tradeoff: small chunks enable precise matching, large parent chunks give the LLM sufficient context.

### 3.7 Hypothetical Document Embeddings (HyDE)

Instead of embedding the query directly, prompt the LLM to generate a hypothetical document that would answer the query, then embed that hypothetical document for retrieval.

**Intuition:** The query "What are Ethereum gas fees?" is short and query-like. A hypothetical answer "Ethereum gas fees are denominated in Gwei and represent the computational cost of executing transactions..." is passage-like and closer in embedding space to actual documentation passages.

Mathematically, instead of retrieving using $\mathbf{q} = \text{Embed}(\text{query})$, we use:

$$\hat{\mathbf{d}} = \text{Embed}(\text{LLM}(\text{query}))$$

as the retrieval vector.

**Problems:** Adds an LLM call in the retrieval path, increasing latency. If the LLM's hypothetical document contains hallucinated facts, those facts steer the retrieval toward wrong documents.

### 3.8 Choosing Chunk Size

| Factor | Recommendation |
|---|---|
| Embedding model max tokens | Never exceed (e.g., 512 for BERT-based, 8192 for OpenAI) |
| Query type: factoid | Small chunks (100-200 tokens) for precision |
| Query type: summarization | Larger chunks (500-1000 tokens) |
| Document type: legal/technical | Larger chunks to preserve argument integrity |
| Document type: FAQ/structured | Smaller chunks, one chunk per QA pair |

---

## 4. Retrieval Methods

### 4.1 Dense Retrieval (Bi-Encoder)

A bi-encoder independently encodes the query and document into dense vectors, then computes similarity.

```
Query ──► Encoder_Q ──► q ∈ R^d
                               \
                                ► sim(q, d) = q · d^T
                               /
Doc   ──► Encoder_D ──► d ∈ R^d
```

At indexing time, all document vectors are precomputed. At query time, only the query is encoded, and then ANN search finds the closest document vectors. This makes retrieval extremely fast ($O(\log N)$ with HNSW).

FAISS (Facebook AI Similarity Search) is the standard library:

```python
import faiss
import numpy as np

d = 768  # dimension
index = faiss.IndexFlatIP(d)  # inner product (for normalized vectors = cosine)
# Normalize vectors to unit length for cosine similarity
embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
index.add(embeddings)

# Retrieve top-K
query_vec = query_vec / np.linalg.norm(query_vec)
scores, indices = index.search(query_vec.reshape(1, -1), k=10)
```

### 4.2 Sparse Retrieval: BM25

BM25 (Best Match 25) is a bag-of-words retrieval function. It scores a document $d$ against a query $q$ by summing term-level IDF-weighted term frequencies with saturation and document length normalization:

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)(k_1+1)}{f(t,d) + k_1\left(1 - b + b\frac{|d|}{\text{avgdl}}\right)}$$

where:
- $f(t,d)$ = raw term frequency of term $t$ in document $d$
- $|d|$ = document length in words
- $\text{avgdl}$ = average document length across the corpus
- $k_1 \in [1.2, 2.0]$ = term frequency saturation parameter (typical: 1.5)
- $b \in [0, 1]$ = length normalization parameter (typical: 0.75)
- $\text{IDF}(t) = \ln\left(\frac{N - n_t + 0.5}{n_t + 0.5} + 1\right)$ where $N$ = corpus size and $n_t$ = number of documents containing $t$

**Why BM25 matters:** BM25 is exact match. If a query contains a specific acronym, product code, or proper noun (e.g., "BNB", "EIP-1559", "Satoshi"), BM25 will find it reliably regardless of whether the embedding model has good representations for that term.

### 4.3 Hybrid Retrieval

Combine dense (semantic) and sparse (lexical) retrieval to get the best of both.

**Score normalization before fusion.** Raw BM25 and cosine similarity scores are on completely different scales. Two approaches:

Min-max normalization:
$$\hat{s}_i = \frac{s_i - s_{\min}}{s_{\max} - s_{\min}}$$

Z-score normalization:
$$\hat{s}_i = \frac{s_i - \mu}{\sigma}$$

**Reciprocal Rank Fusion (RRF).** Rather than fusing scores (which requires calibration), fuse ranks. Given a document $d$ retrieved at rank $r(d)$ in each ranking $R$:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

where $k = 60$ is a smoothing constant that diminishes the importance of very high ranks. Documents retrieved highly in multiple systems will accumulate high RRF scores.

**Why $k=60$?** This was empirically determined to give good performance across multiple TREC runs. The constant prevents very highly ranked documents from dominating.

### 4.4 ColBERT: Multi-Vector Retrieval

ColBERT (Contextualized Late Interaction over BERT) addresses a fundamental limitation of bi-encoders: compressing an entire document into a single vector loses token-level information.

ColBERT produces one vector per token. At retrieval, it computes a **MaxSim** operation:

$$s(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \mathbf{E}_i \cdot \mathbf{F}_j^T$$

where $\mathbf{E}_i$ is the embedding of the $i$-th query token and $\mathbf{F}_j$ is the embedding of the $j$-th document token. For each query token, find the most similar document token; sum these maximum similarities.

**Advantage:** Much richer interaction between query and document. A query token "fees" can match a document token "gas" if they are contextually related.

**Disadvantage:** Storage cost is $O(\text{tokens})$ per document instead of $O(1)$. Retrieval requires comparing all query token vectors against all document token vectors, though this can be made efficient with pre-computed document matrices.

### 4.5 Problems in Retrieval

**Semantic gap.** Query vocabulary and document vocabulary may differ. A user asking "what happens when a smart contract execution runs out of resources?" wants to know about gas limits, but may not use the word "gas" or "gas limit." Dense retrieval helps close this gap but is not perfect.

**False positives.** Documents that are semantically similar to the query but do not actually contain the answer. For example, "Ethereum price prediction" and "Ethereum gas fees" are both about Ethereum but are not interchangeable answers.

**Out-of-distribution queries.** If the embedding model was not trained on text similar to your domain (e.g., financial instruments, DeFi protocols), its representations will be poor and retrieval quality will suffer.

---

## 5. Advanced RAG Architectures

### 5.1 Query Rewriting

The user's natural language query is often a poor retrieval query. It may be conversational, ambiguous, or contain coreferences ("what about this one?" makes no sense out of context). An LLM rewrites the query into a more retrieval-friendly form.

```
User: "what about the fee structure for that?"
      │
      ▼
LLM rewrite: "What are the trading fee tiers and structures on Binance?"
      │
      ▼
Retrieval on rewritten query
```

### 5.2 Query Decomposition

For multi-hop questions that require combining information from multiple documents, decompose the query into sub-questions.

**Sequential (chain-of-thought) decomposition:**
```
Q: "What is the difference in gas fees between Ethereum and Binance Smart Chain?"

Step 1: Retrieve and answer "What are Ethereum gas fees?"
        → "ETH gas fees average X Gwei..."

Step 2: Use step 1 answer + retrieve "What are BSC gas fees?"
        → "BSC gas fees are Y Gwei..."

Step 3: Synthesize comparison
```

**Parallel decomposition:**
```
Q: "What's the cheapest way to move assets from ETH to BNB chain?"

Sub-Q1: "What are bridging options from ETH to BSC?"
Sub-Q2: "What are cross-chain transfer fees for major bridges?"
Sub-Q3: "What are the security trade-offs of different bridges?"

All three retrieved simultaneously → merge context → final answer
```

### 5.3 Step-Back Prompting

Before retrieving, abstract the question to a higher-level concept. This helps when the specific question is rare but a more general question is well-covered in the corpus.

```
Specific: "Why did ETH gas fees spike on January 5, 2023?"
          │
          ▼
Step back: "What factors cause Ethereum gas fee spikes?"
          │
          ▼
Retrieve on abstracted question → get general context
          │
          ▼
Use general context to answer specific question
```

### 5.4 Multi-Query Retrieval

Generate $N$ different phrasings of the query, retrieve for each, and take the union of all retrieved sets.

```python
queries = llm.generate_variants(user_query, n=3)
# e.g.:
# 1. "Binance withdrawal limits for crypto"
# 2. "Maximum daily withdrawal amount Binance"
# 3. "Binance account withdrawal restrictions and caps"

all_docs = set()
for q in queries:
    docs = retriever.retrieve(q, k=5)
    all_docs.update(docs)
# Rerank all_docs
final_docs = reranker.rerank(user_query, list(all_docs), top_k=5)
```

### 5.5 Advanced RAG Pipeline Diagram

```
User Query
    │
    ▼
┌──────────────────────────────────────────────┐
│              Query Processing                 │
│  ┌──────────┐  ┌────────────┐  ┌──────────┐ │
│  │Rewriting │  │Decomposit. │  │Multi-Q   │ │
│  └──────────┘  └────────────┘  └──────────┘ │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│           Multi-Source Retrieval              │
│  ┌──────────┐  ┌──────────┐  ┌────────────┐ │
│  │  Dense   │  │  Sparse  │  │ Knowledge  │ │
│  │  (FAISS) │  │  (BM25)  │  │  Graph     │ │
│  └──────────┘  └──────────┘  └────────────┘ │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│          RRF Fusion + Reranking               │
│  Dense results + Sparse results              │
│  ──────────────────────────────►  RRF Score  │
│                                  Cross-Encoder│
│                                  Reranker    │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│           Context Processing                  │
│  Compression | Deduplication | Citation tags  │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│              LLM Generation                   │
│  System prompt + Context + Query             │
└──────────────────────────────────────────────┘
    │
    ▼
  Response with Citations
```

---

## 6. Reranking in RAG

### 6.1 Why Top-K Retrieval Needs Refinement

Bi-encoder retrieval is a tradeoff: encoding query and document independently enables fast ANN search, but it loses all token-level interaction. As a result, the top-$K$ retrieved set has good recall (the right document is usually in there) but imperfect precision (many documents in the top-$K$ are not truly relevant).

Reranking is a second-stage filtering step that applies a more expensive but more accurate model to the candidate set.

### 6.2 Cross-Encoder Reranking

A cross-encoder takes the concatenation of query and document as input and produces a single relevance score:

```
[CLS] query [SEP] document [SEP]
              │
         BERT encoder
              │
        score ∈ R (fine-tuned on MS-MARCO or similar)
```

This allows full token-level interaction between query and document, producing much higher quality relevance scores. The cost is $O(K)$ forward passes through BERT, each of which is expensive — hence reranking can only be done on a small candidate set (top-50 to top-100 from the initial retrieval).

**Latency vs quality tradeoff:** Bi-encoder retrieval is $O(\log N)$ (ANN search). Cross-encoder reranking is $O(K \cdot L)$ where $L$ is document length. For $K=50$ and $L=512$ tokens, this is roughly 50x slower than retrieval but far more accurate.

### 6.3 Position of Reranker in Full Pipeline

```
Query
  │
  ▼
Dense Retrieval (top-100)
  │
  ├──► Sparse Retrieval (top-100)
  │
  ▼
RRF Fusion (top-100 combined)
  │
  ▼
Cross-Encoder Reranker (top-100 → top-5)
  │
  ▼
Context Assembly + LLM Generation
```

Models: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast), `Cohere Rerank API` (high quality), `bge-reranker-large` (open source).

---

## 7. Context Processing

### 7.1 Lost in the Middle Problem

Liu et al. (2023) showed that LLMs have a pronounced U-shaped attention pattern over long contexts: they reliably attend to the beginning and end of the context window, but attention deteriorates for documents placed in the middle.

In a 20-document context window, if the relevant document is at position 10-15, accuracy drops by 20-30% compared to when it is at position 1 or 20.

**Mitigation strategies:**
1. Place the most relevant document first (rank by reranker score, not retrieval order)
2. Duplicate the most critical document at both the beginning and end of context
3. Keep the context window small — use fewer, more relevant documents rather than many marginal ones

### 7.2 Context Window Management

When retrieved documents exceed the model's context limit:

**Truncation:** Simply cut off after $N$ tokens. Simple but loses information. Prefer truncating lower-relevance chunks.

**Recursive summarization:** Summarize each retrieved document individually, then concatenate summaries.

**Map-reduce:**
```
Retrieved doc 1 ──► LLM("Extract relevant info for query Q") ──► summary 1
Retrieved doc 2 ──► LLM("Extract relevant info for query Q") ──► summary 2
Retrieved doc 3 ──► LLM("Extract relevant info for query Q") ──► summary 3
                                                                     │
                                   LLM("Synthesize: " + summaries) ─┘
                                                │
                                             Final answer
```

### 7.3 Context Compression

**LLMLingua** (Microsoft): applies a small proxy LLM to select which tokens are most important, then compresses the context by removing less important tokens. Can achieve 4-8x compression with minimal accuracy loss.

**Selective sentence extraction:** Use a classification model to score each sentence in each retrieved document on relevance to the query. Keep only the top-$P$% of sentences.

### 7.4 Citation and Attribution

To make the LLM cite which document each statement came from:

1. Tag each retrieved chunk with an identifier: `[DOC_1]`, `[DOC_2]`, etc.
2. Include in the system prompt: "After each claim, cite the document identifier in square brackets."
3. Post-process to verify that cited documents actually support the claim (faithfulness check).

```
System: Answer the question using only the provided documents.
        After each factual claim, cite the source document in brackets.

[DOC_1] Ethereum gas fees are denominated in Gwei...
[DOC_2] Binance Smart Chain uses BNB for gas...

Question: Compare gas fees between ETH and BSC.

Answer: Ethereum gas is denominated in Gwei [DOC_1], while BSC uses
        BNB for transaction fees [DOC_2].
```

---

## 8. Generation

### 8.1 Prompt Template Design

A well-structured RAG prompt has three components:

```
SYSTEM:
You are a helpful assistant for Binance financial products.
Answer questions based ONLY on the provided context documents.
If the context does not contain enough information, say "I don't know."
Do not use any knowledge outside of the provided documents.

CONTEXT:
[Document 1]: {chunk_1_text}
[Document 2]: {chunk_2_text}
...

USER:
{user_query}
```

Key design decisions:
- Explicitly instruct the model to use only the context (reduces parametric override)
- Include a "say I don't know" instruction (reduces hallucination)
- Format context with document labels (enables citation)

### 8.2 Chain-of-Thought for RAG

For complex questions, ask the model to reason step by step before producing the final answer:

```
After reading the documents, reason through the evidence before answering:
1. What does each document say about [topic]?
2. Are there any contradictions between documents?
3. What is the most well-supported answer?
Final Answer: ...
```

### 8.3 Self-RAG

Self-RAG (Asai et al., 2023) trains the LLM with special reflection tokens that allow it to:
1. Decide whether to retrieve at all for a given query
2. Evaluate the quality of retrieved documents
3. Evaluate whether its own output is supported by the retrieved context

Special tokens and their semantics:
- `[Retrieve]` — model decides to retrieve before continuing generation
- `[No Retrieve]` — model decides to answer from parametric memory
- `[Relevant]` — model judges retrieved doc as relevant
- `[Irrelevant]` — model judges retrieved doc as not relevant
- `[Supported]` — model judges its output as supported by context
- `[Partially Supported]` — model judges output as partially supported
- `[Contradictory]` — model output contradicts retrieved context
- `[No support / Not sure]` — model is uncertain about support

This converts RAG from a static pipeline into an adaptive system where retrieval is demand-driven.

### 8.4 FLARE (Forward-Looking Active Retrieval)

FLARE retrieves only when the model is uncertain. The model generates text token by token. When it encounters a sequence of low-probability tokens (the model is "not sure"), it pauses, formulates a retrieval query based on what it was about to say, retrieves relevant documents, and continues generation with the new context.

Uncertainty is measured as: retrieve when $P(\text{token}) < \delta$ for some threshold $\delta$.

This is more efficient than retrieving upfront for every query: short factual queries may not need retrieval, while complex ones trigger multiple retrieval calls exactly where needed.

---

## 9. Agentic RAG

### 9.1 ReAct (Reason + Act)

ReAct (Yao et al., 2022) interleaves reasoning traces and actions. The LLM alternates between:
- **Thought:** "I need to find the current trading volume for BNB"
- **Action:** `search("BNB 24h trading volume")`
- **Observation:** `"BNB trading volume: $1.2B in the past 24 hours"`
- **Thought:** "Now I have the volume. I need the price as well."
- **Action:** `search("BNB current price USD")`
- **Observation:** `"BNB price: $312"`
- **Final Answer:** "BNB has a 24h trading volume of $1.2B at a price of $312."

Retrieval is one tool in the toolbox alongside web search, database queries, code execution, and calculator.

### 9.2 Iterative Retrieval

Rather than a single retrieval step, the agent can retrieve, read, decide if more context is needed, and retrieve again:

```
Initial Query
    │
    ▼
Retrieve (round 1)
    │
    ▼
Read docs: Is this sufficient to answer?
    │
    ├── Yes ──► Generate final answer
    │
    └── No  ──► Formulate follow-up query ──► Retrieve (round 2) ──► ...
```

### 9.3 Graph RAG

For complex multi-hop reasoning, build a knowledge graph (KG) from documents: entities as nodes, relationships as edges. Traverse the graph rather than retrieving flat chunks.

```
Documents ──► Entity extraction ──► Relation extraction ──► Knowledge Graph
                                                                  │
Query ──► Entity linking ──► Graph traversal (multi-hop) ──► Subgraph
                                                                  │
                                                             LLM Synthesis
```

**Example:** "Who founded the company that created Ethereum?" requires linking Ethereum → Vitalik Buterin via "founded by" edge. Flat retrieval would need both facts to appear in the same chunk.

### 9.4 Agentic RAG Loop Diagram

```
    ┌──────────────────────────────────────────────────────┐
    │                   AGENT LOOP                          │
    │                                                        │
    │   User Query                                          │
    │       │                                               │
    │       ▼                                               │
    │   ┌────────┐   Thought    ┌──────────────────────┐   │
    │   │        │ ──────────►  │   Available Tools    │   │
    │   │  LLM   │              │  - VectorSearch(q)   │   │
    │   │        │ ◄──────────  │  - WebSearch(q)      │   │
    │   └────────┘ Observation  │  - Calculator(expr)  │   │
    │       │                   │  - CodeExecute(code) │   │
    │       │                   └──────────────────────┘   │
    │       │ (iterate until done)                          │
    │       ▼                                               │
    │   Final Answer                                        │
    └──────────────────────────────────────────────────────┘
```

---

## 10. Evaluation of RAG Systems

### 10.1 Component-Level vs End-to-End Evaluation

Component-level evaluation isolates each stage:
- Retrieval quality: Precision@K, Recall@K, MRR, NDCG
- Generation quality: ROUGE, BERTScore, human evaluation

End-to-end evaluation measures the final answer quality against a ground truth answer, without caring which component failed.

### 10.2 RAGAS Framework

RAGAS (RAG Assessment) defines four key metrics:

**Faithfulness** measures whether the generated answer is grounded in the retrieved context:

$$\text{Faithfulness} = \frac{\text{number of claims in answer supported by context}}{\text{total number of claims in answer}}$$

Claims are extracted from the answer using an LLM, then each claim is verified against the context.

**Answer Relevance** measures whether the answer addresses the question. Generate $N$ questions from the answer and measure their cosine similarity to the original question:

$$\text{AnswerRelevance} = \frac{1}{N}\sum_{i=1}^{N} \text{sim}(\mathbf{q}_i, \mathbf{q}_{\text{original}})$$

**Context Precision** measures the proportion of retrieved context that is actually relevant, weighted by rank:

$$\text{ContextPrecision@K} = \frac{\sum_{k=1}^{K} (\text{Precision@k} \times \mathbf{1}[\text{chunk}_k \text{ is relevant}])}{K}$$

**Context Recall** measures what fraction of the ground truth information is covered by the retrieved context:

$$\text{ContextRecall} = \frac{\text{number of ground truth facts covered by context}}{\text{total number of ground truth facts}}$$

### 10.3 Other Frameworks

**TruLens:** Open-source framework that adds feedback functions to any LLM app. Provides a dashboard for tracking all four RAGAS metrics across queries over time.

**DeepEval:** Testing framework for LLM applications. Supports G-Eval (LLM-as-judge), hallucination detection, answer correctness metrics.

**LLM-as-Judge:** Use a strong LLM (GPT-4) to evaluate answers on a Likert scale:
- Relevance: Does the answer address the question?
- Accuracy: Are the stated facts correct?
- Completeness: Does the answer cover all aspects?
- Groundedness: Is the answer supported by the provided context?

```python
def llm_judge(question: str, answer: str, context: str) -> dict:
    prompt = f"""
    Question: {question}
    Retrieved Context: {context}
    Generated Answer: {answer}

    Rate the answer on:
    1. Faithfulness (1-5): Is the answer supported by the context?
    2. Relevance (1-5): Does the answer address the question?
    3. Completeness (1-5): Is the answer complete?

    Output as JSON.
    """
    return gpt4(prompt)
```

---

## 11. Vector Databases in Production

### 11.1 Comparison of Vector Databases

| Database | Type | Scale | Strengths | Weaknesses |
|---|---|---|---|---|
| Pinecone | Managed | Billions | Easy to use, managed | Expensive, vendor lock-in |
| Weaviate | Self-hosted/managed | Hundreds of millions | Rich schema, multi-modal | Complex setup |
| Qdrant | Self-hosted/managed | Hundreds of millions | Fast, Rust-based, payload filtering | Smaller ecosystem |
| Chroma | Self-hosted | Millions | Simple API, great for prototyping | Not production-scale |
| pgvector | Self-hosted | Tens of millions | PostgreSQL integration, SQL filters | Slower for pure vector search |
| Milvus | Self-hosted | Billions | Highly scalable, enterprise | Complex deployment |

**Choosing a vector DB:**
- Prototyping / small scale: Chroma or pgvector
- Mid-scale, self-hosted: Qdrant
- Large scale, managed: Pinecone or Weaviate Cloud
- Enterprise, on-premise: Milvus

### 11.2 Metadata Filtering

In production, you almost always need to filter documents before or after vector search. For example: "Find documents from the last 30 days that mention 'staking' and belong to category 'DeFi'."

**Pre-filtering:** Apply metadata filter to narrow the candidate set, then run vector search within the filtered set.
```
Filter: date > 2024-01-01 AND category = "DeFi"
                     │
              [filtered index subset]
                     │
              vector search for "staking"
```

**Post-filtering:** Run vector search on full index, then filter results.

Pre-filtering is generally better: you avoid retrieving irrelevant documents. However, if the filter is too narrow, you may not have enough candidates for meaningful ANN search.

### 11.3 Multi-Tenancy

In a multi-user system (e.g., different enterprise customers), you need to ensure User A cannot retrieve User B's documents.

**Namespace isolation:** Separate namespace per tenant. Each tenant's documents live in their own partition. Vector search is scoped to a namespace.

**Metadata-based filtering:** Store `tenant_id` as metadata, filter on it at query time. Simpler but relies on filtering being correctly applied every time.

**Separate indices:** Most isolated but expensive in resource usage.

### 11.4 Index Updates

**Upsert:** Insert new documents or update existing ones by ID. Most vector DBs support upsert natively.

**Deletes:** Mark vectors as deleted (soft delete) or physically remove them. Some ANN indices (HNSW) require periodic compaction after many deletes.

**Index rebuild:** Periodically rebuild the entire index for optimal search performance. HNSW graph degrades over time with many incremental inserts.

---

## 12. Problems & Mitigations — Dedicated Section

### 12.1 Hallucination Despite Retrieval Context

**Problem:** Even with correct retrieved context, the LLM generates facts not supported by that context (parametric override).

**Mitigations:**
1. **Explicit instruction:** "Answer ONLY using the provided context. Do not use any external knowledge."
2. **Citation enforcement:** Require citations; if a claim has no citation, it was likely hallucinated.
3. **Faithfulness fine-tuning:** Fine-tune the generator on a dataset of (context, question, grounded-answer) triples.
4. **Faithfulness post-check:** Use RAGAS faithfulness metric to auto-flag low-faithfulness responses.
5. **NLI-based verification:** Use a natural language inference model to verify each answer sentence is entailed by the context.

### 12.2 Retrieval Latency

**Problem:** Dense retrieval over billions of vectors can take hundreds of milliseconds; adding reranking adds more.

**Mitigations:**
1. **ANN (approximate) search:** HNSW, IVF-PQ trade small accuracy loss for large speed gain.
2. **Query caching:** Cache embeddings and results for repeated queries. In crypto support, many queries repeat ("what are trading fees?").
3. **Asynchronous retrieval:** Start retrieval as the user is still typing (streaming input).
4. **Two-stage cascade:** Fast BM25 first (< 5ms), then dense retrieval only on BM25 shortlist.
5. **Quantization:** Reduce embedding precision from float32 to int8 (4x smaller index, 2-4x faster search).

### 12.3 Irrelevant Retrieved Documents

**Problem:** Top-$K$ retrieval returns documents that are topically related but do not answer the question.

**Mitigations:**
1. **Cross-encoder reranking:** More expensive but eliminates most irrelevant results.
2. **Query rewriting:** Reformulate the query to be more specific.
3. **Maximum Marginal Relevance (MMR):** Select documents that are relevant to the query AND diverse from each other:

$$\text{MMR} = \arg\max_{d_i \in R \setminus S} \left[\lambda \cdot \text{sim}(d_i, q) - (1-\lambda) \cdot \max_{d_j \in S} \text{sim}(d_i, d_j)\right]$$

where $S$ is the set of already-selected documents, $R$ is the retrieved set. This prevents retrieving 5 nearly-identical documents.

### 12.4 Chunk Boundary Issues

**Problem:** Fixed-size chunking splits logical units (tables, code blocks, arguments).

**Mitigations:**
1. **Overlap:** 10-20% token overlap so boundary content appears in multiple chunks.
2. **Parent-child chunking:** Retrieve small chunks, return parent context.
3. **Sentence-aware splitting:** Never cut mid-sentence.
4. **Document-structure-aware splitting:** Respect semantic boundaries (headers, code fences).

### 12.5 Long Document RAG

**Problem:** A single document (e.g., a 200-page whitepaper) is too long to include in context but too important to chunk naively.

**Mitigations:**
1. **Hierarchical chunking:** Create a document summary + section summaries + paragraph chunks. Query at multiple levels.
2. **Recursive summarization:** Summarize sections, then summarize summaries.
3. **Section routing:** Use the document's table of contents to route the query to the correct section first.

### 12.6 Conflicting Information in Retrieved Chunks

**Problem:** Two retrieved chunks say opposite things (one says "Binance fees are 0.1%", another says "Binance fees are 0.05% with BNB discount").

**Mitigations:**
1. **Provenance metadata:** Tag each chunk with source, date. In conflict, prefer newer source.
2. **LLM resolution:** Prompt the LLM: "If documents conflict, note the conflict and explain the different versions."
3. **Confidence scoring:** When conflict detected, lower the answer's confidence score.

---

## 13. Industry Practices

### 13.1 RAG at Binance Scale

Binance operates a massive knowledge base covering:
- Regulatory documents across 100+ jurisdictions
- DeFi protocol documentation
- Trading rules and fee schedules
- Security advisories and incident reports
- Whitepaper summaries for 1000+ tokens

**Architecture at scale:**
1. **Multi-source ingestion:** Separate ingestion pipelines for each source type with source-specific chunking (legal PDFs: section-level; technical docs: function-level; FAQs: one-chunk-per-QA).
2. **Multilingual index:** Separate indices per language OR cross-lingual model (LaBSE, multilingual-e5).
3. **Tiered retrieval:** BM25 for exact matches (ticker symbols, regulation names), dense for semantic matches.
4. **Freshness management:** Mark documents with `last_updated`. Deprecate old regulatory docs when superseded.

### 13.2 Multilingual RAG

**Cross-lingual retrieval:** A query in Chinese should retrieve English documents if they are semantically relevant. Use a multilingual embedding model (LaBSE, multilingual-e5-large) that maps languages to a shared embedding space.

**Monolingual vs multilingual index:**
- Per-language index: Better precision within a language, no cross-lingual matching.
- Unified multilingual index: One index, cross-lingual matching works, but precision may be lower.

**Translation-based approach:** Translate all documents to English, index English versions. Translate queries to English before retrieval. High quality but expensive.

### 13.3 Real-Time RAG

For retrieval over live data (live trading prices, real-time news):
1. **Streaming ingest:** New documents are embedded and indexed as they arrive. Apache Kafka + consumer workers.
2. **Incremental index updates:** Qdrant and Pinecone support real-time upsert with no downtime.
3. **TTL (time-to-live) on chunks:** Old price data expires automatically.
4. **Hybrid: cached + live:** Static knowledge (regulations) in a stable index; live data in a frequently-refreshed hot index.

### 13.4 RAG for Code

**Chunking:** Function-level or class-level using AST parsing. Never split mid-function.

**Code-specific embedders:**
- `code-search-net` embeddings
- `text-embedding-3-large` (OpenAI) is reasonable for mixed code/text
- StarCoder embeddings for code-dominant corpora

**Additional context:** Include function signature, docstring, and imports even if the function body is the primary chunk.

### 13.5 Cost Optimization

**Cache common queries:** A Redis cache keyed by (`query_hash`, `index_version`). Crypto support sees high query repetition.

**Tiered retrieval:** BM25 first (virtually free), then dense retrieval only when BM25 confidence is low. BM25 handles most FAQ-type queries perfectly well.

**Smaller embedding models for indexing:** Use `text-embedding-3-small` (1536d) instead of `text-embedding-3-large` (3072d) for cost reduction at slight quality loss.

**Batched embedding:** Embed documents in batches during off-peak hours. Avoid embedding one-by-one.

---

## 14. Interview Q&A

### Q1 (Basic): What is RAG and why do we need it?

**Answer:** RAG stands for Retrieval-Augmented Generation. It addresses the core limitations of standalone LLMs: knowledge cutoff (the model doesn't know about recent events), hallucination (the model invents plausible-sounding but false facts), and lack of source attribution (you can't trace where an answer came from).

The idea is to augment the LLM's context at inference time with documents retrieved from an external knowledge base. Instead of asking the model to recall facts from its weights, you hand it the relevant text and ask it to extract/synthesize the answer. The model then acts as a reader and reasoner rather than a memory.

RAG is preferred over fine-tuning for dynamic, factual knowledge because: (a) you can update the knowledge base without retraining the model, (b) you get natural source attribution, and (c) it reduces hallucination by grounding answers in text.

---

### Q2 (Basic): Explain the difference between dense and sparse retrieval.

**Answer:** Sparse retrieval (BM25) uses exact term matching with IDF weighting. It represents documents as sparse vectors over the vocabulary. Its strength is handling specific terms — if the query contains "EIP-1559", BM25 will find documents containing that exact term. Its weakness is the vocabulary mismatch problem: it cannot match "purchase" with "buy."

Dense retrieval (bi-encoder) uses a neural network to encode both query and document into dense vectors in $\mathbb{R}^d$. Similarity is measured by dot product or cosine similarity in this embedding space. It captures semantic meaning but may miss exact lexical matches and requires more compute.

In practice, hybrid retrieval combining both (via RRF) outperforms either alone. Use BM25 for named entities and technical terms; use dense for semantic similarity.

---

### Q3 (Basic): What is chunking and why does it matter?

**Answer:** Chunking is the process of splitting long documents into shorter passages that can be independently indexed and retrieved. It matters for two reasons:

First, embedding models have token limits (512 tokens for BERT-based models, 8192 for newer ones). Documents longer than this cannot be embedded as a unit.

Second, retrieval precision. If you embed an entire 50-page whitepaper as one vector, the resulting embedding captures the "average" of the whole document, not any specific fact. When you retrieve it, you get the whole document but may only need one paragraph. Smaller chunks enable more precise retrieval.

The tradeoff: too small and you lose context (a sentence without its surrounding paragraph may be ambiguous); too large and you lose precision.

---

### Q4 (Intermediate): How would you build a RAG system for Binance's customer support?

**Answer:** I would design the system in layers:

**Data ingestion pipeline:**
- Sources: Binance help center articles, FAQ pages, trading rule PDFs, regulation docs, announcement posts.
- Chunking: Help center articles split at H2/H3 headers (structure-aware). Legal PDFs split at section level. FAQs: one chunk per question-answer pair (natural unit of retrieval).
- Metadata: `source_url`, `last_updated`, `language`, `category` (fees, security, trading, regulations), `jurisdiction`.

**Retrieval:**
- Hybrid BM25 + dense (multilingual-e5-large for multilingual support).
- Pre-filter by `language` and `category` before vector search to reduce irrelevant results.
- Cross-encoder reranker (bge-reranker) for top-50 → top-5 refinement.

**Generation:**
- Strict system prompt: answer only from context, cite sources.
- Chain-of-thought for complex fee calculations.
- Confidence signal: if all retrieved docs score below a threshold, route to human agent.

**Evaluation:**
- RAGAS metrics for faithfulness and context precision.
- A/B testing: RAG vs baseline LLM on held-out support tickets.
- Track escalation rate to human agents (lower = better).

**Production concerns:**
- Cache frequent queries (trading fees, withdrawal limits — asked thousands of times per hour).
- Freshness: regulatory docs need daily refresh pipeline.
- Fallback: if RAG fails (low confidence), route to human or show "I don't have that information, please contact support."

---

### Q5 (Intermediate): Explain RRF and when to use it.

**Answer:** Reciprocal Rank Fusion (RRF) is a rank-based score fusion method for combining results from multiple retrieval systems without requiring score normalization. Given a document $d$ appearing at rank $r(d)$ in each ranked list $R$:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

The constant $k=60$ is a smoothing factor that prevents extremely highly-ranked documents from completely dominating.

**Why rank-based instead of score-based?** Raw BM25 scores and cosine similarity scores are on completely different scales and distributions. Normalizing them requires knowing the distribution of each, which varies per query. Rank fusion sidesteps this problem entirely — rank 1 means "best match for this system" regardless of the raw score.

**When to use it:** Use RRF whenever you are combining results from two or more retrieval systems (e.g., dense + sparse, multiple embedding models, multiple indices). It consistently outperforms score-based fusion in benchmarks. It is also trivial to extend to 3+ systems.

**Limitation:** RRF discards information about score magnitude. If the dense retriever gives document A a score of 0.99 and document B a score of 0.51, RRF treats them as simply rank 1 and rank 2. This can lose signal when score differences are large and meaningful.

---

### Q6 (Intermediate): What is HyDE and when does it help?

**Answer:** Hypothetical Document Embeddings (HyDE) addresses the asymmetry between queries and documents in embedding space. Queries are typically short ("what are ETH gas fees?"), while indexed documents are long passages ("Ethereum gas fees are denominated in Gwei, where 1 Gwei = $10^{-9}$ ETH..."). Even with a good embedder, these live in somewhat different regions of the embedding space.

HyDE generates a hypothetical answer to the query using the LLM (without any retrieval), then embeds that hypothetical answer and uses it for retrieval:

$$\hat{d} = \text{LLM}(\text{"Write a brief answer to: " + query})$$
$$\mathbf{v}_{\text{retrieval}} = \text{Embed}(\hat{d})$$

The hypothetical answer is passage-like in style and vocabulary, making it closer in embedding space to actual documentation passages.

**When it helps:**
- Queries are very short and keyword-sparse.
- Domain-specific terminology: the query uses lay terms but documents use technical terms.
- The embedding model has not seen many queries (trained mostly on documents).

**When it hurts or doesn't help:**
- Short latency requirements (adds an LLM call before retrieval).
- When the LLM's hypothetical answer contains confident hallucinations that steer retrieval to wrong documents.
- Well-phrased, specific queries that already embed well.

---

### Q7 (Intermediate): How do you evaluate a RAG system?

**Answer:** Evaluation must cover both components:

**Retrieval quality:**
- Recall@K: Does the truly relevant document appear in the top-K? (Most important for catching retrieval failures)
- Precision@K: Fraction of top-K that are actually relevant
- MRR (Mean Reciprocal Rank): Average of $1/\text{rank}$ of the first relevant result

**Generation quality via RAGAS:**
- Faithfulness: Are all claims in the answer supported by the retrieved context?
- Answer Relevance: Does the answer actually address the question?
- Context Precision: Are retrieved chunks relevant to the question?
- Context Recall: Does the retrieved context cover all necessary information?

**End-to-end accuracy:** For QA tasks with gold answers, compute Exact Match or F1 against ground truth.

**Human evaluation:** For production systems, have domain experts rate answer quality on (correctness, completeness, groundedness) 1-5 scales.

**LLM-as-judge:** Use GPT-4 to evaluate faithfulness and relevance at scale. Cost-effective but introduces the judge's own biases.

**Process:** Build a test set of 100-500 (question, ground-truth-answer, gold-relevant-documents) tuples. Run this regularly as a regression test when you change chunking strategy, embedding model, or retrieval parameters.

---

### Q8 (Advanced): Explain the Lost in the Middle problem and how you would mitigate it in production.

**Answer:** LLMs process context tokens left to right (and attend right to left via attention). Empirically, Liu et al. (2023) showed that models consistently perform best when relevant information is at the very beginning or end of the context window. Performance degrades substantially for information placed in the middle — a 20-30% accuracy drop for the middle positions vs. the extremes in a 20-document context.

**Why this happens:** Positional attention patterns in transformer models naturally attend more to recent (end) and initial (beginning) tokens. The middle suffers from both effects diminishing.

**Production mitigations:**

1. **Place highest-scoring documents at the boundaries:** After reranking, put rank-1 first and rank-2 last. Put ranks 3-N-1 in the middle. The most important information has the best attention.

2. **Reduce total context size:** Fewer, more relevant documents. Use aggressive reranking to cut from top-20 to top-3. Less context = less middle.

3. **Context compression:** Use LLMLingua or selective extraction to compress each document to its most relevant sentences, reducing total context length.

4. **Duplicate critical information:** For critical facts, include them twice — once at the beginning and once at the end.

5. **Map-reduce pattern:** If context is very long, process each document independently with the query and extract a concise answer, then synthesize all individual answers. This avoids a single long context entirely.

6. **Fine-tune on long-context tasks:** Models like Llama-3-70B with RoPE extension are trained specifically to handle long contexts better.

---

### Q9 (Advanced): How would you handle conflicting retrieved documents?

**Answer:** Conflicting documents arise frequently in real-world RAG when: (a) fees change over time and both old and new documents are indexed; (b) multiple sources cover the same topic with different levels of accuracy; (c) region-specific rules differ (e.g., Binance US vs Binance global fees).

**Detection:** Use an LLM to check for contradictions: "Do any of the following documents contradict each other on the topic of [query]? If so, identify the conflict."

**Resolution strategies:**

1. **Recency bias:** Tag each chunk with `last_updated`. In conflict, weight newer information more heavily. Implement in the system prompt: "If documents conflict, prefer the most recently dated document."

2. **Provenance ranking:** Not all sources are equally authoritative. An official Binance terms of service document should override a blog post. Assign source authority scores.

3. **Explicit conflict reporting:** Rather than silently resolving, tell the user: "Sources disagree. As of [date1], [source1] states X. As of [date2], [source2] states Y. The most recent source indicates Y."

4. **Multi-hop resolution:** If conflict seems resolvable (e.g., "standard fee" vs "fee with BNB discount"), present both as a complete picture rather than treating them as contradictions.

5. **Index hygiene:** The best solution is preventive. Use versioned documents, retire superseded documents from the index, and maintain `valid_from`/`valid_to` metadata.

---

### Q10 (Advanced): Design a RAG system that must answer questions across 50,000 technical documents with sub-500ms latency.

**Answer:**

**Retrieval latency budget breakdown for sub-500ms total:**
- Query embedding: 10-30ms (small model like `text-embedding-3-small`)
- ANN search: 5-50ms (HNSW on 50K docs is fast)
- Reranking (top-50 with cross-encoder): 150-300ms
- LLM generation: variable, stream to user

**Architecture choices:**

1. **Index:** HNSW in Qdrant (self-hosted). 50K documents at 768d float32 = ~150MB, fits in memory easily.

2. **Two-stage retrieval:** BM25 shortlist (top-100) in < 5ms, then dense retrieval within that shortlist. BM25 acts as a pre-filter.

3. **Reranker selection:** Use `cross-encoder/ms-marco-MiniLM-L-6-v2` (6-layer, fast) rather than a large cross-encoder. 50ms reranking time instead of 300ms.

4. **Query caching:** Cache (`query_hash` → `retrieved_chunks`) with 1-hour TTL in Redis. At 50K docs with moderate query overlap, cache hit rates of 30-50% are achievable.

5. **Embedding caching:** Cache query embeddings. If the same query arrives twice, skip the embedding call.

6. **Async generation:** Start streaming LLM tokens to the user as soon as retrieval completes. The user perceives lower latency even if total generation time is 2-3 seconds.

7. **Metadata pre-filtering:** Before any vector search, filter by document type and category using metadata. This reduces the effective search space and improves both speed and precision.

8. **Monitoring:** Track P50/P95/P99 latency per component. Alert when reranker P95 exceeds 200ms.

---

### Q11 (Intermediate): What is Self-RAG and how does it differ from standard RAG?

**Answer:** Standard RAG always retrieves, regardless of whether retrieval is needed. If you ask "What is 2+2?", standard RAG still performs a retrieval step, finds irrelevant documents, and then the LLM hopefully ignores them and answers "4."

Self-RAG (Asai et al., 2023) trains the LLM with special tokens that allow it to:
1. **Decide whether to retrieve** (`[Retrieve]` vs `[No Retrieve]`)
2. **Evaluate retrieved documents** (`[Relevant]` vs `[Irrelevant]`)
3. **Evaluate its own outputs** (`[Supported]` vs `[Partially Supported]` vs `[Contradictory]`)

The model learns these behaviors through fine-tuning on a dataset where all these decisions are annotated.

**Advantage:** Retrieval is demand-driven. For simple factual queries the model knows, no retrieval happens. For genuinely unknown facts, retrieval is triggered exactly where needed.

**Limitation:** Requires fine-tuning the model, which is expensive. The special tokens are model-specific. The model's self-assessment may be unreliable.

---

### Q12 (Advanced): How does ColBERT improve over standard bi-encoder retrieval?

**Answer:** Standard bi-encoder compresses an entire document (potentially 512 tokens) into a single vector in $\mathbb{R}^d$. This creates an information bottleneck: one vector must represent all the content in the document. For a document covering 10 different topics, the single vector is a confused average.

ColBERT keeps all per-token contextual embeddings. For a document with $|d|$ tokens, it stores $|d|$ vectors. For a query with $|q|$ tokens, it computes $|q|$ query token vectors. Relevance is computed as:

$$s(q, d) = \sum_{i=1}^{|q|} \max_{j=1}^{|d|} \mathbf{E}_i \cdot \mathbf{F}_j^T$$

For each query token, find the most similar document token. Sum those max similarities. This is the MaxSim operator.

**Why this is better:** Each query token can independently match its most relevant document token. A query about "ETH fees" has query token "fees" matching document token "gas" if they are contextually similar, and query token "ETH" matching document token "Ethereum." Neither needs to be in the same vector as the other concept.

**Cost:** Storage is $O(\text{avg\_tokens\_per\_doc} \times N)$ instead of $O(N)$ — typically 50-200x more storage. Retrieval is more complex but can be made efficient with precomputed document matrices and approximate MaxSim via PLAID (the ColBERT production system).

---

### Q13 (Intermediate): What is Maximum Marginal Relevance (MMR) and why is it useful in RAG?

**Answer:** When retrieving top-K documents by pure similarity, you often get redundant results: 5 chunks that all say the same thing about a topic, just phrased differently. This wastes context window space.

MMR is a selection algorithm that balances relevance with diversity:

$$\text{MMR} = \arg\max_{d_i \in R \setminus S} \left[\lambda \cdot \text{sim}(d_i, q) - (1-\lambda) \cdot \max_{d_j \in S} \text{sim}(d_i, d_j)\right]$$

At each step, select the document from the remaining candidates that is most similar to the query ($\text{sim}(d_i, q)$) but most different from already-selected documents ($\max_{d_j \in S} \text{sim}(d_i, d_j)$). The $\lambda$ parameter balances relevance vs diversity.

**In RAG:** Instead of retrieving 5 chunks that all explain "what is BNB," MMR ensures you get one chunk on BNB, one on BNB utility, one on BNB tokenomics, and so on — maximizing the information density in the context window.

**$\lambda$ tuning:** $\lambda = 1$ is pure relevance (no diversity penalty). $\lambda = 0$ is pure diversity. $\lambda = 0.5$ is the balanced default.

---

### Q14 (Advanced): How do you handle a multi-hop question in RAG?

**Answer:** A multi-hop question requires combining information from multiple documents that are not co-located. Example: "What is the gas fee for executing a Uniswap V3 swap on the Ethereum network when congestion is high?"

This requires knowing: (1) how Uniswap V3 swaps work gas-wise, (2) how Ethereum gas pricing works, (3) what "high congestion" means for gas prices. No single document likely covers all three.

**Approach 1: Query decomposition + sequential retrieval**

Decompose: ["What gas does Uniswap V3 swap use?", "How does Ethereum gas price increase with congestion?", "What is the total fee formula?"]

Execute sequentially: answer sub-Q1, use the answer to formulate sub-Q2, and so on. Each step's answer informs the next retrieval query.

**Approach 2: Query decomposition + parallel retrieval**

Decompose and retrieve all sub-questions simultaneously. Merge all retrieved context. Ask LLM to synthesize.

**Approach 3: Graph RAG**

Build a knowledge graph where "Uniswap V3" --uses--> "Ethereum", "Ethereum" --has--> "gas pricing mechanism", "gas pricing" --depends-on--> "network congestion". Traverse the graph from the query entity to collect all relevant facts.

**Approach 4: IRCoT (Interleaved Retrieval Chain-of-Thought)**

Interleave retrieval and reasoning: generate a reasoning step, detect when the model needs a new fact, retrieve for that fact, continue reasoning with new context.

---

### Q15 (Advanced): How do you prevent a RAG system from using the LLM's parametric knowledge instead of the retrieved context?

**Answer:** This is called parametric override or faithfulness failure. It happens when the LLM is confident in a memorized answer that conflicts with the retrieved context.

**Prompt-level mitigations:**
1. Strong system instruction: "Answer ONLY based on the provided documents. If the documents don't contain the answer, say 'I don't have this information.' Never use your own knowledge."
2. Explicit instruction to prefer context: "If your own knowledge conflicts with the provided documents, always trust the provided documents."

**Architecture-level mitigations:**
1. **Faithfulness fine-tuning:** Fine-tune the model on a dataset of (context, question, context-grounded-answer) pairs where the correct behavior is to use context even when it conflicts with training data.
2. **Contrastive decoding:** At inference time, penalize tokens that would be generated without the context (context-free generation), reward tokens that are supported by the context.
3. **Post-hoc faithfulness check:** Use NLI or RAGAS faithfulness metric to score the response. If faithfulness < threshold, regenerate with stronger instruction.

**Test cases for faithfulness:**
- Provide context with deliberately wrong facts (e.g., "Ethereum was founded in 2020" when model knows it was 2015). Does the model say 2020 (faithful) or 2015 (parametric override)?
- This is a standard faithfulness evaluation protocol.

---

## 15. Coding Problems

### Problem 1: Implement BM25 from Scratch

```python
import math
from collections import defaultdict, Counter
from typing import List, Dict

class BM25:
    """
    BM25 retrieval model implementation from scratch.

    BM25(q, d) = sum_{t in q} IDF(t) * (f(t,d) * (k1+1)) / (f(t,d) + k1*(1-b+b*|d|/avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.N: int = 0
        self.avgdl: float = 0.0
        self.doc_freqs: Dict[str, int] = defaultdict(int)  # n_t: docs containing term t
        self.doc_lengths: List[int] = []
        self.term_freqs: List[Dict[str, int]] = []  # per-doc term frequencies

    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenizer. Replace with a proper tokenizer in production."""
        return text.lower().split()

    def fit(self, corpus: List[str]) -> None:
        """Index a list of documents."""
        self.corpus = [self.tokenize(doc) for doc in corpus]
        self.N = len(self.corpus)
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avgdl = sum(self.doc_lengths) / self.N

        # Build term frequencies per document and document frequencies
        self.term_freqs = []
        self.doc_freqs = defaultdict(int)

        for doc_tokens in self.corpus:
            tf = Counter(doc_tokens)
            self.term_freqs.append(tf)
            for term in set(doc_tokens):  # unique terms to count doc frequency
                self.doc_freqs[term] += 1

    def idf(self, term: str) -> float:
        """
        IDF(t) = ln((N - n_t + 0.5) / (n_t + 0.5) + 1)
        """
        n_t = self.doc_freqs.get(term, 0)
        return math.log((self.N - n_t + 0.5) / (n_t + 0.5) + 1)

    def score(self, query: str, doc_idx: int) -> float:
        """Compute BM25 score for a single query-document pair."""
        query_terms = self.tokenize(query)
        doc_tf = self.term_freqs[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        score = 0.0
        for term in query_terms:
            f = doc_tf.get(term, 0)  # term frequency in document
            idf = self.idf(term)

            # BM25 term weight with saturation (k1) and length normalization (b)
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)

            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, top_k: int = 10) -> List[tuple]:
        """Retrieve top-K documents sorted by BM25 score."""
        scores = [(i, self.score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# Example usage
if __name__ == "__main__":
    corpus = [
        "Ethereum gas fees are paid in Gwei, a denomination of ETH",
        "Binance Smart Chain has lower gas fees than Ethereum mainnet",
        "DeFi protocols on Ethereum require gas for every transaction",
        "Bitcoin uses a different fee mechanism based on transaction size in bytes",
        "Uniswap V3 swaps require higher gas due to concentrated liquidity logic",
    ]

    bm25 = BM25(k1=1.5, b=0.75)
    bm25.fit(corpus)

    query = "Ethereum gas fees"
    results = bm25.retrieve(query, top_k=3)

    print(f"Query: '{query}'")
    for idx, score in results:
        print(f"  Score: {score:.4f} | Doc: {corpus[idx][:80]}")
```

### Problem 2: Implement Reciprocal Rank Fusion

```python
from typing import List, Dict

def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: int = 60
) -> List[tuple]:
    """
    Reciprocal Rank Fusion: combine multiple ranked lists into a single ranking.

    RRF(d) = sum_{r in R} 1 / (k + r(d))

    Args:
        ranked_lists: List of ranked document ID lists. Each inner list is
                     ordered from most to least relevant.
        k: Smoothing constant (default 60, empirically validated).

    Returns:
        List of (doc_id, rrf_score) sorted by RRF score descending.
    """
    rrf_scores: Dict[int, float] = {}

    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            # rank is 1-indexed; higher rank number = less relevant
            rrf_contribution = 1.0 / (k + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + rrf_contribution

    # Sort by RRF score descending
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def hybrid_retrieve(
    query: str,
    bm25_retriever,
    dense_retriever,
    top_k: int = 10,
    rrf_k: int = 60,
    initial_k: int = 50
) -> List[int]:
    """
    Full hybrid retrieval pipeline with RRF fusion.
    """
    # Step 1: Get ranked lists from each retriever
    bm25_results = bm25_retriever.retrieve(query, top_k=initial_k)
    dense_results = dense_retriever.retrieve(query, top_k=initial_k)

    # Extract just doc IDs (ordered by relevance)
    bm25_ids = [doc_id for doc_id, _ in bm25_results]
    dense_ids = [doc_id for doc_id, _ in dense_results]

    # Step 2: Fuse with RRF
    fused = reciprocal_rank_fusion([bm25_ids, dense_ids], k=rrf_k)

    # Step 3: Return top-K doc IDs
    return [doc_id for doc_id, _ in fused[:top_k]]


# Example
if __name__ == "__main__":
    # Simulate: doc IDs 0-9, two different systems retrieved different orderings
    bm25_ranking    = [0, 3, 7, 1, 5, 2, 8, 4, 9, 6]  # BM25 results
    dense_ranking   = [3, 0, 5, 7, 2, 1, 4, 8, 6, 9]  # Dense results

    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking], k=60)

    print("RRF Fused Ranking:")
    for doc_id, score in fused[:5]:
        print(f"  Doc {doc_id}: RRF score = {score:.6f}")

    # Doc 0: rank 1 in BM25 + rank 2 in dense = 1/(60+1) + 1/(60+2) = 0.03278
    # Doc 3: rank 2 in BM25 + rank 1 in dense = 1/(60+2) + 1/(60+1) = 0.03278
    # Both tied — sensible, both systems agree they are top docs
```

### Problem 3: Build a Full RAG Pipeline with LangChain

```python
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os

# ─── Configuration ────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5

# ─── Step 1: Load and chunk documents ────────────────────────────────────────
def load_and_chunk(file_paths: list[str]) -> list:
    """Load documents and split into chunks."""
    all_docs = []
    for fp in file_paths:
        if fp.endswith(".pdf"):
            loader = PyPDFLoader(fp)
        else:
            loader = TextLoader(fp)
        all_docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        add_start_index=True,  # Track character position in original document
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Created {len(chunks)} chunks from {len(all_docs)} documents")
    return chunks

# ─── Step 2: Build hybrid retriever ──────────────────────────────────────────
def build_hybrid_retriever(chunks):
    """Build an ensemble retriever combining BM25 and dense retrieval."""

    # Dense retriever
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="crypto_docs"
    )
    dense_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K * 3}  # Retrieve more for RRF fusion
    )

    # Sparse retriever (BM25)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = TOP_K * 3

    # Ensemble with equal weights (internally uses RRF)
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retriever

# ─── Step 3: RAG prompt template ─────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are a helpful assistant for Binance users.
Answer questions using ONLY the information from the provided context documents.
If the context does not contain enough information to answer, say "I don't have
enough information to answer this question."

Context documents:
{context}

Question: {question}

Instructions:
- Cite specific documents when making claims (e.g., "According to Document 1...")
- If documents conflict, note the conflict explicitly
- Be concise and factual

Answer:"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"]
)

# ─── Step 4: Build RAG chain ──────────────────────────────────────────────────
def build_rag_chain(retriever):
    """Build a complete RAG chain."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.0,  # Deterministic for factual queries
        max_tokens=1024
    )

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Stuff all retrieved docs into context
        retriever=retriever,
        return_source_documents=True,  # Return sources for attribution
        chain_type_kwargs={"prompt": RAG_PROMPT}
    )

    return rag_chain

# ─── Step 5: Query with attribution ──────────────────────────────────────────
def query_with_attribution(rag_chain, question: str) -> dict:
    """Run a query and return answer with source attribution."""
    result = rag_chain({"query": question})

    answer = result["result"]
    sources = result["source_documents"]

    # Build citation list
    citations = []
    for i, doc in enumerate(sources):
        citations.append({
            "index": i + 1,
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", "N/A"),
            "preview": doc.page_content[:100] + "..."
        })

    return {
        "question": question,
        "answer": answer,
        "citations": citations,
        "num_retrieved": len(sources)
    }

# ─── Main pipeline ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Load documents
    chunks = load_and_chunk(["binance_help_center.txt", "crypto_regulations.pdf"])

    # Build retriever
    retriever = build_hybrid_retriever(chunks)

    # Build RAG chain
    rag = build_rag_chain(retriever)

    # Query
    result = query_with_attribution(rag, "What are Binance's trading fees?")

    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"\nSources ({result['num_retrieved']} retrieved):")
    for cite in result['citations']:
        print(f"  [{cite['index']}] {cite['source']} (page {cite['page']})")
        print(f"      Preview: {cite['preview']}")
```

### Problem 4: Implement RAGAS Faithfulness Metric

```python
from openai import OpenAI
import json
from typing import List, Dict

client = OpenAI()

def extract_claims(answer: str) -> List[str]:
    """
    Use LLM to extract atomic factual claims from an answer.
    Each claim should be a standalone, verifiable statement.
    """
    prompt = f"""Extract all atomic factual claims from the following answer.
    Each claim should be a single, standalone, verifiable statement.
    Return as a JSON array of strings.

    Answer: {answer}

    Claims (JSON array):"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    # Handle different JSON structures the model might return
    if isinstance(result, dict):
        claims = result.get("claims", list(result.values())[0])
    else:
        claims = result
    return claims


def verify_claim_against_context(claim: str, context: str) -> bool:
    """
    Use LLM as NLI model to verify if a claim is supported by the context.
    Returns True if supported, False if not supported or contradicted.
    """
    prompt = f"""Given the following context, determine whether the claim is
    supported by the context.

    Context: {context}

    Claim: {claim}

    Answer with only "SUPPORTED" if the claim is directly supported by the context,
    or "NOT SUPPORTED" if the claim is not supported by or contradicts the context.

    Verdict:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=10
    )

    verdict = response.choices[0].message.content.strip().upper()
    return "SUPPORTED" in verdict


def faithfulness_score(
    answer: str,
    context: str,
    verbose: bool = False
) -> Dict:
    """
    Compute RAGAS faithfulness score.

    Faithfulness = (# claims supported by context) / (total # claims)

    Args:
        answer: The generated answer to evaluate
        context: The retrieved context used to generate the answer
        verbose: If True, print per-claim verdicts

    Returns:
        Dict with score and per-claim details
    """
    # Step 1: Extract claims from answer
    claims = extract_claims(answer)

    if not claims:
        return {"score": 1.0, "claims": [], "supported": 0, "total": 0}

    # Step 2: Verify each claim against context
    results = []
    supported_count = 0

    for claim in claims:
        is_supported = verify_claim_against_context(claim, context)
        results.append({
            "claim": claim,
            "supported": is_supported
        })
        if is_supported:
            supported_count += 1

        if verbose:
            status = "SUPPORTED" if is_supported else "NOT SUPPORTED"
            print(f"  [{status}] {claim}")

    # Step 3: Compute score
    score = supported_count / len(claims)

    return {
        "score": score,
        "claims": results,
        "supported": supported_count,
        "total": len(claims)
    }


def answer_relevance_score(answer: str, question: str, n_questions: int = 3) -> float:
    """
    Compute RAGAS answer relevance score.

    Generate n_questions from the answer, measure cosine similarity to original question.
    High similarity = answer is relevant to the question.
    """
    import numpy as np

    # Generate questions that the answer would answer
    prompt = f"""Given the following answer, generate {n_questions} different questions
    that this answer could be responding to. Return as a JSON array.

    Answer: {answer}

    Questions (JSON array):"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"}
    )

    result = json.loads(response.choices[0].message.content)
    generated_questions = result.get("questions", list(result.values())[0])

    # Embed original question and generated questions
    def embed(text: str) -> np.ndarray:
        resp = client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(resp.data[0].embedding)

    original_emb = embed(question)

    similarities = []
    for gen_q in generated_questions:
        gen_emb = embed(gen_q)
        cos_sim = np.dot(original_emb, gen_emb) / (
            np.linalg.norm(original_emb) * np.linalg.norm(gen_emb)
        )
        similarities.append(float(cos_sim))

    return sum(similarities) / len(similarities)


# ─── Full RAGAS evaluation example ────────────────────────────────────────────
if __name__ == "__main__":
    question = "What are Binance trading fees?"

    context = """
    Binance charges a standard trading fee of 0.1% for spot trading.
    Users who hold BNB (Binance Coin) and choose to pay fees with BNB
    receive a 25% discount, reducing the effective fee to 0.075%.
    VIP tiers with higher 30-day trading volumes receive further discounts,
    with VIP 9 users paying as low as 0.02% maker and 0.04% taker fees.
    """

    answer = """
    Binance's standard spot trading fee is 0.1%. If you pay with BNB,
    you get a 25% discount bringing it to 0.075%. High-volume traders
    in VIP tiers can get fees as low as 0.02%. Binance also offers
    zero-fee Bitcoin trading on select pairs.
    """
    # Note: last sentence about zero-fee Bitcoin trading is potentially not in context

    print("=== RAGAS Faithfulness Evaluation ===")
    faith_result = faithfulness_score(answer, context, verbose=True)
    print(f"\nFaithfulness Score: {faith_result['score']:.3f}")
    print(f"Supported: {faith_result['supported']}/{faith_result['total']} claims")

    print("\n=== RAGAS Answer Relevance Evaluation ===")
    relevance = answer_relevance_score(answer, question)
    print(f"Answer Relevance Score: {relevance:.3f}")
```

---

## Quick Reference Summary

### RAG Pipeline Stages and Key Decisions

| Stage | Key Decision | Default / Recommended |
|---|---|---|
| Chunking | Chunk size | 500 tokens with 50 overlap (RecursiveCharacterSplitter) |
| Embedding | Model | `text-embedding-3-small` (cost) or `bge-large-en` (quality) |
| Indexing | ANN algorithm | HNSW (quality) or IVF-PQ (memory efficiency) |
| Retrieval | Method | Hybrid BM25 + dense with RRF |
| Reranking | Model | `cross-encoder/ms-marco-MiniLM-L-6-v2` (speed) or Cohere Rerank (quality) |
| Context | Management | MMR for diversity, LLMLingua for compression |
| Generation | Temperature | 0.0 for factual, 0.3 for creative |
| Evaluation | Framework | RAGAS (faithfulness + answer relevance + context precision) |

### Key Equations to Remember

BM25 score: $\text{BM25}(q,d) = \sum_{t \in q} \ln\frac{N-n_t+0.5}{n_t+0.5} \cdot \frac{f(t,d)(k_1+1)}{f(t,d)+k_1(1-b+b\frac{|d|}{\text{avgdl}})}$

RRF fusion: $\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$ where $k=60$

ColBERT MaxSim: $s(q,d) = \sum_{i \in q} \max_{j \in d} \mathbf{E}_i \cdot \mathbf{F}_j^T$

MMR diversity: $\text{MMR} = \arg\max_{d_i \notin S}\left[\lambda \cdot \text{sim}(d_i, q) - (1-\lambda)\max_{d_j \in S}\text{sim}(d_i,d_j)\right]$

RAGAS faithfulness: $F = \frac{|\text{supported claims}|}{|\text{total claims}|}$

---

*End of RAG and Retrieval Systems Technical Guide*
