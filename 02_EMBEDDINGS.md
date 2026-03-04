# Embeddings — Comprehensive Technical Guide
### Interview Preparation | ST5230 | Binance Data Science Role

---

## Table of Contents

1. [What Are Embeddings?](#1-what-are-embeddings)
2. [TF-IDF and BM25 (Baselines)](#2-tf-idf-and-bm25-as-baseline)
3. [Word2Vec](#3-word2vec)
4. [Contextual Embeddings — BERT](#4-contextual-embeddings--bert)
5. [Sentence-BERT (SBERT)](#5-sentence-bert-sbert--in-depth)
6. [Modern Embedding Models](#6-modern-embedding-models)
7. [Multi-Modal Embeddings (CLIP)](#7-multi-modal-embeddings-clip)
8. [Embedding Space Analysis](#8-embedding-space-analysis)
9. [Vector Similarity Search & Indexing](#9-vector-similarity-search--indexing)
10. [Problems & Mitigations](#10-problems--mitigations)
11. [Industry Practices](#11-industry-practices)
12. [Interview Q&A](#12-interview-qa)
13. [Coding Problems](#13-coding-problems)

---

## 1. What Are Embeddings?

### 1.1 Core Intuition: Discrete to Continuous Mapping

An **embedding** is a learned mapping from a discrete, symbolic input space (words, tokens, users, items, images) into a continuous, dense vector space $\mathbb{R}^d$. The fundamental insight is that **geometry encodes semantics**: things that are similar should be close together in the embedding space.

Formally, an embedding function is:

$$E: \mathcal{V} \rightarrow \mathbb{R}^d$$

where $\mathcal{V}$ is a discrete vocabulary of size $|\mathcal{V}|$ and $d \ll |\mathcal{V}|$. For example, a vocabulary of 50,000 words might be embedded into $\mathbb{R}^{768}$ or $\mathbb{R}^{1536}$.

The power of embeddings lies in what they capture:
- **Semantic similarity**: $\text{cosine}(E(\text{"bitcoin"}), E(\text{"cryptocurrency"})) \approx 1$
- **Analogical relationships**: $E(\text{"king"}) - E(\text{"man"}) + E(\text{"woman"}) \approx E(\text{"queen"})$
- **Topical clustering**: financial terms cluster together, technical blockchain terms cluster separately

### 1.2 Why Not One-Hot Encodings?

Before embeddings, the standard representation was one-hot encoding: each word $w_i$ maps to a binary vector of length $|\mathcal{V}|$ with a single 1 at position $i$.

**Problem 1: Curse of Dimensionality.** With $|\mathcal{V}| = 50{,}000$, every vector lives in a 50,000-dimensional space. Distances lose meaning in very high dimensions (concentration of measure phenomenon). The ratio of the maximum to minimum pairwise distance approaches 1 as $d \rightarrow \infty$.

**Problem 2: No Semantic Information.** One-hot vectors are completely orthogonal:

$$\langle e_i, e_j \rangle = \delta_{ij}$$

The words "bitcoin" and "ethereum" are exactly as distant from each other as "bitcoin" and "potato". The representation carries zero information about semantic relatedness.

**Problem 3: Orthogonality Problem.** Any downstream model operating on one-hot inputs must learn all semantic relationships from scratch using labeled data. Embeddings allow this knowledge to be pre-learned from massive unlabeled corpora and transferred.

**Problem 4: Memory and Computation.** A lookup table with $|\mathcal{V}| = 50{,}000$ and batch size 256 produces a $256 \times 50{,}000$ sparse matrix. Dense embeddings of dimension 768 produce a $256 \times 768$ dense matrix — far more tractable.

### 1.3 What Makes a Good Embedding Space?

**Isotropy.** A good embedding space distributes vectors uniformly across directions. An isotropic space has no preferred axis; the average cosine similarity between random pairs of embeddings approaches 0. Formally, an embedding distribution is isotropic if for any unit vector $u$:

$$\mathbb{E}_{x \sim p}[(u^\top E(x))^2] = \frac{\|E(x)\|^2}{d}$$

Anisotropic spaces (like raw BERT) have embeddings concentrated in a narrow cone, degrading cosine similarity as a discriminative metric.

**Meaningful Distances.** The distance metric used downstream must correlate with semantic distance. If $d(E(x), E(y)) < d(E(x), E(z))$ then $x$ should be more semantically similar to $y$ than to $z$.

**Compositionality.** The embedding space should support arithmetic: $E(\text{"not good"})$ should be somewhere near $E(\text{"bad"})$. This is approximately true for Word2Vec but breaks down for static embeddings on complex phrases.

**Linearity.** Linear transformations in embedding space should correspond to semantic transformations. This is the foundation of the analogy property.

### 1.4 Sparse vs Dense Embeddings

| Property | Sparse (TF-IDF / BM25) | Dense (Neural) |
|---|---|---|
| Representation | High-dimensional, mostly zeros | Low-dimensional, all non-zero |
| Vocabulary handling | Exact term match | Semantic generalization |
| OOV handling | Drops terms | Subword tokenization |
| Interpretability | High (term weights) | Low (distributed) |
| Training data needed | None (unsupervised) | Large corpus or labeled pairs |
| Latency | Very fast (inverted index) | Slower (ANN search) |
| Best for | Keyword-heavy queries | Semantic/paraphrase queries |

**Hybrid search** combines both: retrieve candidates with BM25 (recall-oriented) and rerank with dense embeddings (precision-oriented). Reciprocal Rank Fusion (RRF) is a common fusion strategy (see Section 11).

---

## 2. TF-IDF and BM25 (as Baseline)

### 2.1 TF-IDF: Full Derivation

**Term Frequency (TF):** How often does term $t$ appear in document $d$?

$$\text{tf}(t, d) = \frac{f_{t,d}}{\sum_{t' \in d} f_{t',d}}$$

where $f_{t,d}$ is the raw count of $t$ in $d$. Variants include log-normalized TF: $1 + \log(f_{t,d})$.

**Inverse Document Frequency (IDF):** How rare is term $t$ across the corpus?

$$\text{idf}(t) = \log\frac{N}{\text{df}(t)}$$

where $N$ is the total number of documents and $\text{df}(t)$ is the number of documents containing $t$. Common terms like "the" get low IDF; rare but informative terms like "Merkle" get high IDF.

**TF-IDF score:**

$$\text{tfidf}(t, d) = \text{tf}(t, d) \cdot \log\frac{N}{\text{df}(t)}$$

**Intuition from Information Theory.** IDF is related to surprisal. If a term appears in all $N$ documents, $\log(N/N) = 0$: it carries no discriminative information. If a term appears in only 1 document, $\log(N/1) = \log N$: it is maximally discriminative.

**Problems & Mitigations for TF-IDF:**
- TF grows linearly with term frequency — a document mentioning "bitcoin" 100 times gets 10x the score of one mentioning it 10 times, even if the latter is more relevant. **Mitigation**: Use BM25's saturated TF.
- No document length normalization: long documents are artificially favored. **Mitigation**: BM25's $b$ parameter.
- Bag-of-words: ignores word order and context. **Mitigation**: Use dense embeddings for semantic matching.

### 2.2 BM25: Okapi BM25 Full Formula

BM25 (Best Match 25) addresses TF-IDF's shortcomings with **saturation** and **length normalization**:

$$\text{BM25}(t, d) = \text{idf}(t) \cdot \frac{f_{t,d} \cdot (k_1 + 1)}{f_{t,d} + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

For a query $Q = \{t_1, t_2, \ldots, t_n\}$ against document $d$:

$$\text{BM25}(Q, d) = \sum_{i=1}^{n} \text{idf}(t_i) \cdot \frac{f_{t_i, d} \cdot (k_1 + 1)}{f_{t_i, d} + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$

**Parameters:**
- $k_1 \in [1.2, 2.0]$: **saturation parameter**. Controls how much TF contributes. As $f_{t,d} \rightarrow \infty$, the TF factor saturates to $k_1 + 1$ rather than growing unboundedly.
- $b \in [0, 1]$: **length normalization**. $b = 0$ means no length normalization; $b = 1$ means full normalization. Default $b = 0.75$.
- $\text{avgdl}$: average document length in the corpus.

**Saturation intuition.** With $k_1 = 1.5$, the TF factor for a term appearing once is $\frac{1 \cdot 2.5}{1 + 1.5} = 1.0$. For 10 occurrences: $\frac{10 \cdot 2.5}{10 + 1.5} = 2.17$. For 100 occurrences: $\frac{100 \cdot 2.5}{100 + 1.5} = 2.46$. The gain from additional occurrences rapidly diminishes — a realistic model of relevance.

**IDF in BM25** (Robertson-Sparck Jones variant):

$$\text{idf}(t) = \log\frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}$$

This is smoother than the classic log formula and handles corner cases.

### 2.3 Why BM25 Remains Competitive

Despite being from the 1990s, BM25 persists in production because:
1. **Exact keyword matching**: crucial when users type exact product codes, contract addresses, or ticker symbols.
2. **Zero latency for indexing**: inverted index lookup is O(1) per term.
3. **No embedding model needed**: works on any text without training.
4. **Strong recall**: BM25 as a first-stage retriever typically has 95%+ recall at 100, competitive with dense retrievers at 1000.
5. **Interpretability**: you can explain why a document was retrieved (term weights).

In modern production systems (Elasticsearch, Solr, OpenSearch), BM25 is the default ranking function and serves as the first-stage retriever in multi-stage pipelines.

---

## 3. Word2Vec

Word2Vec (Mikolov et al., 2013) learns word embeddings by training a shallow neural network to predict words from context. It comes in two architectures: Skip-gram and CBOW.

### 3.1 Skip-gram: Full Objective Derivation

**Setup.** Given a corpus, define a window of size $c$. For each center word $w_t$, predict context words $w_{t-c}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+c}$.

**Full log-likelihood objective** to maximize:

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^{T} \sum_{\substack{-c \leq j \leq c \\ j \neq 0}} \log P(w_{t+j} \mid w_t)$$

**Prediction probability** using softmax over the full vocabulary:

$$P(w_O \mid w_I) = \frac{\exp(v_{w_O}'^{\top} v_{w_I})}{\sum_{w=1}^{W} \exp(v_w'^{\top} v_{w_I})}$$

where $v_{w_I} \in \mathbb{R}^d$ is the **input embedding** of the center word and $v_{w_O}' \in \mathbb{R}^d$ is the **output embedding** of the context word. Each word has two vectors: one when it appears as center, one when it appears as context.

**Problem: the denominator is intractable.** Computing $\sum_{w=1}^{W} \exp(v_w'^{\top} v_{w_I})$ requires a pass over the entire vocabulary for every training step. With $W = 50{,}000$ and billions of training examples, this is computationally infeasible.

### 3.2 Negative Sampling: Surrogate Objective

Instead of the softmax, use a **binary classification objective**: distinguish the true context word (positive) from $k$ randomly sampled "noise" words (negatives).

**Surrogate objective:**

$$\mathcal{L}_{\text{neg}} = \log \sigma(v_{w_O}'^{\top} v_{w_I}) + \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n(w)} \left[\log \sigma(-v_{w_i}'^{\top} v_{w_I})\right]$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function and $P_n(w)$ is the noise distribution. For each positive pair $(w_I, w_O)$, we push $v_{w_O}'^{\top} v_{w_I}$ high (push their dot product positive) and push $k$ random $v_{w_i}'^{\top} v_{w_I}$ low.

**The $\frac{3}{4}$ power for the noise distribution.** Mikolov et al. found empirically that sampling from:

$$P_n(w) \propto \text{count}(w)^{3/4}$$

outperforms both uniform sampling ($\propto 1$) and unigram sampling ($\propto \text{count}(w)$). Why $3/4$?

- Unigram sampling ($\alpha = 1$): extremely common words ("the", "a", "is") dominate as negatives, providing poor learning signal since the model already knows they are unlikely context words for most center words.
- Uniform sampling ($\alpha = 0$): rare words sampled too often, also poor signal.
- $\alpha = 3/4$ **smooths the distribution**: common words are sampled proportionally less often than their frequency warrants, and rare words more often. This balances the learning signal.

Mathematically, for word $w$ with count $c_w$: $P_n(w) = \frac{c_w^{3/4}}{\sum_{w'} c_{w'}^{3/4}}$. The value $3/4$ was chosen empirically, not derived theoretically.

**Typical values**: $k = 5$ for large corpora, $k = 15$ for small corpora.

### 3.3 CBOW (Continuous Bag of Words)

CBOW predicts the center word from the average of context word embeddings:

```
Context words:  [w_{t-2}, w_{t-1}, w_{t+1}, w_{t+2}]
                      |         |        |        |
                   embed     embed    embed    embed
                      \         |        |       /
                       ------[average]------
                                |
                            [linear]
                                |
                           [softmax]
                                |
                          predict w_t
```

$$P(w_t \mid \text{ctx}) = \frac{\exp(v_{w_t}'^{\top} \bar{v})}{\sum_w \exp(v_w'^{\top} \bar{v})}, \quad \bar{v} = \frac{1}{2c}\sum_{\substack{-c \leq j \leq c \\ j \neq 0}} v_{w_{t+j}}$$

**CBOW vs Skip-gram:**
- CBOW trains faster (one prediction per window vs $2c$ predictions for skip-gram).
- CBOW is better for frequent words (averaging smooths noise).
- Skip-gram is better for rare words (each occurrence of a rare word directly affects its embedding).
- In practice, skip-gram with negative sampling (SGNS) produces slightly better embeddings for NLP tasks.

### 3.4 The Analogy Property and PMI Connection

**The analogy $v(\text{king}) - v(\text{man}) + v(\text{woman}) \approx v(\text{queen})$.**

This is not a designed feature — it emerges from training. The geometric explanation:

$$v(\text{king}) - v(\text{man}) \approx v(\text{queen}) - v(\text{woman})$$

This vector encodes the "royalty + female = queen" relationship. The direction $v(\text{king}) - v(\text{man})$ roughly corresponds to the "royalty" direction in embedding space.

**Connection to PMI Matrix Factorization.** Levy and Goldberg (2014) showed that skip-gram with negative sampling implicitly factorizes a shifted PMI matrix.

Define the Pointwise Mutual Information between word $w$ and context $c$:

$$\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w) P(c)} = \log \frac{\#(w,c) \cdot |D|}{\#(w) \cdot \#(c)}$$

SGNS implicitly factorizes:

$$M_{wc} = \text{PMI}(w, c) - \log k$$

where $k$ is the number of negative samples. This is the **Shifted PMI (SPMI)** matrix. The embeddings $v_w$ and $v_c'$ approximate this factorization: $v_w^{\top} v_c' \approx \text{SPMI}(w, c)$.

**Why does the analogy work given PMI?** If two word pairs $(w_1, w_2)$ and $(w_3, w_4)$ have similar co-occurrence patterns relative to their marginals, their PMI vectors will be similar. The difference $v(\text{king}) - v(\text{man})$ cancels out the "common word" directions, leaving behind the "royalty" direction.

### 3.5 GloVe: Global Vectors

GloVe (Pennington et al., 2014) makes the PMI factorization explicit. Instead of predicting context words locally, GloVe minimizes a weighted reconstruction of the **global co-occurrence matrix** $X$ where $X_{ij}$ = number of times word $j$ appears in context of word $i$.

**GloVe Objective:**

$$\mathcal{L} = \sum_{i,j=1}^{V} f(X_{ij}) \left(v_i^{\top} \tilde{v}_j + b_i + \tilde{b}_j - \log X_{ij}\right)^2$$

where:
- $v_i, \tilde{v}_j \in \mathbb{R}^d$: word and context vectors
- $b_i, \tilde{b}_j$: bias terms
- $f(x)$: weighting function

**The weighting function $f(x)$:**

$$f(x) = \begin{cases} (x / x_{\max})^\alpha & \text{if } x < x_{\max} \\ 1 & \text{otherwise} \end{cases}$$

with $x_{\max} = 100$ and $\alpha = 3/4$ (again!) as default values. This weights rare co-occurrences less (they are noisy) and caps the weight for very common co-occurrences (to prevent "the", "a" from dominating).

**Key insight**: GloVe directly models $\log X_{ij}$, which is approximately $\text{PMI}(i, j) + \log(\text{total co-occurrences})$. The model learns $v_i^{\top} \tilde{v}_j \approx \log X_{ij}$.

### 3.6 Problems & Mitigations for Word2Vec/GloVe

**Static Embeddings (Polysemy).** "Bank" has one vector regardless of context. Financial bank and river bank are conflated. This is the fundamental limitation of context-free embeddings. **Mitigation**: Use contextual embeddings (BERT, ELMo).

**No OOV Handling.** Words not seen during training have no embedding. New ticker symbols, named entities, or technical jargon are unseen. **Mitigation**: FastText character n-grams, subword tokenization.

**Requires Large Corpus.** Word2Vec needs billions of tokens to produce good embeddings. Rare words (frequency < 5) are typically discarded. **Mitigation**: Use pretrained models (GloVe 840B, fastText CC).

**Frequency Bias.** Common words dominate training; embeddings of rare words are unreliable. **Mitigation**: Subsampling of frequent words during training (Word2Vec discards high-frequency tokens with probability proportional to their frequency).

---

## 4. Contextual Embeddings — BERT

BERT (Devlin et al., 2018) produces **contextual embeddings**: the vector for a word depends on all words in its sentence. "Bank" in "bank transfer" gets a different vector than in "river bank".

### 4.1 How BERT Produces Contextual Embeddings

BERT is a transformer encoder pretrained with Masked Language Modeling (MLM) and Next Sentence Prediction (NSP). Given input sequence $[x_1, x_2, \ldots, x_n]$, the final hidden states $[h_1, h_2, \ldots, h_n] \in \mathbb{R}^d$ are contextual embeddings — each $h_i$ is a function of the entire sequence via self-attention.

**Three strategies for sentence-level embeddings:**

**[CLS] Token Pooling.** BERT prepends a special `[CLS]` token. After all transformer layers, the hidden state of `[CLS]` is intended to aggregate sequence-level information:

$$e_{\text{sentence}} = h_{[CLS]}$$

This is what BERT's classification head uses. However, for similarity tasks, raw `[CLS]` embeddings perform poorly (see anisotropy below).

**Mean Pooling.** Average all token embeddings (excluding padding):

$$e_{\text{sentence}} = \frac{1}{n} \sum_{i=1}^{n} h_i \odot m_i$$

where $m_i \in \{0, 1\}$ is the attention mask. Mean pooling treats all tokens equally.

**Max Pooling.** Take the element-wise maximum across all token embeddings:

$$e_{\text{sentence}}[j] = \max_{i=1}^{n} h_i[j]$$

This captures the most salient feature in each dimension.

**Comparison.** For semantic similarity tasks without fine-tuning:
- Mean pooling > Max pooling > [CLS] pooling
- None of them work well without fine-tuning (Reimers & Gurevych, 2019)

### 4.2 The Anisotropy Problem

**Definition.** BERT's embedding space is highly anisotropic: most token vectors point in a small cone, occupying a tiny region of $\mathbb{R}^{768}$.

**Formal measurement.** The average cosine similarity between random BERT embeddings is:

$$\bar{s} = \frac{1}{|\mathcal{V}|^2} \sum_{i \neq j} \cos(e_i, e_j)$$

For a truly isotropic space, $\bar{s} \approx 0$. For raw BERT (bert-base-uncased), $\bar{s} \approx 0.6$ — random embeddings have high similarity on average! This means cosine similarity is a poor discriminator.

**Why does anisotropy occur?**

1. **Frequency bias**: High-frequency tokens (punctuation, "the", "of") dominate the embedding space. Because they appear everywhere, their embeddings are pulled toward the origin of the cone.
2. **Self-attention geometry**: The dot-product attention mechanism encourages embeddings to have large norms for attended tokens and small norms otherwise, breaking isotropy.
3. **MLM objective**: The model optimizes for predicting masked tokens, not for producing geometrically spread representations. The softmax over the full vocabulary implicitly creates high-entropy distributions, and Arora et al. showed this leads to a low-dimensional "rogue dimension" that dominates variance.

**Why BERT Can Be Worse Than GloVe for Sentence Similarity.** This is counterintuitive! Reimers & Gurevych (2019) showed that on STS (Semantic Textual Similarity) benchmarks:
- GloVe average: Spearman $\rho \approx 0.36$
- BERT CLS pooling: Spearman $\rho \approx 0.20$
- BERT mean pooling: Spearman $\rho \approx 0.54$

BERT CLS is *worse* than averaging random GloVe vectors! Mean pooling helps but still doesn't beat SBERT. The reason: BERT was never trained to produce geometry-preserving sentence embeddings — it was trained for classification via a task-specific head.

### 4.3 Whitening as a Fix

**Whitening** transforms embeddings to have zero mean and identity covariance:

$$\tilde{e} = W(e - \mu)$$

where $\mu$ is the empirical mean of all embeddings and $W$ is derived from the SVD of the covariance matrix $\Sigma = U \Lambda U^\top$:

$$W = \Lambda^{-1/2} U^\top$$

This standardizes each principal component to have variance 1, spreading embeddings more uniformly. Su et al. (2021) showed whitening closes much of the gap between raw BERT and SBERT for STS tasks.

**Limitations**: whitening requires computing covariance over a large sample, doesn't generalize to new domains without recomputation, and still doesn't fully solve the alignment problem.

### 4.4 Problems & Mitigations for Raw BERT Embeddings

**Anisotropy.** All embeddings occupy a narrow cone. **Mitigation**: Whitening, SBERT fine-tuning, contrastive post-training.

**Task mismatch.** MLM objective does not align with similarity geometry. **Mitigation**: Fine-tune on similarity data with appropriate loss.

**CLS token unreliability.** CLS was trained for NSP (next sentence prediction), a weak task. **Mitigation**: Use mean pooling, or better yet, fine-tune with SBERT.

---

## 5. Sentence-BERT (SBERT) — In Depth

Sentence-BERT (Reimers & Gurevych, 2019) fine-tunes BERT with a **Siamese network** architecture to produce semantically meaningful sentence embeddings directly optimizable for similarity.

### 5.1 Siamese Network Architecture

```
Sentence A                    Sentence B
    |                              |
[BERT Encoder]            [BERT Encoder]
(shared weights)          (shared weights)
    |                              |
[Mean Pooling]            [Mean Pooling]
    |                              |
    u (768-dim)                v (768-dim)
    |                              |
    |----------[concat]------------|
               [u ; v ; |u-v|]
               (2304-dim)
                    |
               [Linear]
                    |
               [Softmax]
               (3 classes: entail/neutral/contra)
```

**Why Siamese?** Parameter sharing between the two towers ensures that:
1. The same sentence always gets the same embedding regardless of which input slot it's in.
2. Contrastive learning is natural: similar inputs should produce similar hidden representations.
3. Training is sample-efficient: a single pair $(A, B)$ trains both towers simultaneously.

### 5.2 Pooling Strategies

SBERT's original paper found **mean pooling** over all token outputs performs best for most tasks. Empirically across the STS benchmark:
- Mean pooling: Spearman $\rho \approx 0.869$
- [CLS] pooling: Spearman $\rho \approx 0.853$
- Max pooling: Spearman $\rho \approx 0.860$

Mean pooling is the de facto standard for modern bi-encoders.

### 5.3 Training Objectives by Data Type

#### NLI Data: Softmax Classification Loss

With Natural Language Inference data (entailment/neutral/contradiction labels), SBERT trains a classifier on the concatenation $[u; v; |u-v|]$.

**Why element-wise difference matters.** The vector $|u-v|$ encodes **how different** $u$ and $v$ are dimension by dimension. The model can learn:
- Dimensions where $u$ and $v$ agree: close to zero, evidence for entailment
- Dimensions where they disagree sharply: large magnitude, evidence for contradiction

Without $|u-v|$, the classifier only sees the concatenated embeddings and must learn distance implicitly. Including $|u-v|$ makes the comparison explicit and dramatically improves classification accuracy.

The combined feature has dimension $d + d + d = 3d = 2304$ for BERT-base.

Loss: standard cross-entropy over 3 classes.

#### STS Data: Cosine Similarity MSE Loss

With Semantic Textual Similarity data (human-annotated similarity scores in $[0, 1]$ or $[-1, 1]$):

$$\mathcal{L}_{\text{STS}} = \frac{1}{N}\sum_{i=1}^{N} \left(\cos(u_i, v_i) - y_i\right)^2$$

This directly trains the model to produce embeddings whose cosine similarity matches human judgments. Simple but effective when ground-truth scores are available.

#### Retrieval Data: Multiple Negatives Ranking Loss (MNRL)

MNRL is the most powerful training signal for retrieval tasks. Given a batch of $N$ positive pairs $(q_i, p_i)$ for $i = 1, \ldots, N$:

**Treat all other positives in the batch as negatives** for each query $q_i$. For query $q_i$:
- Positive: $p_i$
- In-batch negatives: $\{p_j : j \neq i\}$

**MNRL Loss (InfoNCE-style):**

$$\mathcal{L}_{\text{MNRL}} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(q_i, p_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(q_i, p_j) / \tau)}$$

where $\text{sim}(u, v) = \frac{u^\top v}{\|u\| \|v\|}$ (cosine similarity) and $\tau$ is a temperature parameter.

**Key properties:**
- Each batch provides $O(N^2)$ training signal from $N$ pairs.
- Large batches are crucial: with $N = 256$, each query has 255 negatives.
- Hard negatives in the batch (semantically similar but not the paired positive) provide the strongest signal.
- Temperature $\tau$ controls the "peakiness" of the distribution: low $\tau$ makes the model more discriminative but can cause instability.

### 5.4 Cross-Encoder vs Bi-Encoder Speed Comparison

**Cross-encoder** feeds both sentences jointly into BERT with full self-attention between them:

```
[CLS] Sentence A [SEP] Sentence B [SEP]
              |
          BERT (all layers, full attention between A and B)
              |
         [CLS] hidden state
              |
         [Linear] -> score
```

**Bi-encoder** (SBERT) encodes each sentence independently:

```
Sentence A -> BERT -> u      Sentence B -> BERT -> v
                      \           /
                       cos(u, v)
```

**Concrete speed numbers** for 10,000 sentence corpus:

| Method | Time to rank 10,000 pairs | Precomputation |
|---|---|---|
| Cross-encoder | ~650 seconds (65ms/pair) | None |
| Bi-encoder | ~0.01 seconds + precompute | ~0.32 seconds |

Cross-encoder takes 65ms per pair, so 10,000 pairs takes 10.8 minutes. Bi-encoder encodes each sentence once, then does 10,000 dot products at ~1 microsecond each, making it effectively real-time.

**Quality vs speed tradeoff**: Cross-encoders consistently outperform bi-encoders on similarity tasks by 3-8 Spearman points. Production systems use a **two-stage pipeline**: bi-encoder retrieves top-100 candidates, cross-encoder reranks to top-10.

### 5.5 Problems & Mitigations

**Quality gap vs cross-encoder.** Bi-encoders cannot do full pair interaction — each sentence is encoded in isolation. **Mitigation**: (1) knowledge distillation from cross-encoder to bi-encoder teacher-student setup, (2) iterative hard negative mining where the bi-encoder itself generates hard negatives for retraining.

**Domain mismatch.** A model trained on Wikipedia + NLI performs poorly on financial texts. **Mitigation**: domain-adaptive pretraining on unlabeled domain text, then fine-tuning.

**Catastrophic forgetting.** Fine-tuning on new data causes forgetting of previous capability. **Mitigation**: replay buffers, elastic weight consolidation (EWC), or LoRA fine-tuning.

---

## 6. Modern Embedding Models

### 6.1 E5: Text Embeddings from Weakly Supervised Contrastive Pre-training

E5 (Wang et al., 2022) introduced **instruction-based training** and a multi-stage training approach.

**Key innovation — query/passage prefixes.** E5 prepends "query: " to queries and "passage: " to documents during both training and inference. This asymmetric treatment allows the model to learn different representations for queries (short, keyword-like) and passages (long, information-rich).

```
Input:  "query: What is Bitcoin halving?"
Output: [1536-dim embedding optimized for query matching]

Input:  "passage: Bitcoin halving is the event where..."
Output: [1536-dim embedding optimized for passage retrieval]
```

**Training stages:**
1. **Weak supervision at scale**: Mine (query, passage) pairs from Common Crawl using heuristics (titles as queries, bodies as passages). Train on 270M pairs with MNRL.
2. **Supervised fine-tuning**: Fine-tune on high-quality labeled pairs (MS MARCO, NLI, etc.).

The instruct variant (E5-instruct) allows task descriptions: "Instruct: Given a crypto news article, retrieve relevant on-chain data. Query: {query}".

### 6.2 BGE: BAAI General Embeddings

BGE (Zhang et al., 2023) uses **RetroMAE** pretraining and an LLM-based reranker for hard negative mining.

**RetroMAE**: A masked autoencoder where the encoder produces a single vector representation and a decoder reconstructs the original text from this bottleneck. This forces the encoder to compress all information into a single vector — ideal for bi-encoder embeddings.

**LLM-based hard negative mining**: Use an LLM to score candidate negatives and select semantically similar but non-matching passages as hard negatives. Hard negatives dramatically improve retrieval performance over random negatives.

### 6.3 Matryoshka Representation Learning (MRL)

MRL (Kusupati et al., 2022) addresses the dilemma between embedding quality and cost: larger embeddings are more expressive but more expensive to store and compare.

**Core idea.** Train a single model such that the first $d'$ dimensions of the full $d$-dimensional embedding are themselves a meaningful $d'$-dimensional embedding, for a set of target dimensions $\mathcal{D} = \{64, 128, 256, 512, 1024, 2048\}$.

This is analogous to Russian Matryoshka dolls: smaller representations are nested inside larger ones.

**MRL Loss:**

$$\mathcal{L}_{\text{MRL}} = \sum_{d' \in \mathcal{D}} \lambda_{d'} \cdot \mathcal{L}(f(x)[:d'])$$

where:
- $f(x) \in \mathbb{R}^d$ is the full embedding
- $f(x)[:d']$ is the first $d'$ dimensions
- $\mathcal{L}(\cdot)$ is the task loss (e.g., MNRL) computed using only the first $d'$ dimensions
- $\lambda_{d'}$ is a weighting coefficient (often uniform: $\lambda_{d'} = 1/|\mathcal{D}|$)

**Why frontloading information works.** The loss at all resolutions simultaneously trains the model to pack the most important information into early dimensions. Gradient from the $d'=64$ loss flows directly to the first 64 dimensions, creating strong pressure to make those dimensions maximally informative.

**Practical tradeoffs:**

| Dimensions | MTEB Recall@10 | Storage per embedding | Comparison cost |
|---|---|---|---|
| 64 | ~82% | 256 bytes | 64 FLOPs |
| 128 | ~87% | 512 bytes | 128 FLOPs |
| 256 | ~90% | 1024 bytes | 256 FLOPs |
| 512 | ~92% | 2048 bytes | 512 FLOPs |
| 1024 | ~94% | 4096 bytes | 1024 FLOPs |

MRL allows **adaptive retrieval**: use 64 dimensions for fast first-stage retrieval, then re-score with 1024 dimensions for top candidates.

### 6.4 LLM-Based Embeddings: E5-Mistral, GTE-Qwen

**Surprising result**: decoder-only LLMs (GPT-style) produce excellent embeddings despite being designed for generation, not representation.

**Architecture trick.** Take the last token's hidden state from the final layer as the embedding. For decoder-only models, the last token attends to all previous tokens (due to causal masking), so it aggregates all contextual information.

**Why they work despite causal masking:**
1. LLMs are pretrained on vastly more data (trillions of tokens vs billions for BERT).
2. Larger capacity (7B+ parameters vs 110M for BERT-base).
3. Better general world knowledge, especially for reasoning-intensive retrieval.
4. Instruction tuning aligns them well with query understanding.

**E5-Mistral-7B** and **GTE-Qwen2-7B** consistently top the MTEB leaderboard, with the latter achieving state-of-the-art across retrieval, reranking, and STS tasks as of 2024.

### 6.5 Problems & Mitigations for Modern Embedding Models

**Computational cost.** LLM-based models (7B parameters) require significant GPU resources for both fine-tuning and inference. **Mitigation**: Use quantization (INT8/INT4), distillation to smaller models, or PEFT methods like LoRA.

**Long-context handling.** Most models have 512-token limits; long documents require chunking, causing loss of cross-chunk information. **Mitigation**: Hierarchical encoding, sliding window with overlap, document summary embeddings.

**MTEB overfitting.** Models fine-tuned specifically for MTEB tasks may not generalize to real-world domains. **Mitigation**: Always evaluate on domain-specific benchmarks alongside MTEB.

---

## 7. Multi-Modal Embeddings (CLIP)

CLIP (Radford et al., 2021) learns to align images and text in a shared embedding space through contrastive pretraining on 400 million image-text pairs.

### 7.1 Architecture

```
Images: [img_1, img_2, ..., img_N]    Texts: [txt_1, txt_2, ..., txt_N]
           |                                         |
    [Image Encoder]                         [Text Encoder]
    (ViT or ResNet)                       (Transformer)
           |                                         |
    [Linear Projection]                    [Linear Projection]
           |                                         |
   I_1, I_2, ..., I_N                    T_1, T_2, ..., T_N
   (in R^d, L2-normalized)               (in R^d, L2-normalized)
           |                                         |
           |----------[Similarity Matrix]------------|
                       I @ T^T  (N x N matrix)
                       S[i,j] = cos(I_i, T_j)
```

The model is trained so that the diagonal of this matrix (matched pairs) is maximized while off-diagonal elements (mismatched pairs) are minimized.

### 7.2 Symmetric InfoNCE Loss — Full Derivation

For a batch of $N$ (image, text) pairs, define the similarity matrix $S \in \mathbb{R}^{N \times N}$ where $S_{ij} = \frac{I_i^\top T_j}{\tau}$ and $\tau$ is a learned temperature parameter.

**Image-to-text loss** (each image should retrieve its paired text):

$$\mathcal{L}_{I \rightarrow T} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(S_{ii})}{\sum_{j=1}^{N} \exp(S_{ij})}$$

**Text-to-image loss** (each text should retrieve its paired image):

$$\mathcal{L}_{T \rightarrow I} = -\frac{1}{N} \sum_{j=1}^{N} \log \frac{\exp(S_{jj})}{\sum_{i=1}^{N} \exp(S_{ij})}$$

**Symmetric CLIP loss:**

$$\mathcal{L}_{\text{CLIP}} = \frac{1}{2}\left(\mathcal{L}_{I \rightarrow T} + \mathcal{L}_{T \rightarrow I}\right)$$

The symmetry ensures that the alignment is mutual: similar images and texts attract in both directions.

### 7.3 Temperature Parameter

The temperature $\tau$ (initialized at $1/0.07 \approx 14.3$) is a **learned scalar** that controls how peaked the similarity distribution is.

- **Low $\tau$ (high temperature scaling):** Distribution is very sharp, model is penalized heavily for any confusion, hard learning but can overfit.
- **High $\tau$ (low temperature scaling):** Distribution is flat, softer training signal, slower convergence.

The optimal $\tau$ is learned through backpropagation. In CLIP, it's clipped to prevent instability. The choice is critical: CLIP's paper reports that the learned $\tau$ corresponds to a "temperature" that makes the softmax distribution approximately uniform over in-batch negatives.

### 7.4 Zero-Shot Classification

CLIP enables zero-shot classification without any task-specific training:

1. **Template prompting**: For each class label $c_k$, create text prompt: "a photo of a {$c_k$}".
2. **Encode text**: $T_k = \text{TextEncoder}(\text{"a photo of a } c_k\text{"})$
3. **Encode image**: $I = \text{ImageEncoder}(\text{image})$
4. **Predict**: $\hat{k} = \arg\max_k \cos(I, T_k)$

This allows classification into arbitrary categories without any labeled images, purely from the aligned embedding space.

### 7.5 Limitations & Problems

**Bag-of-words behavior.** CLIP treats text as a bag of concepts and struggles with compositional descriptions like "a red circle on top of a blue square" vs "a blue circle on top of a red square".

**Compositional failures.** Attribute binding is weak: "a dog chasing a cat" and "a cat chasing a dog" produce similar embeddings.

**Typographic attacks.** Attaching a text label to an image (e.g., labeling an apple "iPod") can fool CLIP into misclassifying the image, because the text encoder overrides visual evidence.

**Mitigations:**
- **Ensemble of prompts**: Use multiple text templates ("a photo of a {}", "an image of a {}", "a picture of the {}") and average embeddings.
- **Adversarial fine-tuning**: Fine-tune on adversarial examples to improve robustness.
- **BLIP-2**: Adds cross-attention between modalities, improving compositional understanding.

**Extensions:**
- **ALIGN** (Jia et al., 2021): Similar to CLIP but trained on 1.8B noisy image-text pairs.
- **BLIP-2** (Li et al., 2023): Adds a Q-Former module between frozen image encoder and LLM for more sophisticated cross-modal reasoning.
- **ImageBind** (Girdhar et al., 2023): Binds 6 modalities (image, text, audio, depth, thermal, IMU) into a single shared space using image as the "anchor" modality.

---

## 8. Embedding Space Analysis

### 8.1 Similarity Metrics

**Cosine Similarity:**

$$\text{cos}(u, v) = \frac{u^\top v}{\|u\| \|v\|}$$

- Range: $[-1, 1]$
- Scale-invariant: only direction matters, not magnitude
- Best for: when you care about semantic direction, not magnitude (most NLP tasks)
- Geometric interpretation: angle between vectors

**Dot Product:**

$$\text{dot}(u, v) = u^\top v = \|u\| \|v\| \cos\theta$$

- Combines magnitude and direction
- Best for: maximum inner product search (MIPS) in recommendation systems where item popularity matters
- Used by: many production recommendation systems, asymmetric retrieval

**Euclidean Distance:**

$$d_E(u, v) = \|u - v\|_2 = \sqrt{\sum_i (u_i - v_i)^2}$$

- Best for: when absolute position matters (e.g., anomaly detection on embeddings)
- Equivalent to cosine similarity when all vectors are L2-normalized: $\|u - v\|^2 = 2 - 2u^\top v$ for unit vectors

**When to use which:**
- Cosine: sentence similarity, information retrieval, most semantic tasks
- Dot product: recommendation systems, learned metric where scale is informative
- Euclidean: clustering, anomaly detection, when normalization would be harmful

### 8.2 Isotropy Measurement

A distribution over embeddings is **isotropic** if it is invariant to rotation. The measure of isotropy:

$$\text{I}(\mathcal{E}) = \frac{\lambda_{\min}}{\lambda_{\max}}$$

where $\lambda_{\min}$ and $\lambda_{\max}$ are the smallest and largest eigenvalues of the covariance matrix $\Sigma = \text{Cov}(\{e_i\})$. A perfectly isotropic distribution has $\text{I} = 1$; highly anisotropic spaces have $\text{I} \approx 0$.

Alternative measure: partition function of the embedding distribution $Z = \sum_w \exp(v_w^\top u)$ should be independent of direction $u$ for isotropic embeddings.

### 8.3 Johnson-Lindenstrauss Lemma

**Statement.** For any set of $n$ points in $\mathbb{R}^d$ and any $\epsilon \in (0, 1)$, there exists a mapping $f: \mathbb{R}^d \rightarrow \mathbb{R}^k$ with:

$$k = O\left(\frac{\log n}{\epsilon^2}\right)$$

such that for all pairs of points $u, v$:

$$(1 - \epsilon) \|u - v\|^2 \leq \|f(u) - f(v)\|^2 \leq (1 + \epsilon) \|u - v\|^2$$

**Implication for embeddings.** To preserve pairwise distances within $\epsilon$ relative error for a corpus of $n = 10^6$ documents, we only need $k \approx \frac{2\log(10^6)}{\epsilon^2} = \frac{2 \times 13.8}{\epsilon^2}$ dimensions. For $\epsilon = 0.1$: $k \approx 2760$ dimensions. For $\epsilon = 0.3$: $k \approx 307$ dimensions.

**Practical use.** Random Gaussian projections $f(x) = \frac{1}{\sqrt{k}} Rx$ where $R \in \mathbb{R}^{k \times d}$ with $R_{ij} \sim \mathcal{N}(0,1)$ achieve this bound. This justifies both dimensionality reduction and the compressibility of embedding information.

### 8.4 Dimensionality Reduction for Visualization

**PCA.** Project onto the top-$k$ principal components. Preserves maximum variance but uses linear projection — cannot capture nonlinear structure.

**UMAP (Uniform Manifold Approximation and Projection).** Nonlinear dimensionality reduction that preserves both local and global structure better than t-SNE. For embedding visualization:

```python
import umap
reducer = umap.UMAP(n_components=2, metric='cosine')
embedding_2d = reducer.fit_transform(embeddings)  # N x 2
```

UMAP with cosine metric is particularly appropriate for embedding spaces since embeddings are typically compared with cosine similarity.

### 8.5 Analogy Evaluation and Embedding Quality Metrics

**Word Analogy Benchmark.** The Google Analogy Dataset contains 19,544 analogies in the form "a is to b as c is to d". Accuracy is measured by:

$$\text{acc} = \frac{1}{|\mathcal{A}|} \sum_{(a,b,c,d) \in \mathcal{A}} \mathbb{1}\left[d = \arg\max_{w \neq a,b,c} \cos(v_w, v_b - v_a + v_c)\right]$$

**MTEB (Massive Text Embedding Benchmark).** The most comprehensive embedding evaluation suite, with 56 tasks across 8 categories: classification, clustering, pair classification, reranking, retrieval, STS, summarization, and BitextMining. MTEB score is the primary benchmark for comparing modern embedding models.

---

## 9. Vector Similarity Search & Indexing

### 9.1 Brute Force: $O(Nd)$

For a query $q$ and database of $N$ vectors each of dimension $d$, brute force computes all $N$ dot products:

$$\text{scores} = q^\top X, \quad X \in \mathbb{R}^{d \times N}$$

Cost: $O(Nd)$ per query. Acceptable for $N < 10{,}000$; impractical for $N > 10^6$.

### 9.2 IVF: Inverted File Index

**Algorithm:**
1. **Offline**: Run k-means on database vectors, producing $n_{\text{list}}$ centroids $\{c_k\}$.
2. **Offline**: Assign each database vector to its nearest centroid, creating $n_{\text{list}}$ inverted lists.
3. **Query**: Compute distance from query to all $n_{\text{list}}$ centroids, select top $n_{\text{probe}}$ lists, brute force within those lists only.

**Complexity**: $O(n_{\text{list}} \cdot d + n_{\text{probe}} \cdot N/n_{\text{list}} \cdot d)$ per query.

**Key tradeoff**: `nlist` (number of clusters) vs `nprobe` (clusters searched per query). Larger `nprobe` gives better recall but is slower. Typical settings: `nlist = sqrt(N)`, `nprobe = 8` to `64`.

**Problem**: IVF doesn't compress vectors, so memory usage remains $O(Nd)$.

### 9.3 Product Quantization (PQ)

PQ compresses each vector into a compact code, enabling in-RAM storage of hundreds of millions of vectors.

**Algorithm:**

1. **Split** each $d$-dimensional vector into $M$ sub-vectors of dimension $d' = d/M$:

$$v = [v^{(1)}, v^{(2)}, \ldots, v^{(M)}], \quad v^{(m)} \in \mathbb{R}^{d'}$$

2. **Train $M$ sub-quantizers**: For each sub-space $m$, run k-means with $K$ centroids (codebook). Typical $K = 256$ (8-bit code per sub-vector).

3. **Encode**: Replace each sub-vector $v^{(m)}$ with the index of its nearest centroid: $c^{(m)} \in \{0, \ldots, K-1\}$. Full code: $M$ bytes per vector (vs $4d$ bytes for float32).

4. **Decode (ADC — Asymmetric Distance Computation)**: For a query $q$, precompute distances from $q^{(m)}$ to all $K$ centroids in sub-space $m$:

$$\text{LUT}[m][k] = \|q^{(m)} - \text{centroid}_m^{(k)}\|^2$$

This lookup table has $M \times K$ entries. Approximate distance from query $q$ to compressed vector $c$:

$$\tilde{d}(q, v) \approx \sum_{m=1}^{M} \text{LUT}[m][c^{(m)}]$$

5. **Complexity**: Query time $O(N \cdot M)$ table lookups (vs $O(N \cdot d)$ for brute force). Memory: $N \cdot M$ bytes (vs $N \cdot d \cdot 4$ bytes float32).

**Example**: $d = 768$, $M = 96$, $K = 256$. Memory: $N \times 96$ bytes vs $N \times 3072$ bytes, giving **32x compression**. At $N = 10^6$: 96MB vs 3GB.

### 9.4 HNSW: Hierarchical Navigable Small World

HNSW builds a **multi-layer proximity graph** enabling greedy graph traversal for approximate nearest neighbor search.

```
Layer 2 (sparse):    o - - - - - o - - - - - o
                     |                       |
Layer 1 (medium):    o - o - - - o - o - - - o
                     |   |       |   |       |
Layer 0 (dense):     o-o-o-o-o-o-o-o-o-o-o-o-o
```

**Construction**: Each vector is inserted at a random maximum layer (exponentially distributed). In each layer, it connects to the $M$ nearest existing neighbors (found by greedy search from the entry point).

**Search**: Start at the top layer, greedily descend toward the query, expanding to lower layers when converged at each level.

**Complexity**: Search $O(\log N)$ per query. Construction: $O(N \log N)$.

**Key parameters**:
- `M`: number of connections per node (affects recall and memory, typical 16-64)
- `ef_construction`: beam size during construction (affects quality, typical 100-200)
- `ef_search`: beam size during search (tradeoff recall/latency)

### 9.5 ScaNN: Scalable Nearest Neighbors

Google's ScaNN uses **anisotropic vector quantization**: instead of minimizing reconstruction error uniformly, it prioritizes preserving the component of each vector in the direction of likely query vectors. This reduces the error that matters most for inner product search.

### 9.6 Method Comparison

| Method | Build Time | Query Time | Recall@10 | Memory | Notes |
|---|---|---|---|---|---|
| Brute Force (FAISS Flat) | O(N) | O(Nd) | 100% | O(Nd) float32 | Baseline |
| IVF | O(N) | O(nprobe * N/nlist * d) | 90-99% | O(Nd) float32 | Needs nprobe tuning |
| IVF+PQ | O(N) | O(nprobe * N/nlist * M) | 85-95% | O(N*M) bytes | 32x compression |
| HNSW | O(N log N) | O(log N * M * ef) | 95-99% | O(N*M) float32 | Fast, no compression |
| HNSW+SQ | O(N log N) | O(log N) | 92-98% | O(N*d/4) | Scalar quantized |

### 9.7 Problems & Mitigations

**Index Staleness.** When new vectors are added, HNSW graphs become sub-optimal. **Mitigation**: periodic graph rebuilding, or use IVF which supports incremental updates via re-assignment.

**Rebuild Cost.** Rebuilding HNSW on $10^9$ vectors takes hours. **Mitigation**: incremental indexing with periodic consolidation, or use LSH (Locality Sensitive Hashing) which supports instant updates.

**Recall vs Latency Tradeoff.** Every approximate index sacrifices recall for speed. **Mitigation**: two-stage pipeline (approximate retrieval then exact reranking on top-$k$), calibrate `ef_search` on your query distribution.

---

## 10. Problems & Mitigations (Dedicated Section)

### 10.1 Bi-Encoder Quality Gap vs Cross-Encoder

**Problem.** Bi-encoders cannot model interaction between query and document — each is encoded independently. Cross-encoders consistently achieve 3-8% higher NDCG on retrieval benchmarks.

**Mitigation 1: Knowledge Distillation.** Train a bi-encoder to mimic cross-encoder scores:

$$\mathcal{L}_{\text{KD}} = \text{KL}(p_{\text{cross-encoder}} \| p_{\text{bi-encoder}})$$

The cross-encoder's soft probability distribution over candidates is used as a teacher signal. This is the approach used in TASB, RocketQA, and AR2.

**Mitigation 2: Iterative Hard Negative Mining.**
1. Train bi-encoder with random negatives.
2. Use trained bi-encoder to retrieve top-$k$ candidates for each training query.
3. Use cross-encoder to score candidates; hard negatives are those ranked high by bi-encoder but labeled non-relevant.
4. Retrain bi-encoder with hard negatives.
5. Repeat.

This iteratively closes the quality gap because the bi-encoder is forced to distinguish hard cases.

### 10.2 Domain Mismatch

**Problem.** Embedding models trained on general web text (Wikipedia, Common Crawl) perform poorly on specialized domains: medical, legal, financial, blockchain.

**Mitigation 1: Domain-Adaptive Pretraining (DAPT).** Continue MLM pretraining on domain-specific unlabeled text before fine-tuning. For crypto: continue pretraining on CoinDesk articles, whitepapers, on-chain data descriptions.

**Mitigation 2: Synthetic Data Generation.** Use an LLM to generate (query, passage) pairs from domain documents. Instruction: "Given this paragraph from a DeFi protocol whitepaper, generate 5 questions a user might ask that this paragraph answers." Fine-tune embedding model on these synthetic pairs.

**Mitigation 3: Benchmark on Your Domain.** Do not rely on MTEB general scores. Create a small domain-specific eval set (100-500 query-document pairs) and measure actual performance.

### 10.3 Embedding Drift on Model Updates

**Problem.** When the embedding model is updated (new version), all pre-computed embeddings in the index become stale. Recomputing embeddings for millions of documents is expensive.

**Mitigation 1: Versioning Strategy.** Maintain multiple index versions. Route queries to the appropriate index based on the model version. Gradually migrate: rebuild index for new documents immediately, batch-recompute for old documents.

**Mitigation 2: Backward-Compatible Training.** Add a consistency loss to new model training:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \cdot \mathbb{E}_x\left[\|f_{\text{new}}(x) - f_{\text{old}}(x)\|^2\right]$$

This encourages the new model to produce similar embeddings to the old one, reducing drift.

**Mitigation 3: Linear Mapping.** Train a linear projection $A \in \mathbb{R}^{d \times d}$ such that $A \cdot f_{\text{new}}(x) \approx f_{\text{old}}(x)$. Apply $A$ to new model outputs, preserving compatibility with the old index.

### 10.4 Anisotropy

**Problem.** Embeddings occupy a narrow cone, degrading cosine similarity as a semantic metric.

**Mitigation 1: Whitening.** Apply PCA whitening (see Section 4.3). Fast and effective but requires a representative sample.

**Mitigation 2: Contrastive Training.** MNRL loss directly pushes embeddings of different items apart, naturally spreading them across the space.

**Mitigation 3: RMS Normalization Post-Processing.** Subtract the mean embedding (the "rogue dimension") and normalize: $\tilde{e} = (e - \mu) / \|e - \mu\|$.

### 10.5 OOV (Out-of-Vocabulary) Tokens

**Problem.** Words not in the training vocabulary have no learned embedding.

**Mitigation 1: Subword Tokenization.** WordPiece (BERT), BPE (GPT), or SentencePiece tokenize unknown words into known subwords. "DeFi" gets split into ["De", "##Fi"]; "CryptoKitties" gets split into ["Crypto", "##Kit", "##ties"].

**Mitigation 2: Character-Level Fallback.** Models like FastText learn character n-gram embeddings, enabling any word to be represented as the sum of its n-gram embeddings.

**Mitigation 3: Adaptive Vocabulary Expansion.** Periodically retrain the tokenizer on new domain text and extend the embedding matrix with fine-tuned representations.

### 10.6 Multilingual Quality Disparity

**Problem.** Multilingual models are trained on unbalanced corpora: English dominates, leading to poor performance on low-resource languages. "Token fertility" (tokens needed per word) is much higher for agglutinative languages (Turkish, Finnish, Korean), making models less efficient.

**Mitigation 1: Language-Balanced Sampling.** Upsample low-resource languages during pretraining using temperature-based sampling:

$$P_{\text{sample}}(l) \propto \left(\frac{P_l}{\sum_{l'} P_{l'}}\right)^{1/T}$$

Higher $T$ produces a more uniform language distribution. mBERT uses $T = 0.7$.

**Mitigation 2: Multilingual Contrastive Training.** Train on parallel translation pairs: $(x_{\text{en}}, x_{\text{zh}})$ as positive pairs for MNRL. This directly trains cross-lingual alignment.

---

## 11. Industry Practices

### 11.1 Choosing an Embedding Model

**Step 1: Check MTEB but don't stop there.** MTEB provides general benchmark scores. Use it for initial shortlisting.

**Step 2: Build a domain-specific eval set.** For Binance: 100-500 (query, `relevant_document`) pairs from your domain (crypto news, user queries, on-chain data descriptions). Use a cross-encoder or human annotators to create relevance labels.

**Step 3: Benchmark on your eval set.** Measure NDCG@10, Recall@100 using your domain eval set. The MTEB winner may not win here.

**Step 4: Consider latency and cost.** 7B LLM-based embeddings are 10-50x slower and more expensive than BERT-based models. For latency-sensitive applications, 110M models may be the right choice.

**Step 5: Consider context length.** For long documents (whitepapers, technical reports), use models with longer context: E5-mistral (4096 tokens), nomic-embed-text-v1.5 (8192 tokens).

### 11.2 Hybrid Search with BM25 + Dense Retrieval

**Architecture:**

```
Query
  |
  |-----> BM25 Retriever -----> [doc_1, doc_3, doc_7, ...]  (BM25 scores)
  |
  |-----> Dense Retriever ----> [doc_2, doc_3, doc_5, ...]  (cosine scores)
  |
  |-----> [Reciprocal Rank Fusion] -----> Combined ranking
```

**Reciprocal Rank Fusion (RRF):**

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

where $R$ is the set of rankers (BM25 and dense), $\text{rank}_r(d)$ is document $d$'s rank in ranker $r$, and $k = 60$ is a constant (empirically robust).

RRF is robust to score scale differences and doesn't require normalization of BM25 and cosine scores.

**Alternatively**, use linear interpolation if scores are normalized:

$$s_{\text{hybrid}}(d) = \alpha \cdot s_{\text{dense}}(d) + (1 - \alpha) \cdot s_{\text{BM25}}(d)$$

where $\alpha$ is tuned on a validation set.

### 11.3 Embedding Caching Strategies

**Document embeddings**: Always precompute and cache. Documents are static; computing embeddings online wastes GPU compute.

**Query embeddings**: Can cache for repeated queries. Use query normalization (lowercase, strip punctuation) before caching to increase cache hit rate.

**Cache invalidation**: When model is updated (version bump), invalidate all cached embeddings and recompute.

**Implementation**: Redis or Memcached for query embedding cache (sub-millisecond lookup). S3/GCS + local SSD cache for document embeddings.

### 11.4 When to Fine-Tune vs Use Out-of-the-Box

**Use out-of-the-box when:**
- General domain query types (semantic search over news, Q&A)
- Small evaluation budget — no domain eval set
- Prototype stage — time to market matters
- Domain is well-represented in MTEB training data

**Fine-tune when:**
- Domain-specific terminology (DeFi protocols, tokenomics, Layer-2 solutions)
- Custom query patterns (structured on-chain queries, API requests)
- You have labeled data (even 1000 pairs can significantly improve performance)
- Production system with sustained traffic justifying the engineering cost

**Fine-tuning data sources for crypto:**
- User query to clicked document pairs from search logs
- Support ticket to resolved FAQ pairs
- News article to related on-chain events (from labeled datasets)
- LLM-generated synthetic pairs from whitepapers and documentation

### 11.5 Crypto-Specific Embedding Applications

**Financial News Embeddings.** Embed CoinDesk, Cointelegraph articles with domain-adapted models. Cluster by topic (regulatory news, DeFi exploits, ETF announcements). Track semantic drift over time to detect emerging narratives.

**Tokenomics Report Retrieval.** Build a knowledge base of tokenomics documents. Given a user question ("What is the vesting schedule for BNB?"), retrieve relevant sections using dense retrieval.

**On-Chain Transaction Pattern Embeddings.** Represent wallet transaction histories as sequences and embed with a sequence model (Transformer or LSTM). Use embeddings to detect:
- Wash trading (circular transaction patterns)
- Airdrop farming (similar behavior to known farmers)
- Whale accumulation patterns

**Crypto Entity Resolution.** Embed entity descriptions (exchange names, token names, protocol names) to resolve aliases: "BTC" = "Bitcoin" = "XBT", "ETH" = "Ethereum".

---

## 12. Interview Q&A

### Basic Questions (5)

---

**Q1: What is the difference between a word embedding and a sentence embedding?**

A word embedding maps a single token to a vector, typically independently of context (Word2Vec, GloVe) or contextually given the sentence (BERT token embeddings). A sentence embedding maps an entire sentence to a single vector that represents the overall meaning. Sentence embeddings are produced by pooling over token embeddings (mean, CLS, max) or by models specifically trained for sentence-level tasks (SBERT). For retrieval tasks, you almost always want sentence embeddings, because you need to compare entire queries to entire documents. The distinction matters practically: "bank" in Word2Vec has one vector regardless of context (financial bank vs river bank), but in BERT-based sentence embeddings, the contextual representation changes the whole sentence vector.

---

**Q2: Why is cosine similarity preferred over Euclidean distance for comparing text embeddings?**

Cosine similarity is scale-invariant: it measures only the angle between vectors, not their magnitude. Text embeddings can have varying magnitudes depending on sentence length, input token distributions, and model internals. Two semantically identical sentences might produce embeddings of very different norms. Cosine similarity correctly identifies them as similar because $\cos(\theta) = 1$ regardless of norm. Euclidean distance would incorrectly penalize the norm difference. Additionally, most embedding training objectives (MNRL, InfoNCE) normalize vectors to the unit sphere, making cosine and dot product equivalent and both appropriate. For L2-normalized vectors: $\|u - v\|^2 = 2 - 2u^\top v = 2(1 - \cos(u,v))$, so Euclidean and cosine are monotonically related and equivalent for ranking.

---

**Q3: Explain BM25 in simple terms. What makes it better than raw TF-IDF?**

BM25 improves TF-IDF in two ways. First, **saturation**: in TF-IDF, mentioning a word 100 times gives 10x more weight than mentioning it 10 times. This is unrealistic — after a certain point, additional occurrences add diminishing relevance. BM25's $k_1$ parameter causes TF contribution to saturate: the gain from the 100th occurrence is much smaller than from the 10th. Second, **length normalization**: BM25's $b$ parameter adjusts for document length. A long document naturally mentions more words, but that shouldn't inflate its score. BM25 normalizes by the ratio of document length to average document length. The IDF component is also smoothed: $\log\frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5}$ instead of the raw $\log\frac{N}{\text{df}(t)}$, preventing extreme values for very rare terms.

---

**Q4: What is the [CLS] token in BERT, and why doesn't it work well for sentence similarity out of the box?**

The `[CLS]` token is a special token prepended to every BERT input. In pretraining, its final hidden state is used as the sequence representation for classification tasks (Next Sentence Prediction). However, this pretraining objective doesn't require the `[CLS]` embedding to capture semantic similarity — it only needs to predict whether two sentences are adjacent in a document. As a result, the geometric relationship between `[CLS]` embeddings of different sentences is not meaningful for similarity. SBERT's key contribution is fine-tuning BERT with a Siamese architecture and contrastive or MSE objectives so that the resulting embeddings (from mean pooling, not `[CLS]`) are geometrically meaningful for similarity tasks. Reimers & Gurevych (2019) showed that BERT CLS pooling achieves only Spearman $\rho \approx 0.20$ on STS tasks, worse than averaging random GloVe vectors ($\rho \approx 0.36$).

---

**Q5: What are in-batch negatives in contrastive learning?**

In a training batch of $N$ (query, document) positive pairs, **in-batch negatives** treat all other documents in the same batch as negatives for each query. Query $q_i$ is paired positively with $p_i$, and negatively with $p_1, \ldots, p_{i-1}, p_{i+1}, \ldots, p_N$. This provides $N-1$ negatives per query for free, without needing explicit negative annotation. The quality of these negatives depends on batch composition: larger and more diverse batches provide harder and more informative negatives. The Multiple Negatives Ranking Loss (MNRL) formalizes this approach. The key insight is efficiency: a batch of $N=256$ pairs provides $256 \times 255 = 65,280$ training signals in a single forward pass.

---

### Intermediate Questions (5)

---

**Q6: Derive why BERT embeddings are anisotropic.**

BERT is pretrained with Masked Language Modeling (MLM). At each step, the model predicts masked tokens by computing:

$$P(w \mid \text{context}) = \text{softmax}(W h_{[MASK]})$$

where $W \in \mathbb{R}^{|\mathcal{V}| \times d}$ is the output projection. To produce a good probability distribution, the model needs $W h$ to cover a large range. High-frequency tokens (punctuation, function words) appear as masked tokens constantly and their embeddings are pulled toward certain directions due to the optimization pressure of being predicted accurately.

Arora et al. (2019) showed this creates a "partition function" effect: the embedding of each word absorbs a component proportional to $\log P(w)$, the word's log-frequency. This common component (the "rogue dimension") causes all embeddings to point in a similar direction. Additionally, the softmax normalization in self-attention encourages tokens to attend to a few dominant tokens, concentrating information flow and breaking isotropy.

The measured consequence: average cosine similarity between random BERT embeddings is $\approx 0.6$ instead of the isotropic baseline of $\approx 0$. This means cosine similarity fails to discriminate semantically different sentences because background similarity is already very high.

---

**Q7: Explain the connection between Word2Vec and PMI matrix factorization.**

Levy and Goldberg (2014) showed that skip-gram with negative sampling (SGNS) implicitly factorizes a shifted PMI matrix. At the optimal solution of the SGNS objective:

$$v_w^{\top} v_c' = \text{PMI}(w, c) - \log k$$

where $k$ is the number of negative samples. PMI (Pointwise Mutual Information) is:

$$\text{PMI}(w, c) = \log \frac{P(w, c)}{P(w) P(c)}$$

High PMI means $w$ and $c$ co-occur much more than chance — exactly what the skip-gram objective rewards. The dot product of the embeddings approximates the Shifted PMI (SPMI). GloVe makes this connection explicit by directly minimizing a weighted reconstruction of $\log X_{ij}$ (the log co-occurrence count), which is proportional to PMI. Both methods are therefore learning low-rank approximations of word co-occurrence statistics, explaining why they produce similar-quality embeddings and both capture analogical relationships. The analogy property emerges because PMI differences encode semantic relationships: $\text{SPMI}(\text{king}, c) - \text{SPMI}(\text{man}, c) \approx \text{SPMI}(\text{queen}, c) - \text{SPMI}(\text{woman}, c)$ for all context words $c$ relating to royalty.

---

**Q8: Why is MNRL loss effective for training retrieval models, and what is the role of temperature?**

MNRL (also called InfoNCE) works by training the model to correctly identify the positive document among $N$ candidates (1 positive + $N-1$ in-batch negatives). The gradient signal is informative because: (1) it provides $O(N^2)$ learning signal from $N$ examples; (2) hard negatives in the batch (semantically similar but incorrect documents) provide strong gradients when the model nearly confuses them with the positive; (3) the objective directly optimizes ranking, not just binary classification.

Temperature $\tau$ controls the sharpness of the distribution:

$$p_i = \frac{\exp(s_{ii} / \tau)}{\sum_j \exp(s_{ij} / \tau)}$$

Low $\tau$ (e.g., 0.01): very peaked distribution, loss only gets gradient from near-confusable negatives, efficient but unstable (vanishing gradients for already-separated pairs). High $\tau$ (e.g., 1.0): flat distribution, gradient from all negatives equally, more stable but less efficient learning. The optimal $\tau$ is typically 0.05-0.1, often learned via backpropagation. The temperature also connects to the InfoNCE mutual information lower bound: $I(X; Y) \geq \log N - \mathcal{L}_{\text{InfoNCE}}$, tighter at lower temperatures with larger batches.

---

**Q9: What is Matryoshka Representation Learning and when would you use it in production?**

MRL trains a single embedding model such that the first $d'$ dimensions of the full embedding are themselves a valid $d'$-dimensional embedding, for multiple values of $d' \in \{64, 128, 256, 512, 1024, 2048\}$. This is achieved by applying the training loss at each resolution simultaneously:

$$\mathcal{L}_{\text{MRL}} = \sum_{d' \in \mathcal{D}} \lambda_{d'} \cdot \mathcal{L}(f(x)[:d'])$$

The gradient from the $d'=64$ loss flows directly to the first 64 dimensions, creating strong pressure to make those dimensions maximally informative — "frontloading" information.

In production, MRL enables **adaptive retrieval**: first stage retrieves top-1000 candidates using 64-dim embeddings (fast, compact), second stage re-scores top-1000 with 1024-dim embeddings (accurate). This is useful at Binance for high-throughput search over millions of documents where latency constraints prevent using full-dimensional embeddings everywhere. A single MRL model replaces what would otherwise require two separate model deployments. The key metric: 64-dim achieves ~87% of the recall of 1024-dim, at 1/16th the computation and storage cost.

---

**Q10: How does CLIP perform zero-shot classification, and what are its failure modes?**

CLIP aligns image and text embeddings in a shared space via contrastive pretraining. For zero-shot classification: given $K$ classes $\{c_1, \ldots, c_K\}$, create text prompts "a photo of a {$c_k$}" and encode each as $T_k$. Encode the query image as $I$. Predict:

$$\hat{k} = \arg\max_k \cos(I, T_k)$$

This works because training has aligned image regions with their textual descriptions.

**Failure modes:**
1. **Compositional failures**: "red ball on blue cube" and "blue ball on red cube" produce nearly identical embeddings because CLIP learns bags of concepts, not compositional structure.
2. **Typographic attacks**: Placing text "iPod" on an apple image causes CLIP to classify it as an iPod — the text signal overrides visual evidence.
3. **Fine-grained classification**: CLIP struggles to distinguish between visually similar but semantically distinct categories (dog breeds, bird species) without additional prompting.
4. **Negation**: "a dog without a leash" is not well-handled because the embedding space doesn't natively support negation.
5. **Distribution shift**: CLIP was trained on internet images and captions; performance degrades on medical imaging, satellite imagery, or highly technical diagrams.

---

### Advanced Questions (5)

---

**Q11: Design a multilingual crypto news search system from scratch. Walk through every component.**

**Requirements**: Handle queries in English, Chinese, Korean, Japanese; documents are crypto news articles; latency < 200ms; support 10M documents.

**Step 1: Embedding Model.** Choose multilingual-e5-large (560M parameters, trained on 100+ languages, state-of-the-art on multilingual MTEB). Fine-tune on domain-specific (query, article) pairs generated synthetically: use GPT-4 to generate questions from article snippets in each language.

**Step 2: Tokenization.** Use the multilingual model's tokenizer (SentencePiece with 250K vocabulary). Handle crypto-specific terms: add domain vocabulary if token fertility is high (e.g., "DeFi" shouldn't split unnecessarily).

**Step 3: Document Processing.** Articles can be long (2000+ tokens, model max=512). Strategy: chunk each article into overlapping 512-token segments, embed each segment independently, store segment embeddings. At query time, retrieve top-$k$ segments, then group by article.

**Step 4: Index.** HNSW index with $M=32$, $ef_{\text{construction}}=200$ for the 10M document segments (~50M vectors). Build index offline, serve with FAISS. Memory: 50M x 1024 dim x 4 bytes = 200GB: use IVF+PQ for compression: 50M x 128 bytes = 6.4GB, fits in RAM.

**Step 5: Hybrid Search.** Maintain separate BM25 index (Elasticsearch) per language. For each query: (1) detect language, (2) run BM25 in detected language, (3) run dense retrieval (language-agnostic), (4) fuse with RRF: $\text{RRF}(d) = \frac{1}{60 + \text{rank}_{\text{BM25}}(d)} + \frac{1}{60 + \text{rank}_{\text{dense}}(d)}$.

**Step 6: Reranking.** Use multilingual cross-encoder (e.g., mmarco-mMiniLMv2-L12-H384) to rerank top-100, return top-10. Latency budget: dense retrieval 50ms + BM25 20ms + RRF 5ms + reranking 80ms = 155ms total.

**Step 7: Monitoring.** Track: query latency (p50, p95, p99), recall (via click-through rate), embedding space drift (cosine similarity distribution over time), query language distribution.

---

**Q12: How would you handle embedding drift in production at scale?**

**The problem.** Model v2 replaces v1. All 10M pre-computed embeddings were produced by v1. Re-embedding with v2 takes $10^7 \times 0.05$s per embedding = 138 GPU-hours. Meanwhile, the index is stale and performance degrades.

**Strategy 1: Dual-index period.** Run both v1 and v2 indices simultaneously. Route 5% of traffic to v2 index (A/B test). Measure quality. Gradually shift traffic.

**Strategy 2: Backward-compatible training.** During v2 training, add consistency regularization:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \alpha \|f_{\text{v2}}(x) - f_{\text{v1}}(x)\|^2$$
This minimizes drift for unchanged documents.

**Strategy 3: Priority-based reindexing.** Reindex documents by importance. Hot documents (accessed in last 7 days): reindex first. Tail documents: batch-reindex over 2 weeks.

**Strategy 4: Linear compatibility layer.** Learn $A \in \mathbb{R}^{d \times d}$ such that $A \cdot e_{\text{v2}}(x) \approx e_{\text{v1}}(x)$ using a sample of shared documents. Apply $A$ to all v2 query embeddings, making them compatible with the v1 index. This is a fast stopgap: a single matrix multiply per query instead of reindexing the entire corpus.

**Operational practice:** Version all embeddings in your storage system. Use a model registry to track which model version produced which embeddings. Set SLAs for reindexing: new documents must be indexed within X hours, old documents within Y days after model update.

---

**Q13: Derive and explain why the $\frac{3}{4}$ power works as the noise distribution exponent in Word2Vec negative sampling.**

The exact derivation is empirical rather than theoretical. Consider what we want from the noise distribution $P_n(w)$:

**Too uniform** ($P_n \propto 1$): Rare words are oversampled as negatives. For query word "satoshi", negatives like "aardvark" (rare) are sampled as often as "the" (common). But "aardvark" is an easy negative — the model trivially distinguishes it. Poor learning signal.

**Unigram** ($P_n \propto f_w$): Common words dominate as negatives. "The", "of", "a" are sampled constantly. These are easy negatives too, since any meaningful word is unlikely to be a context word for "the" in a semantic sense. The model learns to avoid common words but doesn't learn rich semantics.

**Sublinear** ($P_n \propto f_w^\alpha$, $0 < \alpha < 1$): Intermediate. Common words are sampled proportionally less (relative to their frequency), and rare words proportionally more. The $3/4$ power empirically provides the best balance of hard negatives across the frequency spectrum.

Mathematically: if $f_w = 1000$ and $f_{w'} = 1$, then $P_n(w)/P_n(w') = 1000^{3/4}/1^{3/4} = 177.8$ (vs 1000 for unigram). The $3/4$ power compresses the frequency distribution, giving rare words roughly $1000/177.8 \approx 5.6\times$ more relative sampling mass than they'd get under unigram. In general, for $\alpha = 3/4$, the effective frequency ratio between two words with frequencies $f_1$ and $f_2$ is $(f_1/f_2)^{3/4}$ instead of $f_1/f_2$. This geometric compression produces a noise distribution that is neither too peaked (unigram) nor too flat (uniform), yielding hard and varied negatives. The value $3/4$ was chosen by grid search over $\alpha \in \{0.5, 0.6, 0.7, 0.75, 0.8, 1.0\}$ and found consistently better across multiple tasks.

---

**Q14: Explain Product Quantization fully. What are the theoretical and practical tradeoffs?**

PQ splits each $d$-dimensional vector into $M$ sub-vectors of dimension $d' = d/M$ and quantizes each independently with $K$ centroids. A vector is stored as $M$ bytes (for $K = 256$).

**Full encoding pipeline:**

$$v = [v^{(1)}, \ldots, v^{(M)}] \xrightarrow{\text{k-means per sub-space}} [c^{(1)}, \ldots, c^{(M)}]$$

Each $c^{(m)} \in \{0, \ldots, 255\}$ is an 8-bit integer. The full code is $M$ bytes.

**Asymmetric Distance Computation (ADC):**
1. Precompute LUT: $\text{LUT}[m][k] = \|q^{(m)} - \mu_m^{(k)}\|^2$ for all $m, k$. Cost: $O(M \cdot K \cdot d')$.
2. Score each database vector: $\tilde{d}(q, c) = \sum_{m=1}^M \text{LUT}[m][c^{(m)}]$. Cost: $O(M)$ per vector.

**Theoretical distortion.** The quantization error for each sub-vector is bounded by the k-means quantization error in $\mathbb{R}^{d'}$. By rate-distortion theory, with $K$ centroids in $d'$ dimensions:

$$\mathbb{E}[\|v^{(m)} - \hat{v}^{(m)}\|^2] \leq C \cdot \sigma^2 \cdot K^{-2/d'}$$

The distance approximation error for the full vector sums over $M$ sub-vectors:

$$\mathbb{E}\left[(\tilde{d}(q,v) - d(q,v))^2\right] \leq M \cdot \epsilon_{\text{sub}}^2$$

**Practical tradeoffs:**
- More sub-vectors $M$ → more compression but higher approximation error per sub-vector.
- More centroids $K$ → less quantization error but exponentially more codebook memory ($M \times K \times d'$ floats).
- $K = 256$ and $M = d/8$ is the standard configuration ($d = 768$, $M = 96$).

**Compression ratio**: $4d$ bytes (float32) to $M$ bytes: ratio = $4d/M = 4 \times 8 = 32$x compression.

**Key limitation**: PQ approximates distances, not exact scores. The recall degradation depends on the dataset's intrinsic dimensionality and cluster structure. On natural language embeddings with strong clustering, PQ recall is surprisingly high (>90% at top-100) because semantically similar vectors naturally cluster in sub-spaces.

---

**Q15: How does the InfoNCE loss relate to mutual information maximization, and what does this imply for contrastive learning design?**

The InfoNCE loss:

$$\mathcal{L}_{\text{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f(x)^\top f(y) / \tau)}{\frac{1}{N}\sum_{j=1}^N \exp(f(x)^\top f(y_j) / \tau)}\right]$$

is a lower bound on the mutual information $I(X; Y)$ between views $X$ and $Y$:

$$I(X; Y) \geq \log N - \mathcal{L}_{\text{InfoNCE}}$$

This was derived by van den Oord et al. (2018). Minimizing $\mathcal{L}_{\text{InfoNCE}}$ maximizes this lower bound. The bound is tight when $N$ is large (many negatives) and the distribution is well-covered (diverse batch).

**Design implications:**

1. **Larger batch size is always better**: the bound $\log N$ increases with $N$, so more in-batch negatives tighten the lower bound and provide richer training signal.

2. **Hard negatives help convergence but not the bound**: hard negatives don't change $N$, but they make the optimization landscape better conditioned.

3. **Temperature controls bound tightness**: lower $\tau$ makes the bound tighter but the gradient landscape more jagged, potentially causing instability.

4. **What the model actually learns**: InfoNCE trains $f$ to capture information shared between views $X$ and $Y$ while discarding information unique to each view. For (query, passage) pairs, this means learning semantic content (shared) while discarding stylistic differences (not shared). For (image, caption) pairs (CLIP), this means learning visual-semantic correspondences.

5. **Failure mode of mode collapse**: if $f(x)$ ignores the input and outputs a constant, $\mathcal{L} = \log N$ (random chance). The loss never goes below $\log N$ divided by how informative the learned representation is. Monitoring training: if loss stagnates near $\log N$, representations have collapsed.

---

## 13. Coding Problems

### Problem 1: Cosine Similarity from Scratch

```python
import numpy as np
from typing import Union

def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        u: First vector of shape (d,)
        v: Second vector of shape (d,)

    Returns:
        Cosine similarity in [-1, 1]
    """
    # Compute dot product
    dot_product = np.dot(u, v)

    # Compute norms
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    # Guard against division by zero
    if norm_u == 0 or norm_v == 0:
        return 0.0

    return dot_product / (norm_u * norm_v)


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    Args:
        A: Matrix of shape (N, d)
        B: Matrix of shape (M, d)

    Returns:
        Similarity matrix of shape (N, M)
    """
    # L2-normalize each row
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity = dot product of normalized vectors
    return A_norm @ B_norm.T


# Example usage
if __name__ == "__main__":
    v_bitcoin  = np.array([0.8,  0.6,  0.1, -0.2,  0.9])
    v_ethereum = np.array([0.7,  0.5,  0.2, -0.1,  0.8])
    v_dog      = np.array([-0.3, 0.1,  0.9,  0.7, -0.5])

    print(f"bitcoin <-> ethereum: {cosine_similarity(v_bitcoin, v_ethereum):.4f}")
    print(f"bitcoin <-> dog:      {cosine_similarity(v_bitcoin, v_dog):.4f}")
```

### Problem 2: SBERT Mean Pooling from Scratch

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def mean_pooling(model_output: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Perform mean pooling over token embeddings, weighted by attention mask.

    Args:
        model_output:   Last hidden states, shape (batch_size, seq_len, hidden_dim)
        attention_mask: Binary mask,        shape (batch_size, seq_len)
                        1 = real token, 0 = padding token

    Returns:
        Pooled embeddings of shape (batch_size, hidden_dim)
    """
    token_embeddings = model_output  # (B, L, H)

    # Expand mask to match token embedding shape: (B, L) -> (B, L, H)
    mask_expanded = (
        attention_mask
        .unsqueeze(-1)                         # (B, L, 1)
        .expand(token_embeddings.size())       # (B, L, H)
        .float()
    )

    # Zero-out padding token positions, then sum
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)  # (B, H)

    # Count real tokens per sequence to compute the mean
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)             # (B, H)

    return sum_embeddings / sum_mask                                  # (B, H)


def encode_sentences(
    sentences: list,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> torch.Tensor:
    """
    Full pipeline: tokenize -> encode -> mean pool -> L2-normalize.

    Returns:
        Sentence embeddings of shape (N, hidden_dim), L2-normalized
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded)

    # Mean pool over token dimension
    sentence_embeddings = mean_pooling(
        outputs.last_hidden_state, encoded["attention_mask"]
    )

    # L2-normalize so that dot product equals cosine similarity
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

    return sentence_embeddings


if __name__ == "__main__":
    sentences = [
        "Bitcoin price reaches all-time high",
        "BTC achieves record value",
        "The weather today is sunny"
    ]

    embeddings = encode_sentences(sentences)
    sim_matrix = embeddings @ embeddings.T

    print("Similarity matrix:")
    print(sim_matrix.numpy().round(3))
    # Expected: [0,1] high (same topic), [0,2] and [1,2] low
```

### Problem 3: BM25 from Scratch

```python
import math
from collections import defaultdict
from typing import List, Tuple


class BM25:
    """
    Okapi BM25 implementation.

    Score(q, d) = sum_t IDF(t) * f(t,d)*(k1+1) / (f(t,d) + k1*(1-b+b*|d|/avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: TF saturation (1.2 to 2.0 typical)
            b:  Length normalization (0 = none, 1 = full, 0.75 default)
        """
        self.k1 = k1
        self.b = b
        self.corpus: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.df: dict = defaultdict(int)
        self.idf: dict = {}
        self.N: int = 0

    def fit(self, corpus: List[List[str]]) -> "BM25":
        """Index the corpus (list of tokenized documents)."""
        self.corpus = corpus
        self.N = len(corpus)
        self.doc_lengths = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_lengths) / self.N

        # Compute document frequencies
        for doc in corpus:
            for term in set(doc):
                self.df[term] += 1

        # Robertson-Sparck Jones IDF
        for term, df_t in self.df.items():
            self.idf[term] = math.log(
                (self.N - df_t + 0.5) / (df_t + 0.5) + 1
            )

        return self

    def _score(self, query: List[str], doc_idx: int) -> float:
        """BM25 score for one document."""
        doc = self.corpus[doc_idx]
        doc_len = self.doc_lengths[doc_idx]

        tf: dict = defaultdict(int)
        for term in doc:
            tf[term] += 1

        score = 0.0
        for term in query:
            if term not in self.idf:
                continue
            f = tf[term]
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (
                1 - self.b + self.b * doc_len / self.avgdl
            )
            score += self.idf[term] * numerator / denominator

        return score

    def retrieve(self, query: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """Return top-k (doc_idx, score) pairs, sorted descending."""
        scores = [(i, self._score(query, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


if __name__ == "__main__":
    corpus = [
        "bitcoin price surges to record high as institutional investors enter".split(),
        "ethereum defi protocol launches new yield farming mechanism".split(),
        "bitcoin halving event reduces block reward by half every four years".split(),
        "federal reserve raises interest rates impacting crypto markets".split(),
        "binance launches new trading pair for bitcoin and ethereum".split(),
    ]

    bm25 = BM25(k1=1.5, b=0.75).fit(corpus)
    query = "bitcoin price record".split()
    results = bm25.retrieve(query, top_k=3)

    print(f"Query: {' '.join(query)}")
    for doc_idx, score in results:
        print(f"  Score {score:.3f}: {' '.join(corpus[doc_idx])}")
```

### Problem 4: Semantic Search Engine with FAISS

```python
import numpy as np
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple


class SemanticSearchEngine:
    """
    End-to-end semantic search engine with FAISS ANN index.
    Supports flat (exact), IVF, and HNSW index types.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10,
    ):
        self.model_name = model_name
        self.index_type = index_type
        self.nlist = nlist
        self.nprobe = nprobe

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.index = None
        self.documents: List[str] = []

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    def _encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                outputs = self.model(**encoded)

            # Mean pool
            mask = (
                encoded["attention_mask"]
                .unsqueeze(-1)
                .expand(outputs.last_hidden_state.size())
                .float()
            )
            emb = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = F.normalize(emb, p=2, dim=1)
            all_embeddings.append(emb.numpy())

        return np.vstack(all_embeddings).astype("float32")

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------
    def _build_index(self, embeddings: np.ndarray) -> None:
        d = embeddings.shape[1]

        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(d)

        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            self.index = faiss.IndexIVFFlat(
                quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT
            )
            self.index.train(embeddings)
            self.index.nprobe = self.nprobe

        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 64

        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        self.index.add(embeddings)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def index_documents(self, documents: List[str]) -> None:
        """Encode and index documents."""
        print(f"Encoding {len(documents)} documents...")
        self.documents = documents
        embeddings = self._encode(documents)
        print(f"Building {self.index_type} index (dim={embeddings.shape[1]})...")
        self._build_index(embeddings)
        print(f"Index built: {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Return top-k (document, score) pairs for a query string."""
        assert self.index is not None, "Call index_documents() first."
        q_emb = self._encode([query])                          # (1, d)
        scores, indices = self.index.search(q_emb, top_k)
        return [
            (self.documents[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx >= 0
        ]


if __name__ == "__main__":
    corpus = [
        "Bitcoin halving reduces the block reward miners receive by 50%.",
        "Ethereum's proof-of-stake transition significantly reduced energy use.",
        "DeFi protocols allow users to earn yield through liquidity provision.",
        "Binance Smart Chain enables faster and cheaper transactions.",
        "NFT marketplaces saw record trading volumes in 2021.",
        "The Lightning Network enables instant Bitcoin micropayments.",
        "Stablecoins like USDT maintain a 1:1 peg to the US dollar.",
        "Layer-2 rollups process transactions off-chain to reduce gas fees.",
    ]

    engine = SemanticSearchEngine(index_type="flat")
    engine.index_documents(corpus)

    query = "How does Bitcoin reduce mining rewards over time?"
    results = engine.search(query, top_k=3)
    print(f"\nQuery: {query}")
    for doc, score in results:
        print(f"  [{score:.3f}] {doc}")
```

### Problem 5: Multiple Negatives Ranking Loss (MNRL)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultipleNegativesRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss (MNRL / InfoNCE).

    For a batch of N (query, positive) pairs, treats all other
    positives in the batch as negatives for each query.

    Mathematical form:
        L = -mean_i log [ exp(sim(q_i, p_i)/tau) /
                          sum_j exp(sim(q_i, p_j)/tau) ]

    Args:
        temperature: Softmax temperature (default 0.05)
    """

    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
        negative_embeddings: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            query_embeddings:    (N, d)
            positive_embeddings: (N, d) — one positive per query
            negative_embeddings: (N, d) optional hard negatives

        Returns:
            Scalar loss
        """
        q = F.normalize(query_embeddings,    p=2, dim=-1)  # (N, d)
        p = F.normalize(positive_embeddings, p=2, dim=-1)  # (N, d)

        # Similarity matrix: (N, N), scaled by temperature
        sim = (q @ p.T) / self.temperature  # sim[i,j] = cos(q_i, p_j)/tau

        # Optionally append hard negatives as extra columns
        if negative_embeddings is not None:
            n = F.normalize(negative_embeddings, p=2, dim=-1)
            neg_sim = (q @ n.T) / self.temperature
            sim = torch.cat([sim, neg_sim], dim=1)  # (N, 2N)

        # Labels: the correct positive for query i is at column i
        labels = torch.arange(q.size(0), device=q.device)

        # Cross-entropy: for each row i, maximize column i
        return F.cross_entropy(sim, labels)

    @torch.no_grad()
    def compute_metrics(
        self,
        query_embeddings: torch.Tensor,
        positive_embeddings: torch.Tensor,
    ) -> dict:
        """Retrieval accuracy metrics (unscaled cosine similarity)."""
        q = F.normalize(query_embeddings,    p=2, dim=-1)
        p = F.normalize(positive_embeddings, p=2, dim=-1)
        sim = q @ p.T                              # (N, N)

        # Rank of the correct positive for each query (1-indexed)
        diag_scores = sim.diag().unsqueeze(1)      # (N, 1)
        ranks = (sim > diag_scores).sum(dim=1).float() + 1  # (N,)

        return {
            "accuracy@1":          (ranks <= 1).float().mean().item(),
            "accuracy@3":          (ranks <= 3).float().mean().item(),
            "accuracy@5":          (ranks <= 5).float().mean().item(),
            "mean_rank":           ranks.mean().item(),
            "mean_reciprocal_rank": (1.0 / ranks).mean().item(),
        }


if __name__ == "__main__":
    torch.manual_seed(42)
    N, d = 32, 768

    q_emb   = torch.randn(N, d)
    pos_emb = torch.randn(N, d)
    neg_emb = torch.randn(N, d)

    criterion = MultipleNegativesRankingLoss(temperature=0.05)
    loss = criterion(q_emb, pos_emb, neg_emb)
    print(f"MNRL Loss: {loss.item():.4f}")

    metrics = criterion.compute_metrics(q_emb, pos_emb)
    print("Metrics (random embeddings — expect ~random performance):")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # ----------------------------------------------------------------
    # Sketch of a training loop using MNRL
    # ----------------------------------------------------------------
    # from transformers import AutoModel
    # model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    # criterion = MultipleNegativesRankingLoss(temperature=0.05)
    #
    # for epoch in range(num_epochs):
    #     for batch in dataloader:          # batch has "query" and "positive" fields
    #         q_emb   = encode(model, batch["query"])    # (N, d)
    #         pos_emb = encode(model, batch["positive"]) # (N, d)
    #
    #         loss = criterion(q_emb, pos_emb)
    #         loss.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
```

---

## Summary: Key Formulas at a Glance

| Concept | Formula |
|---|---|
| TF-IDF | $\text{tfidf}(t,d) = \text{tf}(t,d) \cdot \log\frac{N}{\text{df}(t)}$ |
| BM25 | $\sum_t \text{idf}(t) \cdot \frac{f_{t,d}(k_1+1)}{f_{t,d}+k_1(1-b+b\frac{|d|}{\text{avgdl}})}$ |
| Skip-gram objective | $\frac{1}{T}\sum_t \sum_{j \neq 0} \log P(w_{t+j} \mid w_t)$ |
| Neg. sampling prob. | $P_n(w) \propto \text{count}(w)^{3/4}$ |
| GloVe loss | $\sum_{i,j} f(X_{ij})(v_i^\top \tilde{v}_j + b_i + \tilde{b}_j - \log X_{ij})^2$ |
| MNRL loss | $-\frac{1}{N}\sum_i \log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau)}$ |
| MRL loss | $\sum_{d' \in \mathcal{D}} \lambda_{d'} \cdot \mathcal{L}(f(x)[:d'])$ |
| CLIP loss | $\frac{1}{2}(\mathcal{L}_{I\to T} + \mathcal{L}_{T\to I})$ |
| Cosine similarity | $\cos(u,v) = \frac{u^\top v}{\|u\|\|v\|}$ |
| RRF | $\text{RRF}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)}$ |
| JL Lemma | $k = O(\log n / \epsilon^2)$ |
| InfoNCE MI bound | $I(X;Y) \geq \log N - \mathcal{L}_{\text{InfoNCE}}$ |
| Whitening | $\tilde{e} = \Lambda^{-1/2} U^\top (e - \mu)$ |
| PQ distance approx. | $\tilde{d}(q,v) \approx \sum_{m=1}^M \text{LUT}[m][c^{(m)}]$ |

---

*End of Document — Embeddings Comprehensive Technical Guide*

*ST5230 | Binance Interview Preparation | Last updated: 2026-03-04*
