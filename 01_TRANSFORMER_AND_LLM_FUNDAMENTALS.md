# Transformer & LLM Fundamentals
## Interview Preparation — Comprehensive Technical Reference Guide

> **Audience**: Assumes familiarity with basic ML (loss functions, backpropagation, linear algebra) but NOT with transformers.
> **Style**: Lecture-note level. Every claim is justified. All math is derived, not stated.

---

## Table of Contents

1. [The Attention Mechanism](#1-the-attention-mechanism)
2. [Multi-Head Attention](#2-multi-head-attention)
3. [Positional Encoding](#3-positional-encoding)
4. [The Full Transformer Architecture](#4-the-full-transformer-architecture)
5. [Pretraining Objectives](#5-pretraining-objectives)
6. [Layer Normalization](#6-layer-normalization)
7. [Tokenization](#7-tokenization)
8. [Scaling Laws](#8-scaling-laws)
9. [Inference Optimizations](#9-inference-optimizations)
10. [Problems & Mitigations (Dedicated Section)](#10-problems--mitigations-dedicated-section)
11. [Industry Practices at Binance Scale](#11-industry-practices-at-binance-scale)
12. [Interview Q&A](#12-interview-qa)

---

## 1. The Attention Mechanism

### 1.1 The Problem: Why Does Attention Exist?

Before attention, sequence models (RNNs, LSTMs) processed tokens one at a time and compressed the entire history into a fixed-size hidden state. This created a **bottleneck**: the model had to remember everything in one vector.

Consider the sentence:

> "The animal didn't cross the street because **it** was too tired."

What does "it" refer to? To a human, obviously "the animal." But an RNN processing left-to-right must somehow encode, in its hidden state at position 7 ("it"), enough information about position 2 ("animal") to resolve this coreference. Over long sequences, this signal degrades via the vanishing gradient problem.

**Attention solves this directly**: at every position, the model can directly look back at all previous positions and decide how much weight to give each. The word "it" can attend directly to "animal" with high weight, regardless of distance.

This is the core intuition: **attention is a soft, differentiable lookup mechanism** that lets each output position selectively read from all input positions.

### 1.2 Deriving Scaled Dot-Product Attention

#### Step 1: Linear Projections to Q, K, V

Given an input sequence of $n$ tokens, each represented as a $d_{\text{model}}$-dimensional vector, we form a matrix $X \in \mathbb{R}^{n \times d_{\text{model}}}$.

We project $X$ into three matrices via learned weight matrices:

$$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$$

where $W^Q, W^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$.

**Database analogy for Q, K, V**: Think of attention as a soft database query.
- **K (Keys)**: What each entry in the database is "about." Like a column index.
- **V (Values)**: The actual content stored at each entry.
- **Q (Query)**: What you're looking for right now.

When you look up "animal" while processing "it," the query for "it" will have a high dot product with the key for "animal," so the value of "animal" gets retrieved with high weight. The difference from a hard lookup is that all values are retrieved and weighted by similarity — it is a **weighted average retrieval**.

#### Step 2: Compute Raw Attention Scores

$$\text{scores} = QK^T \in \mathbb{R}^{n \times n}$$

The $(i, j)$ entry of this matrix is $q_i \cdot k_j$, the dot product between the query at position $i$ and the key at position $j$. A large positive value means "position $i$ should attend heavily to position $j$."

#### Step 3: Why Scale by $\frac{1}{\sqrt{d_k}}$? — Variance Argument

This is a subtle but important point. Assume the elements of $q$ and $k$ are i.i.d. standard normal:

$$q_i, k_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$$

The dot product is:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$

Since $q_i$ and $k_i$ are independent, each term $q_i k_i$ has:

$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$

$$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - 0 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1 \cdot 1 = 1$$

Since the $d_k$ terms are independent:

$$\text{Var}(q \cdot k) = d_k$$

$$\text{Std}(q \cdot k) = \sqrt{d_k}$$

So as $d_k$ grows (e.g., $d_k = 64$, $\sqrt{64} = 8$), the dot products can be very large. When you feed large values into softmax, the gradients become vanishingly small because softmax becomes nearly a one-hot distribution (the largest logit dominates completely). This is the **saturation problem**.

**The fix**: divide by $\sqrt{d_k}$ to normalize the variance back to 1:

$$\frac{q \cdot k}{\sqrt{d_k}} \text{ has variance } \frac{d_k}{d_k} = 1$$

#### Step 4: Softmax to Get Attention Weights

$$\alpha_{i,:} = \text{softmax}\left(\frac{Q_i K^T}{\sqrt{d_k}}\right) \in \mathbb{R}^n$$

The softmax turns raw scores into a probability distribution over positions. Each row of the attention weight matrix sums to 1: $\sum_j \alpha_{i,j} = 1$.

#### Step 5: Weighted Sum of Values

$$\text{output}_i = \sum_j \alpha_{i,j} V_j$$

In matrix form, the complete formula is:

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}$$

### 1.3 Self-Attention vs Cross-Attention

| Type | Q source | K, V source | Used in |
|---|---|---|---|
| Self-attention | Same sequence | Same sequence | Encoder, decoder causal self-attn |
| Cross-attention | Decoder hidden states | Encoder output | Encoder-decoder cross-attn layer |

### 1.4 Causal (Masked) Self-Attention

In autoregressive models (GPT, Llama), token $i$ must not attend to tokens $j > i$:

$$\text{mask}[i][j] = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

$$\text{scores\_masked} = \frac{QK^T}{\sqrt{d_k}} + \text{mask}$$

Setting future positions to $-\infty$ makes $e^{-\infty} = 0$, so their softmax weight is exactly zero.

### 1.5 ASCII Diagram: Attention Computation

```
Input X (n x d_model)
        |
   +----|----+
   |    |    |
  W^Q  W^K  W^V    (learned projection matrices)
   |    |    |
   Q    K    V
(n×dk)(n×dk)(n×dv)
   |    |
   +----+
   QK^T              <-- (n×n) raw scores
   |
   ÷ sqrt(dk)        <-- scaling: normalize variance to 1
   |
 + mask (optional)   <-- -inf for future positions (causal) or padding
   |
 softmax             <-- row-wise: each row sums to 1 → attention weights A (n×n)
   |
   A × V             <-- (n×n) × (n×dv) = (n×dv) output
   |
 Output (n × dv)
```

### 1.6 Computational Complexity

- **Time**: Computing $QK^T$ is an $(n \times d_k) \times (d_k \times n)$ matrix multiply: $O(n^2 d_k)$. Then multiplying $A \times V$ is $(n \times n) \times (n \times d_v)$: $O(n^2 d_v)$. Total: $O(n^2 d)$.
- **Memory**: The attention matrix $A \in \mathbb{R}^{n \times n}$ requires $O(n^2)$ memory.

For a 100K-token context window, $n^2 = 10^{10}$. At float16 (2 bytes), that is 20 GB just for the attention matrix — **per layer, per head**. This is the fundamental scalability barrier.

### 1.7 Problems & Mitigations

| Problem | Description | Mitigation |
|---|---|---|
| **Quadratic complexity** | $O(n^2)$ time and memory in sequence length | Sparse attention (Longformer), linear attention (Performer), Flash Attention (IO-aware tiling) |
| **Attention sink** | The [BOS] token accumulates disproportionate attention weight, stealing signal | StreamingLLM fixes by always retaining sink tokens; attention temperature tuning |
| **Attention to padding** | Padding tokens in batches receive and contribute spurious attention | Mask out padding positions by setting scores to $-\infty$ before softmax |
| **Numerical instability** | Large logits before softmax cause overflow in $e^x$ | Use the log-sum-exp trick: subtract max before softmax. Flash Attention implements online softmax |
| **Causal masking overhead** | Decoder must mask upper triangle at every layer | Fused kernel implementations (Flash Attention) handle causal masking efficiently |

---

## 2. Multi-Head Attention

### 2.1 Intuition: Why One Head Is Not Enough

A single attention head computes one set of attention weights — one "view" of the sequence. But sequences have multiple types of relationships simultaneously:
- Syntactic dependencies (subject-verb agreement)
- Coreference (pronoun to antecedent)
- Semantic roles (who did what to whom)
- Local vs. long-range relationships

A single head must somehow represent all of these with one attention pattern. In practice, we allow the model to compute $h$ different attention patterns in parallel, each looking at the sequence through a different learned projection. The results are concatenated and re-projected. This is **Multi-Head Attention (MHA)**.

### 2.2 Full Mathematical Formulation

For head $i$ ($i = 1, \ldots, h$):

$$\text{head}_i = \text{Attention}(Q W_i^Q,\ K W_i^K,\ V W_i^V)$$

where $W_i^Q, W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$ and $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, with $d_k = d_v = d_{\text{model}} / h$.

All heads are computed in parallel, then concatenated:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\ W^O$$

where $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$ is the output projection.

### 2.3 Parameter Count Derivation

For $h$ heads with $d_k = d_v = d_{\text{model}} / h$:

- $W_i^Q$: $d_{\text{model}} \times d_k$ per head, $h$ heads $\rightarrow$ $d_{\text{model}} \times (h \cdot d_k) = d_{\text{model}}^2$ total
- $W_i^K$: same $\rightarrow d_{\text{model}}^2$
- $W_i^V$: same $\rightarrow d_{\text{model}}^2$
- $W^O$: $h d_v \times d_{\text{model}} = d_{\text{model}} \times d_{\text{model}}$ $\rightarrow d_{\text{model}}^2$

Total parameters in MHA (ignoring biases):

$$\boxed{4 d_{\text{model}}^2}$$

For GPT-3 ($d_{\text{model}} = 12288$, 96 layers): MHA parameters $\approx 4 \times 12288^2 \times 96 \approx 58.5\text{B}$ out of 175B total.

### 2.4 ASCII Diagram: Multi-Head Attention

```
Input Q, K, V  (n x d_model each)
        |
   +----------+----------+----------+
   | Head 1   | Head 2   | Head h   |
   |          |          |          |
  W1^Q,K,V  W2^Q,K,V   Wh^Q,K,V   (each projects to d_k)
   |          |          |
 Attn_1    Attn_2    Attn_h        (each: n x d_v)
   |          |          |
   +----------+----------+
          Concat              (n x h*d_v = n x d_model)
             |
            W^O               (d_model x d_model output projection)
             |
          Output              (n x d_model)
```

### 2.5 Multi-Query Attention (MQA)

**Motivation**: During autoregressive inference, the **KV cache** stores keys and values for all past tokens. With MHA and $h$ heads, the cache size grows as $O(n \times h \times d_k)$. For large models with many heads, this becomes a severe memory bottleneck.

**MQA** (Shazeer, 2019): All heads share a **single** set of K, V projections, while each head still has its own Q projection:

$$\text{head}_i = \text{Attention}(Q W_i^Q,\ K W^K,\ V W^V)$$

KV cache memory reduction: $h\times$ reduction (e.g., 32× for 32-head model).

**Quality tradeoff**: MQA is slightly worse than MHA on perplexity but the difference is modest and well worth the memory savings in production.

### 2.6 Grouped-Query Attention (GQA)

**GQA** (Ainslie et al., 2023): Intermediate between MHA and MQA. Heads are divided into $g$ groups. Within each group, heads share a single K, V pair:

$$\text{Groups}: g \ll h, \quad \text{e.g., } g = 8, h = 64$$

KV cache memory: $g/h$ fraction of MHA. Quality: between MHA and MQA.

**Real usage**: Llama 2 70B uses GQA with $g = 8$ groups and $h = 64$ heads, reducing KV cache by 8×. Mistral 7B uses GQA with $g = 8$, $h = 32$.

```
Standard MHA:  Q has h heads, K has h heads, V has h heads
GQA:           Q has h heads, K has g heads, V has g heads  (g < h)
MQA:           Q has h heads, K has 1 head,  V has 1 head
```

### 2.7 Problems & Mitigations

| Problem | Description | Mitigation |
|---|---|---|
| **Head redundancy** | Many heads learn similar attention patterns and can be pruned | Attention head pruning (Michel et al., 2019): remove 60-80% of heads with minimal quality loss |
| **Head collapse** | All heads converge to the same pattern during training | Diverse head initialization, regularization on attention entropy |
| **MHA inference memory** | $h$ full K, V copies in cache — memory pressure | MQA, GQA for inference efficiency |
| **Compute per head** | Each head processes the full sequence independently | Flash Attention: fused kernel computes all heads efficiently on GPU |

---

## 3. Positional Encoding

### 3.1 Why Attention Is Permutation Equivariant

Consider permuting the input tokens: $X' = \Pi X$ for a permutation matrix $\Pi$. Then:

$$Q' = \Pi X W^Q = \Pi Q, \quad K' = \Pi K, \quad V' = \Pi V$$

$$\text{Attention}(Q', K', V') = \text{softmax}\!\left(\frac{\Pi Q (\Pi K)^T}{\sqrt{d_k}}\right) \Pi V = \text{softmax}\!\left(\frac{\Pi Q K^T \Pi^T}{\sqrt{d_k}}\right) \Pi V = \Pi \cdot \text{Attention}(Q, K, V)$$

The output is permuted in exactly the same way as the input. This means attention does not know the order of tokens — the sentences "dog bites man" and "man bites dog" would produce the same attention patterns (just with rows shuffled). **Order information must be explicitly injected.**

### 3.2 Sinusoidal Positional Encoding (Vaswani et al., 2017)

For position $\text{pos}$ and dimension index $i$ (out of $d_{\text{model}}$):

$$PE_{(\text{pos},\, 2i)} = \sin\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(\text{pos},\, 2i+1)} = \cos\!\left(\frac{\text{pos}}{10000^{2i/d_{\text{model}}}}\right)$$

**Intuition**: Each pair of dimensions encodes position using a sinusoid of a specific frequency. Low-index dimensions oscillate rapidly (short-range), high-index dimensions oscillate slowly (long-range). This is analogous to a Fourier decomposition of position.

**Why relative positions can be encoded**: Using angle addition formulas, $PE_{\text{pos}+k}$ can be expressed as a linear function of $PE_{\text{pos}}$. For any fixed $k$, there exists a linear transformation $M_k$ such that $M_k \cdot PE_{\text{pos}} = PE_{\text{pos}+k}$. The model can learn to compute relative offsets from the PE vectors.

### 3.3 Learned Positional Encoding (BERT)

BERT learns an embedding table $E \in \mathbb{R}^{L_{\max} \times d_{\text{model}}}$, where $L_{\max} = 512$. Position $\text{pos}$ is encoded as $E[\text{pos}]$.

**Limitation**: Hard upper bound on sequence length. If you need 1024 tokens at inference time and trained on 512, there is no embedding for positions 513–1024 — the model simply cannot handle them.

### 3.4 RoPE (Rotary Position Embedding)

RoPE (Su et al., 2021) is used in Llama, Mistral, Qwen, and most modern open-weight LLMs. Instead of adding position information to embeddings, it **rotates** the Q and K vectors before computing the dot product.

#### Mathematical Derivation

Partition the $d_k$-dimensional query $q$ and key $k$ into $d_k/2$ pairs: $(q_1, q_2), (q_3, q_4), \ldots$

For position $m$, rotate the $j$-th pair by angle $m\theta_j$, where $\theta_j = 10000^{-2j/d_k}$:

$$R_m^{(j)} = \begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix}$$

The full rotation is block-diagonal: $R_m = \text{diag}(R_m^{(1)}, R_m^{(2)}, \ldots, R_m^{(d_k/2)})$.

Query and key at positions $m$ and $n$ are transformed:

$$\tilde{q}_m = R_m q_m, \quad \tilde{k}_n = R_n k_n$$

**Key property — relative position encoding**:

$$\tilde{q}_m^T \tilde{k}_n = (R_m q_m)^T (R_n k_n) = q_m^T R_m^T R_n k_n = q_m^T R_{n-m} k_n$$

Because rotation matrices satisfy $R_m^T R_n = R_{n-m}$, the dot product depends only on the **relative position** $n - m$, not on absolute positions $m$ or $n$ individually. This is exactly the property we want.

#### Why RoPE Extrapolates Better

Sinusoidal PE adds a fixed vector; learned PE hits a hard lookup limit. RoPE encodes relative position in the geometry of the dot product itself. When the model sees positions beyond training length, the rotation angles simply increase — the geometry is smooth and well-defined even for unseen positions. Combined with NTK-aware scaling (Section 10), RoPE can be extended to 2–4× context lengths with minimal fine-tuning.

### 3.5 ALiBi (Attention with Linear Biases)

ALiBi (Press et al., 2022) takes a different approach: it does not modify the embeddings at all. Instead, it adds a linear bias to attention scores based on distance:

$$\text{AttentionScore}_{i,j} = \frac{q_i \cdot k_j}{\sqrt{d_k}} - m \cdot |i - j|$$

where $m$ is a head-specific slope (different for each head, forming a geometric sequence). Closer tokens get less penalty; distant tokens get more.

**Advantage**: No position embeddings at all — clean extrapolation to longer contexts at inference time.

### 3.6 Comparison Diagram

```
                SINUSOIDAL        LEARNED PE        RoPE              ALiBi
                ----------        ----------        ----              -----
Parameters:     None (fixed)      L_max × d_model   None (computed)   None (fixed slopes)
Extrapolation:  Poor beyond 2×    Hard cutoff       Good (+ NTK)      Excellent
Relative pos:   Implicit          No                Exact (proven)    Implicit (via bias)
Used in:        Original Transformer  BERT          Llama, Mistral    MPT, BLOOM
Implementation: Add to embedding  Lookup table      Rotate Q, K       Add to score matrix
Trainable:      No                Yes               No                No
```

### 3.7 Problems & Mitigations

| Problem | Description | Mitigation |
|---|---|---|
| **Hard extrapolation limit** | Learned PE cannot generalize to positions not seen in training | Use RoPE or ALiBi; fine-tune with longer contexts |
| **Positional embedding bottleneck** | With sinusoidal/learned PE, entire position encoded in one vector | RoPE distributes position across all dimensions via rotation |
| **Long context degradation** | Even RoPE degrades for very long contexts (4× training length) | RoPE NTK scaling, YaRN, LongRoPE — rescale base frequency or interpolate |

---

## 4. The Full Transformer Architecture

### 4.1 Encoder-Only (BERT, RoBERTa)

```
Input tokens:  [CLS]  The   animal   ...   [SEP]
                 |     |      |              |
         Token Embeddings (lookup table, d_model)
                 |     |      |              |
         + Positional Encoding (added element-wise)
                 |     |      |              |
            +----+-----+------+--------------+----+
            |  Layer 1:                           |
            |   LayerNorm                         |
            |   Multi-Head Self-Attention         |
            |   (BIDIRECTIONAL: all tokens see    |
            |    all other tokens)                |
            |   + Residual connection             |
            |   LayerNorm                         |
            |   FFN (2-layer MLP)                 |
            |   + Residual connection             |
            +-------------------------------------+
                             ...
            +-------------------------------------+
            |  Layer N                            |
            +-------------------------------------+
                 |     |      |              |
            Contextual representations (n × d_model)
                 |
               [CLS] representation --> Classification head
               All token representations --> Token-level tasks (NER)
```

**What it is used for**: Classification (sentiment, toxicity), NER (token-level labels), question answering (span extraction), embedding generation. Since it sees all tokens bidirectionally, it CANNOT generate text autoregressively.

**Why bidirectional helps**: For understanding tasks, seeing the full context is better. "I saw the man with the telescope" — does "with" modify "man" or "saw"? Future context disambiguates.

### 4.2 Decoder-Only (GPT, Llama, Mistral)

```
Input tokens:  The    cat    sat    on    the
                |      |      |      |     |
         Token Embeddings
                |      |      |      |     |
         Positional Encoding (RoPE applied during attention)
                |      |      |      |     |
       +--------+------+------+------+-----+--------+
       | Layer 1:                                    |
       |  LayerNorm                                  |
       |  Causal Self-Attention                      |
       |  Causal Mask (upper-triangular = -inf):     |
       |                                             |
       |     T    c    a    s    o                   |
       |  T [0   -inf -inf -inf -inf]                |
       |  c [0    0   -inf -inf -inf]                |
       |  a [0    0    0   -inf -inf]                |
       |  s [0    0    0    0   -inf]                |
       |  o [0    0    0    0    0  ]                |
       |                                             |
       |  + Residual connection                      |
       |  LayerNorm                                  |
       |  FFN (SwiGLU in Llama)                      |
       |  + Residual connection                      |
       +---------------------------------------------+
                        ... (N layers)
                |
           Linear unembedding (d_model -> vocab_size)
                |
           Softmax -> Next token probability distribution
```

**What it is used for**: Text generation, chat, code completion, in-context learning. Each token only sees past tokens — can autoregressively generate new tokens.

**Why decoder-only dominates today**: Simplicity (one architecture), emergent in-context learning at scale, natural fit for RLHF/instruction tuning, and empirical scaling results favor it.

### 4.3 Encoder-Decoder (T5, BART)

```
SOURCE TEXT: "Translate to French: The cat sat."

ENCODER (bidirectional):
   The   cat   sat   .
    |     |     |    |
  [Bidirectional self-attention: all tokens attend to all]
    |     |     |    |
  [Layer 1 ... Layer N]
    |     |     |    |
 Encoder hidden states (n_src x d_model)
         |
         |   (passed to decoder via cross-attention)
         v

DECODER (causal + cross-attention):
  <bos>   Le    chat  s'est
    |      |      |      |
  [Causal self-attention: each position sees only past]
    |      |      |      |
  [Cross-attention:
     Q = from decoder hidden states (n_tgt x d_model)
     K, V = from encoder hidden states (n_src x d_model)
     Decoder "reads" source at every layer]
    |      |      |      |
  [FFN]
    |
  Le    chat   s'est  assis   <eos>
  (generated autoregressively, one token at a time)
```

**What it is used for**: Translation, summarization, structured generation. Encoder specializes in understanding source; decoder specializes in generating target. Cross-attention lets each generated token "read" relevant parts of the source at every layer.

### 4.4 Feed-Forward Network (FFN)

Each transformer layer contains a two-layer MLP applied position-wise (independently to each token):

**Original Transformer**:

$$\text{FFN}(x) = \max(0,\ x W_1 + b_1)\ W_2 + b_2$$

with $W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$, $W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$, and $d_{\text{ff}} = 4 d_{\text{model}}$.

**SwiGLU (Llama, PaLM)**:

$$\text{FFN}_{\text{SwiGLU}}(x) = (\text{Swish}(xW_1) \odot (xW_2))\ W_3$$

where $\text{Swish}(x) = x \cdot \sigma(x)$ (also called SiLU) and $\odot$ is element-wise multiplication. SwiGLU uses three matrices instead of two, with $d_{\text{ff}} = \frac{8}{3} d_{\text{model}}$ to keep total parameters comparable to $4d_{\text{model}}^2$. Empirically, SwiGLU outperforms ReLU significantly on downstream tasks.

**Parameter count of FFN**: $2 \times d_{\text{model}} \times d_{\text{ff}} = 2 \times d_{\text{model}} \times 4 d_{\text{model}} = 8 d_{\text{model}}^2$.

**What FFN does**: While attention aggregates information across positions, the FFN transforms the information at each position independently. Mechanistic interpretability research (Geva et al., 2021) shows FFN layers act as "key-value memories" — individual neurons activate for specific semantic patterns and produce corresponding output vectors.

### 4.5 Pre-Norm vs. Post-Norm

**Original Transformer (Post-Norm)**:

$$x_{l+1} = \text{LayerNorm}(x_l + \text{Sublayer}(x_l))$$

**Modern LLMs (Pre-Norm)**:

$$x_{l+1} = x_l + \text{Sublayer}(\text{LayerNorm}(x_l))$$

**Why Pre-Norm dominates**: In post-norm, the gradient must flow through the LayerNorm at every layer, which can cause instability at initialization in deep networks. In pre-norm, the residual stream $x_l$ is always preserved — the gradient can flow directly through the residual connections without passing through any normalization. This enables training deeper models more stably.

**Tradeoff**: Pre-norm models tend to have weaker representation at the final layer because they accumulate raw residuals, not normalized ones. Post-norm often achieves slightly better perplexity when training is stable, but pre-norm is much more reliable in practice.

### 4.6 Residual Connections and Gradient Flow

$$x_{l+1} = x_l + F_l(x_l)$$

The gradient of the loss $\mathcal{L}$ with respect to $x_l$:

$$\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+1}} \cdot \left(I + \frac{\partial F_l}{\partial x_l}\right) = \frac{\partial \mathcal{L}}{\partial x_{l+1}} + \frac{\partial \mathcal{L}}{\partial x_{l+1}} \cdot \frac{\partial F_l}{\partial x_l}$$

The key insight: the gradient always has a direct path ($\frac{\partial \mathcal{L}}{\partial x_{l+1}}$ alone, the identity term) that does not pass through any learned weights. This prevents gradient vanishing in deep networks. Even if $\frac{\partial F_l}{\partial x_l}$ is near zero (layer is effectively skipped), the gradient still flows at full strength.

### 4.7 KV Cache

During autoregressive inference, to generate token $t+1$, the model needs Q, K, V for all tokens $1, \ldots, t$. But K and V for tokens $1, \ldots, t-1$ were already computed when generating token $t$. Without caching, you recompute the entire forward pass for all past tokens at each step — $O(n^2)$ total compute.

**KV cache**: Store K and V for all past tokens. When generating token $t+1$, only compute Q, K, V for the new token, then concatenate new K, V to the cache.

**Memory formula**: For a model with $L$ layers, $H$ heads, $d_k$ dimension per head, generating sequence of length $n$:

$$\text{KV Cache size} = 2 \times L \times H \times d_k \times n \times \text{bytes\_per\_element}$$

For Llama 2 70B ($L=80$, $H=64$, $d_k=128$, float16): 1000-token context $\approx 80 \times 64 \times 128 \times 1000 \times 2 \approx 1.3\ \text{GB}$.

With GQA (8 groups instead of 64 heads): reduces to $\approx 160\ \text{MB}$ for 1000 tokens — an 8× reduction.

### 4.8 Parameter Count Estimation

For a transformer with $L$ layers, $d = d_{\text{model}}$, vocabulary size $V$:

| Component | Parameters |
|---|---|
| Token embeddings | $V \times d$ |
| Per-layer MHA | $4d^2$ |
| Per-layer FFN | $8d^2$ (if $d_{\text{ff}} = 4d$) |
| Per-layer norms | $4d$ (negligible) |
| Total per layer | $\approx 12d^2$ |
| **Total model** | $\approx V \cdot d + 12 L d^2$ |

Example — GPT-3 175B: $d=12288$, $L=96$, $V=50257$:
- Embedding: $50257 \times 12288 \approx 617\text{M}$
- Layers: $96 \times 12 \times 12288^2 \approx 173.9\text{B}$
- Total $\approx 174.6\text{B}$ (matches reported 175B)

---

## 5. Pretraining Objectives

### 5.1 Masked Language Modeling (MLM) — BERT

BERT masks 15% of input tokens and predicts the masked tokens. The masking is not always `[MASK]`:
- **80% of the time**: Replace with `[MASK]` token
- **10% of the time**: Replace with a random token
- **10% of the time**: Keep the original token unchanged

**Why 80/10/10?** If you always used `[MASK]`, the model would only learn to predict when the input token is `[MASK]` — it would have no incentive to build good representations for non-masked tokens. At inference time, there are no `[MASK]` tokens, so this creates a train-test mismatch. The 10% random replacement forces the model to maintain good representations for all tokens (it cannot be sure any given token is correct), and the 10% unchanged forces the model to produce a good representation even for the correct token.

**Training signal**: Only the masked positions contribute to the loss — approximately 15% of tokens per example, making MLM less data-efficient than CLM.

**What MLM cannot do**: Autoregressive generation. Since MLM uses bidirectional context, you cannot use BERT to generate text by predicting the next token.

### 5.2 Causal Language Modeling (CLM) — GPT, Llama

**Objective**: At each position $i$, predict the next token given all previous tokens:

$$\mathcal{L}_{\text{CLM}} = -\sum_{i=1}^{n} \log P(x_i \mid x_1, x_2, \ldots, x_{i-1})$$

**Data efficiency**: Every token in the sequence is a training signal. A 1000-token document produces 999 training examples (predict each token from context). Compare to MLM, which only trains on 15% of positions.

**Why CLM scales better**: The dense supervision signal means CLM models extract more information per token. At scale (billions of tokens, hundreds of billions of parameters), this difference compounds. All frontier models (GPT-4, Llama, Mistral, Claude, Gemini) use CLM.

### 5.3 Span Corruption (T5)

T5 (Raffel et al., 2020) masks **spans** (consecutive runs of tokens) rather than individual tokens. Masked spans are replaced by a single sentinel token (`<extra_id_0>`, `<extra_id_1>`, ...). The decoder must predict all the masked spans in order.

**Advantage**: Longer masked spans teach the model to generate coherent multi-token sequences, better for generation tasks.

### 5.4 Why CLM Scales Better Than MLM

| Property | MLM | CLM |
|---|---|---|
| Training signal density | 15% of tokens | 100% of tokens |
| Bidirectional context | Yes (better for understanding) | No (causal only) |
| Natural generation | No | Yes |
| In-context learning | Weak | Strong |
| Scaling behavior | Saturates earlier | Continues to improve |
| Dominant use | BERT-era NLU tasks | All frontier LLMs today |

---

## 6. Layer Normalization

### 6.1 Batch Norm vs. Layer Norm

**Batch Normalization** (Ioffe & Szegedy, 2015) normalizes across the batch dimension for each feature:

$$\text{BN}(x_{b,d}) = \gamma_d \cdot \frac{x_{b,d} - \mu_d}{\sqrt{\sigma_d^2 + \epsilon}} + \beta_d$$

where $\mu_d = \frac{1}{B}\sum_b x_{b,d}$ and $\sigma_d^2 = \frac{1}{B}\sum_b (x_{b,d} - \mu_d)^2$.

**Why BN fails for NLP**:
- Sequences have variable lengths — padding distorts batch statistics
- At inference with batch size 1, batch statistics are unreliable
- Autoregressive generation always runs with effective batch size 1 per step

**Layer Normalization** (Ba et al., 2016) normalizes across the feature dimension for each sample independently:

$$\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu = \frac{1}{d}\sum_{i=1}^d x_i$ and $\sigma^2 = \frac{1}{d}\sum_{i=1}^d (x_i - \mu)^2$.

Layer Norm does not depend on other samples in the batch — it works for any batch size, any sequence length.

### 6.2 RMSNorm

RMSNorm (Zhang & Sennrich, 2019) simplifies LayerNorm by removing the mean subtraction:

$$\text{RMSNorm}(x) = \gamma \odot \frac{x}{\text{RMS}(x)}, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2}$$

**Why does removing mean subtraction work?** The mean subtraction in LayerNorm is the re-centering step. Empirically, the re-centering is less important than the re-scaling — the model can learn to center via bias terms elsewhere. RMSNorm is faster (one fewer pass over the data, simpler backward pass) and achieves comparable performance. Used in Llama, Mistral, Qwen.

### 6.3 Pre-Norm vs. Post-Norm Gradient Flow

```
POST-NORM (Original Transformer):
  x_l --> [Sublayer F_l] --> (+) --> [LayerNorm] --> x_{l+1}
                              ^
                              | (residual added before norm)
  Gradient path:
  dL/dx_l = dL/dx_{l+1} * d(LN)/dx * (I + dF_l/dx_l)
  LN Jacobian applied at every layer --> can vanish at init

PRE-NORM (GPT-2+, Llama):
  x_l --> [LayerNorm] --> [Sublayer F_l] --> (+) --> x_{l+1}
  |                                          ^
  |__________________________________________|
            (direct residual bypass)

  Gradient path:
  dL/dx_l = dL/dx_{l+1}  (direct path, always intact)
           + dL/dx_{l+1} * dF_l(LN(x_l))/dx_l  (through sublayer)

  Even if sublayer gradient vanishes at init, dL/dx_{l+1} flows!
  --> STABLE at initialization for arbitrary depth
```

---

## 7. Tokenization

### 7.1 BPE (Byte-Pair Encoding) — Step by Step

BPE (Sennrich et al., 2016) starts with a character vocabulary and iteratively merges the most frequent adjacent pair.

**Example**:
```
Corpus: "low low low lower lowest"

Initial chars: l o w _ l o w _ l o w _ l o w e r _ l o w e s t
(underscore represents word boundary)

Count pairs: (l,o)=5, (o,w)=5, (w,_)=3, (w,e)=2, ...

Merge (l,o) -> 'lo':   lo w _ lo w _ lo w _ lo w e r _ lo w e s t
Merge (lo,w) -> 'low': low _ low _ low _ low e r _ low e s t
Merge (low,_) -> 'low_': low_ low_ low_ low e r _ low e s t

Continue until vocabulary size reached.
Final: "lower" -> ["low", "e", "r"], "lowest" -> ["low", "e", "s", "t"]
```

**Result**: Common words and subwords become single tokens; rare words are split into subword pieces.

### 7.2 WordPiece vs. SentencePiece

**WordPiece** (BERT): Like BPE but instead of merging the most frequent pair, merges the pair that maximizes the language model likelihood:

$$\text{score}(A, B) = \frac{\text{freq}(AB)}{\text{freq}(A) \times \text{freq}(B)}$$

Non-initial subwords are marked with `##` prefix: "playing" $\rightarrow$ ["play", "##ing"]. Requires pre-tokenization (whitespace splitting).

**SentencePiece** (T5, Llama): Treats input as a raw byte stream — no pre-tokenization step. Works directly on Unicode code points. Handles any language without language-specific rules. Uses a unigram language model or BPE variant. Llama uses SentencePiece with BPE, vocabulary size 32,000.

### 7.3 Tokenization Failures

**Numbers**: "12345" might become ["12", "34", "5"] or ["1", "2", "3", "4", "5"]. Arithmetic is hard because the model must reason over split digit tokens and reassemble them.

**Multilingual imbalance**: An English-trained BPE tokenizer assigns single tokens to common English words but splits Chinese/Japanese/Korean characters at nearly 1 token per character. This inflates sequence length and compute cost for non-English languages.

**Rare words**: "Transformerization" might become ["Transform", "er", "ization"] — 3 tokens for one word. The model must aggregate these pieces to understand the full concept.

### 7.4 Token Fertility

**Token fertility**: Average number of tokens per word for a given language with a specific tokenizer.

| Language | Approx. fertility (English-trained BPE) |
|---|---|
| English | ~1.1 tokens/word |
| French/German | ~1.5 tokens/word |
| Chinese | ~1.5–2 chars/token |
| Arabic | ~3–4 tokens/word |
| Code (Python) | ~1.3 tokens/token (keywords single) |

**Impact**: A Chinese sentence of 50 characters might use 50+ tokens, while the equivalent English sentence uses 30 tokens. Chinese text is effectively "longer" in token space, using more context window and costing more compute. This is a fairness and efficiency concern for multilingual systems.

---

## 8. Scaling Laws

### 8.1 Kaplan et al. (2020) — OpenAI Scaling Laws

The loss of a language model follows a power law in model size $N$ (parameters), dataset size $D$ (tokens), and compute $C$ (FLOPs):

$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

$$L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

$$L(C) \approx \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050$$

When training with a fixed compute budget $C \approx 6ND$ FLOPs (6 FLOPs per parameter per token for dense transformers), Kaplan et al. found it optimal to scale $N$ much faster than $D$:

$$N_{\text{opt}} \propto C^{0.73}, \quad D_{\text{opt}} \propto C^{0.27}$$

This led to the practice of training very large models on relatively few tokens (GPT-3: 175B params, 300B tokens — only 1.7 tokens per parameter).

### 8.2 Chinchilla (Hoffmann et al., 2022) — Compute-Optimal Training

Hoffmann et al. re-ran the scaling analysis more carefully and found a crucial correction: **$N$ and $D$ should scale equally** with compute:

$$N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}$$

The rule of thumb: **train on approximately 20 tokens per parameter**. Chinchilla (70B params, 1.4T tokens) outperforms Gopher (280B params, 300B tokens) despite using 4× fewer parameters.

**Unified formula for compute-optimal loss**:

$$L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

where $E \approx 1.69$ (irreducible entropy), $\alpha \approx 0.34$, $\beta \approx 0.28$, $A = 406.4$, $B = 410.7$.

Minimizing subject to the constraint $C = 6ND$ yields:

$$N_{\text{opt}} = G \cdot C^{0.5}, \quad D_{\text{opt}} = G^{-1} \cdot C^{0.5}, \quad G = \left(\frac{\alpha A}{\beta B}\right)^{1/(\alpha+\beta)}$$

### 8.3 Over-Training for Inference Efficiency (Llama Philosophy)

Chinchilla gives the compute-optimal training configuration. But **compute-optimal is not inference-optimal**. Once trained, a smaller model costs less per inference token than a larger model. Meta's insight with Llama:

> Train on far more tokens than Chinchilla-optimal. The model will not be the best of its parameter class during training efficiency, but it will be the best small model ever trained — and inference is cheap.

| Model | Parameters | Tokens | Tokens/Param | Notes |
|---|---|---|---|---|
| GPT-3 | 175B | 300B | 1.7 | Under-trained (Kaplan-era) |
| Chinchilla | 70B | 1.4T | 20 | Compute-optimal |
| Llama 1 (7B) | 7B | 1T | 143 | Heavily over-trained |
| Llama 2 (70B) | 70B | 2T | 28.6 | Moderately over-trained |

Llama 2 70B was trained on 2T tokens (Chinchilla says ~1.4T would be optimal). This over-training yields a smaller model that matches the quality of a larger Chinchilla-optimal model, at much lower inference cost.

### 8.4 Emergent Abilities

**Definition**: Capabilities that are not present (or near chance) for smaller models but appear suddenly above a threshold scale (Wei et al., 2022).

Examples: few-shot arithmetic (appears around 10B params), chain-of-thought reasoning (appears around 100B params), instruction following.

**Why do they emerge?** Several hypotheses:
1. **Phase transitions**: The model crosses a threshold where it can compose multiple sub-skills learned separately. Each sub-skill improves smoothly, but the combined task only succeeds when all sub-skills clear a minimum threshold simultaneously.
2. **Measurement artifact**: Some metrics are step functions (e.g., exact match on math problems). Smooth underlying improvement looks like sudden emergence. If you use smoother metrics (e.g., token-level accuracy), improvement is smooth.
3. **Task complexity**: Some tasks require many capabilities simultaneously; all must reach minimum quality before the combined task succeeds.

**Debate**: Schaeffer et al. (2023) argue most emergence is a metric artifact. When using continuous metrics, scaling is smooth. This remains an active area of research.

---

## 9. Inference Optimizations

### 9.1 KV Cache Memory Formula

As derived in Section 4.7:

$$\text{KV Cache} = 2 \times L \times H \times d_k \times n \times \text{bytes\_per\_element}$$

Factor of 2 for K and V. With GQA ($g$ groups): replace $H$ with $g$.

### 9.2 Flash Attention (Dao et al., 2022)

**The problem**: Standard attention materializes the full $n \times n$ attention matrix in GPU HBM (high-bandwidth memory). For large $n$, this dominates runtime and memory — even if compute is fast, memory bandwidth is the bottleneck.

**Flash Attention**: An IO-aware algorithm that computes the exact same attention as standard attention but never materializes the full $n \times n$ matrix.

**Key technique — tiling**: Partition Q, K, V into blocks that fit in SRAM (fast on-chip memory, ~20MB on A100 vs. 40–80GB HBM). Process one block at a time:

```
Algorithm: Flash Attention (simplified)
---------------------------------------
for each block of Q rows (block Q_i):
    initialize output accumulator O_i = 0, normalizer l_i = 0, max m_i = -inf
    for each block of K, V columns (block K_j, V_j):
        load Q_i, K_j, V_j from HBM into SRAM
        compute S_ij = Q_i @ K_j.T / sqrt(d_k)    (partial scores)
        m_new = max(m_i, rowmax(S_ij))             (running max for stability)
        P_ij = exp(S_ij - m_new)                   (shifted exp)
        l_i = exp(m_i - m_new) * l_i + rowsum(P_ij)  (update normalizer)
        O_i = exp(m_i - m_new) * O_i + P_ij @ V_j    (update output)
        m_i = m_new
    O_i = O_i / l_i    (normalize)
    write O_i to HBM
```

**Online softmax trick**: Softmax requires knowing all scores before normalizing. The algorithm maintains running max $m$ and normalizer $l$ that are updated incrementally — this allows computing the exact softmax incrementally as new K, V blocks arrive.

**Complexity**:
- Standard attention: $O(n^2)$ HBM memory, $O(n^2 d)$ HBM reads/writes
- Flash Attention: $O(n)$ HBM memory (no $n \times n$ matrix stored), fewer HBM I/O operations

**Speedup**: 2–4× faster than standard attention on A100, with exactly the same numerical output (up to floating-point rounding order).

**Flash Attention is NOT an approximation** — it is an exact algorithm that exploits the GPU memory hierarchy.

### 9.3 Speculative Decoding (Chen et al., 2023)

**Problem**: Autoregressive generation is sequential — you must generate token $t$ before token $t+1$. Large models are slow because each forward pass is large.

**Idea**: Use a small **draft model** $M_q$ (e.g., 7B) to generate $k$ candidate tokens quickly, then verify all $k$ tokens with one forward pass of the large **target model** $M_p$ (e.g., 70B).

**Algorithm**:
1. Draft model generates $k$ tokens: $\hat{x}_1, \hat{x}_2, \ldots, \hat{x}_k$ with probabilities $q(\hat{x}_i \mid x_{<i})$
2. Target model processes input + all $k$ draft tokens in **one** forward pass, getting $p(\cdot \mid x_{<i})$ for each position
3. Accept draft token $\hat{x}_i$ with probability:

$$\min\!\left(1,\; \frac{p(\hat{x}_i \mid x_{<i})}{q(\hat{x}_i \mid x_{<i})}\right)$$

4. If rejected at position $i$, sample from the adjusted distribution $(p - q)_+ / Z$ and stop
5. If all $k$ accepted, also sample one new token from $M_p$

**Expected speedup**: Expected accepted tokens per target forward pass $= k\alpha + 1$ where $\alpha$ is acceptance rate. With $k=4$, $\alpha=0.8$: $4.2$ tokens per pass vs. $1$ without speculation $\rightarrow$ ~4× speedup.

**Key property**: Speculative decoding produces **exactly the same distribution** as target-only sampling — it is not an approximation. This follows from the rejection sampling identity.

### 9.4 Quantization

**Post-Training Quantization (PTQ)**: Quantize weights after training, no retraining needed.
- **GPTQ**: Second-order quantization (uses Hessian information). INT4, INT8. 4-bit Llama 70B fits on a single A100 80GB.
- **AWQ** (Activation-aware Weight Quantization): Identifies salient weights (large activations) and keeps them in higher precision. More accurate than GPTQ on some tasks.
- **GGUF**: Format used by llama.cpp for CPU/GPU inference. Supports mixed precision.

**Quantization-Aware Training (QAT)**: Simulate quantization noise during training. Better quality than PTQ but requires access to the training pipeline. Used less often for LLMs due to high training cost.

| Precision | Bits | Memory | Quality | Used In |
|---|---|---|---|---|
| FP32 | 32 | 1× | Best | Training (legacy) |
| BF16/FP16 | 16 | 0.5× | Same as FP32 | Training + inference standard |
| INT8 | 8 | 0.25× | Minimal drop | Inference (LLM.int8, SmoothQuant) |
| INT4 | 4 | 0.125× | Small drop | Inference (GPTQ, AWQ, GGUF) |

### 9.5 Continuous Batching and PagedAttention (vLLM)

**Problem with static batching**: Traditional batch inference waits for all sequences in a batch to finish. Sequences of different lengths mean shorter sequences sit idle while the longest finishes — poor GPU utilization.

**Continuous batching** (Orca, Yu et al., 2022): Process requests at the iteration level, not the request level. When one sequence finishes (outputs EOS), immediately replace it with a new request from the queue. Dramatically improves GPU utilization.

**PagedAttention** (Kwon et al., 2023, vLLM): KV cache stored in fixed-size "pages" (like OS virtual memory paging). A block table maps logical positions to physical pages.

```
Without PagedAttention:
  Request A (length 100):  [KV block: 0...99] -- 100 slots allocated upfront
  Request B (length 50):   [KV block: 0...49] -- 50 slots allocated
  Request C (length 200):  [KV block: 0...199] -- 200 slots allocated
  Fragmentation: if B finishes, its memory sits idle until explicitly freed

With PagedAttention:
  Page size = 16 tokens
  Request A: pages [p1, p2, p3, p4, p5, p6, p7]   (7 pages allocated on demand)
  Request B: pages [p8, p9, p10]                    (3 pages)
  Request C: pages [p11..p22]                       (12 pages)
  When B finishes, p8, p9, p10 immediately available to new requests
  Prefix caching: shared system prompt -> shared physical pages
```

**Advantages**:
- Near-zero KV cache fragmentation
- Memory sharing for common prefixes (system prompts)
- Near-100% memory utilization vs. 20–40% with static allocation

### 9.6 Problems with Each Optimization

| Optimization | Problem | Mitigation |
|---|---|---|
| KV cache | Memory grows linearly with context length | GQA, MQA, quantized KV cache, StreamingLLM eviction |
| Flash Attention | Requires custom CUDA kernels, not straightforward to implement | Available in PyTorch 2.0 as `F.scaled_dot_product_attention`; Triton implementation |
| Speculative decoding | Draft model must match target domain; two models in memory | Use same tokenizer and similar architecture; quality degrades if draft is too different |
| Quantization | Accuracy loss especially for INT4; not all operations quantize well | Mixed precision: sensitive layers in FP16; calibration data selection |
| Continuous batching | Increased system complexity | vLLM, TGI handle this transparently behind API |

---

## 10. Problems & Mitigations (Dedicated Section)

### 10.1 Hallucination

**Definition**: The model generates plausible-sounding but factually incorrect content, stated with apparent confidence.

**Root causes**:
- CLM training maximizes likelihood of training data, including incorrect data
- Model conflates "fluent text" with "true text"
- Rare facts are underrepresented; model distributes probability mass to common-but-wrong alternatives
- No grounding mechanism — pure parametric memory

**Mitigations**:
- **RLHF**: Human raters penalize hallucinations; PPO updates the model to avoid them
- **RAG (Retrieval-Augmented Generation)**: Retrieve relevant documents and include them in context. The model generates from retrieved facts, not pure memory
- **Chain-of-Thought**: Forces the model to commit to intermediate reasoning steps that can be checked
- **Calibration**: Train the model to express uncertainty (requires uncertainty-annotated training data)
- **Factuality fine-tuning**: Fine-tune on verified factual data with rewards for verifiable claims

### 10.2 Context Window Limitations

**Problem**: Transformers have a fixed maximum context length. Scaling is $O(n^2)$ in memory and compute.

**Approaches**:
- **RoPE NTK scaling**: Scale the base frequency $\theta$ by a factor proportional to the context extension ratio. Reduces perplexity degradation significantly for 2–4× extension without fine-tuning.
- **YaRN** (Peng et al., 2023): More principled interpolation strategy that separately handles different frequency components of RoPE. Extends Llama 2 to 128K tokens with minimal quality degradation.
- **Sliding window attention** (Mistral): Each position attends only to the last $W$ tokens (e.g., $W=4096$). Linear memory, but cannot attend to distant tokens.
- **Memory augmentation**: External memory systems store past hidden states; retrieved via k-NN at each step.

### 10.3 Quadratic Attention Complexity

**Problem**: $O(n^2)$ time and memory in sequence length.

**Approaches**:
- **Sparse attention** (Longformer, BigBird): Combine local window attention ($O(n \cdot w)$ for window $w$) with global attention for special tokens
- **Linear attention** (Performer, Katharopoulos et al.): Approximate softmax attention using random feature maps, achieving $O(nd)$ complexity. Quality tradeoff varies
- **Flash Attention**: Does NOT reduce complexity but dramatically reduces memory usage and HBM reads via IO-aware tiling. Still $O(n^2)$ compute
- **State Space Models** (Mamba, S4): Replace attention with recurrent state-space formulation. $O(n)$ complexity with different inductive biases

### 10.4 KV Cache Memory Pressure

**Problem**: KV cache grows as $O(L \times H \times d_k \times n)$. For large batches and long contexts, this exhausts GPU memory before compute is saturated.

**Mitigations**:
- **GQA/MQA**: Reduce number of K, V heads (8× reduction for GQA with 8 groups vs. 64 heads)
- **KV cache quantization**: Quantize cached K, V to INT8 or INT4 — 2–4× memory reduction with small quality impact
- **KV eviction**: Selectively drop old or low-attention cache entries. H2O: evict tokens with lowest accumulated attention. StreamingLLM: always keep attention sink tokens + recent window
- **PagedAttention** (vLLM): Eliminates fragmentation, enables near-100% utilization

### 10.5 Exposure Bias in Autoregressive Training

**Definition**: During training, the model is conditioned on ground-truth previous tokens. At inference, it is conditioned on its own generated tokens. If it makes one mistake, the error compounds.

**Mitigations**:
- **Scheduled sampling**: With probability $\epsilon$ (annealed during training), replace ground-truth input with the model's own prediction. Trains recovery from errors
- **RLHF/DPO**: Directly optimize on the model's own outputs, removing the train-test mismatch entirely

### 10.6 Position Extrapolation

**Problem**: Transformers trained on length $L$ perform poorly at length $> L$ due to out-of-distribution position encodings.

**Mitigations**:
- **RoPE NTK-aware scaling**: Set base $\theta' = \theta \cdot (s)^{d_k/(d_k-2)}$ where $s$ is the scale factor. Keeps high-frequency dimensions unchanged while scaling low-frequency dimensions
- **YaRN**: Additionally applies temperature scaling to attention scores ($1/\sqrt{t}$ factor) to compensate for attention entropy changes at longer contexts
- **LongRoPE**: Non-uniform scaling for each RoPE dimension found by search. Claims to extend to 2M context
- **Fine-tuning on longer sequences**: Even a small amount of fine-tuning on longer examples (1000–10000 examples) significantly improves extrapolation

---

## 11. Industry Practices at Binance Scale

### 11.1 How Crypto Exchanges Use LLMs

**Market Sentiment Analysis**:
- Ingest social media (Twitter/X, Telegram, Reddit) and news articles in real time
- Fine-tuned encoder models (FinBERT, domain-adapted RoBERTa) classify sentiment per asset
- Sentiment signals feed into trading strategies or risk alerts
- Decoder models generate natural language summaries of market conditions for traders
- Named Entity Recognition to identify specific coins, projects, people mentioned

**Customer Support Chatbots**:
- RAG-based systems: retrieval over FAQ, help articles, and transaction history
- Instruction-tuned LLMs (e.g., fine-tuned Llama) handle intent classification, account queries
- Critical: strict hallucination mitigation — cannot give wrong financial information
- Use retrieval + explicit "I cannot confirm" fallback for anything account-specific

**Smart Contract Analysis**:
- Code LLMs (DeepSeek Coder, CodeLlama) for vulnerability detection in Solidity/Rust
- Automated auditing pipeline: submit contract → LLM identifies common patterns (reentrancy, integer overflow) → human review
- Classification models for scam/rug-pull contract detection

**Compliance and KYC**:
- NER models extract entities from documents
- Classification models flag suspicious activity descriptions
- Multilingual models handle global user base — SentencePiece-based tokenizers that handle CJK, Arabic

### 11.2 Production Serving

**vLLM**: Open-source LLM serving framework.
- PagedAttention for KV cache management
- Continuous batching
- Tensor parallelism across multiple GPUs
- Target: >50% GPU utilization vs. ~20% with naive serving

**TensorRT-LLM** (NVIDIA): Compiled inference engine.
- Fuses operations, applies INT8/INT4 quantization
- Generates highly optimized CUDA kernels
- Used for latency-critical serving (P99 latency < 500ms)

**TGI (Text Generation Inference, Hugging Face)**: Production-ready serving with Flash Attention, continuous batching, quantization support.

**Serving architecture at scale**:
```
                         Incoming Requests
                               |
                        Load Balancer
                               |
               Router (model selection, request priority)
                               |
             +-----------------+-----------------+
             |                 |                 |
         vLLM Pod          vLLM Pod          vLLM Pod
         (4x A100)         (4x A100)         (4x A100)
         7B model          70B model         70B model
             |                 |                 |
             +-----------------+-----------------+
                               |
                     Prefix Cache Layer
                     (KV states for common system prompts)
                               |
                     Response + Latency Monitoring
                     (Prometheus, Grafana dashboards)
```

### 11.3 Cost Optimization

**Model quantization**: Deploy INT4 or INT8 quantized models (GPTQ, AWQ). A 70B INT4 model runs on 2× A100 40GB vs. 4× A100 for FP16 — 2× GPU cost reduction.

**Speculative decoding**: Draft model (7B) + target model (70B). Net speedup 2–3×, same output quality. Reduces GPU-hours per 1M tokens by roughly 50%.

**Prefix caching**: If many requests share the same system prompt, cache the KV states for the prefix. Subsequent requests skip prompt processing entirely. Common for instruction-preamble in chatbots.

**Model selection routing**: Use a small classifier to route simple queries to a 7B model and complex queries to 70B. 80% of queries can often be handled by the smaller model at ~10× lower cost.

**Batching efficiency**: Continuous batching ensures no GPU time is wasted between requests. PagedAttention maximizes effective batch size.

### 11.4 Safety for Financial Context

**RLHF**: Align the model to refuse giving specific investment advice, to add disclaimers, to refer users to licensed advisors. Human raters trained on financial compliance guidelines.

**Constitutional AI** (Anthropic approach): Define constitutional principles ("Do not give personalized financial advice," "Do not make price predictions"), have the model critique its own outputs against these principles before responding.

**Output filtering**: Rule-based post-processing to catch outputs that contain specific patterns — price predictions presented as fact, guaranteed return statements, specific investment recommendations.

**Adversarial robustness**: Prompt injection attacks are a real threat in financial chatbots. Implement:
- Input sanitization
- Prompt structure isolation (system prompt cannot be overridden by user input via careful delimiters)
- Output monitoring for policy violations

---

## 12. Interview Q&A

### Basic Questions (5)

---

**Q1. What is the attention mechanism and why was it introduced?**

**A**: The attention mechanism was introduced to address the fixed-size bottleneck in seq2seq RNN models, where the entire input sequence had to be compressed into a single vector. For long sequences, this compression loses critical information.

Attention allows the decoder to directly access all encoder hidden states when generating each output token, computing a weighted average of encoder states where the weights reflect relevance. This is computed via:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

The model learns which positions to attend to via gradient descent — no manual feature engineering. The Transformer (Vaswani et al., 2017) extended this to replace recurrence entirely, using self-attention where queries, keys, and values all come from the same sequence.

---

**Q2. Why do we scale attention scores by $\frac{1}{\sqrt{d_k}}$ before softmax?**

**A**: Without scaling, the dot products $QK^T$ grow in magnitude as $d_k$ increases. If $q_i, k_i \sim \mathcal{N}(0,1)$ are independent:

$$\text{Var}\!\left(\sum_{i=1}^{d_k} q_i k_i\right) = d_k$$

With $d_k = 64$, the standard deviation is 8. Large scores push softmax into a near-one-hot regime where gradients vanish (softmax saturates — one weight $\approx 1$, rest $\approx 0$, derivative effectively zero). Dividing by $\sqrt{d_k}$ normalizes variance back to 1, keeping the softmax in a well-conditioned, gradient-friendly regime.

---

**Q3. What is the difference between encoder-only, decoder-only, and encoder-decoder transformers? When would you use each?**

**A**:
- **Encoder-only (BERT)**: Bidirectional attention — each token sees all other tokens. Best for understanding: classification, NER, semantic embeddings, span extraction. Cannot generate text.
- **Decoder-only (GPT, Llama)**: Causal attention — each token only sees past tokens. Natural for text generation, chat, code. Enables in-context learning at scale. Dominates modern LLMs.
- **Encoder-Decoder (T5, BART)**: Encoder reads input bidirectionally; decoder generates autoregressively with cross-attention to encoder. Best for seq2seq: translation, summarization.

Decoder-only dominates today because it scales naturally, emerges strong in-context learning, and works seamlessly with instruction tuning and RLHF.

---

**Q4. What is the KV cache and why is it important for inference?**

**A**: During autoregressive generation, computing token $t$ requires keys and values for all past tokens $1, \ldots, t-1$. Without caching, you would recompute all past K, V at every step — $O(n^2)$ total compute for a sequence of length $n$.

The KV cache stores K, V for all previously seen tokens. When generating token $t+1$, only compute Q, K, V for the new token, then concatenate new K, V to the cache and run attention against the full cached history.

This reduces generation to $O(n)$ total compute. The tradeoff is memory: cache size is $2 \times L \times H \times d_k \times n$ elements — grows linearly with sequence length. GQA/MQA reduce this by sharing K, V across heads (8× reduction for Llama 2 70B with GQA).

---

**Q5. How does BPE tokenization work and what are its limitations?**

**A**: Byte-Pair Encoding starts with a character vocabulary and iteratively merges the most frequent adjacent pair until the vocabulary reaches a target size. This creates a subword vocabulary where common words become single tokens and rare words are split into frequent subword pieces.

Limitations:
1. **Numbers**: "12345" becomes ["12", "34", "5"] or similar — arithmetic is difficult.
2. **Multilingual imbalance**: English-trained BPE assigns 1 token to most English words but 1 token per character for many CJK characters — higher token fertility, longer sequences, more compute.
3. **Rare words**: Uncommon words are split into many subword pieces, harder to learn as units.
4. **Fixed vocabulary**: Cannot natively handle new words without splitting them.

---

### Intermediate Questions (5)

---

**Q6. Derive why the total parameter count of Multi-Head Attention is $4d_{\text{model}}^2$.**

**A**: For $h$ attention heads, each head has per-head projections:

- $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$ for $i = 1, \ldots, h$: total $h \times d_{\text{model}} \times d_k = d_{\text{model}} \times (h \cdot d_k)$. Since $d_k = d_{\text{model}}/h$, we get $d_{\text{model}} \times d_{\text{model}} = d_{\text{model}}^2$.
- Similarly $W_i^K$: $d_{\text{model}}^2$ total.
- Similarly $W_i^V$: $d_{\text{model}}^2$ total.
- Output projection $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$: since $h \cdot d_v = d_{\text{model}}$, this is $d_{\text{model}} \times d_{\text{model}} = d_{\text{model}}^2$.

Total (ignoring biases): $4d_{\text{model}}^2$.

---

**Q7. Explain RoPE mathematically and prove that it achieves relative position encoding.**

**A**: RoPE encodes position by rotating query and key vectors. For position $m$, the $j$-th 2D subspace of $q$ is rotated by angle $m\theta_j$ ($\theta_j = 10000^{-2j/d_k}$):

$$R_m^{(j)} = \begin{pmatrix} \cos(m\theta_j) & -\sin(m\theta_j) \\ \sin(m\theta_j) & \cos(m\theta_j) \end{pmatrix}$$

The full rotation is $R_m = \text{diag}(R_m^{(1)}, \ldots, R_m^{(d_k/2)})$.

**Proof of relative encoding**: For query at position $m$ and key at position $n$:

$$\tilde{q}_m^T \tilde{k}_n = (R_m q_m)^T (R_n k_n) = q_m^T R_m^T R_n k_n$$

Rotation matrices are orthogonal and compose: $R_m^T R_n = R_{-m} R_n = R_{n-m}$. Therefore:

$$\tilde{q}_m^T \tilde{k}_n = q_m^T R_{n-m} k_n$$

The dot product depends only on the relative position $n - m$, not on absolute positions. This is exactly the property we want: attention scores should be a function of how far apart tokens are.

---

**Q8. What is Flash Attention and how does it achieve its speedup? Is it an approximation?**

**A**: Flash Attention (Dao et al., 2022) is an IO-aware **exact** attention algorithm. It achieves speedup not by reducing FLOPs but by reducing HBM (GPU high-bandwidth memory) accesses, which is the actual bottleneck for large $n$.

Standard attention materializes the full $n \times n$ attention matrix in HBM — reading and writing $O(n^2)$ data. Flash Attention tiles the computation into blocks that fit in fast SRAM (on-chip memory). For each Q block, it iterates over K, V blocks, computing partial attention using the **online softmax trick** — maintaining running max and normalizer to compute softmax incrementally without ever storing the full $n \times n$ matrix.

HBM memory: $O(n)$ vs. $O(n^2)$. Speedup: 2–4× on A100.

Flash Attention is **not an approximation** — it produces the exact same output as standard attention up to floating-point rounding order.

---

**Q9. Explain the Chinchilla scaling law. What are its practical implications for LLM training and deployment?**

**A**: Chinchilla (Hoffmann et al., 2022) corrected the Kaplan et al. scaling analysis by showing that model size $N$ and data $D$ should scale equally with compute $C$: $N_{\text{opt}} \propto C^{0.5}$, $D_{\text{opt}} \propto C^{0.5}$. The practical rule is ~20 tokens per parameter for compute-optimal training.

This showed that GPT-3 (175B params, 300B tokens = 1.7 tokens/param) was massively under-trained. Chinchilla (70B params, 1.4T tokens) matched Gopher (280B) with 4× fewer parameters.

**Practical implications**:
1. **Smaller models with more data** are compute-optimal. The field shifted toward this.
2. **Inference cost drives "over-training"**: Llama trains 7B–70B models on 1–2T tokens (well beyond Chinchilla-optimal) because once deployed, the smaller model pays off in inference savings. A 7B model costs ~10× less to serve than a 70B model.
3. **Data quality matters more at smaller $N$**: With fewer parameters, each training token must count more.

---

**Q10. What is speculative decoding and why does it preserve the target distribution exactly?**

**A**: Speculative decoding uses a small fast draft model $M_q$ to generate $k$ candidate tokens, then verifies them all with one forward pass of the large target model $M_p$.

The verification uses rejection sampling: draft token $\hat{x}$ (with probability $q(\hat{x})$) is accepted with probability $\min(1, p(\hat{x})/q(\hat{x}))$. When rejected, sample from $(p - q)_+ / Z$.

**Why it preserves the target distribution**: This is a direct application of the rejection sampling theorem. The accepted tokens (kept with probability $p/q$, drawn from distribution $q$) follow distribution $\propto q \cdot (p/q) = p$. The rejected replacement (drawn from $(p-q)_+$) fills in the remainder. Together, the marginal distribution of each generated token is exactly $p$ — identical to sampling from the target model alone.

Expected speedup: $k\alpha + 1$ tokens per target forward pass, where $\alpha$ is acceptance rate. With $k=4$, $\alpha=0.8$: 4.2 tokens per pass vs. 1 without speculation.

---

### Advanced Questions (5)

---

**Q11. Walk through the mathematical argument for why pre-norm transformers are more stable than post-norm, especially at initialization.**

**A**: Consider $L$ layers. In post-norm:

$$x_{l+1} = \text{LN}(x_l + F_l(x_l))$$

The gradient from loss to $x_0$:

$$\frac{\partial \mathcal{L}}{\partial x_0} = \prod_{l=1}^{L} \frac{\partial \text{LN}}{\partial (x_l + F_l)} \cdot \left(I + \frac{\partial F_l}{\partial x_l}\right)$$

At initialization, sublayer weights are small (Xavier/Kaiming init), so $\frac{\partial F_l}{\partial x_l} \approx 0$. The gradient reduces to a product of $L$ LayerNorm Jacobians:

$$\frac{\partial \mathcal{L}}{\partial x_0} \approx \prod_{l=1}^{L} \frac{\partial \text{LN}_l}{\partial x_l}$$

LayerNorm's Jacobian has singular values $\leq 1$. Multiplying $L$ such matrices: exponential gradient vanishing at initialization. This makes training unstable without careful warmup.

In pre-norm:

$$x_{l+1} = x_l + F_l(\text{LN}(x_l))$$

$$\frac{\partial \mathcal{L}}{\partial x_l} = \underbrace{\frac{\partial \mathcal{L}}{\partial x_{l+1}}}_{\text{direct path}} + \underbrace{\frac{\partial \mathcal{L}}{\partial x_{l+1}} \cdot \frac{\partial F_l(\text{LN}(x_l))}{\partial x_l}}_{\text{sublayer path}}$$

The identity (direct path) always propagates the gradient at full strength, regardless of the sublayer Jacobian. At initialization, the sublayer path $\approx 0$, but the direct path carries the full gradient. This is stable for arbitrary depth at initialization and throughout training.

---

**Q12. Derive the GQA memory savings formula and explain its impact on serving throughput.**

**A**: In standard MHA with $H$ heads, $d_k$ per head, $L$ layers, FP16, sequence length $n$:

$$\text{KV Cache}_{\text{MHA}} = 2 \times L \times H \times d_k \times n \times 2\ \text{bytes}$$

With GQA using $G$ KV head groups ($G \ll H$):

$$\text{KV Cache}_{\text{GQA}} = 2 \times L \times G \times d_k \times n \times 2\ \text{bytes}$$

Memory reduction factor: $H/G$.

**Impact on serving throughput**: GPU memory is shared between model weights $M_w$, activations, and KV cache. The maximum feasible batch size $\times$ context length is bounded by:

$$B \times n \leq \frac{M_{\text{GPU}} - M_w}{4 L G d_k}$$

With GQA ($G = H/8$): the KV budget is $8\times$ larger. This means either:
- $8\times$ longer contexts at the same batch size, or
- $8\times$ larger batch sizes at same context — $8\times$ more throughput (up to GPU compute saturation)

For Llama 2 70B on 8× A100 80GB: MHA ($H=64$) would allow ~2K context at batch 4; GQA ($G=8$) allows ~16K context at batch 4 or batch 32 at 2K context — directly translating to throughput and cost per token.

---

**Q13. Why do emergent abilities appear at scale, and what is the debate around whether they are truly emergent?**

**A**: **The phase transition argument**: Some tasks require combining multiple sub-skills. Multi-step arithmetic requires: digit recognition, carry operations, positional tracking, and final assembly. Each sub-skill improves smoothly with scale. But the combined task only succeeds when ALL sub-skills exceed a minimum competence threshold simultaneously. Below that threshold, any chain failure causes the full task to fail. Above it, the chain succeeds. Smooth underlying improvement appears as a step function at the task level.

**The measurement artifact argument** (Schaeffer et al., 2023): Most "emergent" benchmarks use discontinuous metrics — exact match, pass/fail. Smooth underlying logit improvement maps to a step function in the binary metric. If you plot continuous metrics (e.g., token-level accuracy on arithmetic, or the probability assigned to the correct answer), the improvement is smooth, not sudden.

**Concrete test**: Take a math benchmark where "emergence" is reported at 100B params using exact-match accuracy. If you instead plot the probability assigned to the correct answer vs. model size on a log-log scale, the relationship is often a smooth power law from 1B to 100B+. The apparent emergence was a metric artifact.

**Current consensus**: Both arguments have merit. Some behaviors (in-context learning as a qualitative capability) appear genuinely novel at scale. Others (exact-match arithmetic) are likely metric artifacts. The distinction matters for forecasting what future models can do.

---

**Q14. Explain the full pretraining objective of BERT (MLM + NSP) and why the community abandoned NSP.**

**A**: BERT uses two pretraining objectives:

**MLM**: 15% of input tokens are selected. Of these: 80% replaced by `[MASK]`, 10% by a random token, 10% kept unchanged. Loss computed only at masked positions:

$$\mathcal{L}_{\text{MLM}} = -\sum_{i \in \text{masked}} \log P(x_i \mid \tilde{x})$$

The 80/10/10 scheme avoids train-test mismatch (no `[MASK]` at inference) and forces the model to build good representations for all tokens, not just masked ones.

**NSP (Next Sentence Prediction)**: Given two segments A and B, predict whether B actually follows A in the document (50% positive, 50% random negative from a different document). The `[CLS]` token is used:

$$\mathcal{L}_{\text{NSP}} = -\log P(\text{IsNext} \mid [\text{CLS}])$$

**Why NSP was abandoned (RoBERTa, 2019)**: Liu et al. showed that NSP is trivially solvable by detecting topic shift — the random negative examples come from different documents with obviously different topics. The model learns topic matching rather than discourse coherence. When NSP was removed and training was continued with more data and longer sequences (RoBERTa), performance improved on all downstream tasks. NSP was providing misleading gradient signal, consuming compute that would be better spent on MLM, and potentially hurting representation quality.

---

**Q15. How would you design an LLM-based crypto market sentiment system at production scale for Binance? Address model choice, latency, cost, and accuracy.**

**A**: **System Design**:

**Data ingestion pipeline**:
- Real-time streams: Twitter API, Telegram crypto channels, Reddit API, RSS feeds from CoinDesk/CoinTelegraph
- On-chain event summaries (large transfers, liquidations, whale movements)
- Structured metadata: timestamp, source credibility score, asset mentions

**Two-tier model architecture**:

*Tier 1 — Fast path* (high throughput, low latency): Fine-tuned FinBERT or custom DeBERTa-v3-base (180M params) for per-document sentiment classification. Outputs: {positive, negative, neutral, uncertain} per named entity (BTC, ETH, BNB, ...). Latency: <10ms GPU, ~50ms CPU. Throughput: ~50K docs/second.

*Tier 2 — Slow path* (complex analysis): Llama 3 8B or 70B (INT4 quantized) for nuanced analysis of longer documents — regulatory filings, technical analysis threads, project updates. Generates structured JSON summaries. Latency: 200–500ms.

**Routing logic**:
- Length < 280 chars (tweet), confidence > 0.8 on Tier 1: serve from Tier 1
- Length > 500 chars or Tier 1 confidence < 0.6: route to Tier 2
- ~85% traffic handled by Tier 1

**Serving infrastructure**:
```
Kafka stream (raw docs)
    |
Preprocessor (entity extraction, deduplication)
    |
Router (length + confidence classifier)
    |               |
Tier 1 service   Tier 2 service
(FastAPI + ONNX  (vLLM, PagedAttention,
 RT, 8x A10G)    4x A100 80GB INT4,
                 continuous batching)
    |               |
Aggregator (entity-level sentiment scores)
    |
Redis cache (aggregated signals, TTL=60s)
    |
Trading strategy API + Dashboard
```

**Cost optimization**:
- Tier 1 handles 85% of traffic at ~50× cheaper per token than Tier 2
- INT4 quantization on Tier 2: 4× memory reduction, 2× throughput
- Prefix caching for standard analysis preambles (~30% cache hit rate)

**Accuracy**:
- Fine-tune Tier 1 on labeled crypto-specific dataset: 50K manually labeled examples + data augmentation
- Entity-level F1 target: >85% on held-out crypto test set
- Calibration evaluation: reliability diagrams, expected calibration error < 0.05

**Safety and compliance**:
- Outputs are informational sentiment signals, never investment advice
- Explicit disclaimers at API level
- Output filtering for price predictions framed as recommendations
- Regular adversarial red-teaming (prompt injection attempts on Tier 2)
- Audit logging of all LLM outputs for compliance review

---

## Summary Quick Reference

| Topic | Key Formula / Concept | Used In |
|---|---|---|
| Scaled Dot-Product Attention | $\text{softmax}(QK^T/\sqrt{d_k})V$ | All transformers |
| Why scale by $1/\sqrt{d_k}$ | $\text{Var}(q \cdot k) = d_k$ without scaling | Vaswani et al. 2017 |
| MHA Parameter Count | $4d_{\text{model}}^2$ per MHA block | All transformers |
| RoPE relative position | $\tilde{q}_m^T\tilde{k}_n = q_m^T R_{n-m} k_n$ | Llama, Mistral, Qwen |
| Chinchilla rule of thumb | $D_{\text{opt}} \approx 20N$ | LLM training |
| KV Cache Memory | $2LHd_k n \times \text{bytes}$ | All autoregressive inference |
| Flash Attention | Tiling + online softmax, $O(n)$ HBM memory | All modern serving |
| Speculative Decoding | Acceptance probability $= \min(1, p/q)$ | Fast inference |
| RMSNorm | $x / \text{RMS}(x) \times \gamma$ | Llama, Qwen, Mistral |
| CLM Objective | $-\sum_i \log P(x_i \mid x_{<i})$ | GPT, Llama, all frontier LLMs |
| MLM Objective | 80/10/10 masking, predict masked tokens | BERT, RoBERTa |
| GQA memory savings | Replace $H$ heads with $G$ groups: $H/G$ reduction | Llama 2 70B, Mistral 7B |
| Pre-norm gradient | Direct residual path always carries full gradient | GPT-2+, all modern LLMs |
| BPE | Iterative frequency-based pair merging | GPT, Llama |
| ALiBi | Score $-= m \cdot |i-j|$ per head | BLOOM, MPT |

---

*Technical reference for Binance interview preparation. All equations are derived, not stated.*

*Key references: "Attention Is All You Need" (Vaswani et al., 2017), "BERT" (Devlin et al., 2019), "Language Models are Few-Shot Learners" (Brown et al., 2020), "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022), "FlashAttention" (Dao et al., 2022), "GQA" (Ainslie et al., 2023), "Llama 2" (Touvron et al., 2023), "RoPE" (Su et al., 2021), "YaRN" (Peng et al., 2023).*
