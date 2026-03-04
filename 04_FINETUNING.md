# LLM Fine-tuning: A Comprehensive Technical Guide

> Lecture notes for ST5230 / Binance ML Interview Preparation
> Assumes familiarity with Transformers, attention mechanisms, and basic ML training loops.

---

## Table of Contents

1. [Why Fine-tune?](#1-why-fine-tune)
2. [Full Fine-tuning](#2-full-fine-tuning)
3. [Adapter Methods](#3-adapter-methods)
4. [LoRA (Low-Rank Adaptation)](#4-lora-low-rank-adaptation)
5. [QLoRA](#5-qlora)
6. [Prefix Tuning and Prompt Tuning](#6-prefix-tuning-and-prompt-tuning)
7. [Instruction Tuning](#7-instruction-tuning)
8. [RLHF](#8-rlhf-reinforcement-learning-from-human-feedback)
9. [DPO (Direct Preference Optimization)](#9-dpo-direct-preference-optimization)
10. [Fine-tuning for Embeddings](#10-fine-tuning-for-embeddings)
11. [Evaluation](#11-evaluation)
12. [Problems & Mitigations](#12-problems--mitigations)
13. [Industry Practices](#13-industry-practices)
14. [Interview Q&A](#14-interview-qa)
15. [Coding Problems](#15-coding-problems)

---

## 1. Why Fine-tune?

### 1.1 The Pretraining vs Fine-tuning Conceptual Split

Modern LLMs are trained in two distinct phases that serve fundamentally different purposes.

**Pretraining** is unsupervised learning on massive corpora (trillions of tokens). The objective is typically next-token prediction:

$$\mathcal{L}_{PT} = -\sum_{t=1}^{T} \log P_\theta(x_t | x_{<t})$$

The model learns:
- Grammar and syntax of language
- World knowledge and facts
- Reasoning patterns and common sense
- Representations of concepts

Pretraining is extremely expensive — GPT-3 cost roughly $4.6M in compute. The resulting model is a *foundation model*: a strong prior over language but not aligned to any specific task.

**Fine-tuning** adapts this foundation model to a specific task or behavior using a (usually much smaller) labeled or curated dataset. The key insight is that the model already "knows" how to do most things — fine-tuning just teaches it *when* and *how* to apply that knowledge.

```
PRETRAINING                          FINE-TUNING
============                         ===========
Web data (CommonCrawl)               Task-specific data
Books, Wikipedia, Code               (instructions, labels, preferences)
          |                                     |
          v                                     v
  [Random Init]                       [Pretrained Weights]
          |                                     |
     Months of                           Hours/Days of
     compute                             compute
          |                                     |
          v                                     v
  Foundation Model                  Task-Adapted Model
  (knows language,                  (follows instructions,
   facts, reasoning)                 classifies, retrieves, etc.)
```

### 1.2 Task-Specific Adaptation

Fine-tuning covers a spectrum of downstream tasks:

| Task Type | Example | Label Type |
|-----------|---------|------------|
| Classification | Sentiment analysis, NLI | Hard labels (0/1/2) |
| Sequence labeling | NER, POS tagging | Per-token labels |
| Generation | Summarization, translation | Target sequences |
| Instruction following | Chatbots, assistants | (instruction, response) pairs |
| Preference alignment | RLHF, DPO | (prompt, chosen, rejected) triples |
| Retrieval | Semantic search | (query, positive, negative) triples |

For **classification**, a linear head is added on top of the `[CLS]` token:

$$\hat{y} = \text{softmax}(W_c \cdot h_{[CLS]} + b_c)$$

For **generation**, the same language modeling head is used but trained on target sequences with teacher forcing.

### 1.3 When NOT to Fine-tune

Fine-tuning is not always the right answer. Consider alternatives first:

**Use prompting/few-shot when:**
- Data is scarce (< 100 labeled examples)
- Task is straightforward and well within model capabilities
- You need rapid prototyping without training infrastructure
- The model is accessed via API (no weight access)
- GPT-4 / Claude already handles the task well with a good system prompt

**Use RAG (Retrieval-Augmented Generation) when:**
- The task requires up-to-date or proprietary knowledge
- Facts change frequently (crypto prices, regulatory updates)
- You need attribution/citations
- The knowledge base is too large to fit in context

**Use fine-tuning when:**
- Style/format consistency is critical (e.g., always output JSON)
- Task requires domain-specific reasoning not in pretraining data
- You need to reduce token costs (smaller fine-tuned model > larger prompted model)
- You have > 1000 high-quality labeled examples
- Latency is critical and system prompts are expensive

### 1.4 The Three Regimes

```
PARAMETER EFFICIENCY vs PERFORMANCE TRADEOFF
=============================================

Performance
    ^
    |    Full Fine-tune (FFT)  xxxxxxxxx (best but expensive)
    |                      xxxx
    |    LoRA / PEFT     xxx
    |                  xx
    |    Prompt Tuning x
    |                x
    |   Few-shot   x
    |             x
    +----------------------------> Compute / # Parameters Trained
```

1. **Full Fine-tuning (FFT):** Update all parameters. Best performance, highest cost.
2. **Parameter-Efficient Fine-tuning (PEFT):** Freeze base model, train small adapter modules. Near-FFT performance with 0.1-1% of trainable parameters.
3. **Prompting:** Zero or few-shot. No training. Fast but limited.

---

## 2. Full Fine-tuning

### 2.1 How It Works

In full fine-tuning, all $N$ parameters $\theta$ of the pretrained model are updated with gradient descent on the task-specific loss:

$$\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{task}}(\theta)$$

For a classification task with cross-entropy loss:

$$\mathcal{L}_{\text{task}} = -\frac{1}{N}\sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log \hat{y}_{ic}$$

The entire forward and backward pass flows through all transformer layers, updating every weight matrix — embedding layers, attention projections, FFN layers, layer norms.

### 2.2 Catastrophic Forgetting

**Definition:** When a neural network is trained on new task B after task A, it "forgets" how to perform task A. The gradients from task B's loss surface push the weights away from the minimum found for task A.

**Why it happens — gradient interference:**

The gradient of the new loss $\mathcal{L}_{new}$ has no knowledge of curvature in the old loss $\mathcal{L}_{old}$:

$$\mathcal{L}_{\text{new}} \text{ gradient pushes weights away from } \mathcal{L}_{\text{old}} \text{ minimum}$$

Geometrically, the loss landscapes of task A and task B have different minima. Gradient descent on task B's landscape walks the weights to a region that is high-loss for task A.

```
LOSS LANDSCAPE — CATASTROPHIC FORGETTING
=========================================

Loss
  ^
  |  Task A minimum           Task B minimum
  |       *                         *
  |      / \                       / \
  |     /   \                     /   \
  |    /     \   <-- weights -->  \   /
  |   /       \  move this way     \ /
  +--------------------------------------------> Weight space

After training on B: weights at Task B minimum, far from Task A minimum.
Catastrophic forgetting has occurred.
```

**Problems & Mitigations:** See Section 12.

### 2.3 Elastic Weight Consolidation (EWC)

EWC is a continual learning approach that slows down learning for parameters important to the previous task. Importance is measured using the **Fisher Information Matrix**:

$$F_i = \mathbb{E}\left[\left(\frac{\partial \log p(y|x, \theta)}{\partial \theta_i}\right)^2\right]$$

The Fisher diagonal $F_i$ approximates the curvature of the loss with respect to parameter $\theta_i$. High $F_i$ means parameter $i$ is crucial to the old task — penalize moving it.

The EWC regularized loss:

$$\mathcal{L}_{EWC} = \mathcal{L}_{\text{new}} + \frac{\lambda}{2} \sum_i F_i(\theta_i - \theta_{i}^*)^2$$

where $\theta^*$ are the parameters after task A, and $\lambda$ controls the regularization strength.

**Intuition:** The penalty term acts like a spring anchoring each parameter to its old value, with the spring stiffness proportional to how important that parameter was. Unimportant parameters can move freely; important ones resist change.

**Limitation:** Computing the full Fisher matrix is $O(N^2)$ — intractable for billion-parameter models. Diagonal approximation is used but misses parameter interactions.

### 2.4 Learning Rate Considerations

Fine-tuning requires much smaller learning rates than pretraining:

| Phase | Typical LR | Rationale |
|-------|-----------|-----------|
| Pretraining | $3 \times 10^{-4}$ to $1 \times 10^{-3}$ | Random init, need large steps |
| Full fine-tune | $1 \times 10^{-5}$ to $5 \times 10^{-5}$ | Pretrained weights, small perturbations |
| LoRA fine-tune | $1 \times 10^{-4}$ to $3 \times 10^{-4}$ | Only adapter params, can be larger |

**Why smaller LR for fine-tuning?** The pretrained weights encode millions of learned features. A large LR would destroy these representations in early updates before the optimizer "knows" what to preserve. Small LR allows gradual refinement.

**Layer-wise LR decay** is often used — deeper (earlier) layers receive smaller LRs since they encode more general features:

$$\eta_l = \eta_{base} \times \alpha^{L - l}$$

where $L$ is total layers, $l$ is current layer, and $\alpha \in [0.8, 0.95]$ is the decay factor.

### 2.5 Multi-task Fine-tuning

Train simultaneously on multiple tasks by mixing their datasets:

$$\mathcal{L}_{multi} = \sum_{t=1}^{T} w_t \mathcal{L}_t$$

**Advantages:**
- Tasks can share representations, improving generalization
- Model learns to handle diverse inputs
- Regularizing effect prevents overfitting on any single task

**Task interference:** Gradients from different tasks can conflict. Task A might want to increase a weight while task B wants to decrease it. Gradient projection methods address this but add complexity.

**Data mixing strategy:** Simple uniform mixing often works. Temperature-scaled mixing $p_t \propto |D_t|^{1/T}$ balances large and small datasets.

---

## 3. Adapter Methods

### 3.1 Architecture

Adapters insert small bottleneck modules into frozen transformer layers. The adapter is placed after the attention sublayer and/or the FFN sublayer.

The adapter forward pass:

$$h \leftarrow h + f(hW_{\text{down}})W_{\text{up}}$$

where:
- $W_{\text{down}} \in \mathbb{R}^{d \times r}$ projects down to a low-dimensional bottleneck
- $f(\cdot)$ is a nonlinear activation (typically ReLU or GELU)
- $W_{\text{up}} \in \mathbb{R}^{r \times d}$ projects back up to residual dimension
- $r \ll d$ (e.g., $r=64$ when $d=768$)

The residual connection $h \leftarrow h + (\cdot)$ ensures the adapter starts as near-identity (if $W_{\text{up}}$ is initialized to zero).

### 3.2 ASCII Diagram — Transformer Layer with Adapter

```
TRANSFORMER LAYER WITH ADAPTER (Houlsby et al., 2019)
======================================================

Input: h
  |
  v
+------------------+
|   Multi-Head     |
|   Attention      |
+------------------+
  |
  v
+------------------+
|   Adapter        |   <-- TRAINABLE (small)
|  h + W_up(f(     |
|    h * W_down))   |
|  W_down: d -> r  |
|  W_up:   r -> d  |
+------------------+
  |
  v
+------------------+
|   Layer Norm     |
+------------------+
  |
  v
+------------------+
|   Feed-Forward   |
|   Network        |
+------------------+
  |
  v
+------------------+
|   Adapter        |   <-- TRAINABLE (small)
|  (same structure)|
+------------------+
  |
  v
+------------------+
|   Layer Norm     |
+------------------+
  |
  v
Output: h'

Base model weights = FROZEN (gray)
Adapter weights    = TRAINABLE (highlighted)
```

### 3.3 Parameter Efficiency

For a transformer with hidden dimension $d$ and $L$ layers:

| Component | Parameters |
|-----------|-----------|
| Attention weight (one matrix) | $d^2$ |
| Adapter (one module) | $2dr$ (down + up) |
| Ratio | $r/(d/2)$ |

For $d=768, r=64$: attention has 589,824 params, adapter has 98,304 — an 84% reduction per module.

Two adapters per layer (after attention + after FFN) with $L=12$ layers and $r=64$: total adapter params = $2 \times 12 \times 2 \times 768 \times 64 = 2.36M$ vs $\sim$110M for BERT-base. That is roughly **2.1% of total parameters**.

### 3.4 Sequential vs Parallel Adapters

**Sequential adapters** (original Houlsby): placed after sublayer in series. The adapter computation is on the critical path — adds latency.

**Parallel adapters** (He et al., 2022): adapter runs in parallel with the sublayer:

$$h' = \text{Sublayer}(h) + f(hW_{\text{down}})W_{\text{up}}$$

Parallel adapters can be computed simultaneously with the main sublayer, reducing latency at the cost of more memory bandwidth.

### 3.5 Problems with Adapter Methods

| Problem | Description | Mitigation |
|---------|-------------|------------|
| Inference latency | Sequential adapters add computation depth | Use parallel adapters or merge at inference |
| Multi-task fragility | Different tasks need different adapters, managing them is complex | LoRA merging / task arithmetic |
| Limited expressivity | Bottleneck may discard task-relevant information | Increase rank $r$ |
| Adapter incompatibility | Adapters trained for different base models are not interchangeable | Standardize base model |

---

## 4. LoRA (Low-Rank Adaptation) — Most Important

### 4.1 Motivation

**The key insight:** Pretrained model weights live on a low-dimensional manifold. When fine-tuning, the *change* in weights $\Delta W$ has low intrinsic rank — you don't need to update all $d \times k$ parameters to learn a new task.

Evidence: Aghajanyan et al. (2020) showed that fine-tuning can be done in a subspace of dimension far smaller than the full parameter space (intrinsic dimensionality). For BERT, the intrinsic dimension of NLP tasks is often < 1000, far less than 110M parameters.

This motivates representing $\Delta W$ as a low-rank decomposition.

### 4.2 Core Mathematics

Instead of updating the full weight matrix $W_0 \in \mathbb{R}^{d \times k}$, LoRA decomposes the update:

$$W = W_0 + \Delta W = W_0 + BA$$

where:
- $W_0 \in \mathbb{R}^{d \times k}$ — frozen pretrained weights
- $B \in \mathbb{R}^{d \times r}$ — trainable low-rank factor
- $A \in \mathbb{R}^{r \times k}$ — trainable low-rank factor
- $r \ll \min(d, k)$ — the rank hyperparameter

The modified forward pass:

$$h = W_0 x + \frac{\alpha}{r} BAx$$

The scaling factor $\frac{\alpha}{r}$ is crucial — it controls the magnitude of the LoRA update relative to the pretrained weights. Keeping $\alpha$ fixed while varying $r$ allows rank sweeps without re-tuning the effective learning rate.

### 4.3 Initialization

- $A \sim \mathcal{N}(0, \sigma^2)$ — random Gaussian initialization (e.g., $\sigma = 0.02$)
- $B = 0$ — initialized to all zeros

This ensures $\Delta W = BA = 0$ at the start of training, so the model starts identically to the pretrained model. This is important — it ensures a stable starting point.

### 4.4 ASCII Diagram — LoRA Decomposition

```
LORA DECOMPOSITION
==================

Input x (d-dim)
     |
     |-----------------------------+
     |                             |
     v                             v
  [W_0 (d x k)]            [A (r x k)]   <- Trainable
  FROZEN                         |
     |                             v
     |                    [B (d x r)]   <- Trainable
     |                         |
     v                         v (scale: alpha/r)
  W_0 * x              +    B * A * x
     |                         |
     +-------------------------+
                 |
                 v
            Output h (d-dim)

W_0: FROZEN pretrained weights
B, A: TRAINABLE low-rank matrices
r: rank (e.g., 4, 8, 16, 64)
alpha: scaling hyperparameter
```

### 4.5 Parameter Count Comparison

For a weight matrix with $d = k = 4096$ and rank $r = 16$:

| Method | Parameters | Formula |
|--------|-----------|---------|
| Full fine-tune | 16,777,216 | $d \times k = 4096^2$ |
| LoRA | 131,072 | $(d + k) \times r = (4096 + 4096) \times 16$ |
| Reduction | **127x fewer** | |

For a 7B model with LoRA applied to Q and V projections at $r=16$:

$$\text{LoRA params} \approx 2 \times L \times 2 \times (d + k) \times r$$

With $L=32$ layers, $d=k=4096$: approximately $2 \times 32 \times 2 \times 8192 \times 16 \approx 16.8M$ trainable parameters out of 7B total — about **0.24%**.

### 4.6 No Inference Overhead

This is LoRA's killer feature vs adapters. After training, the update can be merged into the base weights:

$$W' = W_0 + \frac{\alpha}{r} BA$$

This is a single matrix addition done once before deployment. The merged $W'$ has the same shape as $W_0$, so inference is identical to the base model — zero added latency.

To switch between tasks, you can:
1. Unmerge: $W_0 = W' - \frac{\alpha}{r} BA$
2. Re-merge with a different adapter: $W'' = W_0 + \frac{\alpha}{r} B'A'$

### 4.7 Which Layers to Apply LoRA To?

The original LoRA paper applies LoRA to query ($W_Q$) and value ($W_V$) projections only. Subsequent work found:

| Configuration | Performance | Params |
|---------------|-------------|--------|
| Q, V only | Good baseline | Lowest |
| Q, K, V, O | Better | 2x |
| Q, K, V, O + FFN | Often best | 4-5x |
| All linear layers | Marginal improvement | Highest |

**Empirical finding (Hu et al., 2022):** Applying LoRA to all attention weight matrices ($W_Q, W_K, W_V, W_O$) with smaller rank outperforms applying only to $W_Q, W_V$ with larger rank, given the same parameter budget.

### 4.8 Rank Selection

$r$ is the key hyperparameter:

| Rank | Use case | Notes |
|------|---------|-------|
| $r = 1$-$4$ | Very simple task adaptation, style transfer | Risk of underfitting |
| $r = 8$-$16$ | Standard instruction tuning, most NLP tasks | Good default |
| $r = 32$-$64$ | Complex reasoning, code generation | More capacity |
| $r = 128$+ | Near full fine-tuning quality | High memory, diminishing returns |

**Rule of thumb:** Start with $r=16, \alpha=32$ (so $\alpha/r = 2$). Sweep $r \in \{4, 8, 16, 32\}$ if needed.

### 4.9 Why $\alpha/r$ Scaling?

Without scaling, changing $r$ changes the effective learning rate of the LoRA parameters: larger $r$ means more parameters, but each contributes less. The $\alpha/r$ factor normalizes this so the effective learning rate of $\Delta W$ is approximately constant as $r$ varies.

Setting $\alpha = r$ makes $\alpha/r = 1$ (equivalent to no scaling). Setting $\alpha = 2r$ doubles the LoRA update strength. In practice, $\alpha$ is treated as a fixed constant (e.g., $\alpha=16$ or $\alpha=32$) and only $r$ is swept.

### 4.10 LoRA Variants

- **LoRA+:** Different LRs for $A$ and $B$ matrices improve performance
- **AdaLoRA:** Adaptively allocates rank budget across layers using SVD-based importance scores
- **DoRA (Weight-Decomposed LoRA):** Decomposes weights into magnitude and direction components, applies LoRA to direction
- **GaLore:** Applies gradient projection to allow full-parameter training in low-memory setting

---

## 5. QLoRA

### 5.1 Motivation

A 65B parameter model in FP16 requires approximately $65 \times 10^9 \times 2 \approx 130$ GB of GPU memory — far beyond any single consumer GPU. QLoRA (Dettmers et al., 2023) makes fine-tuning such models accessible on a single 48GB A100 or even a 24GB consumer GPU.

### 5.2 Three Key Innovations

#### Innovation 1: 4-bit NormalFloat (NF4) Quantization

Standard INT4 quantization uses evenly spaced quantile levels. But neural network weights are approximately normally distributed — most weights are near zero, few are large.

NF4 uses quantile levels derived from the normal distribution $\mathcal{N}(0, 1)$. Specifically, the $2^4 = 16$ quantile values are chosen so each bin contains equal probability mass:

$$q_i = \Phi^{-1}\left(\frac{i}{2^k + 1}\right), \quad i = 1, \ldots, 2^k - 1$$

where $\Phi^{-1}$ is the inverse normal CDF and $k=4$ bits.

This is **information-theoretically optimal** for normally distributed data — the quantization error is minimized because the quantile bins are denser where data is denser.

Each weight is quantized by finding its nearest NF4 quantile value. During computation, weights are dequantized back to BF16.

#### Innovation 2: Double Quantization

The NF4 quantization uses per-block quantization constants (scale factors). For block size 64, each block needs one FP32 constant — adding $32/64 = 0.5$ bits per parameter overhead.

Double quantization quantizes these constants too: the quantization constants are quantized with 8-bit precision, with block size 256. This saves approximately $0.37$ bits per parameter (from $0.5$ to $0.127$ bits for the constants).

**Total memory:** 4-bit weights + 0.127 bits for double-quantized constants ≈ 4.127 bits per parameter, vs 16 bits for FP16. A roughly **4x memory reduction**.

#### Innovation 3: Paged Optimizers

Adam optimizer states for a single parameter are two additional values (first and second moment), requiring $3 \times$ memory vs the parameter itself in FP32. For a 7B model: $7 \times 10^9 \times 3 \times 4 = 84$ GB just for optimizer states.

Paged optimizers use NVIDIA's unified memory API to transparently page optimizer states between GPU VRAM and CPU RAM during gradient accumulation steps, preventing out-of-memory errors.

### 5.3 QLoRA Training Flow

```
QLORA TRAINING FLOW
===================

Base Model Weights (NF4, 4-bit) -- FROZEN
         |
         | Dequantize to BF16 on-the-fly during forward pass
         v
   [Forward Pass in BF16]
         |
         v
   [Compute Loss]
         |
         v
   [Backward Pass]
         |
         +-----> Gradients for LoRA adapters (BF16) -- TRAINABLE
         |
         v
   [Update A, B matrices in BF16]
         |
         | Paged to CPU if GPU OOM
         v
   [Optimizer States (CPU RAM if needed)]

Memory breakdown for 65B model:
  NF4 weights:  ~32.5 GB (vs 130 GB FP16)
  LoRA adapters:  ~0.5 GB (BF16)
  Optimizer states: ~4 GB (paged, BF16)
  Activations:    ~variable
  Total: ~40 GB -- fits on 2x A100 or 1x 80GB A100
```

### 5.4 Memory Savings

For a 65B model:

| Representation | Memory |
|---------------|--------|
| FP32 | 260 GB |
| FP16 / BF16 | 130 GB |
| 8-bit (INT8) | 65 GB |
| 4-bit NF4 (QLoRA) | ~32.5 GB |

### 5.5 Problems with QLoRA

| Problem | Description | Mitigation |
|---------|-------------|------------|
| Quantization error | NF4 introduces irreducible approximation error | Use NF4 only for base; keep adapters in BF16 |
| Slower training | Dequantization on every forward pass adds overhead | Accept 30-50% slowdown vs full BF16 LoRA |
| Not suitable for full fine-tune | You cannot fine-tune the quantized weights | Only LoRA adapters are trained |
| Accuracy gap | QLoRA models slightly underperform BF16 LoRA on same hardware | Use QLoRA only when BF16 LoRA doesn't fit |

---

## 6. Prefix Tuning and Prompt Tuning

### 6.1 Prompt Tuning

Prepend $k$ trainable "soft token" embeddings to the input sequence. These are continuous vectors in embedding space, not discrete tokens.

$$\text{Input} = [v_1, v_2, \ldots, v_k, x_1, x_2, \ldots, x_n]$$

where $v_i \in \mathbb{R}^d$ are trainable soft tokens and $x_i$ are the actual input token embeddings.

**Parameter count:** $k \times d$ total. For $k=100, d=4096$: 409,600 parameters — almost nothing.

**Key finding (Lester et al., 2021):** Prompt tuning performance approaches full fine-tuning quality as model scale increases. At 11B parameters (T5-XXL), prompt tuning matches full fine-tuning.

```
PROMPT TUNING
=============

[v1][v2]...[vk][x1][x2]...[xn]
  ^   ^      ^
  |   |      |
Trainable   Frozen input
soft tokens  embeddings

Only v1..vk are updated during training.
All model weights W are frozen.
```

### 6.2 Prefix Tuning

Extends prompt tuning by prepending trainable vectors to the key-value matrices at **every attention layer**, not just the input embedding layer.

$$K_{\text{new}} = [P_K; K], \quad V_{\text{new}} = [P_V; V]$$

where $P_K, P_V \in \mathbb{R}^{k \times d}$ are learned prefix matrices for each layer, $K$ and $V$ are the standard key-value matrices computed from input.

At each attention head in each layer:

$$\text{Attention}(Q, K_{\text{new}}, V_{\text{new}}) = \text{softmax}\left(\frac{Q K_{\text{new}}^T}{\sqrt{d_k}}\right) V_{\text{new}}$$

The prefix tokens act as "virtual context" that can directly influence every layer's computation.

**Parameter count:** $2 \times L \times k \times d$ for $L$ layers. For $L=24, k=100, d=1024$: 4.9M parameters.

### 6.3 Problems with Prefix/Prompt Tuning

| Problem | Description |
|---------|-------------|
| Optimization difficulty | Soft token gradients flow through many layers; training is unstable |
| Limited expressivity | Fixed-length prefix cannot capture complex task structure |
| Underperforms LoRA | Empirically, LoRA consistently outperforms prefix tuning at same param budget |
| Inference overhead | Prefix tokens increase KV-cache size; for prefix tuning, every layer has extra KV pairs |
| Poor performance at small scale | Requires large models (>10B) to be competitive |

---

## 7. Instruction Tuning

### 7.1 What Is Instruction Tuning?

Instruction tuning fine-tunes a base LLM on a dataset of (instruction, response) pairs. The goal is to teach the model to follow natural language instructions, transforming a next-token predictor into a helpful assistant.

**Without instruction tuning:** Given "What is the capital of France?", a base model might continue with "... and Germany? Let me ask you another geography question..."

**With instruction tuning:** The model responds "The capital of France is Paris."

### 7.2 Data Formats

Standard template format:

```
### Instruction:
Summarize the following article in 3 bullet points.

### Input:
{article text here}

### Response:
• Point 1
• Point 2
• Point 3
```

**Key datasets:**
- **FLAN (Wei et al., 2022):** 1800+ NLP tasks with templates, multi-task instruction tuning
- **Alpaca:** 52K GPT-3.5-turbo generated instruction-response pairs (Stanford)
- **Dolly 2.0 (Databricks):** 15K human-written instruction pairs
- **OpenAssistant:** Multi-turn conversation trees with human feedback

### 7.3 Why Instruction Tuning Works

A base model is a conditional distribution $P(x_t | x_{<t})$ — it can *complete* text but doesn't understand the *intent* to help. Instruction tuning teaches the model the mapping:

$$\text{intent} \rightarrow \text{helpful completion}$$

The model learns stylistic patterns: start responses directly, be concise, follow specified formats, avoid harmful content.

### 7.4 Data Quality > Quantity

**LIMA paper (Zhou et al., 2023):** Fine-tuning LLaMA-65B on just 1,000 carefully curated examples produces a competitive assistant. Key findings:

1. A base model already has the knowledge — instruction tuning just unlocks it
2. 1000 high-quality examples > 100,000 noisy examples
3. Diversity of tasks matters more than total volume

**What makes high-quality data?**
- Clear, unambiguous instructions
- Accurate, factual responses
- Diverse formatting (lists, paragraphs, code)
- Appropriate length (not too short, not padded)
- No hallucinations or sycophantic filler

### 7.5 Problems with Instruction Tuning

| Problem | Description | Mitigation |
|---------|-------------|------------|
| Sycophancy | Model agrees with user even when user is wrong | Add calibration examples, RLHF |
| Instruction-factuality tradeoff | Model follows format but hallucinates facts | Train on high-quality factual data |
| Format rigidity | Model learns to expect exact template format | Use diverse templates during training |
| Style imitation | Model imitates style of annotators, not actual correctness | Filter by quality, not just style |

---

## 8. RLHF: Reinforcement Learning from Human Feedback

### 8.1 Overview

RLHF is the alignment technique behind ChatGPT, Claude, and Gemini. It addresses a fundamental problem: language modeling loss $-\log P(y|x)$ does not capture helpfulness, harmlessness, or honesty.

RLHF has three stages. The full pipeline:

```
RLHF THREE-STAGE PIPELINE
==========================

STAGE 1: Supervised Fine-Tuning (SFT)
---------------------------------------
Human-written          SFT model
demonstrations  --->  (base + instruction tuned)
                       pi_SFT

STAGE 2: Reward Model Training
--------------------------------
Human preference       Reward model
comparisons     --->  r_phi(x, y)
(y_w preferred         (LLM + scalar head)
 over y_l)

For each (x, y_w, y_l) triple:
   P(y_w > y_l) = sigmoid(r(x,y_w) - r(x,y_l))

STAGE 3: PPO Fine-tuning
------------------------
              +---> KL penalty <-----+
              |   (stay near SFT)   |
              |                     |
  pi_SFT  --->  [PPO agent]  ---> r_phi(x,y)
  (init)         pi_theta           reward
              |                     signal
              +---------------------+

4 models in GPU memory simultaneously:
  1. SFT model (reference, frozen)
  2. Policy model (trainable, initialized from SFT)
  3. Reward model (frozen)
  4. Value model (critic, initialized from reward model)
```

### 8.2 Stage 1: Supervised Fine-tuning (SFT)

Start with a pretrained base model. Fine-tune on high-quality human demonstrations of helpful conversations. This is standard instruction tuning (Section 7). The resulting model $\pi_{SFT}$ is the starting point for RLHF.

### 8.3 Stage 2: Reward Model Training

Human annotators compare pairs of responses $(y_w, y_l)$ to the same prompt $x$, indicating which they prefer. The reward model learns to predict human preference.

**Architecture:** Same LLM backbone as SFT model, with the final language modeling head replaced by a scalar regression head:

$$r_\phi(x, y) = \text{LLM}_\phi(x, y)[\text{last token}] \cdot w^T$$

where $w \in \mathbb{R}^d$ is a trainable weight vector.

**Bradley-Terry preference model:** The probability that $y_w$ is preferred over $y_l$:

$$P(y_w \succ y_l | x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

**Training loss** (cross-entropy over preferences):

$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

### 8.4 Stage 3: PPO Fine-tuning

The policy $\pi_\theta$ (initialized from $\pi_{SFT}$) generates responses and receives reward signals from the frozen reward model. The PPO objective with KL divergence constraint:

$$\mathcal{L}_{PPO} = \mathbb{E}_{x \sim D, y \sim \pi_\theta(\cdot|x)}\left[r(x,y)\right] - \beta D_{KL}(\pi_\theta(y|x) \| \pi_{SFT}(y|x))$$

The KL term $\beta D_{KL}$ prevents the policy from drifting too far from $\pi_{SFT}$. Without it, the model would exploit the reward model (reward hacking).

**PPO clipping:** Standard PPO uses clipped importance ratios:

$$\mathcal{L}_{CLIP} = \mathbb{E}\left[\min\left(\rho_t A_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t\right)\right]$$

where $\rho_t = \pi_\theta(a_t|s_t)/\pi_{old}(a_t|s_t)$ is the importance ratio and $A_t$ is the advantage estimate.

### 8.5 Reward Hacking (Goodhart's Law)

> "When a measure becomes a target, it ceases to be a good measure." — Goodhart's Law

The reward model is an imperfect proxy for human preferences. The PPO agent finds ways to maximize $r_\phi$ that don't actually improve quality:
- Verbose responses that "sound" confident but are wrong
- Sycophantic agreement with user premises
- Specific stylistic patterns the reward model liked during training

**The KL constraint is the primary defense** — limiting how far the policy can stray from $\pi_{SFT}$.

### 8.6 RLHF Problems Summary

| Problem | Description | Mitigation |
|---------|-------------|------------|
| 4-model memory burden | SFT, policy, reward, value models all in GPU | LoRA for policy; gradient checkpointing |
| Reward hacking | Policy exploits reward model flaws | KL constraint; ensemble reward models |
| PPO instability | RL training is notoriously brittle | Careful hyperparameter tuning; use DPO instead |
| Distribution shift | Policy drifts from training distribution of reward model | Periodic reward model retraining |
| Human annotation bias | Annotators have preferences and blind spots | Diverse annotator pools; calibration |
| Annotation cost | Human preference data is expensive | Use AI feedback (RLAIF), synthetic preferences |

---

## 9. DPO (Direct Preference Optimization)

### 9.1 Full Derivation

DPO (Rafailov et al., 2023) eliminates the need for a separate reward model and PPO by directly optimizing the language model on preference data. The derivation is elegant — follow it carefully.

**Starting point:** The optimal RLHF policy has a closed-form solution. Given any reward function $r(x,y)$, the optimal policy satisfying the KL-constrained objective is:

$$\pi^*(y|x) \propto \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$$

This can be verified by solving the constrained optimization with Lagrange multipliers.

**Step 1: Normalize to get $Z(x)$.**

$$\pi^*(y|x) = \frac{\pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)}{Z(x)}$$

where $Z(x) = \sum_y \pi_{ref}(y|x) \exp\left(\frac{1}{\beta} r(x,y)\right)$ is the partition function.

**Step 2: Rearrange to express $r$ in terms of policies.**

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

**Step 3: Substitute into the Bradley-Terry model.**

The Bradley-Terry preference probability uses the reward difference:

$$P(y_w \succ y_l | x) = \sigma(r(x,y_w) - r(x,y_l))$$

Substituting Step 2:

$$r(x,y_w) - r(x,y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}$$

Note: $\beta \log Z(x)$ cancels out! This is the key mathematical insight.

**Step 4: Replace $\pi^*$ with trainable $\pi_\theta$ to get the DPO loss.**

$$\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

### 9.2 What DPO Is Doing

The argument to $\sigma$ can be written as:

$$\beta \underbrace{\left(\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}\right)}_{\text{implicit reward of chosen}} - \beta \underbrace{\left(\log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)}_{\text{implicit reward of rejected}}$$

DPO is maximizing the margin between the **implicit reward** assigned to the chosen response vs the rejected response. The implicit reward is the log-ratio of the policy to the reference model.

```
DPO CONCEPTUAL DIAGRAM
=======================

(x, y_chosen, y_rejected)
           |
           v
+----------+----------+
|                     |
v                     v
pi_theta(y_w|x)   pi_theta(y_l|x)
pi_ref(y_w|x)     pi_ref(y_l|x)
|                     |
v                     v
log-ratio_w       log-ratio_l
(implicit rew_w)  (implicit rew_l)
|                     |
+---------+-----------+
          |
          v
     rew_w - rew_l
          |
          v
   -log(sigma(beta * diff))
          |
          v
     Minimize loss
     (maximize margin)
```

### 9.3 Practical Advantages of DPO over RLHF

| Aspect | RLHF | DPO |
|--------|------|-----|
| Models needed | 4 (SFT, policy, reward, value) | 2 (policy + reference) |
| RL training | PPO (unstable) | Supervised loss (stable) |
| Reward model | Required (separate training) | Implicit in loss |
| Memory footprint | Very high | Moderate |
| Hyperparameter sensitivity | High (RL is brittle) | Low |
| Performance | State of art | Competitive, sometimes better |

### 9.4 DPO Problems

| Problem | Description | Mitigation |
|---------|-------------|------------|
| Mode collapse | Model assigns near-zero probability to rejected responses | IPO (adds regularization term) |
| Chosen reward hacking | Model inflates log-ratio for chosen by memorizing it | Diverse training data, data augmentation |
| Reference model importance | Performance depends heavily on quality of $\pi_{ref}$ | Use strong SFT model as reference |
| No online samples | DPO uses offline data — distribution may not match policy | Iterative DPO, online DPO variants |
| Overoptimization | At high $\beta$, model barely changes from reference | Tune $\beta$ carefully; typical range 0.01-0.5 |

### 9.5 DPO Variants

| Method | Key Modification | Advantage |
|--------|-----------------|-----------|
| **IPO** (Azar et al.) | Adds regularization to prevent overoptimization | More stable than DPO |
| **KTO** (Ethayarajh et al.) | Uses Kahneman-Tversky value function; works with unpaired data | No preference pairs needed |
| **ORPO** (Hong et al.) | Combines SFT loss + DPO-style odds ratio penalty; no reference model | 1 model, no reference required |
| **SimPO** | Uses length-normalized average log-prob as reward | Better length calibration |
| **CPO** | Contrastive Preference Optimization | Improved stability |

---

## 10. Fine-tuning for Embeddings

### 10.1 Contrastive Fine-tuning (SBERT Style)

For embedding models (used in semantic search, retrieval, clustering), the goal is metric learning: similar texts should have similar embeddings.

**Multiple Negatives Ranking (MNR) loss** — the most common:

$$\mathcal{L}_{MNR} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(\text{sim}(q_i, p_i)/\tau)}{\sum_{j=1}^{N} \exp(\text{sim}(q_i, p_j)/\tau)}$$

where $q_i$ is the query, $p_i$ is the positive passage, other $p_j$ in the batch are in-batch negatives, and $\tau$ is temperature.

**Cosine similarity:**

$$\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$

### 10.2 Hard Negative Mining

**In-batch negatives** (as above) are easy — the model quickly learns to ignore unrelated passages. **Hard negatives** are passages that are similar to the query but not the correct answer — much harder and more informative.

**BM25 negatives:** Retrieve top-k BM25 results for each query, exclude the positive, use remaining as hard negatives. These are lexically similar but semantically different.

**Cross-encoder negatives:** Use a cross-encoder (slow but accurate) to re-rank BM25 results and pick negatives that cross-encoder scores highly but are not the true positive.

**Embedding similarity negatives (ANN negatives):** Use the current embedding model to retrieve nearest neighbors, exclude positives — these are the hardest negatives for the current model state.

### 10.3 Synthetic Data Generation with LLMs

**Generative approach:** Use an LLM to generate (query, passage) pairs from a document corpus:

```python
prompt = f"""
Given the following passage, generate a realistic question that this passage answers:

Passage: {passage}

Question:
"""
query = llm.generate(prompt)
# Creates (query, passage) positive pair
```

This technique (used in E5, BGE, GTE models) dramatically increases training data volume.

### 10.4 GradCache

Contrastive training benefits from large batch sizes (more negatives = better signal). But large batches don't fit in GPU memory due to activations.

**GradCache** decouples the forward pass and gradient computation:
1. Compute embeddings for all examples in mini-batches without storing activations
2. Compute the contrastive loss using all embeddings
3. Compute gradients of loss w.r.t. embeddings
4. Recompute activations in mini-batches and backpropagate

This allows effective batch sizes of 32,768+ while using the same GPU memory as batch size 256.

---

## 11. Evaluation

### 11.1 Held-out Evaluation

Split your fine-tuning data into train/val/test:

| Split | Typical Ratio | Purpose |
|-------|--------------|---------|
| Train | 80-90% | Model training |
| Validation | 5-10% | Hyperparameter tuning, early stopping |
| Test | 5-10% | Final evaluation (use only once) |

For small datasets (<1000 examples), use k-fold cross-validation.

### 11.2 Benchmark Evaluation

For instruction-tuned models:
- **MT-Bench:** 80 multi-turn questions across 8 categories, GPT-4 judges responses on 1-10 scale
- **Alpaca Eval:** 805 prompts, GPT-4 Turbo as judge, compares against GPT-4 reference
- **MMLU:** 57 academic subjects, 4-choice MCQ, tests knowledge
- **HumanEval:** Python programming problems, tests code generation
- **TruthfulQA:** Tests for hallucination and truthfulness

For embedding models:
- **BEIR:** Heterogeneous retrieval benchmark, 18 datasets
- **MTEB:** Massive Text Embedding Benchmark, 56 tasks across 7 task categories

### 11.3 Avoiding Contamination

**Test set leakage** occurs when benchmark examples appear in training data. This is a serious problem as it inflates benchmark scores artificially.

**Detection methods:**
- n-gram overlap checking between train and test
- Embedding similarity search for near-duplicate detection
- Track data provenance carefully

**Best practice:** If fine-tuning on web-scraped data, always check for overlap with your evaluation benchmarks (e.g., MMLU questions are widely circulated online).

### 11.4 Human Evaluation

For alignment-focused fine-tuning, human evaluation remains the gold standard:
- **Pairwise preference:** Present two model outputs, ask which is better
- **Absolute rating:** Rate response on 1-5 Likert scale for helpfulness, accuracy, safety
- **Side-by-side comparison:** Compare fine-tuned vs baseline on same prompts

**LLM-as-judge** (GPT-4 or Claude as evaluator) has become a scalable substitute, though it inherits the judge model's biases.

---

## 12. Problems & Mitigations (Dedicated Section)

### 12.1 Catastrophic Forgetting

**Symptoms:** Model performs well on new task but degrades on original tasks/capabilities.

| Mitigation | Mechanism | Cost |
|------------|-----------|------|
| EWC | Penalize updates to important parameters | Compute Fisher matrix once |
| Replay / Experience replay | Mix old task data into new training batches | Need to store old data |
| LoRA | Base weights frozen; only adapters change | Default LoRA behavior |
| Multi-task training | Train on all tasks simultaneously | Need all task data at once |
| Progressive neural networks | Add new columns of neurons for new tasks | Architecture overhead |

**LoRA is the most practical solution:** Since $W_0$ is frozen, the model can never forget what it learned during pretraining. Only the $\Delta W = BA$ component encodes the new task.

### 12.2 Overfitting with Small Datasets

**Symptoms:** Training loss decreases but validation loss diverges or plateaus.

| Mitigation | Notes |
|------------|-------|
| Reduce LoRA rank $r$ | Fewer parameters = less capacity to overfit |
| Weight decay (L2 regularization) | Standard, add to optimizer |
| Dropout on adapter | Add dropout inside adapter/LoRA modules |
| Early stopping | Stop when val loss stops improving |
| Data augmentation | Paraphrase, backtranslation, synonym swap |
| Fewer epochs | Often 1-3 epochs is enough for instruction tuning |

### 12.3 Sycophancy in RLHF

**Symptoms:** Model agrees with incorrect user statements; changes answer when pushed even when originally correct.

| Mitigation | Notes |
|------------|-------|
| Diverse annotator pool | Reduces individual bias effects |
| Include calibration data | Explicitly train model to disagree when correct |
| Constitutional AI | Have model critique its own responses |
| DPO with sycophancy negatives | Add examples where sycophantic response is rejected |

### 12.4 Reward Hacking

**Symptoms:** High reward model score but poor actual quality; verbose, meandering responses; exaggerated positivity.

| Mitigation | Notes |
|------------|-------|
| KL divergence penalty | Primary defense in PPO |
| Ensemble reward models | Average multiple RMs to reduce individual flaws |
| Conservative KL coefficient $\beta$ | Higher $\beta$ = more conservative policy update |
| Periodic RM retraining | Retrain RM on policy outputs to catch new exploits |
| Constitutional AI / RLAIF | Use LLM-based reward as additional signal |

### 12.5 LoRA Rank Too Low

**Symptoms:** Fine-tuned model shows limited improvement; fails on complex task-specific queries.

| Mitigation | Notes |
|------------|-------|
| Increase rank $r$ | Try $r \in \{16, 32, 64\}$ |
| Apply LoRA to more matrices | Add FFN layers, not just attention |
| Full fine-tune if budget allows | Sometimes necessary for complex domains |
| AdaLoRA | Automatically allocates rank budget |

### 12.6 Quantization Quality Loss (QLoRA)

**Symptoms:** QLoRA-trained model underperforms equivalent BF16 LoRA model.

| Mitigation | Notes |
|------------|-------|
| Use BF16 adapters (always) | Never quantize the LoRA matrices themselves |
| Calibration dataset | NF4 quantization is calibrated on representative data |
| Larger adapter rank | Compensate for quantization noise with more adapter capacity |
| Accept the tradeoff | QLoRA is for memory-constrained settings; BF16 is better if it fits |

---

## 13. Industry Practices

### 13.1 Decision Framework: When to Use What

```
DECISION FLOWCHART
==================

Start: I have a new task/domain
         |
         v
Can GPT-4/Claude do it well with a good prompt?
   YES  ---> Use prompting. Done.
   NO
   |
   v
Is it a knowledge retrieval problem?
   YES  ---> Use RAG + prompting
   NO
   |
   v
Do I have GPU access to the model weights?
   NO   ---> Few-shot prompting or fine-tune via API
   YES
   |
   v
How many training examples do I have?
   < 100    ---> Few-shot or prompt tuning
   100-1K   ---> LoRA (low rank, r=8)
   1K-100K  ---> LoRA (r=16-32) or full fine-tune
   > 100K   ---> Full fine-tune if you can afford it
   |
   v
Memory constraints?
   < 24GB   ---> QLoRA
   < 80GB   ---> LoRA (BF16)
   Multiple GPUs  ---> Full fine-tune with FSDP/DeepSpeed
```

### 13.2 LoRA Model Merging

After training multiple LoRA adapters, you can merge them using **task arithmetic** (Ilharco et al., 2023):

$$W_{merged} = W_0 + \sum_{t} \lambda_t \Delta W_t = W_0 + \sum_{t} \lambda_t B_t A_t$$

where $\lambda_t$ are merge coefficients (typically $\lambda_t \in [0, 1]$).

**TIES-Merging** (Yadav et al., 2023) improves on simple linear combination by:
1. Trimming small-magnitude task vectors
2. Resolving sign conflicts between task vectors
3. Disjoint merging of resolved vectors

This allows combining specialized adapters (e.g., "code + math + reasoning") without interference.

### 13.3 Serving LoRA at Scale

**LoRAX** (Predibase): Production system for serving many LoRA adapters on a single GPU:
- Shared base model weights in GPU memory
- Multiple LoRA adapters loaded on demand
- Dynamic batching: requests for different adapters in the same batch
- Cost: ~1 adapter overhead per request vs dedicated model per customer

**vLLM with LoRA support:** Open-source serving with PagedAttention supports multiple LoRA adapters with minimal overhead.

### 13.4 The Data Flywheel

Production ML systems improve through a feedback loop:

```
DATA FLYWHEEL
=============

User queries
     |
     v
[Model inference]
     |
     v
User feedback (implicit: clicks, dwell time; explicit: thumbs up/down)
     |
     v
[Hard negative mining from failures]
     |
     v
[Retrain embedding model / fine-tune policy]
     |
     v
Better model --> more users --> more data
     ^                               |
     +-------------------------------+
```

For embedding models: mine hard negatives from retrieval failures, add to training set, retrain. Repeat.

### 13.5 Crypto/Finance-Specific Fine-tuning (Binance Relevance)

**Domain-specific fine-tuning targets:**
- Blockchain documentation (Ethereum yellowpaper, Solana docs, BEP standards)
- DeFi protocol mechanics (AMM formulas, liquidation logic, tokenomics)
- Regulatory filings and compliance documents
- On-chain data interpretation (transaction types, smart contract interactions)
- Market microstructure terminology (order books, funding rates, basis trading)

**Practical approach:**
1. Collect domain corpus: whititepapers, audit reports, governance proposals, docs
2. Continual pre-training on domain corpus (update base model knowledge)
3. Instruction tuning on curated (question, answer) pairs from domain experts
4. RLHF/DPO with crypto-domain experts as annotators
5. Evaluate on domain-specific benchmarks (financial QA, code execution for smart contracts)

---

## 14. Interview Q&A

### Basic Level

**Q1: What is the difference between pretraining and fine-tuning?**

**A:** Pretraining is self-supervised training on massive unlabeled corpora (web text, books, code) using next-token prediction. The model learns general language understanding and world knowledge. Fine-tuning adapts this pretrained model to a specific task using a smaller labeled or curated dataset. Pretraining is expensive (millions of dollars, weeks of compute); fine-tuning is cheap (hours on a few GPUs). The pretrained model provides a rich initialization that makes fine-tuning sample-efficient — you need far less data than training from scratch.

**Q2: What is catastrophic forgetting?**

**A:** When you fine-tune a neural network on task B after it was trained on task A, the model "forgets" how to do task A. This happens because gradient descent on task B's loss pushes weights to new regions of parameter space that are suboptimal for task A. The gradients from task B have no knowledge of the curvature of task A's loss landscape. In LLM fine-tuning, you might fine-tune a model for medical QA and find it forgets how to do math or code. Mitigations include: EWC (penalizing changes to important parameters), multi-task training (training on both tasks simultaneously), and LoRA (keeping base weights frozen so they never change).

**Q3: What is LoRA and why is it used?**

**A:** LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning method. Instead of updating all weights in a pretrained model, LoRA decomposes the weight update as a product of two low-rank matrices: $\Delta W = BA$ where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$. The base weights $W_0$ are frozen; only $B$ and $A$ are trained. This reduces trainable parameters by 100-1000x. After training, $\Delta W$ can be merged into $W_0$ with no inference overhead. LoRA is used because it allows fine-tuning large models on consumer GPUs, prevents catastrophic forgetting (base weights are frozen), and enables easy adapter swapping.

**Q4: What is instruction tuning?**

**A:** Instruction tuning fine-tunes a base language model on (instruction, response) pairs. A base model is a next-token predictor — it completes text but doesn't "know" to be helpful. Instruction tuning teaches the model to follow natural language instructions, producing an assistant-style model. Examples: FLAN uses 1800+ NLP task templates; Alpaca uses 52K GPT-3.5-generated pairs. Key finding from LIMA (2023): 1,000 high-quality examples can be more effective than 100,000 noisy ones. Quality and diversity of instruction data matters more than volume.

---

### Intermediate Level

**Q5: Walk me through the LoRA parameter count calculation. Why is it efficient?**

**A:** For a weight matrix $W_0 \in \mathbb{R}^{d \times k}$:
- Full fine-tuning updates all $d \times k$ parameters
- LoRA uses $B \in \mathbb{R}^{d \times r}$ and $A \in \mathbb{R}^{r \times k}$, totaling $(d+k) \times r$ parameters

For $d=k=4096, r=16$: Full = $4096^2 = 16.8M$; LoRA = $8192 \times 16 = 131K$. That is a **127x reduction**.

For a 7B model with LoRA on Q,K,V,O in all 32 layers at $r=16$: approximately $4 \times 32 \times 2 \times 4096 \times 16 \approx 16.8M$ trainable parameters out of 7B total, which is 0.24%. The model can be fine-tuned with far less GPU memory (only adapter gradients and optimizer states, not full model gradients).

**Q6: What is the purpose of the $\alpha/r$ scaling in LoRA?**

**A:** The forward pass is $h = W_0 x + \frac{\alpha}{r} BAx$. The scaling factor $\frac{\alpha}{r}$ serves two purposes:

1. **Magnitude control:** $\alpha$ controls how strongly the LoRA update affects the output relative to the frozen weights $W_0$. If $\alpha = r$, the scaling is 1 (no amplification). If $\alpha = 2r$, LoRA updates are doubled in effect.

2. **Rank-invariant effective learning rate:** As $r$ increases (more parameters), each parameter contributes less to the output. The $1/r$ factor normalizes this so that changing $r$ doesn't implicitly change the effective learning rate of $\Delta W$. This means you can sweep $r$ without retuning $\alpha$.

In practice, $\alpha$ is fixed (e.g., 16 or 32) and only $r$ is swept as a hyperparameter.

**Q7: How does QLoRA reduce memory usage? What are the three innovations?**

**A:** QLoRA (Dettmers 2023) combines three innovations:

1. **NF4 quantization:** Base model weights stored in 4-bit NormalFloat format. NF4 quantile bins are spaced according to normal distribution quantiles, making them information-theoretically optimal for normally distributed weights. This reduces base model memory by 4x vs FP16.

2. **Double quantization:** NF4 uses per-block scale factors (quantization constants). These constants are themselves quantized to 8-bit with block size 256, saving ~0.37 bits per parameter.

3. **Paged optimizers:** Adam optimizer states (first and second moment) are paged between GPU VRAM and CPU RAM when GPU memory is full, preventing OOM errors.

Combined, a 65B model that requires 130GB in FP16 can be fine-tuned in ~40GB with QLoRA. LoRA adapters remain in BF16 (trainable); base model weights dequantize on-the-fly during forward pass.

**Q8: What is the RLHF pipeline? What problem does each stage solve?**

**A:** RLHF has three stages:

**Stage 1 — SFT:** Fine-tune base model on high-quality human demonstrations. Converts the next-token predictor into a model that can at least approximate helpful behavior. Necessary because the base model would score poorly on instruction-following tasks even with a good reward model.

**Stage 2 — Reward Model:** Train a scalar reward predictor from human preference data (pairwise comparisons). This is necessary because we can't write a formal loss function for "helpfulness" — but humans can identify which of two responses is better. The Bradley-Terry model $P(y_w \succ y_l) = \sigma(r_w - r_l)$ lets us learn from ordinal comparisons.

**Stage 3 — PPO:** Use the reward model as a training signal to optimize the policy via RL. The KL constraint $\beta D_{KL}(\pi_\theta \| \pi_{SFT})$ prevents the policy from deviating so far that it exploits reward model flaws. PPO's clipped objective ensures stable updates.

---

### Advanced Level

**Q9: Derive the DPO loss from the RLHF objective.**

**A:** Starting from the KL-constrained RLHF objective:

$$\max_\pi \mathbb{E}_{y \sim \pi(\cdot|x)}[r(x,y)] - \beta D_{KL}(\pi \| \pi_{ref})$$

The optimal solution (via calculus of variations) is:

$$\pi^*(y|x) = \frac{\pi_{ref}(y|x) \exp(r(x,y)/\beta)}{Z(x)}$$

where $Z(x) = \sum_y \pi_{ref}(y|x) \exp(r(x,y)/\beta)$.

Rearranging to express $r$ as a function of $\pi^*$:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

Substituting into Bradley-Terry: $P(y_w \succ y_l|x) = \sigma(r(x,y_w) - r(x,y_l))$:

$$r(x,y_w) - r(x,y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{ref}(y_l|x)}$$

The $Z(x)$ terms cancel. Replace $\pi^*$ with trainable $\pi_\theta$ and maximize log-likelihood of preferences:

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

DPO avoids RL entirely — it is a simple cross-entropy loss. The reward model is implicit in the log-ratio. Only 2 models needed: $\pi_\theta$ (trainable) and $\pi_{ref}$ (frozen).

**Q10: What is the Fisher Information Matrix and how is it used in EWC?**

**A:** The Fisher Information Matrix (FIM) measures how sensitive the model's output distribution is to changes in parameters. The $(i,j)$ entry is:

$$F_{ij} = \mathbb{E}\left[\frac{\partial \log P(y|x,\theta)}{\partial \theta_i} \cdot \frac{\partial \log P(y|x,\theta)}{\partial \theta_j}\right]$$

The diagonal $F_{ii}$ measures the variance of the gradient of parameter $i$. A high $F_{ii}$ means the model's output changes a lot when $\theta_i$ changes — i.e., $\theta_i$ is important for the task.

EWC uses the diagonal FIM as importance weights in a regularization term:

$$\mathcal{L}_{EWC} = \mathcal{L}_{new} + \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$$

Parameters with high Fisher values (important for old task) are strongly penalized for moving away from $\theta^*$. Parameters with low Fisher values are free to update. This is analogous to a spring system where $F_i$ is the spring constant.

**Limitation:** The full FIM for a billion-parameter model is $N^2$ — completely intractable. The diagonal approximation ignores parameter correlations. In practice, EWC is rarely used for large-scale LLM fine-tuning; LoRA (frozen base weights) is preferred.

**Q11: What is reward hacking and how do you prevent it?**

**A:** Reward hacking (Goodhart's Law applied to RL) occurs when the policy finds ways to maximize the reward model score without actually improving quality. The reward model is a learned proxy for human preferences; it has blind spots and distributional biases.

Examples of reward hacking:
- Longer responses score higher → model becomes verbose with filler text
- Certain linguistic patterns (confident tone, hedging) score higher → model overuses them
- Formatting artifacts from training data → model produces similar artifacts

Prevention:

1. **KL divergence constraint:** $\beta D_{KL}(\pi_\theta \| \pi_{SFT})$ in the PPO objective limits how far the policy drifts from the SFT model. The SFT model doesn't know the reward model's blind spots.

2. **Ensemble reward models:** Average rewards from multiple independently trained reward models. Hacking one requires hacking all.

3. **Conservative RL:** Use higher $\beta$ (stronger KL penalty) to limit policy deviation.

4. **DPO:** By directly optimizing preferences without a RL loop, DPO is less susceptible to reward hacking. The log-ratio is bounded by the reference model.

5. **Constitutional AI (Anthropic):** Use LLM-based critique as auxiliary signal, harder to hack than a learned reward model.

**Q12: Compare LoRA, prefix tuning, and full fine-tuning in terms of performance, efficiency, and use cases.**

**A:**

| Aspect | Full Fine-tune | LoRA | Prefix Tuning |
|--------|---------------|------|---------------|
| Trainable params | All (~100%) | 0.1-1% | 0.01-0.1% |
| Performance | Best | Near-best | Underperforms LoRA |
| Catastrophic forgetting | High risk | None (base frozen) | None (base frozen) |
| Inference latency | Same as base | Same (after merge) | Higher (longer KV cache) |
| Memory (training) | Very high | Low | Low |
| Optimization stability | High | High | Low (tricky to optimize) |
| Scale sensitivity | Any scale | Any scale | Works best >10B |
| Use case | Maximum quality, large data | Standard fine-tuning | Mostly superseded by LoRA |

**When to use each:**
- **Full fine-tune:** You have hundreds of thousands of examples, multiple GPUs, and need maximum performance. Common for production models at large companies.
- **LoRA:** Default choice for most fine-tuning. Works well from 1B to 70B+ models.
- **Prefix tuning:** Largely superseded by LoRA; only consider for very large models where you want extreme parameter efficiency.

**Q13: How does hard negative mining improve embedding model training?**

**A:** Contrastive learning with random in-batch negatives saturates quickly — after initial training, the model easily distinguishes random passages from queries. The loss gradient becomes negligible. Hard negatives are passages that are semantically similar to the query but are not the correct answer — the model is "almost fooled" and the gradient is large and informative.

**BM25 negatives:** High keyword overlap with query but wrong passage. Trains the model to go beyond lexical matching.

**Cross-encoder negatives:** A cross-encoder scores all BM25 candidates; top-scored non-positive passages are the hardest. Cross-encoders are slow but accurate, so they're run offline to generate a static hard negative set.

**ANN negatives (online mining):** Use the current model to find its nearest neighbors in the corpus; exclude positives. These are the hardest negatives for the current model state and change as the model improves. Most effective but requires running inference on the full corpus periodically.

**Deduplication:** Hard negatives must be filtered to remove false negatives (passages that are actually relevant but not labeled as such). Common: use BM25 or exact match to filter out query-adjacent passages.

**Q14: Explain the difference between DPO and IPO. When would you prefer IPO?**

**A:** DPO optimizes:

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

DPO is derived from the Bradley-Terry model which assumes: if $r(y_w) - r(y_l) \to \infty$, then $P(y_w \succ y_l) \to 1$. But the Bradley-Terry model can be overconfident — it may push log-ratios to arbitrarily large values during training (overoptimization).

IPO (Identity Preference Optimization, Azar et al., 2023) modifies the loss to directly regularize the log-ratio gap:

$$\mathcal{L}_{IPO} = \mathbb{E}\left[\left(\log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

IPO drives the implicit reward gap toward $\frac{1}{2\beta}$ rather than infinity. This prevents overoptimization and mode collapse.

**Prefer IPO when:**
- You observe the policy collapsing (very high confidence on chosen, near-zero on rejected)
- Training data is noisy (human annotations are inconsistent)
- You want stronger regularization than DPO provides

**Q15: What is ORPO and why does it not need a reference model?**

**A:** ORPO (Odds Ratio Preference Optimization, Hong et al., 2024) combines standard SFT training with a preference learning term in a single loss, without requiring a reference model.

The ORPO loss:

$$\mathcal{L}_{ORPO} = \mathcal{L}_{SFT} + \lambda \mathcal{L}_{OR}$$

where $\mathcal{L}_{SFT}$ is the standard NLL loss on chosen responses and $\mathcal{L}_{OR}$ is the odds ratio penalty:

$$\mathcal{L}_{OR} = -\mathbb{E}\left[\log \sigma\left(\log \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}\right)\right]$$

with $\text{odds}_\theta(y|x) = \frac{\pi_\theta(y|x)}{1 - \pi_\theta(y|x)}$.

Why no reference model? The SFT component serves as the implicit regularizer. The policy is simultaneously trained to increase probability of chosen responses (SFT) and reduce relative probability of rejected responses (OR penalty). The baseline is established by the current policy itself, not a frozen reference.

**Advantages:** 1 model vs 2 models (saves 50% memory); simpler training loop; often competitive with DPO in practice.

---

## 15. Coding Problems

### 15.1 Implement LoRA from Scratch in PyTorch

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    A linear layer augmented with a LoRA decomposition.

    Forward pass: h = W_0 @ x + (alpha/r) * B @ A @ x

    W_0 is frozen; A and B are trainable.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 32.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank  # the alpha/r factor

        # Frozen base weight (would normally load pretrained weights here)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features), requires_grad=False
        )
        self.bias = nn.Parameter(torch.zeros(out_features), requires_grad=False)

        # LoRA trainable parameters
        # A: initialized with kaiming_uniform (similar to normal)
        # B: initialized to zero (ensures delta_W = 0 at start)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # lora_B is already zero from torch.zeros

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base model output (frozen)
        base_out = nn.functional.linear(x, self.weight, self.bias)

        # LoRA delta: (alpha/r) * B @ A @ x
        # x shape: (..., in_features)
        # lora_A: (rank, in_features)  -> A @ x: (..., rank)
        # lora_B: (out_features, rank) -> B @ Ax: (..., out_features)
        lora_out = self.dropout(x) @ self.lora_A.T  # (..., rank)
        lora_out = lora_out @ self.lora_B.T          # (..., out_features)
        lora_out = lora_out * self.scaling

        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """
        Merge LoRA weights into base weight for deployment.
        Returns a standard Linear layer with merged weights.
        """
        merged_weight = self.weight + self.scaling * (self.lora_B @ self.lora_A)
        merged_layer = nn.Linear(self.in_features, self.out_features)
        merged_layer.weight = nn.Parameter(merged_weight)
        merged_layer.bias = nn.Parameter(self.bias.clone())
        return merged_layer

    def count_parameters(self) -> dict:
        total = self.weight.numel() + self.bias.numel()
        lora = self.lora_A.numel() + self.lora_B.numel()
        return {
            "total_params": total + lora,
            "frozen_params": total,
            "trainable_lora_params": lora,
            "reduction_factor": total / lora,
        }


# Test the implementation
if __name__ == "__main__":
    torch.manual_seed(42)

    d_in, d_out, rank = 4096, 4096, 16
    layer = LoRALinear(d_in, d_out, rank=rank, alpha=32.0)

    x = torch.randn(8, 128, d_in)  # batch=8, seq_len=128
    out = layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Parameter stats: {layer.count_parameters()}")

    # Verify lora_B=0 means delta_W=0 at init
    delta_w = layer.scaling * (layer.lora_B @ layer.lora_A)
    print(f"Delta W norm at init (should be ~0): {delta_w.norm().item():.6f}")

    # Merge and verify
    merged = layer.merge_weights()
    out_merged = merged(x.view(-1, d_in)).view(8, 128, d_out)
    print(f"Max diff between LoRA and merged outputs: {(out - out_merged).abs().max().item():.6f}")
```

### 15.2 Apply LoRA to a HuggingFace BERT Model

```python
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import List, Optional

def inject_lora_to_bert(
    model: BertModel,
    target_modules: List[str] = ["query", "value"],
    rank: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.1,
) -> BertModel:
    """
    Injects LoRA adapters into specified modules of a BERT model.

    Freezes all parameters except LoRA adapters.
    """
    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace target linear layers with LoRA versions
    for name, module in model.named_modules():
        # Check if this module's name matches any target
        module_name = name.split(".")[-1]
        if module_name in target_modules and isinstance(module, nn.Linear):
            # Get parent module and attribute name
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)

            attr_name = parts[-1]
            original_layer = getattr(parent, attr_name)

            # Create LoRA layer
            lora_layer = LoRALinear(
                in_features=original_layer.in_features,
                out_features=original_layer.out_features,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

            # Copy pretrained weights (frozen)
            lora_layer.weight = nn.Parameter(
                original_layer.weight.data.clone(), requires_grad=False
            )
            if original_layer.bias is not None:
                lora_layer.bias = nn.Parameter(
                    original_layer.bias.data.clone(), requires_grad=False
                )

            # Replace the layer
            setattr(parent, attr_name, lora_layer)
            print(f"Injected LoRA into: {name}")

    return model


def count_trainable_params(model: nn.Module) -> dict:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {
        "trainable": trainable,
        "total": total,
        "trainable_pct": 100 * trainable / total,
    }


# Usage example
if __name__ == "__main__":
    # Load BERT
    config = BertConfig()
    model = BertModel(config)

    print("Before LoRA injection:")
    stats = count_trainable_params(model)
    print(f"  Trainable: {stats['trainable']:,} ({stats['trainable_pct']:.1f}%)")

    # Inject LoRA
    model = inject_lora_to_bert(
        model,
        target_modules=["query", "key", "value", "dense"],
        rank=16,
        alpha=32.0,
    )

    print("\nAfter LoRA injection:")
    stats = count_trainable_params(model)
    print(f"  Trainable: {stats['trainable']:,} ({stats['trainable_pct']:.2f}%)")
    print(f"  Total: {stats['total']:,}")

    # Forward pass test
    input_ids = torch.randint(0, 30522, (2, 64))
    attention_mask = torch.ones(2, 64)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    print(f"\nOutput shape: {outputs.last_hidden_state.shape}")


# Alternative: Use PEFT library (production approach)
def peft_lora_example():
    """
    Using HuggingFace PEFT library (recommended for production)
    """
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import BertForSequenceClassification

    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        target_modules=["query", "value"],  # which modules to apply LoRA to
        lora_dropout=0.1,
        bias="none",  # don't train biases
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Output: trainable params: 888,576 || all params: 109,770,240 || trainable%: 0.8095

    return model
```

### 15.3 Implement DPO Loss

```python
import torch
import torch.nn.functional as F
from typing import Tuple


def compute_sequence_log_probs(
    logits: torch.Tensor,   # (batch, seq_len, vocab_size)
    labels: torch.Tensor,   # (batch, seq_len)
    attention_mask: torch.Tensor,  # (batch, seq_len)
) -> torch.Tensor:
    """
    Compute sum of log probabilities for each sequence.
    Returns: (batch,) tensor of log-probs
    """
    # Shift: predict token t+1 from token t
    # logits: (batch, seq_len-1, vocab_size)
    # labels: (batch, seq_len-1)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous()

    # Log-probabilities: (batch, seq_len-1, vocab_size)
    log_probs = F.log_softmax(shift_logits, dim=-1)

    # Select log-prob of actual next token
    # Gather: (batch, seq_len-1)
    selected_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask padding tokens and sum
    selected_log_probs = selected_log_probs * shift_mask
    return selected_log_probs.sum(dim=-1)  # (batch,)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # (batch,) log P_theta(y_w|x)
    policy_rejected_logps: torch.Tensor,  # (batch,) log P_theta(y_l|x)
    ref_chosen_logps: torch.Tensor,       # (batch,) log P_ref(y_w|x)
    ref_rejected_logps: torch.Tensor,     # (batch,) log P_ref(y_l|x)
    beta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute DPO loss.

    L_DPO = -E[log sigma(beta * (log pi_theta(y_w)/pi_ref(y_w)
                                - log pi_theta(y_l)/pi_ref(y_l)))]

    Returns:
        loss: scalar DPO loss
        chosen_rewards: implicit rewards for chosen responses
        rejected_rewards: implicit rewards for rejected responses
    """
    # Implicit rewards: log-ratio of policy to reference
    # r_implicit(x, y) = beta * log(pi_theta(y|x) / pi_ref(y|x))
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)

    # DPO loss: -log sigma(r_chosen - r_rejected)
    reward_diff = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(reward_diff).mean()

    return loss, chosen_rewards.detach(), rejected_rewards.detach()


def dpo_training_step(
    policy_model,
    ref_model,
    batch: dict,
    beta: float = 0.1,
    optimizer=None,
) -> dict:
    """
    Full DPO training step.

    batch should contain:
        - chosen_input_ids, chosen_attention_mask, chosen_labels
        - rejected_input_ids, rejected_attention_mask, rejected_labels
    """
    # Policy model: compute log-probs for chosen and rejected
    policy_chosen_logits = policy_model(
        input_ids=batch["chosen_input_ids"],
        attention_mask=batch["chosen_attention_mask"],
    ).logits

    policy_rejected_logits = policy_model(
        input_ids=batch["rejected_input_ids"],
        attention_mask=batch["rejected_attention_mask"],
    ).logits

    policy_chosen_logps = compute_sequence_log_probs(
        policy_chosen_logits, batch["chosen_labels"], batch["chosen_attention_mask"]
    )
    policy_rejected_logps = compute_sequence_log_probs(
        policy_rejected_logits, batch["rejected_labels"], batch["rejected_attention_mask"]
    )

    # Reference model: no gradient
    with torch.no_grad():
        ref_chosen_logits = ref_model(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"],
        ).logits
        ref_rejected_logits = ref_model(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"],
        ).logits

        ref_chosen_logps = compute_sequence_log_probs(
            ref_chosen_logits, batch["chosen_labels"], batch["chosen_attention_mask"]
        )
        ref_rejected_logps = compute_sequence_log_probs(
            ref_rejected_logits, batch["rejected_labels"], batch["rejected_attention_mask"]
        )

    loss, chosen_rewards, rejected_rewards = dpo_loss(
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps,
        beta=beta,
    )

    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
        optimizer.step()

    return {
        "loss": loss.item(),
        "chosen_rewards_mean": chosen_rewards.mean().item(),
        "rejected_rewards_mean": rejected_rewards.mean().item(),
        "reward_margin": (chosen_rewards - rejected_rewards).mean().item(),
        "accuracy": (chosen_rewards > rejected_rewards).float().mean().item(),
    }


# Test with random tensors
if __name__ == "__main__":
    batch_size = 4

    # Simulate log-probs (typically large negative numbers)
    policy_chosen = torch.tensor([-10.5, -8.2, -12.1, -9.8])
    policy_rejected = torch.tensor([-15.3, -11.7, -16.4, -13.2])
    ref_chosen = torch.tensor([-11.0, -8.5, -12.5, -10.1])
    ref_rejected = torch.tensor([-14.8, -11.2, -15.9, -12.9])

    loss, chosen_r, rejected_r = dpo_loss(
        policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=0.1
    )

    print(f"DPO Loss: {loss.item():.4f}")
    print(f"Chosen rewards: {chosen_r}")
    print(f"Rejected rewards: {rejected_r}")
    print(f"Reward margin: {(chosen_r - rejected_r).mean().item():.4f}")
    print(f"Accuracy: {(chosen_r > rejected_r).float().mean().item():.2%}")
```

### 15.4 Fine-tune a Sentence Transformer with Hard Negatives

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Optional
import random


class TripletDataset(Dataset):
    """
    Dataset for contrastive training with hard negatives.
    Each sample: (query, positive_passage, hard_negative_passage)
    """
    def __init__(
        self,
        triplets: List[Tuple[str, str, str]],  # (query, pos, neg)
        tokenizer,
        max_length: int = 256,
    ):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        query, positive, negative = self.triplets[idx]

        def tokenize(text):
            return self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        return {
            "query": tokenize(query),
            "positive": tokenize(positive),
            "negative": tokenize(negative),
        }


def mean_pool(
    token_embeddings: torch.Tensor,  # (batch, seq_len, hidden)
    attention_mask: torch.Tensor,    # (batch, seq_len)
) -> torch.Tensor:
    """Mean pooling — average token embeddings weighted by attention mask."""
    mask_expanded = attention_mask.unsqueeze(-1).float()
    sum_embeddings = (token_embeddings * mask_expanded).sum(dim=1)
    sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
    return sum_embeddings / sum_mask  # (batch, hidden)


class SentenceTransformer(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        # Optional projection head (can help for contrastive learning)
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 256),  # project to 256-dim space
        )

    def encode(self, input_ids, attention_mask) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = mean_pool(outputs.last_hidden_state, attention_mask)
        embeddings = self.projection(embeddings)
        return F.normalize(embeddings, p=2, dim=-1)  # L2 normalize

    def forward(self, query_batch, pos_batch, neg_batch):
        q_emb = self.encode(
            query_batch["input_ids"].squeeze(1),
            query_batch["attention_mask"].squeeze(1),
        )
        p_emb = self.encode(
            pos_batch["input_ids"].squeeze(1),
            pos_batch["attention_mask"].squeeze(1),
        )
        n_emb = self.encode(
            neg_batch["input_ids"].squeeze(1),
            neg_batch["attention_mask"].squeeze(1),
        )
        return q_emb, p_emb, n_emb


def multiple_negatives_ranking_loss(
    query_emb: torch.Tensor,    # (batch, dim)
    pos_emb: torch.Tensor,      # (batch, dim)
    neg_emb: Optional[torch.Tensor] = None,  # (batch, dim) hard negatives
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    MNR loss with optional hard negatives.

    For each query q_i, positive is p_i.
    Negatives are all other p_j in batch (in-batch) + hard negatives if provided.
    """
    batch_size = query_emb.shape[0]

    if neg_emb is not None:
        # Concatenate positives and hard negatives as candidate set
        all_passages = torch.cat([pos_emb, neg_emb], dim=0)  # (2*batch, dim)
    else:
        all_passages = pos_emb  # (batch, dim)

    # Similarity matrix: (batch, 2*batch) or (batch, batch)
    similarity = torch.matmul(query_emb, all_passages.T) / temperature

    # Labels: query i should match passage i (the first batch_size entries)
    labels = torch.arange(batch_size, device=query_emb.device)

    # Cross-entropy loss
    loss = F.cross_entropy(similarity, labels)

    # Metrics
    with torch.no_grad():
        predictions = similarity.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean()

    return loss, accuracy


def train_epoch(
    model: SentenceTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    temperature: float = 0.05,
    device: str = "cuda",
) -> dict:
    model.train()
    total_loss = 0
    total_acc = 0

    for batch in dataloader:
        # Move to device
        query_batch = {k: v.to(device) for k, v in batch["query"].items()}
        pos_batch = {k: v.to(device) for k, v in batch["positive"].items()}
        neg_batch = {k: v.to(device) for k, v in batch["negative"].items()}

        # Forward pass
        q_emb, p_emb, n_emb = model(query_batch, pos_batch, neg_batch)

        # Loss with hard negatives
        loss, acc = multiple_negatives_ranking_loss(
            q_emb, p_emb, n_emb, temperature=temperature
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_acc += acc.item()

    n = len(dataloader)
    return {"loss": total_loss / n, "accuracy": total_acc / n}


# Example usage with synthetic data
if __name__ == "__main__":
    # Synthetic triplets (in practice, these come from your corpus + hard negative mining)
    triplets = [
        (
            "What is Bitcoin?",
            "Bitcoin is a decentralized digital currency created in 2009.",
            "Ethereum is a blockchain platform supporting smart contracts.",
        ),
        (
            "How does proof of work function?",
            "Proof of work requires miners to solve computationally expensive puzzles.",
            "Proof of stake selects validators based on their staked tokens.",
        ),
        # ... many more triplets
    ]

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    dataset = TripletDataset(triplets * 100, tokenizer)  # repeat for demo
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Train for a few epochs
    for epoch in range(3):
        metrics = train_epoch(model, dataloader, optimizer, device=device)
        print(f"Epoch {epoch+1}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}")

    # Encode a query for inference
    model.eval()
    with torch.no_grad():
        encoding = tokenizer(
            "What is DeFi?",
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding=True,
        ).to(device)
        embedding = model.encode(encoding["input_ids"], encoding["attention_mask"])
        print(f"Query embedding shape: {embedding.shape}")
        print(f"Query embedding norm: {embedding.norm().item():.4f}")  # Should be ~1.0
```

---

## Quick Reference Summary

### Key Equations

| Concept | Equation |
|---------|---------|
| LoRA forward pass | $h = W_0 x + \frac{\alpha}{r} BAx$ |
| LoRA parameter savings | $(d+k)r$ vs $dk$ |
| EWC loss | $\mathcal{L}_{EWC} = \mathcal{L}_{new} + \frac{\lambda}{2}\sum_i F_i(\theta_i - \theta_i^*)^2$ |
| Reward model loss | $\mathcal{L}_{RM} = -\mathbb{E}[\log\sigma(r(y_w) - r(y_l))]$ |
| PPO objective | $\mathcal{L}_{PPO} = \mathbb{E}[r(x,y)] - \beta D_{KL}(\pi_\theta \| \pi_{SFT})$ |
| DPO loss | $\mathcal{L}_{DPO} = -\mathbb{E}[\log\sigma(\beta\log\frac{\pi_\theta(y_w)}{\pi_{ref}(y_w)} - \beta\log\frac{\pi_\theta(y_l)}{\pi_{ref}(y_l)})]$ |
| Optimal RLHF policy | $\pi^*(y|x) \propto \pi_{ref}(y|x)\exp(r(x,y)/\beta)$ |
| MNR loss | $\mathcal{L} = -\frac{1}{N}\sum_i\log\frac{\exp(\text{sim}(q_i,p_i)/\tau)}{\sum_j\exp(\text{sim}(q_i,p_j)/\tau)}$ |

### Decision Matrix

| Situation | Recommendation |
|-----------|---------------|
| < 100 examples, API access only | Few-shot prompting |
| Knowledge retrieval, facts change | RAG |
| 100-10K examples, 1-2 GPUs | LoRA ($r$=16, apply to Q,K,V,O) |
| Large model (>30B), single GPU | QLoRA |
| Need best quality, many GPUs | Full fine-tune with FSDP |
| Alignment/preference learning | DPO (simpler) or RLHF (more control) |
| Semantic search | Contrastive fine-tune with hard negatives |
| Multiple specialized tasks | LoRA + task arithmetic / TIES merging |

---

*End of LLM Fine-tuning Guide. Word count: ~8,500 words.*
