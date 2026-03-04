# Loss Functions for LLMs and Representation Learning
## ST5230 — Lecture Notes Series, Chapter 3

> **Scope:** This chapter derives, analyzes, and implements every major loss function used in modern NLP, embedding learning, and RLHF-aligned LLMs. Every formula is derived from first principles. Every loss comes with a Problems & Mitigations section. Python/PyTorch implementations are provided for all major losses. Interview Q&A appears at the end.

---

## Table of Contents

1. Foundations: What Makes a Good Loss Function?
2. Cross-Entropy Loss
3. Contrastive Loss (Siamese Networks)
4. Triplet Loss
5. InfoNCE / NT-Xent Loss
6. Multiple Negatives Ranking Loss (MNRL)
7. Listwise Ranking Losses
8. Focal Loss
9. Knowledge Distillation Loss
10. RLHF Losses
11. DPO (Direct Preference Optimization)
12. Embedding Fine-tuning Summary Table
13. Problems & Mitigations (Dedicated Section)
14. Industry Practices
15. Interview Q&A
16. Coding Problems

---

## 1. Foundations: What Makes a Good Loss Function?

### 1.1 The Learning Signal

A loss function $\mathcal{L}(\theta)$ encodes the task: it quantifies how wrong the model's current predictions are. Gradient descent then adjusts $\theta$ to reduce $\mathcal{L}$. A good loss function must satisfy:

1. **Faithful task alignment** — minimizing $\mathcal{L}$ should genuinely improve performance on the real metric (accuracy, NDCG, BLEU, etc.).
2. **Everywhere differentiable** (or subgradients must exist) — so that backpropagation can compute $\nabla_\theta \mathcal{L}$.
3. **Non-trivial gradient landscape** — should provide useful signal even when predictions are "almost correct." Easy examples should contribute less; hard examples should drive learning.
4. **Numerical stability** — no log(0), no overflow from large exponentials.

### 1.2 Cross-Entropy as Maximum Likelihood Estimation — Full Derivation

Suppose we have a dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ where $y_i$ is a class label. We model the probability of class $c$ given input $x$ as $p_\theta(c \mid x)$, parameterized by a neural network.

**The likelihood** of the entire dataset under the model:

$$\mathcal{L}(\theta) = \prod_{i=1}^N p_\theta(y_i \mid x_i)$$

**Taking the log** (log-likelihood, which we maximize):

$$\ell(\theta) = \sum_{i=1}^N \log p_\theta(y_i \mid x_i)$$

**Negating** gives the Negative Log-Likelihood (NLL), which we minimize:

$$\text{NLL}(\theta) = -\sum_{i=1}^N \log p_\theta(y_i \mid x_i)$$

For a single example with a one-hot target distribution $q$ (the true label) and model distribution $p_\theta$:

$$\text{NLL} = -\sum_c q(c) \log p_\theta(c \mid x) = H(q, p_\theta)$$

This is exactly the **cross-entropy** $H(q, p)$ between the true distribution $q$ and the model $p$. Therefore:

> **Minimizing cross-entropy = Maximizing log-likelihood under the empirical data distribution.**

### 1.3 KL Divergence and Its Relationship to Cross-Entropy

The **Kullback-Leibler divergence** from $Q$ to $P$:

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \sum_x P(x) \log P(x) - \sum_x P(x) \log Q(x)$$

$$D_{KL}(P \| Q) = -H(P) + H(P, Q)$$

where $H(P)$ is the entropy of $P$ and $H(P, Q)$ is the cross-entropy.

Since $H(P)$ is constant w.r.t. model parameters $\theta$:

$$\min_\theta H(P, Q_\theta) \equiv \min_\theta D_{KL}(P \| Q_\theta)$$

**Cross-entropy minimization is equivalent to KL minimization when the target distribution is fixed.**

### 1.4 Forward vs. Reverse KL: Mean-Seeking vs. Mode-Seeking

This distinction is critical for understanding generation and distillation:

**Forward KL** (also called "inclusive KL"): $D_{KL}(P \| Q_\theta)$

$$D_{KL}(P \| Q_\theta) = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q_\theta(x)}\right]$$

- When $P(x) > 0$ and $Q_\theta(x) \approx 0$: the term $\log \frac{P(x)}{Q_\theta(x)} \to \infty$. This is **infinity** — catastrophic.
- $Q_\theta$ is forced to cover all modes of $P$ (cannot ignore any region where $P > 0$).
- Result: **mean-seeking** — $Q_\theta$ spreads mass over all modes, potentially producing blurry/averaged outputs.

**Reverse KL** (also called "exclusive KL"): $D_{KL}(Q_\theta \| P)$

$$D_{KL}(Q_\theta \| P) = \mathbb{E}_{x \sim Q_\theta}\left[\log \frac{Q_\theta(x)}{P(x)}\right]$$

- When $Q_\theta(x) > 0$ and $P(x) \approx 0$: the term blows up — $Q_\theta$ is penalized for placing mass where $P$ is low.
- When $Q_\theta(x) \approx 0$: the term contributes 0, so $Q_\theta$ can safely ignore modes of $P$.
- Result: **mode-seeking** — $Q_\theta$ collapses onto a single sharp mode of $P$.

**Practical implications:**
- Training LLMs with teacher forcing uses forward KL (cross-entropy) → mean-seeking, which produces "average" tokens.
- RLHF can use reverse KL → mode-seeking, which can cause reward hacking (collapse to one high-reward mode).
- The KL penalty in PPO keeps the policy close to the reference by penalizing $D_{KL}(\pi_\theta \| \pi_\text{ref})$ (reverse KL from the policy perspective).

---

## 2. Cross-Entropy Loss

### 2.1 Binary Cross-Entropy — Full Derivation from Bernoulli Likelihood

For binary classification, $y \in \{0, 1\}$, model outputs $\hat{y} = \sigma(z) \in (0,1)$.

The Bernoulli likelihood of observing $y$:

$$P(y \mid x) = \hat{y}^y (1-\hat{y})^{1-y}$$

Log-likelihood:

$$\log P(y \mid x) = y \log \hat{y} + (1-y) \log(1-\hat{y})$$

Negating gives the **Binary Cross-Entropy (BCE)**:

$$\boxed{\mathcal{L}_{\text{BCE}} = -[y \log \hat{y} + (1-y) \log(1-\hat{y})]}$$

**Gradient computation.** Let $\hat{y} = \sigma(z)$:

$$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$$

This is a beautifully clean gradient — it is just the prediction error. The sigmoid and the log cancel perfectly, which is why BCE + sigmoid is the standard pairing (not sigmoid → MSE, which produces vanishing gradients).

### 2.2 Categorical Cross-Entropy

For $K$-class classification with one-hot target $\mathbf{y}$ and softmax predictions $\hat{\mathbf{p}}$:

$$\boxed{\mathcal{L}_{\text{CCE}} = -\sum_{c=1}^K y_c \log \hat{p}_c}$$

Since $\mathbf{y}$ is one-hot with $y_{c^*} = 1$, this reduces to:

$$\mathcal{L}_{\text{CCE}} = -\log \hat{p}_{c^*}$$

i.e., minimize the negative log-probability assigned to the correct class.

**Combined softmax + cross-entropy for numerical stability:**

Naive softmax $\hat{p}_c = \frac{e^{z_c}}{\sum_k e^{z_k}}$ then $\log \hat{p}_c = z_c - \log \sum_k e^{z_k}$ suffers overflow if $z_k$ is large.

**Log-Sum-Exp (LSE) trick:** Let $m = \max_k z_k$:

$$\log \sum_k e^{z_k} = m + \log \sum_k e^{z_k - m}$$

Since $z_k - m \leq 0$, the exponentials are bounded in $[0, 1]$ — no overflow. PyTorch's `F.cross_entropy` applies this internally via `log_softmax`.

### 2.3 Label Smoothing

**Problem:** With hard one-hot targets, the model is incentivized to push $z_{c^*} \to \infty$ to make $\hat{p}_{c^*} \to 1$. This causes overconfident predictions and poor calibration.

**Label smoothing** replaces the hard target with a soft mixture:

$$\boxed{y_{\text{smooth},c} = (1-\epsilon) \cdot y_c + \frac{\epsilon}{K}}$$

where $\epsilon \in [0.1, 0.2]$ is the smoothing parameter and $K$ is the number of classes.

The loss becomes:

$$\mathcal{L}_{\text{LS}} = -(1-\epsilon) \log \hat{p}_{c^*} - \frac{\epsilon}{K} \sum_c \log \hat{p}_c$$

**Why it helps:**
1. Prevents the model from assigning probability 1 to any class — the optimal prediction under label smoothing is $\hat{p}_{c^*} = 1 - \epsilon + \epsilon/K$, not 1.
2. Improves calibration: the model's confidence better reflects true accuracy.
3. Acts as a regularizer — equivalent to adding a KL term pushing predictions toward the uniform distribution.

### 2.4 Temperature Scaling

After training, model outputs can be poorly calibrated (overconfident). **Temperature scaling** applies a scalar $T$ to logits:

$$\boxed{\hat{p}_c = \frac{\exp(z_c/T)}{\sum_k \exp(z_k/T)}}$$

- $T = 1$: original distribution.
- $T > 1$: softer/more uniform distribution (higher entropy). Used to create soft labels in knowledge distillation.
- $T < 1$: sharper/more peaked distribution (lower entropy). Used to sharpen predictions at inference.

Temperature $T$ is typically calibrated on a validation set by minimizing NLL, keeping model weights frozen.

### 2.5 Problems & Mitigations for Cross-Entropy

| Problem | Description | Mitigation |
|---|---|---|
| Class imbalance | Loss dominated by majority class | Weighted CE, Focal Loss |
| Overconfidence | Model assigns near-zero prob to wrong classes | Label smoothing, temperature calibration |
| Numerical instability | Overflow in softmax | Log-sum-exp trick, use `F.cross_entropy` |
| Poor calibration | Confidence ≠ accuracy | Temperature scaling post-training |

---

## 3. Contrastive Loss (Siamese Networks)

### 3.1 Intuition

Contrastive learning trains an encoder $f: \mathcal{X} \to \mathbb{R}^d$ such that:
- **Similar pairs** (same class/semantics) have **small distance** in embedding space.
- **Dissimilar pairs** (different class/semantics) have **large distance** in embedding space.

The network is "Siamese" — the same encoder $f$ processes both inputs $x_1$ and $x_2$.

### 3.2 Full Formula

Let $d = \|f(x_1) - f(x_2)\|_2$ (Euclidean distance), $y = 1$ if the pair is similar, $y = 0$ if dissimilar, and $m > 0$ a margin hyperparameter.

$$\boxed{\mathcal{L}_{\text{contrastive}} = y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2}$$

**Gradient analysis:**

For similar pairs ($y=1$): $\nabla \mathcal{L} = 2d \cdot \frac{\partial d}{\partial \theta}$ — pulls the pair closer.

For dissimilar pairs ($y=0$, $d < m$): $\nabla \mathcal{L} = -2(m-d) \cdot \frac{\partial d}{\partial \theta}$ — pushes the pair apart.

For dissimilar pairs ($y=0$, $d \geq m$): $\nabla \mathcal{L} = 0$ — already far enough, no gradient.

### 3.3 The Margin $m$: Geometric Meaning

The margin $m$ defines a **minimum acceptable separation** between dissimilar pairs. Geometrically, it is the radius of a "safe zone" around each embedding: dissimilar pairs are only penalized if their distance falls within this zone.

**Setting $m$:**
- Too small: dissimilar pairs are pushed apart weakly, embeddings cluster insufficiently.
- Too large: most dissimilar pairs are within the margin and receive large gradients, causing instability.
- Common heuristic: $m = 1.0$ for L2-normalized embeddings (since max distance is 2).

### 3.4 Problems & Mitigations for Contrastive Loss

**Problem 1: Easy negatives dominate.**
In a random batch, most dissimilar pairs are already well-separated ($d \geq m$), contributing zero gradient. The model stops learning.

**Problem 2: Pair selection is critical.**
With $N$ samples, there are $O(N^2)$ pairs. Random sampling produces mostly uninformative pairs.

**Mitigation: Hard Negative Mining**

**Semi-hard negatives** (recommended): for anchor $a$, find negatives $n$ such that:
$$d(a, p) < d(a, n) < d(a, p) + m$$
These are harder than the positive but still within the margin — they provide gradient signal without being so hard that they are likely false negatives.

**Online hard negative mining:** within each mini-batch, identify the hardest pairs (smallest $d$ among negatives) and compute the loss only on those.

---

## 4. Triplet Loss

### 4.1 Intuition and Geometry

```
Embedding Space:

        n (negative)
        *
       /
      /  d(a,n)
     /
    a (anchor) -------- p (positive)
    *          d(a,p)   *

Goal: d(a,p) + m < d(a,n)
      [anchor-positive dist + margin < anchor-negative dist]

After training:

    a---p    n
    *---*    *
    (close)  (far)
```

We want the positive to be closer to the anchor than the negative by at least margin $m$.

### 4.2 Full Formula

Given a triplet $(a, p, n)$ where $a$ is the anchor, $p$ is a positive (same class), $n$ is a negative (different class):

$$\boxed{\mathcal{L}_{\text{triplet}} = \max\left(0,\ \|f(a) - f(p)\|^2 - \|f(a) - f(n)\|^2 + m\right)}$$

**Interpretation:** The loss is zero only when the negative is at least $\sqrt{m}$ further from the anchor than the positive. Otherwise, a positive loss drives the model to fix the ordering.

**Gradient for non-zero loss:**

$$\frac{\partial \mathcal{L}}{\partial f(a)} = 2(f(n) - f(p))$$
$$\frac{\partial \mathcal{L}}{\partial f(p)} = -2(f(a) - f(p))$$
$$\frac{\partial \mathcal{L}}{\partial f(n)} = 2(f(a) - f(n))$$

The gradient pushes $a$ and $p$ together, and $a$ and $n$ apart simultaneously.

### 4.3 Margin $m$: Effect on Learning

- $m$ too small (e.g., $m = 0.01$): trivial solutions — the model can satisfy the constraint by making all embeddings nearly identical with tiny separations.
- $m$ too large (e.g., $m = 10$): most triplets are active (non-zero loss), including false negatives, leading to noisy/unstable gradients.
- Common values: $m \in [0.2, 2.0]$ for L2-normalized embeddings, $m \in [0.5, 5.0]$ for unnormalized.

### 4.4 Hard Negative Mining — Three Categories

```
Hard Negative Types:

d(a,p)                          d(a,p)+m
  |                                |
  |<------ zone A ------->|<-- B-->|<------------ C ---------->|

A: d(a,n) < d(a,p)          --> HARD NEGATIVE (highest gradient, noisy)
B: d(a,p) < d(a,n) < d+m    --> SEMI-HARD NEGATIVE (recommended)
C: d(a,n) > d(a,p) + m      --> EASY NEGATIVE (zero gradient, useless)
```

**Easy negatives (Zone C):** $d(a,n) > d(a,p) + m$

Triplet loss = 0. Zero gradient. The model learns nothing from these.

**Semi-hard negatives (Zone B):** $d(a,p) < d(a,n) < d(a,p) + m$

Loss is positive, gradient is non-zero. The negative is harder than the positive but not so hard that it is likely a mislabeled/false negative. **Recommended for stable training.**

**Hard negatives (Zone A):** $d(a,n) < d(a,p)$

Highest gradient magnitude — the negative is currently closer to the anchor than the positive. However, these often include **false negatives** (samples from the same class that were assigned a different label due to noise). Training on false negatives drives the model in the wrong direction.

### 4.5 In-Batch Negatives

In-batch negatives treat all non-paired examples in the batch as negatives for each anchor. With batch size $B$, this gives $B-1$ negatives per anchor at no additional computation cost.

**Benefit:** Efficient use of computation — every example in the batch is reused as a negative for every other anchor.

**Risk:** False negatives — if two different (anchor, positive) pairs in the batch have similar semantics, they will be pushed apart incorrectly.

### 4.6 Triplet Explosion and the $O(n^3)$ Problem

With $N$ samples, there are $O(N^3)$ possible triplets. Most are easy (zero gradient) and useless. Exhaustive enumeration is computationally infeasible.

**Mitigations:**
1. **Online mining:** Within each mini-batch, enumerate triplets and compute the loss only on semi-hard (or all active) ones.
2. **Curriculum learning:** Start with easy triplets, gradually introduce harder ones as the model improves.
3. **Reservoir sampling:** Maintain a buffer of hard negatives from recent batches; sample triplets mixing fresh and buffered negatives.

### 4.7 Problems & Mitigations for Triplet Loss

| Problem | Description | Mitigation |
|---|---|---|
| Triplet explosion | $O(N^3)$ triplets, mostly useless | Online mining per mini-batch |
| Easy negatives | Zero gradient, slow convergence | Semi-hard mining, hard negative mining |
| False negatives | Mislabeled/semantically similar negatives | Deduplication, soft labels, SimCSE |
| Margin sensitivity | Wrong $m$ → trivial or noisy solutions | Grid search, use normalized embeddings |
| Slow convergence | vs. InfoNCE | Switch to InfoNCE for large-scale tasks |

---

## 5. InfoNCE / NT-Xent Loss (Contrastive Learning at Scale)

### 5.1 Used In

- **SimCLR** (self-supervised image representation learning)
- **CLIP** (contrastive image-text pretraining at OpenAI)
- **SBERT** with MultipleNegativesRankingLoss
- **E5, BGE, GTE** (embedding model fine-tuning)

### 5.2 Full Derivation from Scratch

Given a batch of $N$ pairs $(x_i, x_i^+)$ where $x_i^+$ is the positive for $x_i$. After encoding, we have $2N$ embeddings $\{z_1, z_1^+, z_2, z_2^+, \ldots, z_N, z_N^+\}$.

For each $z_i$, the positive is $z_i^+$. All other $2N-2$ embeddings are treated as negatives (in-batch negatives).

**Define** the similarity function (typically cosine similarity):

$$\text{sim}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$

**InfoNCE loss for sample $i$:**

$$\mathcal{L}_i = -\log \frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

**Full batch loss:**

$$\boxed{\mathcal{L}_{\text{InfoNCE}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(\text{sim}(z_i, z_i^+)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}}$$

**NT-Xent (Normalized Temperature-scaled Cross-Entropy)** from SimCLR is the symmetric version:

$$\mathcal{L}_{\text{NT-Xent}} = \frac{1}{2N} \sum_{i=1}^N \left[ \ell(i, i^+) + \ell(i^+, i) \right]$$

where $\ell(i, j) = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$.

### 5.3 InfoNCE as a Lower Bound on Mutual Information

**Claim:** InfoNCE lower bounds $I(X; Y)$ (mutual information between the two views).

**Derivation sketch:**

$$I(X; Y) = \mathbb{E}\left[\log \frac{p(x,y)}{p(x)p(y)}\right]$$

The optimal critic function that maximizes the InfoNCE objective is $f^*(x,y) = \log \frac{p(x,y)}{p(x)p(y)} + \text{const}$, the log density ratio. It can be shown:

$$I(X; Y) \geq \mathcal{L}_{\text{InfoNCE}} + \log(N)$$

where $N$ is the number of negative samples. More negatives tighten the bound, which is why **larger batch sizes improve InfoNCE training** — not just more optimization signal, but a tighter estimate of mutual information.

### 5.4 Temperature $\tau$: Effect Analysis

The temperature controls how sharply the model distinguishes the positive from negatives.

**Low $\tau$ (e.g., 0.05):**
- The softmax becomes very peaked: the highest-similarity negative dominates the denominator.
- Gradient flows almost entirely from the hardest negatives.
- Easy negatives contribute negligible gradient (exponentially suppressed).
- Risk: if the hardest negative is a false negative, the gradient is misleading.

**High $\tau$ (e.g., 1.0):**
- The softmax is nearly uniform: all negatives contribute equally.
- Loss is dominated by the average similarity across the batch.
- No emphasis on hard negatives; the model learns slowly.

**Optimal $\tau$** (typically 0.05–0.2 for text, 0.07 in SimCLR) balances between:
- Sufficient attention to hard negatives (low $\tau$)
- Stability from not over-weighting potential false negatives (not too low $\tau$)

### 5.5 Batch Size Effect

With $2N$ embeddings in a batch, there are $2N - 2$ negatives per anchor. The InfoNCE loss is the log of a softmax over $2N-1$ terms:

$$\mathcal{L}_i \approx \log(2N-1) - \text{sim}(z_i, z_i^+)/\tau + \text{something positive}$$

The maximum possible loss is $\log(2N-1)$ (when the model is random). With more negatives:
1. The denominator contains more diverse hard negatives, making the task harder.
2. Each gradient step provides richer signal.
3. The MI lower bound $I(X;Y) \geq \mathcal{L} + \log N$ is tighter.

This is why SimCLR required batch size 4096 and CLIP used 32768 pairs.

### 5.6 Connection to Cross-Entropy

InfoNCE is mathematically equivalent to $(N+1)$-way cross-entropy classification where:
- The task is: "which of the $2N-1$ candidates is the positive?"
- The logits are $\text{sim}(z_i, z_k)/\tau$
- The correct class is $k = i^+$

This means all the tools for stable cross-entropy training (mixed precision, label smoothing, learning rate schedules) apply directly.

### 5.7 Problems & Mitigations for InfoNCE

| Problem | Description | Mitigation |
|---|---|---|
| False negatives | In-batch negatives may be semantically similar | Deduplication, False Negative Cancellation (FNC) |
| Large batch required | Small batch → few negatives → weak signal | GradCache (gradient checkpointing), Queue (MoCo) |
| Symmetric loss overhead | 2x forward passes | Use asymmetric variant when appropriate |
| Mode collapse | All embeddings collapse to one point | Stop-gradient (BYOL), momentum encoder (MoCo) |

---

## 6. Multiple Negatives Ranking Loss (MNRL)

### 6.1 Overview

MNRL is the primary fine-tuning loss for **Sentence-BERT (SBERT)** for retrieval tasks. It is essentially a simplified, asymmetric version of InfoNCE.

### 6.2 Full Formula

Given $N$ pairs $(a_i, p_i)$ (anchor, positive), encode all to get $\{e_{a_i}\}$ and $\{e_{p_i}\}$.

Compute the $N \times N$ similarity matrix:

$$S_{ij} = \text{sim}(e_{a_i}, e_{p_j})$$

The loss treats $p_j$ for $j \neq i$ as negatives for anchor $a_i$:

$$\boxed{\mathcal{L}_{\text{MNRL}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^N \exp(S_{ij}/\tau)}}$$

This is equivalent to cross-entropy on an $N \times N$ matrix where the diagonal entries are correct classes.

### 6.3 Why It Is Efficient: One Positive Pair → N-1 Free Negatives

With $N$ training pairs in a batch:
- Standard training would compute $N$ losses, each using only 1 positive and 0 negatives.
- MNRL computes $N$ losses, each using 1 positive and $N-1$ negatives.

**Efficiency ratio:** MNRL extracts $N \times (N-1)$ useful learning signals from the same $N$ forward passes — a factor of $N-1$ more signal per computation.

At batch size 512, that is 511 negatives per anchor at no extra compute.

### 6.4 Hard Negatives in MNRL

MNRL can be augmented with hard negatives: for each anchor $a_i$, in addition to in-batch negatives $p_j$ ($j \neq i$), also include explicitly mined hard negatives $h_i$ (passages that are hard to distinguish from the positive but are actually irrelevant):

$$\mathcal{L}_{\text{MNRL+HN}} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{ii}/\tau)}{\sum_{j=1}^N \exp(S_{ij}/\tau) + \sum_k \exp(\text{sim}(e_{a_i}, e_{h_{ik}})/\tau)}$$

### 6.5 False Negatives in MNRL

When two pairs $(a_i, p_i)$ and $(a_j, p_j)$ are semantically similar (e.g., paraphrases of the same question), $p_j$ is treated as a negative for $a_i$ incorrectly. This pushes similar pairs apart.

**Mitigations:**
1. **Deduplication** before batching — remove near-duplicate pairs.
2. **False Negative Cancellation (FNC):** Compute a similarity threshold; pairs above the threshold are removed from the denominator.
3. **Stratified batching:** Ensure pairs in the same batch are diverse.

### 6.6 GradCache (Cached Negatives) for Effectively Larger Batches

GradCache (Gao et al., 2021) enables training with effective batch sizes much larger than GPU memory allows by:
1. Encoding a large set of passages in a forward-only pass (no gradients stored).
2. Computing the full similarity matrix using these cached embeddings.
3. Computing gradients only for the current mini-batch anchors.

This decouples embedding computation memory from contrastive loss computation, enabling effective batch sizes of 32K+ on a single GPU.

### 6.7 Problems & Mitigations for MNRL

| Problem | Description | Mitigation |
|---|---|---|
| False negatives | Semantically similar in-batch pairs | Deduplication, FNC, stratified batching |
| Memory for large batches | GPU memory limits effective batch size | GradCache |
| Asymmetric loss | Only anchor→positive direction | Symmetric MNRL, add positive→anchor term |
| Cold start | Random encoder → all sims near 0 → flat loss | Initialize from strong pretrained model |

---

## 7. Listwise Ranking Losses

### 7.1 Pairwise vs. Listwise vs. Pointwise

**Pointwise:** Predict an absolute score for each document. Train with regression/classification loss on individual items.
- Example: MSE on relevance scores 0/1/2/3.
- Problem: Ignores relative ordering; can work poorly when absolute scores have noise.

**Pairwise:** For each (query, doc_i, doc_j) triple with $\text{rel}(d_i) > \text{rel}(d_j)$, train the model to score $d_i$ higher.
- Example: Contrastive loss, Triplet loss, RankNet.
- Problem: Does not optimize the full ranking metric (NDCG, MAP) directly.

**Listwise:** Optimize directly over the full permutation of documents.
- Example: ListNet, LambdaRank.
- Advantage: Directly optimizes NDCG-like objectives.

### 7.2 ListNet

ListNet defines a probability distribution over all permutations using the **Plackett-Luce model**. For a list of documents with scores $\mathbf{s} = [s_1, \ldots, s_k]$, the probability of permutation $\pi$:

$$P(\pi \mid \mathbf{s}) = \prod_{j=1}^k \frac{\exp(s_{\pi(j)})}{\sum_{l=j}^k \exp(s_{\pi(l)})}$$

The loss minimizes the KL divergence between the model's ranking distribution and the ground-truth ranking distribution:

$$\mathcal{L}_{\text{ListNet}} = -\sum_\pi P(\pi \mid \mathbf{y}) \log P(\pi \mid \mathbf{s})$$

In practice, a **top-one approximation** is used (marginalizing over all but the top position):

$$\mathcal{L}_{\text{ListNet}} \approx -\sum_i P_{\text{top-1}}(i \mid \mathbf{y}) \log P_{\text{top-1}}(i \mid \mathbf{s})$$

where $P_{\text{top-1}}(i \mid \mathbf{s}) = \frac{\exp(s_i)}{\sum_j \exp(s_j)}$ is a softmax over all scores.

### 7.3 LambdaRank and LambdaMART

**LambdaRank** directly defines gradients that approximate the change in NDCG when two documents are swapped:

$$\lambda_{ij} = \frac{-\sigma}{1 + e^{\sigma(s_i - s_j)}} \cdot |\Delta\text{NDCG}_{ij}|$$

where $|\Delta\text{NDCG}_{ij}|$ is the absolute change in NDCG from swapping documents $i$ and $j$.

The gradient with respect to document $i$'s score $s_i$:

$$\frac{\partial \mathcal{L}}{\partial s_i} = \sum_{j: y_i > y_j} \lambda_{ij} - \sum_{j: y_j > y_i} \lambda_{ji}$$

**LambdaMART** combines LambdaRank gradients with gradient boosted decision trees (MART), making it the strongest tree-based ranking model (used in web search). The neural analog is LambdaRank applied to a neural scorer.

### 7.4 Connection to RLHF Reward Modeling

In RLHF, the reward model is trained on human preference data. This is fundamentally a **pairwise ranking** problem: given a query $x$ and two responses $y_w > y_l$, train $r(x, \cdot)$ to rank $y_w$ above $y_l$.

The Bradley-Terry model (Section 10) is the pairwise ranking loss for RLHF. LambdaRank / listwise losses could be applied when full preference rankings (not just pairwise comparisons) are available.

### 7.5 Problems & Mitigations for Listwise Losses

| Problem | Description | Mitigation |
|---|---|---|
| Permutation intractability | $k!$ permutations | Top-1 approximation, sampled approximations |
| Sparse NDCG signal | NDCG is non-differentiable | Lambda trick (use NDCG as gradient weight) |
| Data requirements | Need full relevance grades | Can adapt to pairwise comparisons |

---

## 8. Focal Loss

### 8.1 Problem: Class Imbalance and Easy Negative Domination

In object detection, fraud detection, and anomaly detection, the class ratio can be 1:1000 or worse. With standard cross-entropy:

$$\mathcal{L}_{\text{CE}} = -\log \hat{p}_t$$

where $\hat{p}_t$ is the probability assigned to the true class, easy negatives (background objects in detection) accumulate large total loss even though each individual easy negative has small $-\log \hat{p}_t \approx 0.01$. With 10,000 easy negatives and 10 hard positives:

$$\mathcal{L}_{\text{total}} \approx 10000 \times 0.01 + 10 \times 3 = 100 + 30 = 130$$

The gradient is dominated by easy negatives (100 vs 30), so the model barely learns from the rare class.

### 8.2 Focal Loss Formula

Lin et al. (2017) introduce a **modulating factor** $(1-\hat{p}_t)^\gamma$:

$$\boxed{\mathcal{L}_{\text{focal}} = -(1-\hat{p}_t)^\gamma \log \hat{p}_t}$$

where $\gamma \geq 0$ is the **focusing parameter**.

- For well-classified easy examples: $\hat{p}_t \to 1$, so $(1-\hat{p}_t)^\gamma \to 0$. The easy example is downweighted.
- For hard misclassified examples: $\hat{p}_t \to 0$, so $(1-\hat{p}_t)^\gamma \to 1$. The loss is almost unmodified.

**Effect by $\gamma$:**
- $\gamma = 0$: standard cross-entropy.
- $\gamma = 2$: an example with $\hat{p}_t = 0.9$ has its loss downweighted by $(0.1)^2 = 0.01$ — a factor of 100.
- $\gamma = 5$: even more extreme downweighting of easy examples.

In practice, $\gamma = 2$ and an $\alpha$ class balance factor are used:

$$\mathcal{L}_{\text{focal}} = -\alpha_t (1-\hat{p}_t)^\gamma \log \hat{p}_t$$

### 8.3 Applications in Crypto / Finance

**Fraud detection:** Fraudulent transactions may be 0.01% of all transactions. Focal loss ensures the model does not simply predict "not fraud" for everything.

**Anomaly detection:** Market anomalies (flash crashes, unusual order flow) are rare events. Focal loss keeps the model focused on detecting the rare anomalous pattern.

**Comparison with class-weighted CE:**

$$\mathcal{L}_{\text{weighted}} = -w_t \log \hat{p}_t$$

Class weighting uses a fixed weight per class. Focal loss uses a dynamic weight based on how well the current sample is classified — it adapts to the model's current state, focusing on whatever the model currently finds hard.

### 8.4 Problems & Mitigations for Focal Loss

| Problem | Description | Mitigation |
|---|---|---|
| Sensitive to $\gamma$ | Wrong $\gamma$ → underfits or focuses on noise | Grid search $\gamma \in \{0.5, 1, 2, 5\}$ |
| Noisy hard examples | Hard examples may be mislabeled | Denoising labels before training |
| Not suitable for multi-class | Standard form is for binary/detection | Generalize to multi-class with per-class $\alpha$ |

---

## 9. Knowledge Distillation Loss

### 9.1 Teacher-Student Framework

A large, expensive **teacher** model $T$ has learned a rich representation. We want to train a small, efficient **student** model $S$ to mimic the teacher's behavior.

**Why students can match teachers:** The teacher's output probabilities (soft labels) contain more information than one-hot hard labels. Even incorrect class probabilities reveal the model's "opinion" about which classes are similar — this is **dark knowledge**.

### 9.2 Full Loss Formula

$$\boxed{\mathcal{L}_{\text{KD}} = \alpha \cdot \mathcal{L}_{\text{CE}}(\mathbf{y}, \hat{\mathbf{y}}_S) + (1-\alpha) \cdot T^2 \cdot \mathcal{L}_{\text{CE}}(\mathbf{p}_T, \mathbf{p}_S)}$$

where:
- $\mathbf{y}$: one-hot ground truth labels
- $\hat{\mathbf{y}}_S$: student predictions (temperature 1)
- $\mathbf{p}_T = \text{softmax}(z_T/T)$: teacher soft labels at temperature $T$
- $\mathbf{p}_S = \text{softmax}(z_S/T)$: student predictions at temperature $T$
- $T^2$ scaling factor: compensates for the reduced gradient magnitude at high temperature (Hinton et al., 2015)
- $\alpha \in [0,1]$: interpolation weight (typically 0.1–0.5)

### 9.3 Why Soft Labels Contain More Information (Dark Knowledge)

Consider a 3-class problem: cat, dog, automobile. A teacher trained on images might assign:
- Hard label: [1, 0, 0] (cat)
- Soft label: [0.7, 0.29, 0.01] (cat, but somewhat similar to dog, very different from automobile)

The soft label reveals:
1. **Inter-class similarity:** cats and dogs are more similar than cats and automobiles.
2. **Model uncertainty:** the teacher is not perfectly confident, perhaps because the image is ambiguous.

Training on soft labels transfers this relational knowledge — the student learns not just what is right, but how similar different categories are. This is richer than one-hot supervision.

At high temperature $T$: $p_T(c) \approx \frac{1}{K} + \frac{z_c - \bar{z}}{KT}$, so the soft labels become approximately linear in logits, revealing the full ordering of scores.

### 9.4 DistilBERT

DistilBERT (Sanh et al., 2019) distills BERT-base (110M params) to a 6-layer model (66M params, 40% smaller) achieving 97% of BERT's performance on GLUE.

**Distillation components:**
1. **Soft label loss:** KL divergence between teacher and student MLM predictions.
2. **Hard label loss:** Standard MLM cross-entropy with ground truth.
3. **Hidden state loss:** Cosine similarity between teacher and student hidden states (layer alignment with linear projection).
4. **Attention transfer:** MSE between teacher and student attention matrices.

Combined:
$$\mathcal{L}_{\text{DistilBERT}} = \alpha \mathcal{L}_{\text{MLM}} + \beta \mathcal{L}_{\text{KD}} + \gamma \mathcal{L}_{\text{cosine}}$$

### 9.5 Cross-Encoder to Bi-Encoder Distillation for Retrieval

**Problem:** Cross-encoders (BERT over concatenated query+doc) are accurate but slow (O(N) forward passes per query for N documents). Bi-encoders are fast (pre-compute doc embeddings) but less accurate.

**Solution:** Use a cross-encoder teacher to generate soft relevance scores for (query, doc) pairs. Fine-tune a bi-encoder student to reproduce these scores.

This is often done with **margin MSE loss:**

$$\mathcal{L}_{\text{MarginMSE}} = \text{MSE}\left(\text{sim}(q, d^+) - \text{sim}(q, d^-),\ r_T(q, d^+) - r_T(q, d^-)\right)$$

where $r_T$ is the cross-encoder teacher's relevance score. This preserves relative ordering rather than absolute scores.

### 9.6 Problems & Mitigations for Knowledge Distillation

| Problem | Description | Mitigation |
|---|---|---|
| Capacity gap | Student too small to mimic teacher | Progressive distillation, intermediate layers |
| Distribution shift | Teacher trained on different domain | Distill on target domain data |
| Wrong $\alpha$ | Too much weight on hard vs soft labels | Cross-validate $\alpha$, task-dependent |
| Teacher errors | Student learns teacher mistakes | Ensemble teachers, filter high-uncertainty examples |

---

## 10. RLHF Losses

### 10.1 Reward Model Loss: Bradley-Terry Model

**Setup:** Given a prompt $x$ and two responses $y_w$ (winner/preferred) and $y_l$ (loser/rejected), train a reward model $r_\phi(x, y)$ to assign higher rewards to preferred responses.

**Bradley-Terry model** defines the probability that $y_w$ is preferred:

$$P(y_w \succ y_l \mid x) = \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))$$

The reward model loss (negative log-likelihood):

$$\boxed{\mathcal{L}_{\text{RM}} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\left[\log \sigma\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]}$$

This is a **pairwise ranking loss** — it only cares about the relative ordering, not absolute reward values. The model is therefore translation-invariant: adding a constant to all rewards does not change the loss.

**Gradient:** $\frac{\partial \mathcal{L}}{\partial r_w} = -\sigma(r_l - r_w)$ and $\frac{\partial \mathcal{L}}{\partial r_l} = \sigma(r_l - r_w)$. When $r_w > r_l$, the gradient pushes $r_w$ higher and $r_l$ lower.

### 10.2 PPO Objective for LLM Fine-tuning

PPO (Proximal Policy Optimization) optimizes the policy $\pi_\theta$ (the LLM) to maximize expected reward while staying close to the reference policy $\pi_\text{ref}$ (the original SFT model).

**Full PPO objective:**

$$\boxed{\mathcal{L}_{\text{PPO}} = \mathbb{E}_t\left[\min\left(r_t(\theta) A_t,\ \text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) A_t\right)\right] - \beta \cdot D_{\text{KL}}\left(\pi_\theta \| \pi_{\text{ref}}\right)}$$

where:
- $r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_\text{old}}(a_t \mid s_t)}$: probability ratio between new and old policy
- $A_t$: advantage estimate (estimated reward - baseline)
- $\text{clip}(\cdot, 1-\epsilon, 1+\epsilon)$: clipping for stability (typically $\epsilon = 0.2$)
- $\beta$: KL penalty coefficient
- $D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})$: KL divergence from the reference policy

**The clipping mechanism:** If $r_t > 1+\epsilon$, the policy has moved too far from the old policy; clip the objective to prevent taking steps that are too large. This is the "proximal" in PPO.

**The KL penalty:** Penalizes divergence from the reference (SFT) model. This prevents **reward hacking** — the policy cannot become arbitrarily different from the original model just to maximize the reward signal.

**What happens without KL penalty:** The LLM learns to exploit the reward model's weaknesses. Since the reward model is imperfect (trained on limited human preference data), the policy can find high-reward outputs that are nonsensical or harmful but happen to fool the reward model. This is reward hacking.

---

## 11. DPO (Direct Preference Optimization)

### 11.1 Motivation

RLHF with PPO is complex: it requires training a separate reward model, running online generation to collect trajectories, and careful PPO hyperparameter tuning. Can we derive a simpler objective that directly optimizes preference data without an explicit reward model?

**Yes — DPO (Rafailov et al., 2023).**

### 11.2 Full Derivation from First Principles

**Step 1: RLHF objective.** We want to find a policy that maximizes reward while staying close to the reference policy:

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D},\ y \sim \pi_\theta(y|x)}\left[r(x,y)\right] - \beta \cdot D_{\text{KL}}\left(\pi_\theta(y|x) \| \pi_{\text{ref}}(y|x)\right)$$

**Step 2: Derive the optimal policy.** This is a KL-constrained optimization problem. The optimal policy in closed form is:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$ is the partition function.

**Step 3: Solve for the reward.** Rearranging:

$$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**Step 4: Insert into Bradley-Terry.** The reward model loss uses $r(x, y_w) - r(x, y_l)$. The $\beta \log Z(x)$ terms cancel (same $x$!):

$$r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$$

**Step 5: Replace $\pi^*$ with $\pi_\theta$.** Since $\pi_\theta$ is our parameterized policy replacing the optimal $\pi^*$:

$$\boxed{\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]}$$

### 11.3 Intuition

The DPO loss has a beautiful interpretation:
- $\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$: the **implicit reward** — how much more likely the current policy assigns to $y$ compared to the reference.
- The loss increases the implicit reward for $y_w$ (preferred) and decreases it for $y_l$ (rejected), **relative to the reference policy**.
- The reference policy acts as a regularizer, preventing the model from diverging too far.

**Gradient analysis:**

$$\frac{\partial \mathcal{L}}{\partial \theta} = -\beta \cdot (1 - \sigma(\hat{r})) \cdot \left[\frac{\partial \log \pi_\theta(y_w|x)}{\partial \theta} - \frac{\partial \log \pi_\theta(y_l|x)}{\partial \theta}\right]$$

where $\hat{r} = \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}$ is the implicit reward gap.

- When the model already strongly prefers $y_w$ over $y_l$ ($\hat{r} \gg 0$): $\sigma(\hat{r}) \to 1$, $(1-\sigma(\hat{r})) \to 0$ — small gradient, model is already right.
- When the model prefers $y_l$ over $y_w$ ($\hat{r} \ll 0$): large gradient — strong learning signal.

### 11.4 Problems with DPO

1. **Distribution shift:** DPO is an offline method — it trains on a fixed dataset of preference pairs. The policy may diverge from the reference, and future preferences may not match offline data.

2. **Chosen reward hacking:** The model can increase $\log \pi_\theta(y_w|x)$ by simply making the model more likely for ALL sequences in a given context, inflating the likelihood of chosen responses without truly learning the preference.

3. **Data quality sensitivity:** Unlike PPO which can explore, DPO entirely depends on the quality of offline preference data. Noisy labels strongly affect the outcome.

4. **Length bias:** The model tends to learn to prefer shorter or longer responses based on superficial correlations in preference data, not true quality.

### 11.5 Variants

**IPO (Identity Preference Optimization, Azar et al., 2024):**

Replaces the $\log \sigma$ with a squared loss to avoid the overconfidence issue in DPO:

$$\mathcal{L}_{\text{IPO}} = \mathbb{E}\left[\left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} - \frac{1}{2\beta}\right)^2\right]$$

**KTO (Kahneman-Tversky Optimization, Ethayarajh et al., 2024):**

Uses prospect theory — humans are more sensitive to losses than gains. No need for paired preferences; can use unpaired good/bad examples.

**ORPO (Odds Ratio Preference Optimization, Hong et al., 2024):**

Eliminates the need for a reference model by using odds ratios between chosen and rejected directly. Simplifies the training setup.

### 11.6 Problems & Mitigations for DPO

| Problem | Description | Mitigation |
|---|---|---|
| Distribution shift | Offline data doesn't match current policy | Iterative DPO (re-collect preferences from $\pi_\theta$) |
| Chosen reward hacking | Inflate all likelihoods | NLL regularization on chosen responses |
| Data quality | Noisy preferences | Confidence filtering, Bradley-Terry filtering |
| Length bias | Learn length shortcuts | Length normalization in implicit reward |
| No reference model | IPO, ORPO avoid reference model | Use IPO/ORPO when no reference available |

---

## 12. Embedding Fine-tuning Summary Table

| Loss | Data Needed | Batch Requirement | Best For | Weakness |
|------|-------------|-------------------|---------|---------|
| Contrastive | Labeled pairs (y=0/1) | Small OK | Siamese networks, duplicate detection | Pair selection critical |
| Triplet | (anchor, pos, neg) triplets | Small OK | Metric learning, face recognition | $O(N^3)$ triplets, slow |
| InfoNCE/NT-Xent | Positive pairs | Large (>256) required | Large-scale self-supervised, CLIP | False negatives, memory |
| MNRL | Positive pairs | Large (>128) preferred | Retrieval fine-tuning (SBERT) | False negatives |
| CoSENT | Pairs + similarity scores | Small OK | STS (Semantic Textual Similarity) | Needs continuous scores |
| MSE | Pairs + similarity scores | Any | STS fine-tuning, score regression | Ignores relative ordering |
| Listwise (LambdaRank) | Full relevance grades | Any | Reranking, NDCG optimization | Data-intensive, complex |
| Margin MSE | (query, pos, neg) + teacher scores | Any | Cross-encoder → bi-encoder distillation | Needs teacher model |

---

## 13. Problems & Mitigations (Dedicated Section)

### 13.1 Mode Collapse in Contrastive Learning

**Problem:** All embeddings collapse to a single point (or a low-dimensional subspace), making all cosine similarities equal. Loss reaches a degenerate minimum where the denominator in InfoNCE equals the numerator.

**Detection:** Monitor the average pairwise cosine similarity in embeddings. If it approaches 1.0, collapse is occurring.

**Mitigations:**
- **Stop-gradient (BYOL):** Bootstrap Your Own Latent (Grill et al., 2020) uses an online network and a target network (exponential moving average). A stop-gradient on the target prevents the collapse because the gradient cannot flow back to trivialize both networks simultaneously.
- **Momentum encoder (MoCo):** A slowly-updating key encoder (momentum coefficient ~0.999) maintains a diverse queue of negatives that cannot immediately collapse.
- **Batch normalization in projector:** Prevents uniform collapse by normalizing the projection head outputs, ensuring variance is maintained.
- **Non-contrastive methods (SimSiam):** Explicitly uses stop-gradient to avoid needing negatives while preventing collapse.

### 13.2 False Negatives

**Problem:** In-batch negatives may be semantically similar to the anchor's positive. Treating them as negatives pushes apart semantically similar pairs, corrupting the embedding space.

**Mitigation 1: Deduplication.** Before training, deduplicate the dataset using MinHash LSH or embedding similarity. Ensure no near-duplicates appear in the same batch.

**Mitigation 2: False Negative Cancellation (FNC).** During training, compute similarity between all pairs. If a pair's similarity exceeds a threshold $\tau_\text{FNC}$, remove it from the denominator:

$$\mathcal{L}_i = -\log \frac{\exp(s_{ii}/\tau)}{\sum_j \exp(s_{ij}/\tau) \cdot \mathbf{1}[s_{ij} < \tau_\text{FNC}] + \exp(s_{ii}/\tau)}$$

**Mitigation 3: Supervised contrastive learning.** If class labels are available, use them to identify in-batch examples of the same class and exclude them from negatives.

### 13.3 Gradient Vanishing from Easy Negatives

**Problem:** In triplet/contrastive loss, easy negatives have zero gradient (they are already well-separated). Training stalls after the model solves the easy negatives.

**Mitigations:**
- **Hard negative mining:** Maintain a buffer of hard negatives from recent batches (online hard mining).
- **Focal weighting:** Apply a $(1-\hat{p})^\gamma$ type modulation to downweight easy and upweight hard negatives.
- **Temperature annealing:** Start with high temperature (all negatives equally hard) and gradually lower it (focus on hard negatives).
- **InfoNCE with low temperature:** At $\tau = 0.05$, the loss is naturally focused on hard negatives due to the exponential weighting.

### 13.4 Training Instability with Large Margins

**Problem:** A large margin in triplet/contrastive loss forces large gradient magnitudes on many active triplets simultaneously, causing loss spikes and divergence.

**Mitigations:**
- **Gradient clipping:** Clip gradients to $\|\nabla\|_2 \leq 1.0$ to prevent any single batch from causing huge updates.
- **Warm-up schedule:** Start with a small margin and gradually increase it, or start with a high learning rate warm-up to let embeddings organize first.
- **L2 normalization of embeddings:** Constrains the embedding space to the unit hypersphere, bounding distances to $[0, 2]$ and making margin hyperparameters more interpretable.
- **Curriculum learning:** Train on easy triplets first, then progressively introduce harder ones.

### 13.5 Reward Hacking in RLHF

**Problem:** The LLM learns to maximize the reward model's score without genuinely improving response quality. Since the reward model is imperfect (trained on limited human data), the policy exploits its weaknesses.

**Examples of reward hacking:**
- Generating very long responses (if the reward model conflates length with quality).
- Inserting phrases that frequently appear in highly-rated responses regardless of context.
- Adversarial strings that fool the reward model.

**Mitigations:**
- **KL constraint:** $\beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_\text{ref})$ in PPO prevents the policy from diverging too far.
- **Ensemble reward models:** Train multiple reward models; use the minimum or average score. Harder to hack consistently.
- **Constitutional AI / RLAIF:** Use a powerful AI to critique responses, providing more nuanced reward signals.
- **Iterative reward model updates:** Periodically retrain the reward model on generations from the current policy.
- **Length normalization:** Normalize reward by response length to prevent length exploitation.

---

## 14. Industry Practices

### 14.1 Which Loss for Which Task at Scale

**Query-document retrieval (e.g., Binance search, crypto news search):**
- Fine-tune bi-encoder with MNRL + hard negatives mined from BM25 top-K.
- Use cross-encoder teacher to generate soft scores for margin MSE distillation.
- Final system: bi-encoder for first-stage retrieval, cross-encoder for reranking.

**Semantic deduplication / duplicate detection:**
- Contrastive loss with online hard negative mining.
- Alternatively: SimCSE (in-batch negatives from dropout augmentation).

**Large-scale pretraining (text+image, multilingual):**
- InfoNCE / NT-Xent with batch size 4K–32K.
- GradCache to achieve large effective batch on single machines.

**Classification with imbalanced classes (fraud, anomaly):**
- Focal loss with $\gamma = 2$, class balance factor $\alpha$.
- Combined with oversampling (SMOTE) for severe imbalance.

**LLM alignment:**
- DPO for offline preference optimization (simpler setup, no reward model training).
- PPO+RLHF for online alignment where exploration matters.
- Iterative DPO for distribution shift mitigation.

### 14.2 Mining Hard Negatives from Production Query Logs

Production query logs contain real user behavior that is extremely valuable for hard negative mining:

1. **Clicked but not top-ranked:** Documents that appear in top-K BM25 but were not clicked by users are likely hard negatives — the retriever thinks they are relevant, but users disagree.
2. **Co-occurrence analysis:** Queries that co-occur frequently in sessions but have different results indicate hard negatives.
3. **Temporal mining:** Documents retrieved by the model but rated low by cross-encoder are hard negatives for model-aware training.

**Pipeline:**
```
Query logs → BM25 top-100 → Cross-encoder scoring →
Documents with high BM25 score but low CE score → Hard negatives
```

### 14.3 Mixing Multiple Losses with Annealing Schedules

In practice, multiple objectives are combined with time-varying weights:

**Phase 1 (warm-up, epochs 1–5):** Train with MNRL only (stable, no hard negatives). Let the encoder develop a basic embedding space.

**Phase 2 (refinement, epochs 6–15):** Add hard negatives mined from BM25. Switch from MNRL to MNRL + hard negative augmentation.

**Phase 3 (distillation, epochs 16–20):** Add margin MSE loss from cross-encoder teacher. This polishes the embedding space to match the cross-encoder's ranking.

**Annealing:** Gradually increase the weight on the distillation term:
$$\mathcal{L} = (1 - \lambda_t) \mathcal{L}_{\text{MNRL}} + \lambda_t \mathcal{L}_{\text{MarginMSE}}, \quad \lambda_t = \frac{t - 15}{5}$$

---

## 15. Interview Q&A

### Basic Level

**Q1: What is the connection between cross-entropy and maximum likelihood estimation?**

Cross-entropy $H(q, p) = -\sum_x q(x) \log p(x)$ is exactly the negative log-likelihood when $q$ is the empirical data distribution and $p$ is the model. Minimizing cross-entropy is therefore identical to maximizing the log-likelihood of the training data under the model. This connection means that any cross-entropy minimization implicitly assumes a probabilistic model and performs MLE under that model.

**Q2: What is label smoothing and why does it help?**

Label smoothing replaces one-hot targets $y_c \in \{0,1\}$ with soft targets $y_{\text{smooth},c} = (1-\epsilon)y_c + \epsilon/K$. It helps by: (1) preventing overconfidence — the model cannot achieve zero loss by making one class probability exactly 1; (2) improving calibration — predicted probabilities better reflect true accuracy; (3) acting as a regularizer equivalent to a KL term toward the uniform distribution. Typical values are $\epsilon \in [0.05, 0.2]$.

**Q3: What is the purpose of the margin in triplet loss?**

The margin $m$ enforces a minimum gap between the anchor-positive distance and the anchor-negative distance. Without a margin ($m=0$), the model can achieve zero loss by making all embeddings identical (degenerate solution). The margin forces the model to push negatives at least $\sqrt{m}$ further from anchors than positives, creating a more structured embedding space. The margin also controls the difficulty of the learning problem — larger margin requires more separation.

**Q4: Why does InfoNCE require a large batch size?**

InfoNCE uses in-batch negatives: the denominator of the softmax contains $2N-2$ negatives where $N$ is batch size. More negatives means: (1) harder learning task (must identify the positive among more distractors); (2) tighter lower bound on mutual information ($I(X;Y) \geq \mathcal{L}_{\text{InfoNCE}} + \log N$); (3) more diverse gradient signal since more types of negatives are seen per step. Small batches provide few negatives, making the task trivially easy and learning inefficient.

**Q5: What is the difference between pairwise, pointwise, and listwise losses?**

Pointwise losses score each document independently (regression/classification on individual items). Pairwise losses optimize the relative ordering of pairs (e.g., RankNet: "score A higher than B"). Listwise losses optimize the full ranked list directly (e.g., LambdaRank approximates NDCG, ListNet uses Plackett-Luce model over permutations). Pairwise and listwise methods are better aligned with ranking metrics; pointwise can work but requires accurate absolute scores.

### Intermediate Level

**Q6: Explain the difference between forward KL and reverse KL and their practical implications.**

Forward KL: $D_{\text{KL}}(P \| Q)$ — must cover all modes of $P$ (mean-seeking). Standard cross-entropy training uses forward KL; it produces "average" token predictions covering all likely next tokens.

Reverse KL: $D_{\text{KL}}(Q \| P)$ — can ignore modes of $P$ (mode-seeking). If $Q$ places mass where $P$ is small, the term blows up; if $Q$ ignores a mode of $P$ (probability 0), there is no penalty. This leads to $Q$ collapsing onto a single high-probability mode of $P$.

In RLHF, the KL penalty $D_{\text{KL}}(\pi_\theta \| \pi_\text{ref})$ is reverse KL from the policy's perspective — it penalizes the policy for exploring regions where the reference model has low probability, which constrains exploration and can contribute to reward hacking if the reference model is narrow.

**Q7: Explain hard negative mining strategies for triplet loss.**

Three categories of negatives: (1) Easy negatives: $d(a,n) > d(a,p) + m$ — zero gradient, useless; (2) Semi-hard negatives: $d(a,p) < d(a,n) < d(a,p) + m$ — positive loss, provides gradient, relatively safe since these negatives are still further than the positive; (3) Hard negatives: $d(a,n) < d(a,p)$ — highest gradient, but often include false negatives (mislabeled samples from the same class), so training on pure hard negatives can be unstable.

Best practice: semi-hard mining in the early training phase, transitioning to mixed semi-hard/hard mining as the model improves. For retrieval: mine hard negatives using BM25 (lexical retrieval finds hard false negatives for semantic models).

**Q8: Derive why the DPO loss does not require an explicit reward model.**

Starting from the KL-constrained RL objective $\max_\pi \mathbb{E}[r(x,y)] - \beta D_{\text{KL}}(\pi \| \pi_\text{ref})$, the optimal solution is $\pi^*(y|x) \propto \pi_\text{ref}(y|x) \exp(r(x,y)/\beta)$. Solving for $r$: $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_\text{ref}(y|x)} + \beta \log Z(x)$. Substituting into the Bradley-Terry preference model and noting that $\log Z(x)$ cancels for same-prompt pairs, we get: $P(y_w \succ y_l) = \sigma(\beta \log \frac{\pi^*(y_w|x)}{\pi_\text{ref}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_\text{ref}(y_l|x)})$. Replacing $\pi^*$ with $\pi_\theta$ and maximizing log-likelihood gives the DPO loss. The reward is **implicit** in the policy ratio — we never train a separate reward model.

**Q9: What is "dark knowledge" in knowledge distillation?**

Dark knowledge refers to the information embedded in a teacher model's soft probability outputs that is not present in hard one-hot labels. For example, for an image of a cat, the teacher might output [cat: 0.85, lynx: 0.10, dog: 0.04, automobile: 0.01]. The near-zero probability on "automobile" vs. the small but non-trivial probability on "lynx" and "dog" reveals the teacher's learned similarity structure — cats are more similar to lynx and dogs than to automobiles. This relational knowledge helps the student generalize better, especially for out-of-distribution examples, and is entirely absent from the hard one-hot label [cat: 1, other: 0].

**Q10: How does temperature affect InfoNCE loss?**

Temperature $\tau$ controls the sharpness of the similarity distribution in the softmax. Low $\tau$ (e.g., 0.05): the exponential function amplifies differences — the hardest negative (highest similarity) completely dominates the denominator, and all gradients flow from this single hard negative. High $\tau$ (e.g., 1.0): all negatives are treated nearly equally, providing a "democratic" but unfocused gradient signal. Optimal $\tau$ (typically 0.05–0.2 for text) focuses on hard negatives without being so sensitive that false negatives dominate. Temperature acts analogously to the inverse-temperature in statistical physics: low temperature concentrates probability on extreme states.

### Advanced Level

**Q11: Why might DPO suffer from "chosen reward hacking" and how would you mitigate it?**

In DPO, the implicit reward for a chosen response is $\beta \log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)}$. The model can increase this ratio not by genuinely improving $y_w$ but by making $\pi_\theta$ assign higher probability to all sequences in context $x$, effectively inflating likelihood broadly. This is because the DPO gradient increases $\log \pi_\theta(y_w|x)$ — but the model cannot distinguish between "be more likely for $y_w$ specifically" and "be more likely for everything in this context."

**Mitigations:**
1. Add an NLL regularization term: $\mathcal{L} = \mathcal{L}_{\text{DPO}} - \lambda \log \pi_\theta(y_w|x)$ which prevents the model from collapsing the implicit reward by inflating $\pi_\theta(y_w|x)$ beyond the reference level.
2. Use IPO which optimizes $(r_w - r_l - 1/(2\beta))^2$ directly, providing a more stable target.
3. Iterative DPO: re-collect preferences from the current $\pi_\theta$ to prevent the training distribution from diverging from the policy.

**Q12: How would you train a bi-encoder for financial document retrieval at Binance, where query logs are available but labeled query-document pairs are scarce?**

**Phase 1 — Unsupervised warm-up:** Fine-tune with SimCSE (in-batch negatives using dropout augmentation). This creates a reasonable embedding space without labeled data.

**Phase 2 — Weak supervision from query logs:** Use BM25 to retrieve top-K documents per query. Treat the clicked document as a positive, the non-clicked top-K documents as hard negatives. Fine-tune with MNRL + hard negatives.

**Phase 3 — Cross-encoder distillation:** Train a cross-encoder (BERT-based) on whatever labeled pairs are available (even a few thousand). Use it to score all (query, doc) pairs from query logs. Fine-tune the bi-encoder with Margin MSE loss using the cross-encoder scores as teacher signals.

**Phase 4 — Evaluation-guided mining:** After deployment, collect implicit feedback (clicks, time-spent, successful trades after search). Use this to iteratively improve hard negative quality.

**Loss combination:** MNRL + $\alpha$ × Margin MSE, with $\alpha$ annealed from 0 to 0.5 over training.

**Q13: Explain the relationship between InfoNCE and contrastive predictive coding (CPC).**

InfoNCE was originally proposed in Contrastive Predictive Coding (Oord et al., 2018) for self-supervised speech/image representation learning. The connection to mutual information is: maximizing InfoNCE is equivalent to maximizing a lower bound on $I(X; C)$ where $X$ is the future observation and $C$ is the context. The InfoNCE loss with $N$ negatives gives: $I(X;C) \geq \mathcal{L}_{\text{InfoNCE}} + \log N$. This bound tightens as $N \to \infty$, which is why larger batch sizes improve representation quality — they tighten the mutual information bound, forcing the encoder to capture more task-relevant information in the embeddings.

**Q14: Compare and contrast the gradient dynamics of focal loss vs. class-weighted cross-entropy.**

Class-weighted CE: $\mathcal{L} = -w_c \log \hat{p}_t$. The weight $w_c$ is static — it depends only on the class, not on how well the current example is classified.

Focal loss: $\mathcal{L} = -(1-\hat{p}_t)^\gamma \log \hat{p}_t$. The modulating factor $(1-\hat{p}_t)^\gamma$ is dynamic — it depends on the current model's prediction for this specific example.

**Key difference:** For an easy negative (well-classified, $\hat{p}_t = 0.95$):
- Weighted CE gradient: $w_c \cdot 0.05 / \hat{p}_t = w_c \cdot 0.053$ (non-negligible if $w_c$ is large)
- Focal gradient ($\gamma=2$): $(1-0.95)^2 \times 0.053 = 0.0025 \times 0.053 \approx 0.0001$ (100× smaller)

Focal loss dynamically reduces the gradient for examples the model handles well, regardless of class. Weighted CE reduces the gradient by a fixed factor for the majority class. In practice, focal loss is preferred when the hard/easy split cuts across class boundaries (e.g., some frauds are easy to detect, some are hard), while weighted CE is simpler and preferred when all samples of the rare class are genuinely hard.

**Q15: How would you implement a curriculum learning strategy for triplet loss training?**

Curriculum learning schedules training samples from easy to hard:

**Stage 1 (epochs 1–5, easy):** Use random triplets. The model learns basic cluster structure without being misled by hard negatives that may be false negatives.

**Stage 2 (epochs 6–15, semi-hard):** Switch to semi-hard online mining: for each anchor, find negatives satisfying $d(a,p) < d(a,n) < d(a,p) + m$. The model refines boundaries.

**Stage 3 (epochs 16+, hard):** Mix semi-hard (70%) and hard negatives (30%). Hard negatives provide maximum gradient but risk false negatives — the 70/30 mix limits this risk.

**Margin annealing:** Start with small margin $m_0 = 0.1$ and increase to $m_\text{final} = 0.5$ linearly. This prevents early training from seeing too many active triplets with random embeddings.

**Temperature annealing (for InfoNCE analog):** Start with $\tau = 0.5$ and anneal to $\tau = 0.05$, gradually focusing on harder negatives.

---

## 16. Coding Problems

### 16.1 Contrastive Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Siamese network contrastive loss (Hadsell et al., 2006).

    L = y * d^2 + (1-y) * max(0, margin - d)^2

    Args:
        margin: minimum distance for dissimilar pairs (default: 1.0)
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        emb1: torch.Tensor,  # (B, D)
        emb2: torch.Tensor,  # (B, D)
        labels: torch.Tensor  # (B,)  1=similar, 0=dissimilar
    ) -> torch.Tensor:
        # Compute pairwise L2 distances
        # Using squared distance for numerical stability before sqrt
        diff = emb1 - emb2  # (B, D)
        dist_sq = (diff ** 2).sum(dim=1)  # (B,)
        dist = torch.sqrt(dist_sq + 1e-8)  # (B,)  add eps to avoid grad issues at 0

        # Similar pair loss: pull together
        loss_similar = labels * dist_sq  # (B,)

        # Dissimilar pair loss: push apart (only if within margin)
        loss_dissim = (1 - labels) * F.relu(self.margin - dist) ** 2  # (B,)

        loss = 0.5 * (loss_similar + loss_dissim)
        return loss.mean()


# Usage example
if __name__ == "__main__":
    B, D = 32, 128
    emb1 = F.normalize(torch.randn(B, D), dim=1)
    emb2 = F.normalize(torch.randn(B, D), dim=1)
    # First half similar, second half dissimilar
    labels = torch.cat([torch.ones(B//2), torch.zeros(B//2)])

    criterion = ContrastiveLoss(margin=1.0)
    loss = criterion(emb1, emb2, labels)
    print(f"Contrastive Loss: {loss.item():.4f}")
```

### 16.2 Triplet Loss with Hard Negative Mining

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TripletLossWithMining(nn.Module):
    """
    Triplet loss with online hard/semi-hard negative mining.

    L = max(0, d(a,p)^2 - d(a,n)^2 + margin)

    Args:
        margin: separation margin (default: 0.2)
        mining: 'semi-hard', 'hard', or 'all'
        distance: 'euclidean' or 'cosine'
    """
    def __init__(
        self,
        margin: float = 0.2,
        mining: str = "semi-hard",
        distance: str = "cosine"
    ):
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.distance = distance

    def _pairwise_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise squared Euclidean distances."""
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        dot_product = embeddings @ embeddings.T  # (N, N)
        sq_norms = (embeddings ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        distances = sq_norms + sq_norms.T - 2 * dot_product  # (N, N)
        # Numerical stability: clamp negatives from floating point errors
        distances = F.relu(distances)
        return distances

    def _pairwise_cosine_sim(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise cosine similarities."""
        normed = F.normalize(embeddings, dim=1)
        return normed @ normed.T  # (N, N), values in [-1, 1]

    def forward(
        self,
        embeddings: torch.Tensor,  # (N, D) — all embeddings in the batch
        labels: torch.Tensor        # (N,)  — class labels
    ) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            loss: scalar triplet loss
            n_active: number of active (non-zero) triplets
        """
        N = embeddings.shape[0]

        if self.distance == "cosine":
            # Convert cosine similarity to distance: d = 1 - sim
            sim_matrix = self._pairwise_cosine_sim(embeddings)
            dist_matrix = 1.0 - sim_matrix  # (N, N)
        else:
            dist_matrix = self._pairwise_distances(embeddings)

        # Build masks
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)  # (N, N)
        not_same_idx = ~torch.eye(N, dtype=torch.bool, device=embeddings.device)

        # Positive mask: same class, not same index
        pos_mask = labels_equal & not_same_idx  # (N, N)
        # Negative mask: different class
        neg_mask = ~labels_equal  # (N, N)

        # For each anchor, get hardest positive (furthest positive)
        # and mine negatives based on strategy
        losses = []
        n_active = 0

        for anchor_idx in range(N):
            pos_indices = pos_mask[anchor_idx].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[anchor_idx].nonzero(as_tuple=True)[0]

            if len(pos_indices) == 0 or len(neg_indices) == 0:
                continue

            # Use hardest positive (maximum d(a,p))
            d_ap = dist_matrix[anchor_idx, pos_indices]
            hardest_pos_dist = d_ap.max()

            # Mine negatives
            d_an = dist_matrix[anchor_idx, neg_indices]

            if self.mining == "hard":
                # Hard negatives: d(a,n) < d(a,p)
                mask = d_an < hardest_pos_dist
                if mask.sum() == 0:
                    # Fall back to semi-hard if no hard negatives
                    mask = (d_an > hardest_pos_dist) & \
                           (d_an < hardest_pos_dist + self.margin)
                if mask.sum() == 0:
                    continue
                selected_d_an = d_an[mask].min()  # Use hardest of the hard

            elif self.mining == "semi-hard":
                # Semi-hard: d(a,p) < d(a,n) < d(a,p) + margin
                mask = (d_an > hardest_pos_dist) & \
                       (d_an < hardest_pos_dist + self.margin)
                if mask.sum() == 0:
                    continue
                selected_d_an = d_an[mask].min()

            else:  # 'all' — batch hard negative
                selected_d_an = d_an.min()

            triplet_loss = F.relu(hardest_pos_dist - selected_d_an + self.margin)
            losses.append(triplet_loss)
            if triplet_loss > 0:
                n_active += 1

        if len(losses) == 0:
            return torch.tensor(0.0, requires_grad=True, device=embeddings.device), 0

        return torch.stack(losses).mean(), n_active


# Usage example
if __name__ == "__main__":
    N, D = 64, 256
    embeddings = F.normalize(torch.randn(N, D), dim=1)
    # 8 classes, 8 samples each
    labels = torch.arange(8).repeat_interleave(8)

    criterion = TripletLossWithMining(margin=0.2, mining="semi-hard", distance="cosine")
    loss, n_active = criterion(embeddings, labels)
    print(f"Triplet Loss: {loss.item():.4f}, Active triplets: {n_active}")
```

### 16.3 InfoNCE / NT-Xent Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE / NT-Xent loss for contrastive representation learning.
    Used in SimCLR, CLIP, SBERT with in-batch negatives.

    L_i = -log [ exp(sim(z_i, z_i+)/tau) / sum_{k != i} exp(sim(z_i, z_k)/tau) ]

    Args:
        temperature: softmax temperature tau (default: 0.07)
        symmetric: if True, compute both directions (NT-Xent style)
        reduction: 'mean' or 'sum'
    """
    def __init__(
        self,
        temperature: float = 0.07,
        symmetric: bool = True,
        reduction: str = "mean"
    ):
        super().__init__()
        self.temperature = temperature
        self.symmetric = symmetric
        self.reduction = reduction

    def forward(
        self,
        z_i: torch.Tensor,  # (N, D) — first view embeddings
        z_j: torch.Tensor,  # (N, D) — second view / positive pair embeddings
    ) -> torch.Tensor:
        """
        z_i[k] and z_j[k] are positive pairs.
        All other (z_i, z_j) combinations are in-batch negatives.
        """
        N = z_i.shape[0]
        device = z_i.device

        # L2 normalize for cosine similarity
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate all 2N embeddings: [z_i_0, ..., z_i_{N-1}, z_j_0, ..., z_j_{N-1}]
        z = torch.cat([z_i, z_j], dim=0)  # (2N, D)

        # Compute all pairwise cosine similarities
        sim_matrix = z @ z.T / self.temperature  # (2N, 2N)

        # Mask out self-similarities (diagonal)
        mask_self = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask_self, -1e9)

        # Positive pairs:
        # - z_i[k] pairs with z_j[k] at position k+N
        # - z_j[k] pairs with z_i[k] at position k (if symmetric)
        # Labels for cross-entropy: position of positive in the 2N-1 remaining entries
        labels = torch.cat([
            torch.arange(N, 2 * N, device=device),  # z_i[k] -> z_j[k] at index k+N
            torch.arange(0, N, device=device)         # z_j[k] -> z_i[k] at index k
        ])  # (2N,)

        # Cross-entropy loss over 2N - 1 candidates (excluding self)
        # sim_matrix has -inf on diagonal, so softmax ignores self
        if self.symmetric:
            loss = F.cross_entropy(sim_matrix, labels, reduction=self.reduction)
        else:
            # Asymmetric: only compute loss for z_i side
            loss = F.cross_entropy(
                sim_matrix[:N],
                labels[:N],
                reduction=self.reduction
            )

        return loss


class NTXentLoss(InfoNCELoss):
    """NT-Xent: Normalized Temperature-scaled Cross-Entropy (SimCLR)."""
    def __init__(self, temperature: float = 0.07):
        super().__init__(temperature=temperature, symmetric=True)


# Usage example
if __name__ == "__main__":
    N, D = 256, 768
    # Simulate two augmented views of the same sentences
    z_i = torch.randn(N, D)
    z_j = z_i + 0.1 * torch.randn(N, D)  # Small perturbation = positive pair

    criterion = InfoNCELoss(temperature=0.07, symmetric=True)
    loss = criterion(z_i, z_j)
    print(f"InfoNCE Loss: {loss.item():.4f}")
    print(f"  (Random baseline would be log({2*N-1}) = {torch.log(torch.tensor(2*N-1.0)).item():.4f})")
```

### 16.4 Multiple Negatives Ranking Loss (MNRL)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultipleNegativesRankingLoss(nn.Module):
    """
    MNRL: Multiple Negatives Ranking Loss (Henderson et al., 2017; used in SBERT).

    Given N (anchor, positive) pairs, treats all other positives as negatives.
    Equivalent to cross-entropy on an N x N similarity matrix with diagonal labels.

    L = -1/N * sum_i log( exp(S_ii/tau) / sum_j exp(S_ij/tau) )

    Args:
        temperature: softmax temperature (default: 0.05)
        scale: if True, scale cosine similarity to [-1, 1] using F.normalize
    """
    def __init__(self, temperature: float = 0.05, scale: bool = True):
        super().__init__()
        self.temperature = temperature
        self.scale = scale

    def forward(
        self,
        anchors: torch.Tensor,    # (N, D)
        positives: torch.Tensor,  # (N, D)
        hard_negatives: torch.Tensor = None  # (N, D) optional hard negatives
    ) -> torch.Tensor:
        N = anchors.shape[0]
        device = anchors.device

        # Normalize for cosine similarity
        if self.scale:
            anchors = F.normalize(anchors, dim=1)
            positives = F.normalize(positives, dim=1)

        # Similarity matrix: S[i,j] = sim(anchor_i, positive_j)
        sim_matrix = anchors @ positives.T / self.temperature  # (N, N)

        # Augment with hard negatives if provided
        if hard_negatives is not None:
            if self.scale:
                hard_negatives = F.normalize(hard_negatives, dim=1)
            # Hard neg similarities: S_hn[i] = sim(anchor_i, hard_neg_i)
            sim_hard = (anchors * hard_negatives).sum(dim=1, keepdim=True) / self.temperature
            # Concatenate to similarity matrix: now (N, N+1)
            # The hard negative for anchor i is at position N (last column)
            # But we need to be careful: hard_neg_i should only be negative for anchor_i
            # Standard approach: add hard negs as extra negatives in denominator
            # Simple approach: diagonal hard negatives (hard_neg_i for anchor_i)
            sim_matrix = torch.cat([sim_matrix, sim_hard], dim=1)  # (N, N+1)

        # Labels: diagonal (anchor_i matches positive_i, i.e., column i for row i)
        labels = torch.arange(N, device=device)  # (N,)

        # Cross-entropy: equivalent to softmax over all positives/negatives
        loss = F.cross_entropy(sim_matrix, labels)
        return loss


# Usage example with SBERT-like setup
if __name__ == "__main__":
    N, D = 128, 768  # 128 query-doc pairs, 768-dim embeddings

    # Simulate bi-encoder outputs
    query_embs = torch.randn(N, D)
    doc_embs = query_embs + 0.2 * torch.randn(N, D)  # Positive docs
    hard_neg_embs = query_embs + 0.5 * torch.randn(N, D)  # Hard negatives

    criterion = MultipleNegativesRankingLoss(temperature=0.05)

    # Without hard negatives
    loss_basic = criterion(query_embs, doc_embs)
    print(f"MNRL Loss (basic): {loss_basic.item():.4f}")

    # With hard negatives
    loss_hn = criterion(query_embs, doc_embs, hard_neg_embs)
    print(f"MNRL Loss (with hard negs): {loss_hn.item():.4f}")
```

### 16.5 DPO Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def compute_log_probs(
    logits: torch.Tensor,     # (B, L, V)
    labels: torch.Tensor,     # (B, L)
    attention_mask: torch.Tensor  # (B, L)
) -> torch.Tensor:
    """
    Compute per-token log probabilities and sum over sequence length.
    Returns: (B,) scalar log prob per sequence.
    """
    # Shift for causal LM: predict token t+1 from t
    logits = logits[:, :-1, :]    # (B, L-1, V)
    labels = labels[:, 1:]         # (B, L-1)
    mask = attention_mask[:, 1:]   # (B, L-1)

    # Log-softmax over vocabulary
    log_probs = F.log_softmax(logits, dim=-1)  # (B, L-1, V)

    # Gather log prob of actual tokens
    token_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)  # (B, L-1)

    # Mask padding and sum
    token_log_probs = token_log_probs * mask
    return token_log_probs.sum(dim=-1)  # (B,)


class DPOLoss(nn.Module):
    """
    Direct Preference Optimization loss (Rafailov et al., 2023).

    L_DPO = -E[ log sigma( beta * (log pi(y_w|x)/pi_ref(y_w|x)
                                  - log pi(y_l|x)/pi_ref(y_l|x)) ) ]

    Requires the policy model (trainable) and reference model (frozen).

    Args:
        beta: KL regularization coefficient (typical: 0.1-0.5)
        label_smoothing: smoothing for the sigmoid (IPO-like, default: 0.0)
    """
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,    # (B,) log pi_theta(y_w|x)
        policy_rejected_logps: torch.Tensor,  # (B,) log pi_theta(y_l|x)
        ref_chosen_logps: torch.Tensor,       # (B,) log pi_ref(y_w|x)
        ref_rejected_logps: torch.Tensor,     # (B,) log pi_ref(y_l|x)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            loss: scalar DPO loss
            chosen_rewards: (B,) implicit rewards for chosen
            rejected_rewards: (B,) implicit rewards for rejected
        """
        # Implicit rewards: beta * log(pi_theta / pi_ref)
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)    # (B,)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)  # (B,)

        # Reward margin: r(y_w) - r(y_l)
        reward_margin = chosen_rewards - rejected_rewards  # (B,)

        # DPO loss: -log sigma(reward_margin)
        if self.label_smoothing > 0.0:
            # IPO-style: use squared loss instead
            loss = (reward_margin - 1.0 / (2 * self.beta)) ** 2
        else:
            # Standard DPO: binary cross-entropy with margin
            loss = -F.logsigmoid(reward_margin)

        return loss.mean(), chosen_rewards.detach(), rejected_rewards.detach()


# Full training loop example
class DPOTrainer:
    """
    Minimal DPO training demonstration.
    In practice, use HuggingFace TRL's DPOTrainer.
    """
    def __init__(self, policy_model, ref_model, beta: float = 0.1):
        self.policy = policy_model
        self.ref = ref_model
        self.criterion = DPOLoss(beta=beta)
        self.ref.eval()  # Reference model is frozen
        for p in self.ref.parameters():
            p.requires_grad_(False)

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        batch contains:
            chosen_input_ids: (B, L_c)
            chosen_attention_mask: (B, L_c)
            chosen_labels: (B, L_c)
            rejected_input_ids: (B, L_r)
            rejected_attention_mask: (B, L_r)
            rejected_labels: (B, L_r)
        """
        # Policy forward pass (with gradients)
        policy_chosen_logits = self.policy(
            input_ids=batch["chosen_input_ids"],
            attention_mask=batch["chosen_attention_mask"]
        ).logits

        policy_rejected_logits = self.policy(
            input_ids=batch["rejected_input_ids"],
            attention_mask=batch["rejected_attention_mask"]
        ).logits

        # Reference forward pass (no gradients)
        with torch.no_grad():
            ref_chosen_logits = self.ref(
                input_ids=batch["chosen_input_ids"],
                attention_mask=batch["chosen_attention_mask"]
            ).logits

            ref_rejected_logits = self.ref(
                input_ids=batch["rejected_input_ids"],
                attention_mask=batch["rejected_attention_mask"]
            ).logits

        # Compute log probabilities
        policy_chosen_logps = compute_log_probs(
            policy_chosen_logits,
            batch["chosen_labels"],
            batch["chosen_attention_mask"]
        )
        policy_rejected_logps = compute_log_probs(
            policy_rejected_logits,
            batch["rejected_labels"],
            batch["rejected_attention_mask"]
        )
        ref_chosen_logps = compute_log_probs(
            ref_chosen_logits,
            batch["chosen_labels"],
            batch["chosen_attention_mask"]
        )
        ref_rejected_logps = compute_log_probs(
            ref_rejected_logits,
            batch["rejected_labels"],
            batch["rejected_attention_mask"]
        )

        loss, chosen_rewards, rejected_rewards = self.criterion(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps
        )

        # Log reward accuracy: how often is chosen reward > rejected reward?
        reward_accuracy = (chosen_rewards > rejected_rewards).float().mean()

        return loss, reward_accuracy


# Standalone DPO loss test
if __name__ == "__main__":
    B = 16
    # Simulate log probabilities (negative values, as log probs are <= 0)
    policy_chosen_logps = -torch.rand(B) * 50 - 10     # (B,)
    policy_rejected_logps = -torch.rand(B) * 50 - 15   # (B,) slightly lower
    ref_chosen_logps = -torch.rand(B) * 50 - 12
    ref_rejected_logps = -torch.rand(B) * 50 - 13

    criterion = DPOLoss(beta=0.1)
    loss, chosen_r, rejected_r = criterion(
        policy_chosen_logps, policy_rejected_logps,
        ref_chosen_logps, ref_rejected_logps
    )
    print(f"DPO Loss: {loss.item():.4f}")
    print(f"Mean chosen reward: {chosen_r.mean().item():.4f}")
    print(f"Mean rejected reward: {rejected_r.mean().item():.4f}")
    print(f"Reward accuracy: {(chosen_r > rejected_r).float().mean().item():.2%}")
```

### 16.6 Knowledge Distillation Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class KnowledgeDistillationLoss(nn.Module):
    """
    Hinton et al. (2015) Knowledge Distillation loss.

    L = alpha * CE(y, y_hat_student) + (1 - alpha) * T^2 * KL(softmax(z_t/T), softmax(z_s/T))

    The T^2 factor compensates for reduced gradient magnitudes at high temperature.

    Args:
        temperature: distillation temperature T (default: 4.0)
        alpha: weight for hard label loss (default: 0.5)
    """
    def __init__(self, temperature: float = 4.0, alpha: float = 0.5):
        super().__init__()
        self.T = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,  # (B, K) raw logits from student
        teacher_logits: torch.Tensor,  # (B, K) raw logits from teacher
        labels: torch.Tensor           # (B,) hard ground truth labels
    ) -> torch.Tensor:
        """
        Returns combined distillation + hard label loss.
        """
        # Hard label loss: standard cross-entropy at temperature 1
        loss_hard = F.cross_entropy(student_logits, labels)

        # Soft label loss: KL divergence at temperature T
        # Teacher soft targets
        teacher_soft = F.softmax(teacher_logits / self.T, dim=-1)  # (B, K)
        # Student log-softmax at temperature T
        student_log_soft = F.log_softmax(student_logits / self.T, dim=-1)  # (B, K)

        # KL divergence: KL(p_teacher || p_student) = sum p_t * (log p_t - log p_s)
        # F.kl_div expects log-probabilities for input, probabilities for target
        loss_soft = F.kl_div(
            student_log_soft,
            teacher_soft,
            reduction="batchmean"
        )

        # T^2 scaling compensates for 1/T^2 gradient scaling at high temperature
        loss_soft = loss_soft * (self.T ** 2)

        # Combined loss
        loss = self.alpha * loss_hard + (1 - self.alpha) * loss_soft
        return loss


class MarginMSELoss(nn.Module):
    """
    Margin MSE loss for cross-encoder → bi-encoder distillation.

    Preserves relative ordering (margin) from teacher scores.

    L = MSE( sim(q, d+) - sim(q, d-), score_teacher(q, d+) - score_teacher(q, d-) )

    Reference: Hofstatter et al. (2021), "Efficiently Teaching an Effective Dense Retriever"
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query_embs: torch.Tensor,      # (B, D)
        pos_doc_embs: torch.Tensor,    # (B, D)
        neg_doc_embs: torch.Tensor,    # (B, D)
        teacher_pos_scores: torch.Tensor,  # (B,) cross-encoder scores for (q, d+)
        teacher_neg_scores: torch.Tensor,  # (B,) cross-encoder scores for (q, d-)
    ) -> torch.Tensor:
        # Compute student (bi-encoder) cosine similarities
        query_embs = F.normalize(query_embs, dim=1)
        pos_doc_embs = F.normalize(pos_doc_embs, dim=1)
        neg_doc_embs = F.normalize(neg_doc_embs, dim=1)

        student_pos_sim = (query_embs * pos_doc_embs).sum(dim=1)  # (B,)
        student_neg_sim = (query_embs * neg_doc_embs).sum(dim=1)  # (B,)

        # Student margin: how much more similar is the positive?
        student_margin = student_pos_sim - student_neg_sim  # (B,)

        # Teacher margin: same concept from cross-encoder scores
        teacher_margin = teacher_pos_scores - teacher_neg_scores  # (B,)

        # MSE on margins: train student to reproduce teacher's relative ordering
        loss = F.mse_loss(student_margin, teacher_margin)
        return loss


# Usage example: distilling BERT-large → DistilBERT style
if __name__ == "__main__":
    B, K = 32, 10  # Batch size, number of classes

    # Simulate teacher (stronger model) and student (smaller model) logits
    teacher_logits = torch.randn(B, K) * 2  # Teacher more confident
    student_logits = torch.randn(B, K)
    labels = torch.randint(0, K, (B,))

    kd_criterion = KnowledgeDistillationLoss(temperature=4.0, alpha=0.5)
    kd_loss = kd_criterion(student_logits, teacher_logits, labels)
    print(f"KD Loss: {kd_loss.item():.4f}")

    # Margin MSE example for retrieval distillation
    D = 768
    query_embs = torch.randn(B, D)
    pos_doc_embs = torch.randn(B, D)
    neg_doc_embs = torch.randn(B, D)
    teacher_pos_scores = torch.rand(B) * 3 + 1  # High relevance
    teacher_neg_scores = torch.rand(B) * 1      # Low relevance

    margin_mse = MarginMSELoss()
    mmse_loss = margin_mse(query_embs, pos_doc_embs, neg_doc_embs,
                           teacher_pos_scores, teacher_neg_scores)
    print(f"Margin MSE Loss: {mmse_loss.item():.4f}")
```

### 16.7 Focal Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal loss for class-imbalanced problems (Lin et al., RetinaNet 2017).

    L_focal = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma: focusing parameter (default: 2.0)
        alpha: class balance weights. If float, used as weight for class 1 in binary.
               If list/tensor, per-class weights for multi-class.
        reduction: 'mean', 'sum', or 'none'
    """
    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is not None:
            if isinstance(alpha, (float, int)):
                self.alpha = torch.tensor([1 - alpha, alpha])
            else:
                self.alpha = torch.tensor(alpha)
        else:
            self.alpha = None

    def forward(
        self,
        inputs: torch.Tensor,   # (B, C) logits or (B,) for binary
        targets: torch.Tensor   # (B,) class indices
    ) -> torch.Tensor:
        B = inputs.shape[0]

        # Compute standard cross-entropy (per-element, no reduction)
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")  # (B,)

        # Compute p_t: probability of the true class
        probs = F.softmax(inputs, dim=-1)  # (B, C)
        p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)

        # Modulating factor
        focal_weight = (1 - p_t) ** self.gamma  # (B,)

        # Alpha balancing
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]  # (B,)
            focal_weight = alpha_t * focal_weight

        focal_loss = focal_weight * ce_loss  # (B,)

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# Usage example: fraud detection
if __name__ == "__main__":
    B = 1000
    # Severe class imbalance: 990 non-fraud, 10 fraud
    targets = torch.zeros(B, dtype=torch.long)
    targets[:10] = 1  # 10 fraud cases

    logits = torch.randn(B, 2)

    # Standard CE — dominated by easy non-fraud examples
    ce_loss = F.cross_entropy(logits, targets)

    # Focal loss — downweights easy non-fraud examples
    focal = FocalLoss(gamma=2.0, alpha=0.25)  # alpha=0.25 for class 1 (fraud)
    focal_loss = focal(logits, targets)

    print(f"Standard CE Loss:  {ce_loss.item():.4f}")
    print(f"Focal Loss (γ=2):  {focal_loss.item():.4f}")
```

---

## Summary and Key Takeaways

1. **All major losses derive from MLE:** Cross-entropy, contrastive, InfoNCE, DPO — all are special cases of maximizing log-likelihood under different model assumptions.

2. **Negative mining is the central engineering challenge** in contrastive/triplet learning. Easy negatives give no gradient. Hard negatives risk false negatives. Semi-hard mining is the pragmatic middle ground.

3. **Temperature is a critical hyperparameter** in InfoNCE/MNRL. It controls the focus on hard negatives. $\tau \approx 0.05$ is a common starting point for text.

4. **DPO eliminates the explicit reward model** by showing that the optimal RLHF policy implicitly encodes the reward in the policy ratio $\pi_\theta / \pi_\text{ref}$.

5. **Knowledge distillation transfers dark knowledge** — the teacher's soft probability distribution encodes inter-class similarity that one-hot labels cannot express.

6. **Focal loss is gradient reweighting** — it dynamically downweights easy examples based on the current model's confidence, making it adaptive in a way static class weights cannot be.

7. **The KL penalty in RLHF prevents reward hacking** by keeping the policy close to the reference model, limiting how far the LLM can deviate to exploit reward model weaknesses.

8. **In industry, losses are combined and annealed:** Start with stable broad losses (MNRL), add hard negatives progressively, then add distillation from a stronger teacher.
