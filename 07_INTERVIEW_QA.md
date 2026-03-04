# Binance Data Scientist Intern — Technical Interview Q&A

> 60+ hard questions with full answers. Covers embeddings, loss functions, rerankers, RAG, fine-tuning, architecture, system design, and coding.

---

## Part 1: Embeddings

---

### Q1. Why do raw BERT `[CLS]` embeddings perform worse than averaged GloVe embeddings on semantic textual similarity? This seems counterintuitive — BERT is a much stronger model.

**A:** This is one of the most important results in the embedding literature (Reimers & Gurevych, 2019). The cause is **anisotropy**.

BERT was pretrained with MLM (Masked Language Modeling), which optimizes a token-level prediction objective. The `[CLS]` token was trained for Next Sentence Prediction (NSP) — a binary "does sentence B follow sentence A?" task — not for producing a general-purpose sentence representation.

The deeper issue is geometric. The MLM objective pushes token representations to be useful for predicting masked tokens via a softmax over the vocabulary:

$$P(w \mid h) = \frac{\exp(e_w^T h)}{\sum_{w'} \exp(e_{w'}^T h)}$$

This creates a **narrow cone** in the embedding space — the model needs representations to be close to a large number of word embedding vectors in the output layer, which constrains the space. Ethayarajh (2019) showed that the average cosine similarity between random BERT sentence embeddings is ~0.6–0.99 (depending on layer), meaning almost all pairs look "similar." When your baseline similarity is already 0.85, the signal-to-noise ratio for distinguishing semantically similar vs. dissimilar sentences is terrible.

GloVe averaging doesn't suffer from this because GloVe vectors are trained on co-occurrence statistics without a softmax bottleneck — the space is more isotropic.

**Fix:** SBERT (Sentence-BERT) fine-tunes BERT with a Siamese architecture using contrastive/NLI objectives, which explicitly pushes similar sentences together and dissimilar sentences apart, producing an isotropic, discriminative embedding space.

**Key insight:** A strong model doesn't automatically produce strong embeddings — the training objective must align with the downstream use case.

---

### Q2. Explain the mathematical connection between Word2Vec skip-gram with negative sampling and implicit matrix factorization.

**A:** Levy & Goldberg (2014) proved that when skip-gram with negative sampling (SGNS) converges, the word vectors implicitly factorize a **shifted PMI (Pointwise Mutual Information) matrix**.

The SGNS objective for a (word, context) pair $(w, c)$ with $k$ negative samples is:

$$\mathcal{L} = \log \sigma(\vec{w} \cdot \vec{c}) + k \cdot \mathbb{E}_{c_n \sim P_n} [\log \sigma(-\vec{w} \cdot \vec{c}_n)]$$

At the global optimum, taking the derivative w.r.t. $\vec{w} \cdot \vec{c}$ and setting it to zero:

$$\sigma(\vec{w} \cdot \vec{c}) = \frac{\#(w,c)}{\#(w,c) + k \cdot \frac{\#(w) \cdot \#(c)}{|D|}}$$

Solving for the dot product:

$$\vec{w} \cdot \vec{c} = \log \frac{\#(w,c) \cdot |D|}{\#(w) \cdot \#(c)} - \log k = \text{PMI}(w,c) - \log k$$

So at convergence:

$$W \cdot C^T = M_{\text{SPMI}}$$

where $M_{\text{SPMI}}[i,j] = \max(\text{PMI}(w_i, c_j) - \log k, \; 0)$ (PPMI with shift). The word and context embedding matrices together approximate a low-rank factorization of this shifted PMI matrix.

This explains why both neural (Word2Vec) and count-based (SVD on PMI matrix) methods produce comparable embeddings — they're both extracting the same statistical signal, just via different optimization paths.

**Key insight:** Word2Vec is not "magic" — it's doing matrix factorization. Understanding this reveals that the quality of embeddings fundamentally depends on co-occurrence statistics.

---

### Q3. You have a bi-encoder that gives cosine similarity 0.87 for (query, doc_A) and 0.84 for (query, doc_B). A cross-encoder reranker reverses the ranking, giving doc_B a higher score. Which should you trust, and why?

**A:** Trust the **cross-encoder**.

The bi-encoder encodes query and document **independently** — the query embedding has no knowledge of the document, and vice versa. The relevance score is computed as a simple dot product / cosine between two vectors. This means the bi-encoder cannot capture fine-grained token-level interactions between the query and document.

The cross-encoder concatenates query and document as a single input: `[CLS] query [SEP] document [SEP]`, and runs full cross-attention. Every query token can attend to every document token. This captures:

- Exact term matching nuances
- Negation ("BTC is **not** a security" vs. "BTC is a security")
- Complex compositional semantics
- Specificity (query asks about "Binance staking rewards for ETH" — doc A mentions staking generally, doc B mentions ETH staking specifically)

The quality gap is well-documented. On MS MARCO, cross-encoders typically outperform bi-encoders by 5–15 points in MRR@10.

The 0.87 vs 0.84 difference in bi-encoder scores is within the margin of error for bi-encoders on fine-grained ranking. Cross-encoder scores are more reliable for pairwise comparisons.

**Caveat:** If the cross-encoder was not trained on your domain, domain mismatch could be an issue. Also check that the cross-encoder's max sequence length isn't truncating the documents.

**Key insight:** Bi-encoders are for recall (find 1000 candidates from millions); cross-encoders are for precision (rerank top candidates). When they disagree, the cross-encoder is almost always right.

---

### Q4. What is Matryoshka Representation Learning? Why does it work?

**A:** MRL (Kusupati et al., 2022) trains embeddings so that **any prefix** of the full-dimensional vector is itself a valid, useful embedding.

**Training:** Compute the contrastive loss at multiple truncation levels simultaneously:

$$\mathcal{L}_{\text{MRL}} = \sum_{d' \in \{32, 64, 128, 256, 512, ..., d\}} \lambda_{d'} \cdot \mathcal{L}_{\text{contrastive}}\!\left(\frac{f(x)[:d']}{\|f(x)[:d']\|}, \; \frac{f(y)[:d']}{\|f(y)[:d']\|}\right)$$

Each prefix is normalized independently before computing the loss. Typically $\lambda_{d'}$ is uniform (all truncation levels weighted equally).

**Why it works:** The multi-scale loss creates a natural information hierarchy. The first few dimensions must carry the most globally discriminative information because they contribute to losses at every truncation level. Dimensions further in carry progressively finer-grained distinctions. This is analogous to PCA — the first few principal components capture the most variance.

**Practical benefit:** At inference time, you choose the truncation level based on your latency/quality budget:
- **d'=64**: very fast approximate pre-filtering (Hamming distance on binary quantized)
- **d'=256**: good general-purpose search (4× less storage than full)
- **d'=1024**: high-precision final scoring

OpenAI's `text-embedding-3-small/large` supports this — you set `dimensions=256` in the API call and get a valid 256-dim embedding that's a prefix of the full vector.

**Key insight:** MRL frontloads information into early dimensions, giving you a single model that serves multiple latency/quality tradeoffs without retraining.

---

### Q5. How would you detect and handle embedding drift in a production system that serves 100M queries/day?

**A:** Embedding drift occurs when the embedding model is updated (retrained, fine-tuned, or version-bumped) and old embeddings in the index are no longer compatible with new query embeddings.

**Detection:**
1. **Canary set monitoring:** Maintain a fixed set of ~1000 (query, relevant_doc) pairs with known ground truth. After any model update, compute recall@10 on this canary set. If recall drops > 2%, flag drift.
2. **Distribution shift detection:** Track the distribution of cosine similarities for random (query, retrieved_doc) pairs. If the mean similarity shifts significantly (KS test p < 0.01), the embedding space has changed.
3. **Online A/B metrics:** Click-through rate, user satisfaction signals degrade when embeddings drift.

**Handling strategies:**
1. **Full re-indexing** (simplest, most expensive): Re-embed entire corpus with new model. For 100M docs at 1536 dims, this takes ~8 hours on 8 A100s. Do offline, hot-swap index.
2. **Versioned indices:** Maintain model version alongside each embedding. Route queries to the matching-version index. Gradually re-embed old docs in background.
3. **Backward-compatible training:** Add a consistency loss during model training that penalizes large divergence from the previous model's embedding space:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{contrastive}} + \alpha \cdot \sum_i \|f_{\text{new}}(x_i) - f_{\text{old}}(x_i)\|^2$$

This constrains the new model to stay close to the old embedding space while still improving.
4. **Linear alignment:** Learn a linear projection $W$ that maps old embeddings to the new space: $e_{\text{aligned}} = W e_{\text{old}}$. Train $W$ on a small calibration set. Cheap ($d^2$ parameters), works well when drift is moderate.

**At Binance scale (100M queries/day):** Use versioned indices + linear alignment as a bridge during re-indexing. Full re-index runs nightly. Canary set checks before any index swap.

**Key insight:** Embedding drift is a production reality. Plan for it from the start with versioning and monitoring — don't wait until quality degrades.

---

### Q6. Explain in-batch negatives. Why does a larger batch size improve contrastive learning, and is there a diminishing return?

**A:** In-batch negatives is a training strategy where, within a batch of $B$ positive (query, document) pairs, every other document serves as a negative for each query. One batch of $B$ pairs yields $B$ positive examples and $B(B-1)$ negative examples — quadratic negative scaling with batch size, for free.

The InfoNCE loss with in-batch negatives for query $i$:

$$\mathcal{L}_i = -\log \frac{\exp(\text{sim}(q_i, d_i^+) / \tau)}{\sum_{j=1}^{B} \exp(\text{sim}(q_i, d_j) / \tau)}$$

**Why larger batch helps:** More negatives tighten the lower bound on mutual information:

$$\mathcal{L}_{\text{InfoNCE}} \geq \log(B) - I(Q; D)$$

With $B=32$, the denominator contains 32 candidates — it's easy to distinguish the positive from 31 random documents. With $B=65536$ (CLIP's batch size), the task is much harder, forcing finer-grained representations.

**Diminishing returns:** Yes.
1. **Statistical:** Once $B$ is large enough that the hardest negatives in the batch are genuinely hard, adding more random negatives doesn't help much. The gradient is dominated by the hardest few negatives.
2. **False negatives:** As $B$ grows, the probability of accidentally sampling a semantically similar document as a negative increases. If your corpus has 10 documents about "ETH staking rewards," a batch of 64K will likely include multiple of them, creating false negatives that corrupt the gradient.
3. **Compute:** Batch size is limited by GPU memory. The similarity matrix is $B \times B$.

**Practical workaround:** GradCache (Gao et al., 2021) decouples gradient computation from memory — it computes embeddings in small chunks but accumulates gradients as if the batch were much larger. This enables effective batch sizes of 32K+ on a single GPU.

**Key insight:** In-batch negatives give you quadratic scaling of negative examples for free. Large batch size improves contrastive learning up to a point, after which false negatives and diminishing gradient signal limit returns.

---

### Q7. What's the difference between cosine similarity and dot product in practice? When does the choice matter fundamentally?

**A:** For L2-normalized vectors (unit vectors), cosine and dot product are identical:

$$\cos(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|} = \mathbf{u} \cdot \mathbf{v} \quad \text{when } \|\mathbf{u}\| = \|\mathbf{v}\| = 1$$

The choice matters when **magnitude carries information.**

**Cosine similarity** only measures direction. Two documents — one 3 words long, one 3000 words — could have identical cosine similarity to a query if they point in the same direction, even though the long document likely has richer, more confident information.

**Dot product** measures both direction and magnitude. If the model learns to encode "confidence" or "importance" in the vector norm, dot product preserves this signal. For example, a document embedding with $\|\mathbf{d}\| = 2.5$ is "more about" its topic than one with $\|\mathbf{d}\| = 0.8$.

**When it matters fundamentally:**
1. **Recommendation systems:** User/item embeddings often encode popularity in magnitude. Dot product surfaces popular items; cosine surfaces niche but topically relevant ones.
2. **MIPS (Maximum Inner Product Search):** Some models are explicitly trained for dot product (not cosine). Using cosine would discard the magnitude signal the model learned.
3. **Asymmetric search:** Query embeddings might be short (few tokens → smaller norm), document embeddings longer (larger norm). Cosine normalizes this away; dot product doesn't.

**Practical check:** If your model documentation says "normalize before computing similarity," use cosine. If it says "use inner product," don't normalize. When in doubt, evaluate both on a held-out set.

**Key insight:** The choice is only meaningful when vector norms carry information. Most modern embedding models normalize internally or recommend normalization, making the choice moot. But for models that don't (like some recommendation embeddings), the difference is critical.

---

### Q8. How does CLIP prevent embedding collapse — where all images and texts map to the same point?

**A:** Embedding collapse is a real risk in contrastive learning. If all embeddings converge to a single point $\mathbf{c}$, then every pair has similarity $\|\mathbf{c}\|^2$ — the loss becomes constant and no learning happens.

CLIP prevents this through several mechanisms:

1. **Symmetric loss structure:** The loss has two components — image-to-text and text-to-image classification. Each row AND each column of the similarity matrix must select the correct match. If all embeddings collapsed, every row and column would be uniform — the loss would be $\log(B)$ (maximum), not minimized.

2. **Temperature scaling:** CLIP uses a **learned** temperature $\tau$:
$$\mathcal{L} = -\frac{1}{2}\sum_i \left[\log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ij}/\tau}} + \log \frac{e^{s_{ii}/\tau}}{\sum_j e^{s_{ji}/\tau}}\right]$$
A low $\tau$ sharpens the distribution, creating strong gradients away from uniform (collapsed) solutions. $\tau$ is initialized at ~0.07 and learned — the model finds the right sharpness.

3. **Large batch size (32K):** With 32K negative pairs, the model must discriminate among many candidates. Collapse would make discrimination impossible, producing maximum loss.

4. **L2 normalization:** Both image and text embeddings are projected to the unit sphere. This prevents a trivial collapse where norms go to infinity or zero — the model must use the angular structure of the sphere.

5. **Diverse data:** 400M (image, text) pairs from the internet. The sheer diversity of concepts prevents the model from finding a single "average" representation that works.

**Key insight:** The combination of symmetric cross-entropy loss + low temperature + large batch size + normalization makes collapse a high-loss state that gradient descent moves away from. No single mechanism prevents it — it's the system design.

---

### Q9. Design an embedding pipeline for 500M crypto transaction descriptions at Binance. Cover model, index, and serving.

**A:**

**Requirements analysis:**
- 500M documents, growing ~1M/day
- Queries: natural language ("show me large ETH transfers to exchanges in the last week")
- Latency: < 50ms p99 for search
- Languages: primarily English, Chinese, Korean, Japanese

**Model selection:**
- Use `multilingual-e5-large` (1024 dims) — strong multilingual performance, instruction-aware prefixes (`query:` / `passage:`), open-source
- Fine-tune on 50K labeled (query, transaction_description) pairs mined from Binance support logs
- Output: 1024-dim normalized embeddings

**Indexing architecture:**

```
500M docs × 1024 dims × float32 = 2 TB (raw)

Strategy: IVF-PQ (Inverted File + Product Quantization)
- IVF: nlist = 32768 clusters (sqrt(500M) ≈ 22K, round up)
- PQ:  m = 64 sub-quantizers, 8 bits each → 64 bytes/vector
- Compressed: 500M × 64 bytes = 32 GB (fits in RAM)
- Full vectors stored on SSD for re-scoring top candidates
```

**Two-stage retrieval:**
1. **Stage 1 — Approximate search:** IVF-PQ with `nprobe=128`. Returns top-1000 candidates in ~10ms. Recall@1000 ≈ 95%.
2. **Stage 2 — Re-scoring:** Load full-precision vectors for top-1000 from SSD, exact cosine similarity. Returns top-100 in ~5ms.
3. **Stage 3 (optional):** Cross-encoder reranker on top-100 → top-10. Adds 30ms.

**Serving architecture:**

```
         Query
           |
    Embedding Service (GPU, batch queries)
           |
    ┌──────┴──────┐
    │   FAISS     │ (IVF-PQ, 32 GB RAM, 4 shards × 8 GB)
    │  Cluster    │
    └──────┬──────┘
           | top-1000
    ┌──────┴──────┐
    │  Re-scorer  │ (full vectors from SSD, exact cosine)
    └──────┬──────┘
           | top-100
    ┌──────┴──────┐
    │  Reranker   │ (cross-encoder, optional)
    └──────┬──────┘
           | top-10
        Response
```

**Incremental updates:** New transactions embedded on ingestion, added to a small "hot" flat index. Periodically (nightly) merge hot index into main IVF-PQ index with re-clustering.

**Metadata filtering:** Timestamp, token type, amount range — pre-filter using Elasticsearch before vector search, or use hybrid filtering in Qdrant/Milvus.

**Key insight:** At 500M scale, you can't afford brute-force. IVF-PQ compresses 2 TB to 32 GB with ~95% recall. The two-stage architecture (approximate → exact → reranker) gives you the latency and quality you need.

---

## Part 2: Loss Functions

---

### Q10. Derive the InfoNCE loss and explain why it's a lower bound on mutual information.

**A:** Starting from mutual information estimation.

Given a batch of $N$ positive pairs $(x_i, y_i)$ drawn from the joint distribution $p(x, y)$, and $N-1$ negatives $y_j$ ($j \neq i$) drawn from the marginal $p(y)$:

The InfoNCE loss for sample $i$:

$$\mathcal{L}_i = -\log \frac{\exp(f(x_i, y_i))}{\sum_{j=1}^{N} \exp(f(x_i, y_j))}$$

where $f(x, y) = \text{sim}(g(x), h(y)) / \tau$ is a learned critic function (typically cosine similarity / temperature).

This is a categorical cross-entropy loss over $N$ classes — "which of the $N$ candidates is the true positive?"

**Mutual information bound:** Oord et al. (2018) proved:

$$I(X; Y) \geq \log(N) - \mathcal{L}_{\text{InfoNCE}}$$

**Proof sketch:** The optimal critic $f^*(x, y) = \log \frac{p(y|x)}{p(y)} + c$ achieves the bound. Substituting into the loss:

$$\mathcal{L}^* = -\mathbb{E}\left[\log \frac{e^{\log p(y_i|x_i)/p(y_i)}}{\frac{1}{N}\sum_j e^{\log p(y_j|x_i)/p(y_j)}}\right]$$

By Jensen's inequality and properties of the softmax, $\mathcal{L}^* \geq \log(N) - I(X; Y)$, hence $I(X; Y) \geq \log(N) - \mathcal{L}^*$.

**Implications:**
- The bound is tighter with more negatives ($N$ larger) — this is why CLIP uses 32K batch size
- With $N=1$, the bound is $I \geq 0$ (trivial)
- The bound can never exceed $\log(N)$ — you cannot estimate MI beyond $\log(N)$ bits with $N$ negatives

**Key insight:** InfoNCE turns mutual information maximization into a classification problem. More negatives = tighter bound = better representations.

---

### Q11. What happens to contrastive learning when temperature $\tau \to 0$ and $\tau \to \infty$?

**A:** Temperature controls the sharpness of the softmax distribution over negatives.

**When $\tau \to 0$ (very low temperature):**

$$\text{softmax}(s_j / \tau) \to \begin{cases} 1 & \text{if } j = \arg\max_k s_k \\ 0 & \text{otherwise} \end{cases}$$

The distribution becomes a one-hot over the **hardest negative** (the negative with highest similarity to the query). Gradients come only from this hardest negative — all others contribute zero.

**Problems:**
- Extremely noisy gradients — a single false negative can dominate training
- Gradient magnitude explodes: $\frac{\partial \mathcal{L}}{\partial s_j} \propto 1/\tau$
- Training becomes unstable

**When $\tau \to \infty$ (very high temperature):**

$$\text{softmax}(s_j / \tau) \to \frac{1}{N} \quad \forall j$$

The distribution becomes uniform — every negative is weighted equally regardless of how similar or dissimilar it is to the query.

**Problems:**
- Easy negatives (clearly irrelevant) dominate the gradient because there are many of them
- The model wastes capacity pushing away things that are already far
- Very slow convergence

**Optimal $\tau$:** Typically 0.05–0.1 for embedding training, 0.07 for CLIP (learned). This balances focusing on informative hard negatives while maintaining gradient signal from the full batch.

**Gradient analysis:** The gradient of InfoNCE w.r.t. the similarity of negative $j$:

$$\frac{\partial \mathcal{L}}{\partial s_j} = \frac{1}{\tau} \cdot p_j$$

where $p_j$ is the softmax probability of negative $j$. Low $\tau$ → $p_j$ concentrated on hardest negative. High $\tau$ → $p_j$ spread uniformly.

**Key insight:** Temperature controls the hardness of the contrastive task. It's one of the most important hyperparameters in embedding training.

---

### Q12. Derive the DPO loss from first principles. Start from the RLHF objective.

**A:** This is a 5-step derivation.

**Step 1 — RLHF objective:** Find policy $\pi$ that maximizes reward while staying close to reference policy $\pi_{\text{ref}}$:

$$\max_\pi \; \mathbb{E}_{x \sim \mathcal{D}, \, y \sim \pi(\cdot|x)} [r(x, y)] - \beta \, D_{KL}(\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x))$$

**Step 2 — Closed-form optimal policy:** This KL-constrained optimization has an analytical solution:

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x,y)/\beta)$ is the partition function.

**Step 3 — Express reward in terms of policies:** Rearranging the optimal policy:

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

**Step 4 — Substitute into Bradley-Terry preference model:** Human preferences follow:

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

Substituting the reward expression (the $Z(x)$ terms cancel!):

$$P(y_w \succ y_l | x) = \sigma\!\left(\beta \log \frac{\pi^*(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi^*(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

**Step 5 — DPO loss:** Replace $\pi^*$ with trainable $\pi_\theta$ and maximize log-likelihood of preferences:

$$\boxed{\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)}\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right]}$$

**Intuition:** DPO increases the log-probability ratio of the preferred response (relative to the reference) and decreases it for the rejected response. The $\beta$ parameter controls how far the policy can deviate from the reference.

**Advantages over RLHF:** No reward model, no PPO, only 2 models (policy + reference) instead of 4. Single-stage supervised training on preference pairs.

**Problems with DPO:**
- Chosen reward hacking: model may increase chosen probability by decreasing reference probability of both, rather than genuinely improving
- Sensitive to data quality — wrong preference labels cause larger damage than in RLHF
- No explicit reward signal to monitor

**Key insight:** The critical mathematical trick is that the partition function $Z(x)$ cancels in the preference probability, eliminating the need to compute an intractable normalization constant.

---

### Q13. You are training a reranker on a dataset where each query has 1 relevant document out of 1000 retrieved candidates. What loss function do you use?

**A:** This is an extreme positive:negative ratio (1:999). Several approaches:

**Option 1 — Binary cross-entropy with hard negative mining:** Don't train on all 1000. Select the top-$k$ hardest negatives (highest bi-encoder score but actually irrelevant) plus the positive. Typical $k=7$ to $15$:

$$\mathcal{L} = -\log \sigma(s^+) - \sum_{i=1}^{k} \log \sigma(-s_i^-)$$

Mining hard negatives from BM25 or bi-encoder top results gives more informative gradients than random negatives.

**Option 2 — Listwise softmax (cross-entropy over the full ranking):**

$$\mathcal{L} = -\log \frac{\exp(s^+ / \tau)}{\exp(s^+ / \tau) + \sum_{i=1}^{k} \exp(s_i^- / \tau)}$$

This is InfoNCE applied to reranking. The positive must be ranked above all negatives.

**Option 3 — Margin MSE (knowledge distillation):** If you have a teacher cross-encoder with scores $\hat{s}$:

$$\mathcal{L} = \sum_{(i,j)} ((\hat{s}_i - \hat{s}_j) - (s_i - s_j))^2$$

The student learns to reproduce the teacher's **score differences**, which is more informative than binary labels.

**Option 4 — LambdaRank:** Weight gradients by the change in NDCG if two documents were swapped. Documents near the top of the ranking get stronger gradients.

**What I would use:** Option 2 (listwise softmax) with hard negative mining (sample top-20 BM25/bi-encoder negatives per query, not all 1000). This gives clean gradients, handles the imbalance naturally (softmax normalizes), and scales well.

**Key insight:** Don't train on all 1000 negatives — most are trivially easy and provide no gradient signal. Mine hard negatives. The loss should be pairwise or listwise, not pointwise, because ranking is inherently relative.

---

### Q14. What is label smoothing? Derive its effect on the gradient.

**A:** Standard cross-entropy uses hard one-hot labels: $y = [0, 0, 1, 0, ..., 0]$. Label smoothing replaces this with:

$$y_{\text{smooth}} = (1 - \epsilon) \cdot y_{\text{hard}} + \frac{\epsilon}{K}$$

For $K$ classes and smoothing parameter $\epsilon$ (typically 0.1):
- True class gets target $(1 - \epsilon + \epsilon/K) = 1 - \epsilon(1 - 1/K)$
- Other classes get target $\epsilon/K$

**Gradient effect:** Standard CE gradient for logit $z_c$ of the correct class:

$$\frac{\partial \mathcal{L}}{\partial z_c} = p_c - 1$$

With label smoothing:

$$\frac{\partial \mathcal{L}_{\text{LS}}}{\partial z_c} = p_c - (1 - \epsilon + \epsilon/K)$$

The target for the correct class is less than 1. The gradient pushes the model toward $p_c = 1 - \epsilon + \epsilon/K \approx 0.9$ instead of $p_c = 1.0$.

**Why this helps:**
1. **Prevents overconfidence:** Without label smoothing, the model drives logits to $\pm\infty$ to achieve $p \to 1$. This wastes capacity and hurts generalization.
2. **Better calibration:** Model probabilities become more meaningful (a prediction of 0.9 is actually ~90% accurate).
3. **Implicit regularization:** Encourages the model to maintain non-zero probability mass on incorrect classes, which means the embedding space doesn't collapse.

**Key insight:** Label smoothing prevents the model from becoming infinitely confident, which improves generalization and calibration at essentially no cost.

---

### Q15. Explain knowledge distillation. Why do "soft labels" from a teacher carry more information than hard one-hot labels?

**A:** Knowledge distillation (Hinton et al., 2015) trains a small "student" to mimic a large "teacher."

**Loss:**

$$\mathcal{L} = (1-\alpha) \cdot \mathcal{L}_{CE}(y_{\text{hard}}, p_s) + \alpha \cdot T^2 \cdot D_{KL}(p_t^{(T)} \| p_s^{(T)})$$

where $p^{(T)} = \text{softmax}(z/T)$ is the temperature-scaled probability.

**Why $T^2$?** When you scale logits by $1/T$, the gradient is scaled by $1/T^2$. Multiplying by $T^2$ restores the gradient magnitude so that the distillation loss and hard label loss are on the same scale.

**Why soft labels carry more information:**

Consider a cat image. Hard label: $[0, 0, 1, 0, 0]$ (cat). Teacher soft label at $T=3$: $[0.01, 0.15, 0.60, 0.20, 0.04]$ (cat=0.60, dog=0.20, tiger=0.15, ...).

The soft label encodes:
- **Inter-class similarity:** Cat is more similar to dog (0.20) than to car (0.01). This is "dark knowledge" — information about class relationships that hard labels completely discard.
- **Uncertainty:** The teacher is only 60% confident it's a cat — the image might be ambiguous. Hard labels force 100% confidence on everything.
- **Information content:** A hard label over $K$ classes carries $\log_2(K)$ bits. A soft distribution over $K$ classes carries up to $K \log_2(K)$ bits (entropy of the distribution), vastly more.

**Application in retrieval:** Distill a cross-encoder teacher into a bi-encoder student. The cross-encoder provides soft relevance scores for (query, document) pairs. The bi-encoder learns to reproduce these soft scores, absorbing the teacher's fine-grained ranking knowledge:

$$\mathcal{L}_{\text{distill}} = \text{MSE}(s_{\text{bi-encoder}}(q, d), \; s_{\text{cross-encoder}}(q, d))$$

**Key insight:** Soft labels transfer structural knowledge about the task — how classes relate, which confusions are reasonable — not just which answer is correct. This is why DistilBERT achieves 97% of BERT's performance at 40% the size.

---

### Q16. What causes mode collapse in contrastive learning and how do you fix it?

**A:** Mode collapse in contrastive learning means all embeddings converge to a small number of points (or one point) in the embedding space. This is a degenerate solution where the model "collapses" all inputs to similar representations.

**Causes:**
1. **Learning rate too high:** Aggressive updates push all embeddings in the same direction.
2. **Temperature too low:** Only the hardest negative matters; the model "short-circuits" by collapsing everything together (loss becomes constant).
3. **No negatives or too-easy negatives:** Without hard negatives, there's no repulsive force to spread embeddings apart.
4. **Asymmetric architecture bugs:** If the two branches of a Siamese network share parameters incorrectly, both encoders can converge to a constant function.

**Fixes:**
1. **Momentum encoder (MoCo):** One encoder is updated slowly (EMA of the other), preventing both branches from collapsing simultaneously.
2. **Stop-gradient (SimSiam, BYOL):** One branch does not receive gradients from the contrastive loss. This prevents the trivial collapse solution where both branches co-adapt.
3. **Batch normalization in the projection head:** Prevents the hidden representation from collapsing by normalizing statistics across the batch.
4. **Regularization:** Add a term that maximizes the variance of embeddings within a batch (VICReg):

$$\mathcal{L}_{\text{variance}} = \frac{1}{d} \sum_{j=1}^{d} \max(0, \gamma - \text{Std}(z_j))$$

This explicitly penalizes any dimension whose variance falls below threshold $\gamma$.

5. **Monitor:** Track the average pairwise cosine similarity of embeddings in each batch. If it increases toward 1.0 over training, collapse is occurring.

**Key insight:** Contrastive learning has two forces: attraction (positives) and repulsion (negatives). Collapse happens when attraction overwhelms repulsion. The fix is ensuring both forces remain balanced.

---

## Part 3: Rerankers

---

### Q17. Why does ColBERT's MaxSim work better than single-vector similarity?

**A:** In a bi-encoder, the entire query and document are each compressed into a single vector. This is a severe information bottleneck — a 500-word document must be captured in one 768-dim vector.

ColBERT keeps **per-token embeddings**. For query tokens $\{q_1, ..., q_m\}$ and document tokens $\{d_1, ..., d_n\}$:

$$s(q, d) = \sum_{i=1}^{m} \max_{j=1}^{n} q_i^T d_j$$

Each query token finds its **best matching** document token (MaxSim), then scores are summed.

**Why this is better:**

1. **Token-level alignment:** "What is ETH staking APR?" — the token "APR" matches exactly with the document token "APR" (high dot product). A single-vector bi-encoder might lose this exact match signal in the mean/CLS pooling.

2. **Partial matching:** If the query asks about "ETH staking rewards" and the document discusses "Ethereum proof-of-stake yields," ColBERT can match:
   - "ETH" ↔ "Ethereum" (semantic match)
   - "staking" ↔ "proof-of-stake" (semantic match)
   - "rewards" ↔ "yields" (semantic match)
   Each contributes to the total score independently.

3. **No information bottleneck:** A single vector must compress all token interactions. ColBERT preserves $n$ vectors for the document, keeping full token-level information.

**Tradeoff:** ColBERT's index is $n \times d$ per document (one vector per token), vs. $1 \times d$ for a bi-encoder. For 1M documents averaging 200 tokens at 128 dims: ColBERT = 100 GB, bi-encoder = 500 MB. The PLAID algorithm compresses this via centroid-based token pruning.

**Key insight:** MaxSim captures fine-grained, token-level alignment that single-vector similarity cannot. It's the sweet spot between bi-encoder efficiency and cross-encoder quality.

---

### Q18. Derive NDCG@5 with a numerical example.

**A:** Given a query with 5 retrieved documents, each with a graded relevance score:

| Rank | Document | Relevance |
|------|---------|-----------|
| 1 | Doc A | 3 (highly relevant) |
| 2 | Doc B | 0 (irrelevant) |
| 3 | Doc C | 2 (relevant) |
| 4 | Doc D | 1 (marginally relevant) |
| 5 | Doc E | 3 (highly relevant) |

**Step 1 — DCG@5** (Discounted Cumulative Gain):

$$\text{DCG@5} = \sum_{i=1}^{5} \frac{2^{r_i} - 1}{\log_2(i + 1)}$$

$$= \frac{2^3 - 1}{\log_2 2} + \frac{2^0 - 1}{\log_2 3} + \frac{2^2 - 1}{\log_2 4} + \frac{2^1 - 1}{\log_2 5} + \frac{2^3 - 1}{\log_2 6}$$

$$= \frac{7}{1} + \frac{0}{1.585} + \frac{3}{2} + \frac{1}{2.322} + \frac{7}{2.585}$$

$$= 7 + 0 + 1.5 + 0.431 + 2.708 = 11.639$$

**Step 2 — Ideal DCG (IDCG@5):** Sort by relevance descending: [3, 3, 2, 1, 0]

$$\text{IDCG@5} = \frac{7}{1} + \frac{7}{1.585} + \frac{3}{2} + \frac{1}{2.322} + \frac{0}{2.585}$$

$$= 7 + 4.416 + 1.5 + 0.431 + 0 = 13.347$$

**Step 3 — NDCG@5:**

$$\text{NDCG@5} = \frac{\text{DCG@5}}{\text{IDCG@5}} = \frac{11.639}{13.347} = \boxed{0.872}$$

**Interpretation:** The ranking achieves 87.2% of the ideal. The main loss comes from placing the irrelevant Doc B at rank 2 (pushing the relevant Doc C and Doc E to lower positions).

**When to use NDCG:** When relevance is graded (not binary). For binary relevance, MAP or MRR is simpler and sufficient.

**Key insight:** NDCG penalizes relevant documents appearing at lower ranks via the logarithmic discount. The $2^{r_i} - 1$ term in the numerator gives exponentially more credit to higher-relevance documents.

---

### Q19. Your reranker adds 80ms latency but improves NDCG@10 from 0.65 to 0.72. How do you make the production decision?

**A:** This is a business decision, not just a technical one. Framework:

**1. Quantify the value of the NDCG improvement:**
- What does 0.65 → 0.72 mean for users? Run an online A/B test measuring:
  - Click-through rate on top results
  - Query abandonment rate (user reformulates or gives up)
  - Task completion rate (user finds what they need)
- If NDCG 0.72 increases CTR by 15% and reduces abandonment by 10%, that's significant.

**2. Analyze the latency impact:**
- 80ms additional latency. Total p50 latency goes from ~50ms to ~130ms.
- Research shows each 100ms of latency costs ~1% of revenue for e-commerce (Google/Amazon data). For search, similar principles apply.
- But: for knowledge retrieval (not shopping), users tolerate higher latency if results are better.

**3. Can you reduce the latency?**
- **Reduce top-K input to reranker:** Rerank top-50 instead of top-200. This might reduce latency from 80ms to 20ms with only marginal NDCG loss.
- **Model distillation:** Distill the cross-encoder into a faster model (MiniLM, 6 layers instead of 12). ~4× faster with ~90% of the quality gain.
- **TensorRT/ONNX optimization:** Compile the reranker. Typical 2–3× speedup.
- **Caching:** If 30% of queries are repeated, cache reranker outputs. Amortized latency drops to ~56ms.

**4. Decision matrix:**

| Scenario | Latency Impact | NDCG | Recommendation |
|----------|---------------|------|----------------|
| No reranker | 0ms | 0.65 | Baseline |
| Full reranker, top-200 | +80ms | 0.72 | Too slow for real-time |
| Distilled reranker, top-50 | +15ms | 0.70 | **Best tradeoff** |
| Full reranker + cache | +56ms avg | 0.72 | Good for non-real-time |

**My recommendation:** Distilled reranker on top-50, optimized with TensorRT. 15ms latency for 0.70 NDCG. Deploy full cross-encoder for offline batch scoring.

**Key insight:** Never present this as "do we add the reranker yes/no." Present it as "here are 4 configurations with different latency/quality tradeoffs, and here's which I'd pick for our use case."

---

### Q20. How would you train a cross-encoder reranker without labeled data?

**A:** Several approaches:

**1. Distillation from an LLM (zero-shot):** Use GPT-4 or a large instruction-tuned LLM to generate relevance judgments:

```
Prompt: "Rate the relevance of this document to the query on a scale of 0-3.
Query: {query}
Document: {document}
Relevance (0=irrelevant, 3=highly relevant):"
```

Generate labels for 100K (query, document) pairs sampled from your production query logs + BM25 top-50 results. Train the cross-encoder on these soft labels.

**2. Self-training / bootstrapping:**
- Start with BM25 as the initial retriever
- Use BM25 top-1 as "positive" and BM25 rank 50–100 as "negative" (assumption: rank 1 is more relevant than rank 50+)
- Train cross-encoder v0 on this noisy data
- Use cross-encoder v0 to re-score and re-label → create cleaner (query, doc, label) triples
- Train cross-encoder v1 on the cleaned labels
- Iterate 2–3 times

**3. Cross-lingual / cross-domain transfer:**
- Start from a cross-encoder trained on MS MARCO (publicly available, e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- The model already knows what relevance means — it just needs domain adaptation
- Fine-tune on domain-specific data using either method 1 or 2

**4. Contrastive from click-through data:**
- If you have search logs with clicks: (query, clicked_doc) = positive, (query, shown_but_not_clicked) = negative
- Noisy but effective at scale. De-bias for position bias (users click higher-ranked results more).

**My approach for Binance:** Start with MS MARCO cross-encoder (free, strong baseline). Generate 50K labels using an LLM on Binance-specific queries. Fine-tune. Iterate with self-training using production click data.

**Key insight:** You almost never have perfect labeled data. The art is bootstrapping from weak signals (BM25 rankings, clicks, LLM judgments) and iteratively improving.

---

### Q21. Explain Reciprocal Rank Fusion. What happens at the extreme values of k?

**A:** RRF combines results from multiple ranking systems using rank position:

$$\text{RRF}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + \text{rank}_r(d)}$$

where $k = 60$ by convention (Cormack et al., 2009).

**Why it works:** RRF is rank-based, not score-based. This means:
- No score normalization needed (BM25 scores and cosine similarities are on completely different scales)
- Robust to outlier scores
- Documents ranked highly by multiple systems get a strong combined score

**Extreme values of $k$:**

**$k \to 0$:** $\text{RRF}(d) \approx \sum_r \frac{1}{\text{rank}_r(d)}$. This is MRR-like — the rank-1 result from each ranker dominates completely (score = 1), while rank-2 is 0.5, rank-10 is 0.1. Very "winner-take-all" — strongly favors any system's top-1 result, ignoring agreement at lower ranks.

**$k \to \infty$:** $\text{RRF}(d) \approx \sum_r \frac{1}{k}$ (constant, independent of rank). All positions are treated nearly equally — rank 1 and rank 100 have almost the same score. RRF degenerates into a count of "how many systems returned this document" — pure recall fusion, ignoring ranking quality.

**$k = 60$ (default):** Balances between these extremes. Rank 1 scores $1/61 \approx 0.0164$, rank 60 scores $1/120 = 0.0083$ (half). So the top-60 results contribute meaningfully; beyond 60, the contribution diminishes. This is a reasonable "effective depth" for most retrieval systems.

**Worked example:**

| Document | BM25 Rank | Dense Rank | RRF Score |
|----------|----------|------------|-----------|
| Doc X | 1 | 5 | $\frac{1}{61} + \frac{1}{65} = 0.0318$ |
| Doc Y | 3 | 1 | $\frac{1}{63} + \frac{1}{61} = 0.0323$ |
| Doc Z | 2 | 8 | $\frac{1}{62} + \frac{1}{68} = 0.0308$ |

RRF ranking: **Y > X > Z**. Doc Y wins because rank 1 in dense retrieval outweighs its worse BM25 position. Agreement matters more than dominance in one system.

**Key insight:** RRF is simple, requires no tuning, handles heterogeneous score scales, and empirically outperforms most score-based fusion methods. The $k=60$ default rarely needs changing.

---

## Part 4: RAG & Retrieval

---

### Q22. What is the "lost in the middle" problem and how do you mitigate it in production?

**A:** LLMs perform significantly worse at utilizing information placed in the **middle** of long contexts compared to the beginning or end (Liu et al., 2023). Given a prompt with 20 retrieved documents, the model reliably uses documents at positions 1–3 and 18–20, but often ignores documents at positions 8–12.

**Causes:**
1. **Positional bias from training data:** Web text tends to have the most important information at the beginning (inverted pyramid structure in journalism, abstracts in papers).
2. **Attention pattern decay:** In causal attention, recent tokens have more direct influence paths than middle tokens.
3. **U-shaped attention distribution:** Models develop a primacy (beginning) and recency (end) bias.

**Mitigations:**

1. **Strategic document placement:** Put the most relevant documents at the beginning and end of the context. Place less relevant ones in the middle.

2. **Reduce retrieval count:** Instead of stuffing 20 documents into context, use a strong reranker and only include the top 3–5. Less context = less "middle" to get lost in.

3. **Map-reduce approach:** Process each document separately with the LLM, extract key points, then synthesize:
   - Step 1: For each doc, ask "What does this say about {query}?" → short answer
   - Step 2: Combine all short answers → final answer
   - No middle problem because each doc gets full attention.

4. **Recursive summarization:** Summarize chunks of 3–4 documents into a single paragraph each, then feed summaries into the final prompt.

5. **Citation enforcement:** Ask the model to cite `[Doc N]` for every claim. This forces it to attend to specific documents rather than generating from parametric memory.

6. **Re-ordering by relevance with interleaving:** Place documents in relevance order: 1st most relevant at position 1, 2nd at the end, 3rd at position 2, 4th at second-to-last, etc. This distributes important content across the U-shaped attention curve.

**Key insight:** The practical fix is simple: retrieve fewer, higher-quality documents. If you must use many documents, use map-reduce. Don't rely on the LLM to attend equally to a 20-document context.

---

### Q23. How do you evaluate a RAG system end-to-end? Explain the key metrics.

**A:** RAG evaluation requires measuring both retrieval quality and generation quality. The RAGAS framework (Es et al., 2023) defines four core metrics:

**1. Context Precision** (Is retrieved context relevant?):

$$\text{Context Precision@K} = \frac{1}{K} \sum_{k=1}^{K} \frac{\text{number of relevant chunks in top-}k}{k} \times v_k$$

where $v_k = 1$ if chunk at rank $k$ is relevant, else 0. Higher-ranked relevant chunks score more. Requires relevance labels (or LLM-as-judge).

**2. Context Recall** (Did retrieval find all needed information?):

$$\text{Context Recall} = \frac{|\text{ground truth facts covered by retrieved context}|}{|\text{total ground truth facts}|}$$

If the ground truth answer has 5 key facts and your retrieved context supports 4 of them, recall = 0.8.

**3. Faithfulness** (Does the answer only use retrieved context?):

$$\text{Faithfulness} = \frac{|\text{claims in answer supported by context}|}{|\text{total claims in answer}|}$$

Decompose the answer into individual claims. For each claim, check if it can be inferred from the retrieved context. Faithfulness = 1.0 means no hallucination.

**4. Answer Relevance** (Does the answer address the question?):

Generate $n$ synthetic questions from the answer. Compute average cosine similarity between these synthetic questions and the original query. If the answer is on-topic, the synthetic questions will resemble the original query.

**Practical evaluation pipeline:**

```
For each (query, ground_truth_answer) in test set:
    retrieved = retriever(query, top_k=10)
    context = format_context(retrieved)
    answer = llm(query, context)

    # Component metrics
    retrieval_recall_at_10 = recall(retrieved, relevant_docs)
    retrieval_mrr = mrr(retrieved, relevant_docs)

    # End-to-end metrics (use LLM-as-judge)
    faithfulness = judge_faithfulness(answer, context)
    correctness = judge_correctness(answer, ground_truth)
    relevance = judge_relevance(answer, query)
```

**Key insight:** Evaluate component-level (retrieval) and end-to-end (generation) separately. A system can have perfect retrieval but poor generation (LLM ignores context), or vice versa (LLM hallucinates despite missing context). You need both levels to diagnose issues.

---

### Q24. A RAG system faithfully cites its retrieved sources but the answer is still wrong. What went wrong?

**A:** This is a **retrieval failure** manifesting as an **accurate-but-wrong** answer. The system is doing its job correctly — being faithful to context — but the context itself is wrong or insufficient. Causes:

1. **Retrieved documents are outdated:** The knowledge base contains old information. "ETH staking APR is 5.2%" was correct 6 months ago but it's now 3.8%. The model faithfully cites the old document.

2. **Retrieved documents are topically relevant but don't answer the question:** Query: "What is the minimum BTC withdrawal?" Retrieved: "BTC transactions require network confirmations." Relevant topic (BTC), wrong information need.

3. **Conflicting information in retrieved documents:** Doc 1 says "minimum withdrawal is 0.001 BTC," Doc 2 says "minimum withdrawal is 0.0005 BTC." The model picks one — faithfully citing it — but picks the wrong one. (Perhaps Doc 1 is the current policy and Doc 2 is archived.)

4. **Chunk boundary issue:** The relevant information spans two chunks. Chunk A: "The minimum withdrawal amount is" (end of chunk). Chunk B: "0.001 BTC for regular accounts and 0.0001 for VIP." Only chunk A was retrieved, so the model hallucinated the actual number while faithfully citing chunk A.

5. **Ambiguous query → wrong retrieval:** "How much does it cost?" — cost of what? Trading fee? Withdrawal fee? Gas fee? The retriever guesses wrong, retrieves documents about trading fees, and the model faithfully answers about trading fees.

**Fixes:**
- Add timestamps to documents and filter by recency for factual queries
- Use metadata (document type, last_updated) to prefer authoritative/current sources
- Overlap chunks to prevent boundary splits
- Query decomposition to disambiguate vague queries before retrieval
- Conflict resolution: when multiple documents disagree, flag uncertainty or prefer the most recent / most authoritative source

**Key insight:** Faithfulness ≠ correctness. A faithful model that retrieves bad documents will confidently give wrong answers with citations. Always evaluate retrieval quality independently from generation quality.

---

## Part 5: LLM Architecture & Fine-tuning

---

### Q25. Why does LoRA work? What is the "low intrinsic rank" hypothesis?

**A:** LoRA (Hu et al., 2021) freezes the pretrained weight matrix $W_0$ and adds a low-rank update:

$$W = W_0 + \Delta W = W_0 + BA$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, $r \ll \min(d, k)$.

**Low intrinsic rank hypothesis:** The weight changes during fine-tuning, $\Delta W = W_{\text{fine-tuned}} - W_{\text{pretrained}}$, live on a **low-dimensional subspace** of the full weight space.

**Evidence:**
1. Aghajanyan et al. (2021) showed that pretrained models have a low "intrinsic dimensionality" — you can achieve 90% of full fine-tuning performance by optimizing in a random subspace of dimension $d' \ll D$ (total parameters). For RoBERTa on MRPC, $d' \approx 200$ out of 125M parameters.

2. Hu et al. measured the rank of $\Delta W$ empirically: the singular values of $W_{\text{fine-tuned}} - W_{\text{pretrained}}$ decay rapidly. Most of the change is captured by the top 1–8 singular vectors.

3. **Theoretical intuition:** Pretraining already learns a good representation space. Fine-tuning doesn't need to restructure the entire weight matrix — it only needs to make a small, structured adjustment. If the original weight captures general language knowledge in an $d$-dimensional space, the task-specific adjustment only needs to redirect attention in a few key dimensions.

**Why rank $r = 4$ to 64 is enough:**
- Classification: the model only needs to learn to map from existing representations to a few output classes — rank 1–4 often suffices.
- Instruction following: slightly higher rank (8–16) to redirect generation behavior.
- Domain adaptation (e.g., crypto-specific language): rank 32–64 for substantial vocabulary/knowledge updates.

**Key insight:** The pretrained model is already "almost right" for most tasks. Fine-tuning is a low-rank perturbation because only a few directions in weight space need to change.

---

### Q26. Explain QLoRA. What is NF4 and why is it better than standard INT4?

**A:** QLoRA (Dettmers et al., 2023) enables fine-tuning of 65B+ models on a single 48 GB GPU. Three innovations:

**1. NormalFloat 4-bit (NF4):** Observation: pretrained weights are approximately normally distributed. Standard INT4 quantization uses uniformly spaced quantization levels, which wastes levels on the tails and under-represents the dense center.

NF4 places quantization levels at the **quantiles of the normal distribution**:

$$q_i = \Phi^{-1}\!\left(\frac{i + 0.5}{2^4}\right) \quad \text{for } i = 0, 1, ..., 15$$

where $\Phi^{-1}$ is the inverse CDF of $\mathcal{N}(0,1)$. This is **information-theoretically optimal** for normally distributed data — each quantization bin contains equal probability mass, minimizing expected quantization error.

Compared to uniform INT4, NF4 has more levels near 0 (where most weights are) and fewer levels in the tails (where few weights are).

**2. Double quantization:** The quantization constants (scale factors) themselves consume memory — one FP32 scale per block of 64 weights = 0.5 bits/param overhead. QLoRA quantizes these constants too (FP32 → FP8), reducing overhead to 0.127 bits/param. Saves ~3 GB for a 65B model.

**3. Paged optimizers:** Optimizer states (Adam has 2 states per parameter) are moved to CPU RAM when GPU memory is insufficient, then paged back in on demand. Prevents OOM during gradient accumulation.

**Training flow:**
```
Forward:  NF4 weights → dequantize to BF16 → compute → BF16 LoRA adapters
Backward: Gradients in BF16 → update only LoRA adapters (BF16)
          Base weights stay frozen in NF4
```

**Memory:** 65B model in NF4 ≈ 32.5 GB. With LoRA adapters + optimizer states (BF16): ~40 GB total. Fits on one A6000 (48 GB).

**Key insight:** NF4 is not a generic quantization format — it's specifically optimized for the weight distribution of pretrained neural networks. This domain-specific design is what makes it work so well.

---

### Q27. Chinchilla says 20 tokens/parameter is optimal. Llama 3 8B trained on 15T tokens (1875 tokens/param). Isn't that wasteful?

**A:** No. This is one of the most important strategic insights in modern LLM training.

**Chinchilla is compute-optimal, not deployment-optimal.**

Chinchilla answers: "Given a fixed training compute budget $C$, what $N$ (parameters) and $D$ (tokens) minimize loss?" Answer: $N \propto C^{0.5}$, $D \propto C^{0.5}$, giving ~20 tokens/param.

But Chinchilla ignores **inference cost**. In production:

- Training cost is a one-time expenditure
- Inference cost is paid **per query, forever**
- Inference cost $\propto N$ (model size) per token generated

So the real optimization is:

$$\text{Total cost} = C_{\text{train}} + C_{\text{inference/query}} \times \text{expected\_queries}$$

For high-volume services (billions of queries), the inference term dominates. A smaller model that's "over-trained" on more data is cheaper to serve at the same quality level.

**Concrete example:**
- Chinchilla-optimal 40B model on 800B tokens: trains in $X$ GPU-hours. Inference: $Y$ ms/token.
- Over-trained 8B model on 15T tokens: trains in $2X$ GPU-hours (more data). Inference: $Y/5$ ms/token (5× smaller).

If you serve 1B queries/month, the 8B model pays off its extra training cost in days.

**Llama 3 philosophy:** Spend 5× more on training compute to get a 5× smaller model with equivalent quality. The extra training cost is amortized over billions of inference tokens.

**Key insight:** Chinchilla is about training efficiency. Over-training is about deployment efficiency. Meta chose deployment efficiency because training is a one-time cost but inference scales with usage.

---

### Q28. What is the attention sink phenomenon?

**A:** Attention sink (Xiao et al., 2023) refers to the observation that in autoregressive LLMs, the **first token** (BOS/system token) accumulates disproportionately high attention weights across all layers and heads, even though it carries no meaningful semantic content.

**Why it happens:** Softmax forces attention weights to sum to 1. When no particular token is relevant (e.g., for general context aggregation), the model needs to "dump" attention mass somewhere. The first token becomes the default dump location because:
1. It's always present in every sequence (consistent during training)
2. Its value vector is learned to be a good "no-op" — it returns a residual that doesn't interfere with computation
3. Positional encoding gives it a unique, consistent fingerprint

**Problem for streaming/long context:** If you use a sliding window attention (e.g., only attend to last 4096 tokens), the initial BOS token eventually falls out of the window. When this happens, the model loses its attention sink and performance degrades catastrophically — even though the BOS token carried no semantic information.

**StreamingLLM fix:** Always keep the first $k$ tokens (typically $k=4$, the "sink tokens") in the attention window, regardless of how far the window has moved:

$$\text{Attention window} = \{\text{tokens}[0:k]\} \cup \{\text{tokens}[t-W:t]\}$$

where $W$ is the sliding window size and $t$ is the current position. This maintains the sink and enables theoretically infinite context without quality degradation (for tasks that don't require long-range retrieval from the middle).

**Key insight:** The first token acts as an attention "garbage collector." It's a learned artifact of softmax normalization. Removing it breaks the model; preserving it enables efficient streaming inference.

---

## Part 6: System Design

---

### Q29. Design a semantic search system for Binance's customer support with 50 languages, < 100ms latency, and zero tolerance for hallucination on account-specific queries.

**A:**

**Architecture overview:**

```
User query (any of 50 languages)
        |
  Language detection (fasttext lid, <1ms)
        |
  Query classifier: account-specific vs. general knowledge
        |
  ┌─────┴─────┐
  |           |
Account     General
queries     queries
  |           |
  v           v
Direct DB    RAG pipeline
lookup       |
(no LLM)     ├─ BM25 (multilingual Elasticsearch)
  |          ├─ Dense retrieval (multilingual-e5-large, FAISS)
  |          ├─ RRF fusion
  |          ├─ Cross-encoder reranker (top-50 → top-5)
  |          └─ LLM generation (Llama 3 8B, instruction-tuned)
  |                |
  v                v
Template         Generated answer + citations
response         + faithfulness check
  |                |
  └────────┬───────┘
           |
    Safety filter (regex + classifier)
           |
    Response to user
```

**Key design decisions:**

**1. Account queries → NO LLM.** "What is my balance?" "Where is my withdrawal?" These must return exact data from the database. An LLM would hallucinate. Use slot-filling NER to extract entities (account ID, asset, transaction type) and template-based responses with exact database values. Zero hallucination by construction.

**2. Multilingual retrieval:** Use `multilingual-e5-large` which maps all 50 languages to a shared embedding space. A Korean query will retrieve the relevant English FAQ article and vice versa. BM25 runs per-language (Elasticsearch with language-specific analyzers). RRF combines both.

**3. Latency budget:**
- Language detection: 1ms
- Query classification: 3ms
- BM25 retrieval: 10ms
- Dense retrieval: 10ms (HNSW, pre-computed)
- RRF fusion: 1ms
- Reranker (top-50): 40ms (distilled MiniLM, TensorRT)
- LLM generation: 30ms (first token) — stream remaining tokens
- **Total: ~95ms to first token**

**4. Faithfulness enforcement:**
- System prompt: "Answer ONLY using the provided documents. If the information is not in the documents, say 'I don't have information about that.'"
- Post-generation check: lightweight NLI model classifies if answer is supported by context. If confidence < 0.8, fall back to "Let me connect you with a human agent."
- All LLM outputs logged for offline quality audit.

**5. Safety filter:** Block any output containing specific investment advice, price predictions, or references to competitor exchanges.

**Key insight:** The most important design decision is separating account-specific queries (no LLM, template only) from general knowledge queries (RAG). This eliminates the highest-risk hallucination category entirely.

---

### Q30. Design a recommendation system for crypto assets using embeddings. How do you handle the cold start problem?

**A:**

**Core approach:** Represent both users and crypto assets as embeddings in the same space. Recommendation = nearest neighbor search.

**User embedding:** Aggregate from interaction history:
$$\mathbf{u} = \alpha \cdot \text{mean}(\text{asset\_embeds\_traded}) + (1-\alpha) \cdot \text{attention\_pool}(\text{article\_embeds\_read})$$

The first term captures trading behavior; the second captures information-seeking behavior.

**Asset embedding:** Multi-modal:
- **Text:** Embed whitepaper abstract + recent news using fine-tuned E5
- **On-chain features:** Market cap, volume, holder distribution → MLP to embedding
- **Social graph:** GNN embedding from who-holds-what co-occurrence
- Concatenate and project to shared $d$-dimensional space

**Training:** Two-tower model with MNRL loss:
$$\mathcal{L} = -\log \frac{\exp(\mathbf{u}_i \cdot \mathbf{a}_i^+ / \tau)}{\sum_j \exp(\mathbf{u}_i \cdot \mathbf{a}_j / \tau)}$$

Positive pairs: (user, asset the user traded/researched). Negatives: in-batch.

**Cold start — new users:**
1. **Content-based fallback:** Embed the user's first few interactions immediately. Even 1–2 data points give a rough position in embedding space.
2. **Onboarding quiz:** "Are you interested in DeFi, NFTs, or L1s?" Map answers to cluster centers in the asset embedding space.
3. **Popularity-weighted exploration:** For zero-data users, recommend popular assets weighted by diversity (one from each category).

**Cold start — new assets:**
1. **Text embedding is available immediately:** Embed the whitepaper and launch announcement. New asset is positioned in the space before anyone trades it.
2. **Transfer from similar assets:** Find the nearest existing assets by text similarity. Initialize the new asset's behavioral embedding as the average of its text-neighbors.
3. **Exploration bonus:** Add a novelty score that decays over time, ensuring new assets get some exposure regardless of embedding position.

**Key insight:** The text modality solves cold start for assets — a whitepaper exists before any trading data. For users, onboarding signals + fast adaptation from sparse interactions + popularity fallback cover the gap.

---

## Part 7: Coding Questions

---

### Q31. Implement InfoNCE loss with temperature and in-batch negatives in PyTorch.

```python
import torch
import torch.nn.functional as F

def info_nce_loss(query_embeds: torch.Tensor,
                  doc_embeds: torch.Tensor,
                  temperature: float = 0.05) -> torch.Tensor:
    """
    InfoNCE loss with in-batch negatives.

    Args:
        query_embeds: (B, d) - normalized query embeddings
        doc_embeds:   (B, d) - normalized document embeddings
                      query_embeds[i] and doc_embeds[i] are a positive pair
        temperature:  scaling factor

    Returns:
        scalar loss
    """
    # Similarity matrix: (B, B)
    # sim[i][j] = cosine(query_i, doc_j) / temperature
    similarity = query_embeds @ doc_embeds.T / temperature

    # Labels: diagonal entries are positive pairs
    labels = torch.arange(similarity.size(0), device=similarity.device)

    # Symmetric loss: query→doc and doc→query
    loss_q2d = F.cross_entropy(similarity, labels)
    loss_d2q = F.cross_entropy(similarity.T, labels)

    return (loss_q2d + loss_d2q) / 2
```

**Key points:**
- Embeddings must be L2-normalized before calling this function
- Labels are the identity permutation: `[0, 1, 2, ..., B-1]`
- Cross-entropy over the similarity matrix is equivalent to the InfoNCE formula
- Symmetric loss means query→doc and doc→query both contribute

---

### Q32. Implement NDCG@K, MRR, and MAP from scratch.

```python
import numpy as np

def dcg_at_k(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain at K."""
    relevances = np.array(relevances[:k])
    positions = np.arange(1, len(relevances) + 1)
    discounts = np.log2(positions + 1)
    gains = 2**relevances - 1
    return float(np.sum(gains / discounts))


def ndcg_at_k(relevances: list[float], k: int) -> float:
    """Normalized DCG at K."""
    dcg = dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(ranked_relevances: list[list[int]]) -> float:
    """
    Mean Reciprocal Rank.
    ranked_relevances: list of queries, each a list of binary relevances at each rank.
    """
    reciprocal_ranks = []
    for rels in ranked_relevances:
        for rank, rel in enumerate(rels, 1):
            if rel == 1:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return float(np.mean(reciprocal_ranks))


def average_precision(relevances: list[int]) -> float:
    """Average Precision for a single query."""
    relevances = np.array(relevances)
    n_relevant = relevances.sum()
    if n_relevant == 0:
        return 0.0
    cumsum = np.cumsum(relevances)
    precisions = cumsum / np.arange(1, len(relevances) + 1)
    return float(np.sum(precisions * relevances) / n_relevant)


def mean_average_precision(ranked_relevances: list[list[int]]) -> float:
    """MAP over multiple queries."""
    return float(np.mean([average_precision(r) for r in ranked_relevances]))


# --- Example ---
if __name__ == "__main__":
    # Graded relevance: 3 docs with relevance [3, 0, 2, 1, 3]
    rels = [3, 0, 2, 1, 3]
    print(f"NDCG@5: {ndcg_at_k(rels, 5):.4f}")  # 0.8722

    # Binary relevance for MRR and MAP
    binary = [[0, 1, 0, 1],   # first relevant at rank 2 → RR = 0.5
              [1, 0, 0, 0],   # first relevant at rank 1 → RR = 1.0
              [0, 0, 1, 0]]   # first relevant at rank 3 → RR = 0.333
    print(f"MRR:    {mrr(binary):.4f}")     # 0.6111
    print(f"MAP:    {mean_average_precision(binary):.4f}")
```

---

### Q33. Implement a LoRA layer from scratch in PyTorch.

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Linear layer with Low-Rank Adaptation.
    Wraps a frozen pretrained linear layer with trainable low-rank matrices.
    """
    def __init__(self, pretrained_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.in_features = pretrained_linear.in_features
        self.out_features = pretrained_linear.out_features
        self.rank = rank
        self.scaling = alpha / rank  # LoRA scaling factor

        # Freeze pretrained weights
        self.pretrained = pretrained_linear
        self.pretrained.weight.requires_grad_(False)
        if self.pretrained.bias is not None:
            self.pretrained.bias.requires_grad_(False)

        # LoRA matrices
        # A: (rank, in_features) — initialized with Kaiming uniform
        # B: (out_features, rank) — initialized to zero
        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        # Initialize A with Kaiming uniform (same as nn.Linear default)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen output
        base_output = self.pretrained(x)

        # LoRA output: x @ A^T @ B^T * scaling
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

        return base_output + lora_output

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA into the base weights for zero-overhead inference."""
        merged = nn.Linear(self.in_features, self.out_features,
                          bias=self.pretrained.bias is not None)
        merged.weight.data = (
            self.pretrained.weight.data + self.scaling * (self.lora_B @ self.lora_A)
        )
        if self.pretrained.bias is not None:
            merged.bias.data = self.pretrained.bias.data
        return merged


# --- Example: Apply LoRA to a pretrained layer ---
if __name__ == "__main__":
    # Simulate a pretrained linear layer (e.g., from BERT's attention)
    pretrained = nn.Linear(768, 768)

    # Wrap with LoRA (rank=8)
    lora_layer = LoRALinear(pretrained, rank=8, alpha=16.0)

    # Count parameters
    total = sum(p.numel() for p in lora_layer.parameters())
    trainable = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    print(f"Total params:     {total:,}")       # 590,592 + 12,288 = 602,880
    print(f"Trainable params: {trainable:,}")    # 12,288 (rank*in + out*rank = 8*768 + 768*8)
    print(f"Reduction:        {total/trainable:.1f}x")  # ~49x

    # Test forward pass
    x = torch.randn(4, 10, 768)  # batch=4, seq_len=10, d=768
    output = lora_layer(x)
    print(f"Output shape:     {output.shape}")  # (4, 10, 768)

    # Merge for inference (no extra latency)
    merged = lora_layer.merge_weights()
    output_merged = merged(x)
    print(f"Max merge error:  {(output - output_merged).abs().max():.2e}")  # ~1e-7
```

---

### Q34. Implement BM25 from scratch.

```python
import math
import numpy as np
from collections import Counter

class BM25:
    """BM25 ranking function from scratch."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1  # term frequency saturation
        self.b = b     # document length normalization

    def fit(self, documents: list[list[str]]):
        """
        Index a corpus.
        documents: list of tokenized documents (each is a list of strings)
        """
        self.corpus_size = len(documents)
        self.doc_lengths = [len(doc) for doc in documents]
        self.avgdl = sum(self.doc_lengths) / self.corpus_size

        # Document frequency: how many documents contain each term
        self.df = Counter()
        # Term frequency per document
        self.tf = []

        for doc in documents:
            tf = Counter(doc)
            self.tf.append(tf)
            for term in set(doc):  # count each term once per doc
                self.df[term] += 1

    def _idf(self, term: str) -> float:
        """Robertson-Sparck Jones IDF."""
        n = self.df.get(term, 0)
        return math.log((self.corpus_size - n + 0.5) / (n + 0.5) + 1)

    def score(self, query: list[str], doc_index: int) -> float:
        """Score a single document against a query."""
        doc_tf = self.tf[doc_index]
        doc_len = self.doc_lengths[doc_index]

        score = 0.0
        for term in query:
            if term not in doc_tf:
                continue
            tf = doc_tf[term]
            idf = self._idf(term)
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
            score += idf * numerator / denominator
        return score

    def search(self, query: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        """Return top-k (doc_index, score) pairs, sorted by score descending."""
        scores = [(i, self.score(query, i)) for i in range(self.corpus_size)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# --- Example ---
if __name__ == "__main__":
    corpus = [
        "bitcoin price prediction for next month".split(),
        "ethereum staking rewards and APR guide".split(),
        "how to buy bitcoin on binance exchange".split(),
        "defi yield farming strategies for beginners".split(),
        "bitcoin and ethereum price comparison analysis".split(),
    ]

    bm25 = BM25()
    bm25.fit(corpus)

    query = "bitcoin price".split()
    results = bm25.search(query, top_k=3)
    for idx, score in results:
        print(f"  Score {score:.3f}: {' '.join(corpus[idx])}")
```

---

### Q35. Implement Reciprocal Rank Fusion.

```python
def reciprocal_rank_fusion(
    rankings: list[list[int]],
    k: int = 60
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion.

    Args:
        rankings: list of ranked document ID lists.
                  rankings[i] = [doc_id_rank1, doc_id_rank2, ...]
                  from ranker i.
        k: RRF constant (default 60).

    Returns:
        List of (doc_id, rrf_score) sorted by score descending.
    """
    rrf_scores: dict[int, float] = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)

    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


# --- Example ---
if __name__ == "__main__":
    bm25_ranking = [101, 204, 305, 102, 200]
    dense_ranking = [305, 101, 102, 204, 400]
    colbert_ranking = [101, 305, 200, 102, 204]

    fused = reciprocal_rank_fusion([bm25_ranking, dense_ranking, colbert_ranking])

    print("RRF Results:")
    for doc_id, score in fused[:5]:
        print(f"  Doc {doc_id}: {score:.5f}")
    # Doc 101 should be top (ranked 1st by BM25 and ColBERT, 2nd by dense)
```

---

### Q36. Implement a cross-encoder reranker using HuggingFace.

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score_pairs(self, query: str, documents: list[str],
                    batch_size: int = 32) -> list[float]:
        """Score (query, doc) pairs. Returns relevance scores."""
        all_scores = []

        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            # Cross-encoder input: [CLS] query [SEP] document [SEP]
            pairs = [[query, doc] for doc in batch_docs]

            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model(**inputs)
            scores = outputs.logits.squeeze(-1).cpu().tolist()

            if isinstance(scores, float):
                scores = [scores]
            all_scores.extend(scores)

        return all_scores

    def rerank(self, query: str, documents: list[str],
               top_k: int = 10) -> list[tuple[int, str, float]]:
        """Rerank documents by relevance to query."""
        scores = self.score_pairs(query, documents)

        # Sort by score descending
        ranked = sorted(enumerate(zip(documents, scores)),
                       key=lambda x: x[1][1], reverse=True)

        return [(idx, doc, score) for idx, (doc, score) in ranked[:top_k]]


# --- Example ---
if __name__ == "__main__":
    reranker = CrossEncoderReranker()

    query = "What is ETH staking APR?"
    docs = [
        "Ethereum staking currently offers approximately 3.8% annual percentage rate.",
        "Bitcoin mining difficulty has increased 15% this quarter.",
        "The Ethereum network uses proof-of-stake consensus mechanism.",
        "Staking rewards vary by validator and network conditions.",
        "BNB Chain offers staking with variable APR depending on the amount staked.",
    ]

    results = reranker.rerank(query, docs, top_k=3)
    for idx, doc, score in results:
        print(f"  [{score:.3f}] {doc}")
```

---

## Quick Reference: What Binance Interviewers Are Really Testing

| Question Type | What They Want to See |
|---|---|
| **"Explain X"** | Can you teach it clearly? Do you understand the *why*, not just the *what*? |
| **"Derive the loss"** | Do you understand the math, or just memorize formulas? |
| **"Design a system"** | Can you make tradeoffs? Do you think about latency, cost, scale? |
| **"What would you do if..."** | Can you debug and troubleshoot? Do you have practical experience? |
| **"Code this from scratch"** | Can you implement, not just use libraries? |
| **"Compare A vs B"** | Do you understand tradeoffs, or do you have a one-size-fits-all answer? |

**Binance-specific priorities:**
- **Multilingual** (global exchange, 50+ languages)
- **Low latency** (trading context — milliseconds matter)
- **Safety** (financial advice hallucination is a regulatory risk)
- **Scale** (hundreds of millions of users)
- **Crypto domain knowledge** (blockchain, DeFi, tokenomics vocabulary)
