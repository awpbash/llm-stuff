# 09 — Modern Techniques: MoE, CPT, GRPO, Alignment Variants, SPLADE, Graph RAG, Self-RAG/CRAG, and Model Merging

> Lecture-note style technical guide covering frontier methods that appear increasingly in industry interviews and production systems (2024-2026).

---

## Table of Contents

1. [Mixture of Experts (MoE)](#1-mixture-of-experts-moe)
2. [Continued Pretraining (CPT) / Domain-Adaptive Pretraining](#2-continued-pretraining-cpt--domain-adaptive-pretraining)
3. [GRPO (Group Relative Policy Optimization)](#3-grpo-group-relative-policy-optimization)
4. [SimPO, ORPO, KTO](#4-simpo-orpo-kto)
5. [SPLADE (Sparse Lexical and Expansion Model)](#5-splade-sparse-lexical-and-expansion-model)
6. [Graph RAG](#6-graph-rag)
7. [Self-RAG and Corrective RAG (CRAG)](#7-self-rag-and-corrective-rag-crag)
8. [Model Merging Techniques](#8-model-merging-techniques)
9. [Problems & Mitigations](#9-problems--mitigations)
10. [Interview Q&A](#10-interview-qa)

---

## 1. Mixture of Experts (MoE)

### 1.1 Core Idea

In a standard Transformer, every parameter is activated for every input token. This means that scaling the model (adding more parameters) scales compute linearly. **Mixture of Experts (MoE)** breaks this coupling: the model has many parameters, but only a *sparse subset* is activated for any given token.

The key insight is: **different tokens need different knowledge**. A token about "liquidation" in a DeFi context activates different experts than a token about "weather." This conditional computation allows MoE models to have far more total parameters (capacity) while keeping per-token FLOPs roughly constant.

### 1.2 Architecture

An MoE layer **replaces the Feed-Forward Network (FFN)** in a standard Transformer block. Instead of one FFN, there are $N$ expert FFNs and a small **router network** (also called a gating network) that decides which experts process each token.

```
                          Standard Transformer Block
                          ┌─────────────────────────┐
                          │    Multi-Head Attention  │
                          │    + Add & LayerNorm     │
                          ├─────────────────────────┤
                          │    FFN (all params used) │
                          │    + Add & LayerNorm     │
                          └─────────────────────────┘

                           MoE Transformer Block
                          ┌─────────────────────────┐
                          │    Multi-Head Attention  │
                          │    + Add & LayerNorm     │
                          ├─────────────────────────┤
                          │     ┌─── Router ───┐    │
                          │     │  g(x) = TopK  │    │
                          │     │  (softmax(Wx))│    │
                          │     └───┬───┬───┬───┘    │
                          │         │   │   │        │
                          │   ┌─────┴┐ ┌┴──┐┌┴─────┐ │
                          │   │Exp 1 │ │...││Exp N │ │
                          │   │ FFN  │ │   ││ FFN  │ │
                          │   └──┬───┘ └─┬─┘└──┬───┘ │
                          │      └───┬───┘     │     │
                          │     Weighted Sum ◄─┘     │
                          │    + Add & LayerNorm     │
                          └─────────────────────────┘
```

### 1.3 Router Network

The router is a small learned linear layer that maps each token representation $x \in \mathbb{R}^d$ to a probability distribution over $N$ experts:

$$g(x) = \text{TopK}(\text{softmax}(W_r x + b_r))$$

where $W_r \in \mathbb{R}^{N \times d}$. After softmax, only the top-$k$ experts are retained; all other gating weights are set to zero. The output of the MoE layer is:

$$\text{MoE}(x) = \sum_{i \in \text{TopK}} g_i(x) \cdot E_i(x)$$

where $E_i(x)$ is the output of expert $i$ applied to token $x$, and $g_i(x)$ is the gating weight for expert $i$.

### 1.4 Load Balancing Loss

Without explicit balancing, the router often converges to sending all tokens to a few "favorite" experts. The remaining experts receive no gradient signal and become useless — this is called **expert collapse**.

To prevent this, a load balancing auxiliary loss is added to the training objective:

$$\mathcal{L}_{\text{balance}} = N \sum_{i=1}^{N} f_i \cdot P_i$$

where:
- $f_i$ = fraction of tokens in the batch routed to expert $i$
- $P_i$ = average router probability assigned to expert $i$ across all tokens in the batch
- $N$ = number of experts

This loss is minimized when both $f_i$ and $P_i$ are uniform (each equal to $1/N$). The product $f_i \cdot P_i$ penalizes situations where an expert is both highly selected and highly probable, encouraging diversity.

The total training loss becomes:

$$\mathcal{L} = \mathcal{L}_{\text{task}} + \alpha \cdot \mathcal{L}_{\text{balance}}$$

where $\alpha$ is a small coefficient (typically $0.01$ to $0.1$).

### 1.5 Token Dropping

When an expert's buffer is full (more tokens routed to it than its capacity allows), excess tokens are **dropped** — they skip the MoE layer entirely and pass through via the residual connection. This is necessary for efficient batched computation on hardware but introduces noise. During inference, token dropping is typically disabled.

### 1.6 Expert Parallelism

MoE models distribute experts across GPUs. In **expert parallelism**, each GPU holds a subset of experts. Tokens are dispatched to the correct GPU via **all-to-all communication**: each GPU sends tokens to the GPU hosting the selected expert, that GPU runs the expert FFN, and results are sent back.

```
  GPU 0           GPU 1           GPU 2           GPU 3
 ┌──────┐       ┌──────┐       ┌──────┐       ┌──────┐
 │Exp 0 │       │Exp 2 │       │Exp 4 │       │Exp 6 │
 │Exp 1 │       │Exp 3 │       │Exp 5 │       │Exp 7 │
 └──┬───┘       └──┬───┘       └──┬───┘       └──┬───┘
    │               │               │               │
    └───────────────┴───────────────┴───────────────┘
              All-to-All Token Dispatch
```

This is orthogonal to tensor parallelism and pipeline parallelism and can be combined with them.

### 1.7 Landmark Models

**Switch Transformer (Fedus et al., 2022):** Uses top-1 routing (each token goes to exactly one expert). Simplifies implementation and reduces communication. Showed that scaling to 1.6T parameters with MoE outperformed dense models at equivalent compute.

**Mixtral 8x7B (Mistral AI, 2024):** 8 experts, top-2 routing per token. Total parameters: 46.7B, but only 12.9B active per forward pass. Competitive with Llama 2 70B while being much cheaper to run.

**DeepSeek-MoE (DeepSeek, 2024):** Introduces **shared experts** that process every token (providing common knowledge) plus **routed experts** selected per token. Uses finer-grained experts (more, smaller experts) for better specialization. Example: 2 shared experts + 64 routed experts with top-6 routing.

```
  DeepSeek-MoE Architecture
  ┌──────────────────────────────────────┐
  │             Input Token x            │
  │                  │                   │
  │     ┌────────────┼────────────┐      │
  │     ▼            ▼            ▼      │
  │ ┌────────┐  ┌────────┐  ┌────────┐  │
  │ │Shared  │  │Shared  │  │ Router │  │
  │ │Expert 1│  │Expert 2│  │ g(x)   │  │
  │ └───┬────┘  └───┬────┘  └──┬─────┘  │
  │     │           │     TopK selection │
  │     │           │    ┌──┬──┬──┬──┐   │
  │     │           │    ▼  ▼  ▼  ▼  ▼   │
  │     │           │   E3 E7 E12 E45 ..│
  │     │           │    │  │  │   │     │
  │     └─────┬─────┘    └──┴──┴───┘     │
  │           │              │           │
  │           └──── Sum ─────┘           │
  └──────────────────────────────────────┘
```

### 1.8 Why MoE Matters

The fundamental equation governing LLM scaling relates loss $L$ to compute $C$, data $D$, and parameters $N$. In dense models, increasing $N$ directly increases $C$. MoE decouples them:

- **Dense:** Active params $\approx$ Total params $\Rightarrow$ FLOPs $\propto N$
- **MoE:** Active params $\ll$ Total params $\Rightarrow$ FLOPs $\propto N_{\text{active}}$

This means an MoE model can store more knowledge (larger $N$) without proportionally increasing compute at inference time.

### 1.9 Problems

| Problem | Description |
|---------|-------------|
| Load imbalance | Some experts receive far more tokens; others starve |
| Expert collapse | Router converges to using only a few experts |
| Training instability | Router gradients can be noisy; loss spikes are common |
| Fine-tuning difficulty | LoRA on MoE is non-trivial — do you adapt experts, router, or both? |
| Serving complexity | Expert parallelism requires all-to-all communication; memory for all experts must be available |

---

## 2. Continued Pretraining (CPT) / Domain-Adaptive Pretraining

### 2.1 What It Is

Continued Pretraining (CPT) — also called Domain-Adaptive Pretraining (DAPT) — is the practice of resuming the causal language modeling (next-token prediction) training of a pretrained base model on a domain-specific corpus, *before* instruction tuning or alignment.

The pipeline is:

```
  ┌──────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐
  │  Base LLM │───►│  CPT on      │───►│  SFT on      │───►│  RLHF /   │
  │ (Llama 3) │    │  Domain Data │    │  Instructions│    │  DPO      │
  └──────────┘    └──────────────┘    └──────────────┘    └───────────┘
                   Crypto corpus       Q&A pairs           Preference
                   Whitepapers         Tool use            alignment
                   DeFi docs           Chat format
```

### 2.2 Why It Works

General-purpose LLMs are trained on web crawl data that covers many domains shallowly. A crypto/DeFi domain has specialized vocabulary ("impermanent loss," "slippage," "MEV," "liquidity pool"), concepts, and reasoning patterns that are underrepresented in general training data.

CPT allows the model to:
1. **Learn domain vocabulary and distributional semantics** — how domain-specific terms relate to each other
2. **Absorb factual knowledge** — protocol mechanics, regulatory frameworks, market microstructure
3. **Adjust internal representations** — shift the model's "world model" toward the domain

This is not achievable through prompting or few-shot examples alone because those approaches don't update the model's parameters.

### 2.3 Data Sources for Binance

For a crypto-focused CPT, data sources would include:
- Cryptocurrency whitepapers (Bitcoin, Ethereum, Solana, etc.)
- DeFi protocol documentation (Uniswap, Aave, Compound, etc.)
- Blockchain technical specifications (EIPs, BIPs)
- Crypto news articles (CoinDesk, The Block, etc.)
- Regulatory filings and guidelines (SEC, MAS, MiCA)
- Binance Academy articles
- Smart contract code (Solidity, Rust)
- On-chain data descriptions and schemas

### 2.4 Key Considerations

**Learning Rate.** CPT uses a much lower learning rate than original pretraining — typically $1 \times 10^{-5}$ to $5 \times 10^{-5}$, compared to $1 \times 10^{-4}$ to $3 \times 10^{-4}$ during original pretraining. The model has already learned general language structure; you are making incremental adjustments, not learning from scratch. Too high a learning rate causes catastrophic forgetting.

**Data Mixing.** A critical technique is to mix domain-specific data with a proportion of general data during CPT. A common ratio is 70-90% domain data, 10-30% general data. This prevents the model from forgetting general capabilities:

$$\mathcal{D}_{\text{CPT}} = \alpha \cdot \mathcal{D}_{\text{domain}} + (1 - \alpha) \cdot \mathcal{D}_{\text{general}}, \quad \alpha \in [0.7, 0.9]$$

**Tokenizer Extension (Optional).** If the domain has many unseen tokens (e.g., contract addresses, protocol names), you can extend the tokenizer by adding new tokens and initializing their embeddings as the mean of semantically related existing token embeddings. This reduces sequence length for domain text.

**Replay.** To further mitigate forgetting, some approaches periodically replay a fraction of the original pretraining data, or use elastic weight consolidation (EWC):

$$\mathcal{L}_{\text{EWC}} = \mathcal{L}_{\text{CLM}} + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_i^*)^2$$

where $F_i$ is the Fisher information of parameter $i$ and $\theta_i^*$ are the pretrained weights.

### 2.5 Notable Examples

| Model | Domain | Base | CPT Data |
|-------|--------|------|----------|
| CodeLlama | Code | Llama 2 | 500B tokens of code |
| BioMedLM | Biomedical | GPT-2 architecture | PubMed abstracts + papers |
| BloombergGPT | Finance | Trained from scratch | 363B tokens of financial data |
| Minerva | Math | PaLM | Mathematical web pages + papers |
| Galactica | Science | Trained from scratch | 106B tokens of scientific text |

### 2.6 Problems

- **Catastrophic forgetting**: the model loses general knowledge. Mitigated by data mixing and low LR.
- **Data quality**: domain corpora are often noisy (scraped docs, PDFs, malformed text). Requires extensive cleaning.
- **Compute cost**: CPT on 50-100B tokens still requires significant GPU hours.
- **Evaluation difficulty**: hard to measure domain knowledge gain without domain-specific benchmarks.

---

## 3. GRPO (Group Relative Policy Optimization)

### 3.1 Background: The PPO Burden

Standard RLHF uses PPO (Proximal Policy Optimization) which requires four models in memory simultaneously:

1. **Policy model** $\pi_\theta$ (the LLM being trained)
2. **Reference model** $\pi_{\text{ref}}$ (frozen copy for KL penalty)
3. **Reward model** $R_\phi$
4. **Value model** $V_\psi$ (critic that estimates expected future reward)

The value model is typically the same size as the policy model, so PPO effectively requires 4x the memory of inference. This is the main bottleneck.

### 3.2 GRPO Core Idea

**GRPO (Group Relative Policy Optimization)**, introduced in DeepSeek-R1, eliminates the value model entirely. Instead of estimating advantages via a learned critic, GRPO estimates advantages by **sampling a group of responses** for each prompt and using the **relative reward ranking within the group**.

### 3.3 Algorithm

For each prompt $x$ in the batch:

1. **Sample a group** of $G$ responses: $\{y_1, y_2, \ldots, y_G\}$ from the current policy $\pi_\theta(\cdot | x)$
2. **Score each response** with the reward model: $r_1, r_2, \ldots, r_G$
3. **Normalize rewards within the group** to get advantages:

$$\hat{A}_i = \frac{r_i - \text{mean}(\{r_1, \ldots, r_G\})}{\text{std}(\{r_1, \ldots, r_G\})}$$

4. **Update policy** using a clipped objective (similar to PPO's clipping):

$$\mathcal{L}_{\text{GRPO}} = -\frac{1}{G} \sum_{i=1}^{G} \min\left(\frac{\pi_\theta(y_i | x)}{\pi_{\text{ref}}(y_i | x)} \hat{A}_i, \; \text{clip}\left(\frac{\pi_\theta(y_i | x)}{\pi_{\text{ref}}(y_i | x)}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i\right) - \beta \, D_{KL}(\pi_\theta \| \pi_{\text{ref}})$$

where:
- $\epsilon$ is the clipping parameter (typically 0.2)
- $\beta$ is the KL penalty coefficient
- The ratio $\frac{\pi_\theta(y_i|x)}{\pi_{\text{ref}}(y_i|x)}$ is computed at the token level and aggregated

### 3.4 Why Group Normalization Works

The key insight is that for RL to work, you don't need absolute advantage estimates — you just need to know **which responses are better than others for the same prompt**. By normalizing within the group:

- Responses with above-average reward get positive advantage (reinforced)
- Responses with below-average reward get negative advantage (suppressed)
- The scale is automatically calibrated by the standard deviation

This is essentially a **REINFORCE with baseline** algorithm where the baseline is the group mean, but it avoids the need to train a separate value network.

### 3.5 Comparison with PPO and DPO

```
  ┌────────────────────────────────────────────────────────┐
  │                    Models in Memory                     │
  ├──────────┬────────────┬──────────┬────────────┬────────┤
  │ Method   │ Policy     │ Reference│ Reward     │ Value  │
  ├──────────┼────────────┼──────────┼────────────┼────────┤
  │ PPO      │     ✓      │    ✓     │     ✓      │   ✓    │
  │ DPO      │     ✓      │    ✓     │     ✗      │   ✗    │
  │ GRPO     │     ✓      │    ✓     │     ✓      │   ✗    │
  │ SimPO    │     ✓      │    ✗     │     ✗      │   ✗    │
  └──────────┴────────────┴──────────┴────────────┴────────┘
```

| Aspect | PPO | DPO | GRPO |
|--------|-----|-----|------|
| Needs value model | Yes | No | No |
| Needs reward model | Yes | No (implicit) | Yes |
| Online generation | Yes | No (offline) | Yes |
| Reward type | Scalar | Binary preference | Scalar |
| Memory footprint | 4 models | 2 models | 3 models |

### 3.6 GRPO in DeepSeek-R1

DeepSeek used GRPO with $G = 64$ responses per prompt. The reward was a combination of:
- **Correctness reward**: binary (1 if the final answer is correct, 0 otherwise)
- **Format reward**: encourages `<think>...</think>` reasoning format

The result was a model that spontaneously learned chain-of-thought reasoning, self-verification, and even "aha moments" — all emerging from RL without explicit supervision of the reasoning process.

### 3.7 Problems

- **High variance**: group normalization introduces variance, especially with small $G$. With $G=4$, the advantage estimates are very noisy.
- **Sample inefficiency**: generating $G$ responses per prompt is expensive. With $G=64$ and sequence length 2048, that is 131K tokens per prompt.
- **Reward hacking**: the model may find shortcuts to maximize the reward signal without genuine improvement.
- **Distribution shift**: as the policy improves, the group's reward distribution shifts, potentially causing instability.

---

## 4. SimPO, ORPO, KTO

These are post-DPO alignment methods, each addressing a specific limitation of DPO.

### 4.1 SimPO (Simple Preference Optimization)

**Motivation.** DPO requires a reference model $\pi_{\text{ref}}$ in memory, doubling the memory requirement. SimPO eliminates it.

**Key idea.** Use the **length-normalized average log-probability** of the response as an implicit reward:

$$\hat{r}(y | x) = \frac{\beta}{|y|} \log \pi_\theta(y | x)$$

The length normalization by $|y|$ prevents the model from favoring shorter responses (which naturally have higher total log-probability).

**Loss function:**

$$\mathcal{L}_{\text{SimPO}} = -\log \sigma\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w | x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l | x) - \gamma\right)$$

where:
- $y_w$ = preferred (winning) response
- $y_l$ = dispreferred (losing) response
- $\beta$ = scaling parameter (controls reward sensitivity)
- $\gamma$ = margin parameter (ensures a minimum gap between preferred and dispreferred)
- $\sigma$ = sigmoid function

The margin $\gamma > 0$ pushes the model to assign meaningfully higher probability to the preferred response, not just marginally higher.

**Derivation insight.** In DPO, the implicit reward is $r(y|x) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$. SimPO replaces $\pi_{\text{ref}}$ with a uniform distribution, making the reward depend only on $\pi_\theta$. The length normalization compensates for the fact that longer sequences accumulate more log-probability mass.

### 4.2 ORPO (Odds Ratio Preference Optimization)

**Motivation.** Standard alignment is a two-stage process: SFT first, then preference optimization (DPO/PPO). ORPO combines both stages into one.

**Key idea.** Add an odds-ratio-based preference loss directly to the SFT loss:

$$\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}}$$

where the odds ratio loss is:

$$\mathcal{L}_{\text{OR}} = -\log \sigma\left(\log \frac{P_\theta(y_w | x)}{1 - P_\theta(y_w | x)} - \log \frac{P_\theta(y_l | x)}{1 - P_\theta(y_l | x)}\right)$$

Here, $P_\theta(y|x) = \exp\left(\frac{1}{|y|} \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t})\right)$ is the geometric mean token probability of the response.

**Why odds ratios?** The odds ratio $\frac{P}{1-P}$ naturally separates preferred from dispreferred responses in probability space. When $P$ is close to 1, the odds ratio diverges, providing a strong gradient signal. When $P$ is close to 0, the odds ratio approaches 0.

**Advantage.** Single-stage training saves compute and avoids the "alignment tax" where DPO can degrade SFT performance.

### 4.3 KTO (Kahneman-Tversky Optimization)

**Motivation.** DPO requires *paired* preferences $(y_w, y_l)$ for the same prompt. In practice, collecting paired data is expensive. Often you only have "this response is good" or "this response is bad" labels — unpaired binary feedback.

**Key idea.** Inspired by Kahneman and Tversky's prospect theory, KTO uses a loss that treats gains (good responses) and losses (bad responses) asymmetrically:

For desirable outputs $y_w$:

$$\mathcal{L}_w = 1 - \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - z_{\text{ref}}\right]\right)$$

For undesirable outputs $y_l$:

$$\mathcal{L}_l = 1 - \sigma\left(\beta \left[z_{\text{ref}} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right]\right)$$

where $z_{\text{ref}}$ is a reference point (typically the KL divergence between policy and reference on the batch). The overall loss is:

$$\mathcal{L}_{\text{KTO}} = \lambda_w \mathbb{E}[\mathcal{L}_w] + \lambda_l \mathbb{E}[\mathcal{L}_l]$$

The asymmetric weighting ($\lambda_w \neq \lambda_l$) reflects loss aversion: humans weigh negative outcomes more heavily than positive ones.

### 4.4 Comparison Table

| Method | Reference Model | Paired Data | Online Gen | Stages | Memory |
|--------|:-:|:-:|:-:|:-:|:-:|
| DPO | Yes | Yes | No | 2 (SFT+DPO) | 2x |
| GRPO | Yes | No (uses rewards) | Yes | 2+ | 3x |
| SimPO | **No** | Yes | No | 2 (SFT+SimPO) | **1x** |
| ORPO | **No** | Yes | No | **1 (combined)** | **1x** |
| KTO | Yes | **No (unpaired)** | No | 2 (SFT+KTO) | 2x |

---

## 5. SPLADE (Sparse Lexical and Expansion Model)

### 5.1 The Problem with BM25

BM25 matches documents to queries based on **exact lexical overlap**: the query term must appear in the document. This fails when:
- Query says "dog" but document says "puppy" (synonym problem)
- Query says "how to fix a flat tire" but document uses "tyre puncture repair" (vocabulary mismatch)

Dense retrievers (e.g., sentence transformers) solve this via learned semantic representations, but they lose interpretability and are slower for large-scale retrieval without approximate nearest neighbor (ANN) indexes.

### 5.2 SPLADE Core Idea

SPLADE (Sparse Lexical AnD Expansion) uses a BERT-based model to produce **sparse, vocabulary-sized representations** for both queries and documents. These representations:
1. **Re-weight existing terms** (learn which terms are important, like TF-IDF but learned)
2. **Expand to related terms** (the model can assign non-zero weight to terms that don't appear in the text)

```
  SPLADE Architecture
  ┌─────────────────────────────────────────────┐
  │  Input: "Bitcoin liquidation event"          │
  │                    │                         │
  │              ┌─────┴─────┐                   │
  │              │   BERT    │                   │
  │              │ Encoder   │                   │
  │              └─────┬─────┘                   │
  │                    │                         │
  │         h_1, h_2, ..., h_T  (hidden states)  │
  │                    │                         │
  │              ┌─────┴─────┐                   │
  │              │  MLM Head  │ (project to vocab)│
  │              └─────┬─────┘                   │
  │                    │                         │
  │     logits: [V]-dim vector per token         │
  │                    │                         │
  │         log(1 + ReLU(logits))                │
  │                    │                         │
  │         Max-pool over token positions        │
  │                    │                         │
  │  Output: sparse [V]-dim vector               │
  │  e.g., {"bitcoin": 2.1, "liquidation": 1.8, │
  │         "crypto": 0.9, "margin": 0.7,       │
  │         "forced": 0.4, "selling": 0.3, ...}  │
  └─────────────────────────────────────────────┘
```

### 5.3 Mathematical Formulation

Given input text with token representations $\{h_1, h_2, \ldots, h_T\}$ from BERT, each $h_i \in \mathbb{R}^d$:

1. Project each token through the MLM head to get vocabulary-sized logits: $l_i = W_{\text{MLM}} h_i + b_{\text{MLM}} \in \mathbb{R}^{|V|}$

2. Apply log-saturation activation for sparsity:

$$w_j = \sum_{i=1}^{T} \log(1 + \text{ReLU}(l_i^{(j)}))$$

where $w_j$ is the weight for vocabulary term $j$ in the final representation.

The $\log(1 + \cdot)$ function saturates large values, encouraging sparsity. The $\text{ReLU}$ ensures non-negativity. The sum over token positions aggregates evidence from all input tokens.

3. Retrieval via **dot product** between sparse query vector $q$ and sparse document vector $d$:

$$\text{score}(q, d) = \sum_{j \in V} q_j \cdot d_j$$

Because both vectors are sparse, this can be computed using an **inverted index**, just like BM25.

### 5.4 FLOPS Regularization

To control sparsity, SPLADE adds a regularization term that penalizes the total number of floating point operations (proportional to the number of non-zero entries):

$$\mathcal{L}_{\text{FLOPS}} = \sum_{j=1}^{|V|} \bar{a}_j^2$$

where $\bar{a}_j$ is the average weight of term $j$ across all documents in the batch. This encourages most terms to have zero weight on average.

Total loss:

$$\mathcal{L} = \mathcal{L}_{\text{ranking}} + \lambda_q \mathcal{L}_{\text{FLOPS}}^{q} + \lambda_d \mathcal{L}_{\text{FLOPS}}^{d}$$

### 5.5 SPLADE++ and Distillation

SPLADE++ improves quality by distilling from a cross-encoder reranker. The cross-encoder provides soft relevance labels, and SPLADE is trained to match these scores via KL divergence or MSE loss. This brings the quality of the sparse retriever closer to the cross-encoder while maintaining the efficiency of inverted index retrieval.

### 5.6 Where SPLADE Fits in a RAG Pipeline

```
  Query: "What happens during Bitcoin liquidation?"
       │
       ├─── BM25 ──────────────────── Top-100 docs ──┐
       │                                               │
       ├─── SPLADE ─────────────────── Top-100 docs ──┼── RRF ── Top-20 ── Reranker ── Top-5
       │                                               │
       └─── Dense (bi-encoder) ────── Top-100 docs ──┘
```

SPLADE is complementary to both BM25 (provides learned expansion) and dense retrieval (provides interpretable sparse matching). Fusion via Reciprocal Rank Fusion (RRF) combines their strengths.

### 5.7 Advantages and Limitations

| Advantage | Limitation |
|-----------|-----------|
| Term expansion (synonym matching) | Requires fine-tuning BERT (not zero-shot) |
| Uses inverted index (fast, scalable) | Index size larger than BM25 (expanded terms) |
| Interpretable (can see which terms matched) | Vocabulary-bound (no cross-lingual by default) |
| Complements dense retrieval | Training needs relevance labels or distillation |

---

## 6. Graph RAG

### 6.1 Motivation

Standard vector-based RAG retrieves individual chunks independently. It fails when answering questions that require:
- **Multi-hop reasoning**: "Which DeFi protocols were affected by the same exploit as the one that hit Euler Finance?"
- **Global summarization**: "What are the main themes in Binance's 2024 compliance reports?"
- **Relationship awareness**: "How are Lido, Rocket Pool, and Coinbase connected in the liquid staking ecosystem?"

These questions require understanding *relationships between entities* across multiple documents.

### 6.2 Graph RAG Pipeline (Microsoft, 2024)

```
  ┌──────────────┐
  │  Raw Corpus   │
  └──────┬───────┘
         │ Step 1: Chunk
         ▼
  ┌──────────────┐
  │   Chunks      │
  └──────┬───────┘
         │ Step 2: LLM extracts entities & relationships
         ▼
  ┌──────────────────────────────────────┐
  │  (Entity: "Euler Finance")           │
  │  (Entity: "Flash Loan Attack")       │
  │  (Relationship: "was exploited via") │
  └──────┬───────────────────────────────┘
         │ Step 3: Build Knowledge Graph
         ▼
  ┌──────────────────────────┐
  │   Knowledge Graph         │
  │   Nodes: entities         │
  │   Edges: relationships    │
  └──────┬───────────────────┘
         │ Step 4: Community Detection (Leiden)
         ▼
  ┌──────────────────────────┐
  │  Communities (clusters)   │
  │  C1: DeFi exploits        │
  │  C2: Stablecoin mechanics │
  │  C3: Regulatory actions   │
  └──────┬───────────────────┘
         │ Step 5: LLM summarizes each community
         ▼
  ┌──────────────────────────┐
  │  Community Summaries      │
  │  "This cluster covers..." │
  └──────────────────────────┘
```

### 6.3 Two Query Modes

**Local Search.** For specific questions like "What vulnerabilities did Euler Finance have?"
1. Extract entities from query ("Euler Finance")
2. Find node in graph
3. Traverse local neighborhood (1-2 hops)
4. Collect related entities and their source text
5. Feed subgraph context to LLM for answer generation

**Global Search.** For broad questions like "What are the main themes in this corpus?"
1. Retrieve community summaries at the appropriate hierarchy level
2. Map-reduce: each community summary is sent to the LLM for partial answers
3. Aggregate partial answers into a final response

### 6.4 Leiden Algorithm for Community Detection

The Leiden algorithm is an improved version of Louvain for detecting communities in graphs. It optimizes **modularity**:

$$Q = \frac{1}{2m} \sum_{ij} \left[A_{ij} - \frac{k_i k_j}{2m}\right] \delta(c_i, c_j)$$

where $A_{ij}$ is the adjacency matrix, $k_i$ is the degree of node $i$, $m$ is the total number of edges, and $\delta(c_i, c_j) = 1$ if nodes $i$ and $j$ are in the same community.

The algorithm produces a hierarchical community structure, allowing Graph RAG to operate at different levels of granularity.

### 6.5 When Graph RAG Excels vs. Fails

| Excels | Fails |
|--------|-------|
| Multi-hop questions | Simple factual lookup |
| "What are the main themes?" | "What is the price of BTC?" |
| Relationship-heavy domains | Domains with few entities |
| Corpus-level understanding | Single-document questions |

### 6.6 Problems

- **Entity extraction quality**: LLM-based extraction is imperfect; missed entities or spurious relationships degrade the graph
- **Graph maintenance**: when documents are updated, the graph must be rebuilt or incrementally updated
- **Compute cost**: building the graph requires many LLM calls (one per chunk for extraction, one per community for summarization)
- **Latency**: graph traversal adds latency compared to vector retrieval
- **Cost at scale**: for large corpora, the indexing cost (in LLM API calls) can be very high

---

## 7. Self-RAG and Corrective RAG (CRAG)

### 7.1 The Problem with Naive RAG

In standard RAG, retrieved documents are always concatenated into the prompt. This creates problems:
- **Irrelevant context**: if retrieval returns poor results, the LLM may hallucinate based on irrelevant text
- **Always-retrieve overhead**: some questions don't need retrieval ("What is 2+2?")
- **No quality control**: the LLM has no mechanism to judge whether retrieved context is helpful

### 7.2 Self-RAG (Asai et al., 2023)

Self-RAG trains the LLM itself to *decide* when to retrieve and to *evaluate* the quality of retrieved documents using special reflection tokens.

**Special Tokens:**
- `[Retrieve]` — Yes/No: does the model need to retrieve external information?
- `[IsRel]` — Relevant/Irrelevant: is the retrieved passage relevant to the query?
- `[IsSup]` — Fully Supported / Partially Supported / Not Supported: is the generated response supported by the retrieved passage?
- `[IsUse]` — Utility score (1-5): how useful is the overall response?

**Generation Flow:**

```
  Query: "What is the TVL of Aave?"
       │
       ▼
  ┌────────────────────────┐
  │ LLM generates tokens   │
  │ ...                    │
  │ Outputs: [Retrieve=Yes]│
  └────────┬───────────────┘
           │
           ▼
  ┌────────────────────────┐
  │ Retriever fetches docs │
  │ d1, d2, d3             │
  └────────┬───────────────┘
           │
           ▼
  ┌────────────────────────┐
  │ LLM evaluates each doc │
  │ d1: [IsRel=Yes]        │
  │ d2: [IsRel=No]  ← skip│
  │ d3: [IsRel=Yes]        │
  └────────┬───────────────┘
           │ Use d1, d3
           ▼
  ┌────────────────────────┐
  │ LLM generates answer   │
  │ with d1, d3 as context │
  │ Outputs: [IsSup=Full]  │
  │ Outputs: [IsUse=4]     │
  └────────────────────────┘
```

**Training.** Self-RAG is trained by augmenting the training data with reflection tokens. A critic model (or GPT-4) annotates training examples with the special tokens, and the LLM is fine-tuned to predict both the content tokens and the reflection tokens.

**Inference-time control.** At inference time, you can adjust thresholds for the reflection tokens:
- Set high `[IsSup]` threshold for factual accuracy (conservative)
- Set low `[Retrieve]` threshold to retrieve more often (thorough)

### 7.3 Corrective RAG (CRAG)

CRAG (Yan et al., 2024) adds a **lightweight retrieval evaluator** that classifies retrieved documents into three categories before they reach the LLM.

**Pipeline:**

```
  Query ──► Retriever ──► Retrieved Docs ──► Evaluator
                                                │
                          ┌─────────────────────┼─────────────────┐
                          ▼                     ▼                 ▼
                      "Correct"            "Ambiguous"        "Incorrect"
                          │                     │                 │
                          ▼                     ▼                 ▼
                   Use retrieved          Refine query        Web search
                   docs directly          + Re-retrieve       as fallback
                          │                     │                 │
                          └─────────────────────┴─────────────────┘
                                           │
                                           ▼
                                    LLM generates
                                    final answer
```

**Evaluator.** The retrieval evaluator is a small classifier (e.g., fine-tuned DeBERTa) that scores each document on a scale. Based on an aggregate score:
- **Correct** (score > $\tau_{\text{high}}$): at least one document is highly relevant. Use them directly.
- **Ambiguous** ($\tau_{\text{low}} < \text{score} < \tau_{\text{high}}$): documents are partially relevant. Decompose the query, refine, and re-retrieve.
- **Incorrect** (score < $\tau_{\text{low}}$): documents are irrelevant. Discard them and fall back to web search.

**Knowledge Refinement.** CRAG also applies a **decompose-then-recompose** step: it extracts relevant sentences from retrieved documents (filtering out noise) before passing to the LLM.

### 7.4 Self-RAG vs. CRAG Comparison

| Aspect | Self-RAG | CRAG |
|--------|----------|------|
| Evaluator | Built into the LLM | External lightweight model |
| Training | Requires SFT with special tokens | Plug-and-play evaluator |
| Flexibility | High (adjustable thresholds) | Moderate |
| Latency | One LLM call (with reflection) | Additional evaluator call |
| Fallback | Skip irrelevant docs | Web search fallback |
| Complexity | Higher (custom training) | Lower (modular design) |

### 7.5 Problems

- **Self-RAG**: requires training with special tokens — cannot be applied to off-the-shelf models; added latency from reflection steps; training data annotation is expensive
- **CRAG**: evaluator quality is a bottleneck; web search fallback adds latency and may not be available in all deployments; threshold tuning ($\tau_{\text{high}}$, $\tau_{\text{low}}$) requires validation data

---

## 8. Model Merging Techniques

### 8.1 Motivation

You fine-tune separate LoRA adapters for different tasks:
- Adapter A: crypto sentiment analysis
- Adapter B: named entity recognition (NER) for blockchain entities
- Adapter C: regulatory document classification

At serving time, you don't want to load three separate adapters — you want one merged model that handles all tasks. Model merging combines multiple fine-tuned models into a single model *without additional training*.

### 8.2 Task Arithmetic

The simplest approach. Define a **task vector** as the difference between the fine-tuned weights and the base weights:

$$\tau_i = W_{\text{task}_i} - W_{\text{base}}$$

The merged model is:

$$W_{\text{merged}} = W_{\text{base}} + \lambda_1 \tau_1 + \lambda_2 \tau_2 + \cdots + \lambda_k \tau_k$$

where $\lambda_i$ are scaling coefficients (typically in $[0.3, 1.0]$) that control the influence of each task.

**Intuition.** Fine-tuning moves the model in a specific "direction" in weight space. Task arithmetic assumes these directions are approximately orthogonal, so they can be combined linearly.

**Problem.** Task vectors often *interfere*: when one task pushes a parameter up and another pushes it down, the average cancels out the signal. This is called **parameter interference**.

### 8.3 TIES-Merging (Trim, Elect Sign, Merge)

TIES-Merging (Yadav et al., 2023) addresses parameter interference through three steps:

```
  Step 1: TRIM                Step 2: ELECT SIGN         Step 3: MERGE
  ┌──────────────┐           ┌──────────────┐           ┌──────────────┐
  │ Task Vector  │           │ Remaining    │           │ Unified Sign │
  │ [0.5, -0.01, │  ──────► │ [0.5, 0,     │  ──────► │ [0.5, 0,     │
  │  0.3, -0.02, │  Keep    │  0.3, 0,     │  Majority│  0.3, 0,     │
  │  -0.4]       │  top-k%  │  -0.4]       │  vote on │  -0.4]       │
  └──────────────┘  by mag  └──────────────┘  sign     └──────┬───────┘
                                                              │
  For each parameter:                                         ▼
  1. If |delta| < threshold → set to 0 (trim noise)   Average remaining
  2. For conflicting signs across tasks → majority vote deltas with
  3. Average the surviving, sign-aligned deltas        agreed sign
```

Formally:

1. **Trim:** For each task vector $\tau_i$, set entries with magnitude below the $p$-th percentile to zero. This removes noise — small parameter changes that are likely not meaningful.

2. **Elect Sign:** For each parameter position $j$, look at the signs of $\tau_{1,j}, \tau_{2,j}, \ldots$ across tasks. Take a **majority vote** to determine the "elected" sign. Set any task vector entry that disagrees with the elected sign to zero.

3. **Merge:** Average the remaining non-zero entries:

$$\Delta W_{\text{merged}} = W_{\text{base}} + \sum_{t=1}^{k} \lambda_t \cdot \text{TIES}(\tau_t)$$

### 8.4 DARE (Drop And REscale)

DARE (Yu et al., 2024) takes a different approach: it randomly drops delta parameters and rescales the survivors.

For each task vector $\tau_i$:

1. Create a random binary mask $m$ where each entry is 0 with probability $p$ and 1 with probability $1-p$
2. Apply mask and rescale:

$$\tilde{\tau}_i = \frac{m \odot \tau_i}{1 - p}$$

3. Merge:

$$W_{\text{merged}} = W_{\text{base}} + \sum_{i} \lambda_i \tilde{\tau}_i$$

**Intuition.** Most delta parameters are redundant (the "lottery ticket" hypothesis applied to fine-tuning deltas). By randomly dropping them, you reduce interference between task vectors. The rescaling by $\frac{1}{1-p}$ preserves the expected magnitude.

DARE can be combined with TIES: apply DARE's random dropping first, then TIES's sign election.

### 8.5 Model Soups

The simplest merging strategy: average multiple checkpoints of the *same* fine-tuning run (or multiple runs with different hyperparameters):

$$W_{\text{soup}} = \frac{1}{K} \sum_{k=1}^{K} W_k$$

This is surprisingly effective because:
- Different random seeds explore different regions of the loss landscape
- Averaging tends to land in flatter minima (better generalization)
- It is equivalent to an ensemble at the weight level

### 8.6 Practical Application at Binance

```
  Base Model (Llama 3 8B)
       │
       ├── LoRA fine-tune on sentiment ──► Adapter A
       │
       ├── LoRA fine-tune on NER ────────► Adapter B
       │
       ├── LoRA fine-tune on QA ─────────► Adapter C
       │
       └── TIES-Merge (A, B, C) ────────► Single Merged Model
                                               │
                                          Deploy once
                                          Serve all tasks
```

### 8.7 Problems

| Problem | Description | Mitigation |
|---------|-------------|------------|
| Task interference | Different tasks push parameters in conflicting directions | TIES sign election, DARE random dropping |
| Quality degradation | Merged model is worse than individual models | Careful $\lambda$ tuning, validation on all tasks |
| Incompatible architectures | Can only merge models with identical architectures | Ensure same base model and LoRA config |
| No guarantees | Merging is empirical; no theoretical guarantee of success | Always benchmark merged model |

---

## 9. Problems & Mitigations

### 9.1 MoE

| Problem | Mitigation |
|---------|-----------|
| Load imbalance | Auxiliary load balancing loss $\mathcal{L}_{\text{balance}}$; capacity factor limits; random routing noise |
| Expert collapse | Load balancing loss; dropout on router logits; initialization diversity |
| Training instability | Router z-loss: $\mathcal{L}_z = \frac{1}{B} \sum_i (\log \sum_j e^{z_{ij}})^2$ penalizes large logits; gradient clipping |
| Fine-tuning difficulty | Fine-tune only shared layers + router; or use MoLoRA (LoRA on each expert) |
| Serving memory | Expert offloading to CPU; expert parallelism; quantize inactive experts more aggressively |

### 9.2 CPT

| Problem | Mitigation |
|---------|-----------|
| Catastrophic forgetting | Data mixing (30% general data); low learning rate ($\leq 5 \times 10^{-5}$); EWC regularization |
| Data quality | Rigorous deduplication; perplexity filtering; domain expert review |
| Compute cost | Efficient training (Flash Attention, gradient checkpointing); smaller focused corpus |
| Evaluation | Create domain-specific benchmarks before CPT; measure both domain and general performance |

### 9.3 GRPO

| Problem | Mitigation |
|---------|-----------|
| High variance | Increase group size $G$ (32-64); use reward normalization with running statistics |
| Sample inefficiency | Importance sampling from replay buffer; reject sampling to reuse high-quality generations |
| Reward hacking | Multi-objective rewards; diverse prompt sampling; human spot-checks |
| Distribution shift | Periodic reference model updates; KL penalty scheduling |

### 9.4 Alignment Variants (SimPO, ORPO, KTO)

| Problem | Mitigation |
|---------|-----------|
| SimPO: length bias despite normalization | Tune $\gamma$ margin carefully; add explicit length penalty |
| ORPO: unstable joint training | Curriculum: start with SFT-dominated loss, increase $\lambda$ over training |
| KTO: need balanced good/bad data | Oversample the minority class; adjust $\lambda_w / \lambda_l$ ratio |
| All: preference data quality | Use strong annotators; filter low-agreement examples; iterative data refinement |

### 9.5 SPLADE

| Problem | Mitigation |
|---------|-----------|
| Index size larger than BM25 | Aggressive FLOPS regularization; top-k sparsification at index time |
| Training data requirement | Distill from cross-encoder; use hard negatives from BM25 |
| Domain adaptation | Fine-tune on domain relevance judgments; or use GPL (generative pseudo-labeling) |
| Vocabulary-bound | Combine with dense retriever via hybrid search (RRF) |

### 9.6 Graph RAG

| Problem | Mitigation |
|---------|-----------|
| Entity extraction errors | Use multiple extraction passes; ensemble extractors; human-in-the-loop validation |
| Graph maintenance | Incremental graph updates; version the graph alongside the corpus |
| High build cost | Batch extraction with cheaper models; cache extracted entities |
| Latency | Pre-compute community summaries; index graph for fast traversal |

### 9.7 Self-RAG / CRAG

| Problem | Mitigation |
|---------|-----------|
| Self-RAG: not plug-and-play | Use existing Self-RAG fine-tuned models; or use CRAG instead |
| Self-RAG: annotation cost | Use GPT-4 for automated annotation; active learning for annotation selection |
| CRAG: evaluator bottleneck | Use lightweight evaluator (DeBERTa-base); cache evaluator results |
| CRAG: web search dependency | Configure multiple fallback sources; rate-limit web search |

### 9.8 Model Merging

| Problem | Mitigation |
|---------|-----------|
| Parameter interference | TIES sign election; DARE random dropping; orthogonal fine-tuning (OFT) |
| Quality degradation | Grid search over $\lambda$ values; evaluate on all task benchmarks |
| Incompatible models | Standardize base model and LoRA configuration across all tasks |
| No theoretical guarantees | Treat merging as a heuristic; always validate empirically; keep individual adapters as fallback |

---

## 10. Interview Q&A

### Q1 (Basic): What is a Mixture of Experts model and why is it useful?

**Answer.** A Mixture of Experts (MoE) model replaces the feed-forward network (FFN) in a Transformer block with multiple "expert" FFNs and a router that selects which experts process each token. Only a subset (top-$k$) of experts are activated per token, making the model **sparsely activated**. This means the model can have far more total parameters (and thus more capacity to store knowledge) while keeping the per-token compute cost roughly constant. For example, Mixtral 8x7B has 46.7B total parameters but only 12.9B are active per token, achieving performance comparable to much larger dense models at a fraction of the inference cost.

---

### Q2 (Basic): What is Continued Pretraining and when would you use it?

**Answer.** Continued Pretraining (CPT) is the process of resuming language model pretraining on a domain-specific corpus (e.g., crypto text, biomedical literature) before instruction tuning. You would use it when the target domain has specialized vocabulary, concepts, and reasoning patterns that are underrepresented in the base model's training data. The pipeline is: Base Model -> CPT on domain data -> SFT on instructions -> RLHF/DPO alignment. Key considerations include using a low learning rate ($1 \times 10^{-5}$ to $5 \times 10^{-5}$) and mixing domain data with general data (70/30 ratio) to prevent catastrophic forgetting.

---

### Q3 (Basic): What is SPLADE and how does it differ from BM25?

**Answer.** SPLADE is a learned sparse retrieval model that uses BERT to produce sparse, vocabulary-sized representations for queries and documents. Unlike BM25, which can only match on exact terms present in the document, SPLADE can **expand** terms: a query with "dog" can match a document containing "puppy" because SPLADE assigns non-zero weight to semantically related terms. Like BM25, SPLADE representations are sparse and can be stored in an inverted index for efficient retrieval. The key formula is $w_j = \sum_i \log(1 + \text{ReLU}(h_i^j))$, which aggregates evidence from BERT's hidden states projected through the MLM head.

---

### Q4 (Intermediate): Explain how GRPO eliminates the need for a value model compared to PPO.

**Answer.** In PPO, a value model $V_\psi(s)$ estimates the expected future reward from a given state, which is used to compute advantages: $A(s, a) = R - V_\psi(s)$. This requires training and maintaining a separate neural network of comparable size to the policy. GRPO replaces this with **group-relative advantages**: for each prompt, it generates a group of $G$ responses, scores them with the reward model, and normalizes rewards within the group: $\hat{A}_i = (r_i - \text{mean}(r)) / \text{std}(r)$. This group mean acts as a baseline, serving the same variance-reduction role as the value model. The tradeoff is that you need to generate many samples per prompt ($G = 32$-$64$) rather than just one, but you save the memory and training complexity of a separate value network.

---

### Q5 (Intermediate): Compare DPO, SimPO, and ORPO. When would you choose each?

**Answer.**
- **DPO**: requires paired preferences and a reference model. Choose when you have high-quality paired preference data and sufficient GPU memory for two models. It is the most established method with well-understood properties.
- **SimPO**: eliminates the reference model by using length-normalized log-probability as the implicit reward. Choose when memory is constrained — you only need one model in memory. The margin parameter $\gamma$ provides an extra degree of control.
- **ORPO**: combines SFT and alignment into a single training stage. Choose when you want to reduce training complexity and compute — instead of SFT then DPO, you do one pass. It uses odds ratios rather than log-probability ratios, which provides good gradient signal near the extremes of the probability range.

In all cases, the quality of preference data matters more than the choice of algorithm.

---

### Q6 (Intermediate): How does Graph RAG's global search mode work and when is it better than standard RAG?

**Answer.** In Graph RAG's global search, the system does not retrieve individual chunks. Instead, it: (1) builds a knowledge graph from the corpus via LLM-based entity/relationship extraction, (2) runs community detection (Leiden algorithm) to identify topic clusters, and (3) pre-generates summaries of each community. At query time, relevant community summaries are retrieved and passed to the LLM in a map-reduce fashion — each summary produces a partial answer, which are then aggregated.

This is better than standard RAG for **corpus-level** questions like "What are the main regulatory trends discussed in these documents?" Standard RAG would retrieve a few random relevant chunks and miss the global picture. Graph RAG's community summaries provide a structured, hierarchical overview. However, for simple factual lookup ("What is the TVL of Aave?"), standard RAG is faster and cheaper.

---

### Q7 (Intermediate): Explain the TIES-Merging algorithm step by step.

**Answer.** TIES-Merging resolves parameter interference when combining multiple task-specific models:

1. **Trim**: For each task vector $\tau_i = W_{\text{task}_i} - W_{\text{base}}$, remove (set to zero) entries whose magnitude is below the $p$-th percentile. This eliminates noise — small parameter changes that are not meaningful to the task.

2. **Elect Sign**: For each parameter position, examine the signs of all surviving task vector entries. Take a **majority vote** to determine the elected sign. Zero out any task vector entry that disagrees with the majority. This resolves conflicts where task A wants to increase a parameter and task B wants to decrease it.

3. **Merge**: Average the remaining non-zero entries (which now agree in sign) and add to the base model: $W_{\text{merged}} = W_{\text{base}} + \sum_t \lambda_t \cdot \text{TIES}(\tau_t)$.

The $\lambda_t$ coefficients control each task's influence and are typically tuned on a validation set.

---

### Q8 (Intermediate): What is Self-RAG and how does it reduce hallucination?

**Answer.** Self-RAG is a framework where the language model itself decides when to retrieve external information and evaluates the quality of what was retrieved, using special reflection tokens. When generating, the model can output `[Retrieve=Yes]` to trigger retrieval. After receiving documents, it outputs `[IsRel]` to judge relevance and `[IsSup]` to judge whether its answer is supported by the retrieved text. If a document is deemed irrelevant, it is skipped.

This reduces hallucination because: (1) the model only retrieves when it recognizes uncertainty (avoiding contamination from unnecessary retrieval), and (2) it explicitly evaluates whether its claims are supported by evidence. In standard RAG, irrelevant retrieved text can actually increase hallucination by providing a misleading context that the model tries to reconcile.

---

### Q9 (Intermediate): What is KTO and why is it useful when you don't have paired preferences?

**Answer.** KTO (Kahneman-Tversky Optimization) is an alignment method that works with unpaired binary feedback — each response is independently labeled as "good" or "bad" without needing a paired comparison. The loss function is inspired by prospect theory: for good responses, it encourages the policy to assign higher probability relative to the reference; for bad responses, it encourages lower probability. The losses are weighted asymmetrically ($\lambda_l > \lambda_w$) reflecting loss aversion.

This is useful because paired preference data is expensive — you need two responses to the same prompt, with one clearly better. In practice, production systems often have thumbs-up/thumbs-down feedback on individual responses, which is exactly what KTO can use. This makes KTO much more data-efficient in real-world deployment scenarios.

---

### Q10 (Advanced): How would you design an MoE architecture for a Binance multi-task NLP system?

**Answer.** I would design a DeepSeek-MoE style architecture with shared and routed experts:

- **Shared experts (2-3)**: Process every token, providing common crypto/financial language understanding. These encode the broad domain knowledge that all tasks need.
- **Routed experts (16-32)**: Specialized for different sub-tasks. Top-4 routing per token.
- **Expert granularity**: Use smaller experts (e.g., FFN hidden dim 1024 instead of 4096) but more of them, for finer specialization.
- **Router**: Learned linear router with load balancing loss ($\alpha = 0.01$) and router z-loss for stability.
- **Training**: CPT on crypto corpus first (as a dense model on the shared backbone), then convert FFN layers to MoE and continue training. This gives the shared experts a head start.
- **Serving**: Expert parallelism across 4 GPUs (8 experts per GPU). Use vLLM's MoE support for efficient batched inference.
- **Fine-tuning**: Apply LoRA to the shared experts and the router. For task-specific LoRA, merge using TIES.

The total parameter count might be 14B (with 3B active per token), giving us the capacity of a much larger model at the cost of a 3B dense model.

---

### Q11 (Advanced): Compare GRPO, PPO, and DPO for training a reasoning model. Which would you choose and why?

**Answer.** For a reasoning model (like DeepSeek-R1):

- **PPO**: Most flexible. Can use arbitrary reward functions, including programmatic rewards (e.g., code execution correctness). But requires 4 models in memory and is notoriously difficult to tune (learning rate, KL coefficient, value function clipping, GAE lambda, etc.).

- **DPO**: Simplest to implement. But it requires *offline* preference pairs and works best with binary preferences. For reasoning, you'd need pairs of (correct reasoning trace, incorrect reasoning trace), which are hard to collect at scale. Also, DPO cannot generate its own training data — it operates on a fixed dataset.

- **GRPO**: The best fit for reasoning. It generates its own data online, works with scalar rewards (not just binary), and eliminates the value model. For math/code reasoning, you can use *verifiable* rewards (check if the answer is correct). The group normalization naturally identifies which reasoning strategies work better.

I would choose GRPO because: (1) reasoning correctness is verifiable (ground truth available), making reward model training straightforward; (2) online generation means the model explores diverse reasoning strategies; (3) no value model means lower memory and simpler training. The main cost is generating $G$ responses per prompt, but this is amortized over the quality gains.

---

### Q12 (Advanced): How would you build a hybrid retrieval system using SPLADE, dense retrieval, and BM25 for a crypto knowledge base?

**Answer.** Architecture:

1. **Indexing**: For each document chunk, compute three representations:
   - BM25 index (Elasticsearch/Lucene) — zero-cost, standard tokenization
   - SPLADE sparse vector (fine-tuned on crypto relevance data) — stored in an inverted index (Anserini or Vespa)
   - Dense embedding (fine-tuned bi-encoder, e.g., BGE-large) — stored in a vector DB (Milvus/Qdrant)

2. **Retrieval**: For each query, run all three retrievers in parallel (latency = max, not sum):
   - BM25: top-100
   - SPLADE: top-100
   - Dense: top-100

3. **Fusion**: Apply Reciprocal Rank Fusion: $\text{RRF}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)}$ with $k=60$. This produces a single ranked list without needing to calibrate scores across different retrieval systems.

4. **Reranking**: Take top-20 from RRF, pass through a cross-encoder reranker (fine-tuned on domain data). Output top-5.

Why hybrid? BM25 handles exact term matching (ticker symbols, contract addresses). SPLADE handles synonym expansion (query "rug pull" matches document "exit scam"). Dense handles deep semantic similarity (query about "yield farming risks" matches document about "impermanent loss in liquidity provision"). No single method covers all three.

---

### Q13 (Advanced): Explain how CRAG's corrective mechanism improves upon naive RAG and give a concrete example.

**Answer.** In naive RAG, all retrieved documents are concatenated into the context regardless of quality. CRAG adds a retrieval evaluator that classifies documents as Correct, Ambiguous, or Incorrect, and takes different actions for each.

**Concrete example:** User asks: "What were the regulatory implications of Binance's 2024 settlement?"

- Naive RAG retrieves 5 chunks. Chunks 1-3 are about the actual settlement (relevant). Chunk 4 is about a different exchange's settlement (misleading). Chunk 5 is about Binance's trading features (irrelevant). The LLM might conflate details from chunk 4, producing a hallucinated answer.

- CRAG evaluator scores: chunks 1-3 as "Correct" (high relevance), chunk 4 as "Ambiguous" (related topic but wrong entity), chunk 5 as "Incorrect." The system uses only chunks 1-3, and for the ambiguous chunk 4, it refines the query to "Binance SEC settlement 2024 terms" and re-retrieves, getting a more targeted result. Chunk 5 is discarded.

The knowledge refinement step further extracts only relevant sentences from the surviving chunks, removing tangential information within otherwise relevant documents. The result is a more focused context and a more accurate answer.

---

### Q14 (Advanced): How does DARE improve upon simple model averaging, and what is the theoretical justification?

**Answer.** Simple model averaging ($W_{\text{avg}} = \frac{1}{K} \sum W_k$) suffers from parameter interference: when task A's fine-tuning increases parameter $j$ by +0.1 and task B decreases it by -0.1, the average is 0 — both signals are lost.

DARE (Drop And REscale) randomly drops delta parameters with probability $p$, then rescales survivors by $\frac{1}{1-p}$. If task A's delta for parameter $j$ is dropped, task B's delta survives uncontested. Across many parameters, each task's signal survives in approximately $(1-p)$ of positions, with reduced interference.

The theoretical justification draws on the **dropout/sparse approximation** literature. Fine-tuning deltas are empirically over-parameterized — the task-specific information can be represented by a sparse subset of the parameter changes. DARE exploits this by stochastically selecting different subsets, analogous to how dropout creates implicit ensembles of sub-networks. The rescaling factor $\frac{1}{1-p}$ ensures that the expected magnitude of the surviving deltas matches the original, preserving the learned adaptation in expectation.

Empirically, DARE with $p = 0.9$ (dropping 90% of deltas) often outperforms simple averaging because the 10% of surviving parameters carry task-specific information with minimal cross-task interference.

---

### Q15 (Advanced): If you were building a domain-specific LLM for Binance from scratch, outline the complete training pipeline including MoE, CPT, GRPO, and RAG integration.

**Answer.** Complete pipeline:

**Phase 1 — Base Model Selection:** Start with an open-source base (e.g., Llama 3 70B) or a smaller MoE model (e.g., Mixtral 8x22B).

**Phase 2 — Continued Pretraining:** Run CPT on ~50B tokens of crypto/financial data: whitepapers, DeFi docs, Binance Academy, regulatory filings, blockchain code. Mix 80% domain / 20% general data. Learning rate: $3 \times 10^{-5}$ with cosine schedule. Optionally extend tokenizer with 1000 crypto-specific tokens.

**Phase 3 — MoE Conversion (Optional):** If starting from a dense model, convert FFN layers to MoE by duplicating the FFN into $N$ experts and adding a randomly initialized router. Continue training for 5-10B tokens to let the router learn specialization. Add 2 shared experts initialized from the original FFN.

**Phase 4 — Supervised Fine-Tuning:** SFT on ~100K high-quality instruction-response pairs covering: crypto Q&A, sentiment analysis, NER, regulatory queries, tool use (API calls to Binance), multi-turn conversation. Use chat template formatting.

**Phase 5 — GRPO Alignment:** Train a reward model on human preferences (crypto expert annotators). Then run GRPO with $G=32$, using both the reward model score and format adherence as reward components. Include a KL penalty ($\beta = 0.1$) to prevent divergence from the SFT model.

**Phase 6 — RAG Integration:** Deploy with a hybrid retrieval system: BM25 + SPLADE + dense retriever on Binance's knowledge base. Implement CRAG-style evaluation to filter irrelevant retrievals. Use community-level Graph RAG for complex multi-hop questions about the crypto ecosystem.

**Phase 7 — Task Adaptation:** Fine-tune LoRA adapters for specific tasks (sentiment, NER, classification). Merge using TIES-Merging for the production multi-task endpoint.

**Phase 8 — Evaluation:** Benchmark on domain-specific tasks (crypto QA accuracy, NER F1, sentiment accuracy), general benchmarks (MMLU, HellaSwag to check for catastrophic forgetting), and safety evaluations (toxicity, refusal of financial advice without disclaimer).

This pipeline produces a domain-expert LLM that is efficient (MoE sparsity), knowledgeable (CPT), well-aligned (GRPO), factually grounded (RAG), and multi-capable (merged adapters).

---

*End of Modern Techniques Guide*
