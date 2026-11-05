# Model Architectures II: Beyond the Standard Transformer

The decoder-only transformer, with its modern additions (pre-norm/RMSNorm, RoPE, GQA, MLA, MoE), is covered in the sibling chapter [Model Architectures](architectures.md). This chapter covers what sits next to it: sub-quadratic sequence mixers (linear attention, RetNet, RWKV, GLA, DeltaNet, state space models), hybrids that mix them with attention, sparse attention, the encoders that still matter (ViT, CLIP/SigLIP, Whisper), multimodal wiring, diffusion models, and what reasoning models change at the systems level.

The aim is to explain why each design exists and what it costs to train and serve: state size per token, what can run in parallel during training, how inference scales with sequence length, and where each one is weak. The goal is not a full list of papers.

Conventions:

- `T` = sequence length, `d` = model width, `L` = number of layers, `N` = parameter count, `n` = SSM state dimension, `H` = heads, `dk`/`dv` = per-head key/value dims.
- Byte counts assume bf16 (2 bytes) unless stated. `GiB` = 2^30 bytes.
- Paper citations give arXiv ids. Ids I could not check while writing are marked "(id unverified)". Claims are attributed to the paper that made them. Where I am unsure of a detail (exact layer counts, ratios, release dates), I say so.
- Anything about kernel or serving-engine support describes roughly late 2025. This area moves monthly, so check the current release notes before you rely on it.

All the PyTorch snippets in this chapter were run with torch 2.6 on CPU. The equivalence checks they claim (recurrent vs parallel form, and so on) pass to fp32 tolerance. They are naive reference code, not fast kernels.


## Why look beyond attention

Softmax attention has two separate scaling problems. Most arguments for alternatives mix them up, so take them one at a time.

### Problem 1: training and prefill compute is O(T^2)

For causal attention, the forward QK^T and AV matmuls cost about `4 * t * d` FLOPs per layer for the token at position `t`. Averaged over a sequence, that is `2 * T * d` per token per layer. The dense (weight) FLOPs are about `2N` per token. So:

```
attention FLOPs / dense FLOPs  ~=  (2 * T * d * L) / (2 * N)  =  T * d * L / N
```

| model shape | T = 8k | T = 128k | T = 1M |
|---|---|---|---|
| Llama-3-8B-like (d=4096, L=32, N=8e9) | 0.13 | 2.1 | 17.2 |
| Llama-3-70B-like (d=8192, L=80, N=70e9) | 0.08 | 1.2 | 9.8 |

At 8k, attention is a rounding error compared to the MLPs. At 128k it is already 1-2x the whole dense compute. At 1M it is 10-17x. Pretraining usually runs at 4k-8k and extends context later, which is why the O(T^2) term rarely shows up in pretraining budgets. It does show up in long-context extension, in long prefill at inference, and in RL on long generations.

FlashAttention does not change this. It makes attention IO-efficient: the T x T matrix is never materialized, so memory is O(T). The FLOPs are still O(T^2).

### Problem 2: decode memory grows as O(T) per sequence

At decode time, every new token reads the whole KV cache. Per token, the cache holds `2 (K and V) * L * n_kv_heads * head_dim * bytes`:

| model (GQA, 8 KV heads, head_dim 128) | KV / token | at 128k | at 1M |
|---|---|---|---|
| Llama-3-8B (L=32) | 128 KiB | 16 GiB | 128 GiB |
| Llama-3-70B (L=80) | 320 KiB | 40 GiB | 320 GiB |
| Llama-3-405B (L=126) | 504 KiB | 63 GiB | 504 GiB |

These numbers are per sequence. At 128k, one 70B request needs about as much KV memory as the bf16 weights of a 20B model. Decode is memory-bandwidth bound, so the time per token grows linearly with how much KV has to be read. Batch size, which is where serving gets its throughput, is capped by `(HBM - weights) / (KV per sequence)`.

GQA, MQA, MLA, and KV quantization shrink the constant (see [architectures.md](architectures.md)). They do not change the O(T) growth. Every architecture in the first half of this chapter tries, one way or another, to replace that growing cache with a fixed-size state. Fixed state means O(1) memory per token and O(T) total compute. The open question in this whole area is what you give up for that. The short answer is exact recall.


## Linear attention and the RNN view

### Linear attention (Katharopoulos et al.)

"Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention", Katharopoulos et al., arXiv 2006.16236.

Softmax attention for query `t` is

```
o_t = sum_{s<=t} exp(q_t . k_s) v_s / sum_{s<=t} exp(q_t . k_s)
```

Replace the exponential kernel `exp(q . k)` with any kernel that factorizes, `sim(q, k) = phi(q) . phi(k)`, and use associativity:

```
o_t = phi(q_t)^T [ sum_{s<=t} phi(k_s) v_s^T ]  /  phi(q_t)^T [ sum_{s<=t} phi(k_s) ]
    = phi(q_t)^T S_t / phi(q_t)^T z_t

S_t = S_{t-1} + phi(k_t) v_t^T        # (dk, dv) matrix-valued state
z_t = z_{t-1} + phi(k_t)              # (dk,)   normalizer
```

That is an RNN whose hidden state is a `dk x dv` matrix. The paper used `phi(x) = elu(x) + 1`, which keeps the similarities positive.

```python
import torch, torch.nn.functional as F

def phi(x): return F.elu(x) + 1

def linear_attn_parallel(q, k, v):
    # q, k: (B, T, D)  v: (B, T, Dv)   -- training form: O(T^2 D), matmul-friendly
    q, k = phi(q), phi(k)
    scores = q @ k.transpose(1, 2)                      # (B, T, T)
    scores = scores * torch.tril(torch.ones_like(scores))  # causal mask, no softmax
    return (scores @ v) / scores.sum(-1, keepdim=True)  # (B, T, Dv)

def linear_attn_recurrent(q, k, v):
    # inference form: O(T D Dv) total, O(D Dv) state, O(1) per token
    B, T, D = q.shape
    q, k = phi(q), phi(k)
    S = q.new_zeros(B, D, v.shape[-1])                  # (B, D, Dv)  the entire "KV cache"
    z = q.new_zeros(B, D)                               # (B, D)
    out = []
    for t in range(T):
        S = S + k[:, t, :, None] * v[:, t, None, :]     # += k_t v_t^T
        z = z + k[:, t]
        num = (q[:, t, :, None] * S).sum(1)             # (B, Dv)
        den = (q[:, t] * z).sum(-1, keepdim=True)       # (B, 1)
        out.append(num / den)
    return torch.stack(out, 1)                          # (B, T, Dv)

# max |parallel - recurrent| = 1.2e-7 on (2, 16, 8) inputs
```

Systems consequences:

- **Inference:** the state is `H * dk * dv` per layer no matter what `T` is. With 32 heads of 128 x 128 that is 1 MiB per layer in bf16. A GQA-8 layer with head_dim 128 stores 4 KiB of KV per token, so the recurrent state equals about 256 tokens of KV. Past a few thousand tokens of context, the state is much smaller than a KV cache would be.
- **Training:** there are three equivalent ways to compute it. The *parallel* form is quadratic but uses tensor cores. The *recurrent* form is linear but sequential and runs at memory-bandwidth speed. The *chunkwise* form is the one used in practice: split T into chunks of size C (64-256), use the quadratic form inside each chunk (dense matmul), and pass the state between chunks recurrently. The cost is `O(T * C * d + T * dk * dv)`. With tensor cores this beats FlashAttention in wall-clock somewhere around 2k-8k tokens, depending on the kernel and head dims. That crossover is a rough figure, not a measurement.

**Why plain linear attention lost.** Perplexity lagged softmax attention noticeably at equal size. Two reasons explain most of it:

1. **No forgetting.** `S_t` is a running sum. Nothing is ever removed, so old associations pile up and interfere with new ones.
2. **Low-rank, low-capacity memory.** A `dk x dv` matrix can store about `dk` orthogonal key-value pairs exactly. Beyond that, retrieval `S^T q` returns a blend of values. Softmax's sharp, high-dynamic-range weighting is what lets attention pick one token out of 100k. A positive feature map with no normalization trick cannot do that.

Everything below (RetNet, GLA, Mamba-2, DeltaNet) fixes (1) with decay or gating, (2) with a better write rule, or both. A related line of work approximates the softmax kernel with random features: Performer / FAVOR+, Choromanski et al., arXiv 2009.14794. It is no longer competitive for LMs and is mentioned here only for history.

### RetNet

"Retentive Network: A Successor to Transformer for Large Language Models", Sun et al., arXiv 2307.08621.

RetNet is linear attention with no feature map and a fixed per-head exponential decay `gamma`, plus xPos-style rotation on q and k:

```
S_t = gamma * S_{t-1} + k_t^T v_t           # per head, gamma in (0,1) fixed per head
o_t = q_t S_t
parallel form:  O = (Q K^T  ⊙  D) V,  D[t,s] = gamma^(t-s) if t>=s else 0
```

The paper presents the same three modes as above (parallel, recurrent, chunkwise) and uses a multi-scale design where different heads get different gammas. Its group norm on head outputs takes over the job the softmax denominator used to do.

The claims were O(1) inference, parity with transformers at a few billion parameters, and better throughput and latency. Later independent comparisons, including the GLA and Mamba-2 papers' baselines, generally put RetNet behind data-dependent gating at equal compute. That does not mean the paper's own numbers were wrong. The weakness is that a *fixed* decay is a fixed-horizon memory: the model cannot choose to keep a particular token longer.

### RWKV

"RWKV: Reinventing RNNs for the Transformer Era", Peng et al., arXiv 2305.13048. Later versions: "Eagle and Finch" (RWKV-5/6), arXiv 2404.05892. RWKV-7 "Goose", arXiv 2503.14456.

The RWKV (Eagle/Finch) block - time-mixing and channel-mixing:

![RWKV architecture](images/rwkv-architecture.png)

([source](https://arxiv.org/abs/2404.05892))

RWKV is a community project (it is a Linux Foundation project, as far as I know) that has gone through several distinct architectures under one name:

- **RWKV-4:** vector-valued state with a per-channel time decay `w` and a "bonus" `u` for the current token. The WKV operator is an exponentially decayed softmax-like average:

  ```
  wkv_t = ( sum_{i<t} exp(-(t-1-i) w + k_i) v_i + exp(u + k_t) v_t )
        / ( sum_{i<t} exp(-(t-1-i) w + k_i)     + exp(u + k_t)     )
  ```

  It also uses "token shift" (mixing `x_t` with `x_{t-1}` before projections), which acts like a width-2 causal conv. The state is O(d) per layer, which is tiny.
- **RWKV-5 (Eagle):** matrix-valued multi-head state (`dk x dv` per head), similar to linear attention with decay.
- **RWKV-6 (Finch):** data-dependent decay. `w_t` is a function of the input, which places it in the same family as GLA and Mamba.
- **RWKV-7 (Goose):** a generalized delta rule with vector-valued in-context learning rates (see DeltaNet below). The paper argues this lets it represent things that diagonal-decay models provably cannot, such as certain state-tracking tasks. Treat that as the authors' claim.

Who uses it: RWKV's own releases, plus a fair amount of interest from edge and on-device deployments, because the state really is constant and small. I'm not aware of a frontier lab shipping a pure RWKV model.

### Gated linear attention (GLA)

"Gated Linear Attention Transformers with Hardware-Efficient Training", Yang, Wang, Shen, Panda, Kim, arXiv 2312.06635.

```
S_t = diag(alpha_t) S_{t-1} + k_t^T v_t     # alpha_t = sigmoid(W x_t)^(1/tau), per key-dim, data-dependent
o_t = q_t S_t
```

The decay is now input-dependent: the model learns when to forget. The paper's real contribution is as much systems as modeling. It is a chunkwise algorithm (FlashLinearAttention) that keeps the gating inside matmuls by factoring cumulative products of `alpha` into the q and k sides within a chunk, with a secondary level of chunking for numerical stability. It reported training throughput competitive with FlashAttention-2 at 2k context and faster beyond that. Kernels live in the `flash-linear-attention` (fla-org) Triton library, which has become the reference implementation for most of this family.

### DeltaNet and Gated DeltaNet

- "Linear Transformers Are Secretly Fast Weight Programmers", Schlag, Irie, Schmidhuber, arXiv 2102.11174 (delta rule in linear attention).
- "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", Yang, Wang, Zhang, Shen, Kim, arXiv 2406.06484.
- "Gated Delta Networks: Improving Mamba2 with Delta Rule", Yang, Kautz, Hatamizadeh, arXiv 2412.06464.

Additive linear attention writes `S += v k^T` whether or not `S` already holds something for `k`. The delta rule writes the *error* instead:

```
v_old = S_{t-1} k_t                              # what memory currently returns for k_t
S_t   = S_{t-1} + beta_t (v_t - v_old) k_t^T     # replace, don't add
      = S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T

Gated DeltaNet:
S_t   = alpha_t S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T     # alpha_t: Mamba-2 style scalar decay
```

This is one step of online gradient descent on `||S k_t - v_t||^2` with learning rate `beta_t`. With `beta = 1` and unit-norm `k`, it overwrites exactly:

```python
def delta_rule_recurrent(q, k, v, beta):
    # q, k: (B, T, D) L2-normalized   v: (B, T, Dv)   beta: (B, T) in (0, 1]
    B, T, D = q.shape
    S = q.new_zeros(B, v.shape[-1], D)                      # (B, Dv, D)
    out = []
    for t in range(T):
        kt, vt, bt = k[:, t], v[:, t], beta[:, t, None]
        v_old = (S @ kt[..., None]).squeeze(-1)             # (B, Dv)
        S = S + (bt * (vt - v_old))[..., None] * kt[:, None, :]   # rank-1 correction
        out.append((S @ q[:, t, :, None]).squeeze(-1))      # (B, Dv)
    return torch.stack(out, 1)                              # (B, T, Dv)

# Query with the key just written (q_t = k_t), beta = 1, T=16, D=8:
#   delta rule:                max |o_t - v_t| = 6.6e-7   (exact recall of the latest write)
#   additive linear attention: max |o_t - v_t| = 3.4      (interference from the other 15 keys)
```

That toy compares one rule against another on the same inputs. It shows the mechanism, not a quality gap between trained models.

The training problem is that the transition `(I - beta k k^T)` is a dense matrix, not a diagonal, so the GLA trick does not apply directly. Yang et al. (2406.06484) parallelize it over the sequence using the WY representation for products of Householder-like matrices. That gives a chunkwise algorithm whose cost is still dominated by matmuls. Gated DeltaNet adds the scalar decay `alpha_t`, which combines Mamba-2's fast forgetting with the delta rule's precise overwriting. The paper reports it beating Mamba-2 and DeltaNet on LM and recall-heavy benchmarks at 1.3B scale.

Who uses it: Qwen3-Next uses Gated DeltaNet layers in a hybrid (see below). Kimi Linear (Moonshot, late 2025) uses a refined variant called Kimi Delta Attention. Its paper is "Kimi Linear: An Expressive, Efficient Attention Architecture", arXiv 2510.26692 (id unverified).

### The unifying view

Most of this family can be written as one recurrence on a matrix state, which differs only in the transition:

| model | state update `S_t = ...` | transition type |
|---|---|---|
| linear attention | `S + v k^T` | identity |
| RetNet | `gamma S + v k^T` | constant scalar |
| Mamba-2 (SSD) | `a_t S + v k^T` (with B as k, C as q) | data-dependent scalar |
| GLA / RWKV-6 | `S diag(alpha_t) + v k^T` | data-dependent diagonal |
| DeltaNet | `S (I - b k k^T) + b v k^T` | identity minus rank-1 |
| Gated DeltaNet / RWKV-7 | `a_t S (I - b k k^T) + b v k^T` (RWKV-7 generalizes this) | scalar/diag times (I - rank-1) |

Readout is always `o_t = S_t q_t`. Richer transitions buy more expressivity and better recall, at the price of more complicated chunkwise kernels. A scalar transition is the easiest to make fast. For a fuller version of this table, see the Gated DeltaNet paper and Songlin Yang's write-ups.


## State space models

### S4: structured state spaces

- "HiPPO: Recurrent Memory with Optimal Polynomial Projections", Gu, Dao, Ermon, Rudra, Ré, arXiv 2008.07669.
- "Efficiently Modeling Long Sequences with Structured State Spaces" (S4), Gu, Goel, Ré, arXiv 2111.00396.
- "On the Parameterization and Initialization of Diagonal State Space Models" (S4D), Gu et al., arXiv 2206.11893.

S4 comes from signal processing, not attention. It starts from a continuous-time linear system, applied independently to each of the `d` channels:

```
h'(t) = A h(t) + B u(t)          # h in R^n (state), u scalar input, A (n x n), B (n x 1)
y(t)  = C h(t) + D u(t)          # C (1 x n)
```

**HiPPO.** A specific choice of `A` (HiPPO-LegS) makes `h(t)` the coefficients of the optimal projection of the input history onto Legendre polynomials. In other words, the state is a provably good compression of the past. That initialization was the key to S4 working on very long sequences at all.

**Discretization.** For step size `Delta` (zero-order hold):

```
A_bar = exp(Delta A)
B_bar = (Delta A)^(-1) (exp(Delta A) - I) Delta B
h_t   = A_bar h_{t-1} + B_bar u_t
y_t   = C h_t + D u_t
```

**Convolution view.** `A_bar`, `B_bar`, and `C` do not depend on the input (the system is linear time-invariant, LTI). So you can unroll the recurrence into one long convolution:

```
K = (C B_bar, C A_bar B_bar, C A_bar^2 B_bar, ..., C A_bar^(T-1) B_bar)    # length-T kernel per channel
y = K * u                                                              # causal conv, via FFT
```

Training uses the convolution: FFT in `O(T log T)`, fully parallel, and the state is never materialized. Inference uses the recurrence: O(n) state per channel and O(1) per token. Most of S4's complexity, the normal-plus-low-rank (NPLR) parameterization and Cauchy kernels, exists to compute `K` efficiently for the HiPPO matrix. S4D later showed that a diagonal `A` with a suitable init works about as well, which made the whole thing much simpler.

**Evidence.** S4 was the first model to solve Path-X (length 16k) in the Long Range Arena benchmark, and it was strong on audio and other continuous signals. On language it was clearly behind transformers.

**The weakness that matters:** LTI means the same kernel `K` is applied to every input. The model cannot decide, based on content, to remember this token and ignore that one. Content-based selection is exactly what language needs, and it is also exactly what the convolution trick forbids. H3 (Fu et al., "Hungry Hungry Hippos", arXiv 2212.14052) and Hyena (Poli et al., arXiv 2302.10866) added multiplicative gating around long convolutions to recover some of this. They narrowed the gap but did not close it.

### Mamba: selective state spaces

"Mamba: Linear-Time Sequence Modeling with Selective State Spaces", Gu and Dao, arXiv 2312.00752.

The selective SSM, with the hardware-aware scan keeping the expanded state in SRAM:

![Mamba selective SSM](images/mamba-selection.png)

([source](https://github.com/state-spaces/mamba))

The change is to make `Delta`, `B`, and `C` functions of the input:

```
Delta_t = softplus(W_Delta x_t + b)     # (d,)   per-channel step size
B_t     = W_B x_t                       # (n,)   shared across channels
C_t     = W_C x_t                       # (n,)
A       = -exp(A_log)                   # (d, n) diagonal, input-independent, negative real

h_t = exp(Delta_t A) ⊙ h_{t-1} + (Delta_t B_t) x_t      # per channel: (n,) state
y_t = C_t . h_t + D x_t
```

`Delta_t` acts as a gate. A large `Delta_t` resets the state toward the current input ("focus on this token"). A small `Delta_t` leaves the state almost unchanged ("ignore this token"). The model is now time-varying, so the convolution view is gone. Training has to compute the recurrence, which takes a scan.

```python
def selective_scan(x, delta, A, B, C, D):
    # x:     (b, T, d)   input (after in_proj + causal conv + SiLU)
    # delta: (b, T, d)   softplus'd, input-dependent step
    # A:     (d, n)      negative real, diagonal per channel
    # B, C:  (b, T, n)   input-dependent
    # D:     (d,)        skip connection
    b, T, d = x.shape
    n = A.shape[1]
    dA  = torch.exp(delta[..., None] * A)                      # (b, T, d, n)  ZOH on A
    dBx = delta[..., None] * B[:, :, None, :] * x[..., None]   # (b, T, d, n)  Euler on B (as in the ref impl)
    h = x.new_zeros(b, d, n)                                   # (b, d, n)  the entire recurrent state
    ys = []
    for t in range(T):
        h = dA[:, t] * h + dBx[:, t]                           # (b, d, n)
        ys.append((h * C[:, t, None, :]).sum(-1))              # (b, d)
    return torch.stack(ys, 1) + x * D                          # (b, T, d)
```

Note that this naive version materializes `(b, T, d, n)` tensors. That is exactly the expansion by `n` that makes a naive implementation memory-bound and slow.

**Hardware-aware scan.** The Mamba kernel never writes `(b, T, d, n)` to HBM. It loads `Delta`, `A`, `B`, `C`, and `x` (all O(b T d) or O(b T n)) into SRAM, does the discretization and the scan there, and writes only `y` (O(b T d)) back. For backward it recomputes the intermediate states instead of storing them, the same trick FlashAttention uses. Within a block it uses a parallel (Blelloch-style) associative scan, so the recurrence `h_t = a_t h_{t-1} + b_t` gets O(log T) depth. The kernel is still elementwise work, not matmul, so it runs on CUDA cores rather than tensor cores. That ceiling is what Mamba-2 removes.

**Block structure.** A Mamba block is `in_proj (d -> 2 * E d)`, then one branch goes through a causal depthwise conv1d (width 4), SiLU, and the SSM, gated by `SiLU(z)` from the other branch, then `out_proj`. With expansion `E = 2`, Mamba-1 models stack these blocks with no separate MLP, so the block does both mixing and channel MLP work.

**State size, concretely.** Mamba-2.8B has `d = 2560`, `E = 2`, `n = 16`, and 64 layers. Per layer the state is `5120 * 16` SSM elements plus `5120 * 3` conv-cache elements. Across 64 layers that is about 25 MB in fp32, constant in `T`. A 2.7B-class MHA transformer (32 layers, d=2560) uses 320 KiB of KV per token, so Mamba's entire state equals **about 76 tokens** of KV cache.

**Evidence (as reported in the paper):** Mamba-3B matched or beat transformers twice its size on standard zero-shot evaluations. It solved selective copying and induction-head synthetic tasks, and extrapolated on induction heads to 1M tokens. It also reported about 5x higher generation throughput than a similarly sized transformer.

### Mamba-2 and state space duality (SSD)

"Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality", Dao and Gu, arXiv 2405.21060.

An SSM written as one (semiseparable) matrix acting on the sequence:

![SSMs as semiseparable matrices](images/ssm-semiseparable-matrix.png)

([source](https://arxiv.org/abs/2405.21060))

The chunked SSD algorithm - attention-like matmuls within chunks, a recurrence across them:

![SSD chunked algorithm](images/ssd-algorithm.png)

([source](https://arxiv.org/abs/2405.21060))

Mamba-2 restricts `A` to a **scalar times identity per head** (`a_t` is one number per head per step) and shares `B_t` and `C_t` across a group of channels, the same way GQA shares K and V. With that restriction, the whole sequence map unrolls into a matrix:

```
y = M x,   M[t, s] = C_t^T B_s * prod_{r=s+1..t} a_r    for t >= s, else 0
          = (L ⊙ C B^T)[t, s]                                  L = 1-semiseparable decay mask
```

That is masked linear attention: `C` plays the role of Q, `B` of K, `x` of V, and the mask `L` holds cumulative decays instead of 1s. This is the "duality". The same layer can be computed as a linear-time recurrence (the SSM view) or as a quadratic masked matmul (the attention view).

```python
def ssd_quadratic(x, delta, A, B, C, D):
    # Same math as selective_scan above, attention-style. O(T^2) memory -- reference only.
    dA_log = delta[..., None] * A                                # (b, T, d, n)
    cum = dA_log.cumsum(1)                                       # (b, T, d, n)
    seg = cum[:, :, None] - cum[:, None, :]                      # (b, T, T, d, n)  log-decay s->t
    causal = torch.tril(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool))
    Lmask = torch.exp(seg.masked_fill(~causal[None, :, :, None, None], float('-inf')))
    dBx = delta[..., None] * B[:, :, None, :] * x[..., None]     # (b, T, d, n)
    return torch.einsum('btn,btsdn,bsdn->btd', C, Lmask, dBx) + x * D   # (b, T, d)

# max |ssd_quadratic - selective_scan| = 6.7e-6 on b=2, T=32, d=16, n=4
```

(This check keeps Mamba-1's per-channel `A` for simplicity. Mamba-2's scalar `A` is a special case of it.) Note the mask-before-exp: computing `exp` on the upper triangle and then masking overflows to `inf * 0 = nan`. Real kernels make this mistake too.

**The SSD algorithm** is the chunkwise algorithm from the linear attention section. Split T into chunks of size Q (for example 64-256). Within a chunk, use the quadratic form (matmuls on tensor cores). Across chunks, pass the `(n x headdim)` state with a short recurrence. Because everything is now matmul, Mamba-2 can afford a much larger state (`n = 64-256` instead of 16). The paper reports the SSD kernel being 2-8x faster than Mamba-1's scan, and faster than FlashAttention-2 beyond about 2k tokens.

At `n = 128`, a Mamba-2 2.7B-shaped model (64 layers, d_inner 5120) has about 84 MB of state in bf16, equal to **about 256 tokens** of a comparable MHA model's KV cache. That is still constant in T.

**Systems details that matter:**

- **Tensor parallelism:** Mamba-2 computes `x, z, B, C, Delta` from one parallel input projection at the start of the block, instead of Mamba-1's sequential `x -> (B, C, Delta)` dependency. That makes Megatron-style TP need one all-reduce per block, as an MLP does. It also adds a norm before `out_proj` for stability.
- **Sequence/context parallelism:** you pass the state between devices (a "state passing" step) instead of ring-exchanging KV blocks. The communication is O(state), not O(T).
- **Decode:** O(1) per token, state updated in place. For serving, that means one fixed-size slot per sequence. Prefix caching only works if the state was snapshotted at exactly the prefix boundary, and speculative decoding needs state checkpointing so it can roll back rejected tokens.

### Where SSMs and linear RNNs are weak: recall and copying

This is the most important caveat in the chapter.

- "Zoology: Measuring and Improving Recall in Efficient Language Models", Arora et al., arXiv 2312.04927. Introduces multi-query associative recall (MQAR). The authors report that most of the perplexity gap between gated-convolution models and attention comes from recall of previously seen tokens, and that the gap tracks state size.
- "Simple linear attention language models balance the recall-throughput tradeoff" (Based), Arora et al., arXiv 2402.18668. Makes the recall vs. state-size tradeoff explicit: a fixed-state model can only recall as much as its state can hold.
- "Repeat After Me: Transformers are Better than State Space Models at Copying", Jelassi, Brandfonbrener, Kakade, Malach, arXiv 2402.01032. Proves that a two-layer transformer can copy strings exponentially long in its size, while a fixed-state model cannot copy strings longer than its state allows. Experiments show that Mamba and similar models need far more training to copy, and fail to generalize on copy and phone-book lookup.
- "An Empirical Study of Mamba-based Language Models", Waleffe et al. (NVIDIA), arXiv 2406.07887. 8B Mamba, Mamba-2, and transformer models trained on the same data, up to 3.5T tokens. Pure Mamba-2 matched or beat the transformer on many standard tasks, but lagged on 5-shot MMLU and on phone-book-style retrieval and long-context in-context learning. A hybrid of about 43% Mamba-2, 7% attention, and 50% MLP layers beat the transformer on average across their 12 standard tasks, by about +2.65 points as I recall the abstract. They projected up to 8x faster generation at long context.

The intuition: attention keeps every token, so its memory grows with T and it can look anything up. A fixed-state model must decide at write time what to keep, so it cannot answer a question about a detail it chose to drop. That is a fundamental information limit, not an optimization issue. The delta rule and larger states push the limit out but don't remove it. This is why the field has largely converged on **hybrids**.

Other recurrent designs in the same space, for completeness:

- Griffin / Hawk (De et al., arXiv 2402.19427): RG-LRU recurrence mixed with local attention. Shipped as RecurrentGemma.
- xLSTM (Beck et al., arXiv 2405.04517): sLSTM and mLSTM. mLSTM is a matrix-state, linear-attention-like cell.


## Hybrids: a few attention layers plus many linear ones

The reasoning: put most layers in an O(1)-state mixer for cheap long-context decode, and keep a few full-attention layers to provide exact retrieval. KV memory then scales with the number of *attention* layers. A model with 1 attention layer in 8 carries 1/8 the KV cache of a pure transformer of the same shape. For a 70B-shaped model at 1M context, that is 40 GiB instead of 320 GiB, plus a small constant for the recurrent states.

### Jamba (AI21)

"Jamba: A Hybrid Transformer-Mamba Language Model", Lieber et al., arXiv 2403.19887.

- A 1:7 ratio of attention to Mamba (Mamba-1) layers, with MoE replacing the MLP in every other layer (16 experts, top-2). 52B total parameters, 12B active.
- The paper's own table puts KV cache at 256k context at about 4 GB for Jamba, versus 32 GB for Mixtral and 128 GB for Llama-2-70B. Check: 4 attention layers x 8 KV heads x head_dim 128 x 2 x 2 bytes = 16 KiB/token, and x 262,144 tokens = 4 GiB. That matches.
- Reported findings: no positional encoding was needed in attention layers, because the Mamba layers supply position implicitly. Adding RMSNorm inside Mamba layers helped stability at scale. Pure Mamba struggled with in-context format following, which the attention layers fixed.
- Follow-up: Jamba-1.5 (arXiv 2408.12570, id unverified) at 398B total / 94B active.

### Zamba (Zyphra)

"Zamba: A Compact 7B SSM Hybrid Model", Glorioso et al., arXiv 2405.16712.

Mamba backbone plus a **single shared** attention+MLP block, applied every 6 Mamba blocks. The same weights are reused at each application, and the original embedding is concatenated in. The point is parameter efficiency: you get repeated access to attention without paying for its parameters several times. You still pay KV cache for every place the shared block is applied. Zamba2 refined this with two alternating shared blocks and LoRA per application. I'm less sure of the exact Zamba2 details.

### Samba (Microsoft)

"Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling", Ren et al., arXiv 2406.07522.

Layer-wise interleaving of Mamba, MLP, **sliding-window attention** (window 2048), and MLP. There is no global attention, so total state is bounded: Mamba state plus a 2048-token window. The 3.8B model was trained at 4k context. The paper reports perplexity extrapolation far beyond the training length and perfect passkey retrieval to 256k after short fine-tuning. Passkey retrieval is a weak test of recall, and bounded state still limits arbitrary lookups.

### NVIDIA Nemotron-H

"Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models", NVIDIA, arXiv 2504.03624.

- Mostly Mamba-2 layers with roughly 8% self-attention layers, spread evenly through the depth, plus MLPs. The figure I remember is that the 8B has 4 attention layers out of 52 and the 56B has 10 out of 118. I have not verified those layer counts.
- The paper reports accuracy on par with or better than similarly sized open transformers (Qwen-2.5, Llama-3.1), and up to about 3x faster inference at long sequence lengths.
- Notable systems work: the 56B was trained in FP8, and a 47B version was derived by compression/distillation (MiniPuzzle). The Nemotron-Nano-2 models (late 2025) use the same hybrid backbone. That is my understanding, unverified.

### MiniMax-01 (lightning attention)

"MiniMax-01: Scaling Foundation Models with Lightning Attention", MiniMax, arXiv 2501.08313.

- Lightning attention is a linear-attention variant with an IO-aware intra/inter-block kernel. The prior paper is "Lightning Attention-2", arXiv 2401.04658 (id unverified).
- Hybrid ratio: one softmax-attention layer after every seven lightning-attention layers (7:1). 456B total / 45.9B active MoE. Training context up to 1M and inference extrapolation to 4M, as claimed.
- MiniMax-M1 (arXiv 2506.13585, id unverified) is the reasoning model on the same backbone. It argued that the hybrid makes long-generation RL much cheaper, reporting about 25% of DeepSeek-R1's FLOPs at 100k generation length. That figure is their claim.
- My understanding is that MiniMax's later M2 model went back to full attention, and that MiniMax published commentary on why: efficient-attention variants were harder to evaluate and less robust on complex retrieval and agentic tasks. I'm not confident of the exact reasoning or wording, so treat it as a data point that a strong team found the tradeoff unfavorable at that time.

### Qwen3-Next and Kimi Linear (3:1 linear:full)

- **Qwen3-Next** (Alibaba Qwen, September 2025; described in a blog and model card, no arXiv paper I know of): 80B total / about 3B active, high-sparsity MoE. It mixes Gated DeltaNet layers with gated full-attention layers at roughly 3:1 (three Gated DeltaNet layers per attention layer). The attention layers add an output gate. Qwen positioned it as a preview of the Qwen3.5 architecture direction. Details are from the release materials; I have not verified them against a paper.
- **Kimi Linear** (Moonshot, October/November 2025, arXiv 2510.26692, id unverified): Kimi Delta Attention (a finer-grained gated delta rule) interleaved with MLA at 3:1. The authors report it beating full attention at equal training compute while cutting KV cache by up to about 75% and speeding up decode at 1M context. Those are the authors' claims.

### Choosing the attention ratio

What the published hybrids tell us:

| model | linear/SSM : attention | attention type |
|---|---|---|
| Jamba | 7 : 1 | full (no PE) |
| MiniMax-01 | 7 : 1 | full softmax |
| Nemotron-H | about 11 : 1 (about 8% attention) | full |
| NVIDIA 8B study (2406.07887) | about 6 : 1 among mixers (43% Mamba-2, 7% attn, rest MLP) | full |
| Qwen3-Next | 3 : 1 | full, gated |
| Kimi Linear | 3 : 1 | MLA |
| Samba | 1 : 1 | sliding window only |
| (attention-only analogue) Gemma 3 | 5 local : 1 global | SWA 1024 + full |

Reasoning about the ratio:

1. **Retrieval needs only a few attention layers, but they must exist.** The NVIDIA study and Jamba both found that a small fraction of attention layers recovers most of the in-context-learning and retrieval gap. Going from 0 to 1 attention layer in 8 matters much more than going from 1 in 8 to 1 in 4.
2. **KV cost is linear in the attention fraction.** Going from 1:7 to 1:3 doubles KV. The 3:1 models are betting that more retrieval capacity is worth 2x the KV, at a total that is still 4x below a pure transformer. The 7:1 and 11:1 models are betting on throughput.
3. **Placement:** spread attention layers evenly through the depth rather than at the start. Several papers report the first layer working badly as attention, but I'd treat that as weakly established.
4. **The FLOPs at moderate T barely change.** At 4k-32k, the MLP/MoE FLOPs dominate either way. The win is in decode memory and long-prefill compute, not short-context training throughput. If your workload is mostly 4k-context chat, a hybrid saves you little. If it is 128k+ documents, long reasoning traces, or agents with long histories, it saves you a lot.
5. **Evaluation trap:** standard short benchmarks (MMLU, HellaSwag) don't stress recall. Hybrids and pure linear models look fine on them. Test needle-in-a-haystack variants with multiple keys (RULER-style), phone-book lookup, and long-context QA before concluding that a design is "equivalent".

The same reasoning applies inside pure-attention models. Gemma 2 interleaves sliding-window (4096) and global attention 1:1, and Gemma 3 uses 5:1 with a 1024 window. For KV purposes, a sliding-window layer is a bounded-state layer: it costs `window * per-layer-KV`, constant in T.


## Long-context alternatives inside attention

These approaches keep softmax attention but restrict which (query, key) pairs get computed.

### Fixed sparse patterns: Longformer, BigBird

- "Generating Long Sequences with Sparse Transformers", Child et al., arXiv 1904.10509: strided and fixed factorized patterns, O(T sqrt(T)).
- "Longformer: The Long-Document Transformer", Beltagy, Peters, Cohan, arXiv 2004.05150: sliding window (for example w=512), plus dilated windows in some heads, plus a few task-specific **global** tokens (such as [CLS] or question tokens) that attend to and are attended by everything. O(T w).
- "Big Bird: Transformers for Longer Sequences", Zaheer et al., arXiv 2007.14062: window, global, and **random** blocks. The paper proves that this sparse pattern keeps attention a universal approximator of sequence functions and Turing complete, with the global tokens carrying much of that weight.

Longformer's global + sliding-window pattern:

![global plus sliding window attention](images/longformer-global-sliding.png)

([source](https://arxiv.org/abs/2004.05150))

These were the encoder era's (2020-2021) answer to 4k-16k documents. They were largely displaced by two developments:

1. FlashAttention made dense attention to 16k-32k cheap enough. Block-sparse patterns with irregular memory access often lost to a dense, fused kernel at those lengths.
2. Fixed patterns can't adapt to content. Which tokens matter depends on the query.

The piece that survived is **sliding-window attention (SWA)**: Mistral 7B (window 4096), and Gemma 2/3 interleaving local and global layers. SWA is the only sparse pattern that also bounds KV memory, because tokens outside the window can be evicted. A related finding from "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM, Xiao et al., arXiv 2309.17453): keeping the first few tokens as "sinks" next to the window prevents the collapse you otherwise get when the earliest tokens are evicted.

### Learned/dynamic sparsity: DeepSeek NSA and DSA

"Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention", Yuan et al. (DeepSeek), arXiv 2502.11089.

NSA's three branches - compressed, selected and sliding - combined by learned gates:

![native sparse attention](images/nsa-architecture.png)

([source](https://arxiv.org/abs/2502.11089))

Each query attends through three branches whose outputs are mixed by learned gates:

1. **Compressed:** keys and values are compressed block by block (a small MLP over each block of, say, 32 tokens) into coarse tokens. Attending to these gives a cheap global summary.
2. **Selected:** the compressed attention scores rank blocks. The top-n blocks (for example 16 blocks of 64 tokens) are loaded at full resolution and attended exactly.
3. **Sliding window:** local context, kept as its own branch so that local patterns don't take over the learned selection.

The systems design is what makes it work. Selection is shared across all query heads in a GQA group, so the kernel loads one contiguous KV block per group and reuses it for every head. That keeps tensor-core utilization high, which a per-head token-level top-k would not. The model is trained with sparsity from the start ("natively trainable"), rather than having sparsity applied to a dense model after the fact. Reported results (their claims, at 64k on A100): NSA matches or beats full attention on their benchmarks, with speedups of up to about 9x forward, 6x backward, and 11.6x decode.

**DeepSeek Sparse Attention (DSA)** in DeepSeek-V3.2-Exp (September 2025; tech report, no arXiv id I'm confident of): a small, cheap "lightning indexer" scores every previous token for each query, and the main MLA attention then attends only to the top-k tokens (k=2048, as I recall). The dense model was converted with continued training. The selection step is O(T^2), but with a very small constant. The main attention becomes O(T k).

**A key systems point about sparse attention:** NSA, DSA, and MoBA ("MoBA: Mixture of Block Attention for Long-Context LLMs", Lu et al., arXiv 2502.13189) cut **compute and HBM reads** per decode step. They do **not** cut KV **storage**, because every token stays cached in case it gets selected later. Linear and hybrid models cut both. Choose based on whether you are compute/bandwidth-bound or capacity-bound.

### Ring attention and context parallelism

When the problem is that one device can't hold the sequence (not that attention is too expensive), shard the sequence. "Ring Attention with Blockwise Transformers for Near-Infinite Context", Liu, Zaharia, Abbeel, arXiv 2310.01889, rotates KV blocks around a ring of devices and overlaps communication with blockwise attention. DeepSpeed-Ulysses instead all-to-alls from sequence sharding to head sharding. Both are exact attention, so they change the memory distribution, not the FLOPs. Causal masking leaves ring attention load-imbalanced, which striped and zigzag orderings fix. Details are in [Model Parallelism](../training/model-parallelism/README.md). For SSM and linear layers, the analogue is passing O(state) across devices, which is much cheaper.


## Encoder models that still matter

### ViT

"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", Dosovitskiy et al., arXiv 2010.11929.

ViT - patchify, linearly embed, add positions, run a standard encoder:

![Vision Transformer](images/vit-model.png)

([source](https://arxiv.org/abs/2010.11929))

Split the image into non-overlapping P x P patches, flatten each patch, apply a linear projection, add a position embedding, and run a bidirectional transformer encoder.

```python
img = torch.randn(1, 3, 224, 224)                                  # (B, C, H, W)
p = 16
patches = img.unfold(2, p, p).unfold(3, p, p)                      # (1, 3, 14, 14, 16, 16)
patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(1, 14 * 14, 3 * p * p)   # (1, 196, 768)
tokens = torch.nn.Linear(768, 768)(patches)                       # (1, 196, d) -- equivalently Conv2d(3, d, 16, stride=16)
```

Token count is `(H/P) * (W/P)`. It grows quadratically with resolution and inverse-quadratically with patch size:

| config | tokens | note |
|---|---|---|
| 224px, P=16 (ViT-B/16) | 196 (+1 CLS) | about 34 GFLOP fwd at 86M params (2 * N * T) |
| 224px, P=14 (ViT-L/14) | 256 | CLIP ViT-L/14 |
| 336px, P=14 | 576 | LLaVA-1.5's vision tower |
| 384px, P=14 | 729 | SigLIP so400m/14@384 |
| 518px, P=14 | 1369 | DINOv2 at its native high res |
| 1024px, P=16 | 4096 | SAM-style high-res encoders, where O(T^2) attention starts to matter (SAM uses windowed attention) |

ViTs need lots of data or strong augmentation to beat CNNs, because they lack locality and translation priors. At scale (JFT, LAION, WebLI) that stops mattering. Doubling the resolution multiplies T by 4 and attention FLOPs by 16, which is why VLMs tile images or merge tokens instead of feeding high-res images to a single ViT pass.

### CLIP and SigLIP: contrastive image-text training

- "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), Radford et al., arXiv 2103.00020.
- "Sigmoid Loss for Language Image Pre-Training" (SigLIP), Zhai et al., arXiv 2303.15343. SigLIP 2: arXiv 2502.14786.

A two-tower setup: an image encoder and a text encoder, each projecting into a shared embedding space. They are trained so that matched pairs score high and mismatched pairs score low.

```python
def clip_loss(img, txt, logit_scale):
    # img, txt: (B, E) L2-normalized embeddings of matched pairs
    logits = logit_scale * img @ txt.T                     # (B, B), diagonal = positives
    labels = torch.arange(img.shape[0])
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

def siglip_loss(img, txt, t, b):
    # each (i, j) is an independent binary problem: +1 on the diagonal, -1 elsewhere
    logits = t * img @ txt.T + b                           # (B, B)
    labels = 2 * torch.eye(img.shape[0]) - 1               # (B, B) in {+1, -1}
    return -F.logsigmoid(labels * logits).sum() / img.shape[0]
```

Why the loss matters for systems: CLIP's softmax normalizes over the **global** batch (32k in the paper), so every rank needs every other rank's embeddings (an all-gather), and the B x B logit matrix is spread across devices. Large batches matter for quality, because they supply more negatives. SigLIP's pairwise sigmoid has no global normalizer. The paper computes it in chunks by rotating text embeddings around devices, so memory is O(local B^2) instead of O(global B^2), and it reports better results at small batch sizes. The learnable bias `b` (initialized strongly negative, -10 in the paper) offsets the massive class imbalance of about B^2 negatives to B positives.

Where they are used: CLIP and SigLIP vision towers are the default image encoders in open VLMs (LLaVA used CLIP ViT-L/14; PaliGemma, Idefics, and many others use SigLIP). CLIP text encoders conditioned early latent diffusion models. Contrastive embeddings also power retrieval and zero-shot classification.

Text encoders (BERT-style, bidirectional, MLM-trained) are also still widely used for embeddings, rerankers, and classifiers, because they are cheap: a single forward pass with no decode loop. ModernBERT (Warner et al., arXiv 2412.13663) brought RoPE, GeGLU, alternating local/global attention, and unpadding to that class.


## Multimodal LLMs

The design question is how to get non-text signals into a language model, and how many tokens that costs.

### Adapter designs

**LLaVA (projection into the token stream).** "Visual Instruction Tuning", Liu et al., arXiv 2304.08485, and LLaVA-1.5, arXiv 2310.03744. The ViT patch features (CLIP ViT-L/14, 576 tokens at 336px) go through a linear layer (v1) or a 2-layer MLP (1.5) into the LM's embedding space and are **inserted as tokens** into the sequence. Training typically happens in two stages: first the projector only, on caption data, then projector plus LM on instruction data. It is simple, and the LM attends to image tokens with its normal attention. The cost is that every image token pays full price: it enters prefill, every layer's KV cache, and every later decode step.

**Flamingo (gated cross-attention).** "Flamingo: a Visual Language Model for Few-Shot Learning", Alayrac et al., arXiv 2204.14198. The LM stays frozen. New `GATED XATTN-DENSE` layers are interleaved between LM blocks: text queries cross-attend to vision features, and the output is scaled by `tanh(alpha)` with `alpha` initialized to 0, so at init the model is exactly the original LM. A **Perceiver resampler** compresses a variable number of vision features to a fixed 64 latent tokens per image. Image tokens never occupy the LM's self-attention context. The cost per image is a fixed-size cross-attention KV, and text length isn't consumed. In exchange you get extra parameters, an extra layer type, and weaker fine-grained interaction. Llama 3.2 Vision used cross-attention adapter layers in this spirit.

**Q-Former (BLIP-2).** "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models", Li et al., arXiv 2301.12597. A small transformer (about 188M params) with **32 learned query tokens** that cross-attend to frozen ViT features. It is trained in two stages: representation learning with contrastive, matching, and captioning losses, then generative alignment to a frozen LM. The output is 32 tokens per image, fed as soft prompts. That is very cheap in tokens, but the fixed bottleneck loses detail (OCR, charts). Open models have mostly moved back to LLaVA-style projection with token-reduction tricks.

**Native-resolution projection.** Qwen2-VL ("Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution", arXiv 2409.12191) uses dynamic resolution: 14px patches, merged 2x2 by an MLP, so one LM token covers 28 x 28 px. It also uses M-RoPE, which factors position into temporal, height, and width components. InternVL uses "pixel shuffle" to cut tokens 4x: 448px / 14 = 32 x 32 = 1024 patches, down to 256 tokens per tile.

### Early fusion: Chameleon

"Chameleon: Mixed-Modal Early-Fusion Foundation Models", Chameleon Team (Meta), arXiv 2405.09818.

Images are turned into **discrete tokens** by a VQ tokenizer: 1024 tokens per 512 x 512 image, codebook size 8192. They are interleaved with text tokens in one vocabulary, and one transformer is trained from scratch on mixed sequences with ordinary next-token prediction. The model can then *generate* images as well as read them, with no separate encoder at inference.

The costs: VQ tokenization loses information (fine text in images is a known weakness), and the paper reports substantial training-stability problems at scale. Logit drift across modalities was fixed with QK-norm, revised norm placement, and z-loss. Related: Transfusion (Zhou et al., arXiv 2408.11039) keeps images continuous and trains one transformer with a next-token loss on text and a diffusion loss on image patches.

### Image token arithmetic

Per image token, an LLaVA-style model pays roughly what a text token pays:

- **Prefill compute:** about `2N` FLOPs per token. For a 7B LM, a 576-token image costs 2 * 7e9 * 576 = 8.1 TFLOPs, the same as a 576-token prompt.
- **KV memory:** Llama-2-7B (MHA, 32 layers, d=4096) stores 512 KiB per token, so one 576-token image is **288 MiB of KV per sequence**. With GQA-8 it is 4x less.
- **Tiling multiplies this.** LLaVA-NeXT "AnyRes" at 672 x 672 uses 4 tiles plus a downscaled global view, 5 x 576 = 2880 tokens. Qwen2-VL on a 1080p frame is about 2.6k tokens.
- **Video multiplies again.** 60 s at 1 fps x 256 tokens/frame = 15,360 tokens, before any audio.

This is why token-reduction methods (pixel shuffle, pooling, resamplers, dropping tokens from static video frames) and long-context architectures matter so much for multimodal workloads: vision tokens are high-volume and low in information per token. It also means the ViT encoder itself is usually a small share of compute next to the LM prefill over its output tokens. It is not negligible at high resolution, though, and the encoder can be a latency bottleneck if it runs in a separate, unbatched stage.

### Audio: Whisper

"Robust Speech Recognition via Large-Scale Weak Supervision", Radford et al., arXiv 2212.04356.

Whisper is a classic **encoder-decoder** transformer. The input is a 30 s window converted to a log-mel spectrogram (80 bins, 128 for large-v3) at a 10 ms hop, giving 3000 frames. Two conv layers follow, the second with stride 2, giving **1500 encoder positions**, which the encoder processes bidirectionally. The decoder autoregressively generates text tokens (with special task, language, and timestamp tokens) and cross-attends to the encoder output. Large is 1.55B parameters. Trained on 680k hours of weakly supervised data.

Systems profile:

- The encoder runs once per 30 s chunk, which is compute-bound and batches well.
- The decoder has a self-attention KV cache that grows with the output, plus a **cross-attention KV computed once** from the encoder output and reused at every step.
- Since output text is short relative to the 1500 encoder positions, the encoder dominates for short transcripts. Distil-Whisper and "turbo" variants shrink the decoder, because decode is where latency goes for long outputs.
- Long audio means chunking with overlap and stitching. Streaming needs extra machinery, because the encoder expects whole 30 s windows.

Audio in LLMs follows the vision patterns. Understanding models (such as Qwen-Audio) usually use an audio encoder (often Whisper's) plus a projector. Generation and full-duplex speech models use discrete audio codec tokens, with several codebooks per frame modeled jointly. Moshi (Défossez et al., arXiv 2410.00037) is an example. The token-count arithmetic works the same way: frames/s x codebooks.


## Diffusion models

### DDPM objective

"Denoising Diffusion Probabilistic Models", Ho, Jain, Abbeel, arXiv 2006.11239.

Forward process (fixed, no parameters): `x_t = sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps`, with `eps ~ N(0, I)`, `abar_t = prod_{s<=t} (1 - beta_s)`. With the original linear schedule (`beta` from 1e-4 to 0.02 over 1000 steps), `abar_1000 = 4e-5`, which is almost pure noise. The network learns to predict the noise:

```python
def ddpm_loss(model, x0, alphas_bar):
    # x0: (B, C, H, W)   alphas_bar: (num_steps,)
    B = x0.shape[0]
    t = torch.randint(0, len(alphas_bar), (B,))                    # (B,) random timestep per sample
    a = alphas_bar[t].view(B, 1, 1, 1)
    eps = torch.randn_like(x0)
    xt = a.sqrt() * x0 + (1 - a).sqrt() * eps                      # (B, C, H, W)
    return F.mse_loss(model(xt, t), eps)                           # simplified (unweighted) ELBO
```

Sampling runs the learned reverse process from `x_T ~ N(0, I)`, one network evaluation per step. That was 1000 steps originally. DDIM (Song et al., arXiv 2010.02502) brought it to about 50 deterministic steps without retraining.

**Classifier-free guidance** (Ho and Salimans, arXiv 2207.12598) runs the model with and without the condition and extrapolates: `eps = eps_uncond + w (eps_cond - eps_uncond)`. That **doubles** the cost of every step, unless the two evaluations are batched together (same FLOPs, better utilization) or distilled away.

### Latent diffusion: VAE + U-Net

"High-Resolution Image Synthesis with Latent Diffusion Models", Rombach et al., arXiv 2112.10752.

Diffusing in pixel space at 512 x 512 x 3 is expensive. LDM first trains a VAE that downsamples by f=8: a 512 x 512 x 3 image becomes a 64 x 64 x 4 latent, 48x fewer values. Diffusion runs in that latent space. The denoiser is a **U-Net**: a conv ResNet encoder/decoder with skip connections, with self-attention and text cross-attention blocks at lower resolutions. Text conditioning comes from a frozen text encoder (CLIP for Stable Diffusion 1.x) through cross-attention. SD 1.x's U-Net is about 860M parameters. The VAE decoder runs once at the end, and at high resolutions its cost and memory are significant, so tiled VAE decoding is common.

### DiT: transformer backbone

"Scalable Diffusion Models with Transformers", Peebles and Xie, arXiv 2212.09748.

DiT replaces the U-Net with a ViT over **patchified latents**. A 256px image becomes a 32 x 32 x 4 latent, which at patch size 2 is 256 tokens. Timestep and class conditioning go in through **adaLN-Zero**: per-block scale, shift, and gate regressed from the conditioning embedding, with gates initialized to 0 so each block starts as the identity. The paper's key finding is that sample quality (FID) tracks forward GFLOPs, whether those come from model size or from more tokens (smaller patches). That gives diffusion the same scaling story transformers have for LMs. As I recall, DiT-XL/2 is about 119 GFLOPs per forward at 256px.

DiT descendants now dominate:

- SD3 ("Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", Esser et al., arXiv 2403.03206) introduced MM-DiT, with separate weights for text and image tokens and joint attention over the concatenated sequence.
- FLUX.1 (Black Forest Labs, no paper; about 12B params, a rectified-flow MM-DiT-style transformer, according to the public reference code).
- Video models (Sora's technical report describes "spacetime patches" with a diffusion transformer; open models like HunyuanVideo and Wan are DiT-style).

Token count at 1024px: the VAE (f=8) gives 128 x 128 latents, and at patch 2 that is **4096 image tokens**. With full bidirectional attention, a 4x4 increase in pixel area is a 16x increase in attention FLOPs, hence the windowed and sequence-parallel tricks for video.

### Flow matching / rectified flow

- "Flow Matching for Generative Modeling", Lipman et al., arXiv 2210.02747.
- "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow", Liu, Gong, Liu, arXiv 2209.03003.

Define a straight-line interpolation between data and noise, and regress the velocity:

```python
def flow_matching_loss(model, x0):
    # x0: (B, C, H, W) data; t ~ U(0,1) (SD3 samples t from a logit-normal instead)
    B = x0.shape[0]
    t = torch.rand(B).view(B, 1, 1, 1)
    noise = torch.randn_like(x0)
    xt = (1 - t) * x0 + t * noise                                  # (B, C, H, W)
    return F.mse_loss(model(xt, t.flatten()), noise - x0)          # target velocity dx/dt
```

Sampling integrates the ODE `dx/dt = v(x, t)` from t=1 to t=0 with Euler or a higher-order solver. Straighter paths need fewer steps. "Rectified flow" re-pairs noise and data through the trained model ("reflow") to straighten the paths further. SD3 and FLUX use this formulation. Step distillation goes further:

- consistency models (Song et al., arXiv 2303.01469)
- progressive distillation (Salimans and Ho, arXiv 2202.00512)
- adversarial distillation (SDXL-Turbo)

These get to 1-8 steps at some quality cost.

### Compute profile and serving implications

Diffusion inference is shaped nothing like LLM decode:

| | LLM decode | diffusion sampling |
|---|---|---|
| work per output | 1 forward over 1 new token | S steps x (1 or 2 for CFG) forwards over **all** tokens |
| attention | causal, cached | bidirectional, recomputed every step |
| cache | KV cache, grows with T | none across steps (latents only); text-encoder output computed once |
| bound at batch 1 | memory bandwidth (weights read per token) | **compute** (thousands of tokens per weight read) |
| latency lever | batch, speculative decoding, KV tricks | fewer steps (distillation), smaller/faster model, FP8, parallelism across devices |

The arithmetic for a FLUX-like model (about 12B params, d=3072, 57 blocks, 4096 image + 512 text tokens; shape figures approximate):

```
dense per step   = 2 * 12e9 * 4608               = 110.6 TFLOP
attention/step   = 4 * 4608^2 * 3072 * 57         =  14.9 TFLOP
per step                                          ~ 125  TFLOP
28 steps (guidance-distilled, no CFG doubling)    ~ 3.5  PFLOP per 1024px image
H100 dense bf16 peak 989 TFLOP/s                  -> 3.6 s at 100% MFU, ~8.9 s at 40%
```

Compare a 7B LLM decoding at batch 1: about 14 GB of weights read per token at 3.35 TB/s gives about 4.2 ms/token, bandwidth-bound and with tensor cores mostly idle. For diffusion, by contrast:

- **Batching gives little throughput** once a single image saturates the GPU. Batching is about utilization at small resolutions, not the order-of-magnitude win it is for LLM decode.
- **Latency scales linearly with steps.** Step distillation is the biggest single lever. The next tier: caching and reusing intermediate features across adjacent steps (DeepCache and TeaCache-style methods), and CFG distillation.
- **No KV cache means memory is weights plus activations.** Memory per request is roughly constant, which makes capacity planning easy. Video is the exception: tens of thousands of tokens, so activations and attention dominate and you need sequence parallelism (Ulysses or ring over the token dimension; xDiT is one implementation).
- **The pipeline has several models:** text encoder(s), possibly a large T5-XXL, run once; the denoiser, run S times; the VAE decoder, run once. Offloading the text encoder or keeping it resident is a real memory decision.
- FP8 or INT8 quantization of the denoiser helps directly, because the workload is compute-bound. Weight-only quantization (the LLM decode trick) helps much less here, because the bottleneck isn't reading weights.


## Reasoning and test-time-compute models

Reasoning models (OpenAI o1/o3, DeepSeek-R1 ("DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", arXiv 2501.12948), Qwen3 thinking modes, and so on) are, as far as anything public shows, **the same transformer architectures**: dense or MoE decoders, often with MLA, GQA, or SWA. What changes is the workload.

- **Decode-dominated.** Outputs of 5k-100k tokens per query, versus a few hundred for chat. Per-request cost shifts from prefill to decode, and decode is bandwidth-bound. The KV cache now grows *during generation*, so a request that arrives small becomes large, which makes admission control and preemption harder. Prefix caching helps less, because the expensive part is the unique output.
- **This is where sub-quadratic and hybrid architectures pay off most.** Decoding 64k tokens with a 70B-shaped GQA model ends at 20 GiB of KV per sequence. A 1:7 hybrid ends at about 2.5 GiB plus constant state. MiniMax-M1 made this exact argument for using a lightning-attention hybrid in an RL-trained reasoner.
- **RL training loops.** GRPO ("DeepSeekMath", arXiv 2402.03300) and its relatives drop the critic, but each update still needs G rollouts (for example 8-64) per prompt. Rollout generation is commonly reported as the majority of wall-clock in these pipelines (I'd treat "majority" as typical, not universal). The systems consequences:
  - an inference engine (vLLM/SGLang) runs inside the training loop, either colocated with the trainer or on separate GPUs, with weight sync after each update;
  - long-tail generation lengths create stragglers, which leads to partial rollouts, asynchronous or off-policy updates, and length caps;
  - small numerical mismatches between the inference engine's kernels and the trainer's kernels show up as an off-policy error that needs correcting (importance weighting).
- **Test-time scaling knobs** (best-of-N, self-consistency, longer thinking budgets) multiply decode tokens linearly. Price them with the decode arithmetic, not the prefill arithmetic.

There is no architectural change needed. The pressure is to make decode cheaper, which pushes toward MLA/GQA, KV quantization, speculative decoding, sparse attention, and hybrids.


## Comparison table

T = sequence length. "State per token" means the memory that must be kept per generated or seen token at inference. For fixed-state models it is the total state, which does not grow.

| family | training parallelism | inference state | compute in T (train / per decode step) | strengths | weaknesses | examples |
|---|---|---|---|---|---|---|
| Softmax attention (dense) | fully parallel (matmul, FlashAttention) | O(T) KV: `2 L h_kv d_h` bytes/token | O(T^2) / O(T) | exact recall, best ICL, mature kernels and serving | KV memory and bandwidth at long T | Llama, Qwen, DeepSeek, GPT |
| Sliding-window attention | parallel | O(window) | O(T w) / O(w) | bounded KV, simple | no retrieval beyond the window unless mixed with global layers | Mistral 7B, Gemma 2/3 (interleaved) |
| Sparse attention (NSA/DSA/MoBA) | parallel, custom kernels | O(T) KV (all kept) | about O(T k) + cheap selector / O(k) | cuts compute and bandwidth, keeps full-context access | doesn't cut KV storage; kernels complex; selection may miss | DeepSeek-V3.2-Exp (DSA) |
| Linear attention (vanilla) | chunkwise parallel | O(H dk dv) constant | O(T) / O(1) | tiny state, fast decode | weak recall, no forgetting; quality gap | research |
| RetNet | chunkwise parallel | constant | O(T) / O(1) | simple, three equivalent forms | fixed decay, behind gated variants | research |
| GLA / RWKV-6 | chunkwise (FLA kernels) | constant (matrix) | O(T) / O(1) | data-dependent forgetting, good throughput | recall limited by state | RWKV-6, research |
| DeltaNet / Gated DeltaNet / RWKV-7 | chunkwise (WY repr.) | constant (matrix) | O(T) / O(1) | best recall among fixed-state, state tracking | more complex kernels; still bounded memory | Qwen3-Next (hybrid), Kimi Linear (hybrid), RWKV-7 |
| S4 / LTI SSM | FFT convolution | O(d n) constant | O(T log T) / O(1) | long continuous signals, stable | no content selection; weak on language | audio/time series, LRA |
| Mamba (selective SSM) | parallel scan (CUDA cores) | O(d n) + conv cache, constant | O(T) / O(1) | content-aware gating, fast decode | recall/copying; scan doesn't use tensor cores | Mamba, Falcon Mamba, Jamba (inside) |
| Mamba-2 (SSD) | chunkwise matmul (tensor cores) | O(d n), larger n, constant | O(T) / O(1) | fast training, TP-friendly, larger state | recall/copying still bounded | Nemotron-H (inside), Codestral Mamba, IBM Granite 4.0 hybrids (unverified) |
| Hybrid (linear/SSM + some attention) | parallel (mixed kernels) | O(T) KV x attention fraction + constant | O(T^2 · frac) / O(T · frac) | most of the attention quality at a fraction of the KV | two cache types complicate serving; ratio is a new hyperparameter | Jamba, Zamba, Samba, Nemotron-H, MiniMax-01, Qwen3-Next, Kimi Linear |
| Encoder (ViT/BERT, bidirectional) | parallel | none (single forward pass) | O(T^2) | cheap, one-shot embeddings | no generation | SigLIP, CLIP, DINOv2, ModernBERT |
| Encoder-decoder (Whisper) | parallel (teacher forcing) | self-attn KV (output) + fixed cross-attn KV | O(T_in^2 + T_out^2 + T_in T_out) | efficient conditional generation | two components to serve | Whisper, T5 |
| Diffusion (U-Net / DiT) | parallel, random timestep per sample | none across steps | S x O(T^2) per sample | high-quality continuous generation | many steps; compute-bound latency | SD1.x/XL (U-Net), SD3, FLUX, video DiTs |


## Practical guidance

**Default to a dense or MoE transformer with GQA or MLA** if contexts are mostly under about 32k and you need the strongest quality per parameter with the least engineering risk. Every inference engine, quantization tool, speculative decoder, and fine-tuning recipe assumes this architecture.

**Consider a hybrid** when:

- the workload has long contexts (128k+), long generations (reasoning, agents), or high concurrency at moderate context, so that KV capacity limits batch size;
- you control the serving stack, or the engines you use support the model (see below).

Start from a published ratio (1:3 to 1:7 attention) rather than inventing one. Evaluate on multi-key retrieval and long-context QA, not only short benchmarks.

**Pure SSM or linear models** make sense:

- on-device or streaming, where constant memory is a hard requirement;
- for very long continuous signals (audio, genomics, time series), where exact token recall matters less;
- as research baselines.

Expect a recall gap.

**Sparse attention (NSA/DSA style)** is the option when you want full-context access and compute savings, can afford the KV storage, and can write or adopt custom kernels.

**Sliding-window interleaving** is the cheapest change to a transformer for bounded KV, and is supported nearly everywhere.

**For vision-language:** a LLaVA-style projector on a SigLIP-class encoder, with dynamic resolution and token merging, is the well-trodden path. Budget image tokens explicitly. They are usually the dominant prefill cost.

**For image/video generation:** a DiT/MM-DiT with flow matching. Plan serving around step count and compute (FP8, distillation, sequence parallelism for video), not KV.

### Infrastructure (as of about late 2025; check current releases)

- **Kernels:**
  - `mamba_ssm` and `causal-conv1d` (CUDA, from the Mamba authors) for Mamba-1/2;
  - `flash-linear-attention` (fla-org; Triton) for GLA, DeltaNet, Gated DeltaNet, RWKV-6/7, RetNet, Mamba-2, and many others. This is the de facto reference;
  - the Triton SSD kernels in the Mamba repo;
  - FlashAttention and FlashInfer for the attention layers of hybrids.

  Kernel support on non-NVIDIA hardware for these mixers is much thinner than for attention. I'm not confident about the current ROCm or TPU status. On TPU, JAX implementations of scan and chunkwise forms exist.
- **Serving engines:** vLLM and SGLang support Mamba, Mamba-2, and hybrid models, including Jamba, Nemotron-H, Falcon-H1, and Qwen3-Next, the last with day-0 support announced. As I understand it, vLLM's V1 engine added a hybrid KV-cache manager that allocates attention KV pages and fixed-size recurrent state slots together. The rough edges: prefix caching, speculative decoding, and chunked prefill interactions with recurrent state were all incomplete or recently added at various points. TensorRT-LLM has Mamba and hybrid support. llama.cpp supports Mamba, RWKV, and some hybrids, with a GGUF ecosystem behind them. Verify per model.
- **Training:** Megatron-LM/Megatron-Core supports Mamba hybrids (used for Nemotron-H). Hugging Face `transformers` has modeling code for most of the above, but its pure-PyTorch fallback paths are very slow. Make sure the fused kernels are actually installed and being used. Benchmark tokens/s before and after.
- **Diffusion serving:** `diffusers` is the default library. xDiT and similar tools provide sequence/patch parallelism for DiTs. ComfyUI dominates hobbyist pipelines. Compiled and quantized deployments use TensorRT or torch.compile with FP8.


## Further reading

Sequence mixers and SSMs:

- Katharopoulos et al., "Transformers are RNNs", arXiv 2006.16236
- Sun et al., "Retentive Network", arXiv 2307.08621
- Peng et al., "RWKV", arXiv 2305.13048; "Eagle and Finch", arXiv 2404.05892; "RWKV-7 Goose", arXiv 2503.14456
- Yang et al., "Gated Linear Attention Transformers with Hardware-Efficient Training", arXiv 2312.06635
- Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length", arXiv 2406.06484
- Yang, Kautz, Hatamizadeh, "Gated Delta Networks", arXiv 2412.06464
- Gu et al., "HiPPO", arXiv 2008.07669; "S4", arXiv 2111.00396; "S4D", arXiv 2206.11893
- Gu and Dao, "Mamba", arXiv 2312.00752; Dao and Gu, "Transformers are SSMs" (Mamba-2), arXiv 2405.21060
- De et al., "Griffin", arXiv 2402.19427; Beck et al., "xLSTM", arXiv 2405.04517
- Songlin Yang's blog posts on DeltaNet, and Tri Dao and Albert Gu's blog series on Mamba-2 / SSD, for clearer derivations of the chunkwise algorithms than the papers give

Recall limitations:

- Arora et al., "Zoology", arXiv 2312.04927; "Based", arXiv 2402.18668
- Jelassi et al., "Repeat After Me", arXiv 2402.01032
- Waleffe et al., "An Empirical Study of Mamba-based Language Models", arXiv 2406.07887

Hybrids:

- Jamba, arXiv 2403.19887; Zamba, arXiv 2405.16712; Samba, arXiv 2406.07522
- Nemotron-H, arXiv 2504.03624; MiniMax-01, arXiv 2501.08313; Kimi Linear, arXiv 2510.26692 (id unverified); Qwen3-Next release blog and model card

Sparse and long-context attention:

- Child et al., arXiv 1904.10509; Longformer, arXiv 2004.05150; BigBird, arXiv 2007.14062
- StreamingLLM, arXiv 2309.17453; Ring Attention, arXiv 2310.01889
- NSA, arXiv 2502.11089; MoBA, arXiv 2502.13189; DeepSeek-V3.2-Exp technical report (DSA)

Vision, multimodal, audio:

- ViT, arXiv 2010.11929; CLIP, arXiv 2103.00020; SigLIP, arXiv 2303.15343; SigLIP 2, arXiv 2502.14786
- LLaVA, arXiv 2304.08485; LLaVA-1.5, arXiv 2310.03744; Flamingo, arXiv 2204.14198; BLIP-2, arXiv 2301.12597
- Qwen2-VL, arXiv 2409.12191; Chameleon, arXiv 2405.09818; Transfusion, arXiv 2408.11039
- Whisper, arXiv 2212.04356; Moshi, arXiv 2410.00037

Diffusion:

- DDPM, arXiv 2006.11239; DDIM, arXiv 2010.02502; Classifier-free guidance, arXiv 2207.12598
- LDM, arXiv 2112.10752; DiT, arXiv 2212.09748; SD3, arXiv 2403.03206
- Flow Matching, arXiv 2210.02747; Rectified Flow, arXiv 2209.03003
- Consistency Models, arXiv 2303.01469; Progressive Distillation, arXiv 2202.00512

Reasoning and RL:

- DeepSeekMath (GRPO), arXiv 2402.03300; DeepSeek-R1, arXiv 2501.12948; MiniMax-M1, arXiv 2506.13585 (id unverified)

Sibling chapters: [Model Architectures](architectures.md) (transformer, norms, RoPE, GQA/MLA, MoE), [Model Parallelism](../training/model-parallelism/README.md) (context/sequence parallelism), [Inference](../inference/README.md) (KV cache, batching, serving).
