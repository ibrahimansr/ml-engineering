# Model Architectures I: The Transformer and Its Evolution

This chapter goes through the design choices behind the modern decoder-only LLM one at a time. For each choice it covers what the choice is, what problem it solved, what evidence supports it, what it costs, and who uses it. The goal is not to catalogue every variant. It is to make the reasoning visible, so that a 2025 config file like `num_key_value_heads: 8, rope_theta: 500000, hidden_act: silu, rms_norm_eps: 1e-5` reads as a series of decisions with known motivations, not as magic numbers.

The recurring theme is that many "architecture" choices are really **hardware** choices. Some changes exist because they improve loss per FLOP: pre-LN, SwiGLU, RoPE, MoE. Others exist because they cut bytes moved per generated token: MQA/GQA, MLA, sliding windows, large vocabularies. A third group exists because a run at 10^25 FLOPs must not diverge on day 40: QK-norm, z-loss, no biases, softcapping. Knowing which bucket a choice belongs to tells you when it is safe to drop it.

Out of scope here: state-space models, linear attention, hybrids, diffusion LMs and multimodal architectures. They are covered in the sibling chapter [Model Architectures II: Beyond the Transformer](beyond-transformers.md). For the inference-side arithmetic (prefill vs decode, memory-bandwidth bounds) see [Inference](../inference/README.md). For how these architectures get sharded see [Model Parallelism](../training/model-parallelism/README.md).

Conventions used in the code: `B` = batch, `T` = sequence length, `C` or `d` = model width (`d_model`), `H` = query heads, `KV` = key/value heads, `hd` = head dim, `F` = FFN hidden size, `V` = vocab size, `L` = number of layers, `E` = number of experts.

All numeric claims that come from arithmetic (parameter counts, KV cache sizes, FLOPs) were computed with `python3` while writing this chapter. Claims about quality come from the cited papers. Where an arXiv id or a detail could not be confirmed, it is marked.



## From RNNs to "Attention Is All You Need"

### The seq2seq bottleneck

The 2014 encoder-decoder RNN ([Sutskever et al., Sequence to Sequence Learning with Neural Networks, arXiv:1409.3215](https://arxiv.org/abs/1409.3215)) compressed a whole source sentence into one fixed-size hidden vector and decoded from it. Quality fell off with sentence length, because a single `d`-dimensional state must carry everything.

[Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate, arXiv:1409.0473](https://arxiv.org/abs/1409.0473) fixed the bottleneck with **attention**. At each decoder step, compute a softmax-weighted average of *all* encoder states, with weights from a learned compatibility score. [Luong et al., arXiv:1508.04025](https://arxiv.org/abs/1508.04025) simplified the score to a dot product. The decoder no longer depends on one vector. It can look up whatever it needs, so memory grows with input length.

The RNN was still there, though, and the RNN is the problem from a hardware point of view. `h_t = f(h_{t-1}, x_t)` is a sequential dependency over `T`. During training you cannot compute step `t` before step `t-1`. A GPU with 10^4+ parallel lanes ends up running a `T`-long chain of small matmuls. Each matmul is `(B, d) x (d, d)`, which at realistic `B` is memory-bound, not compute-bound.

### The Transformer

[Vaswani et al., Attention Is All You Need, arXiv:1706.03762](https://arxiv.org/abs/1706.03762) removed the recurrence completely. Every position attends to every other position in one batched matmul:

```python
import torch, torch.nn as nn, torch.nn.functional as F
import math

def attention(q, k, v, causal=True):
    # q, k, v: (B, H, T, hd)
    T = q.shape[-2]
    att = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])     # (B, H, T, T)
    if causal:
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()
        att = att.masked_fill(~mask, float("-inf"))
    return att.softmax(-1) @ v                                 # (B, H, T, hd)
```

The original encoder-decoder Transformer:

![the Transformer](images/transformer-architecture.png)

([source](https://arxiv.org/abs/1706.03762))

Multi-head attention:

![multi-head attention](images/multi-head-attention.png)

([source](https://arxiv.org/abs/1706.03762))

Three properties made this win.

1. **Parallel over T at training time.** With teacher forcing, all `T` positions of a sequence are computed at once. The causal mask stops information leaking from the future. Under the RNN, the same computation is a `T`-step loop. The work becomes a few large GEMMs (`QKV` projection, `QK^T`, `AV`, output projection, MLP), which is exactly what tensor cores are built for.
2. **Constant path length.** Any two tokens interact in one layer. In an RNN, information from token 1 reaches token 1000 through 999 nonlinear steps, and the gradient takes the same path in reverse. This is a statement about optimization, not expressivity, and it is a large part of why Transformers learn long-range dependencies so much more easily.
3. **The `1/sqrt(hd)` scale.** If `q` and `k` have unit-variance entries, `q·k` has variance `hd`. Without the scale, the softmax saturates at init and its gradients vanish. Vaswani et al. give this reason directly.

**The cost** is the `(T, T)` score matrix. For each layer and head, `QK^T` and `AV` each cost `2·T²·hd` FLOPs, and a naive implementation stores `T²` scores per head. Summed over heads (`H·hd = d`), the causal attention cost averaged over the tokens of a length-`T` sequence is about `2·L·T·d` forward FLOPs per token. Compare that with `2N` for the weight matmuls. For Llama-3-8B (`L=32, d=4096, N≈8.03B`):

| T | attention fwd FLOPs/token | `2N` | attention / `2N` |
|---|---|---|---|
| 2K | 0.54 GFLOP | 16.1 GFLOP | 3% |
| 8K | 2.15 GFLOP | 16.1 GFLOP | 13% |
| 32K | 8.6 GFLOP | 16.1 GFLOP | 53% |
| 128K | 34.4 GFLOP | 16.1 GFLOP | 214% |

At pretraining lengths (4K-8K), attention FLOPs are a rounding error next to the MLP. That explains why "quadratic attention" did not stop scaling, and why long-context extension happens as a short final training phase. The `T²` *memory* problem was solved separately by FlashAttention ([Dao et al., arXiv:2205.14135](https://arxiv.org/abs/2205.14135)). It tiles the softmax so that the `(T, T)` matrix never reaches HBM, which makes memory `O(T)` while FLOPs stay `O(T²)`. The FLOPs problem is what sliding windows, sparse attention and the sub-quadratic alternatives in the [sibling chapter](beyond-transformers.md) attack.

**At inference time** the picture flips. Decoding generates one token at a time, so parallelism over `T` is gone. Each new token must read the keys and values of all previous tokens. That cached state, the **KV cache**, is the main reason so many later attention variants exist. Training is compute-bound, and decoding is memory-bandwidth-bound on weights plus KV cache. The rest of this chapter keeps coming back to that asymmetry.



## Encoder-only, encoder-decoder, decoder-only

The original Transformer was an encoder-decoder for translation. Within two years the field split into three families. By 2023 essentially all frontier LLMs were decoder-only. The reasons are worth spelling out, because they are mostly about efficiency and not about expressivity.

### Encoder-only: BERT

[Devlin et al., BERT, arXiv:1810.04805](https://arxiv.org/abs/1810.04805) uses a bidirectional encoder (no causal mask) trained with **masked language modeling**: replace 15% of tokens and predict them from both sides. Bidirectional context is ideal for representation tasks like classification, NER and retrieval embeddings. The model is still widely used in that role (ModernBERT, the E5/GTE embedding families).

Why it lost as a generative pretraining recipe:

- **Only ~15% of positions give a loss signal**, while a causal LM gets a loss at 100% of positions for the same forward/backward FLOPs. ELECTRA ([arXiv:2003.10555](https://arxiv.org/abs/2003.10555)) was motivated exactly by this sample inefficiency.
- **It does not generate.** Iterative unmasking is possible but awkward, and nothing like a KV cache exists, because every token attends to every other token, including ones that have not been produced yet.

### Encoder-decoder: T5

[Raffel et al., T5, arXiv:1910.10683](https://arxiv.org/abs/1910.10683) cast every task as text-to-text. A bidirectional encoder reads the input, and a causal decoder with cross-attention produces the output. Pretraining used span corruption. In the T5 paper's own architecture comparison (Section 3.2, Table 2), the encoder-decoder with a denoising objective beat a decoder-only LM of comparable size and compute on their downstream fine-tuning suite.

Since T5's comparison favored the encoder-decoder, why did that not hold up for LLMs? Three reasons:

1. **The evaluation changed.** T5 evaluated after fine-tuning on each task. From GPT-3 onward ([Brown et al., arXiv:2005.14165](https://arxiv.org/abs/2005.14165)) the target became zero/few-shot prompting. [Wang et al., What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?, arXiv:2204.05832](https://arxiv.org/abs/2204.05832) trained models of about 5B parameters on 170B tokens across architectures and objectives. Their finding: **a causal decoder-only model trained with a plain LM objective gave the best zero-shot generalization after purely self-supervised pretraining**, while encoder-decoder + MLM was best *after* multitask fine-tuning. The target moved, and the ranking moved with it.
2. **The parameter split is awkward.** An encoder-decoder with the same total parameters spends about half of them on the encoder, and the encoder only processes the input. For open-ended continuation, where the "input" is just the earlier part of the same text, the split is arbitrary. You must decide where the input ends, and both stacks have to be sized.
3. **Inference.** In a decoder-only model, every past token's K/V never changes once computed, so the KV cache is append-only. For a multi-turn chat, the whole conversation is one prefix, and that prefix can be cached and even shared across requests (prefix caching). In an encoder-decoder, appending a user turn changes the *encoder's* bidirectional representations of everything already there, so the encoder has to rerun. The serving stack of 2023-2025, built around paged KV caches, prefix sharing and speculative decoding, assumes an append-only causal state.

### Decoder-only: GPT

The GPT line (GPT-1 2018, GPT-2 2019, GPT-3 [arXiv:2005.14165](https://arxiv.org/abs/2005.14165); GPT-1 and GPT-2 were released as OpenAI reports, not on arXiv) uses one stack, a causal mask, and next-token prediction on raw text. Why it won:

- **Training efficiency:** loss at every token, one homogeneous stack, and nothing to tune about how to split the parameters. A uniform stack of identical blocks also pipelines and shards trivially.
- **A single objective covers everything.** Translation, QA, code and dialogue are all continuation. In-context learning (GPT-3) falls out without a task-specific head.
- **KV cache friendly**, as explained above. Generation cost per token is `O(T)` reads of cached K/V, not a recompute.
- **It scales predictably.** Scaling laws ([Kaplan et al., arXiv:2001.08361](https://arxiv.org/abs/2001.08361); [Hoffmann et al., Chinchilla, arXiv:2203.15556](https://arxiv.org/abs/2203.15556)) were fit on decoder-only LMs, so their compute planning carries over directly.

A middle ground is the **prefix LM**, which has bidirectional attention over a prefix and causal attention after it (T5 explored it; UL2, [arXiv:2205.05131](https://arxiv.org/abs/2205.05131), mixed objectives). It survives mostly in multimodal models, which attend bidirectionally over image tokens. For text LLMs the operational simplicity of a pure causal mask won.



## Normalization

### Post-LN vs pre-LN

The original Transformer used **post-LN**: `x = LN(x + F(x))`. GPT-2 moved to **pre-LN**: `x = x + F(LN(x))`, with a final LN before the head. Nearly every LLM since then is pre-norm.

```python
class PreLNBlock(nn.Module):
    def __init__(self, d, attn, mlp, norm=nn.LayerNorm):
        super().__init__()
        self.n1, self.n2, self.attn, self.mlp = norm(d), norm(d), attn, mlp
    def forward(self, x):            # x: (B, T, d)
        x = x + self.attn(self.n1(x))  # residual stream is never normalized
        x = x + self.mlp(self.n2(x))
        return x
```

**Problem solved:** [Xiong et al., On Layer Normalization in the Transformer Architecture, arXiv:2002.04745](https://arxiv.org/abs/2002.04745) showed through mean-field analysis that in post-LN, the expected gradient of the parameters near the output is large at init. That is why post-LN needs a learning-rate warmup to avoid diverging. In pre-LN, gradients are well-behaved at init. The residual stream is a pure sum `x_L = x_0 + Σ F_l(·)`, so there is an identity path from loss to embeddings with no normalization in between. They report training pre-LN *without* warmup at reduced wall-clock cost on BERT and translation.

**The cost:** in pre-LN, the residual stream's norm grows with depth (each block adds to it and nothing rescales it). A late block's contribution `F_l` is then small relative to `x`, so deep pre-LN models use their last layers less effectively. [Liu et al., Understanding the Difficulty of Training Transformers (Admin), arXiv:2004.08249](https://arxiv.org/abs/2004.08249) argue that post-LN, when it trains, can reach better quality, and that its instability comes from heavy dependence on the residual branch, not from gradient vanishing alone. DeepNet ([arXiv:2203.00555](https://arxiv.org/abs/2203.00555)) trained 1000-layer post-LN models by scaling up the residual. In practice the LLM world chose pre-LN: losing a little quality is far cheaper than losing a run.

### RMSNorm

[Zhang & Sennrich, Root Mean Square Layer Normalization, arXiv:1910.07467](https://arxiv.org/abs/1910.07467) drops LayerNorm's mean-centering and bias:

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps, self.weight = eps, nn.Parameter(torch.ones(d))
    def forward(self, x):                                # x: (B, T, d)
        xf = x.float()                                   # norm stats in fp32
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * xf.type_as(x)
```

**Why:** the paper claims that LayerNorm's benefit comes from re-scaling invariance, not re-centering, and reports 7%-64% running-time reductions across their models (that is the paper's range, measured on 2019 hardware and implementations). On a modern GPU with a fused kernel, normalization is a small, memory-bound share of step time either way. The real reasons it stuck are simplicity: one reduction instead of two, no bias, fewer parameters to shard. LLaMA ([arXiv:2302.13971](https://arxiv.org/abs/2302.13971)) made pre-RMSNorm the open-model default. Note the `x.float()`: computing the statistic in bf16 is a classic source of loss spikes.

**Removing biases** happened alongside. PaLM ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311), Section 2) reports that removing biases from dense layers and LayerNorms improved training stability at scale. Llama, Mistral, DeepSeek and Gemma have no linear biases. The well-known exception is Qwen2/2.5, which keeps a bias on Q/K/V. Qwen3 removed it and added QK-norm (Qwen3 technical report, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388)).

### QK-norm

```python
# inside attention, after projecting and splitting heads:
q = self.q_norm(q)   # RMSNorm over hd, per head: (B, H, T, hd)
k = self.k_norm(k)   # (B, KV, T, hd)
# then apply RoPE, then attention as usual
```

**Problem solved:** **attention logit growth.** If `||q||` and `||k||` grow during training, the logits `q·k/sqrt(hd)` grow quadratically, the softmax collapses to one-hot, and training diverges. [Dehghani et al., Scaling Vision Transformers to 22B, arXiv:2302.05442](https://arxiv.org/abs/2302.05442) hit this at 22B and fixed it by normalizing Q and K. [Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities, arXiv:2309.14322](https://arxiv.org/abs/2309.14322) reproduced logit-growth divergence in small models at high learning rates and showed QK-layernorm mitigates it, widening the range of learning rates that train stably. The idea goes back to [Henry et al., Query-Key Normalization for Transformers, arXiv:2010.04245](https://arxiv.org/abs/2010.04245).

**Cost:** two small norms per layer, plus a subtle constraint. QK-norm must be applied *before* RoPE (the rotation preserves norm, so the order matters only for what gets normalized). Also, with the logits bounded by the learned norm gains, the model cannot produce arbitrarily sharp attention without growing those gains.

**Who uses it:** OLMo 2 ([arXiv:2501.00656](https://arxiv.org/abs/2501.00656)), Gemma 3 ([arXiv:2503.19786](https://arxiv.org/abs/2503.19786), which uses QK-norm *instead of* Gemma 2's softcapping), Qwen3, and many 2025 models.

**The alternative, logit softcapping:** Gemma 2 ([arXiv:2408.00118](https://arxiv.org/abs/2408.00118)) bounds logits with `cap · tanh(logits / cap)`, using 50 for attention and 30 for the final LM logits. It works, but it changes the softmax inside the attention kernel, so FlashAttention initially did not support it. Gemma 3 moved to QK-norm. A related stabilizer is **z-loss** (`1e-4 · log²Z` on the output softmax normalizer), used by PaLM and ST-MoE.

### Sandwich and "peri-LN"

Pre-LN normalizes the *input* of each sublayer but leaves the *output* unbounded, and that is how the residual grows. The fix is to also normalize the output before adding it back:

```python
x = x + self.post_attn_norm(self.attn(self.pre_attn_norm(x)))
x = x + self.post_mlp_norm(self.mlp(self.pre_mlp_norm(x)))
```

This appeared as "sandwich LN" in CogView ([arXiv:2105.13290](https://arxiv.org/abs/2105.13290)), which used it to tame fp16 overflow. NormFormer ([arXiv:2110.09456](https://arxiv.org/abs/2110.09456)) added extra norms too. Gemma 2 and Gemma 3 use pre- *and* post-sublayer RMSNorm. OLMo 2 normalizes *only* the sublayer outputs (`x = x + Norm(F(x))`, in the style of Swin-v2) and reports better stability. Kim et al., "Peri-LN: Revisiting Layer Normalization in the Transformer Architecture" ([arXiv:2502.02732](https://arxiv.org/abs/2502.02732), id unverified) give this placement a name and argue it controls activation variance growth better than pure pre-LN. **Cost:** one extra norm per sublayer, which is negligible in FLOPs, plus one extra kernel launch or fusion boundary.



## Positional information

Attention without a mask is permutation-equivariant, so it needs some notion of order. A causal mask does leak position (see NoPE below), but every production model since 2017 has added explicit positional information.

### Sinusoidal and learned absolute

Vaswani et al. added a fixed `PE[pos, 2i] = sin(pos / 10000^(2i/d))`, `PE[pos, 2i+1] = cos(...)` to the embeddings. Their motivation was that `PE[pos+k]` is a linear function of `PE[pos]`, so relative offsets are in principle learnable. GPT-1/2/3 and BERT used **learned** absolute embeddings, a `(T_max, d)` table. Vaswani et al. report that the two performed nearly identically.

**The problem with both:** they are additive in the residual stream, so position information mixes with content in every projection. Learned tables also cannot represent any position beyond `T_max`, and sinusoids extrapolate poorly in practice.

### Relative: Shaw and T5 bias

[Shaw et al., Self-Attention with Relative Position Representations, arXiv:1803.02155](https://arxiv.org/abs/1803.02155) add learned embeddings for clipped relative distances `j - i` into the attention computation. Transformer-XL ([arXiv:1901.02860](https://arxiv.org/abs/1901.02860)) refined this. **T5** simplified it to a learned **scalar bias per head** added to the logits, with relative distances bucketed logarithmically (32 buckets, up to distance 128 in T5's config) and shared across layers. It is cheap and relative, but the bias is a full `(H, T, T)` additive term. That fits awkwardly into fused attention kernels, and distances beyond the last bucket all look alike.

### ALiBi

[Press et al., Train Short, Test Long: Attention with Linear Biases, arXiv:2108.12409](https://arxiv.org/abs/2108.12409) adds no learned parameters, only a fixed linear penalty `-m_h · (i - j)` with head-specific slopes `m_h = 2^(-8h/H)`:

```python
def alibi_bias(H, T):
    slopes = 2.0 ** (-8.0 * torch.arange(1, H + 1) / H)          # (H,)
    dist = torch.arange(T)[None, :] - torch.arange(T)[:, None]    # (T, T): j - i
    return slopes[:, None, None] * dist.clamp(max=0)[None]        # (H, T, T), <= 0
# att = q @ k.T / sqrt(hd) + alibi_bias(H, T)
```

**Evidence:** in the paper's abstract, a 1.3B model trained at length 1024 with ALiBi extrapolates to 2048 and matches the perplexity of a sinusoidal model trained at 2048, while training 11% faster and using 11% less memory. **Tradeoff:** the strong recency bias that makes it extrapolate also discourages using distant tokens. Later long-context work found it weaker than RoPE-with-scaling at actually *retrieving* far-away information. **Users:** BLOOM, MPT. It largely disappeared after 2023.

### RoPE

[Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding, arXiv:2104.09864](https://arxiv.org/abs/2104.09864) is the modern default. The derivation is short.

Split `q` (and `k`) of size `hd` into `hd/2` pairs and treat each pair as a complex number `q_j = q[2j] + i·q[2j+1]`. Give each pair a frequency `θ_j = base^(-2j/hd)`, with `base = 10000` originally. At position `m`, rotate:

```
q_j(m) = q_j · e^{i m θ_j}
k_j(n) = k_j · e^{i n θ_j}
```

The real dot product of two 2D vectors is `Re(a · conj(b))`, so the attention logit is

```
<q(m), k(n)> = Σ_j Re( q_j e^{i m θ_j} · conj(k_j e^{i n θ_j}) )
             = Σ_j Re( q_j conj(k_j) · e^{i (m - n) θ_j} )
```

This depends on `m` and `n` **only through `m - n`**. RoPE is therefore an *absolute* encoding (each token is rotated by its own position, with no `T x T` bias tensor) that produces *relative* attention. Rotations preserve norm, so content magnitudes are untouched. The frequencies span from `θ_0 = 1` rad/token (fast, local) down to about `base^(-1)` rad/token (slow, global), like a multi-scale clock.

```python
def rope(x, pos, base=10000.0):
    # x: (B, H, T, hd), pos: (T,)  -- interleaved-pair convention
    hd = x.shape[-1]
    inv_freq = base ** (-torch.arange(0, hd, 2, device=x.device).float() / hd)  # (hd/2,)
    ang = pos[:, None].float() * inv_freq[None]                                 # (T, hd/2)
    cos, sin = ang.cos(), ang.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]                                         # (B, H, T, hd/2)
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)

# check: logit depends only on m - n
q, k = torch.randn(1, 1, 1, 64), torch.randn(1, 1, 1, 64)
s = lambda m, n: (rope(q, torch.tensor([m])) * rope(k, torch.tensor([n]))).sum()
assert torch.allclose(s(5, 2), s(105, 102), atol=1e-4)
```

(HF Llama uses the "rotate-half" convention, pairing dimension `j` with `j + hd/2` instead of `2j, 2j+1`. The math is the same, but the weight layouts differ, which is a common porting bug.)

**Why it won:** it is relative, has no parameters, is applied element-wise to Q and K *before* the attention kernel (so FlashAttention needs no changes), and composes with the KV cache, because each cached key is rotated once at its own position. The RoFormer paper also argues for a long-term decay of the logit bound with distance. **Users:** GPT-NeoX, PaLM, all Llamas, Mistral, Qwen, DeepSeek, Gemma.

**Cost:** two element-wise multiply-adds on Q and K. The real cost shows up in two places. First, RoPE does not extrapolate: at positions beyond training, the slow dimensions see rotation angles they never saw. Second, it interacts badly with KV compression, which is where MLA's "decoupled RoPE" comes from (see below).

### Extending RoPE: PI, NTK-aware, YaRN, and raising the base

When `m - n` exceeds the training length `L_train`, the low-frequency pairs reach angles never seen in training and quality collapses. All the fixes remap angles back into the trained range.

Extrapolation vs interpolation:

![position interpolation](images/position-interpolation.png)

([source](https://arxiv.org/abs/2306.15595))

- **Position Interpolation (PI)**, [Chen et al., arXiv:2306.15595](https://arxiv.org/abs/2306.15595): use `pos · L_train / L_new`, compressing every frequency uniformly by `s = L_new / L_train`. The abstract reports extending LLaMA to 32768 with about 1000 fine-tuning steps. The downside is that it also squashes the high-frequency dimensions that encode local order, so nearby tokens become harder to tell apart.
- **NTK-aware scaling** (a 2023 Reddit/GitHub post by "bloc97", no paper): instead of scaling positions, raise the base, `base' = base · s^(hd/(hd-2))`. This barely changes the fastest frequencies and interpolates the slowest ones by about `s`. It works reasonably even without fine-tuning.
- **YaRN**, [Peng et al., arXiv:2309.00071](https://arxiv.org/abs/2309.00071): "NTK-by-parts". Dimensions whose wavelength is much shorter than `L_train` are left alone (extrapolate), dimensions whose wavelength exceeds `L_train` are interpolated as in PI, and a linear ramp covers the band in between. It also adds an attention temperature, scaling logits by about `(0.1·ln s + 1)²` overall (the paper writes it as `sqrt(1/t) = 0.1 ln s + 1` applied to both q and k). The abstract claims about 10x fewer tokens and 2.5x fewer steps than prior methods. Users: Qwen2.5/Qwen3 (32K native to 128K via YaRN), DeepSeek-V3 (4K to 128K in two YaRN stages, per the V3 report).
- **Llama 3.1** uses its own frequency-dependent rescaling (scale factor 8, with separate low/high-frequency thresholds), in the same spirit as NTK-by-parts. It is documented in the released code and config rather than in a separate paper.
- **Large base from the start:** Llama 3 trains with `rope_theta = 500000` and Qwen2.5/Mixtral use `1e6`, so the slowest frequencies cover much longer wavelengths before any extension. Gemma 3 uses base 10K for local layers and 1M for global layers.

```python
def scaled_inv_freq(hd, base, s, kind):
    i = torch.arange(0, hd, 2).float() / hd
    if kind == "pi":  return base ** (-i) / s                          # uniform compression
    if kind == "ntk": return (base * s ** (hd / (hd - 2))) ** (-i)    # raise the base
```

### NoPE and interleaved NoPE

A causal decoder can infer position without any positional encoding, for example by counting how attention spreads over the prefix. [Haviv et al., Transformer Language Models without Positional Encodings Still Learn Positional Information, arXiv:2203.16634](https://arxiv.org/abs/2203.16634) showed that NoPE causal LMs come close to models with positional encodings. [Kazemnejad et al., The Impact of Positional Encoding on Length Generalization in Transformers, arXiv:2305.19466](https://arxiv.org/abs/2305.19466) found that NoPE length-generalized better than RoPE/ALiBi on their (small, synthetic) tasks. Pure NoPE has not been adopted at scale. The hybrid has: **some layers without RoPE**. Llama 4 interleaves NoPE layers with RoPE layers ("iRoPE", per Meta's blog, with no paper), and SmolLM3 drops RoPE on every fourth layer (per Hugging Face's blog). The intuition is that RoPE layers handle local order, and NoPE layers do position-agnostic retrieval over very long ranges without the out-of-distribution angle problem. This is still a 2025 design choice, not a settled result.



## Activations and the MLP

The MLP holds about two thirds of a dense model's parameters and FLOPs (for Llama-3-8B, 176M of the 218M parameters per layer), so its design matters more than anything about attention, FLOP for FLOP.

### ReLU to GELU

The original FFN was `W2 · ReLU(W1 x)` with `F = 4d`. BERT and GPT-2 switched to GELU ([Hendrycks & Gimpel, arXiv:1606.08415](https://arxiv.org/abs/1606.08415)), `x · Φ(x)`, a smooth ReLU that does not zero all gradients for negative inputs. The gain is modest and empirical. (Squared ReLU from Primer, [arXiv:2109.08668](https://arxiv.org/abs/2109.08668), is a sparse-friendly alternative that comes back now and then.)

### Gated linear units: SwiGLU

[Shazeer, GLU Variants Improve Transformer, arXiv:2002.05202](https://arxiv.org/abs/2002.05202), building on [Dauphin et al., arXiv:1612.08083](https://arxiv.org/abs/1612.08083), replaces the first projection with a gated product:

```python
class SwiGLU(nn.Module):
    def __init__(self, d, F):
        super().__init__()
        self.w_gate = nn.Linear(d, F, bias=False)
        self.w_up   = nn.Linear(d, F, bias=False)
        self.w_down = nn.Linear(F, d, bias=False)
    def forward(self, x):                                       # (B, T, d)
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))  # (B, T, F) -> (B, T, d)
```

`SiLU(x) = x · sigmoid(x)` (Swish, [arXiv:1710.05941](https://arxiv.org/abs/1710.05941)). GEGLU uses GELU in place of SiLU (Gemma uses GeGLU).

**The 2/3 adjustment.** A GLU has *three* `d x F` matrices instead of two. To match a standard `F = 4d` FFN in parameters and FLOPs, set `3·d·F' = 2·d·4d`, so `F' = 8d/3 ≈ 2.67d`. Llama then rounds up to a hardware-friendly multiple: `F = 256 · ceil(int(8d/3) / 256)` gives **11008** for `d = 4096` (Llama-1/2-7B). Llama 3 adds `ffn_dim_multiplier`: `int(8·4096/3) = 10922`, times 1.3 is 14198, rounded up to a multiple of 1024 is **14336** (= 3.5d). The same rule gives 28672 for 70B (x1.3, multiple 4096) and 53248 for 405B (x1.2). All three match the released configs (checked in Python). So modern models are *not* parameter-matched to a 4d ReLU FFN. They spend more on the MLP.

**Evidence:** Shazeer's Table 1 reports C4 log-perplexity at matched parameters and compute, with GEGLU and SwiGLU lowest (GEGLU 1.942 and SwiGLU 1.944 vs ReLU 1.997 after 65K steps, as I recall the table, so verify the digits before quoting them), and gains on GLUE/SuperGLUE fine-tuning. The paper famously offers no explanation ("we attribute their success ... to divine benevolence"). **Cost:** with the 2/3 adjustment, none in FLOPs. The extra kernel (or a fused gate-up GEMM of width `2F` followed by an element-wise op) is cheap. **Users:** PaLM, all Llamas, Mistral, Qwen, DeepSeek (SwiGLU), Gemma (GeGLU). See also [SwiGLU-based MLP sizing](../training/performance/README.md#swiglu-based-mlp) for the tile-friendly dimension choices.



## Attention variants

### MHA to MQA to GQA: the KV cache arithmetic

In MHA every query head has its own K and V head. The KV cache per token is

```
bytes/token = 2 (K and V) · L · KV · hd · bytes_per_elem
```

For Llama-3-70B (`L=80, hd=128`, bf16) the choice of `KV` dominates:

| scheme | KV heads | KV cache / token | at 128K context (1 sequence) |
|---|---|---|---|
| MHA | 64 | 2560 KiB | 320 GiB |
| GQA (actual) | 8 | 320 KiB | 40 GiB |
| MQA | 1 | 40 KiB | 5 GiB |

The four variants side by side:

![MHA, GQA, MQA and MLA](../inference/images/mha-gqa-mqa-mla.png)

([source](https://arxiv.org/abs/2405.04434))

During decode, each step reads all weights once for the whole batch and reads each sequence's full KV cache. With MHA at 128K context, a single sequence's cache (344 GB) is about 2.4x the bf16 weights (141 GB). The cache sets both the maximum batch size and the per-token latency.

**MQA**, [Shazeer, Fast Transformer Decoding: One Write-Head is All You Need, arXiv:1911.02150](https://arxiv.org/abs/1911.02150): all query heads share a single K/V head, which cuts the cache by a factor of `H` and leaves quality only slightly worse in the paper's experiments. Used by PaLM and Falcon. In practice the quality loss and some training instability were noticeable at large scale, and under tensor parallelism a single KV head must be replicated across every TP rank.

**GQA**, [Ainslie et al., GQA, arXiv:2305.13245](https://arxiv.org/abs/2305.13245): `KV` groups, each shared by `H/KV` query heads. The paper also shows you can "uptrain" an MHA checkpoint into GQA with about 5% of the original pretraining compute by mean-pooling heads, and reports GQA-8 close to MHA quality at speed close to MQA on T5-XXL. `KV = 8` became the de facto choice because it divides cleanly over 8-way tensor parallelism: each GPU in a node holds exactly one KV head.

```python
class GQA(nn.Module):
    def __init__(self, d, H, KV, hd):
        super().__init__()
        self.H, self.KV, self.hd = H, KV, hd
        self.wq = nn.Linear(d, H * hd, bias=False)
        self.wkv = nn.Linear(d, 2 * KV * hd, bias=False)
        self.wo = nn.Linear(H * hd, d, bias=False)
    def forward(self, x, pos):                                         # x: (B, T, d)
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.H, self.hd).transpose(1, 2)     # (B, H, T, hd)
        k, v = self.wkv(x).view(B, T, 2, self.KV, self.hd).permute(2, 0, 3, 1, 4)  # each (B, KV, T, hd)
        q, k = rope(q, pos), rope(k, pos)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)  # (B, H, T, hd)
        return self.wo(o.transpose(1, 2).reshape(B, T, self.H * self.hd))
```

(`enable_gqa` needs PyTorch 2.5 or later. Otherwise use `k.repeat_interleave(H // KV, dim=1)`, which materializes the copies.)

**Arithmetic intensity view:** in decode, a GQA kernel loads each K/V head once and uses it for `H/KV` query heads, so FLOPs per byte rise by a factor of `H/KV`. That matters because decode attention is purely bandwidth-bound.

### MLA: Multi-head Latent Attention (DeepSeek-V2/V3)

[DeepSeek-V2, arXiv:2405.04434](https://arxiv.org/abs/2405.04434) takes a different route. Rather than share K/V heads, it caches a **low-rank latent** and reconstructs per-head K/V from it.

MLA and DeepSeekMoE in one block:

![DeepSeek-V2 architecture](images/deepseek-v2-architecture.png)

([source](https://arxiv.org/abs/2405.04434))

```
c_kv = W_DKV · x                      # (B, T, d_c)    d_c = 512   <- this is cached
k_C  = W_UK · c_kv,  v = W_UV · c_kv  # (B, T, H, 128) per-head "content" keys/values
k_R  = RoPE(W_KR · x)                 # (B, T, 64)     shared across heads <- also cached
q    = [q_C ; RoPE(q_R)]              # (B, T, H, 128 + 64), q itself also low-rank (d_q = 1536)
```

**The absorption trick:** for the content part, `q_C · k_C = q_C^T W_UK c_kv = (W_UK^T q_C) · c_kv`. You can fold `W_UK` into the query side, and `W_UV` into the output projection. Attention then runs directly against the 512-d latent and never materializes per-head K/V during decode. It behaves like MQA with a single 576-d (512 + 64) "head" that all 128 query heads share.

**Why the RoPE has to be decoupled:** absorption needs `W_UK` to commute with everything between it and `q`. RoPE inserts a position-dependent rotation `R_n` between them (`q^T R_m^T R_n W_UK c`), so `W_UK` can no longer be pre-multiplied into a fixed matrix. DeepSeek's fix is to carry position in a *separate*, small, RoPE'd key (64-d, shared across heads), concatenated with the un-rotated content key.

**Arithmetic** for DeepSeek-V3 (61 layers, `d_c = 512`, `d_R = 64`, bf16):

```
MLA:          (512 + 64) · 61 · 2 bytes    = 68.6 KiB / token
MHA-equiv.:   61 · 128 heads · (192 K + 128 V) · 2 bytes = 4880 KiB / token
Llama-3-70B GQA-8 (for scale):                 320 KiB / token
```

So a 671B-parameter model has a KV cache per token about 4.7x *smaller* than Llama-3-70B's. **Evidence:** the V2 abstract reports a 93.3% KV cache reduction and 5.76x maximum generation throughput relative to DeepSeek 67B (a dense MHA model, so this mixes the MoE and MLA effects). The V2 paper's appendix ablation reports MLA beating MHA quality at their scale, which is notable because GQA/MQA lose quality relative to MHA (they did not publish many seeds or scales, so treat it as one data point). **Cost:** more complex kernels (FlashMLA and similar). Prefill is more expensive, because without absorption it materializes the full per-head K/V. MLA does not shard naturally over TP by heads, since the latent is shared. And there are two extra projection matmuls. **Users:** DeepSeek-V2/V3/R1, Kimi K2 (uses the DeepSeek-V3 architecture, per its release), and others that build on DeepSeek-V3.

### Sliding-window and local attention

Restrict each token to the last `W` tokens:

```python
def sliding_window_mask(T, W, device=None):
    i = torch.arange(T, device=device)
    return (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < W)   # (T, T) bool, True = attend
```

Sliding-window attention - stacking layers grows the receptive field to roughly `n_layers * W`:

![sliding window attention](images/mistral-sliding-window-attention.png)

([source](https://arxiv.org/abs/2310.06825))

The rolling-buffer KV cache this enables:

![rolling buffer KV cache](images/mistral-rolling-buffer-cache.png)

([source](https://arxiv.org/abs/2310.06825))

FLOPs per token and KV cache per layer become `O(W)` instead of `O(T)`. Stacking `L` layers still gives a receptive field of `L·W` tokens (Mistral: 32 x 4096, about 131K in theory). In practice, information does not propagate perfectly across hops. Local attention goes back to the Sparse Transformer ([Child et al., arXiv:1904.10509](https://arxiv.org/abs/1904.10509)) and Longformer ([arXiv:2004.05150](https://arxiv.org/abs/2004.05150)). GPT-3 alternated dense and locally banded sparse attention layers (GPT-3 paper, Section 2.1).

- **Mistral 7B** ([arXiv:2310.06825](https://arxiv.org/abs/2310.06825)): SWA with `W = 4096` in every layer and a rolling-buffer KV cache of size `W`. The paper reports a 2x speedup at 16K sequence length with `W = 4096` over a vanilla attention baseline, along with the cache reduction. (Later Mistral releases dropped SWA. Mixtral's paper describes full 32K attention.)
- **Gemma 2:** alternates local (`W = 4096`) and global layers 1:1.
- **Gemma 3:** 5 local layers (`W = 1024`) per global layer. For Gemma 3 27B (62 layers, 16 KV heads, `hd = 128`), bf16 KV cache at 128K context is about **62 GiB** if every layer were global versus about **10.4 GiB** with the 5:1 pattern, where only the 10 global layers store full context (my computation). The Gemma 3 report includes ablations showing small perplexity changes from the local:global ratio and window size.
- **gpt-oss** alternates banded (128-token window) and dense layers, per its model card (arXiv:2508.10925, id unverified).

**Tradeoff:** pure SWA degrades on long-range retrieval. Interleaving keeps a few global layers for retrieval and makes the rest cheap. Since only the global layers' KV cache grows with `T`, the "KV heads x global layers" product is the number that matters.

### Attention sinks

[Xiao et al., Efficient Streaming Language Models with Attention Sinks, arXiv:2309.17453](https://arxiv.org/abs/2309.17453) observed that trained LMs put large attention mass on the first few tokens regardless of their content. Softmax must sum to 1, so a head with "nothing to attend to" needs a dump. If you drop those tokens from a sliding window cache, perplexity explodes. Keeping about 4 "sink" tokens plus a rolling window lets the model stream stably (the paper reports up to 4M tokens, and a 22.2x speedup over sliding-window-with-recompute). Architectural versions bake the sink in, either with a learnable per-head logit added to the softmax denominator (gpt-oss, per its model card) or with an "off-by-one" softmax (`exp(x_i) / (1 + Σ exp(x_j))`, Evan Miller's blog post). **Why it matters for systems:** sinks are also linked to massive activations and outlier features that make INT8/FP8 quantization harder. Giving heads an explicit no-op option is partly a quantization-friendliness choice.

### Differential attention (briefly)

[Ye et al., Differential Transformer, arXiv:2410.05258](https://arxiv.org/abs/2410.05258) computes two softmax maps from split Q/K and subtracts them, `(softmax(Q1 K1^T) - λ · softmax(Q2 K2^T)) V`, with a learnable `λ`. The aim is to cancel common-mode "attention noise". The paper reports better scaling and long-context retrieval than a matched Transformer. It needs two attention passes (or a custom kernel) and has not been adopted by a major open model as of this writing. Treat it as promising but unreplicated at frontier scale.



## Mixture of Experts

### What and why

Replace the dense MLP with `E` expert MLPs and a router that sends each token to `k` of them:

```python
class TopKMoE(nn.Module):
    def __init__(self, d, F, E, k):
        super().__init__()
        self.k = k
        self.router = nn.Linear(d, E, bias=False)
        self.experts = nn.ModuleList(SwiGLU(d, F) for _ in range(E))
    def forward(self, x):                                   # x: (B, T, d)
        xf = x.reshape(-1, x.shape[-1])                     # (N, d), N = B*T
        probs = self.router(xf).softmax(-1)                 # (N, E)
        w, idx = probs.topk(self.k, dim=-1)                 # (N, k)
        w = w / w.sum(-1, keepdim=True)                     # renormalize over chosen experts
        out = torch.zeros_like(xf)
        for e, expert in enumerate(self.experts):           # ponytail: loop for clarity; real code = grouped GEMM
            tok, slot = (idx == e).nonzero(as_tuple=True)
            if tok.numel():
                out.index_add_(0, tok, w[tok, slot, None] * expert(xf[tok]))
        return out.view_as(x), probs, idx
```

**The point is that it decouples parameters from FLOPs.** Per token, compute scales with `k · F` while capacity scales with `E · F`. Mixtral 8x7B has **46.7B** total and **12.9B** active parameters (my count from the config: `E = 8`, `k = 2`, `F = 14336`). DeepSeek-V3 has **671B** total and **37B** active (my count from the config gives 671.0B / 37.6B including embeddings, which matches the report). For a fixed training FLOP budget, an MoE reaches lower loss than a dense model. Switch Transformer reports up to 7x pre-training speedup over T5-Base at matched FLOPs.

### Lineage

- [Shazeer et al., Outrageously Large Neural Networks: The Sparsely-Gated MoE Layer, arXiv:1701.06538](https://arxiv.org/abs/1701.06538): noisy top-k gating between LSTM layers, plus the first load-balancing losses.
- [GShard, Lepikhin et al., arXiv:2006.16668](https://arxiv.org/abs/2006.16668): MoE in every other Transformer layer, top-2, expert capacity, and the sharding compiler. This is where **expert parallelism** begins.
- [Switch Transformer, Fedus et al., arXiv:2101.03961](https://arxiv.org/abs/2101.03961): **top-1** routing (simpler, less communication), a capacity factor, bf16 stability tricks (router in fp32), and the standard auxiliary loss.
- [ST-MoE, Zoph et al., arXiv:2202.08906](https://arxiv.org/abs/2202.08906): the **router z-loss** for stability, plus fine-tuning recipes.
- [Mixtral of Experts, arXiv:2401.04088](https://arxiv.org/abs/2401.04088): 8 experts, top-2, every layer. The abstract reports it matches or beats Llama 2 70B and GPT-3.5 on most of the benchmarks shown. It made MoE mainstream in open models.
- [DeepSeekMoE, Dai et al., arXiv:2401.06066](https://arxiv.org/abs/2401.06066): **fine-grained experts** (split each expert into `m` smaller ones and route to `m·k` of them, so the space of combinations grows combinatorially while FLOPs stay the same) and **shared experts** (always on, so routed experts do not each re-learn common knowledge). DeepSeek-V3 uses 1 shared + 256 routed experts with `F = 2048`, routes to 8, and uses dense MLPs in the first 3 layers.
- Qwen3-235B-A22B (128 experts, top-8, **no** shared expert, per the Qwen3 report), Llama 4 (with a shared expert, per Meta's release), and gpt-oss follow the fine-grained trend.

DeepSeek-V3: fine-grained routed experts plus a shared expert:

![DeepSeek-V3 architecture](images/deepseek-v3-architecture.png)

([source](https://arxiv.org/abs/2412.19437))

### Routing

- **Token choice, top-k (standard):** each token picks its `k` best experts. Load is uneven, so experts can overflow. Under a **capacity factor** `C`, each expert takes at most `C · N·k/E` tokens and the overflow is dropped (it passes through the residual). Dropless implementations (MegaBlocks, [arXiv:2211.15841](https://arxiv.org/abs/2211.15841)) use variable-size grouped GEMMs instead.
- **Expert choice**, [Zhou et al., arXiv:2202.09368](https://arxiv.org/abs/2202.09368): each expert picks its top-`c` tokens. Load is perfectly balanced by construction, and tokens get a variable number of experts. The catch is that this is **non-causal**, because an expert's choice depends on other tokens in the batch, including future ones. That is a problem for autoregressive decoding, so it is mostly used in encoders or training-only settings.
- **Gating function:** softmax over all experts then top-k (Mixtral computes the softmax over the selected top-k logits), or **sigmoid** affinities normalized over the selected experts (DeepSeek-V3).

### Load balancing: aux loss vs aux-loss-free

Without balancing, the router collapses onto a few experts (rich-get-richer). The Switch auxiliary loss:

```python
def switch_aux_loss(probs, idx, E):
    # probs: (N, E) router softmax; idx: (N, k) chosen experts
    f = torch.zeros(E, device=probs.device).scatter_add_(0, idx.flatten(), torch.ones(idx.numel(), device=probs.device))
    f = f / idx.numel()                    # fraction of routed slots per expert
    P = probs.mean(0)                      # mean router prob per expert
    return E * (f * P).sum()               # = 1 when perfectly uniform; scaled by alpha ~1e-2
```

The tension is that the aux loss's gradient competes with the LM loss. Set it too high and routing becomes uniform but useless. Set it too low and you get collapse. **DeepSeek-V3's aux-loss-free balancing** ([Wang et al., Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts, arXiv:2408.15664](https://arxiv.org/abs/2408.15664)) adds a per-expert bias `b_e` to the affinities **only for the top-k selection**, not for the gating weights. After each step it nudges `b_e` down for overloaded experts and up for underloaded ones:

```python
def aux_free_select(scores, bias, k):          # scores: (N, E) sigmoid affinities, bias: (E,)
    idx = (scores + bias).topk(k, dim=-1).indices       # bias affects WHO is chosen...
    w = scores.gather(-1, idx); w = w / w.sum(-1, keepdim=True)   # ...not HOW MUCH they count
    return idx, w

def update_bias(bias, idx, E, gamma=1e-3):
    load = torch.bincount(idx.flatten(), minlength=E).float()
    bias += gamma * torch.sign(load.mean() - load)       # push underloaded up, overloaded down
```

V3 still keeps a very small sequence-wise auxiliary loss as a safeguard, plus node-limited routing (each token goes to at most 4 nodes) to bound all-to-all traffic. The 2408.15664 paper reports better perplexity *and* better balance than aux-loss baselines at 1B-3B scale.

### What MoE costs

- **Memory:** you store all `E` experts. Mixtral is 93 GB in bf16 to run a 12.9B-active model. DeepSeek-V3 is about 671 GB in FP8. The serving footprint is set by **total** parameters, and compute by **active** ones.
- **Decode is memory-bound, and MoE makes it worse at small batch.** At batch 1, each token reads `k` experts' weights per layer. With a large batch, the union of experts touched approaches all `E`, so the bytes read per step approach the full model while FLOPs per token stay at the active count. Arithmetic intensity per expert is about `B · k / E` tokens per weight load. DeepSeek-V3 has `k/E = 8/256`, so you need a batch of about 32x larger than a dense model needs to reach the same intensity on expert weights. This is why MoE serving relies on large-scale **expert parallelism** with big global batches (DeepSeek's deployment description in the V3 report uses EP across many GPUs for decode).
- **Communication:** two all-to-alls per MoE layer (dispatch and combine) in both forward and backward. These are latency-sensitive and topology-sensitive (see [Model Parallelism](../training/model-parallelism/README.md#expert-parallelism)).
- **Training instability:** router logits drift. The fixes are a router in fp32, z-loss, and sometimes dense layers early in the stack (DeepSeek-V3's first 3 layers).
- **Fine-tuning and quantization:** routing can shift under fine-tuning, and rarely activated experts get few gradient updates and poorly calibrated quantization statistics.



## Embeddings, heads, vocab and tokenizers

### Tokenizers

- **BPE**, [Sennrich et al., arXiv:1508.07909](https://arxiv.org/abs/1508.07909): repeatedly merge the most frequent pair.
- **Byte-level BPE** (GPT-2): the base alphabet is the 256 bytes, so any input is encodable. There is no `<unk>`, and that is why every modern LLM tokenizer is byte-level or has byte fallback.
- **SentencePiece**, [Kudo & Richardson, arXiv:1808.06226](https://arxiv.org/abs/1808.06226): a library that treats whitespace as a symbol and works on raw text without pre-tokenization. It provides BPE or the **Unigram LM** tokenizer ([Kudo, arXiv:1804.10959](https://arxiv.org/abs/1804.10959)). Llama 1/2 (32K BPE) and Gemma (256K) use SentencePiece. Llama 3 moved to a tiktoken-based 128K BPE.
- Tokenizer-free byte models (ByT5, [arXiv:2105.13626](https://arxiv.org/abs/2105.13626); Byte Latent Transformer, [arXiv:2412.09871](https://arxiv.org/abs/2412.09871)) remove the vocabulary entirely and pay for it with longer sequences.

### What vocab size costs

Embedding parameters are `V · d`, the LM head is another `V · d` unless tied, and the logits cost `2 · V · d` FLOPs per token (forward):

| V | params per matrix at d=4096 | head fwd FLOPs/token |
|---|---|---|
| 32,000 (Llama 2) | 131 M | 0.26 GFLOP |
| 128,256 (Llama 3) | 525 M | 1.05 GFLOP |
| 256,000 (Gemma) | 1,049 M | 2.10 GFLOP |

For Llama-3-8B, embeddings plus head are **1.05B of the 8.03B parameters (13%)**, and the head's 1.05 GFLOP is about 7% of the 16 GFLOP forward. Going from 32K to 128K added 0.79B parameters, which is why "Llama 3 8B" is not "Llama 2 7B" plus GQA.

**What you get back:** a bigger vocab means **fewer tokens per document**. Meta reported that the 128K tokenizer improves compression over Llama 2's (the Llama 3 paper, [arXiv:2407.21783](https://arxiv.org/abs/2407.21783), Section 3.1, gives 3.17 to 3.94 characters per token on an English sample; figures recalled, verify). Fewer tokens per document means less compute per document, a shorter KV cache per document, and effectively longer context. Multilingual coverage improves a lot (Gemma's 256K and Qwen's 152K are partly about non-English scripts). [Tao et al., Scaling Laws with Vocabulary, arXiv:2407.13623](https://arxiv.org/abs/2407.13623) argue that compute-optimal vocab size grows with model size and that most models' vocabularies were too small.

**The systems costs:** the `(B·T, V)` logits tensor is huge. At `B·T = 16K` tokens and `V = 128K` in fp32, it is 8 GiB *before* the softmax backward, so fused/chunked cross-entropy kernels are needed. The embedding gather and the head's all-gather under tensor parallelism get large. The softmax over `V` in decode is a small but non-zero latency cost. For small models the embedding can dominate: Gemma-3-1B or Qwen3-0.6B spend a large fraction of parameters on embeddings.

### Tied embeddings

Share the input embedding with the output projection, `logits = x @ E.T` ([Press & Wolf, arXiv:1608.05859](https://arxiv.org/abs/1608.05859); [Inan et al., arXiv:1611.01462](https://arxiv.org/abs/1611.01462)):

```python
self.tok_emb = nn.Embedding(V, d)
self.lm_head = nn.Linear(d, V, bias=False)
self.lm_head.weight = self.tok_emb.weight      # tied: (V, d), one copy of the parameters
```

**When it pays off:** it saves `V · d` parameters. That is significant when `V · d` is a large share of the model (GPT-2, Gemma at every size, Llama 3.2 1B/3B, Qwen at 0.5B-4B), and negligible at 70B+. **Why large models untie:** the input embedding (a lookup, sparse gradient) and the output head (a dense classifier) want different geometry. Separate matrices cost little at scale. Also, embedding and head usually live on different pipeline stages, so tying forces a cross-stage gradient all-reduce. Llama 2/3 (8B+), Mistral, Qwen 7B+ and DeepSeek are untied. Gemma ties at all sizes and multiplies embeddings by `sqrt(d)`.



## Multi-token prediction

Next-token prediction supervises one step ahead. **Multi-token prediction (MTP)** adds heads that predict `t+2, t+3, ...` from the same trunk.

Meta's multi-token prediction - n independent heads on a shared trunk:

![multi-token prediction](images/multi-token-prediction.png)

([source](https://arxiv.org/abs/2404.19737))

- **Meta MTP**, [Gloeckle et al., Better & Faster Large Language Models via Multi-token Prediction, arXiv:2404.19737](https://arxiv.org/abs/2404.19737): `n` independent output heads on a shared trunk. The abstract reports that a 13B model solves 12% more HumanEval and 17% more MBPP problems than a next-token baseline, that the gains grow with model size (it *hurts* small models), and that inference is up to 3x faster with self-speculative decoding using the extra heads. The memory trick is to compute each head's forward/backward sequentially so that only one `(B·T, V)` logits tensor is alive at a time.
- **DeepSeek-V3 MTP:** **sequential** modules instead of parallel heads. Module `k` combines the trunk's (or previous module's) hidden state at `t` with the *embedding* of token `t+k` (via RMSNorm, concatenation and a linear projection), runs one Transformer block, and predicts `t+k+1` through the **shared** output head and embedding. The causal chain is preserved at each depth. V3 uses depth 1 (predicts one extra token), and the MTP loss is weighted 0.3 then 0.1 over training.

```python
class MTPModule(nn.Module):                       # DeepSeek-V3 style, depth 1
    def __init__(self, d, block, emb, head):
        super().__init__()
        self.n_h, self.n_e = RMSNorm(d), RMSNorm(d)
        self.proj = nn.Linear(2 * d, d, bias=False)
        self.block, self.emb, self.head = block, emb, head   # emb/head shared with main model
    def forward(self, h, next_tokens):            # h: (B, T, d) trunk output; next_tokens: (B, T) = tokens t+1
        z = self.proj(torch.cat([self.n_h(h), self.n_e(self.emb(next_tokens))], -1))  # (B, T, d)
        return self.head(self.block(z))           # (B, T, V): logits for tokens t+2
```

**Why:** a denser training signal (and, per Gloeckle et al., a bias toward learning "planning" features), plus a **free draft model for speculative decoding**. V3's report says the second-token acceptance rate is 85%-90% across generation topics, giving about 1.8x decoding TPS (Section 5.4.3). The approach is in the same family as Medusa ([arXiv:2401.10774](https://arxiv.org/abs/2401.10774)) and EAGLE ([arXiv:2401.15077](https://arxiv.org/abs/2401.15077)), which train draft heads *after* pretraining, and blockwise parallel decoding ([Stern et al., arXiv:1811.03115](https://arxiv.org/abs/1811.03115)). **Cost:** the extra block plus extra logits in training (V3's MTP module adds about 14B parameters on top of the 671B main model). The modules can be discarded at inference if you do not speculate.



## Depth vs width, and counting parameters

### The parameter-count formula for a Llama-style model

Per layer, with `H` query heads, `KV` KV heads, head dim `hd`, FFN size `F`, no biases:

```
attn  = d·(H·hd)        # W_q
      + 2·d·(KV·hd)     # W_k, W_v
      + (H·hd)·d        # W_o
mlp   = 3·d·F           # gate, up, down
norms = 2·d             # two RMSNorm weights
N     = L·(attn + mlp + norms) + d (final norm) + V·d (embed) + V·d (head, if untied)
```

With `H·hd = d` and `KV·hd = d·KV/H`, this simplifies to `attn = 2d²(1 + KV/H)`. For MHA with `F = 4d` and a 2-matrix MLP, it reduces to the classic `12·d²` per layer.

```python
def llama_params(V, d, L, H, KV, F, hd=None, tied=False):
    hd = hd or d // H
    attn = d * H * hd + 2 * d * KV * hd + H * hd * d
    mlp = 3 * d * F
    return L * (attn + mlp + 2 * d) + d + V * d * (1 if tied else 2)

# Llama-3-8B config: vocab 128256, hidden 4096, 32 layers, 32 heads, 8 kv heads, ffn 14336, untied
n = llama_params(128256, 4096, 32, 32, 8, 14336)
print(f"{n:,}")                     # 8,030,261,248
assert n == 8_030_261_248
```

Output: **8,030,261,248**. That matches the commonly reported 8.03B for Llama-3-8B. The breakdown: per layer 41.9M attention + 176.2M MLP, so 6.98B in the 32 layers, plus 1.05B in embedding and head. The same function gives **70.55B** for Llama-3-70B, **405.85B** for Llama-3.1-405B, **7.24B** for Mistral-7B and **6.74B** for Llama-2-7B, all matching the published totals. Qwen2.5-7B comes out at 7.62B once you add its QKV biases, and Qwen3-8B at 8.19B once you add its QK-norm weights.

Two observations follow from the formula:

1. **GQA shifts parameters to the MLP.** With `KV/H = 1/4`, attention is `2.5d²` against the MLP's `3dF = 10.5d²` (Llama 3). About 80% of non-embedding parameters are in the MLP.
2. **Training FLOPs ≈ `6N` per token plus attention**, where `N` should exclude the embedding (a lookup) but include the head. For Llama-3-8B that is about 6 x 7.5B = 45 GFLOP/token at short context. The attention term from the first section adds 13% at 8K.

### Depth vs width

- [Kaplan et al., arXiv:2001.08361](https://arxiv.org/abs/2001.08361) (Section 3.1, Figure 5) found loss depends **weakly** on shape at fixed non-embedding `N`. Aspect ratios `d/L` spanning a wide range (roughly 20 to 200) changed loss by only a few percent. Total parameters dominate.
- [Levine et al., Limits to Depth Efficiencies of Self-Attention, arXiv:2006.12467](https://arxiv.org/abs/2006.12467) give a theory-driven optimal depth that grows roughly logarithmically with width, so very deep and thin is wasteful at large `N`.
- [Tay et al., Scale Efficiently, arXiv:2109.10686](https://arxiv.org/abs/2109.10686) found that shape matters more for *downstream* tasks than for upstream perplexity, and favored "deep-narrow" (DeepNarrow) for T5.
- At small scale, MobileLLM ([arXiv:2402.14905](https://arxiv.org/abs/2402.14905)) argues that deep-and-thin beats wide-and-shallow for sub-1B models. SmolLM and Qwen3-0.6B (28 layers at `d = 1024`) follow this.

In practice, frontier dense models cluster around `d/L ≈ 100-130`: GPT-3 128, Llama-3-8B 128, Mistral-7B 128, Llama-3-70B 102, Llama-3.1-405B 130, DeepSeek-V3 118, Qwen3-8B 114, Gemma-3-27B 87. **The deciding factors are hardware:**

- **Width gives bigger GEMMs**, and so higher MFU and better tensor-core utilization. It also allows more TP shards: `d` and `H` must divide by the TP degree, `KV` should be a multiple of TP (or replicated), and `hd = 128` is the sweet spot for FlashAttention kernels.
- **Depth gives sequential latency.** Each layer is a serial step in decode, and each layer adds a kernel launch and, under TP, two all-reduces. More layers mean more sync points per token. Depth is also what pipeline parallelism shards, so `L` is chosen to divide the PP degree (Llama-3.1-405B's 126 = 16 x 8 - 2 layers: the Llama 3 paper describes removing one layer from the first and last pipeline stages to balance the embedding and head; recalled from the paper, verify the section).
- **Head dim** is fixed at 128 for most models (Gemma 2 9B uses 256). Splitting `d` into more heads is free in FLOPs, but small `hd` hurts kernel efficiency.

See [the model sizing section](../training/performance/README.md#final-recommendations-for-model-sizing) for the divisibility rules.



## Comparison of real configs

Values come from published papers and released `config.json` files, which I am confident about. "?" marks cells I could not confirm. `hd` = head dim. FFN for MoE models is per expert. Context is the released maximum (native/extended where they differ).

| Model | L | d_model | H | KV | hd | FFN | vocab | context | norm | pos-emb | attention | MLP / MoE | tied |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GPT-2 small (124M) | 12 | 768 | 12 | 12 | 64 | 3072 | 50257 | 1024 | pre-LN (LayerNorm) | learned | MHA | GELU, dense | yes |
| GPT-2 XL (1.5B) | 48 | 1600 | 25 | 25 | 64 | 6400 | 50257 | 1024 | pre-LN | learned | MHA | GELU, dense | yes |
| GPT-3 175B | 96 | 12288 | 96 | 96 | 128 | 49152 | 50257 | 2048 | pre-LN | learned | MHA, alternating dense / locally banded sparse | GELU, dense | ? |
| Llama-2-70B | 80 | 8192 | 64 | 8 | 128 | 28672 | 32000 | 4096 | pre-RMSNorm | RoPE θ=1e4 | GQA | SwiGLU | no |
| Llama-3-8B | 32 | 4096 | 32 | 8 | 128 | 14336 | 128256 | 8K (3.1: 128K) | pre-RMSNorm | RoPE θ=5e5 | GQA | SwiGLU | no |
| Llama-3-70B | 80 | 8192 | 64 | 8 | 128 | 28672 | 128256 | 8K (3.1: 128K) | pre-RMSNorm | RoPE θ=5e5 | GQA | SwiGLU | no |
| Llama-3.1-405B | 126 | 16384 | 128 | 8 | 128 | 53248 | 128256 | 128K | pre-RMSNorm | RoPE θ=5e5 + 3.1 scaling | GQA | SwiGLU | no |
| Mistral-7B v0.1 | 32 | 4096 | 32 | 8 | 128 | 14336 | 32000 | 8K? (window 4096) | pre-RMSNorm | RoPE θ=1e4 | GQA + SWA 4096 | SwiGLU | no |
| Mixtral-8x7B | 32 | 4096 | 32 | 8 | 128 | 14336 | 32000 | 32K | pre-RMSNorm | RoPE θ=1e6 | GQA | 8 experts, top-2 | no |
| Qwen2.5-7B | 28 | 3584 | 28 | 4 | 128 | 18944 | 152064 | 32K (128K YaRN) | pre-RMSNorm | RoPE θ=1e6 | GQA, QKV bias | SwiGLU | no |
| Qwen2.5-72B | 80 | 8192 | 64 | 8 | 128 | 29568 | 152064 | 32K (128K YaRN) | pre-RMSNorm | RoPE θ=1e6 | GQA, QKV bias | SwiGLU | no |
| Qwen3-8B | 36 | 4096 | 32 | 8 | 128 | 12288 | 151936 | 32K (128K YaRN) | pre-RMSNorm + QK-norm | RoPE θ=1e6 | GQA, no bias | SwiGLU | no |
| Qwen3-235B-A22B | 94 | 4096 | 64 | 4 | 128 | 1536 | 151936 | 32K (128K YaRN) | pre-RMSNorm + QK-norm | RoPE θ=1e6 | GQA | 128 experts, top-8, no shared | no |
| DeepSeek-V3 (671B / 37B act.) | 61 | 7168 | 128 | MLA (latent 512 + 64 RoPE) | q/k 128+64, v 128 | 18432 dense (first 3 layers); 2048 per expert | 129280 | 128K (YaRN from 4K) | pre-RMSNorm | decoupled RoPE | MLA | 1 shared + 256 routed, top-8, sigmoid, aux-loss-free; MTP depth 1 | no |
| Gemma 2 9B | 42 | 3584 | 16 | 8 | 256 | 14336 | 256000 | 8K | pre+post RMSNorm | RoPE θ=1e4 | GQA, local 4096 / global 1:1, softcap 50/30 | GeGLU | yes |
| Gemma 2 27B | 46 | 4608 | 32 | 16 | 128 | 36864 | 256000 | 8K | pre+post RMSNorm | RoPE θ=1e4 | GQA, local 4096 / global 1:1, softcap 50/30 | GeGLU | yes |
| Gemma 3 27B | 62 | 5376 | 32 | 16 | 128 | 21504 | 262K | 128K | pre+post RMSNorm + QK-norm | RoPE (local 1e4, global 1e6) | GQA, 5 local (1024) : 1 global | GeGLU | yes |

Notes on the table:

- **GPT-3 tied:** the GPT-3 paper does not say whether embeddings are tied. GPT-2's released code ties them.
- **Mistral-7B context:** the v0.1 paper describes `W = 4096`, and the config lists `max_position_embeddings = 32768`. The trained sequence length is ambiguous across sources, hence "?". v0.2 and later removed SWA and use 32K context.
- **DeepSeek-V3 128 heads** applies to the query side. The KV cache holds one shared 576-d latent per layer, which is what "KV" means for MLA.
- **Gemma 3 vocab:** the paper says 262K entries. The config lists 262,208 (so 262,144 plus special tokens; treat the exact digits as unverified).
- Parameter counts I checked against my formula (embedding included): GPT-2 small 124.4M, GPT-2 XL 1.558B, GPT-3 174.6B (the paper's "175B"), Mixtral 46.7B total / 12.9B active, DeepSeek-V3 671.0B / 37.6B (main model, excluding the MTP module).



## Reasoning summary: how the modern default emerged

The 2023-2025 dense default, established by LLaMA and refined by Llama 3, Mistral, Qwen and Gemma:

> decoder-only, pre-RMSNorm (increasingly + QK-norm), RoPE with a large base (+ YaRN-style extension), SwiGLU with `F ≈ 2.7-3.5d`, GQA with 8 KV heads, `hd = 128`, no biases, BPE with 100K-260K vocab, untied embeddings at 7B+ (tied for small models), bf16 training with fp32 norm/softmax statistics.

At frontier scale, add fine-grained MoE with shared experts, MLA or interleaved local/global attention, and multi-token prediction.

Sorting each choice by *why it exists* tells you which ones transfer to a new setting.

| Choice | Primarily about | What it buys | When it might not pay off |
|---|---|---|---|
| Decoder-only | training efficiency + inference | loss on every token, append-only KV cache, one homogeneous stack | encoder tasks (retrieval, classification) still favor bidirectional |
| Pre-LN | **stability** | identity residual path, no warmup dependence | very deep models underuse late layers; post-norm variants recover some quality |
| RMSNorm | simplicity / small efficiency | one fewer reduction, no bias | essentially never worse |
| No biases | **stability** (PaLM) + simplicity | fewer params to shard, fewer spike sources | Qwen2/2.5 kept a QKV bias (the claimed benefit for length extrapolation is from Jianlin Su's blog, unverified); Qwen3 dropped it |
| QK-norm, softcap, z-loss | **stability** at high LR and large scale | bounded attention/output logits | unneeded at small scale and LR; costs little |
| Post-sublayer (sandwich/peri) norm | **stability** | bounded residual updates | mild extra cost; evidence still accumulating |
| RoPE | **quality** + kernel compatibility | relative positions, no `T x T` bias, FlashAttention-friendly | needs scaling (PI/NTK/YaRN) for long context; complicates KV compression |
| Large RoPE base / YaRN | **long context** | cheap context extension | slightly worse local resolution if overdone |
| SwiGLU | **quality** per FLOP (Shazeer 2020) | better perplexity at matched params | none significant |
| `F ≈ 3.5d` (Llama 3) | quality vs hardware balance | MLP GEMMs are the most efficient FLOPs you have | |
| GQA / MQA | **inference memory bandwidth** | 4-8x smaller KV cache, bigger batches, TP-friendly `KV = 8` | small quality loss vs MHA (GQA paper) |
| MLA | **inference memory bandwidth** | about 5x smaller cache than GQA-8 at comparable or better quality (DeepSeek's ablation) | kernel complexity, more expensive prefill, decoupled RoPE |
| Sliding window / local:global | **long-context compute + memory** | KV cache `O(W)` for most layers | long-range retrieval relies on the few global layers |
| MoE | **training FLOPs per quality** | parameters decoupled from FLOPs | serving memory, all-to-all, small-batch decode inefficiency |
| Large vocab | quality (multilingual) + **sequence compression** | fewer tokens per document means less compute and KV | embedding/head params and logits memory, especially in small models |
| Tied embeddings | **parameter budget** at small scale | saves `V·d` | untie at scale; pipeline-stage coupling |
| MTP | quality (large models) + **decode speed** | denser signal, built-in speculative draft | extra training compute; hurts small models (Gloeckle et al.) |
| `hd = 128`, `KV = 8`, `d` multiple of 128 | **hardware** | kernel tiles, TP=8 divisibility | |

Three patterns stand out.

1. **Quality changes are few and old.** Most of the measured per-FLOP quality gain over the 2017 Transformer comes from pre-norm (by making training possible at all), GLU MLPs, RoPE, and MoE. All of them are constant-factor compute savings, and MoE's is the largest (Switch reports up to 7x pre-training speedup at matched FLOPs). Data and compute did most of the rest.
2. **Most post-2022 changes are about decode.** MQA, GQA, MLA, SWA, local:global, large vocab and MTP each reduce bytes moved per generated token, or make more tokens per weight read possible. This follows from training being compute-bound while decode is bandwidth-bound and drives the cost of a deployed model.
3. **Stability changes are bought insurance.** QK-norm, z-loss, no biases, fp32 norms and post-sublayer norms rarely improve the loss of a run that would have trained anyway. They widen the stable learning-rate range (Wortsman et al.) and reduce the chance of losing a multi-million-dollar run. When you scale a recipe up, keep them. When you prototype small, they barely matter. Do not read their absence from a small ablation as evidence that they are unnecessary.

A practical corollary: when a new paper proposes an architecture change, ask which bucket it is in before believing a small-scale result. Quality claims need multiple scales and seeds. Efficiency claims need a measured roofline on real hardware (KV bytes, achieved TFLOPs). Stability claims need learning-rate sweeps, not one run.



## Further reading

Core papers, in rough chapter order:

- Vaswani et al., Attention Is All You Need, [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Bahdanau et al., Neural Machine Translation by Jointly Learning to Align and Translate, [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)
- Wang et al., What Language Model Architecture and Pretraining Objective Work Best for Zero-Shot Generalization?, [arXiv:2204.05832](https://arxiv.org/abs/2204.05832)
- Xiong et al., On Layer Normalization in the Transformer Architecture, [arXiv:2002.04745](https://arxiv.org/abs/2002.04745)
- Wortsman et al., Small-scale proxies for large-scale Transformer training instabilities, [arXiv:2309.14322](https://arxiv.org/abs/2309.14322)
- Su et al., RoFormer, [arXiv:2104.09864](https://arxiv.org/abs/2104.09864); Peng et al., YaRN, [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)
- Shazeer, GLU Variants Improve Transformer, [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)
- Ainslie et al., GQA, [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)
- DeepSeek-V2, [arXiv:2405.04434](https://arxiv.org/abs/2405.04434) (MLA); DeepSeek-V3, [arXiv:2412.19437](https://arxiv.org/abs/2412.19437) (MoE, aux-loss-free balancing, MTP, FP8)
- Fedus et al., Switch Transformer, [arXiv:2101.03961](https://arxiv.org/abs/2101.03961); Dai et al., DeepSeekMoE, [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)
- Gloeckle et al., Multi-token Prediction, [arXiv:2404.19737](https://arxiv.org/abs/2404.19737)
- Llama 3 Herd of Models, [arXiv:2407.21783](https://arxiv.org/abs/2407.21783); Gemma 2, [arXiv:2408.00118](https://arxiv.org/abs/2408.00118); Gemma 3, [arXiv:2503.19786](https://arxiv.org/abs/2503.19786); Qwen2.5, [arXiv:2412.15115](https://arxiv.org/abs/2412.15115); Qwen3, [arXiv:2505.09388](https://arxiv.org/abs/2505.09388); OLMo 2, [arXiv:2501.00656](https://arxiv.org/abs/2501.00656)

Readable code references:

- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) (GPT-2 in about 300 lines) and [llm.c](https://github.com/karpathy/llm.c)
- Meta's reference [llama-models](https://github.com/meta-llama/llama-models) `model.py` (RoPE scaling, GQA, the FFN-size rule)
- Hugging Face `transformers` `modeling_llama.py`, `modeling_mixtral.py`, `modeling_deepseek_v3.py`, `modeling_gemma3.py`: these are the source of truth for the config values above

In this book:

- [Model Architectures II: Beyond the Transformer](beyond-transformers.md): SSMs, linear attention, hybrids, diffusion, multimodal
- [Inference](../inference/README.md): prefill/decode arithmetic and serving engines
- [Model Parallelism](../training/model-parallelism/README.md): TP/PP/EP and how architecture divisibility constrains them
- [Training instabilities](../training/instabilities/): what the stability choices above are guarding against
- [Software Tune Up](../training/performance/README.md): model sizing for hardware efficiency
