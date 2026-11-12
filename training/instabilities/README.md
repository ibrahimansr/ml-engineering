# Avoiding, Recovering From and Understanding Instabilities

Sub-sections:

* [Understanding Training Loss Patterns](training-loss-patterns.md) - types of spikes, divergences, grokking moments, resumes, etc.

## Learning from Training Logbooks

The best learning is to read [Publicly available training LLM/VLM logbooks](../../resources/README.md#publicly-available-training-llmvlm-logbooks) because there you can see exactly what happened and how the problem has been overcome.


## STD Init

Correctly initializing the initial distribution of the tensors can have a tremendous impact on training's stability. The `std` value (the standard deviation of the normal distribution the weights are sampled from) isn't fixed and depends on the hidden dimension size.

This proved to be a very crucial setting in our pre-BLOOM 104B experiments and we couldn't break past the first few thousands iterations until we figured out that the 0.02 default `--init-method-std` (the command-line flag that sets that standard deviation) in [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (NVIDIA's large-model training framework, which BLOOM was trained with via its Megatron-DeepSpeed fork) was a way too big for our model.

We referred to these two sources:

1. "Transformers without Tears" paper https://arxiv.org/abs/1910.05895 prescribes: $\sqrt{2/(\text{NHIDDEN} \cdot 5)}$, where $\text{NHIDDEN}$ is the model's hidden size ($d_\text{model}$)

2. The 530B training paper https://arxiv.org/abs/2201.11990 they used an even smaller init formula: $\sqrt{1/(\text{NHIDDEN} \cdot 3)}$

and decided to go with the 530B one as it leads to an even smaller init value.

To make it easier to compare the two formulas, they can be rewritten as:
1. $\sqrt{0.4000/\text{NHIDDEN}}$
2. $\sqrt{0.3333/\text{NHIDDEN}}$

The smaller the hidden size, the larger the resulting std. Thus for $\text{NHIDDEN}=14336$ the math was $\sqrt{1/(14336 \cdot 3)} = 0.00482$ and that's what we used. It surely wasn't the only reason why we had no stability issues during BLOOM-176B training, but I think it was one of the crucial ones.


## Numerical instabilities

See also [Detecting problematic tensor values](../../debug/pytorch.md#detecting-problematic-tensor-values) and [Underflow and Overflow Detection](../../debug/pytorch.md#underflow-and-overflow-detection) in the debugging chapter (later in the book) for tooling to locate `inf`/`nan` values.

Certain mathematical operations could be unstable when dealing with low precision numbers.

For example, please see this very interesting [PyTorch guide on numerical stability](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html).

Now let's look at a specific example of this concept in action.

During 104B training experiments where fp16 mixed precision was used - the following improvement was proposed by [Corby Rosset](https://github.com/corbyrosset) to make [self-attention more stable](https://github.com/bigscience-workshop/Megatron-DeepSpeed/pull/118).

Specifically this [line](https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/c839a8aa30731f71b3738d56009be9668508e366/megatron/model/transformer.py#L303) shows that the `norm_factor` may be multiplied after the Query * Key matrix multiplication. (`norm_factor` is the $\sqrt{d_\text{head}}$ from scaled dot-product attention $QK^\top/\sqrt{d_\text{head}}$, so applying it means dividing by it.) If the dim of Q and K are very large, the output may blow up and the `norm_factor` won't be able to save it.

Proposal: move the `norm_factor` inward, so Q and K are scaled down before matrix multiply. In the snippet below `torch.baddbmm(input, batch1, batch2, beta, alpha)` computes $\beta \cdot \text{input} + \alpha \cdot (\text{batch1} @ \text{batch2})$ as a batched matmul, so with `beta=0` it is just the scaled matmul (`beta=1.0` is used to add the [ALiBi](../../models/architectures.md#alibi) position bias that was pre-loaded into `matmul_result`). In the shape comments `b` is batch size, `np` the number of attention heads on this GPU, `sq`/`sk` the query/key sequence lengths and `hn` the per-head dimension:
```python
        matmul_result = torch.baddbmm(
            matmul_result,
            1.0/math.sqrt(self.norm_factor) * query_layer.transpose(0, 1),   # [b * np, sq, hn]
            1.0/math.sqrt(self.norm_factor) * key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0 if alibi is None else 1.0, alpha=1.0)

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)
```

To make the operation mathematically equivalent, moving the norm factor inward requires taking sqrt again if $n$ is a scalar, A and B matrices:

$$
n (A \cdot B) \equiv (\sqrt{n} A) \cdot (\sqrt{n} B)
$$

Now A and B dimensions can be significantly larger.

The same post-multiply scaling problem also exists one level down, in the CUDA matmul APIs themselves.

For CUDA kernel writers [CuBlas](https://docs.nvidia.com/cuda/cublas/index.html)'s `GemmStridedBatchedEx` at the time of this writing has a similar issue. It is defined as:

Here $A$, $B$ and $C$ are pointers to the first matrix of each batch and `strideA` etc. are the element offsets between consecutive matrices, so $A+i \cdot \text{strideA}$ is the $i$-th matrix in the batch, and $\operatorname{op}$ is an optional transpose:

$$
C+i \cdot \text{strideC}=\alpha \operatorname{op}(A+i \cdot \text{strideA}) \operatorname{op}(B+i \cdot \text{strideB})+\beta (C+i \cdot \text{strideC}), \quad \text{for } i \in [0,\text{batchCount}-1]
$$

The issue is that `alpha` is multiplied after the matrix-matrix multiplication is done so it can cause instability.

## "Bad" combination of data batch and model parameter state

PaLM team observed dozens of loss spikes at "highly irregular intervals" when training larger models. While they were not able to track down the root cause, they mitigated the issue by restarting from an earlier checkpoint and skipping potentially problematic data batches (per the paper: restarting about 100 steps before the spike and skipping roughly 200-500 data batches). [Section 5.1 Training instability](https://arxiv.org/pdf/2204.02311)


## Time-domain correlation divergence in Adam

[A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871) performs a rigorous study of divergence spikes while training LLMs at up to 546B parameters - and suggests that the time-domain correlation leads to divergence of Adam. Here epsilon is the small constant Adam adds to the denominator $\sqrt{\hat v_t} + \epsilon$ (see [Adam's epsilon](../arithmetic.md#adams-epsilon)). This is triggered by the epsilon value not being small enough and gradient estimation components become similar to the epsilon.

In section 7.1 they propose practical suggestions, the most interesting one of them is setting epsilon to 0 and possibly dealing with division by zero condition.
