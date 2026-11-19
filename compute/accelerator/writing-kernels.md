# Writing GPU Kernels, Hands-On

## Prerequisites

- [GPU execution model](execution-model.md) - SMs, warps, occupancy, coalescing, bank conflicts, tensor cores, GEMM tiling, the FlashAttention derivation, roofline. This chapter does not explain that theory again. It links to the relevant section whenever a step depends on it.
- [Toolkit](../../foundations/toolkit.md) - the "Reading C" section (pointers, arrays, structs, `#include`, compiling with a C compiler) and the Python-environment section (`venv`, `pip`). Every CUDA file here is C plus a few C++ features, which [Just enough C++ to read CUDA](#just-enough-c-to-read-cuda) below adds.
- [Linux and the OS](../../foundations/linux-and-os.md) - the shell, [environment variables](../../foundations/linux-and-os.md#environment-variables-and-path) (`CUDA_LAUNCH_BLOCKING=1`, `TORCH_CUDA_ARCH_LIST`), [shared libraries](../../foundations/linux-and-os.md#shared-libraries-and-ld_library_path) (the `.so` a build produces), and [driver vs runtime vs toolkit](../../foundations/linux-and-os.md#cuda-driver-vs-runtime-vs-toolkit-why-cuda-version-mismatch-happens).
- [Concurrency](../../foundations/concurrency.md) - races, barriers, atomics, and [streams and events](../../foundations/concurrency.md#part-7-gpu-concurrency). A kernel is a few thousand threads racing on shared memory.
- [Computer architecture](../../foundations/computer-architecture.md) - caches, SIMD, floating point, why bytes cost more than FLOPs.
- [Transformer architectures](../../models/architectures.md) - what softmax, RMSNorm, attention, cross-entropy, GQA and [MoE](../../models/architectures.md#mixture-of-experts) layers compute.

Forward links: [GPU software stack](../gpu-software-stack.md) (the next chapter) goes deeper into driver, CUDA toolkit, `nvcc`, PTX vs SASS, and how PyTorch finds CUDA; this chapter defines what it needs of those in place. [Serving engine internals](../../inference/serving-engine-internals.md) is where the inference-side kernels linked from here (paged and split-KV attention, ragged batching, DeepEP) are taught.

[The execution model chapter](execution-model.md) explains why a kernel is fast or slow. This chapter is the workshop. You write kernels from an empty file, test them until they are correct, measure them, and use a profiler to pick the next optimization. It covers the two languages you will meet in any lab's kernel repository: CUDA C++ (full control, more code) and Triton (a Python DSL, or domain-specific language, where you program whole tiles and the compiler maps them to threads). By the end you should be able to read and modify a production kernel repository (FlashAttention, Liger-Kernel, vLLM's `csrc/` and `fused_moe`, ThunderKittens, DeepGEMM), add a kernel with tests and benchmarks, and expose it to `torch.compile`.

Names you will see throughout: *vLLM* and *SGLang* are open-source LLM inference servers (taught in [serving engine internals](../../inference/serving-engine-internals.md)); *Liger-Kernel* is a library of Triton kernels for LLM training; *cuBLAS* is NVIDIA's closed-source matrix-multiply library, which is what `torch.matmul` calls on a GPU and the baseline every GEMM kernel is measured against. A *PR* (pull request) is a proposed change to a shared git repository that maintainers review before merging it, and *CI* (continuous integration) is the machine that automatically runs the repository's tests on every PR ([git basics are in the toolkit chapter](../../foundations/toolkit.md#git-from-zero)).

The plan:

| part | what you build | why it matters |
| :--- | :------------- | :------------- |
| [CUDA C++ from zero](#cuda-c-from-zero) | vector add, a PyTorch extension, a fused bf16 elementwise kernel | launch syntax, indexing, memory, errors, build |
| [Reductions](#reductions) | sum: atomics, smem tree, shuffles, multi-block | every norm, softmax, loss and optimizer has one |
| [Softmax and norms in CUDA](#softmax-rmsnorm-and-layernorm-in-cuda) | online softmax, RMSNorm, Welford LayerNorm | one-pass statistics, mergeable states |
| [GEMM step by step](writing-kernels-reference.md#gemm-step-by-step) | 6 SGEMM kernels, then tensor cores | the most important kernel, and the clearest optimization ladder |
| [Triton from zero](#triton-from-zero) | softmax, autotuned matmul, fused cross-entropy, FlashAttention forward | what most new ML kernels are written in |
| [Grouped GEMM for MoE](writing-kernels-reference.md#grouped-gemm-for-moe) | one launch for all experts | the core of MoE training and serving |
| [Testing](writing-kernels-reference.md#testing-kernels), [benchmarking](writing-kernels-reference.md#benchmarking-kernels), [ncu](writing-kernels-reference.md#reading-ncu-to-choose-the-next-optimization) | a test harness, a timing harness, a profiler workflow | what reviewers check in a kernel PR |
| [Custom ops](#integrating-kernels-as-pytorch-custom-ops) | `torch.library` ops that work with autograd and `torch.compile` | how a kernel becomes usable |
| [The kernel DSL landscape](writing-kernels-reference.md#cutlass-cute-thunderkittens-and-other-kernel-dsls) | CUTLASS/CuTe, ThunderKittens, TileLang, Gluon, ... | what you will see in other repositories |


## How to run this chapter without a GPU

This chapter was written on an Apple Silicon Mac, which has no NVIDIA GPU. Here is how each kind of code was checked. Every code block says which kind it is.

- **Triton kernels: run in a simulator.** Triton has no macOS wheels (`pip download triton` on this machine: `ERROR: No matching distribution found for triton`), so even `TRITON_INTERPRET=1` (Triton's built-in CPU interpreter, Linux only) was unavailable. Instead, [`tsim.py`](#tsimpy-a-cpu-simulator-of-the-triton-model) below is a ~130-line CPU simulator of the subset of `triton.language` this chapter uses, built on CPU PyTorch. The kernel bodies are ordinary Triton source. Only the import line changes, through a 10-line `backend.py`. With an NVIDIA GPU and real Triton, the same files run on the GPU unchanged. **The pasted outputs come from the simulator. They show the algorithm and indexing are correct. They say nothing about speed, and the kernels have not been compiled by the real Triton compiler.** Real Triton enforces constraints the simulator does not (for example power-of-2 block shapes, `tl.dot` needing tiles of at least 16, and types of loop-carried values), so treat the Triton code as "algorithm verified, compile untested".
- **CUDA C++: syntax-checked, never run.** No `nvcc` here. Every `.cu` file was type-checked with `clang++ -x cuda -fsyntax-only --cuda-gpu-arch=sm_90` against hand-written stub headers. That catches typos, type errors and bad template instantiations, but it is not the real CUDA headers and not a compile. The index arithmetic of the SGEMM kernels and the reduction trees was checked by [mirroring them thread by thread in Python](writing-kernels-reference.md#checking-the-index-arithmetic-without-a-gpu). Running them on a GPU is untested.
- **PyTorch integration: run on CPU.** `torch.utils.cpp_extension.load_inline` (with a C++ CPU function), `torch.library.custom_op`, `opcheck` and `torch.compile(fullgraph=True)` all work on a Mac and were run.

Setup, all inside `/tmp` (virtual environments and `pip` are covered in the [toolkit chapter](../../foundations/toolkit.md#running-the-books-code); `ninja` is a small, fast build tool, a cousin of Make, that PyTorch uses to drive the C++ compiler):

```bash
python3 -m venv /tmp/mle-venv
/tmp/mle-venv/bin/pip install torch numpy ninja      # torch 2.15.0, numpy 2.5.4; ninja for cpp_extension
mkdir -p /tmp/mle-kernels && cd /tmp/mle-kernels      # save each file below under its name
/tmp/mle-venv/bin/python softmax.py
```

### `backend.py`: one import line for GPU and simulator

```python
# backend.py - real Triton on an NVIDIA GPU, the tsim simulator otherwise. Same kernel source.
import torch
if torch.cuda.is_available():
    import triton
    import triton.language as tl
    DEV = "cuda"
else:
    import tsim as triton
    tl = triton.language
    DEV = "cpu"
```

### `tsim.py`: a CPU simulator of the Triton model

Read this after [the Triton programming model](#the-programming-model) if you want to know exactly what the simulator does. It does what Triton's interpreter does: it runs each program instance one at a time on the CPU, treats every `tl.*` op as a whole-tile tensor op, and represents a pointer as "base buffer + tensor of offsets". Two choices make it more useful for debugging than a naive loop. Programs run in a **shuffled order**, because real programs run in no defined order and a kernel that depends on order is wrong. Every load and store is **bounds-checked against the tensor's storage**, so a missing mask raises an error instead of quietly reading a neighbour's memory, which is what usually happens on a GPU.

```python
# tsim.py - a tiny CPU simulator of the Triton programming model, built on CPU PyTorch.
# Kernel bodies written against it are plain Triton; only the import line changes:
#     import tsim as triton; tl = triton.language      # instead of: import triton, triton.language as tl
# It runs one program instance at a time, sequentially, like TRITON_INTERPRET=1 does.
# Note: covers the subset of tl.* used in this chapter; add ops as kernels need them.
import itertools, random, types
import torch


class _Dtype:                                   # what `ptr.dtype.element_ty` returns
    def __init__(self, t): self.element_ty = t


class Ptr:
    """A pointer tensor: one flat storage buffer + a tensor of element offsets into it."""
    def __init__(self, buf, offs):
        self.buf, self.offs = buf, offs
    def __add__(self, o): return Ptr(self.buf, self.offs + o)
    __radd__ = __add__
    def __sub__(self, o): return Ptr(self.buf, self.offs - o)
    @property
    def dtype(self): return _Dtype(self.buf.dtype)


def _as_ptr(t):
    # view the whole underlying storage as 1-D so strided / offset tensors behave like raw pointers
    n = t.untyped_storage().nbytes() // t.element_size()
    flat = torch.as_strided(t, (n,), (1,), 0)
    return Ptr(flat, torch.tensor(t.storage_offset()))


def _check(p, mask):
    offs = p.offs if mask is None else p.offs[torch.broadcast_to(torch.as_tensor(mask), p.offs.shape)]
    if offs.numel() and (offs.min() < 0 or offs.max() >= p.buf.numel()):
        raise IndexError(f"out-of-bounds access (offset {int(offs.max())} >= {p.buf.numel()}); missing mask?")


def load(p, mask=None, other=0.0):
    _check(p, mask)
    idx = p.offs.clamp(0, p.buf.numel() - 1)
    x = p.buf[idx]
    if mask is None:
        return x
    return torch.where(torch.as_tensor(mask), x, torch.as_tensor(other, dtype=x.dtype))


def store(p, val, mask=None):
    _check(p, mask)
    val = torch.broadcast_to(torch.as_tensor(val).to(p.buf.dtype), p.offs.shape)
    m = torch.ones_like(p.offs, dtype=torch.bool) if mask is None else torch.broadcast_to(torch.as_tensor(mask), p.offs.shape)
    p.buf[p.offs[m]] = val[m]


def atomic_add(p, val, mask=None):
    _check(p, mask)
    val = torch.broadcast_to(torch.as_tensor(val).to(p.buf.dtype), p.offs.shape)
    m = torch.ones_like(p.offs, dtype=torch.bool) if mask is None else torch.broadcast_to(torch.as_tensor(mask), p.offs.shape)
    p.buf.index_add_(0, p.offs[m].reshape(-1), val[m].reshape(-1))


_rng = random.Random(0)
_pid = [0, 0, 0]
_grid = [1, 1, 1]
T = torch.as_tensor
language = types.SimpleNamespace(
    constexpr=int, float32=torch.float32, float16=torch.float16, bfloat16=torch.bfloat16,
    int32=torch.int32, int64=torch.int64,
    program_id=lambda axis: _pid[axis],
    num_programs=lambda axis: _grid[axis],
    arange=lambda a, b: torch.arange(a, b),
    zeros=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
    full=lambda shape, v, dtype: torch.full(shape, v, dtype=dtype),
    load=load, store=store, atomic_add=atomic_add,
    sum=lambda x, axis=None: x.sum() if axis is None else x.sum(axis),
    max=lambda x, axis=None: x.max() if axis is None else x.amax(axis),
    min=lambda x, axis=None: x.min() if axis is None else x.amin(axis),
    argmax=lambda x, axis: x.argmax(axis),
    exp=lambda x: torch.exp(T(x)), exp2=lambda x: torch.exp2(T(x)), log=lambda x: torch.log(T(x)),
    sqrt=lambda x: torch.sqrt(T(x)), rsqrt=lambda x: torch.rsqrt(T(x)),
    maximum=lambda a, b: torch.maximum(T(a), T(b)), minimum=lambda a, b: torch.minimum(T(a), T(b)),
    where=lambda c, a, b: torch.where(T(c), T(a), T(b)),
    # tl.dot: tensor-core MMA, fp32 accumulate; inputs are fed at their own precision
    dot=lambda a, b, acc=None: (a.float() @ b.float()) + (0 if acc is None else acc),
    trans=lambda x: x.T,
    cdiv=lambda a, b: (a + b - 1) // b,
    multiple_of=lambda x, _: x, max_contiguous=lambda x, _: x,   # compiler hints: no-ops here
    static_range=range, range=range,
)


def cdiv(a, b): return (a + b - 1) // b
def next_power_of_2(n): return 1 << (n - 1).bit_length()


class Config:
    def __init__(self, kwargs, num_warps=4, num_stages=2):
        self.kwargs, self.num_warps, self.num_stages = kwargs, num_warps, num_stages


class _Kernel:
    def __init__(self, fn, configs=None): self.fn, self.configs, self.calls = fn, configs, 0
    def __getitem__(self, grid):
        def launch(*args, **kw):
            kw.pop("num_warps", None); kw.pop("num_stages", None)   # meaningless on a CPU
            if self.configs:   # "autotune": each call uses the next config, so a test loop covers all
                kw = {**self.configs[self.calls % len(self.configs)].kwargs, **kw}   # (timing here means nothing)
                self.calls += 1
            g = grid(kw) if callable(grid) else grid
            g = tuple(g) + (1,) * (3 - len(g))
            _grid[:] = g
            args = [_as_ptr(a) if isinstance(a, torch.Tensor) else a for a in args]
            pids = list(itertools.product(*(range(n) for n in g)))
            _rng.shuffle(pids)            # real programs run in no particular order; so do these
            for pid in pids:
                _pid[:] = pid
                self.fn(*args, **kw)
        return launch


def jit(fn): return _Kernel(fn)


def autotune(configs, key, **_):
    return lambda k: _Kernel(k.fn, configs)
```


## CUDA C++ from zero

### Just enough C++ to read CUDA

CUDA C++ is C (as read in the [toolkit chapter](../../foundations/toolkit.md#reading-c)'s "Reading C" section) plus some C++ features and a handful of GPU keywords. You do not need to write general C++ to follow this chapter. You need to recognize these features when they appear:

- **`a::b`** is a name `b` inside a *namespace* `a`, which is how C++ groups names, much like a Python module (`torch::Tensor`, `at::cuda::getCurrentCUDAStream`, `wmma::fragment`). `using namespace nvcuda;` makes the names in `nvcuda` usable without the prefix, like `from nvcuda import *`.
- **Templates.** `template <typename T, int N>` before a function or struct makes it generic over a type `T` and a compile-time integer `N`. You pick the values in angle brackets at the call site, e.g. `sgemm_smem<32><<<grid, block>>>(...)` means "the version of `sgemm_smem` with `TILE = 32`". The compiler generates separate machine code for each choice, so `N` is a constant inside the function, and loops over it can be fully unrolled. That is why tile sizes are template parameters.
- **`auto x = ...;`** lets the compiler infer the type. **`constexpr`** marks a value computed at compile time. **`static_assert(cond, "msg")`** is a check the compiler runs while compiling; it costs nothing at run time.
- **Structs with functions.** A C++ `struct` can contain functions. `operator()` makes a struct callable like a function, so `struct Sum { float operator()(float a, float b) const { return a + b; } };` can be passed as an argument and then called as `merge(a, b)`. `Sum{}` creates one; `MD{x, 1.f}` builds a struct from its fields in order (*brace initialization*).
- **`reinterpret_cast<T*>(p)`** reinterprets the bytes at pointer `p` as type `T`, with no conversion. It is how the code treats 8 adjacent bf16 values as one 16-byte `uint4` to load them in one instruction. `(float)i` is the ordinary C cast, which does convert.
- **Preprocessor lines** start with `#` and are processed before compilation. `#include "file.cuh"` pastes a file in (`.cuh` is the CUDA header-file convention). `#pragma once` at the top of a header says "include this at most once". `#pragma unroll` asks the compiler to unroll the next loop. `#define NAME(args) ...` defines a *macro*, a text substitution; a trailing `\` continues it on the next line, and the `do { ... } while (0)` wrapper in `CUDA_CHECK` is the standard trick that makes a multi-statement macro behave like one statement. `__FILE__` and `__LINE__` expand to the current file name and line number.
- **Small C syntax that Python lacks:** `cond ? a : b` is Python's `a if cond else b`; `1.f` and `0.5f` are `float` literals (`1.0` would be a `double`); `++i` adds one; `x >>= 1` halves an integer; `0xffffffffu` is an unsigned hexadecimal constant (all 32 bits set). Bitwise `&`, `|`, `^`, `<<`, `>>` mean the same as in Python.
- **`asm volatile("...")`** embeds raw assembly (here PTX, the GPU's assembly language) inside C++; the `"=f"(d[0])` and `"r"(a[0])` lists say which C++ variables feed each `%0, %1, ...` placeholder. You will only read this, in the tensor core section.
- **CUDA's built-in vector types**: `float4` is a struct of four floats with fields `.x .y .z .w` (16 bytes), `uint4` is four 32-bit unsigned integers (also 16 bytes, used as a raw 16-byte chunk), `dim3` and `uint3` are three integers `.x .y .z` (a `dim3` you do not fill in defaults to 1 in the missing axes), and `make_float4(a, b, c, d)` builds a `float4`. `half` and `__nv_bfloat16` are the fp16 and bf16 types; `__bfloat162float` and `__float2bfloat16` convert to and from `float`.

The same features in a plain C++ program you can compile on any machine with a C++ compiler (run on this Mac with Apple clang 17; the GPU keywords are left out):

```cpp
// cpp_tour.cpp - the C++ features this chapter's CUDA code uses, in plain C++ (no GPU)
// build: clang++ -std=c++17 -o cpp_tour cpp_tour.cpp && ./cpp_tour     (g++ works too)
#include <cstdio>
#include <cstdint>

#define CHECK(cond)                                         \
  do {                                                      \
    if (!(cond)) { printf("failed at line %d\n", __LINE__); return 1; } \
  } while (0)

namespace demo {                                  // demo::sum is the full name
template <typename T, int N>                      // a type and an int fixed at compile time
T sum(const T* x) {
  static_assert(N > 0, "N must be positive");      // checked by the compiler, costs nothing at run time
  T acc = 0;
#pragma unroll                                     // a hint; plain C++ compilers may ignore it
  for (int i = 0; i < N; ++i) acc += x[i];
  return acc;
}
}  // namespace demo

struct Max {                                      // a "function object": a struct you can call
  float operator()(float a, float b) const { return a > b ? a : b; }
};

struct Pair { float m, d; };

int main() {
  float x[4] = {1.f, 2.f, 3.f, 4.f};              // 1.f is a float literal (1.0 would be a double)
  constexpr int n = 4;                             // compile-time constant
  auto s = demo::sum<float, n>(x);                 // auto: the compiler infers the type (float)
  float m = Max{}(3.f, 7.f);                       // build a Max with {}, then call it
  Pair p{m, s};                                    // brace initialization of a struct
  uint32_t bits = *reinterpret_cast<uint32_t*>(&p.d);   // view the same 4 bytes as another type
  printf("sum=%.1f max=%.1f bits=0x%08x\n", p.d, p.m, bits);
  CHECK(s == 10.f && m == 7.f);
  return 0;
}
```

```
sum=10.0 max=7.0 bits=0x41200000
```

`0x41200000` is the IEEE 754 bit pattern of `10.0f` ([floating point](../../foundations/computer-architecture.md#floating-point-ieee-754)): `reinterpret_cast` changed how the bytes are read, not the bytes.

### The anatomy of a kernel

A *kernel* is a C++ function marked `__global__`. The host (CPU) launches it and the device (GPU) runs it once per thread. The launch syntax `kernel<<<grid, block, smem_bytes, stream>>>(args...)` starts `grid.x * grid.y * grid.z` *blocks* (CTAs) of `block.x * block.y * block.z` threads each. Every thread runs the same function and finds out who it is from four built-in variables:

| variable | type | meaning |
| :------- | :--- | :------ |
| `threadIdx` | `uint3` | this thread's coordinates within its block |
| `blockIdx` | `uint3` | this block's coordinates within the grid |
| `blockDim` | `dim3` | the block shape (threads per block, per axis) |
| `gridDim` | `dim3` | the grid shape (blocks per axis) |

The other qualifiers you will see: `__device__` is a function callable only from GPU code (inlined helpers). `__host__ __device__` compiles for both sides. `__shared__` declares a variable in the block's shared memory (smem), visible to all threads of the block. `__restrict__` promises the compiler that pointers do not alias, which lets it use the read-only cache and reorder loads. `__syncthreads()` is a barrier for all threads of one block (there is no barrier across blocks inside a kernel, apart from *cooperative launches*, a special launch mode, `cudaLaunchCooperativeKernel`, that guarantees all blocks are resident at once so a grid-wide barrier cannot deadlock). `__forceinline__` forces a `__device__` helper to be inlined, and `__launch_bounds__(N)` on a kernel promises it is never launched with more than `N` threads per block, so the compiler can budget registers for that size ([occupancy](execution-model.md#occupancy-worked)).

Limits worth memorizing (from the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html), compute capability 7.0 to 10.x): at most 1024 threads per block; `gridDim.x` up to $2^{31} - 1$; `gridDim.y` and `gridDim.z` up to 65535; static `__shared__` up to 48 KB per block (more needs dynamic shared memory plus `cudaFuncSetAttribute`). Blocks should be a multiple of 32 threads, because the hardware runs 32-thread warps ([why](execution-model.md#sms-warps-and-schedulers)).

The one line every elementwise kernel starts with is the global index, plus the check for the partial last block:

```python
# indexing.py - what `int i = blockIdx.x * blockDim.x + threadIdx.x` computes, for n = 1000, 256 threads/block
n, threads = 1000, 256
blocks = (n + threads - 1) // threads              # ceil-divide: 4 blocks = 1024 threads >= n
for b in range(blocks):
    ids = [b * threads + t for t in range(threads)]
    live = sum(i < n for i in ids)
    print(f"block {b}: i = {ids[0]:4d}..{ids[-1]:4d}  threads doing work: {live}")
```

```
block 0: i =    0.. 255  threads doing work: 256
block 1: i =  256.. 511  threads doing work: 256
block 2: i =  512.. 767  threads doing work: 256
block 3: i =  768..1023  threads doing work: 232
```

Without the `if (i < n)`, the last 24 threads write past the end of the array. The GPU usually does not crash on that: it scribbles over whatever the caching allocator placed next. This is the most common kernel bug, and it is why [testing](writing-kernels-reference.md#testing-kernels) uses sizes that are not multiples of the block size.

### A complete program: memory, launch, errors, timing

Device memory is a separate address space. `cudaMalloc` returns a pointer the CPU must not dereference, and `cudaMemcpy` moves bytes between the two. Every CUDA runtime call returns a `cudaError_t`. Kernel launches return nothing, so you check them in two places: `cudaGetLastError()` right after the launch catches *launch* errors (too many threads per block, too much shared memory), and the next synchronizing call (`cudaDeviceSynchronize`, `cudaEventSynchronize`, a blocking `cudaMemcpy`) reports *execution* errors (illegal address, misaligned access). Because execution is asynchronous, an error reported at line 80 may come from a kernel launched at line 40. Set `CUDA_LAUNCH_BLOCKING=1` to make every launch synchronous while you debug.

```cuda
// vadd.cu - a complete CUDA program: allocate, copy, launch, check errors, time, verify
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                              \
  do {                                                                                \
    cudaError_t err_ = (call);                                                        \
    if (err_ != cudaSuccess) {                                                        \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_),         \
              __FILE__, __LINE__, cudaGetErrorString(err_));                          \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

// __global__: runs on the GPU, launched from the host. One thread per element.
__global__ void vadd(const float* __restrict__ a, const float* __restrict__ b,
                     float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;   // global thread index
  if (i < n) c[i] = a[i] + b[i];                   // the last block is partial
}

int main() {
  const int n = 1 << 24;                           // 16M floats = 64 MB per array
  const size_t bytes = (size_t)n * sizeof(float);
  float* ha = (float*)malloc(bytes);
  float* hb = (float*)malloc(bytes);
  float* hc = (float*)malloc(bytes);
  for (int i = 0; i < n; ++i) { ha[i] = (float)i; hb[i] = 2.0f * i; }

  float *da, *db, *dc;                             // device pointers: not dereferenceable on host
  CUDA_CHECK(cudaMalloc(&da, bytes));
  CUDA_CHECK(cudaMalloc(&db, bytes));
  CUDA_CHECK(cudaMalloc(&dc, bytes));
  CUDA_CHECK(cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice));

  const int threads = 256;                         // per block: a multiple of 32 (warp size)
  const int blocks = (n + threads - 1) / threads;  // ceil-divide so every element is covered
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  vadd<<<blocks, threads>>>(da, db, dc, n);        // warmup (first launch loads the module)
  CUDA_CHECK(cudaGetLastError());                  // catches bad launch configs immediately
  CUDA_CHECK(cudaEventRecord(start));
  vadd<<<blocks, threads>>>(da, db, dc, n);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(stop));
  CUDA_CHECK(cudaEventSynchronize(stop));          // faults inside the kernel surface here
  float ms = 0.f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

  CUDA_CHECK(cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; ++i) bad += (hc[i] != 3.0f * i);
  printf("n=%d  %.3f ms  %.0f GB/s  mismatches=%d\n", n, ms, 3.0 * bytes / (ms * 1e6), bad);

  CUDA_CHECK(cudaFree(da)); CUDA_CHECK(cudaFree(db)); CUDA_CHECK(cudaFree(dc));
  free(ha); free(hb); free(hc);
  return bad != 0;
}
```

Build and run (untested here; expected output sketched, numbers depend on the GPU). `nvcc` is NVIDIA's compiler driver, used like the C compiler in the toolkit chapter: `-O3` optimizes, `-o vadd` names the output program, `-arch=sm_90` picks the GPU generation to generate machine code for, and `-lineinfo` keeps a map from machine code back to source lines for the profiler (more on all of these in the next section):

```bash
nvcc -O3 -arch=sm_90 -lineinfo -o vadd vadd.cu     # sm_90 = H100; sm_80 = A100; sm_100 = B200
./vadd
# n=16777216  0.07 ms  2900 GB/s  mismatches=0        <- sketch: 3 x 64 MB moved, near HBM peak on H100
```

The arithmetic behind that sketched line, so you can tell whether your own number makes sense:

```python
# vadd_floor.py - the floor for vadd.cu: 3 arrays of 2^24 fp32 at peak HBM bandwidth
n = 1 << 24
bytes_moved = 3 * n * 4                       # read a, read b, write c (fp32)
for gpu, bw in [("A100 80GB SXM", 2.039e12), ("H100 SXM", 3.35e12)]:
    print(f"{gpu:14s} ideal time {bytes_moved / bw * 1e3:.3f} ms at peak {bw/1e12:.2f} TB/s")
```

```
A100 80GB SXM  ideal time 0.099 ms at peak 2.04 TB/s
H100 SXM       ideal time 0.060 ms at peak 3.35 TB/s
```

Unified memory (`cudaMallocManaged`) gives one pointer valid on both sides, with pages migrating on demand. It is convenient for prototypes. In ML code you rarely touch either API, because PyTorch's caching allocator owns device memory and you receive pointers from tensors, as shown next.

### Compiling: nvcc, PTX, SASS

`nvcc` splits a `.cu` file into host code (passed to the system C++ compiler) and device code. Device code is compiled to *PTX* (a virtual, forward-compatible ISA) and then by `ptxas` to *SASS* (the real machine code of one architecture). The executable embeds a *fatbinary* with SASS for each `-arch`/`-gencode` target and optionally PTX, which the driver can JIT-compile for newer GPUs. Details are in [the software stack chapter](../gpu-software-stack.md), which comes next; this section is enough to build the examples here.

```bash
# SASS for H100 + PTX so future GPUs can JIT it
nvcc -O3 -lineinfo -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 -o vadd vadd.cu
# wgmma / setmaxnreg and other Hopper-only features need the "a" (architecture-specific) target:
nvcc -O3 -arch=sm_90a ...
# per-kernel registers, smem and spills (read this for every kernel you write)
nvcc -O3 -arch=sm_90 --ptxas-options=-v -c vadd.cu
#   ptxas info    : Used 10 registers, 380 bytes cmem[0]          <- sketch
# look at the machine code
cuobjdump -sass vadd | grep -E "LDG|STG|FADD" | head
#   LDG.E.CONSTANT R2, desc[UR4][R2.64] ; FADD R9, R4, R7 ; STG.E desc[UR4][R6.64], R9   <- sketch
```

`-lineinfo` keeps the source-line mapping (no slowdown) so that Nsight Compute can attribute stalls to source lines. `-G` builds debug device code, which is very slow; use it only with `cuda-gdb` (NVIDIA's GPU-aware version of [gdb](../../foundations/linux-and-os.md#gdb-basics)). `--use_fast_math` replaces `expf`/division with approximate versions and flushes denormals ([subnormals](../../foundations/computer-architecture.md#floating-point-ieee-754)) to zero. Inside a kernel you can make that choice per call with intrinsics such as `__expf`.

### Loading a kernel into PyTorch: `load_inline`

`torch.utils.cpp_extension.load_inline` writes your source strings to a build directory, generates a `build.ninja`, compiles with `nvcc`/`c++`, links against libtorch, and imports the result as a Python module. If you pass `functions=[...]`, it also generates the pybind11 bindings (pybind11 is the C++ library that wraps a C++ function so Python can call it). libtorch is PyTorch's C++ library; `torch::Tensor` is the C++ type behind a Python `torch.Tensor`, and `#include <torch/extension.h>` brings in all of it. It caches by a hash of the sources, so the second run is instant. The mechanics are the same for CPU-only C++, so this part runs on a Mac:

```python
# inline_cpu.py - the load_inline mechanics with a C++ CPU function (runs on a Mac; no CUDA needed)
import time, torch
from torch.utils.cpp_extension import load_inline

cpp = r"""
#include <torch/extension.h>
torch::Tensor scale_add(torch::Tensor x, torch::Tensor y, double alpha) {
  TORCH_CHECK(x.sizes() == y.sizes(), "shape mismatch");
  TORCH_CHECK(x.scalar_type() == torch::kFloat && y.scalar_type() == torch::kFloat, "float32 only");
  auto xc = x.contiguous(), yc = y.contiguous();
  auto out = torch::empty_like(xc);
  const float* xp = xc.data_ptr<float>(); const float* yp = yc.data_ptr<float>();
  float* op = out.data_ptr<float>();
  for (int64_t i = 0; i < xc.numel(); ++i) op[i] = alpha * xp[i] + yp[i];
  return out;
}
"""
t0 = time.time()
ext = load_inline(name="scale_add_ext", cpp_sources=cpp, functions=["scale_add"],
                  extra_cflags=["-O3"], verbose=False)
print(f"built + loaded in {time.time() - t0:.1f}s ->", ext.scale_add.__doc__.splitlines()[0])
x, y = torch.randn(1000), torch.randn(1000)
torch.testing.assert_close(ext.scale_add(x, y, 2.0), 2.0 * x + y)
print("scale_add OK")
try:
    ext.scale_add(x, y[:10], 2.0)
except RuntimeError as e:
    print("TORCH_CHECK ->", str(e).splitlines()[0])
```

```
built + loaded in 4.6s -> scale_add(arg0: torch.Tensor, arg1: torch.Tensor, arg2: typing.SupportsFloat | typing.SupportsIndex) -> torch.Tensor
scale_add OK
TORCH_CHECK -> shape mismatch
```

The first run compiles (the time above, with an empty cache). A second run prints `built + loaded in 0.0s`, because the build directory (`~/.cache/torch_extensions/<py>_<cuda>/<name>` on Linux, `~/Library/Caches/torch_extensions/...` here; override with `TORCH_EXTENSIONS_DIR`) holds the `.so` (the compiled [shared library](../../foundations/linux-and-os.md#shared-libraries-and-ld_library_path) that Python imports). `TORCH_CHECK` raises a Python `RuntimeError` with your message. Use it for every precondition your kernel assumes.

The CUDA version (untested). Three details separate a toy from a correct extension: launch on PyTorch's *current stream*, not the default stream, otherwise your kernel is unordered with respect to the surrounding PyTorch ops; set the *device guard* so a tensor on `cuda:1` launches on GPU 1; and check the launch:

```python
# vadd_ext.py - CUDA kernel as a PyTorch function (untested: needs an NVIDIA GPU + CUDA toolkit)
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

__global__ void vadd_kernel(const float* __restrict__ a, const float* __restrict__ b,
                            float* __restrict__ c, int64_t n) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;   // int64: tensors can exceed 2^31 elements
  if (i < n) c[i] = a[i] + b[i];
}

torch::Tensor vadd(torch::Tensor a, torch::Tensor b) {
  TORCH_CHECK(a.is_cuda() && b.is_cuda(), "CUDA tensors expected");
  TORCH_CHECK(a.scalar_type() == torch::kFloat32 && b.scalar_type() == torch::kFloat32, "float32 expected");
  TORCH_CHECK(a.sizes() == b.sizes(), "shape mismatch");
  const at::cuda::OptionalCUDAGuard guard(a.device());
  auto ac = a.contiguous(), bc = b.contiguous();      // kernel assumes dense, unit-stride memory
  auto c = torch::empty_like(ac);
  const int64_t n = ac.numel();
  if (n == 0) return c;                               // a 0-block launch is an error
  const int threads = 256;
  const unsigned blocks = (unsigned)((n + threads - 1) / threads);
  vadd_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
      ac.data_ptr<float>(), bc.data_ptr<float>(), c.data_ptr<float>(), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();                     // cudaGetLastError -> C++ exception -> Python
  return c;
}
"""
ext = load_inline(name="vadd_ext",
                  cpp_sources="torch::Tensor vadd(torch::Tensor a, torch::Tensor b);",   # declaration for bindings
                  cuda_sources=cuda_src, functions=["vadd"],
                  extra_cuda_cflags=["-O3", "-lineinfo"], verbose=True)
x, y = torch.randn(10_000_001, device="cuda"), torch.randn(10_000_001, device="cuda")
torch.testing.assert_close(ext.vadd(x, y), x + y)
```

`TORCH_CUDA_ARCH_LIST="9.0"` (environment variable) selects the architectures to build for; by default it builds for the GPUs it sees. For a real repository you would switch to `torch.utils.cpp_extension.CUDAExtension` in `setup.py` (the classic file that tells `pip install` how to build a package with compiled parts), or to a CMake build (CMake is a widely used generator of C/C++ build scripts), and register the function as a proper operator ([below](#integrating-kernels-as-pytorch-custom-ops)) instead of a pybind function.

### Elementwise fusion and vectorized loads

Elementwise ops have arithmetic intensity well below 1 FLOP/byte, so they are memory-bound and their cost is bytes moved ([roofline](execution-model.md#roofline-per-op)). *Fusion* means doing several ops per load: read `x` once, apply bias, GELU, residual, cast, write once. Eager PyTorch runs one kernel per op and round-trips HBM between them. The fused kernel below also uses *vectorized* 16-byte accesses (8 bf16 values per `uint4`). That issues one wide load instruction instead of eight narrow ones, which cuts instruction count and pressure on the load/store queue; it is the "LG Throttle" fix in the [stall table](execution-model.md#nsight-compute-why-is-this-kernel-slow). Vector accesses need the address aligned to 16 bytes and the row length divisible by 8. Real kernels check both and fall back to a scalar path.

```cuda
// bias_gelu.cu - fused bias + tanh-GELU, bf16, 16-byte (8 x bf16) vector loads and stores
#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>

// requires N % 8 == 0 and 16-byte aligned x, bias, y (true for fresh torch allocations)
__global__ void bias_gelu_bf16(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ bias,
                               __nv_bfloat16* __restrict__ y, int64_t rows, int N) {
  const int64_t nvec = rows * N / 8;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t v = blockIdx.x * (int64_t)blockDim.x + threadIdx.x; v < nvec; v += stride) {
    uint4 xr = reinterpret_cast<const uint4*>(x)[v];                       // 16 B = 8 bf16
    uint4 br = *reinterpret_cast<const uint4*>(bias + (v * 8) % N);
    const __nv_bfloat16* xe = reinterpret_cast<const __nv_bfloat16*>(&xr);
    const __nv_bfloat16* be = reinterpret_cast<const __nv_bfloat16*>(&br);
    uint4 out;
    __nv_bfloat16* oe = reinterpret_cast<__nv_bfloat16*>(&out);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float t = __bfloat162float(xe[j]) + __bfloat162float(be[j]);
      oe[j] = __float2bfloat16(0.5f * t * (1.f + tanhf(0.7978845608f * (t + 0.044715f * t * t * t))));
    }
    reinterpret_cast<uint4*>(y)[v] = out;
  }
}
```

It uses a *grid-stride loop*: launch a fixed grid of a few blocks per SM and let each thread walk the array in steps of the whole grid. One launch configuration then works for any size, and per-thread setup is amortized.

The bytes the fusion saves, per element of a `(rows, N)` bf16 activation:

```python
# fusion_bytes.py - HBM bytes per element: bias + GELU + residual add in eager vs fused, bf16
eager = {"x + bias": 2 + 2, "gelu": 2 + 2, "+ residual": 2 + 2 + 2}   # read(s) + write per kernel
fused = 2 + 2 + 2                                                         # read x, read residual, write y
print("eager kernels:", eager, "->", sum(eager.values()), "B/elem")
print("fused:", fused, "B/elem ->", f"{sum(eager.values()) / fused:.2f}x fewer bytes, 3 launches -> 1")
```

```
eager kernels: {'x + bias': 4, 'gelu': 4, '+ residual': 6} -> 14 B/elem
fused: 6 B/elem -> 2.33x fewer bytes, 3 launches -> 1
```

(The kernel above fuses the first two; adding the residual is one more 16-byte load per thread. The bias vector is tiny and cached, so it is ignored. `torch.compile` does exactly this kind of fusion for you, see [Inductor](execution-model.md#inductor-codegen). You write elementwise kernels by hand when the fusion crosses something the compiler will not fuse, such as the epilogue of a custom GEMM.)


## Reductions

A *reduction* combines many values into one (sum, max, mean, logsumexp). Elementwise kernels are embarrassingly parallel (each output depends only on its own inputs, so threads never need to talk); reductions are not, because every input contributes to one output. They appear in every norm, softmax, loss, optimizer step and gradient-norm clip. The goal is the same as for elementwise kernels: a reduction is memory-bound, so a good one reads its input once at close to HBM bandwidth.

The canonical worklog is Mark Harris, "Optimizing Parallel Reduction in CUDA" (NVIDIA, 2007; [slides](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)). It runs on a G80 (peak 86.4 GB/s) with 2^22 integers. Its numbers, as published:

| Harris kernel | idea | GB/s | cumulative speedup |
| :------------ | :--- | ---: | -----------------: |
| 1 | interleaved addressing, divergent branch (`tid % (2*s) == 0`) | 2.083 | 1x |
| 2 | interleaved addressing, strided index (bank conflicts) | 4.854 | 2.33x |
| 3 | sequential addressing | 9.741 | 4.68x |
| 4 | first add during global load | 17.377 | 8.34x |
| 5 | unroll the last warp | 31.289 | 15.01x |
| 6 | complete unroll (templates) | 43.996 | 21.16x |
| 7 | multiple elements per thread | 62.671 | 30.04x (72% of peak) |

The ideas still hold. Two things changed: kernel 5's "warp-synchronous" trick, which relied on implicit lockstep within a warp, became incorrect with Volta's independent thread scheduling, and warp shuffles replaced it.

### v0 to v3

All four versions are in one file. The shared `block_allreduce` helper is written once and reused by every later kernel in this chapter. "All-reduce" has the same meaning as the [collective](../../foundations/distributed-systems.md#collectives-the-workhorse-of-distributed-training) of that name, scaled down: every thread of the block (instead of every GPU) ends up holding the combined result:

```cuda
// block_reduce.cuh - warp and block all-reduce for any struct of floats and any associative merge
#pragma once

template <typename T>
__device__ __forceinline__ T shfl_xor(T v, int lane_mask) {
  static_assert(sizeof(T) % sizeof(float) == 0, "T must be a struct of 4-byte fields");
  float* f = reinterpret_cast<float*>(&v);
#pragma unroll
  for (int i = 0; i < (int)(sizeof(T) / sizeof(float)); ++i)
    f[i] = __shfl_xor_sync(0xffffffffu, f[i], lane_mask);
  return v;
}

// butterfly: after 5 steps every lane of the warp holds the reduction of all 32 lanes
template <typename T, typename Merge>
__device__ __forceinline__ T warp_allreduce(T v, Merge merge) {
#pragma unroll
  for (int m = 16; m > 0; m >>= 1) v = merge(v, shfl_xor(v, m));
  return v;
}

// requires blockDim.x to be a multiple of 32 and <= 1024; every thread gets the result
template <typename T, typename Merge>
__device__ __forceinline__ T block_allreduce(T v, Merge merge, T identity) {
  __shared__ T part[32];                       // one slot per warp
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nwarps = blockDim.x >> 5;
  v = warp_allreduce(v, merge);
  __syncthreads();                             // a previous call may still be reading part[]
  if (lane == 0) part[warp] = v;
  __syncthreads();
  v = lane < nwarps ? part[lane] : identity;   // each warp redundantly reduces the partials,
  return warp_allreduce(v, merge);             // which saves a third barrier and a broadcast
}

struct Sum { __device__ float operator()(float a, float b) const { return a + b; } };
```

```cuda
// reduce.cu - summing n floats, four ways
#include <cstdint>
#include "block_reduce.cuh"

// v0: one atomic per element. Correct, but 2^24 atomics serialize on one address,
// and the summation order (hence the fp32 result) changes run to run.
__global__ void sum_atomic(const float* __restrict__ x, float* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(out, x[i]);
}

// v1: shared-memory tree with sequential addressing: active threads are contiguous
// (whole warps retire together, no divergence) and s[tid], s[tid + stride] hit distinct banks.
template <int BLOCK>
__global__ void sum_smem_tree(const float* __restrict__ x, float* __restrict__ partial, int n) {
  __shared__ float s[BLOCK];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * BLOCK + tid;
  s[tid] = i < n ? x[i] : 0.f;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (tid < stride) s[tid] += s[tid + stride];
    __syncthreads();                           // log2(BLOCK) barriers
  }
  if (tid == 0) partial[blockIdx.x] = s[0];
}

// v2: warp shuffles: registers only inside a warp, one smem round trip per block
__global__ void sum_shuffle(const float* __restrict__ x, float* __restrict__ partial, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = block_allreduce(i < n ? x[i] : 0.f, Sum{}, 0.f);
  if (threadIdx.x == 0) partial[blockIdx.x] = v;
}

// v3: a fixed grid (a few blocks per SM) where each thread first sums many elements with
// 16-byte loads, so the tree runs once per block instead of once per 256 elements.
// Call twice: n elements -> gridDim.x partials -> 1. Deterministic: no atomics.
__global__ void sum_grid_stride(const float* __restrict__ x, float* __restrict__ partial, int64_t n) {
  float acc = 0.f;
  const int64_t tid = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  const int64_t nthreads = (int64_t)gridDim.x * blockDim.x;
  const int64_t n4 = n / 4;                    // x must be 16-byte aligned (torch allocations are)
  const float4* x4 = reinterpret_cast<const float4*>(x);
  for (int64_t i = tid; i < n4; i += nthreads) {
    float4 v = x4[i];
    acc += (v.x + v.y) + (v.z + v.w);
  }
  for (int64_t i = 4 * n4 + tid; i < n; i += nthreads) acc += x[i];   // 0-3 tail elements
  acc = block_allreduce(acc, Sum{}, 0.f);
  if (threadIdx.x == 0) partial[blockIdx.x] = acc;
}
```

Why each one is faster than the last:

- **v0, atomics**: `atomicAdd` on one address serializes in the L2's atomic unit. It is fine for a few thousand atomics, hopeless for millions. It is also non-deterministic: the order of the floating-point additions depends on timing.
- **v1, shared-memory tree**: $\log_2 256 = 8$ steps, each halving the active threads. *Sequential addressing* (`s[tid] += s[tid + stride]`, active threads `0..stride-1`) keeps active threads contiguous, so whole warps go idle together (no [divergence](execution-model.md#divergence)) and the 32 lanes touch 32 consecutive words (no [bank conflicts](execution-model.md#bank-conflicts)). Cost: 8 barriers and a smem round trip per step.
- **v2, warp shuffles**: `__shfl_xor_sync(mask, v, m)` reads register `v` from lane `lane ^ m` directly, with no shared memory and no barrier inside a warp. Five steps reduce 32 lanes. Only the 8 warp partials go through smem, once. The XOR "butterfly" leaves the total in *every* lane, so every warp can finish the block reduction redundantly and the result needs no broadcast.
- **v3, multi-block with many elements per thread**: launch a fixed grid of `4 x num_SMs` blocks. Each thread sums $n / (\text{blocks} \cdot 256)$ elements in registers with 16-byte loads (Harris's kernel 7 idea), and the tree runs once per block instead of once per 256 elements. A second launch reduces the partials. Two passes, no atomics, so the result is **bitwise deterministic** for a given grid size.

The PyTorch wrapper and test for v3 (untested):

```python
# reduce_ext.py - build reduce.cu's v3 into PyTorch and check it (needs an NVIDIA GPU; untested)
import pathlib, torch
from torch.utils.cpp_extension import load_inline

here = pathlib.Path(__file__).parent
cuda_src = (here / "block_reduce.cuh").read_text() + (here / "reduce.cu").read_text().replace(
    '#include "block_reduce.cuh"', "") + r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

torch::Tensor sum_two_pass(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == torch::kFloat32, "contiguous fp32 CUDA");
  const at::cuda::OptionalCUDAGuard guard(x.device());
  const int blocks = 4 * at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
  auto partial = torch::empty({blocks}, x.options());
  auto out = torch::empty({1}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  sum_grid_stride<<<blocks, 256, 0, stream>>>(x.data_ptr<float>(), partial.data_ptr<float>(), x.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  sum_grid_stride<<<1, 1024, 0, stream>>>(partial.data_ptr<float>(), out.data_ptr<float>(), blocks);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}
"""
ext = load_inline("reduce_ext", cpp_sources="torch::Tensor sum_two_pass(torch::Tensor x);",
                  cuda_sources=cuda_src, functions=["sum_two_pass"], extra_cuda_cflags=["-O3", "-lineinfo"])

for n in [1, 3, 4, 1000, 1 << 20, (1 << 24) + 7]:
    x = torch.randn(n, device="cuda")
    got, ref = ext.sum_two_pass(x), x.double().sum()
    assert abs(got.item() - ref.item()) <= 1e-5 * n ** 0.5 + 1e-5, (n, got.item(), ref.item())
    assert torch.equal(ext.sum_two_pass(x), got)          # bitwise reproducible
print("sum_two_pass OK")
```

### Checking the reduction without a GPU

Before running on a GPU, you can execute the algorithm lane by lane in Python to check it gives the right answer. The mirror below reproduces `__shfl_down_sync` (the "down" variant, where only lane 0 ends with the total), the block structure and the grid-stride split. It also shows the floating-point facts that drive kernel design:

```python
# reduce_mirror.py - the warp-shuffle and block reductions, lane by lane, plus why order matters
import numpy as np

def shfl_down(vals, delta):                  # __shfl_down_sync(0xffffffff, v, delta) for a 32-lane warp
    return [vals[l + delta] if l + delta < 32 else vals[l] for l in range(32)]

def warp_reduce_sum(vals):
    vals = list(vals)
    for off in (16, 8, 4, 2, 1):            # 5 steps for 32 lanes
        other = shfl_down(vals, off)
        vals = [v + o for v, o in zip(vals, other)]
    return vals[0]                          # only lane 0 holds the full sum

def block_reduce_sum(x, threads=256):       # one value per thread -> warp sums -> smem -> warp 0
    warp_sums = [warp_reduce_sum(x[w * 32:(w + 1) * 32]) for w in range(threads // 32)]
    smem = warp_sums + [0.0] * (32 - len(warp_sums))   # warp 0 reads 32 slots, pads with 0
    return warp_reduce_sum(smem)

rng = np.random.default_rng(0)
x = rng.standard_normal(256).astype(np.float32)
print("block_reduce_sum:", np.float32(block_reduce_sum(list(x))), " np.sum:", x.sum(dtype=np.float64))

# grid-stride: each of G*256 threads accumulates a strided subset, then one block reduce per block
n, G = 1_000_003, 8
y = rng.standard_normal(n).astype(np.float32)
partials = []
for b in range(G):
    per_thread = [np.float32(0)] * 256
    for t in range(256):
        per_thread[t] = y[b * 256 + t::G * 256].sum(dtype=np.float32)
    partials.append(np.float32(block_reduce_sum(per_thread)))
print("two-pass multi-block:", np.float32(sum(partials)), " fp64 reference:", y.sum(dtype=np.float64))

# fp32 addition is not associative: the same numbers in a different order give a different sum
orders = {np.float32(np.sum(rng.permutation(y), dtype=np.float32)) for _ in range(5)}
print("5 random orders ->", len(orders), "distinct fp32 sums:", sorted(float(o) for o in orders))

# Welford / Chan merge for LayerNorm: (n, mean, M2) of two halves combine exactly
def merge(a, b):
    n = a[0] + b[0]; d = b[1] - a[1]
    return n, a[1] + d * b[0] / n, a[2] + b[2] + d * d * a[0] * b[0] / n
z = rng.standard_normal(4096) * 3 + 100
st = lambda v: (len(v), v.mean(), ((v - v.mean()) ** 2).sum())
n_, mean, M2 = merge(merge(st(z[:1000]), st(z[1000:3000])), st(z[3000:]))
print(f"welford merge: mean {mean:.10f} var {M2 / n_:.10f} | numpy: {z.mean():.10f} {z.var():.10f}")

# online-softmax pair merge (m, d): max and rescaled sum, associative so any tree order works
def sm_merge(a, b):
    m = max(a[0], b[0]); return m, a[1] * np.exp(a[0] - m) + b[1] * np.exp(b[0] - m)
s = rng.standard_normal(1000) * 10
parts = [(c.max(), np.exp(c - c.max()).sum()) for c in np.split(s, 8)]
m, d = parts[0]
for p in parts[1:]: m, d = sm_merge((m, d), p)
print(f"online softmax merge: logsumexp {m + np.log(d):.12f} | direct {np.log(np.exp(s).sum()):.12f}")
```

```
block_reduce_sum: 0.48477745  np.sum: 0.4847759190015495
two-pass multi-block: 989.6423  fp64 reference: 989.6422559503635
5 random orders -> 2 distinct fp32 sums: [989.642333984375, 989.6424560546875]
welford merge: mean 100.0402787114 var 8.6659856771 | numpy: 100.0402787114 8.6659856771
online softmax merge: logsumexp 32.735291829949 | direct 32.735291829949
```

Three lessons from the output. The block and multi-block trees agree with an fp64 reference to fp32 rounding. Summing the same 1M floats in different orders gives **different fp32 results**, because floating-point addition is not associative. That is why atomics (order depends on timing) and split-K style kernels whose split depends on the shape are non-deterministic, while a fixed two-pass tree is reproducible ([batch invariance](../../inference/serving-engine-internals.md#determinism-and-batch-invariance) in a serving engine, a later chapter, is the same issue: the same request should give the same bits whatever else is in its batch). The Welford and online-softmax *merge* functions are associative, which is exactly what makes them usable inside a tree reduction: any split of the input, merged in any tree order, gives the same answer up to rounding.


## Softmax, RMSNorm and LayerNorm in CUDA

All three are "one row per block, reduce, then normalize" kernels. The only differences are what gets reduced and how two partial states merge:

| op | per-thread state | merge(a, b) | passes over x |
| :- | :--------------- | :---------- | ------------: |
| softmax (naive) | max, then sum | max; sum | 3 (max, sum, write) |
| softmax (online) | $(m, d)$ running max and rescaled sum | $m = \max(m_a, m_b);\ d = d_a e^{m_a - m} + d_b e^{m_b - m}$ | 2 (stats, write) |
| RMSNorm | sum of squares | `+` | 2 (stats, write) |
| LayerNorm (Welford) | `(n, mean, M2)` | Chan et al. parallel merge | 2 (stats, write) |

The online softmax derivation is in [the execution model chapter](execution-model.md#online-softmax). Here it is used as a merge operator, so the `block_allreduce` above works unchanged:

```cuda
// norms.cu - one block per row: online softmax, RMSNorm (bf16), LayerNorm (Welford)
#include <cstdint>
#include <cuda_bf16.h>
#include <math.h>
#include "block_reduce.cuh"

// ---- softmax: (max, sum) in one pass, then one normalizing pass: 2 reads of x instead of 3 ----
struct MD { float m, d; };                     // running max, running sum of exp(x - m)
struct MergeMD {
  __device__ MD operator()(MD a, MD b) const {
    float m = fmaxf(a.m, b.m);
    if (m == -INFINITY) return {m, 0.f};       // both sides saw only -inf: avoid (-inf) - (-inf)
    return {m, a.d * __expf(a.m - m) + b.d * __expf(b.m - m)};
  }
};

__global__ void softmax_online(const float* __restrict__ x, float* __restrict__ y, int N) {
  const float* xr = x + (int64_t)blockIdx.x * N;
  float* yr = y + (int64_t)blockIdx.x * N;
  MD t{-INFINITY, 0.f};
  for (int i = threadIdx.x; i < N; i += blockDim.x) t = MergeMD{}(t, MD{xr[i], 1.f});
  const MD row = block_allreduce(t, MergeMD{}, MD{-INFINITY, 0.f});
  const float inv_d = 1.f / row.d;
  for (int i = threadIdx.x; i < N; i += blockDim.x) yr[i] = __expf(xr[i] - row.m) * inv_d;
}

// ---- RMSNorm, bf16 in/out, fp32 math. The second read of the row hits L1/L2 ----
__global__ void rmsnorm_bf16(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                             __nv_bfloat16* __restrict__ y, int N, float eps) {
  const __nv_bfloat16* xr = x + (int64_t)blockIdx.x * N;
  __nv_bfloat16* yr = y + (int64_t)blockIdx.x * N;
  float ss = 0.f;
  for (int i = threadIdx.x; i < N; i += blockDim.x) {
    float v = __bfloat162float(xr[i]);
    ss += v * v;
  }
  ss = block_allreduce(ss, Sum{}, 0.f);
  const float rstd = rsqrtf(ss / N + eps);
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    yr[i] = __float2bfloat16(__bfloat162float(xr[i]) * rstd * __bfloat162float(w[i]));
}

// ---- LayerNorm: Welford (count, mean, M2) per thread, merged with Chan's formula ----
// One-pass sum / sum-of-squares is E[x^2] - E[x]^2, which cancels catastrophically when
// |mean| >> std; Welford does not.
struct Welford { float n, mean, m2; };
struct MergeW {
  __device__ Welford operator()(Welford a, Welford b) const {
    float n = a.n + b.n;
    if (n == 0.f) return {0.f, 0.f, 0.f};
    float d = b.mean - a.mean, rb = b.n / n;
    return {n, a.mean + d * rb, a.m2 + b.m2 + d * d * a.n * rb};
  }
};

__global__ void layernorm_f32(const float* __restrict__ x, const float* __restrict__ gamma,
                              const float* __restrict__ beta, float* __restrict__ y, int N, float eps) {
  const float* xr = x + (int64_t)blockIdx.x * N;
  float* yr = y + (int64_t)blockIdx.x * N;
  Welford t{0.f, 0.f, 0.f};
  for (int i = threadIdx.x; i < N; i += blockDim.x) {   // Welford update = merge with a 1-sample state
    float v = xr[i];
    t.n += 1.f;
    float d = v - t.mean;
    t.mean += d / t.n;
    t.m2 += d * (v - t.mean);
  }
  const Welford s = block_allreduce(t, MergeW{}, Welford{0.f, 0.f, 0.f});
  const float mean = s.mean, rstd = rsqrtf(s.m2 / N + eps);
  for (int i = threadIdx.x; i < N; i += blockDim.x)
    yr[i] = (xr[i] - mean) * rstd * gamma[i] + beta[i];
}
```

Design notes:

- **Why Welford for LayerNorm.** The tempting one-pass $\operatorname{var} = E[x^2] - E[x]^2$ subtracts two large, nearly equal numbers when $\lvert\text{mean}\rvert \gg \text{std}$, and in fp32 it can lose every significant digit (it can even go negative). Welford's running `(n, mean, M2)` and Chan's merge are stable. The mirror output above confirms the merge on data with mean 100 and std 3. RMSNorm has no mean subtraction, so a plain sum of squares in fp32 is fine.
- **The `-inf` guard** in `MergeMD` handles fully masked rows (attention with padding): `(-inf) - (-inf)` is NaN in IEEE arithmetic, and one NaN poisons the whole row. The Triton softmax below has the same guard, and the [tests](writing-kernels-reference.md#testing-kernels) cover it.
- **The second pass** re-reads `x`. For rows up to a few thousand elements that re-read hits L1/L2. The faster production layout caches the row in registers on the first pass. With $N \le 8192$ bf16 and 256 threads that is at most 32 values per thread (the [register arithmetic](execution-model.md#choices-and-why) in the execution model chapter), and it removes the second HBM read entirely for rows evicted from L2.
- **One block per row** is wrong for small `N`: a 1024-wide row on 256 threads gives each thread 4 elements, and block-level barriers dominate. Production kernels (Apex, FlashAttention's layer norm, vLLM's `csrc/layernorm_kernels.cu`) map *one warp per row* when `N` is small, so the reduction is shuffles only, and one block per row when it is large.

A test for the CUDA norms looks exactly like the Triton tests in [Testing kernels](writing-kernels-reference.md#testing-kernels); `reduce_ext.py` above shows the build pattern.


> **Moved to the reference part:** [GEMM step by step](writing-kernels-reference.md#gemm-step-by-step). Read it when you need it.

## Triton from zero

### The programming model

In CUDA you write the program for **one thread** and reason about warps by hand. In Triton you write the program for **one block of work** (a *program instance*, one per grid point). Every value is a whole tile (a small tensor of shape known at compile time), and the compiler decides how the tile is spread across threads, registers and shared memory, and which instructions (vector loads, `mma`, `wgmma`) to use. The core vocabulary:

| Triton | meaning | CUDA analogue |
| :----- | :------ | :------------ |
| `@triton.jit` | compile this Python function as a kernel (JIT, specialized per dtype / constexpr / alignment) | `__global__` |
| `kernel[grid](args)` | launch `grid` programs; `grid` is a tuple or `lambda meta: (...)` | `<<<grid, block>>>` |
| `tl.program_id(axis)` | which program am I | `blockIdx` |
| `tl.arange(0, BLOCK)` | the tile of indices `0..BLOCK-1` (BLOCK a power of 2) | `threadIdx` loop, implicit |
| `tl.constexpr` | compile-time constant: tile sizes, flags | template parameter |
| `X + offs` | a tile of pointers (tensor arguments arrive as pointers to element 0) | pointer arithmetic |
| `tl.load(p, mask, other)` | gather a tile; masked lanes get `other` instead of touching memory | guarded loads |
| `tl.store(p, v, mask)` | scatter a tile; masked lanes do nothing | guarded stores |
| `tl.sum / max(x, axis)` | reduction along an axis of the tile | everything in [Reductions](#reductions) |
| `tl.dot(a, b, acc)` | tile matmul on tensor cores, fp32 accumulate | WMMA / `mma` / `wgmma` |
| `num_warps`, `num_stages` | threads per program (`32 * num_warps`), software-pipeline depth | `blockDim`, multi-buffering |

NVIDIA's picture of the same split (Triton is a tile model):

![SIMT vs tile programming model](images/simt-vs-tile-model.jpg)

([source](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html))

A 2-D tile is built by broadcasting a column of row indices against a row of column indices. This is the pattern in every matmul and attention kernel below. The simulator shows the offsets and the mask for the last row-block of a 5x6 matrix with 4x8 tiles:

```python
# ptr_tile.py - what a 2-D pointer tile and its mask look like (runs in the simulator)
import torch
from backend import tl

x = torch.arange(5 * 6, dtype=torch.float32).reshape(5, 6)     # (M=5, N=6), row stride 6
rows = 4 + tl.arange(0, 4)                                      # program 1 of BLOCK_M=4: rows 4..7
cols = tl.arange(0, 8)                                          # BLOCK_N=8 > N=6
offs = rows[:, None] * x.stride(0) + cols[None, :]              # (4, 8) element offsets: broadcast
mask = (rows[:, None] < 5) & (cols[None, :] < 6)                # (4, 8)
print("offsets:\n", offs)
print("mask:\n", mask.int())
from tsim import _as_ptr
print("loaded tile (other=-1):\n", tl.load(_as_ptr(x) + offs, mask=mask, other=-1.0))
```

```
offsets:
 tensor([[24, 25, 26, 27, 28, 29, 30, 31],
        [30, 31, 32, 33, 34, 35, 36, 37],
        [36, 37, 38, 39, 40, 41, 42, 43],
        [42, 43, 44, 45, 46, 47, 48, 49]])
mask:
 tensor([[1, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.int32)
loaded tile (other=-1):
 tensor([[24., 25., 26., 27., 28., 29., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1.],
        [-1., -1., -1., -1., -1., -1., -1., -1.]])
```

Look at the masked offsets 30 and 31 in the first row. Here they fall just past the end of the 30-element tensor. In any row except the last, the same column overflow lands on the start of the *next* row, which is a valid address. Without the column mask the kernel would read the wrong data silently instead of crashing.

Masks replace the `if (i < n)` of CUDA. Every load and store whose indices can run past the end needs one. The rule for `other` is "a value that does not change the result": `0` for sums and dot products, `-inf` for max and softmax.

### Vector add, and what a missing mask does

```python
# vadd.py
import torch
from backend import triton, tl, DEV


@triton.jit
def add_kernel(X, Y, OUT, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)                        # which BLOCK-sized chunk is mine
    offs = pid * BLOCK + tl.arange(0, BLOCK)      # (BLOCK,) element indices
    mask = offs < n                               # the last chunk is partial
    x = tl.load(X + offs, mask=mask)
    y = tl.load(Y + offs, mask=mask)
    tl.store(OUT + offs, x + y, mask=mask)


def add(x, y, BLOCK=1024):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)    # number of programs
    add_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


@triton.jit
def add_kernel_nomask(X, Y, OUT, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + offs, tl.load(X + offs) + tl.load(Y + offs))   # BUG: no mask


if __name__ == "__main__":
    torch.manual_seed(0)
    for n in [1, 1000, 1024, 1025, 98_432]:
        x, y = torch.randn(n, device=DEV), torch.randn(n, device=DEV)
        torch.testing.assert_close(add(x, y), x + y)
    print("add: 5 sizes OK")
    x = torch.randn(1000, device=DEV); out = torch.empty_like(x)
    try:
        add_kernel_nomask[(1,)](x, x, out, 1000, BLOCK=1024)
    except IndexError as e:
        print("nomask:", e)
```

```
add: 5 sizes OK
nomask: out-of-bounds access (offset 1023 >= 1000); missing mask?
```

On a GPU, the unmasked kernel would not raise. It would read past the end of `x` and write past the end of `out`, usually into memory the caching allocator gave to another tensor. Real Triton's interpreter (`TRITON_INTERPRET=1`) runs on numpy, and it is your first tool for catching this on a Linux box without a GPU. `compute-sanitizer` ([below](writing-kernels-reference.md#testing-kernels)) catches it on a GPU.

### Fused softmax: one row per program, and online for long rows

This mirrors Triton's tutorial 02 in idea. One program handles one row. If the whole row fits in one tile, the kernel reads `x` once and writes `y` once, which is the memory floor. Eager PyTorch's unfused equivalent (`max`, `sub`, `exp`, `sum`, `div`) reads and writes the row several times. For rows too long for one tile (a 128K-vocab logits row), the online variant loops over chunks with the `(m, d)` state from the CUDA version, then makes a second pass to write.

```python
# softmax.py - row softmax two ways: whole row in registers, and online (chunked) for long rows
import torch
from backend import triton, tl, DEV


@triton.jit
def softmax_row_kernel(X, Y, stride_x, stride_y, N, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)                                  # one program per row
    cols = tl.arange(0, BLOCK_N)                            # (BLOCK_N,)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=float("-inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)                               # -inf padding -> exp() = 0
    num = tl.exp(x)
    y = num / tl.sum(num, axis=0)
    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def softmax_online_kernel(X, Y, stride_x, stride_y, N, BLOCK_N: tl.constexpr):
    # pass 1: running max m and running normalizer d over chunks (Milakov & Gimelshein)
    row = tl.program_id(0)
    m = tl.full((BLOCK_N,), float("-inf"), tl.float32)      # per-lane running max
    d = tl.zeros((BLOCK_N,), tl.float32)                    # per-lane running sum of exp(x - m)
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        x = tl.load(X + row * stride_x + cols, mask=cols < N, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        # lanes that have only seen -inf keep d = 0 (avoid exp(-inf - -inf) = nan)
        d = tl.where(m_new == float("-inf"), 0.0, d * tl.exp(m - m_new) + tl.exp(x - m_new))
        m = m_new
    m_row = tl.max(m, axis=0)                               # combine the BLOCK_N lanes
    d_row = tl.sum(tl.where(m == float("-inf"), 0.0, d * tl.exp(m - m_row)), axis=0)
    # pass 2: normalize and write
    for start in range(0, N, BLOCK_N):
        cols = start + tl.arange(0, BLOCK_N)
        x = tl.load(X + row * stride_x + cols, mask=cols < N, other=float("-inf")).to(tl.float32)
        tl.store(Y + row * stride_y + cols, (tl.exp(x - m_row) / d_row).to(Y.dtype.element_ty), mask=cols < N)


def softmax(x, online=False, BLOCK_N=None):
    if x.stride(-1) != 1:          # the kernel indexes columns as base + cols: needs unit stride
        x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    if online:
        softmax_online_kernel[(M,)](x, y, x.stride(0), y.stride(0), N, BLOCK_N=BLOCK_N or 1024)
    else:
        softmax_row_kernel[(M,)](x, y, x.stride(0), y.stride(0), N, BLOCK_N=triton.next_power_of_2(N))
    return y


if __name__ == "__main__":
    torch.manual_seed(0)
    for M, N in [(1, 1), (3, 7), (64, 1000), (16, 4096), (5, 5003)]:
        for dt in (torch.float32, torch.bfloat16):
            x = (3 * torch.randn(M, N, device=DEV)).to(dt)
            ref = torch.softmax(x.float(), -1).to(dt)
            torch.testing.assert_close(softmax(x), ref)
            torch.testing.assert_close(softmax(x, online=True, BLOCK_N=256), ref)
    x = torch.randn(4, 300, device=DEV); x[1, :] = float("-inf"); x[2, 100:] = float("-inf")
    print("fully masked row -> nan, like torch:", softmax(x)[1, :3].tolist(), torch.softmax(x, -1)[1, :3].tolist())
    torch.testing.assert_close(softmax(x, online=True, BLOCK_N=64)[2:], torch.softmax(x, -1)[2:])
    print("softmax: row and online variants OK for 5 shapes x 2 dtypes")
```

```
fully masked row -> nan, like torch: [nan, nan, nan] [nan, nan, nan]
softmax: row and online variants OK for 5 shapes x 2 dtypes
```

The online kernel keeps a per-lane `(m, d)` vector across the loop and merges lanes once at the end, so the loop body has no cross-lane reduction at all. That is the same trick as v3 of the CUDA reduction: accumulate privately, reduce once.

### Matmul with autotuning

This follows the structure of Triton's tutorial 03, written out with masks so that it handles any `M`, `N`, `K`:

```python
# matmul.py - tiled GEMM C = A @ B with grouped rasterization and autotuning
import torch
from backend import triton, tl, DEV


def configs():
    return [triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": 8}, num_warps=w, num_stages=s)
            for bm, bn, bk, w, s in [(128, 128, 64, 8, 3), (128, 64, 32, 4, 4), (64, 128, 32, 4, 4),
                                     (64, 64, 32, 4, 4), (32, 32, 32, 2, 5)]]


@triton.autotune(configs=configs(), key=["M", "N", "K"])     # re-tune when the shape changes
@triton.jit
def matmul_kernel(A, B, C, M, N, K,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    # --- which output tile is mine: grouped ordering so GROUP_M row-tiles share B tiles in L2 ---
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)       # last group may be short
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)               # (BM,) rows of C
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)               # (BN,) cols of C
    rk = tl.arange(0, BLOCK_K)                                 # (BK,)
    a_ptrs = A + rm[:, None] * stride_am + rk[None, :] * stride_ak   # (BM, BK)
    b_ptrs = B + rk[:, None] * stride_bk + rn[None, :] * stride_bn   # (BK, BN)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)      # fp32 accumulator in registers
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_left = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < k_left) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)                                # (BM, BN) += (BM, BK) @ (BK, BN)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=(rm[:, None] < M) & (rn[None, :] < N))


def matmul(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)
    matmul_kernel[grid](a, b, c, M, N, K,
                        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1))
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    for M, N, K in [(1, 1, 1), (128, 128, 64), (257, 129, 100), (1000, 300, 77), (64, 64, 64)]:
        for dt in (torch.float16, torch.bfloat16):
            a = torch.randn(M, K, device=DEV, dtype=dt)
            b = torch.randn(K, N, device=DEV, dtype=dt)
            ref = (a.float() @ b.float()).to(dt)
            torch.testing.assert_close(matmul(a, b), ref)
    a = torch.randn(300, 200, device=DEV, dtype=torch.float16)
    bt = torch.randn(150, 200, device=DEV, dtype=torch.float16)
    torch.testing.assert_close(matmul(a, bt.t()), (a.float() @ bt.float().t()).half())   # strided B
    print("matmul: 5 shapes x 2 dtypes (10 calls = every config twice) + transposed B OK")
```

```
matmul: 5 shapes x 2 dtypes (10 calls = every config twice) + transposed B OK
```

What each part does:

- **`tl.dot`** compiles to `mma.sync` on Ampere, `wgmma` on Hopper and `tcgen05` on Blackwell. You never see the fragment layouts. `num_stages` controls how many K-tiles are in flight (`cp.async` or TMA pipelining), and `num_warps` how many warps share the tile.
- **`@triton.autotune(configs, key)`**: on the first call for each distinct value of the `key` arguments, Triton benchmarks every config and caches the fastest. Pick configs that span the tradeoffs: large tiles (more reuse, more registers and smem, fewer programs) vs small tiles (more parallelism for small shapes). Autotuning costs time at first call, and it is a source of non-determinism across machines, since a different config means a different summation order. Production code often pins configs per GPU from an offline sweep. (The simulator's "autotune" uses the next config on each call, so the test loop above exercised all five configs.)
- **Grouped ordering (`GROUP_M`)**: programs with consecutive `pid` values run at roughly the same time. With row-major order, the first 9 programs need 1 row-tile of A and 9 column-tiles of B. With groups of 3 rows they need 3 and 3, so tiles are re-read from L2 instead of HBM:

![Row-major vs grouped ordering of output tiles in Triton's matmul tutorial: grouped ordering loads 27+27 blocks instead of 9+81 for the same 9 outputs](images/triton-grouped-vs-row-major-ordering.png)

([source](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html))

The figure's counts, reproduced from the kernel's own mapping code:

```python
# swizzle.py - the matmul kernel's pid -> (pid_m, pid_n) mapping, printed as a grid of launch order
num_pid_m = num_pid_n = 9; GROUP_M = 3
order = [[None] * num_pid_n for _ in range(num_pid_m)]
for pid in range(num_pid_m * num_pid_n):
    num_pid_in_group = GROUP_M * num_pid_n
    first_pid_m = (pid // num_pid_in_group) * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m
    order[pid_m][pid_n] = pid
for row in order[:4]: print(" ".join(f"{p:2d}" for p in row))
# tiles loaded by the first 9 programs (one wave of 9 SMs): A row-tiles + B col-tiles
for name, first9 in [("row-major", [(p // 9, p % 9) for p in range(9)]),
                     ("grouped", [(r, c) for r in range(9) for c in range(9) if order[r][c] < 9])]:
    rows = {r for r, _ in first9}; cols = {c for _, c in first9}
    print(f"{name:9s}: {len(rows)} distinct A row-blocks, {len(cols)} distinct B col-blocks "
          f"-> {9*len(rows) + 9*len(cols)} tile loads of A+B")
```

```
 0  3  6  9 12 15 18 21 24
 1  4  7 10 13 16 19 22 25
 2  5  8 11 14 17 20 23 26
27 30 33 36 39 42 45 48 51
row-major: 1 distinct A row-blocks, 9 distinct B col-blocks -> 90 tile loads of A+B
grouped  : 3 distinct A row-blocks, 3 distinct B col-blocks -> 54 tile loads of A+B
```

### Fused cross-entropy

For a 128K vocabulary, eager `F.cross_entropy` on `(B*T, V)` bf16 logits materializes fp32 softmax and gradient tensors several times the size of the logits. At $B \cdot T = 16384$ and $V = 128256$, one fp32 copy is 8.4 GB. The fused kernel computes the loss with an online logsumexp in one read, then **overwrites the logits buffer with the gradient** $(\operatorname{softmax} - \text{onehot}) / n$ in a second pass. The backward pass then just returns that buffer. This is the trick behind Liger-Kernel's cross-entropy. Its "fused linear cross-entropy" goes further and never materializes the logits at all, chunking the final matmul.

```python
# cross_entropy.py - fused cross-entropy: loss and dlogits in one kernel, no (B, V) softmax tensor
import torch
from backend import triton, tl, DEV


@triton.jit
def ce_fwd_bwd_kernel(LOGITS, TARGET, LOSS, stride, V, n_valid, ignore_index,
                      BLOCK_V: tl.constexpr):
    row = tl.program_id(0)
    t = tl.load(TARGET + row)
    base = LOGITS + row * stride
    if t == ignore_index:                                  # padding token: zero loss, zero grad
        for s in range(0, V, BLOCK_V):
            cols = s + tl.arange(0, BLOCK_V)
            tl.store(base + cols, tl.zeros((BLOCK_V,), tl.float32), mask=cols < V)
        tl.store(LOSS + row, 0.0)
        return
    # pass 1: online logsumexp over the vocab, chunk by chunk (scalar running max m, sum d)
    m = float("-inf")
    d = 0.0
    for s in range(0, V, BLOCK_V):
        cols = s + tl.arange(0, BLOCK_V)
        x = tl.load(base + cols, mask=cols < V, other=float("-inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, axis=0))
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), axis=0)
        m = m_new
    lse = m + tl.log(d)
    x_t = tl.load(base + t).to(tl.float32)
    tl.store(LOSS + row, (lse - x_t) / n_valid)            # mean reduction over non-ignored rows
    # pass 2: overwrite the logits with dloss/dlogits = (softmax - onehot) / n_valid
    for s in range(0, V, BLOCK_V):
        cols = s + tl.arange(0, BLOCK_V)
        x = tl.load(base + cols, mask=cols < V, other=0.0).to(tl.float32)
        g = (tl.exp(x - lse) - tl.where(cols == t, 1.0, 0.0)) / n_valid
        tl.store(base + cols, g.to(LOGITS.dtype.element_ty), mask=cols < V)


class FusedCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, ignore_index=-100):    # logits (B, V), target (B,)
        logits = logits.contiguous().clone()   # clone so the demo can compare; production reuses the buffer
        B, V = logits.shape
        loss = torch.empty(B, device=logits.device, dtype=torch.float32)
        n_valid = max(int((target != ignore_index).sum()), 1)   # host sync; fine for a demo
        ce_fwd_bwd_kernel[(B,)](logits, target, loss, logits.stride(0), V, n_valid, ignore_index,
                                BLOCK_V=1024)
        ctx.save_for_backward(logits)                       # now holds the gradient
        return loss.sum()

    @staticmethod
    def backward(ctx, gout):
        (grad,) = ctx.saved_tensors
        return grad * gout, None, None


if __name__ == "__main__":
    torch.manual_seed(0)
    for B, V in [(1, 10), (8, 1000), (33, 5000), (4, 32_000)]:
        for dt in (torch.float32, torch.bfloat16):
            x = (4 * torch.randn(B, V, device=DEV)).to(dt).requires_grad_()
            t = torch.randint(0, V, (B,), device=DEV)
            if B > 1: t[0] = -100                           # one ignored row (B=1 would be 0/0 = nan in torch)
            loss = FusedCE.apply(x, t); loss.backward(); g = x.grad.clone(); x.grad = None
            ref = torch.nn.functional.cross_entropy(x.float(), t, ignore_index=-100); ref.backward()
            torch.testing.assert_close(loss, ref, rtol=1e-5, atol=1e-5) if dt == torch.float32 else \
                torch.testing.assert_close(loss, ref, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(g, x.grad.to(dt))
    print("fused CE: loss and grad match F.cross_entropy for 4 shapes x 2 dtypes")
```

```
fused CE: loss and grad match F.cross_entropy for 4 shapes x 2 dtypes
```

The memory it saves:

```python
# ce_mem.py - memory for cross-entropy over a Llama-3-sized vocabulary
BT, V = 16384, 128256
gb = lambda n_bytes: f"{n_bytes / 1e9:5.1f} GB"
print("bf16 logits             ", gb(BT * V * 2))
print("fp32 upcast + softmax   ", gb(BT * V * 4 * 2), "(eager, typical)")
print("fused: grad in place    ", gb(0), "extra; loss is", BT * 4, "bytes")
```

```
bf16 logits                4.2 GB
fp32 upcast + softmax     16.8 GB (eager, typical)
fused: grad in place       0.0 GB extra; loss is 65536 bytes
```

Two caveats. The demo clones the logits so it can compare against PyTorch. Production code writes into the caller's buffer and has to document that the input is destroyed (`mutates_args` in [the custom op](#integrating-kernels-as-pytorch-custom-ops)). The host-side `n_valid` count forces a GPU-to-CPU sync; production kernels pass it as a device tensor.

### FlashAttention forward, compact

This is the [FlashAttention-2 algorithm](execution-model.md#flashattention) in about 50 lines: one program per `(query block, batch*head)`, a loop over key/value blocks, online softmax in base 2 (`exp2` is the native instruction; `log2(e)` is folded into the scale), causal block skipping, and the log-sum-exp saved for a backward pass.

```python
# flash_fwd.py - compact FlashAttention-2 style forward: O = softmax(Q K^T * scale) V, plus LSE
import math
import torch
from backend import triton, tl, DEV


@triton.jit
def flash_fwd_kernel(Q, K, V, O, LSE, sm_scale, T,
                     stride_b, stride_h, stride_t,            # q, k, v, o share one contiguous layout
                     H, CAUSAL: tl.constexpr, D: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)                                # which query block
    bh = tl.program_id(1)                                     # which (batch, head)
    off = (bh // H) * stride_b + (bh % H) * stride_h
    rm = start_m * BLOCK_M + tl.arange(0, BLOCK_M)            # (BM,) query positions
    rd = tl.arange(0, D)                                      # (D,)
    q = tl.load(Q + off + rm[:, None] * stride_t + rd[None, :], mask=rm[:, None] < T, other=0.0)  # (BM, D)

    m_i = tl.full((BLOCK_M,), float("-inf"), tl.float32)     # running row max (base-2 units)
    l_i = tl.zeros((BLOCK_M,), tl.float32)                    # running row sum
    acc = tl.zeros((BLOCK_M, D), tl.float32)                  # unnormalized output
    qk_scale = sm_scale * 1.4426950408889634                  # fold log2(e) in: exp(x) = exp2(x*log2e)

    hi = tl.minimum((start_m + 1) * BLOCK_M, T) if CAUSAL else T   # causal: skip blocks above diagonal
    for start_n in range(0, hi, BLOCK_N):
        rn = start_n + tl.arange(0, BLOCK_N)                  # (BN,) key positions
        kv_mask = rn[:, None] < T
        k = tl.load(K + off + rn[:, None] * stride_t + rd[None, :], mask=kv_mask, other=0.0)   # (BN, D)
        v = tl.load(V + off + rn[:, None] * stride_t + rd[None, :], mask=kv_mask, other=0.0)   # (BN, D)
        s = tl.dot(q, tl.trans(k)) * qk_scale                 # (BM, BN) fp32
        valid = rn[None, :] < T
        if CAUSAL:
            valid = valid & (rm[:, None] >= rn[None, :])
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))            # (BM,)  finite: key 0 is always valid
        p = tl.exp2(s - m_new[:, None])                       # (BM, BN)
        alpha = tl.exp2(m_i - m_new)                          # (BM,) rescale for the old state
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v) # P cast to bf16/fp16 for the tensor core
        m_i = m_new

    acc = acc / l_i[:, None]
    tl.store(O + off + rm[:, None] * stride_t + rd[None, :], acc.to(O.dtype.element_ty), mask=rm[:, None] < T)
    lse = m_i / 1.4426950408889634 + tl.log(l_i)              # back to natural log: ln(sum e^s)
    tl.store(LSE + bh * T + rm, lse, mask=rm < T)            # natural-log LSE, for a backward pass


def flash_fwd(q, k, v, causal=False, BLOCK_M=64, BLOCK_N=64):
    B, H, T, D = q.shape                                      # (B, H, T, D), contiguous
    assert q.is_contiguous() and k.is_contiguous() and v.is_contiguous() and D in (16, 32, 64, 128)
    o = torch.empty_like(q)
    lse = torch.empty(B * H, T, device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(T, BLOCK_M), B * H)
    flash_fwd_kernel[grid](q, k, v, o, lse, 1.0 / math.sqrt(D), T,
                           q.stride(0), q.stride(1), q.stride(2), H,
                           CAUSAL=causal, D=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return o, lse.view(B, H, T)


if __name__ == "__main__":
    torch.manual_seed(0)
    F = torch.nn.functional
    for B, H, T, D in [(1, 1, 1, 16), (2, 3, 100, 64), (1, 2, 257, 128)]:
        for causal in (False, True):
            for dt in (torch.float16, torch.bfloat16):
                q, k, v = (torch.randn(B, H, T, D, device=DEV, dtype=dt) for _ in range(3))
                o, lse = flash_fwd(q, k, v, causal)
                ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal)
                s = (q.float() @ k.float().transpose(-1, -2)) / math.sqrt(D)
                if causal:
                    s = s.masked_fill(torch.ones(T, T, dtype=torch.bool).triu(1), float("-inf"))
                torch.testing.assert_close(o, ref.to(dt), atol=2e-2, rtol=2e-2)
                torch.testing.assert_close(lse, torch.logsumexp(s, -1), atol=1e-3, rtol=1e-3)
    print("flash_fwd: 3 shapes x causal/non-causal x fp16/bf16 OK (output and LSE)")
```

```
flash_fwd: 3 shapes x causal/non-causal x fp16/bf16 OK (output and LSE)
```

What this compact version leaves out, in the order you would add it: `num_stages` pipelining and autotuned `BLOCK_M/BLOCK_N` per head dim; splitting the causal loop into the full off-diagonal blocks (no mask needed) and the diagonal block (masked), which removes the `tl.where` from most iterations; GQA (index K/V by `head // group`); variable-length sequences packed into one tensor without padding (`cu_seqlens`, the cumulative sequence lengths that mark where each sequence starts; see [ragged batching](../../inference/serving-engine-internals.md#ragged-batching), a later chapter); and the backward pass (recompute `P` from `Q, K` and the saved LSE). The Triton repository's tutorial 06 (fused attention) has forward + backward. The serving-side variants (split-KV, paged KV) are in [inference attention kernels](../../inference/serving-engine-internals.md#inference-attention-kernels), a later chapter.

### Debugging and inspecting Triton

These commands need Linux and real Triton (untested here; flags and environment variables as of Triton 3.x, check your version). `VAR=1 python ...` sets an [environment variable](../../foundations/linux-and-os.md#environment-variables-and-path) for that one command. MLIR is the compiler framework Triton is built on; TTIR and TTGIR are Triton's own intermediate representations, the stages between your Python and PTX:

```bash
TRITON_INTERPRET=1 python flash_fwd.py      # CPU interpreter: pdb (Python's debugger) works inside kernels, prints show tiles
TRITON_DEBUG=1 python flash_fwd.py          # enables tl.device_assert checks
MLIR_ENABLE_DUMP=1 python flash_fwd.py      # dump the MLIR passes (TTIR -> TTGIR -> LLVM)
ls ~/.triton/cache/*/                        # per-kernel .ttir .ttgir .llir .ptx .cubin, and a .json with
                                             # num_warps, shared memory bytes, etc.
```

Inside a kernel, `tl.device_print("m", m_i)` prints from the device, and `tl.static_print` prints at compile time (shapes and constexprs). To check what the compiler did with your `tl.dot`, look in the `.ptx` for `mma.sync` / `wgmma.mma_async` and in the `.ttgir` for the layouts it chose. If a dot falls back to FMA instructions, the tile shapes or dtypes were unsupported.


> **Moved to the reference part:** [Grouped GEMM for MoE](writing-kernels-reference.md#grouped-gemm-for-moe), [Testing kernels](writing-kernels-reference.md#testing-kernels), [Benchmarking kernels](writing-kernels-reference.md#benchmarking-kernels), [Reading ncu to choose the next optimization](writing-kernels-reference.md#reading-ncu-to-choose-the-next-optimization). Read it when you need it.

## Integrating kernels as PyTorch custom ops

A raw Python function that launches a kernel works in eager mode. But `torch.compile` cannot trace through a pybind call or a Triton launch it does not understand, `torch.export` (which saves a traced model graph for deployment without Python) cannot serialize it, and autograd knows nothing about it. `torch.library` (PyTorch 2.4+; APIs as of 2.15) turns it into a real operator with a schema:

- `@torch.library.custom_op("ns::name", mutates_args=())` defines the op from type annotations. The op is **opaque** to `torch.compile`: it appears as one node, and Dynamo does not graph-break on it.
- `register_fake` is the shape/dtype function used during tracing (on "fake" tensors that carry no data). If it is missing, `torch.compile` fails.
- `register_autograd(backward, setup_context=...)` supplies the gradient.
- `torch.library.opcheck` tests that all of these are consistent: schema, fake vs real outputs, autograd registration, and AOT dispatch with dynamic shapes.

This runs on CPU here, with the softmax kernel (through the simulator) inside:

```python
# custom_op.py - expose a kernel to PyTorch (autograd, torch.compile, opcheck) via torch.library
import torch
from softmax import softmax as softmax_kernel   # the launcher from softmax.py (Triton or tsim)


@torch.library.custom_op("mlek::softmax", mutates_args=())
def softmax_op(x: torch.Tensor) -> torch.Tensor:
    return softmax_kernel(x.contiguous())


@softmax_op.register_fake                       # shape/dtype propagation without running the kernel
def _(x):
    return torch.empty_like(x)


def _setup(ctx, inputs, output):
    ctx.save_for_backward(output)


def _backward(ctx, gy):                         # dsoftmax: y * (gy - sum(gy * y))
    (y,) = ctx.saved_tensors
    return y * (gy - (gy * y).sum(-1, keepdim=True))


softmax_op.register_autograd(_backward, setup_context=_setup)


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(8, 300, requires_grad=True)
    torch.library.opcheck(softmax_op, (x,))      # schema, fake tensor, autograd registration checks
    print("opcheck passed")

    def block(x):
        return (softmax_op(x * 2.0) * torch.linspace(-1, 1, 300)).sum(-1).mean() + x.square().mean()

    compiled = torch.compile(block, fullgraph=True)        # fullgraph: error on any graph break
    out = compiled(x); out.backward()
    g = x.grad.clone(); x.grad = None
    block_ref = lambda x: (torch.softmax(x * 2.0, -1) * torch.linspace(-1, 1, 300)).sum(-1).mean() + x.square().mean()
    out_ref = block_ref(x); out_ref.backward()
    torch.testing.assert_close(out, out_ref); torch.testing.assert_close(g, x.grad)
    print("torch.compile(fullgraph=True) + backward match eager:", round(out.item(), 6))
    print(torch.ops.mlek.softmax.default._schema)
```

```
opcheck passed
torch.compile(fullgraph=True) + backward match eager: 0.925124
mlek::softmax(Tensor x) -> Tensor
```

`mutates_args` must list every input the kernel writes to. The fused cross-entropy, which overwrites its logits, needs `mutates_args=("logits",)`. Getting it wrong breaks `torch.compile`'s functionalization (the pass that rewrites in-place ops into pure ones so the graph can be reordered; see [AOTAutograd](execution-model.md#aotautograd-joint-graph)) silently.

For Triton kernels there is a better option: **`torch.library.triton_op` + `wrap_triton`**. Unlike `custom_op`, the op's body is visible to `torch.compile`, which can trace into the Triton launch, fuse around it and CUDA-graph it. You do not need a separate fake implementation if the body only allocates outputs with PyTorch ops. Untested (needs real Triton):

```python
import torch
from torch.library import triton_op, wrap_triton
from softmax import softmax_row_kernel
import triton

@triton_op("mlek::softmax_t", mutates_args={})
def softmax_t(x: torch.Tensor) -> torch.Tensor:
    x = x.contiguous()
    M, N = x.shape
    y = torch.empty_like(x)
    wrap_triton(softmax_row_kernel)[(M,)](x, y, x.stride(0), y.stride(0), N,
                                          BLOCK_N=triton.next_power_of_2(N))
    return y
# autograd: softmax_t.register_autograd(backward, setup_context=...) exactly as for custom_op
```

For CUDA C++ in a package, register the operator in C++ so it exists without Python bindings, and add the fake in Python (untested). `TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL` are macros from libtorch: the first declares operator `mlek::vadd` and its schema, the second says "on CUDA tensors, run the C++ function `vadd`" (`&vadd` is a pointer to that function):

```cpp
// in the extension's .cu file
TORCH_LIBRARY(mlek, m) { m.def("vadd(Tensor a, Tensor b) -> Tensor"); }
TORCH_LIBRARY_IMPL(mlek, CUDA, m) { m.impl("vadd", &vadd); }
```

```python
torch.ops.load_library("build/mlek_ext.so")       # or load_inline(..., is_python_module=False)
@torch.library.register_fake("mlek::vadd")
def _(a, b):
    return torch.empty_like(a)
y = torch.ops.mlek.vadd(x, x)                       # now traceable by torch.compile
```

Two more things for production. If the kernel uses no Python-specific features, build against the stable ABI subset so one wheel works across PyTorch versions (the ABI, application binary interface, is the binary-level contract between compiled code and the library it calls; a wheel is the prebuilt package file `pip` installs); check the current state in the PyTorch docs, since this is moving (as of 2025-2026). And CUDA-graph-capture your op once in a test: anything that syncs or allocates per call (`.item()`, host-side counts like `n_valid` above) breaks capture.


> **Moved to the reference part:** [CUTLASS, CuTe, ThunderKittens and other kernel DSLs](writing-kernels-reference.md#cutlass-cute-thunderkittens-and-other-kernel-dsls), [A checklist before you open a kernel PR](writing-kernels-reference.md#a-checklist-before-you-open-a-kernel-pr). Read it when you need it.

## You can now read

- [Profiling and optimization](../../training/profiling-and-optimization.md) - finding which kernel matters in a whole training step before you write one.
- [Serving engine internals](../../inference/serving-engine-internals.md#inference-attention-kernels) - paged and split-KV attention kernels, CUDA graphs, MoE serving.
- [Inference arithmetic](../../inference/arithmetic.md) and [training arithmetic](../../training/arithmetic.md) - where the kernels in this chapter sit in the step-time budget.
- [GPU execution model](execution-model.md) - the reference for every "why" in this chapter, in more depth: TMA, clusters, `wgmma` resource arithmetic, FA3.
- Source you can now follow: Triton's tutorials 01-06, [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)'s `ops/`, vLLM's `csrc/` and `vllm/model_executor/layers/fused_moe/`, [FlashAttention](https://github.com/Dao-AILab/flash-attention)'s Triton and CuTe code, CUTLASS's `examples/`.

## Further reading

Worklogs (numbers in this chapter labelled as theirs come from these):
- Simon Boehm, [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM), 2022
- Mark Harris, [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf), NVIDIA, 2007
- Alex Armbruster, [How To Write A Fast Matrix Multiplication From Scratch With Tensor Cores](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html), 2024
- Pranjal Shankhdhar, [Outperforming cuBLAS on H100: a Worklog](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog), 2024

Documentation:
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) - launch syntax, memory, warp intrinsics (`__shfl_*_sync`), WMMA
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) - `mma`, `ldmatrix`, `wgmma`, `cp.async`
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) and [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
- [PyTorch custom C++ and CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html), [Python custom operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html), and [`torch.library`](https://docs.pytorch.org/docs/stable/library.html) (`custom_op`, `triton_op`, `opcheck`)
- [Triton tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) and [`triton.testing`](https://triton-lang.org/main/python-api/triton.testing.html)
- [CUTLASS](https://github.com/NVIDIA/cutlass) (CuTe docs under `media/docs/cute/`, Python DSL under `media/docs/pythonDSL/`)

Papers:
- Tillet, Kung, Cox, "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations", MAPL 2019
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
- Milakov and Gimelshein, "Online normalizer calculation for softmax", [arXiv:1805.02867](https://arxiv.org/abs/1805.02867)
- Spector et al., "ThunderKittens: Simple, Fast, and Adorable AI Kernels", [arXiv:2410.20399](https://arxiv.org/abs/2410.20399)
- Gale et al., "MegaBlocks: Efficient Sparse Training with Mixture-of-Experts", [arXiv:2211.15841](https://arxiv.org/abs/2211.15841)
- Hsu et al., "Liger Kernel: Efficient Triton Kernels for LLM Training", [arXiv:2410.10989](https://arxiv.org/abs/2410.10989) (id from memory)
- Welford, "Note on a method for calculating corrected sums of squares and products", Technometrics 1962; Chan, Golub, LeVeque, "Algorithms for computing the sample variance", 1983

Courses and communities:
- [GPU MODE](https://github.com/gpu-mode) lectures and the kernel leaderboards - the most active public community for exactly this skill
- Kirk, Hwu, El Hajj, *Programming Massively Parallel Processors*, 4th edition, 2022
