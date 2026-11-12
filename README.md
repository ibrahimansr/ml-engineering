# Machine Learning Engineering

A from-zero path through the systems side of modern machine learning: the hardware, operating systems, networks and distributed systems underneath training and serving large models, and the engineering of training runs, inference servers and RL infrastructure on top of them.

It assumes you know Python, the basics of PyTorch (tensors, `nn.Module`, writing a training loop) and the math of deep learning. It assumes nothing else. Read it top to bottom: nothing is used before it is taught, every chapter starts with a **Prerequisites** list, and almost every section has code you can run.

## Table of Contents

**Part 1. Foundations** - systems from zero

1. **[The Working Toolkit](foundations/toolkit.md)** - terminal basics, running the book's code, git, JSON/YAML/TOML, regex, Make, reading C
1. **[Computer Architecture](foundations/computer-architecture.md)** - number formats, CPUs, memory hierarchy, PCIe, what a GPU is, sizing the CPU side of a GPU node
1. **[Linux and Operating Systems](foundations/linux-and-os.md)** - shell, processes, memory, filesystems, ssh, systemd, CUDA versions
1. **[Concurrency and Parallelism](foundations/concurrency.md)** - threads, processes, the GIL, asyncio, DataLoader internals, CUDA streams
1. **[Computer Networking](foundations/networking.md)** - TCP/IP, sockets, DNS/HTTP/TLS, data center networks, RDMA, InfiniBand
1. **[Distributed Systems](foundations/distributed-systems.md)** - failures, clocks, consensus, Raft, leases, linearizability, fault tolerance
1. **[Containers, VMs, MicroVMs and Sandboxing](foundations/containers-and-virtualization.md)** - namespaces, cgroups, Docker, KVM, Firecracker, gVisor, untrusted code
1. **[Storage, Checkpoints and Data Pipelines](foundations/storage-and-data.md)** - file systems, object storage, benchmarking IO, checkpoint I/O, datasets at scale, cleaning up

**Part 2. Model Architectures**

1. **[The Transformer and Its Evolution](models/architectures.md)** - attention, norms, RoPE, SwiGLU, GQA, MLA, sliding windows, MoE - built in code
1. **[Beyond the Standard Transformer](models/beyond-transformers.md)** - linear attention, DeltaNet, SSMs/Mamba, hybrids, multimodal, diffusion - built in code

**Part 3. Hardware and Kernels**

1. **[Accelerators](compute/accelerators.md)** - specs, vendors, TFLOPS, memory, achievable FLOPS and how to measure them
1. **[GPU Execution Model](compute/accelerator/execution-model.md)** - SMs, warps, memory hierarchy, tensor cores, rooflines, FlashAttention, torch.compile
1. **[Writing GPU Kernels](compute/accelerator/writing-kernels.md)** - CUDA C++ and Triton, from vector add to tensor-core GEMM and FlashAttention
1. **[The GPU Node Software Stack](compute/gpu-software-stack.md)** - drivers, CUDA, NCCL, RDMA, node bring-up and GPU troubleshooting
1. **[Interconnect Hardware](network/interconnects.md)** - NVLink, NVSwitch, InfiniBand, RoCE, EFA, SHARP and what they really deliver
1. **[Collective Communication Internals](network/collectives-internals.md)** - communication patterns, ring/tree algorithms, NCCL internals, cost models
1. **[Measuring and Debugging the Network](network/measuring-and-debugging.md)** - benchmarks, NCCL diagnostics, performance settings

**Part 4. Orchestration**

1. **[Slurm From Zero](orchestration/slurm/from-zero.md)** - architecture, jobs, multi-node PyTorch, preemption, day-to-day recipes, admin basics
1. **[Kubernetes From Zero](orchestration/kubernetes-from-zero.md)** - objects, controllers, operators, GPU workloads
1. **[Orchestration Internals](orchestration/internals.md)** - gang scheduling, topology, elastic launch, failure arithmetic

**Part 5. Training**

1. **[Training Arithmetic and Numerics](training/arithmetic.md)** - FLOPs, memory, data types, mixed precision, FP8/FP4, optimizers
1. **[Parallelism From First Principles](training/model-parallelism/derivations.md)** - DDP, ZeRO/FSDP, TP, SP/CP, PP, EP, composing them, implementations
1. **[Profiling and Optimizing Training](training/profiling-and-optimization.md)** - finding the bottleneck, MFU, memory, model sizing, DataLoader, NUMA
1. **[Training Stability](training/stability.md)** - hyperparameters, initialization, instabilities, reading loss curves
1. **[Fault Tolerance, Checkpoints and Reproducibility](training/fault-tolerance-and-reproducibility.md)** - surviving failures and preemption, checkpoints, determinism
1. **[Training Across Datacenters and the Internet](training/decentralized-training.md)** - DiLoCo, compression, elastic and decentralized training

**Part 6. Inference**

1. **[Inference Arithmetic](inference/arithmetic.md)** - vocabulary and metrics, rooflines, KV cache sizing, quantization, capacity planning
1. **[Inside a Serving Engine](inference/serving-engine-internals.md)** - paged attention, scheduling, speculative decoding, disaggregation
1. **[Serving LLMs in Practice](inference/serving-hands-on.md)** - vLLM/SGLang, choosing a framework, load testing, gateways, multi-node serving
1. **[Serving Efficiently on a Fixed Fleet](inference/serving-efficiently.md)** - tuning engines for your traffic, prefix-aware routing, speculative decoding in production, packing models, long-context and multimodal serving

**Part 7. Platforms**

1. **[Control Planes and Fleet Management](platform/control-planes.md)** - placement, node agents, leases, reconciliation, operating services
1. **[Model Deploys, Recovery and Autoscaling](platform/deploys-recovery-autoscaling.md)** - where startup time goes and how to cut it, rolling deploys, GPU failure recovery, autoscaling, what infrastructure to build on
1. **[Operating an Inference Service](platform/operating-inference.md)** - rate limits, fair scheduling across tenants, SLOs and goodput, batch work, cost per token, quality monitoring, incidents
1. **[Infrastructure for Large-Scale RL](platform/rl-infrastructure.md)** - generators vs trainers, async RL, weight sync, environments and sandboxes
1. **[Getting Compute](platform/getting-compute.md)** - renting vs buying, choosing a cloud provider, evaluating a cluster, when to upgrade GPUs
1. **[Running a Small Inference Provider](platform/small-inference-provider.md)** - a worked example on two leased 8xB300 nodes: placement, training-to-serving, isolation, failure planning, the first 30 days

**Part 8. Development**

1. **[Debugging PyTorch Programs](development/debugging-pytorch.md)** - tiny models, memory, tensors, crashes, git bisect
1. **[Debugging Distributed Programs and Performance](development/debugging-distributed.md)** - hangs across ranks, NCCL, multi-node, profiling
1. **[Writing and Running Tests](development/testing.md)** - pytest for ML code, including GPU and distributed tests

**Appendix**

1. **[Further Reading and Quick Reference](appendix/further-reading.md)** - comparison tables, tools, training logbooks, lessons learned

### Reading paths

After Part 1, pick the path closest to the work you want to do. Each path lists its chapters in book order, so prerequisites always come first:

- **Distributed training:** Collective Communication Internals → Slurm From Zero → Training Arithmetic → Parallelism From First Principles → Profiling and Optimizing Training → Training Stability → Fault Tolerance → Training Across Datacenters
- **Inference:** Kubernetes From Zero → Inference Arithmetic → Inside a Serving Engine → Serving LLMs in Practice → Serving Efficiently on a Fixed Fleet → Control Planes → Model Deploys, Recovery and Autoscaling → Operating an Inference Service → Running a Small Inference Provider
- **RL infrastructure:** Parallelism From First Principles → Inference Arithmetic → Inside a Serving Engine → Infrastructure for Large-Scale RL (the sandboxing material is in Part 1's Containers chapter)
- **Sandboxes and compute platforms:** The GPU Node Software Stack → Kubernetes From Zero → Orchestration Internals → Control Planes → Getting Compute (building on Part 1's Containers, VMs and MicroVMs)
- **Kernels and performance:** Accelerators → GPU Execution Model → Writing GPU Kernels → Profiling and Optimizing Training → Inside a Serving Engine (attention kernels)
- **Model architectures:** Part 2, then Inference Arithmetic for what each design costs at serving time

## Reading it

- **On GitHub:** start at [The Working Toolkit](foundations/toolkit.md). Equations render natively.
- **As a website:** the `site/` folder is a ready-to-serve static site with a sidebar, search filter and rendered equations. See [Building the book](build/README.md).
- **As a PDF:** `make pdf` (see [Building the book](build/README.md)).

## License and credits

This book is a fork of, and substantially rewrites, Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering). Material adapted from it is marked at the end of each chapter that uses it. Thanks to Stas Bekman and [the original book's contributors](https://github.com/stas00/ml-engineering/blob/master/contributors.md).

The content is distributed under [Attribution-ShareAlike 4.0 International](LICENSE-CC-BY-SA).
