# Machine Learning Engineering Open Book

<!-- not-in-ebook-start -->
<a href="#ebook-versions-of-the-book"><img src="images/Machine-Learning-Engineering-book-cover-200x259.png" align="right" width="200" height="259" alt="Machine Learning Engineering Open Book cover"></a>
<!-- not-in-ebook-end -->

This is an open collection of methodologies, tools and step by step instructions to help with successful training and fine-tuning of large language models and multi-modal models and their inference.

This is a technical material suitable for LLM/VLM training engineers and operators. That is the content here contains lots of scripts and copy-n-paste commands to enable you to quickly address your needs.

This repo is an ongoing brain dump of my experiences training Large Language Models (LLM) (and VLMs); a lot of the know-how I acquired while training the open-source [BLOOM-176B](https://huggingface.co/bigscience/bloom) model in 2022 and [IDEFICS-80B](https://huggingface.co/HuggingFaceM4/idefics-80b-instruct) multi-modal model in 2023, and RAG models at [Contextual.AI](https://contextual.ai/) in 2024.

I've been compiling this information mostly for myself so that I could quickly find solutions I have already researched in the past and which have worked, but as usual I'm happy to share these notes with the wider ML community.


## Table of Contents

New to systems? Read Part 1 in order first - every later chapter links back to it. The new chapters are written to be read in the order listed: nothing is used before it is taught or defined. Each chapter starts with a **Prerequisites** list, so you can also jump straight to a topic and follow the links backwards.

Entries marked *(reference)* are Stas Bekman's original chapters: practical notes written for experienced engineers. Read them after the matching from-zero chapters, or use them as lookup material.

**Part 1. Foundations** - systems from zero

1. **[The Working Toolkit](./foundations/toolkit.md)** - terminal basics, running the book's code, git, JSON/YAML/TOML, regex, Make, reading C
1. **[Computer Architecture](./foundations/computer-architecture.md)** - number formats, CPUs, memory hierarchy, PCIe, what a GPU is
1. **[Linux and Operating Systems](./foundations/linux-and-os.md)** - shell, processes, memory, filesystems, ssh, systemd, CUDA versions
1. **[Concurrency and Parallelism](./foundations/concurrency.md)** - threads, processes, the GIL, asyncio, DataLoader internals, CUDA streams
1. **[Computer Networking](./foundations/networking.md)** - TCP/IP, sockets, DNS/HTTP/TLS, data center networks, RDMA, InfiniBand
1. **[Distributed Systems](./foundations/distributed-systems.md)** - failures, clocks, consensus, leases, rendezvous, fault tolerance
1. **[Containers, VMs, MicroVMs and Sandboxing](./foundations/containers-and-virtualization.md)** - namespaces, cgroups, Docker, KVM, Firecracker, gVisor, untrusted code
1. **[Storage, Checkpoints and Data Pipelines](./foundations/storage-and-data.md)** - block/file/object storage, safetensors, checkpoint I/O, datasets at scale

**Part 2. Model Architectures**

1. **[The Transformer and Its Evolution](./models/architectures.md)** - attention, norms, RoPE, SwiGLU, GQA, MLA, sliding windows, MoE - built in code
1. **[Beyond the Standard Transformer](./models/beyond-transformers.md)** - linear attention, DeltaNet, SSMs/Mamba, hybrids, multimodal, diffusion - built in code

**Part 3. Insights**

1. **[The AI Battlefield Engineering](./insights/ai-battlefield.md)** - what you need to know in order to succeed
1. **[How to Choose a Cloud Provider](./insights/how-to-choose-cloud-provider.md)** - questions that will empower you to have a successful compute cloud experience
1. **[When Is It Worth Upgrading GPUs?](./insights/when-to-upgrade-gpus/README.md)** - a framework for deciding whether a GPU upgrade is worth its cost

**Part 4. Hardware and Kernels**

1. **[Compute](compute)** - accelerators, CPUs, CPU memory *(reference)*
1. **[GPU Execution Model](compute/accelerator/execution-model.md)** - SMs, warps, memory hierarchy, tensor cores, rooflines, FlashAttention, torch.compile
1. **[Writing GPU Kernels](compute/accelerator/writing-kernels.md)** - CUDA C++ and Triton, from vector add to tensor-core GEMM and FlashAttention
1. **[The GPU Node Software Stack](compute/gpu-software-stack.md)** - drivers, CUDA, NCCL, RDMA and node bring-up
1. **[Storage](storage)** - local, distributed and shared file systems *(reference)*
1. **[Network](network)** - intra- and inter-node networking *(reference)*
1. **[Collective Communication Internals](network/collectives-internals.md)** - ring/tree algorithms, NCCL internals, cost models

**Part 5. Orchestration**

1. **[Orchestration Systems](orchestration)** - overview
1. **[Slurm From Zero](orchestration/slurm/from-zero.md)** - architecture, jobs, multi-node PyTorch, preemption, admin basics
1. **[SLURM](orchestration/slurm)** - cheatsheets, admin, performance, launchers *(reference)*
1. **[Kubernetes From Zero](orchestration/kubernetes-from-zero.md)** - objects, controllers, operators, GPU workloads
1. **[Orchestration Internals](orchestration/internals.md)** - gang scheduling, topology, elastic launch, failure arithmetic

**Part 6. Training**

1. **[Training](training)** - model training-related guides *(reference)*
1. **[Training Arithmetic and Numerics](training/arithmetic.md)** - FLOPs, memory, mixed precision, FP8, optimizers
1. **[Parallelism From First Principles](training/model-parallelism/derivations.md)** - DDP, ZeRO/FSDP, TP, SP/CP, PP, EP, composing them
1. **[Profiling and Optimizing Training](training/profiling-and-optimization.md)** - hands-on bottleneck hunting
1. **[Training Across Datacenters and the Internet](training/decentralized-training.md)** - DiLoCo, compression, elastic and decentralized training

**Part 7. Inference**

1. **[Inference](inference)** - model inference insights *(reference)*
1. **[Inference Arithmetic](inference/arithmetic.md)** - rooflines, KV cache sizing, quantization, capacity planning
1. **[Inside a Serving Engine](inference/serving-engine-internals.md)** - paged attention, scheduling, speculative decoding, disaggregation
1. **[Serving LLMs in Practice](inference/serving-hands-on.md)** - vLLM/SGLang, load testing, gateways, multi-node serving

**Part 8. Platforms**

1. **[Control Planes and Fleet Management](platform/control-planes.md)** - placement, node agents, leases, reconciliation, operating services
1. **[Infrastructure for Large-Scale RL](platform/rl-infrastructure.md)** - generators vs trainers, async RL, weight sync, environments and sandboxes

**Part 9. Development**

1. **[Debugging and Troubleshooting](debug)** - how to debug easy and difficult issues
1. **[And more debugging](https://github.com/stas00/the-art-of-debugging)**
1. **[Testing](testing)** - numerous tips and tools to make test writing enjoyable

**Part 10. Miscellaneous**

1. **[Resources](resources)** - LLM/VLM chronicles

### Reading paths

After Part 1, pick the path closest to the work you want to do:

- **Distributed training:** Training Arithmetic → Parallelism From First Principles → Collective Communication Internals → Profiling and Optimizing Training → Training Across Datacenters → Slurm From Zero
- **Inference:** Inference Arithmetic → Inside a Serving Engine → Serving LLMs in Practice → Kubernetes From Zero → Control Planes
- **RL infrastructure:** Inference Arithmetic → Inside a Serving Engine → Parallelism From First Principles → Infrastructure for Large-Scale RL → Containers/Sandboxing
- **Sandboxes and compute platforms:** Containers, VMs and MicroVMs → Kubernetes From Zero → Control Planes → The GPU Node Software Stack → Orchestration Internals
- **Kernels and performance:** GPU Execution Model → Writing GPU Kernels → Profiling and Optimizing Training → Inside a Serving Engine (attention kernels)
- **Model architectures:** Part 2, then Inference Arithmetic for what each design costs at serving time

## Updates

I announce any significant updates on my twitter channel [https://twitter.com/StasBekman](https://twitter.com/StasBekman).

## Ebook versions of the book

You can download various ebook formats of this book:
* [PDF](https://huggingface.co/stas/ml-engineering-book/resolve/main/Stas%20Bekman%20-%20Machine%20Learning%20Engineering.pdf?download=true)
* [EPUB](https://huggingface.co/stas/ml-engineering-book/resolve/main/Stas%20Bekman%20-%20Machine%20Learning%20Engineering.epub?download=true)


I will try to rebuild these once in a few weeks or so, but if you want the latest ebook versions, the instructions for building are [here](build).

Thanks to HuggingFace for giving me permission to host my book's ebook formats at the [HF hub](https://huggingface.co/stas/ml-engineering-book).


## Skills for AI agents

I maintain two [SKILL.md](skills/README.md) files that you can use to teach your AI agent:

- [Machine Learning Engineering](skills/ml-engineering/SKILL.md) — train and operate large-scale ML models.
- [Evaluate a GPU cluster](skills/evaluate-cluster/SKILL.md) — trial-node / acceptance loop: compute, network, and storage, then a markdown report.

See also the companion skills: [The Art of Debugging](https://github.com/stas00/the-art-of-debugging/blob/master/SKILL.md) and [Stas' Python Cookbook](https://github.com/stas00/python-cookbook/blob/master/SKILL.md).


## Courses

- **[Lessons Learned from Training LLMs](courses/lesson-learned)** - provides a very different way of reading my open books, by going over the terse learned insights and allowing you to quickly dive deeper when you need to.

## Lectures/Talks

- [Building resilient ML Engineering skills](https://www.youtube.com/watch?v=IBJUt9JPKHk) given on 2026-01-10 for the [GPU Mode community](https://github.com/gpu-mode). Only had time to discuss performance reality of accelerators, network and storage and how each of them can be crucial to the ensemble's performance. Thanks to [Mark Saroufim](https://github.com/msaroufim) for organizing and providing an awesome support during the talk.

## Discussions

If you want to discuss something related to ML engineering this repo has the [community discussions](https://github.com/stas00/ml-engineering/discussions) available - so please don't hesitate to share your experience or start a new discussion about something you're passionate about.


## Key comparison tables

High end accelerators:

- [Theoretical accelerator TFLOPS](compute/accelerator/README.md#tflops-comparison-table)
- [Accelerator memory size and speed](compute/accelerator/README.md#accelerator-memory-size-and-speed)

Networks:

- [Theoretical inter-node speed](network/README.md#inter-node-networking)
- [Theoretical intra-node speed](network/README.md#intra-node-networking)

## Shortcuts

Things that you are likely to need to find quickly and often.

Tools:

- [torch-dist-bench.py](network/benchmarks/torch-dist-bench.py) - a much easier way to benchmark network throughput than nccl-tests (used to be called `all_reduce_bench.py`).
- [torch-distributed-gpu-test.py](debug/torch-distributed-gpu-test.py) - a tool to quickly test your inter-node connectivity
- [mamf-finder.py](compute/accelerator/benchmarks/mamf-finder.py) - what is the actual TFLOPS measurement you can get from your accelerator.

Guides:

- [debugging pytorch applications](debug/pytorch.md) - quick copy-n-paste solutions to resolve hanging or breaking pytorch applications
- [slurm for users](orchestration/slurm/users.md) - a slurm cheatsheet and tricks
- [make tiny models/datasets/tokenizers](debug/pytorch.md#faster-debug-and-development-with-tiny-models-tokenizers-and-datasets)
- [LLM/VLM chronicles collection](resources/README.md#publicly-available-training-llmvlm-logbooks)


## Gratitude

None of this would have been possible without me being entrusted with doing the specific LLM/VLM trainings I have learned the initial know-how from. This is a privilege that only a few enjoy due to the prohibitively expensive cost of renting huge ML compute clusters. So hopefully the rest of the ML community will vicariously learn from these notes.

Special thanks go to [Thom Wolf](https://github.com/thomwolf) who proposed that I lead the BLOOM-176B training back when I didn't know anything about large scale training. This was the project that catapulted me into the intense learning process. And, of course, HuggingFace for giving me the opportunity to work full time on BLOOM-176B and later on IDEFICS-80B trainings.

Recently, I continued expanding my knowledge and experience while training models and building scalable training/inference systems at [Contextual.AI](https://contextual.ai/) and I'm grateful for that opportunity to Aman and Douwe.

I'd also like to thank the numerous [contributors](contributors.md) who have been making this text awesome and error-free.

## Contributing

If you found a bug, typo or would like to propose an improvement please don't hesitate to open an [Issue](https://github.com/stas00/ml-engineering/issues) or contribute a PR.

## Related Open Books

- [The Art of Debugging Open Book](https://github.com/stas00/the-art-of-debugging) — methodologies and recipes for debugging Unix, Python and PyTorch programs.
- [Stas' Python Cookbook](https://github.com/stas00/python-cookbook) — everyday Python and standard-library recipes.

## License

The content of this site is distributed under [Attribution-ShareAlike 4.0 International](LICENSE-CC-BY-SA).


## Citation

```bibtex
@misc{bekman2024mlengineering,
  author = {Bekman, Stas},
  title = {Machine Learning Engineering Open Book},
  year = {2023-2026},
  publisher = {Stasosphere Online Inc.},
  journal = {GitHub repository},
  url = {https://github.com/stas00/ml-engineering}
}
```

## My repositories map

✔ **Books:**
 [Machine Learning Engineering](https://github.com/stas00/ml-engineering) |
 [The Art of Debugging](https://github.com/stas00/the-art-of-debugging) |
 [Stas' Python Cookbook](https://github.com/stas00/python-cookbook)

✔ **Applications:**
 [ipyexperiments](https://github.com/stas00/ipyexperiments)

✔ **Tools and Cheatsheets:**
 [bash](https://github.com/stas00/bash-tools) |
 [conda](https://github.com/stas00/conda-tools) |
 [git](https://github.com/stas00/git-tools) |
 [jupyter-notebook](https://github.com/stas00/jupyter-notebook-tools) |
 [make](https://github.com/stas00/make-tools) |
 [python](https://github.com/stas00/python-tools) |
 [tensorboard](https://github.com/stas00/tensorboard-tools) |
 [unix](https://github.com/stas00/unix-tools)

✔ **Other Machine Learning:**
 [ML ways](https://github.com/stas00/ml-ways) |
 [Porting](https://github.com/stas00/porting)
