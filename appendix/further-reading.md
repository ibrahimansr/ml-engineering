# Appendix: Further Reading and Quick Reference

This appendix is the part of the book you come back to while working. It has three jobs:

1. **Quick reference.** Where the comparison tables (accelerator TFLOPS, memory size and speed, intra- and inter-node network speeds) and the handful of tools you reach for most often (`torch-dist-bench.py`, `torch-distributed-gpu-test.py`, `mamf-finder.py`) live in this book, with the one command you need to run each.
2. **Further reading.** Other guides worth your time, a few useful compilations, and every public training logbook and hardware-setup logbook we know of, with a note on what makes each one worth reading.
3. **Lessons learned.** A list of hard-won lessons from training and operating large models (BLOOM-176B in 2022, IDEFICS-80B in 2023, production RAG and RL systems at Contextual.AI in 2024, and Arctic Training open source work at Snowflake in 2025-2026). Each lesson is short, says what happened and what to do about it, and links to the chapter that teaches the material behind it. Read the bold headlines to find what concerns you, then follow the link.

Conventions, as in the rest of the book: blocks marked **[ran]** were executed while writing this appendix on a MacBook (Apple M4, Python 3.14) and the output shown is from that run; blocks marked **[GPU, untested]** need GPUs and were not run here.

## Prerequisites

This appendix assumes you have read the whole book; every term in it is taught in an earlier chapter, and each entry links back to where. The ones used most:

- [Accelerators](../compute/accelerators.md): [TFLOPS](../compute/accelerators.md#tflops), [MAMF and MSMF](../compute/accelerators.md#mamf-and-msmf) (the measured matmul ceilings).
- [Interconnects](../network/interconnects.md): [unidirectional vs duplex bandwidth](../network/interconnects.md#unidirectional-vs-bidirectional-duplex), [SHARP](../network/interconnects.md#sharp), [intra-node](../network/interconnects.md#intra-node-networking) and [inter-node](../network/interconnects.md#inter-node-networking) networking.
- [Measuring and debugging networks](../network/measuring-and-debugging.md): [torch-dist-bench](../network/measuring-and-debugging.md#torch-dist-bench).
- [Storage and data](../foundations/storage-and-data.md), [Slurm from zero](../orchestration/slurm/from-zero.md), [Profiling and optimization](../training/profiling-and-optimization.md), [Training stability](../training/stability.md), [Fault tolerance and reproducibility](../training/fault-tolerance-and-reproducibility.md), [Getting compute](../platform/getting-compute.md).
- [Linux and the OS](../foundations/linux-and-os.md): [gdb basics](../foundations/linux-and-os.md#gdb-basics), [py-spy](../foundations/linux-and-os.md#py-spy-what-is-python-doing-right-now), [shared libraries and LD_LIBRARY_PATH](../foundations/linux-and-os.md#shared-libraries-and-ld_library_path).


## Quick reference

### Comparison tables

The tables you look up most often while choosing hardware or checking a measurement against the spec. All of them are theoretical (vendor) numbers except the MAMF table, which is measured; [Unidirectional vs bidirectional (duplex)](../network/interconnects.md#unidirectional-vs-bidirectional-duplex) explains why you must check which kind of bandwidth a network number is before comparing it to anything.

| What you want to look up | Where it lives |
|---|---|
| Theoretical dense TFLOPS per accelerator and dtype | [TFLOPS comparison table](../compute/accelerators.md#tflops-comparison-table) |
| What you can actually get from a matmul (measured) | [MAMF and MSMF comparison table](../compute/accelerators.md#mamf-and-msmf-comparison-table) |
| Accelerator memory size and bandwidth | [Accelerator memory size and speed](../compute/accelerators.md#accelerator-memory-size-and-speed) |
| Intra-node (GPU-to-GPU inside one box) speed | [Intra-node networking](../network/interconnects.md#intra-node-networking) |
| Inter-node (box-to-box) speed | [Inter-node networking](../network/interconnects.md#inter-node-networking) |

A habit that saves you from most mistakes with these tables: express every measurement as a fraction of the matching spec. Healthy systems land around 80% of the unidirectional network spec and, depending on the accelerator, roughly 60-85% of the theoretical matmul TFLOPS (see the MAMF table for your chip):

```python
# spec_fraction.py - is my measurement healthy?
measured_busbw, spec_unidir = 376, 450        # GBps, all-reduce on 8x H200 vs NVLink 4 unidirectional spec
measured_tflops, spec_tflops = 1745, 2250     # B200 bf16 matmul, measured vs theoretical
print(f"network {measured_busbw / spec_unidir:.0%} of spec, matmul {measured_tflops / spec_tflops:.0%} of spec")
```

**[ran]** output:

```text
network 84% of spec, matmul 78% of spec
```

How to read it: 84% of the unidirectional spec is a healthy network; if you had divided by the duplex number (900GBps) you would have seen 42% and gone looking for a problem that isn't there. For matmul, compare against the measured MAMF of your chip, not the theoretical peak.

### Tools

Three scripts in this repository you will run again and again. Each is explained in full in the linked section; here is just the command.

| Tool | What it answers | Explained in |
|---|---|---|
| [torch-dist-bench.py](../network/benchmarks/torch-dist-bench.py) | What network bandwidth do collectives actually get? An easier alternative to nccl-tests (it used to be called `all_reduce_bench.py`). | [torch-dist-bench](../network/measuring-and-debugging.md#torch-dist-bench), [Running the benchmark across nodes](../network/measuring-and-debugging.md#running-the-benchmark-across-nodes) |
| [torch-distributed-gpu-test.py](../debug/torch-distributed-gpu-test.py) | Can all my GPUs, within a node and across nodes, talk to each other at all? | [Use a small test script, not your application](../network/measuring-and-debugging.md#use-a-small-test-script-not-your-application) |
| [mamf-finder.py](../compute/accelerator/benchmarks/mamf-finder.py) | What TFLOPS can a matmul really reach on this accelerator and software stack? | [The MAMF finder: `mamf-finder.py`](../compute/accelerators.md#the-mamf-finder-mamf-finderpy) |

**[GPU, untested]** (run from the repository root):

```bash
# network bandwidth, one node, 8 GPUs
python -u -m torch.distributed.run --nproc_per_node 8 --rdzv_endpoint localhost:6000 --rdzv_backend c10d \
    network/benchmarks/torch-dist-bench.py

# connectivity check with NCCL's setup logged
NCCL_DEBUG=INFO torchrun --nproc_per_node 8 --nnodes 1 debug/torch-distributed-gpu-test.py

# maximum achievable matmul FLOPS, auto search, log to a timestamped file
python compute/accelerator/benchmarks/mamf-finder.py --output_file=$(date +'%Y-%m-%d-%H:%M:%S').txt
```

How to read the outputs: `torch-dist-bench.py` prints `busbw` per payload size, which you compare to the unidirectional spec as above; `torch-distributed-gpu-test.py` prints one OK line per rank, and a hang or missing rank is the finding; `mamf-finder.py` prints the best TFLOPS it found and the matmul shape that reached it, which you compare to the [MAMF table](../compute/accelerators.md#mamf-and-msmf-comparison-table).

### Guides

| You need | Go to |
|---|---|
| Copy-paste fixes for a PyTorch program that hangs or crashes | [Debugging PyTorch](../development/debugging-pytorch.md), and for multi-GPU / multi-node hangs [Debugging distributed](../development/debugging-distributed.md) |
| A Slurm cheatsheet and tricks | [Day-to-day user recipes](../orchestration/slurm/from-zero.md#day-to-day-user-recipes) |
| Tiny models, tokenizers and datasets for fast debugging | [Debugging PyTorch](../development/debugging-pytorch.md) |
| Other teams' LLM/VLM training chronicles | [Public LLM and VLM training logbooks](#public-llm-and-vlm-training-logbooks) below |


## Similar online guides

Other long-form guides that cover parts of this book's ground from a different angle. Reading a second explanation of a hard topic (pipeline parallelism, the roofline model) is one of the fastest ways to make it stick.

- **Boris Dayma, [A Recipe for Training Large Models](https://wandb.ai/craiyon/report/reports/Recipe-Training-Large-Models--VmlldzozNjc4MzQz).** Practical advice from training the Craiyon (DALL-E mini) models; a compact companion to [Training stability](../training/stability.md).
- **Hugging Face, [The Ultra-Scale Playbook: Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook).** Data, tensor, pipeline, context and expert parallelism built up step by step with measurements; pairs with [Model parallelism: derivations](../training/model-parallelism/derivations.md).
- **Jacob Austin et al., [How to Scale Your Model: A Systems View of LLMs on TPUs](https://jax-ml.github.io/scaling-book/).** Rooflines, sharding and the arithmetic of training and inference, worked out for TPUs; the arithmetic transfers directly to GPUs ([Training arithmetic](../training/arithmetic.md), [Inference arithmetic](../inference/arithmetic.md)).
- **Harvard Edge, [Machine Learning Systems](https://github.com/harvard-edge/cs249r_book)** (the CS249r textbook; site: [mlsysbook.ai](http://mlsysbook.ai/)). A full university-course textbook on ML systems, broader and gentler than this book.


## Useful compilations

- **[Common LLM Settings spreadsheet](https://docs.google.com/spreadsheets/d/14vbBbuRMEHoqeuMHkTfw3uiZVmyXNuoSp8s-aHvfvZk/edit#gid=0)**, created by [@StellaAthena](https://github.com/StellaAthena). Hyperparameters and settings (sizes, learning rates, batch sizes, and more) of many known LLM trainings in one table. Very useful when you are about to start a new LLM training and want to see what others chose at your scale ([Training stability](../training/stability.md) explains why those choices matter).
- **[Model pre-training precision database](https://discuss.huggingface.co/t/model-pre-training-precision-database-fp16-fp32-bf16/5671)**, started by Stas Bekman: which dtype (fp16, bf16, fp32) various models were pretrained in. It covers only a handful of models, but is still useful if you research dtypes. He used it while trying to write a [model pretraining dtype auto-detector](https://github.com/stas00/ml-ways/blob/master/numbers/detect-model-pretrained-in-bf16-fp16-fp32.ipynb), and there is a related [float16 vs bfloat16 numerical properties comparison](https://github.com/stas00/ml-ways/blob/master/numbers/bfloat16-vs-float16-study.ipynb). Why the pretraining dtype matters when you fine-tune is covered in [Training arithmetic](../training/arithmetic.md).


## Public LLM and VLM training logbooks

A **VLM** is a vision-language model: an LLM that also takes images as input (see [Multimodal LLMs](../models/beyond-transformers.md#multimodal-llms)). A **logbook** (or **chronicle**) is the running diary a team keeps during a training run: what broke, what they tried, what the loss did.

Logbooks, chronicles and technical reports of LLM/VLM trainings are among the best sources to learn from about training instabilities, infrastructure and scaling problems, and choosing good hyperparameters. The older entries below are diaries kept during the run. Increasingly the same lessons appear as dedicated sections of technical reports, typically titled *Infrastructure, Scaling, and Efficiency* or *Engineering Challenges and Solutions*, and often covering reliability and operational problems that never make it into a model card.

How to read one efficiently: the loss curve tells you *when* something happened, and the diary tells you *why*. Search for the words that mark incidents, then read around them. For a chronicle you have downloaded as markdown, for example:

```bash
# list incident-like lines in a chronicle, with line numbers to jump to (untested on these exact files)
grep -n -i -E "spike|diverg|nan|hang|restart|rollback|lr |skip" chronicles.md | head -40
```

Each hit is a place where the team noticed something; the paragraphs after it are usually the diagnosis and the fix. [Training loss patterns](../training/stability.md) teaches you to recognize the curves they describe.

If you know of a public logbook, or a technical report with such a section, that is missing here, the upstream book welcomes it via [an Issue](https://github.com/stas00/ml-engineering/issues) or a PR. The list is grouped by year and in no particular order within a year. Several of the original repositories have been moved or deleted over time, so Stas Bekman keeps backups; they are listed next to each entry.

### 2021

- **BigScience pre-BLOOM 104B training experiments** (the repository directory says 104B; it is often referred to as 108B): [chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr8-104B-wide/chronicles.md) | [the full spec and discussions](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr8-104B-wide) (backup: [1](https://github.com/stas00/bigscience-backup/blob/master/train/tr8-104B-wide/chronicles.md) | [2](https://github.com/stas00/bigscience-backup/tree/master/train/tr8-104B-wide)). Worth reading because it *failed*: repeated early divergence, with fp16 and uncleaned web data as the main suspects. The post-mortem became the recipe for BLOOM-176B ([A very failed training](../training/stability.md#a-very-failed-training)).

### 2022

- **BigScience BLOOM-176B**: [chronicles-prequel](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles-prequel.md) | [chronicles](https://github.com/bigscience-workshop/bigscience/blob/master/train/tr11-176B-ml/chronicles.md) | [the full spec and discussions](https://github.com/bigscience-workshop/bigscience/tree/master/train/tr11-176B-ml) (backup: [1](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles-prequel.md) | [2](https://github.com/stas00/bigscience-backup/blob/master/train/tr11-176B-ml/chronicles.md) | [3](https://github.com/stas00/bigscience-backup/tree/master/train/tr11-176B-ml)). The prequel covers the weeks of throughput tuning before launch; the chronicles cover a ~3-month run on 384 A100s, hardware failures, the kill switch and watchdogs. Many of the lessons below come from here.
- **Meta OPT-175B**: [logbook](https://github.com/facebookresearch/metaseq/tree/main/projects/OPT/chronicles) | [video](https://www.youtube.com/watch?v=p9IxoSkvZ-M) (backup: [1](https://github.com/stas00/metaseq-backup/tree/main/projects/OPT/chronicles)). Unusually candid: many restarts, loss spikes handled by rolling back and lowering the learning rate, and a lot of hardware trouble. The best single document on what manual babysitting of a large run looks like.
- **THUDM GLM-130B**: [English logbook](https://github.com/zai-org/GLM-130B/blob/main/logs/main-log-en.md) | [Mandarin version](https://github.com/zai-org/GLM-130B/blob/main/logs/main-log.md) (backup: [1](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log-en.md) | [2](https://github.com/stas00/GLM-130B-backup/blob/main/logs/main-log.md)). A bilingual 130B model; the log documents the stability tricks they converged on, a useful counterpoint to the BLOOM and OPT choices.

### 2023

- **Hugging Face IDEFICS-80B** (multimodal, an open reproduction of DeepMind's Flamingo): [learning log](https://github.com/huggingface/m4-logs/blob/master/memos/README.md) | [training chronicles](https://github.com/huggingface/m4-logs/blob/master/tr-190-80b/chronicles.md) (backup: [1](https://github.com/stas00/m4-logs-backup/blob/master/memos/README.md) | [2](https://github.com/stas00/m4-logs-backup/blob/master/tr-190-80b/chronicles.md)). Read it for the VLM-specific pain: streaming a multi-TB image-text dataset from cloud storage, DataLoader memory, and a slow CPU memory leak (lessons 1.4, 5.2 and 6.7 below).
- **BloombergGPT 50B**: section C of [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564). A training chronicle inside a paper appendix: the loss-curve incidents and how they were handled.

### 2024

- **ByteDance, [MegaScale: Scaling Large Language Model Training to More Than 10,000 GPUs](https://arxiv.org/abs/2402.15627).** Covers many training problems and their resolutions at more than 10,000 GPUs: diagnosis tooling, stragglers, network tuning, fault tolerance. The models are proprietary, but the infrastructure lessons are just as useful.
- **Imbue, [From bare metal to a 70B model: infrastructure set-up and scripts](https://imbue.com/blog/70b-infrastructure).** A very detailed technical post on the training problems they had to overcome while training a proprietary 70B model (it also appears under [Hardware setup logbooks](#hardware-setup-logbooks)).
- **Meta, [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783).** [§3.3 Infrastructure, Scaling, and Efficiency](https://arxiv.org/pdf/2407.21783#page=12) covers the training stack, including §3.3.4 *Reliability and Operational Challenges*. Unique among these reports: it documents how the environment perturbed training throughput, through diurnal (day-night) temperature swings and through the fluctuating power draw of tens of thousands of GPUs stressing the datacenter's power grid.

### 2025

- **Project Apertus** (ETH Zürich, EPFL, CSCS), [Apertus v1 Technical Report: Democratizing Open and Compliant LLMs for Global Language Environments](https://arxiv.org/abs/2509.14233). A fully open 8B/70B model suite trained on 15T tokens across more than 1800 languages, using up to 4096 GPUs on CSCS's Alps supercomputer. Its *Infrastructure, Scaling, and Efficiency* and *Engineering Challenges and Solutions* sections (~5 pages) read like a logbook of the run.

### 2026

- **Ai2, [Supercharging Olmo-core for Efficient and Scalable MoE Training](https://allenai.org/papers/olmocore3).** Practical, current recommendations for training large [Mixture of Experts](../models/architectures.md#mixture-of-experts) (MoE) models.
- **Marin 535B-A23B MoE, [live tracker](https://mtracker.oa.dev/hero-run-535b).** "535B-A23B" means 535B total parameters of which 23B are active per token, the usual way of naming an MoE model. Each short tracker entry links to a GitHub Issue with the details, so you can follow the run as it happens. Retrospectives of finished Marin trainings: [8B](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/), [32B](https://marin.readthedocs.io/en/latest/reports/marin-32b-retro/).
- **Project Apertus, [Apertus v1.5 Technical Report: From Open Text Models to Open Multimodal Reasoners](https://arxiv.org/abs/2610.15542).** Continued multimodal pretraining of the Apertus 8B/70B models; like v1 it includes *Infrastructure, Scaling, and Efficiency* and *Engineering Challenges and Solutions* sections (~5 pages).


## Hardware setup logbooks

The training logbooks above start once the cluster works. These start earlier: racking, cabling, burn-in and acceptance testing.

- **Imbue, [From bare metal to a 70B model: infrastructure set-up and scripts](https://imbue.com/blog/70b-infrastructure).** A detailed log of how they set up a 512-node [InfiniBand](../network/interconnects.md#infiniband) cluster wired as a [fat tree](../foundations/networking.md#top-of-rack-leaf-spine-fat-tree) and got it to work. They also open-sourced the [cluster tooling](https://github.com/imbue-ai/cluster-health) they wrote in the process (health checks for GPUs, InfiniBand and hosts), which is worth reading next to [Fault tolerance and reproducibility](../training/fault-tolerance-and-reproducibility.md).
- **SemiAnalysis, [What it takes to set up a Neocloud cluster](https://newsletter.semianalysis.com/p/ai-neocloud-playbook-and-anatomy).** A detailed write-up of building and operating a GPU cloud. A **neocloud** is a GPU-focused cloud provider such as CoreWeave or Lambda, as opposed to the big general-purpose clouds ([Tier 1 vs Tier 2 clouds](../platform/getting-compute.md#tier-1-vs-tier-2-clouds)). It is the source of the burn-in and acceptance-test durations quoted in [Accelerators need to be burned in](../platform/getting-compute.md#accelerators-need-to-be-burned-in).


## Lessons learned

![Machine Learning Engineering - Lessons Learned from Training LLMs](../courses/lesson-learned/lesson-learned-cover.png)

These lessons come from Stas Bekman's work on BLOOM-176B (2022), IDEFICS-80B (2023), production RAG and RL systems at Contextual.AI (2024) and Arctic Training open source work at Snowflake (2025-2026), and from his book [The Art of Debugging](https://github.com/stas00/the-art-of-debugging) (section 9). You can use them as a self-guided course, or teach them: read the headlines, skim the bullets for the reasoning, and follow the "Covered in" link for the full explanation, scripts and benchmarks. Lessons whose content is not taught elsewhere in this book are kept here in full, with code.

Most lessons hinge on a number. This script re-derives them, so you can see where each comes from:

```python
# lesson_numbers.py - re-derive the numbers quoted in the lessons below
import math

# 1.3 block size waste: 16 MiB blocks holding 16 KiB files
print(f"1.3  block waste            : {(16 << 20) // (16 << 10)}x")

# 2.1 duplex vs unidirectional
for name, meas, spec in [("A100 vs duplex 600", 235, 600), ("A100 vs uni 300", 235, 300),
                         ("H200 vs uni 450", 376, 450)]:
    print(f"2.1  {name:<22}: {meas / spec:.0%}")

# 2.3 all-reduce sends per rank: ring vs NVLS (in-switch reduction), N = 8 GPUs
N = 8
print(f"2.3  sends, ring 2N vs NVLS N+1 (N={N}): {2 * N} vs {N + 1}")

# 4.1 achieved vs theoretical matmul TFLOPS on B200
print(f"4.1  B200 MAMF / spec        : {1745.0 / 2250:.1%}")

# 6.2 checkpoint overhead: 40 s per save, every 3 h, ~90-day run
days, save_s, every_h = 90, 40, 3
n = days * 24 / every_h
hours = n * save_s / 3600
print(f"6.2  {n:.0f} saves, {hours:.1f} h     : {hours / (days * 24):.2%} of the run")
print(f"6.2  same on 5x slower IO   : {5 * hours / (days * 24):.2%} of the run")
print(f"6.5  disk per day           : {24 / every_h * 2.3:.1f} TB")

# 7.1 init std for BLOOM-176B hidden size
h = 14336
print(f"7.1  sqrt(2/(5h)) = {math.sqrt(2 / (5 * h)):.5f}, sqrt(1/(3h)) = {math.sqrt(1 / (3 * h)):.5f}")

# 9.2 git bisect rounds for 1024 candidate commits
print(f"9.2  bisect rounds for 1024 : {math.ceil(math.log2(1024))}")
```

**[ran]** output:

```text
1.3  block waste            : 1024x
2.1  A100 vs duplex 600    : 39%
2.1  A100 vs uni 300       : 78%
2.1  H200 vs uni 450       : 84%
2.3  sends, ring 2N vs NVLS N+1 (N=8): 16 vs 9
4.1  B200 MAMF / spec        : 77.6%
6.2  720 saves, 8.0 h     : 0.37% of the run
6.2  same on 5x slower IO   : 1.85% of the run
6.5  disk per day           : 18.4 TB
7.1  sqrt(2/(5h)) = 0.00528, sqrt(1/(3h)) = 0.00482
9.2  bisect rounds for 1024 : 10
```

How to read it: each line is labelled with the lesson that quotes it. The "5x slower IO" line is the ~2% checkpoint tax of lesson 6.2; the two std values are the two init formulas of lesson 7.1.

### 1. Storage and filesystems

**1.1. Fast parallel storage is a game changer for checkpointing.**

- Parallel distributed filesystems such as GPFS and WekaIO let hundreds to thousands of clients hit the same storage at once without hotspots, and the difference is dramatic.
- At the JeanZay HPC in 2021, a 2.3TB checkpoint was written in parallel by 384 processes in 40 seconds, because the cluster ran GPFS over NVMe drives.
- GPFS over NVMe versus a generic cloud NFS mount can differ by many multiples in wall-clock time.
- Takeaway: benchmark your filesystem with your workload to know its overhead.

Covered in: [Which file system to choose](../foundations/storage-and-data.md#which-file-system-to-choose).

**1.2. Slow shared storage kills developer productivity, not just training throughput.**

- On GCP's Zonal Filestore over NFS, `python -c "import torch"` took 20 seconds uncached (about 2 seconds once cached), and creating a conda environment with a handful of prebuilt packages could take 20-30 minutes.
- None of this shows up in a training-throughput benchmark, but it slows every developer every day, multiplied across the team.
- The fix: before committing to any storage, run a "perception benchmark" (timed `import torch`, `conda create`, `git status`, `pytest --collect-only`) and compare it with a local NVMe baseline.
- If a proposed shared filesystem is only 2x faster than what you have, it is probably not worth switching. Aim for something within roughly 5-10x of local NVMe before it stops being a daily frustration.
- Use `uv` instead of `pip` to install packages: it is 5-50x faster, and fewer seconds spent per install means fewer slow small-file operations on the shared filesystem.

A minimal perception benchmark, run once on local NVMe and once on the candidate filesystem:

```bash
# perception.sh - run from a directory on the filesystem under test (untested as a whole; each line is standard)
time python -c "import torch"
time uv venv /path/on/fs/under/test/venv && time uv pip install --python /path/on/fs/under/test/venv numpy
time git status
time pytest --collect-only -q > /dev/null
```

Compare the `real` times line by line between the two filesystems; the ratio is what your team will feel.

Covered in: [Which file system to choose](../foundations/storage-and-data.md#which-file-system-to-choose), [Usability perception IO benchmarks](../foundations/storage-and-data.md#usability-perception-io-benchmarks). Not covered elsewhere: the "2x is not worth switching" rule and the `uv` speed-up.

**1.3. Python's tens of thousands of tiny files can exhaust inodes long before you run out of bytes.**

- On a partition with a large block size (16MiB, common for GPFS partitions tuned for big files), files averaging 16KiB waste up to 1,024x the space each, and more importantly they burn through the inode budget rather than the byte budget.
- This happened at JeanZay: each conda environment installs tens of thousands of tiny files, the team kept running out of inodes on the normal GPFS partitions, and had to request a dedicated partition just for conda environments.
- The fix: keep at least two differently tuned partitions, a small block size (4-8KiB) for millions of tiny files (venvs, package caches) and a large one (2-16MiB) for a few huge files (checkpoints, datasets). Modern GPFS sub-blocks help by packing tiny files together.
- Always check `df -i` next to `df -h`: a partition can show plenty of free space and have no inodes left.

```bash
df -h /path   # bytes:  Size Used Avail Use%
df -i /path   # inodes: Inodes IUsed IFree IUse%  - 100% here means "disk full" even if df -h says otherwise
```

Covered in: [File block size](../foundations/storage-and-data.md#file-block-size).

**1.4. Avoid loading training data directly from cloud storage if you possibly can.**

- During IDEFICS-80B there wasn't enough local disk for the multi-TB multimodal dataset, so it was streamed from cloud storage. Weeks of effort went into making that robust, and it never quite got there.
- The core problem: the streaming DataLoader of the time didn't save and restore the sampler's RNG state across resumes, so every crash-and-resume either skipped or repeated data, and the run never saw a single clean epoch of unique data.
- Prefer enough local disk for the data. Offload only checkpoints to the cloud, asynchronously (from cron or a separate Slurm job), and always keep the last couple of checkpoints locally for a fast resume.
- If you must stream, verify that your solution resumes exactly, without losing or repeating data, and without needing a huge number of local DataLoader worker processes.

Covered in: [Local storage beats cloud storage](../foundations/storage-and-data.md#local-storage-beats-cloud-storage).

### 2. Network and interconnects

**2.1. Check whether a bandwidth number is unidirectional or bidirectional (duplex) before comparing it to a measurement.**

- Most benchmark tools report unidirectional bandwidth; vendor marketing prefers duplex, which is about 2x larger.
- Rule of thumb: if you measure about 40% of the advertised number, the advertisement was probably duplex. Halve it and your efficiency becomes a normal ~80%.
- Example: an A100 node advertised at 600GBps intra-node measured 235GBps, a disappointing 39%, until it was compared with the 300GBps unidirectional spec (78%).
- Same today on H200: NVLink is advertised at 900GBps (duplex), and an all-reduce benchmark reaching about 376GBps checks out at 84% of the 450GBps unidirectional spec.

Covered in: [Unidirectional vs bidirectional (duplex)](../network/interconnects.md#unidirectional-vs-bidirectional-duplex). The H200 all-reduce figure (376GBps) is quoted only here; that chapter gives H200 copy-bandwidth measurements instead.

**2.2. A slow inter-node network can dictate your training framework, not just your throughput ceiling.**

- At JeanZay the Omni-Path interconnect was stuck at 135Gbps.
- That forced BLOOM-176B onto Megatron-DeepSpeed (tensor + pipeline parallelism, which moves far less data between nodes) instead of the much simpler DeepSpeed ZeRO (which needs much more bandwidth to shard optimizer states and gradients).
- Such a choice can't be undone once a multi-month run is underway, so measure the network and factor it into your parallelism strategy at the very start.
- On a slow inter-node network, prefer strategies that exchange only activations (pipeline and tensor parallelism) over those that exchange full gradients and optimizer state (ZeRO-DP, FSDP).

Covered in: [Omni-Path](../network/interconnects.md#omni-path), and the bytes each strategy moves in [Model parallelism: derivations](../training/model-parallelism/derivations.md).

**2.3. An all-reduce that beats the wire's theoretical maximum is not a measurement bug: it is SHARP.**

- On an H100 node, an intra-node NVLink 4 all-reduce reported 480GBps for a 4GiB payload, above the 450GBps unidirectional spec.
- NVIDIA's SHARP lets NCCL do the reduction inside the switch. When NCCL detects InfiniBand or NVSwitch SHARP support it uses the `NVLS` algorithm, which needs about `N+1` sends instead of the ring's `2N`.
- So the standard `busbw` formula, which assumes `2N`, over-states the bytes on the wire once NVLS is active: a number that looks "too good" is a smarter algorithm, not an instrumentation error.
- You need more than about 4 GPUs in the collective (the exact switch-over depends on the hardware), and `NCCL_NVLS_ENABLE` must not be set to `0`.
- SHARP applies only to `all-reduce`. A full-node `all-gather` or `reduce-scatter` runs the ring path and lands at the ordinary ~80% of the unidirectional spec.

Covered in: [SHARP](../network/interconnects.md#sharp).

**2.4. On a shared inter-node network, performance tuning is nearly impossible: you can't tell whether your change or your neighbours changed the number.**

- At JeanZay, before BLOOM-176B, the identical configuration gave noticeably different throughput every run, because other users' jobs shared the fabric at unpredictable intensity.
- When the noise floor moves by double digits between runs, you can't attribute a change to your tweak.
- The fix: just before the BLOOM-176B launch the team got an exclusive, isolated Slurm partition on the then-new A100s, and tuning finally became tractable.
- For serious pre-training performance work, insist on a dedicated, isolated network partition for the benchmarking phase.

Covered in: [Shared internode network](../network/interconnects.md#shared-internode-network), [Crucial reproducibility requirements](../network/measuring-and-debugging.md#crucial-reproducibility-requirements).

### 3. Compute, accelerators and memory

**3.1. Coherent (unified) memory hardware can OOM unpredictably, and the fix is a driver mode, not a code change.**

- On NVIDIA's coherent-memory platforms (DGX Station, DGX Spark, GH200, GB300), Linux can transparently borrow GPU HBM when it runs out of CPU RAM, but not the other way round, and `nvidia-smi` may not even show the borrowing.
- While tuning the maximum sequence length for Qwen3-32B post-training on a DGX Station, with optimizer states offloaded to CPU RAM, the team saw seemingly random alternation between CPU OOM and GPU OOM, depending on which background daemons happened to be running.
- The fix was a system setting: switching from the default NUMA coherent-memory mode to CDMM (Coherent Driver-based Memory Management, "driver" mode, where the NVIDIA driver alone manages GPU memory) via `/etc/modprobe.d/nvidia-openrm.conf` and a reboot. After that, CPU and GPU memory stopped silently borrowing from each other.
- If you see unexplained, apparently non-deterministic OOMs on coherent-memory NVIDIA hardware, check the `CoherentGPUMemoryMode` setting before debugging your training script.

```bash
# [GPU, untested] which coherent memory mode is the driver in?
nvidia-smi -q | grep -i -A1 coherent
```

Covered in: [Debugging PyTorch](../development/debugging-pytorch.md) (the coherent memory section, with the exact `modprobe` configuration).

**3.2. Cryptic asynchronous CUDA errors may need the debug flag on permanently, and the flag may cost nothing.**

- CUDA runs asynchronously, so by the time an error surfaces the program has moved on: tracebacks point at the wrong line, and NCCL may refuse to produce one at all.
- `CUDA_LAUNCH_BLOCKING=1` makes every kernel launch synchronous. Normally that costs real performance, but in exchange you get an accurate, in-context traceback.
- BLOOM-176B ran its *entire* training with `CUDA_LAUNCH_BLOCKING=1` (training otherwise hung for a reason never diagnosed), and saw no measurable throughput loss.
- Don't assume a debug flag is a performance tax until you've measured it for your case; it might also be the fix.

```bash
CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node 8 train.py   # [GPU, untested] compare tokens/s with and without
```

Covered in: [Debugging PyTorch](../development/debugging-pytorch.md) (async CUDA bugs).

**3.3. Memory profilers catch leaks that are invisible in code review.**

- While implementing a tricky custom `TiledMLP` `torch.autograd.Function`, a leak hid in the implementation and became obvious only when rendered by PyTorch's memory profiler: `torch.cuda.memory._record_memory_history()` plus `torch.cuda.memory._dump_snapshot()`, viewed at [pytorch.org/memory_viz](https://pytorch.org/memory_viz).
- Normal allocations appear as bars that start and end. The leaked tensors appeared as bars that started in one layer's forward pass and never ended, repeated for every following layer.
- Clicking a bar gives the Python traceback of the allocating line, turning "there's a leak somewhere" into "line X of file Y" in minutes.
- The same view is great for comparing how two implementations of "the same" algorithm use GPU memory over time, for example normal versus CPU-offloaded activation checkpointing.

```python
# [GPU, untested]
import torch
torch.cuda.memory._record_memory_history(max_entries=100_000)
run_a_few_steps()                                   # your code
torch.cuda.memory._dump_snapshot("snapshot.pickle") # drag this file onto pytorch.org/memory_viz
torch.cuda.memory._record_memory_history(enabled=None)
```

Covered in: [Debugging PyTorch](../development/debugging-pytorch.md) (PyTorch memory profiler), and reading memory snapshots in [Profiling and optimization](../training/profiling-and-optimization.md).

**3.4. Precisely placed memory sampling finds framework bugs too, not just bugs in your code.**

- Explicit control over *when* you sample GPU and CPU memory (a "sample now" call at strategic points, via a helper derived from DeepSpeed's `see_memory_usage`) can isolate a leak a passive profiler misses.
- This technique exposed a real leak inside PyTorch's own `all_gather_object` ([pytorch#150798](https://github.com/pytorch/pytorch/issues/150798), since fixed), and several other PyTorch-internal leaks were found the same way.
- Releasing a Python variable doesn't free its tensor immediately, because Python's garbage collector runs on its own schedule. Call `gc.collect()` before sampling or you'll see phantom "still allocated" numbers.
- Turn this tooling off in production: frequent `gc.collect()` calls and sampling cost real time.

Covered in: [Debugging PyTorch](../development/debugging-pytorch.md) (strategic memory allocation tracing).

**3.5. Hyper-Threading can wreck network throughput on some hardware; the only way to know is to benchmark both settings.**

- On AWS p4 nodes, enabling Hyper-Threading (HT, normally "free" extra CPU parallelism) cut network throughput to about a quarter of the non-HT value.
- Because of that measurement, HT stayed disabled on that setup, rather than left at the Slurm configuration's (often HT-enabled) default.
- It isn't universal in either direction: on other setups disabling HT (`--hint=nomultithread`) is what hurts.
- Benchmark your workload's network throughput with HT on and off on your hardware and keep the winner.

Covered in: [Hyper-Threads: on or off?](../orchestration/slurm/from-zero.md#hyper-threads-on-or-off).

### 4. Training performance and throughput

**4.1. Spec sheets give unreachable numbers: benchmark matmul on your own hardware.**

- Theoretical peak TFLOPS assume perfectly shaped matrices, no memory-movement overhead, and boost clocks held indefinitely. None of that happens in real training.
- Example: a B200 SXM may reach only ~1745 TFLOPS (bf16) out of a 2250 TFLOPS spec (~77.6%), and even that depends on the software-stack version and the exact matrix shapes.
- That's why MAMF (Maximum Achievable Matmul FLOPS) exists as a metric distinct from the spec: `mamf-finder.py` brute-force searches matmul shapes on your hardware and software and gives the realistic ceiling to optimize against.
- Once your measured training TFLOPS approach your MAMF (not the theoretical peak), stop optimizing and start training.

Covered in: [MAMF and MSMF comparison table](../compute/accelerators.md#mamf-and-msmf-comparison-table) (whose current B200 entry, from a newer software stack, is 1737 TFLOPS, 77.2%: a reminder that the number moves with the stack).

**4.2. Throughput tuning is a journey with diminishing returns, and you have to decide when to stop.**

- In BLOOM-176B's tuning phase the team started below 100 TFLOPS per GPU and, over a few weeks of accumulated small wins (no silver bullet), reached 150 TFLOPS by launch.
- Once there was no meaningful headroom left relative to what similar setups achieved (~155 TFLOPS for 80GB A100s in 2022), they stopped optimizing and started training.
- Budget explicit calendar time for this phase before launch. Treated as an afterthought, it means either months of compute at low efficiency or an endlessly delayed launch.

Covered in: [TFLOPS as a performance metric](../training/profiling-and-optimization.md#tflops-as-a-performance-metric).

**4.3. Global batch size ramp-up matters far more than intuition suggests, especially with pipeline parallelism.**

- For BLOOM-176B, starting small was disastrous: GBS=16 gave only 8 TFLOPS, because with pipeline parallelism a small global batch means a large pipeline bubble of idle GPU time.
- The team started at GBS=192 (73 TFLOPS) and ramped up by 16 every ~9.77M samples, reaching GBS=2048 at 150 TFLOPS.
- Why ramp at all: early training is nearly random and doesn't yet benefit from large, precise batches, but there's no reason to pay the bubble cost of a tiny batch either.
- With pipeline parallelism, benchmark achieved TFLOPS at a few candidate starting GBS values before choosing a ramp schedule.

Covered in: [Global batch size ramp up](../training/stability.md#global-batch-size-ramp-up).

### 5. DataLoaders

**5.1. Several DataLoaders in one process can hang randomly; the workaround may be to stop using one of them.**

- In BLOOM-176B the training DataLoader ran fine with `num_workers=2`, but adding an evaluation DataLoader caused random hangs, a known, long-unresolved class of PyTorch multi-worker DataLoader bugs.
- The workaround: `num_workers=0` for the eval loader only (accepting blocking IO, since eval ran rarely), and eventually dropping in-training eval altogether in favour of asynchronous evaluation of saved checkpoints with EleutherAI's [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) (a tool that runs standard benchmarks against a saved model).
- Bonus: removing in-loop eval also sped up training, since nothing shared resources with a second DataLoader any more.

Covered in: [Asynchronous DataLoader](../training/profiling-and-optimization.md#asynchronous-dataloader).

**5.2. A streaming DataLoader can quietly become the bottleneck when workers use a lot of memory and CPU RAM is limited.**

- In IDEFICS-80B the multimodal streaming DataLoader used huge, spiky amounts of memory per worker, and with ~1TiB of CPU RAM per node the team couldn't spawn enough workers to keep the GPUs fed.
- Unlike a GPU OOM, which fails loudly, an under-provisioned DataLoader just caps throughput without crashing anything.
- There was no time to engineer a fix, so the run finished at that DataLoader-bound ceiling.
- For multimodal or other memory-hungry streaming pipelines, budget CPU RAM for DataLoader workers as carefully as GPU memory for the model.

Covered in: [Asynchronous DataLoader](../training/profiling-and-optimization.md#asynchronous-dataloader).

### 6. Fault tolerance and reliability

**6.1. Beware of newly minted GPUs: a fresh batch often has a 3-10% failure rate, worse for the first units of a new generation.**

- This is common enough that burn-in practices exist: OEMs typically burn in for 3-4 weeks, and cloud providers then run a further 2-3 day acceptance test, though not every provider does this rigorously or says whether it did.
- Plan for spare nodes (5-10% extra is reasonable) and make sure your scheduler can drain and replace bad nodes automatically.
- Once you've spent weeks separating good accelerators from bad ones, insist on a **fixed** allocation, so the good nodes stay yours across reboots instead of returning to a shared pool.
- Ask any provider how long and how rigorously your specific nodes were burned in. "We burned them in" without a duration is not an answer.

Covered in: [Always plan to have more nodes than needed](../training/fault-tolerance-and-reproducibility.md#always-plan-to-have-more-nodes-than-needed), [Accelerators need to be burned in](../platform/getting-compute.md#accelerators-need-to-be-burned-in), [Keep your good accelerators on reboot](../platform/getting-compute.md#keep-your-good-accelerators-on-reboot).

**6.2. Checkpoint overhead scales inversely with IO speed.**

- BLOOM-176B checkpoints were fast: 2.3TB in 40 seconds, every ~3 hours over a ~3-month run, about 720 checkpoints and only ~8 hours in total, a ~0.37% tax on wall-clock time.
- The same cadence on IO 5x slower (a normal gap between premium and standard cloud storage tiers) grows to about 2% of the run, a significant cost at scale.
- The math: measured save time x number of saves, compared with the expected training duration (the script at the top of this section does it).
- If you offload checkpoints to the cloud, always keep the two most recent ones locally for a fast resume, since the latest may be corrupt from a crash mid-save.

Covered in: [Frequent checkpoint saving](../training/fault-tolerance-and-reproducibility.md#frequent-checkpoint-saving).

**6.3. Build a kill switch: Slurm has no notion of group job ownership.**

- On an HPC cluster, if a colleague launches a long job or job array and goes to sleep, you can't stop or modify it without `sudo`: Slurm permissions are per user, not per team.
- BLOOM-176B's fix was a file-based kill switch: before each iteration the training loop checks for a specific file and, if it exists, saves a checkpoint and exits cleanly, so *any* team member can intervene.
- A second check at the very start of `main()` lets a long queued job array from a sleeping teammate "burn through" in seconds instead of each job running first.
- A few lines of code buy a lot of operational safety; build it into any shared multi-user training setup from day one.

Covered in: [Kill switch](../training/fault-tolerance-and-reproducibility.md#kill-switch).

**6.4. Watch the watchers: a scheduled job-status watchdog catches "training silently stopped and nobody noticed".**

- During BLOOM-176B, `slurm-status.py` ran about hourly (through cron emulation on Slurm, since real `crontab` usually isn't available there) and checked whether the named job was running or at least queued.
- If neither, it emailed an alert, and it also appended its result to the main training log for a paper trail.
- It costs almost nothing and catches a whole class of expensive failures (hours or days of idle compute) that would otherwise surface only when someone happened to look.
- Don't just build the training loop; build the thing that watches it.

Covered in: [Is-job-running watchdog](../training/fault-tolerance-and-reproducibility.md#is-job-running-watchdog).

**6.5. Monitor disk usage and inode usage aggressively; checkpoint-heavy training eats storage faster than you expect.**

- BLOOM-176B checkpointing wrote about 18.4TB per day. On a shared cluster where other jobs write too, that rate burns through a "should be plenty" allocation fast.
- `fs-watchdog.py` monitored both the disk-usage percentage and the inode usage of every relevant partition, since those are two independent ways to run out of space.
- Tune alert thresholds per partition rather than one blanket percentage; what matters is how many days of runway the free space gives at the current daily burn rate.
- This turns "we ran out of disk at 3am and lost the run" into "we got an email three days ago and added storage in time".

Covered in: [Low disk space alerts](../training/fault-tolerance-and-reproducibility.md#low-disk-space-alerts).

**6.6. On some distributed filesystems 80% is the new 100%: free space in `df` is no guarantee.**

- Distributed storage is built from many physical disks (each typically 0.3-2TB). If rebalancing doesn't happen often enough, a single disk can hit 100% and reject writes while the aggregate `df` shows a comfortable 80% used.
- This exact failure hit IDEFICS-80B training, where 80% really was the practical ceiling.
- It depends on the technology: Lustre is known to have this problem (about 80% reliably usable); GPFS vendors say GPFS doesn't.
- Ask your storage provider what fraction of nominal capacity is reliably usable, and provision for that ceiling.

Covered in: [Low disk space alerts](../training/fault-tolerance-and-reproducibility.md#low-disk-space-alerts).

**6.7. Sometimes the right call is to work around a leak instead of chasing its root cause, especially under time pressure.**

- During IDEFICS-80B a slow CPU memory leak took several days to grow into an OOM crash, a timescale that makes root-causing painfully slow to iterate on.
- Rather than halting training to hunt for it, the team added an in-loop watchdog that checked resident memory and exited voluntarily past a threshold, so the next queued job resumed with memory reclaimed.
- Concrete pattern: once per iteration check `psutil.virtual_memory().percent` (or, for GPU, `torch.cuda.mem_get_info()` after `gc.collect()` and `torch.cuda.empty_cache()`) and exit cleanly past a threshold. Cheap insurance, worth adding whether or not you suspect a leak.

Covered in: [Dealing with slow memory leaks](../training/fault-tolerance-and-reproducibility.md#dealing-with-slow-memory-leaks).

**6.8. Handle forced preemption gracefully: know how much time your program needs to save and exit, and tell it up front.**

- Slurm jobs on shared clusters have hard time limits (about 20 hours is common; BLOOM-176B's jobs ran up to 100 hours), and letting the job be killed wastes every step since the last checkpoint.
- BLOOM-176B's approach A, `--exit-duration-in-mins`: tell the program at launch how many minutes it has. It starts a timer and, before each iteration, checks (with an `all_reduce`-synchronized flag so all ranks agree) whether the budget is spent; if so it saves and exits before Slurm's hard kill.
- Sizing the budget: one iteration plus one checkpoint save, then doubled for margin. Exiting early costs nothing; exiting late costs the whole checkpoint.
- Approach B: signals (`--signal=B:USR1@<seconds>`), trapped directly or relayed through the launcher for `torchrun`, `accelerate` or `deepspeed`, for when you can't compute a fixed time budget in advance.

Covered in: [Approach A: tell the program at launch time when it should start the exiting process](../training/fault-tolerance-and-reproducibility.md#approach-a-tell-the-program-at-launch-time-when-it-should-start-the-exiting-process), and approach B in [Dealing with forced job preemption](../training/fault-tolerance-and-reproducibility.md#dealing-with-forced-job-preemption).

### 7. Training stability

**7.1. Initialization std matters enormously; get it wrong and a large model can't get past the first few thousand iterations.**

- In the pre-BLOOM 104B experiments the team couldn't get past early training until they realized Megatron-LM's then-default `--init-method-std` of 0.02 was far too large for that hidden size.
- Two published formulas were considered: "Transformers without Tears", `sqrt(2/(NHIDDEN*5))`, and the 530B paper's `sqrt(1/(NHIDDEN*3))`. The team chose the smaller (530B) one as more conservative.
- For BLOOM-176B's `NHIDDEN=14336` that gives `std=0.00482`, a tiny, easy-to-get-wrong number, and one of the main reasons the 176B run never fully diverged.
- Scale init std with the hidden size using a published formula; never copy a fixed constant from a smaller model's config.

Covered in: [Initialization std](../training/stability.md#initialization-std).

**7.2. Learn from a failed training instead of abandoning it: the failed 104B experiments produced the recipe that made BLOOM-176B work.**

- The 104B attempts kept diverging early. The post-mortem pointed at two suspects: fp16 (narrow dynamic range, prone to overflow) and training data full of uncleaned web-scraped garbage.
- For BLOOM-176B the team changed three things at once: fp16 to bf16, much cleaner data, and an added embedding LayerNorm. The combination made all the difference: an almost perfectly smooth loss curve, with one spike that recovered within 200 steps.
- Note that changing three things at once means nobody knows which one mattered; at that scale the team couldn't afford separate runs. The point isn't "use bf16": a failed multi-month experiment, analysed rather than written off, produced the fix for the next, larger attempt.

Covered in: [A very failed training](../training/stability.md#a-very-failed-training).

### 8. Cloud providers and procurement

**8.1. A crowd on the support call is not support; negotiate real remedies into the contract.**

- "We will do our best" often means ten-plus people on an escalation call with no ability to fix anything. Reaching the two people who matter, one product manager and the responsible engineer, can take weeks to months (faster with personal connections).
- The customer is legally obliged to keep paying and generally can't break the contract unilaterally, so "best effort" is one-sided unless the contract has quantifiable delivery metrics for each critical component.
- The most useful remedy to negotiate: payment pauses until a defined problem is fixed, the one lever that gives the provider a reason to prioritize the fix.
- Negotiate this *before* signing; afterwards you have essentially no leverage.

Covered in: [The "we will do our best" clause](../platform/getting-compute.md#the-we-will-do-our-best-clause).

**8.2. Accelerators are what fail: about 95% of hardware failures on a cluster are the accelerators, not CPUs, storage or network gear.**

- Let this shape fault-tolerance planning: storage and network resilience don't deserve the same engineering effort as GPU-failure resilience.
- Ask for a fast, ideally automated replacement path (an API, not a support ticket), how many spares the provider keeps, and how quickly it commits to swapping a failed unit.
- Since instant replacement is rarely available, the standard mitigation is to pay for ~10% more nodes than needed, using the extras for day-to-day development and as instant fallbacks when a production node fails.
- That overprovisioning is real money, but cheap compared with a multi-hundred-GPU job stalled on a support ticket.

Covered in: [Dealing with accelerator failures](../platform/getting-compute.md#dealing-with-accelerator-failures).

**8.3. The network can take accelerator memory you paid for, without telling you.**

- On one cloud, about 5GiB of *each* GPU's memory was already used by the provider's networking software before any user workload started. This was found after signing, not disclosed during sales.
- As accelerator memory grows (141GiB on H200, about 180GiB usable on B200), a fixed 5GiB matters proportionally less, but on an 80GiB H100 it can be the difference between a batch size fitting and an OOM.
- Ask before signing exactly how much GPU memory the provider's software reserves and how much is available to you, and get the number in writing. You can also check it yourself on a fresh node:

```bash
# [GPU, untested] memory already in use before you run anything
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
```

Covered in: [Does the network steal accelerator memory?](../platform/getting-compute.md#does-the-network-steal-accelerator-memory).

**8.4. Provider software stacks can trap you on old, unsupported OS versions.**

- More than once, a provider's required driver, orchestration or monitoring stack didn't support any Ubuntu release newer than a fairly old one, forcing an outdated OS under a modern ML stack.
- Nobody demos "which Ubuntu can I run", so it's easy to miss during evaluation, but it bites later: security patches, kernel features, drivers and some ML dependencies assume a newer base OS.
- Ask during evaluation, not after signing, which OS and software versions are supported and required, and get a commitment on how quickly newer ones will be supported.

Covered in: [Up-to-date software and OS versions](../platform/getting-compute.md#up-to-date-software-and-os-versions).

### 9. Debugging methodology

These lessons come from [The Art of Debugging](https://github.com/stas00/the-art-of-debugging). This book teaches the tools ([gdb basics](../foundations/linux-and-os.md#gdb-basics), [py-spy](../foundations/linux-and-os.md#py-spy-what-is-python-doing-right-now), [shared libraries](../foundations/linux-and-os.md#shared-libraries-and-ld_library_path)) but not all of the methodology, so the lessons are kept here in full.

**9.1. Reduce a failing import or component to a one-liner instead of debugging it inside a large, slow-starting program.**

- If a large program fails at startup (a failing `import torch`, or a model, tokenizer or config load failing on a missing Hugging Face auth token), extract exactly that operation into a standalone one-line command and reproduce it there.
- Fast debug cycles and small reproductions are the two most important prerequisites for successful debugging. At ten minutes per attempt over thirty attempts, you lose track of which hypothesis you actually tested.
- One-liners such as `python -c "import torch"` are atomic (findable in shell history), trivially copy-pasted to a remote machine, and easy to save in a file of reusable variations.
- The same trick lets you pre-download and cache a large model or dataset in parallel with other work:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c 'import sys; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained(sys.argv[1])' t5-small
```

The first prints the torch version and whether CUDA is visible, in about a second, instead of after your whole program's startup; the second only downloads and caches the `t5-small` tokenizer.

Read more: [The power of one-liner programs](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#the-power-of-one-liner-programs).

**9.2. Bisect to find the breaking commit instead of reading hundreds of commits by eye.**

- The common scenario: you were pinned to a known-good release, upgraded to bleeding-edge `main` for a new feature, and something broke somewhere in the hundreds of commits in between.
- `git bisect` is a binary search: give it one good and one bad commit, it checks out the midpoint, you test and mark it `good` or `bad`, repeat. 1,024 candidate commits take about 10 rounds.
- Standard handles for complications: `git bisect skip` for commits you can't test, and custom terms (`git bisect start --term-new=fixed --term-old=broken`) when you're hunting for the commit that *fixed* something rather than broke it.
- When it converges, the first bad commit's GitHub page almost always links to the PR that introduced it, an immediate, informed place to ask for help.

If the test can be scripted, `git bisect run` does every round for you: the command's exit code (0 = good, non-zero = bad) is the verdict. A self-contained demo, a repository of 16 commits where commit 11 introduces a bug:

```bash
# bisect_demo.sh - 16 commits, commit 11 introduces a bug; let git find it
git init -q . && git config user.email d@d && git config user.name d
for i in $(seq 1 16); do
  if [ $i -ge 11 ]; then bug="/0"; else bug=""; fi
  echo "x = $i$bug" > lib.py                 # commits 11..16: ZeroDivisionError on import
  git add lib.py && git commit -qm "commit $i"
done
git bisect start HEAD HEAD~15 >/dev/null     # bad = newest, good = commit 1
git bisect run python3 -c "import lib" 2>/dev/null | grep -E "first bad commit|^    commit|^running"
git bisect reset >/dev/null
```

**[ran]** in an empty directory, output:

```text
running 'python3' '-c' 'import lib'
running 'python3' '-c' 'import lib'
running 'python3' '-c' 'import lib'
running 'python3' '-c' 'import lib'
cb7fa6171e2359bdb76fb1f198b43837c41b5ce1 is the first bad commit
    commit 11
bisect found first bad commit
Previous HEAD position was cb7fa61 commit 11
Switched to branch 'master'
```

How to read it: four `running` lines means four test rounds for 15 candidate commits ($\lceil\log_2 16\rceil = 4$), and git names commit 11 as the first bad one. In real use the test is your one-liner from 9.1 or a single pytest test.

Read more: [Finding a breaking commit by bisecting revisions](https://github.com/stas00/the-art-of-debugging/blob/master/methodology/README.md#finding-a-breaking-commit-by-bisecting-revisions).

**9.3. Get a backtrace from the core file to turn a useless native crash into one actionable line.**

- When a compiled extension (a CUDA kernel, NCCL, any C/C++ library) segfaults, Python can't raise a meaningful exception; you get nothing, or an unreadable native dump.
- Load the core file into gdb with `gdb -c /path/to/core /path/to/program` and type `bt`: it walks the call stack at the moment of the crash and shows the exact source line. `bt full` also prints local variable values (a pointer showing `0x0`, for example, reveals it was dereferenced uninitialized).
- For multi-threaded programs (essentially every serious PyTorch/CUDA program), `thread apply all bt` shows every thread's stack, since the real cause sometimes lives in a different thread from the one that crashed.
- This is how real PyTorch crash reports get root-caused.

```bash
ulimit -c unlimited                        # allow core files (see gdb basics for where they go)
python train.py                            # ... Segmentation fault (core dumped)
gdb -c /path/to/core $(which python)       # program = the binary that crashed: python, not train.py
(gdb) bt                                   # stack of the crashing thread
(gdb) bt full                              # same, with local variables
(gdb) thread apply all bt                  # every thread
```

Read more: [Get the backtrace from the core file](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#get-the-backtrace-from-the-core-file).

**9.4. Backtrace a live, hung process instead of waiting for it to crash or time out.**

- A deadlock isn't a segfault: the process is alive but makes no progress, so there's no crash to produce a core dump.
- `gdb --pid=<pid>` attaches to the running process without restarting it, and `thread apply all bt` shows what every thread is doing. This is how a real PyTorch deadlock was once diagnosed.
- To keep the state for later analysis (or free the resources right away), `gcore <pid>` writes a core file of the running process, and `kill -ABRT <pid>` makes the process dump core and die. (The original lesson says both terminate the process; `gcore` as shipped with gdb detaches and leaves it running, so check on your system before relying on either behaviour.)
- Use live attach for spinning or fully hung processes, and the core-file technique of 9.3 for processes that already crashed.
- If the hang is in Python code, `py-spy dump --pid <pid>` is the fastest answer ([py-spy](../foundations/linux-and-os.md#py-spy-what-is-python-doing-right-now)).

```bash
gdb --pid=4033 -batch -ex "thread apply all bt" > stacks.txt   # attach, dump all stacks, detach
gcore 4033                                                      # write core.4033 for later
py-spy dump --pid 4033 --native                                 # Python + native frames
```

Read more: [Get the backtrace from the still running process](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#get-the-backtrace-from-the-still-running-process), [py-spy](https://github.com/stas00/the-art-of-debugging/tree/master/pytorch#py-spy). Hangs across many ranks: [Debugging distributed](../development/debugging-distributed.md).

**9.5. Run flaky or segfaulting tests under gdb from the start instead of trying to reproduce the crash blind.**

- For a `pytest` test that segfaults intermittently, launch the whole invocation under gdb rather than trying to catch the crash separately afterwards.
- When it segfaults, gdb stops at its prompt at that exact moment; `bt` or `bt full` works as in 9.3, and `c` continues if you want.
- This sidesteps "it didn't crash when I tried to reproduce it separately".
- Together with the two backtrace techniques above, this is enough to diagnose roughly 95% of real-world native crashes in ML codebases.

```bash
gdb -ex r --args python -m pytest -sv tests/test_failing.py   # -ex r: start running immediately
(gdb) bt
(gdb) c
```

Read more: [Run the program under gdb](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#run-the-program-under-gdb). Writing tests that catch these crashes earlier: [Testing](../development/testing.md).

**9.6. A missing symbol is usually a link-time or library-version problem, and `nm` lets you check what's actually there.**

- `undefined symbol: util_b` at runtime, even though the function exists in your codebase, is often caused by an older build of a shared library being loaded (for example via an `LD_LIBRARY_PATH` pointing at it).
- `nm` lists the symbols (names of functions and global variables) a binary or library defines or needs. In `nm ./your_program | grep symbol_name`, `T` means defined there and `U` means referenced but undefined (to be found in some other library).
- `ldd` shows which library files a binary actually resolved to ([Shared libraries and LD_LIBRARY_PATH](../foundations/linux-and-os.md#shared-libraries-and-ld_library_path)). With both, you can compare two same-named builds and see that one has the symbol and the other doesn't.
- `nm` for "is the symbol here?" plus `ldd` for "which file got loaded?" resolves most missing-symbol errors.

A self-contained demo with two builds of "the same" library:

```bash
# nm_demo.sh - two builds of "the same" library, only the new one defines util_b
echo 'int util_a(void){return 1;}'                               > old.c
echo 'int util_a(void){return 1;} int util_b(void){return 2;}'  > new.c
echo 'int util_b(void); int main(void){return util_b();}'        > app.c
cc -shared -fPIC old.c -o libold.so
cc -shared -fPIC new.c -o libnew.so
cc -c app.c -o app.o
echo "--- app.o needs:";      nm app.o      | grep util
echo "--- libold.so defines:"; nm libold.so | grep util
echo "--- libnew.so defines:"; nm libnew.so | grep util
```

**[ran]** on macOS (which prefixes C symbols with `_`; on Linux the names appear without it), output:

```text
--- app.o needs:
                 U _util_b
--- libold.so defines:
00000000000002d0 T _util_a
--- libnew.so defines:
00000000000002d0 T _util_a
00000000000002d8 T _util_b
```

How to read it: the application needs `util_b` (`U`), only `libnew.so` defines it (`T`), so if `ldd` shows the application resolving to the old build, you have found your "undefined symbol" error. On Linux use `nm -D` for the dynamic symbol table of a stripped `.so`.

Read more: [nm](https://github.com/stas00/the-art-of-debugging/blob/master/compiled-programs/README.md#nm).


## You can now read

You have reached the end of the book. From here, the logbooks above are the natural next step: with every chapter behind you, you can read the BLOOM-176B, OPT-175B and IDEFICS-80B chronicles and the Llama 3, MegaScale and Apertus infrastructure sections and follow every decision in them.

Parts of this chapter are adapted from Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (CC BY-SA 4.0).

## Further reading

- Stas Bekman, [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering), the original source of much of this book's material, and his [The Art of Debugging](https://github.com/stas00/the-art-of-debugging), the source of section 9 of the lessons.
- Stas Bekman, [Building resilient ML Engineering skills](https://www.youtube.com/watch?v=IBJUt9JPKHk), a talk given on 2026-01-10 for the [GPU MODE community](https://github.com/gpu-mode): the performance reality of accelerators, network and storage, and how each one can decide the performance of the whole cluster. A spoken companion to [Getting compute](../platform/getting-compute.md) and the lessons above.
- Stas Bekman, [Python Cookbook](https://github.com/stas00/python-cookbook): everyday Python and standard-library recipes, the third of his open books.
- Everything in [Similar online guides](#similar-online-guides), [Public LLM and VLM training logbooks](#public-llm-and-vlm-training-logbooks) and [Hardware setup logbooks](#hardware-setup-logbooks) above.
