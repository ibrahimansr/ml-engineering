# Debugging Distributed and Multi-Node Programs, and Performance

A single-process bug leaves one traceback. A bug in a 512-GPU job usually leaves nothing: the loss stops printing, GPU utilization sits flat (or pinned at 100% while spinning in a collective), and the job burns GPU-hours until a timeout fires minutes later. The evidence is spread over hundreds of processes on dozens of machines. Most of the work is finding *which* rank is somewhere it shouldn't be, and *where* in the code it is.

This chapter is the multi-process, multi-node companion to [Debugging PyTorch programs](debugging-pytorch.md). It covers, in the order you reach for them:

1. A small test script that checks the GPUs and the network before you blame your code.
2. Stack dumps of every rank with `py-spy`, on one node and across many.
3. PyTorch's own NCCL hang diagnostics: the desync report and the flight recorder.
4. Network-level hangs and isolating bad GPUs.
5. Tracing every Python call when a process stops responding, printing from many ranks without garbled output, and core files from processes stuck in native code.
6. Getting nodes to talk to each other in the first place, and attaching a debugger to one rank.
7. Performance: measuring durations correctly, `torch.profiler` in one shot, and a case study where `cProfile` found what `torch.profiler` could not.

The methods were developed on multi-node multi-GPU PyTorch training, but they work for any multi-process, multi-node Python program.

Labels follow [the toolkit's conventions](../foundations/toolkit.md#how-to-read-this-books-labels). The runs marked **[ran]** were done on an Apple-silicon MacBook (10 CPU cores, macOS) with Python 3.14 and PyTorch 2.15.0 (CPU build), using the `gloo` backend where a process group was needed. Commands that need Linux, NVIDIA GPUs, `sudo` or a cluster are marked **[GPU, untested]** or **untested**. Output from GPU clusters that I could not reproduce here is labelled as quoted from the original source (Stas Bekman's runs), with the PyTorch version where it is known.


## Prerequisites

- [Debugging PyTorch programs](debugging-pytorch.md), the previous chapter: async CUDA errors, `strace`, and reading a `gdb` backtrace from a core file (its section on segfaults and core files). This chapter reuses all three.
- [Linux and the OS](../foundations/linux-and-os.md): [signals](../foundations/linux-and-os.md#signals), [`py-spy`](../foundations/linux-and-os.md#py-spy-what-is-python-doing-right-now) and its `ptrace` permission issue, [`gdb` basics](../foundations/linux-and-os.md#gdb-basics), [ssh](../foundations/linux-and-os.md#working-remotely-ssh-scp-rsync), [quoting](../foundations/linux-and-os.md#quoting-the-source-of-many-bugs).
- [Concurrency: a distributed hang is a deadlock across machines](../foundations/concurrency.md#a-distributed-hang-is-a-deadlock-across-machines): why a skipped collective freezes every rank, and the NCCL watchdog.
- [Distributed systems: the distributed debugging mindset](../foundations/distributed-systems.md#the-distributed-debugging-mindset): [per-rank logs](../foundations/distributed-systems.md#per-rank-logs-and-rank-prefixed-printing), [hangs vs crashes](../foundations/distributed-systems.md#hangs-vs-crashes) (and `faulthandler`), [py-spy on all ranks](../foundations/distributed-systems.md#py-spy-on-all-ranks).
- [Networking](../foundations/networking.md): [ports and sockets](../foundations/networking.md#ports-and-sockets), [interfaces](../foundations/networking.md#interfaces-and-the-routing-table), [firewalls and `MASTER_PORT`](../foundations/networking.md#firewalls-and-why-master_port-gets-blocked), [how PyTorch distributed bootstraps](../foundations/networking.md#how-pytorch-distributed-bootstraps).
- [Measuring and debugging the network](../network/measuring-and-debugging.md): [diagnosing NCCL connectivity](../network/measuring-and-debugging.md#how-to-diagnose-nccl-multi-gpu-and-multi-node-connectivity-issues) and the [NCCL debug environment variables](../network/measuring-and-debugging.md#useful-nccl-debug-environment-variables). This chapter does not repeat them; it uses them.
- [Slurm from zero](../orchestration/slurm/from-zero.md): [`squeue`](../orchestration/slurm/from-zero.md#squeue-what-is-queued-and-running), [`srun`](../orchestration/slurm/from-zero.md#srun-launching-a-step), [hostfiles](../orchestration/slurm/from-zero.md#hostfiles-for-deepspeed-and-mpi-launchers), [`SLURM_PROCID` early interpolation](../orchestration/slurm/from-zero.md#slurm_procid-early-interpolation).
- [Profiling and optimization](../training/profiling-and-optimization.md): [timing GPU code correctly](../training/profiling-and-optimization.md#timing-gpu-code-correctly) and [`torch.profiler` end to end](../training/profiling-and-optimization.md#torchprofiler-end-to-end). The performance half of this chapter adds to those, it does not replace them.


## Start with a test script, not your program

Before debugging your training code, find out whether the GPUs and the network work at all. A full training program takes minutes to start and can fail for a dozen unrelated reasons. A script that only initializes the process group, allocates some GPU memory and runs a few collectives fails fast and for one reason.

The repository ships one: [torch-distributed-gpu-test.py](../debug/torch-distributed-gpu-test.py). It checks that every GPU in the job (one node or many) can allocate memory and talk to every other GPU through NCCL, and prints other useful facts such as each GPU's NUMA affinity. It is mostly a tool for network problems, and a quick way to see how multi-GPU communication works. Its docstring has the launch commands for `torchrun`, for Slurm, and for `[hostname:rank]` log prefixes. The basic single-node form **[GPU, untested]**:

```bash
torchrun --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

If it hangs in a `barrier`, the problem is the network or the GPUs, not your code; [Measuring and debugging the network](../network/measuring-and-debugging.md#how-to-diagnose-nccl-multi-gpu-and-multi-node-connectivity-issues) walks through `NCCL_DEBUG=INFO` output and interface selection. If the script passes and your program still hangs, the problem is in the code, and the rest of this chapter applies.


## Where is every rank stuck? py-spy

[`py-spy`](https://github.com/benfred/py-spy) reads the Python stack of a running process from outside, without stopping it. [Linux and the OS](../foundations/linux-and-os.md#py-spy-what-is-python-doing-right-now) introduced it; here we run it on one process, then every process on a node, then every node.

### One process

Install it with `pip install py-spy`, then attach to a hanging process **[untested here: needs Linux, or `sudo` on macOS]**:

```bash
py-spy dump -n -p PID
```

- `PID` is the process id of the hanging Python process.
- `-n` (`--native`) also shows frames from extensions written in C, C++ and so on. Use it, since the program may be hanging inside one of them.
- You may need `sudo` in front; see [when you need to run as sudo](https://github.com/benfred/py-spy/blob/master/README.md#when-do-you-need-to-run-as-sudo).

Without `sudo`, your sysadmin can relax the kernel's `ptrace` restriction (`ptrace` is the kernel interface that lets one process inspect another; on Ubuntu `kernel.yama.ptrace_scope=1` by default, which allows tracing only your own descendant processes). Any of these sets it to `0` **[untested]**:

```bash
sudo echo 0 > /proc/sys/kernel/yama/ptrace_scope   # often fails: the redirect runs as you, not root
sudo sysctl -w kernel.yama.ptrace_scope=0
sudo bash -c "echo 0 > /proc/sys/kernel/yama/ptrace_scope"
```

If the first form fails with "Permission denied", that is why: `sudo` applies to `echo`, but the `>` redirect is done by your own shell. The other two run the write as root. To make the change survive a reboot, put this line in `/etc/sysctl.d/10-ptrace.conf`:

```
kernel.yama.ptrace_scope = 0
```

This lets you run `py-spy` (and `strace`) without `sudo`. Mind the [security implications](https://wiki.ubuntu.com/SecurityTeam/Roadmap/KernelHardening#ptrace_Protection): any of your processes can now read the memory of any other process you own. On a compute node not reachable from the Internet this is usually an acceptable risk.

Here is a `py-spy dump` of a rank hung during Megatron-DeepSpeed training (quoted from the original source):

```
Thread 835995 (active): "MainThread"
    broadcast (torch/distributed/distributed_c10d.py:1191)
    _aggregate_total_loss (deepspeed/runtime/pipe/engine.py:540)
    train_batch (deepspeed/runtime/pipe/engine.py:330)
    train_step (megatron/training.py:436)
    train (megatron/training.py:851)
    pretrain (megatron/training.py:187)
    <module> (pretrain_gpt.py:239)
```

How to read it: the stack is printed innermost first, so the **first line is where the program is stuck**: inside `broadcast`, a collective, called from DeepSpeed's pipeline engine while aggregating the loss. Very often it is an NCCL collective or a `barrier`. If the hang is inside a C++ extension, `--native` adds the non-Python frames.

A process with several threads gets one stack per thread:

```
Thread 0x7F6D3C29D740 (idle): "MainThread"
    wait (threading.py:312)
    result (concurrent/futures/_base.py:435)
    main (slurmeventd.py:208)
    <module> (slurmeventd.py:217)
Thread 0x7F6CF5FFB700 (idle): "Thread-CallbackRequestDispatcher"
    wait (threading.py:312)
    get (queue.py:171)
    _get_many (pubsub_v1/subscriber/_protocol/helper_threads.py:56)
    __call__ (pubsub_v1/subscriber/_protocol/helper_threads.py:103)
    run (threading.py:892)
    _bootstrap_inner (threading.py:954)
    _bootstrap (threading.py:912)
```

`MainThread` is the one running your program's main code; the others are helper threads (here a Pub/Sub client's dispatcher). `(idle)` vs `(active)` says whether the thread was on a CPU when sampled.

### Every process on one node

Attaching to 8 processes one by one is too slow. `pgrep` finds them for you. If the launch command was `python` **[untested]**:

```bash
pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}
```

`pgrep -o python` returns the PID of the **oldest** process named `python`, which is the launcher; `pgrep -P PID` lists that process's direct children, the per-GPU worker processes; and `xargs -I {}` runs `py-spy dump` once per child PID, substituting it for `{}`. Replace `python` with the launcher's name:

```bash
pgrep -P $(pgrep -o deepspeed) | xargs -I {} py-spy dump --pid {}     # deepspeed
pgrep -P $(pgrep -o accelerate) | xargs -I {} py-spy dump --pid {}    # accelerate
```

This dumps only the launcher's direct children, not other sub-processes and threads they spawned, so with 8 GPUs and 8 processes you get 8 stack traces. To show just the top 5 frames of each main thread:

```bash
pgrep -P $(pgrep -o deepspeed) | xargs -I {} py-spy dump --pid {} | grep -A5 MainThread
```

Variations you will need:

```bash
# defunct zombies from earlier runs that can't be killed: pick the newest launcher (-n), not the oldest (-o)
pgrep -P $(pgrep -n deepspeed) | xargs -I {} py-spy dump --pid {}

# an extra wrapper launcher adds one more Python parent level: add one more pgrep -P
pgrep -P $(pgrep -P $(pgrep -n deepspeed)) | xargs -I {} py-spy dump --pid {}

# every python process and every sub-process (replace python with the launcher's name if needed)
pgrep -f python | xargs -I {} py-spy dump --pid {}
```

What to look for: the odd one out. If 7 stacks end in `all_reduce` and one in `torch.load` or in a different collective, that rank is the bug.

### Every node, via `srun`

With several nodes you can of course `ssh` into each and dump the stacks. On Slurm, `srun` can do it on all nodes at once. First, in another terminal, find the job id (or read it from the `salloc` log):

```bash
squeue -u `whoami` -o "%.16i %9P %26j %.8T %.10M %.8l %.6D %.20S %R"
```

Then, with the job id from that output (here `2180718`), run a second job step inside the same allocation **[untested]**:

```bash
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'ps aux | grep python | egrep -v "grep|srun" | grep `whoami` | awk "{print \$2}" | xargs -I {} py-spy dump --native --pid {}' || echo "failed"
```

The `sh -c '...'` part runs on every node: list your Python processes (excluding the `grep` and `srun` themselves), take the PID column with `awk`, and dump each one. Notes:

- `--gres=gpu:0` is required. Otherwise the monitoring step asks for the GPUs the training step already holds, and blocks until training exits.
- `--output=trace-%N.out` writes one file per node, `trace-<nodename>.out`, which tells you which node(s) are the problem. Drop it to get everything on stdout.
- Some Slurm versions also need `--overlap` to share the allocation.
- In some Slurm versions the job id does not match the one `squeue` shows; take the correct `SLURM_JOB_ID` from the logs of the `srun` job that allocated the GPUs.
- Sometimes `bash` doesn't work where `sh` does, probably because of which dot files get `source`d.
- If you need a custom Python environment, activate it inside the command, or in `~/.bashrc` (or your shell's rc file):

```bash
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'conda activate myenvname; ps auxc | ... ' || echo "failed"
```

The command above dumps every Python process. For only the launcher's children, as in the single-node section (adjust `python` as explained there):

```bash
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}' || echo "failed"
```

If you get nothing back, build the command up one stage at a time and check each stage's output: the first should print a date per node, the second and third PIDs:

```bash
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 --output=trace-%N.out sh -c 'date'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'date'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -o python'
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) '
srun --jobid=2180718 --gres=gpu:0 --nodes=40 --tasks-per-node=1 sh -c 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}'
```

### Every node, via `pdsh`

[`pdsh`](https://github.com/chaos/pdsh) runs one command over ssh on many hosts at once, and is available from the normal OS package installer. With 2 nodes named `nodename-5` and `nodename-8`, first check that remote execution works at all (quoted output from the original source):

```bash
$ PDSH_RCMD_TYPE=ssh pdsh -w nodename-[5,8] "date"
nodename-5: Wed Oct 25 04:32:43 UTC 2023
nodename-8: Wed Oct 25 04:32:45 UTC 2023
```

`PDSH_RCMD_TYPE=ssh` tells `pdsh` to connect with ssh, `-w` takes the host list (`nodename-[5,8]` expands to the two names), and each output line is prefixed with the host it came from. Once `date` works, move to `py-spy` **[untested]**:

```bash
PDSH_RCMD_TYPE=ssh pdsh -w nodename-[5,8] 'pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}'
```

This usually fails because `py-spy` is not on the remote `PATH`. A remote command runs a non-interactive shell, and most `~/.bashrc` files start with:

```bash
# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac
```

so Bash exits the rc file early and whatever it sets up (`PATH`, conda activation) never happens. Either delete that block, or copy `~/.bashrc` to a startup file that contains only what the remote command needs, and `source` it explicitly. The name `~/.pdshrc` is just a convention; nothing reads it automatically:

```bash
PDSH_RCMD_TYPE=ssh pdsh -w nodename-[5,8] 'source ~/.pdshrc; pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}'
```

If `py-spy` needs `sudo` on your nodes (the `ptrace` restriction above):

```bash
PDSH_RCMD_TYPE=ssh pdsh -w nodename-[5,8] 'sudo bash -c "source ~/.pdshrc; pgrep -P $(pgrep -o python) | xargs -I {} py-spy dump --pid {}"'
```

Edit the `pgrep` part to narrow down which processes you watch. To avoid being asked

```
Are you sure you want to continue connecting (yes/no/[fingerprint])?
```

for every node you haven't logged into before, disable host key checking:

```bash
echo "Host *" >> ~/.ssh/config
echo "  StrictHostKeyChecking no" >> ~/.ssh/config
```

This assumes an isolated cluster where replaced nodes routinely come back with new host keys. Disabling the check removes ssh's protection against a machine impersonating your node, so don't do it on machines reachable from the Internet.

### Every node, via `ds_ssh`

Read the `pdsh` section first; this is yet another way, using `ds_ssh`, DeepSpeed's small `pdsh` wrapper (installed by `pip install deepspeed`). It runs one command on every host listed in a DeepSpeed hostfile: one `hostname slots=N` line per node, `N` being the number of GPUs.

In one Slurm environment the original author could not get `py-spy` to work remotely through `ds_ssh`: the remote `ssh` command did not get the same environment as an interactive login. With `sudo` on the compute nodes it does work. First write the hostfile from inside the job:

```bash
function makehostfile() {
perl -e '$slots=split /,/, $ENV{"SLURM_STEP_GPUS"};
$slots=8 if $slots==0; # workaround 8 gpu machines
@nodes = split /\n/, qx[scontrol show hostnames $ENV{"SLURM_JOB_NODELIST"}];
print map { "$_ slots=$slots\n" } @nodes'
}
makehostfile > hostfile
```

It counts the GPU ids in `SLURM_STEP_GPUS` for the slot count (falling back to 8), expands the job's node list with `scontrol show hostnames`, and prints `node slots=N` per node. Adapt `$slots` to your GPUs per node, and the parsing if your `scontrol` prints something different. [Slurm from zero](../orchestration/slurm/from-zero.md#hostfiles-for-deepspeed-and-mpi-launchers) has a cleaner version of the same function, with a test run. Then **[untested]**:

```bash
ds_ssh -f hostfile "source ~/.pdshrc; ps aux | grep python | grep -v grep | grep `whoami` | awk '{print \$2}' | xargs -I {} sudo py-spy dump --pid {} "
```

- Put whatever init code you need in `~/.pdshrc`; if you need none, drop `source ~/.pdshrc;`.
- If you get `rcmd: socket: Permission denied`, `export PDSH_RCMD_TYPE=ssh`.

### Loops hide which iteration a rank is in

Stack traces have one weakness: they show the line, not the iteration. With

```python
for i, d in enumerate(data):
    some_hanging_call(d)
```

one process can hang in the first iteration and another in the second, and `py-spy` reports the same line number for both, although they are not at the same point in the program. While debugging, unroll the loop:

```python
d_iter = iter(data)
some_hanging_call(next(d_iter))
some_hanging_call(next(d_iter))
```

Now each iteration has its own line, so a process stuck in the first iteration reports the first `some_hanging_call` and one stuck in the second reports the second.


## Diagnosing NCCL collective hangs

When a multi-GPU run hangs, `py-spy` shows *where* each rank's Python is parked, but usually every rank is blocked inside the same NCCL wait, which does not say what went wrong. The useful questions are *which* collective failed to match across ranks, and *where in your code* it was issued. PyTorch's ProcessGroupNCCL watchdog (the background thread that monitors collectives and aborts on timeout, introduced in [Concurrency](../foundations/concurrency.md#a-distributed-hang-is-a-deadlock-across-machines)) answers both, in two escalating steps:

1. A **desync report**, which names the rank and the collective that fell out of step.
2. A **flight recorder** dump: the full per-rank history of recent collectives, each with the Python call stack that issued it.

Start with the first; reach for the second only when you need more.

Both come from the `nccl` backend's watchdog. Measured on PyTorch 2.14.0 (in the original source): if the process group uses the [`nccl2` implementation](../training/profiling-and-optimization.md#choosing-between-nccl-nccl2-and-nccl-lazy) instead, selected with `backend="nccl2"` or with `TORCH_DIST_USE_NCCL2=1` (which redirects plain `"nccl"` to it), none of the `TORCH_NCCL_*` and `TORCH_FR_*` variables below do anything. The same hang then reports a single `Operation timed out` line and a Python traceback pointing at the collective the rank was blocked in, rather than the one that diverged, with no desync report and no dump file. Reproduce the hang under `nccl` to diagnose it. (The 2.14 release notes describe a flight recorder that records through process-group hooks rather than only under NCCL. A stock 2.14.0 wheel does not do that for `nccl2`, which is why the advice is to switch back rather than to expect a dump.)

### The bug we will diagnose

[collective_mismatch.py](../debug/code/collective_mismatch.py) has rank 0 issue one extra `all_reduce` that the other rank never joins:

```python
#!/usr/bin/env python
import torch, torch.distributed as dist
from datetime import timedelta

def buggy(x, rank):
    dist.all_reduce(x)          # both ranks take part - fine
    if rank == 0:
        dist.all_reduce(x)      # BUG: only rank 0 calls this -> everyone hangs

def main():
    dist.init_process_group("nccl", timeout=timedelta(seconds=8))
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    x = torch.ones(4, device="cuda")
    buggy(x, rank)
    dist.barrier()

main()
```

Rank 0 waits in its second `all_reduce` for a partner that never comes; rank 1 has moved on to `barrier` and waits there. Neither can proceed.

To see what you get *without* the NCCL tooling, here is the same program on CPU with `gloo` (`init_process_group("gloo", ...)`, no `.cuda()`), which is what you can run on a laptop **[ran]**:

```bash
$ GLOO_SOCKET_IFNAME=lo0 torchrun --nnodes 1 --nproc_per_node=2 --master-addr 127.0.0.1 --master-port 29512 mismatch_gloo.py
[rank0]:   File "/private/tmp/absorb/run/debug2/mismatch_gloo.py", line 8, in buggy
[rank0]:     dist.all_reduce(x)      # BUG: only rank 0 calls this
...
[rank0]: RuntimeError: [.../gloo/transport/uv/unbound_buffer.cc:65] Timed out waiting 8000ms for recv operation to complete
[rank1]:   File "/private/tmp/absorb/run/debug2/mismatch_gloo.py", line 15, in main
[rank1]:     dist.barrier()
...
[rank1]: RuntimeError: [.../gloo/transport/uv/unbound_buffer.cc:65] Timed out waiting 8000ms for recv operation to complete
```

(Trimmed to the relevant frames. `GLOO_SOCKET_IFNAME=lo0` and `--master-addr 127.0.0.1` keep everything on the macOS loopback interface.) How to read it: after 8 s each rank fails where *it* was blocked, rank 0 in the stray `all_reduce` and rank 1 in `barrier`. With two ranks and a short script you can work out the story from that. With 512 ranks and a real training loop, "everyone timed out somewhere" is not a diagnosis. That is what the NCCL tooling adds.

### The desync report

For most hangs the desync report is enough. Enable it before launching the program **[GPU, untested here]**:

```bash
export TORCH_NCCL_DESYNC_DEBUG=1
export TORCH_FR_BUFFER_SIZE=2000
torchrun --standalone --nproc_per_node=2 collective_mismatch.py
```

`TORCH_NCCL_DESYNC_DEBUG=1` makes the watchdog, at timeout, name which collective and which rank fell out of step. `TORCH_FR_BUFFER_SIZE` (any non-zero value turns the flight recorder on; it is the number of recent collectives each rank remembers) is what adds the source stack trace of the stuck call to that report.

The repro passes a short `timeout=timedelta(seconds=8)` to `init_process_group`, which is why the watchdog fires after 8 s. Unless `default_pg_nccl_timeout` is overridden, the NCCL default is 10 minutes, so the watchdog won't fire until then. While chasing a hang, lower it to seconds or a few minutes so the report lands promptly, and remember to undo it before production. From PyTorch 2.14 you don't have to edit the `init_process_group` call: `dist.set_timeout(timedelta(seconds=8))` changes the timeout of an existing group, and the same call can stretch it back out around something slow, like a checkpoint load.

On timeout the watchdog names the culprit and prints the call stack of the stuck collective (quoted from the original source, PyTorch 2.14, 2 GPUs):

```
[Rank 0] Watchdog caught collective operation timeout: WorkNCCL(SeqNum=2, OpType=ALLREDUCE, NumelIn=4, NumelOut=4, Timeout(ms)=8000) ran for 8097 milliseconds before timing out.
 - [0] Timeout at collective: ALLREDUCE, #3
 - To our best knowledge, the lagging/dead/mismatched ranks that caused the desync are:
     [1] finished collective #2, but didn't join collective #3 (count from 1)
Stack trace of the failed collective:
#2 buggy from collective_mismatch.py:8
#3 main from collective_mismatch.py:15
#4 <module> from collective_mismatch.py:18
```

How to read it: line 1 is the timeout itself (`SeqNum=2` is the process group's 0-based sequence number of the operation, `NumelIn=4` the tensor size, and it ran 8097 ms against an 8000 ms limit). The `- [1] finished collective #2, but didn't join collective #3` line is the diagnosis: rank 1 is the one that diverged. The stack trace points at `collective_mismatch.py:8`, the stray `all_reduce`, and, crucially, at `main:15`, which called `buggy`, so you learn *how* the bad call was reached, not just that it happened. That is usually enough to find and fix the bug.

### The flight recorder trace

For a more detailed picture, have every rank write its full flight-recorder history to disk **[GPU, untested here]**:

```bash
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_DUMP_TEMP_FILE=/tmp/fr
torchrun --standalone --nproc_per_node=2 collective_mismatch.py
```

- `TORCH_NCCL_DUMP_ON_TIMEOUT=1` dumps the buffer to disk when the watchdog fires. Without it the recorder still runs, but nothing is written out.
- `TORCH_FR_DUMP_TEMP_FILE=/tmp/fr` is the path prefix for the dump. Each rank appends its rank number, giving `/tmp/fr0`, `/tmp/fr1`, and so on. On a multi-node job, point it at a shared filesystem or collect the files from every node afterwards.
- `TORCH_FR_BUFFER_SIZE` is as above.

`TORCH_FR_BUFFER_SIZE` and `TORCH_FR_DUMP_TEMP_FILE` were named `TORCH_NCCL_TRACE_BUFFER_SIZE` and `TORCH_NCCL_DEBUG_INFO_TEMP_FILE` before PyTorch 2.9; the old names still work but warn. These `TORCH_NCCL_*`/`TORCH_FR_*` variables are PyTorch-level controls, distinct from NCCL's own `NCCL_*` variables such as `NCCL_DEBUG`.

Each dump is a pickle. The helper [collective_mismatch_analyze.py](../debug/code/collective_mismatch_analyze.py) loads one and, for each collective that did not complete, prints its call site and caller with PyTorch's own frames stripped out:

```python
#!/usr/bin/env python
import pickle, glob, os, torch

SHOW_ALL   = False   # False: only collectives that hung; True: every collective
NUM_FRAMES = 2       # user frames to show: the actual call site + who called it

TORCH_DIR = os.path.dirname(torch.__file__)      # wherever torch actually lives
def is_user(frame):                              # skip torch's own collective plumbing
    return not frame["filename"].startswith(TORCH_DIR)

d = pickle.load(open(sorted(glob.glob("/tmp/fr*"))[0], "rb"))
for e in d["entries"]:
    if SHOW_ALL or e["state"] != "completed":
        print(e["state"], e["profiling_name"])
        user = [f for f in e["frames"] if is_user(f)]   # innermost first
        for f in user[:NUM_FRAMES]:                      # call site, then its caller
            print("   ", f["name"], os.path.basename(f["filename"]) + ":" + str(f["line"]))
```

Output (quoted from the original source):

```
started nccl:all_reduce
    buggy collective_mismatch.py:8
    main collective_mismatch.py:15
scheduled nccl:all_reduce_barrier
    main collective_mismatch.py:16
    <module> collective_mismatch.py:18
```

How to read it: each entry has a state, `scheduled` (enqueued), `started` (the kernel began) or `completed`. An entry left in `started` or `scheduled` is the hang, and its first non-torch frame is the offending call, here `collective_mismatch.py:8`. Raise `NUM_FRAMES` for deeper call chains. Set `SHOW_ALL = True` for the whole timeline, including the collectives that completed before the divergence:

```
completed nccl:all_reduce
    buggy collective_mismatch.py:6
    main collective_mismatch.py:15
started nccl:all_reduce
    buggy collective_mismatch.py:8
    main collective_mismatch.py:15
scheduled nccl:all_reduce_barrier
    main collective_mismatch.py:16
    <module> collective_mismatch.py:18
```

This is usually faster than reading raw stacks, because comparing ranks' timelines surfaces the ordering mismatch behind most NCCL hangs: mismatched sizes, a different collective order, or a collective one rank skipped.

Primary reference: [ProcessGroupNCCL environment variables](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html).

### Debug environment variables at a glance

All the variables this chapter uses, in one place. NCCL's own variables are explained in [Measuring and debugging the network](../network/measuring-and-debugging.md#useful-nccl-debug-environment-variables); variable names change between versions, so check them against yours.

| Variable | Layer | What it does |
| --- | --- | --- |
| `TORCH_NCCL_DESYNC_DEBUG=1` | PyTorch (`nccl` backend) | at timeout, report which rank and collective desynced |
| `TORCH_FR_BUFFER_SIZE=N` | PyTorch (`nccl` backend) | enable the flight recorder, keeping the last `N` collectives; was `TORCH_NCCL_TRACE_BUFFER_SIZE` before 2.9 |
| `TORCH_NCCL_DUMP_ON_TIMEOUT=1` | PyTorch (`nccl` backend) | write the flight recorder to disk when the watchdog fires |
| `TORCH_FR_DUMP_TEMP_FILE=/path/prefix` | PyTorch (`nccl` backend) | dump path prefix, rank number appended; was `TORCH_NCCL_DEBUG_INFO_TEMP_FILE` before 2.9 |
| `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` | PyTorch | crash instead of hanging forever when a collective times out ([Slurm from zero](../orchestration/slurm/from-zero.md#nccl-settings-and-per-rank-logging)) |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | PyTorch | extra checks that collectives match across ranks; slow, debugging only |
| `NCCL_DEBUG=INFO` | NCCL | print transports, interfaces, topology; see below |
| `NCCL_DEBUG_FILE=nccl.%h.%p.log` | NCCL | one log file per process ([details](../network/measuring-and-debugging.md#nccl_debug_file)) |
| `NCCL_SOCKET_IFNAME=...` | NCCL | which network interface to use ([details](../network/measuring-and-debugging.md#nccl_socket_ifname)) |
| `NCCL_P2P_DISABLE=1` | NCCL | rule the intra-node P2P path in or out ([details](../network/measuring-and-debugging.md#nccl_p2p_disable)) |
| `CUDA_VISIBLE_DEVICES=0,1` | CUDA | run on a subset of GPUs to isolate a bad one |
| `TORCH_LOGS=recompiles` | PyTorch (`torch.compile`) | log every recompilation and the guard that triggered it ([below](#a-laptop-reproduction-recompiles-under-torchcompile)) |


## Network-level hangs and bad GPUs

### Is it the network?

The hang could be at the network level. Run the test script with `NCCL_DEBUG=INFO` and look for obvious errors. Among other things it tells you which device NCCL uses (quoted from the original source):

```
DeepWhite:21288:21288 [0] NCCL INFO NET/Socket : Using [0]enp67s0:192.168.50.21<0>
```

How to read it: `DeepWhite` is the host, `21288:21288` the PID and thread id, `[0]` the GPU, and `NET/Socket` says the collectives run over plain TCP sockets on interface `enp67s0` with address `192.168.50.21`. Is that address firewalled? Is it a misconfigured network device? Does it work over the loopback device, `127.0.0.1` (single node only) **[GPU, untested]**?

```bash
NCCL_DEBUG=INFO NCCL_SOCKET_IFNAME=lo torchrun --nproc_per_node 4 --nnodes 1 torch-distributed-gpu-test.py
```

If not, list the node's other interfaces with `ifconfig` (or `ip addr`) and try those in place of `lo`. [Measuring and debugging the network](../network/measuring-and-debugging.md#how-to-diagnose-nccl-multi-gpu-and-multi-node-connectivity-issues) explains the `NET/` line in detail, which interfaces to avoid, and the `NCCL_SOCKET_IFNAME` syntax.

### Isolate problematic GPUs

Check whether only some GPUs fail. On a 4-GPU node, try the first pair, then the second **[GPU, untested]**:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node 2 --nnodes 1 torch-distributed-gpu-test.py
```

`CUDA_VISIBLE_DEVICES` hides all other GPUs from the process. If one pair passes and the other hangs, keep halving until one GPU (or one GPU pair's link) is left. The same bisection works at node level: if 64 nodes hang and 32 don't, test each half. If you suspect faulty hardware rather than code, see [Xid errors](../compute/gpu-software-stack.md#xid-errors-reading-finding-and-looking-them-up).

### AMD/ROCm: hangs or slowness with the IOMMU

On AMD GPUs, multi-GPU runs that hang or run slowly are often caused by the IOMMU (the unit that translates device memory addresses) configuration. See [AMD: hangs or slow multi-GPU runs and the IOMMU](../compute/gpu-software-stack.md#amd-hangs-or-slow-multi-gpu-runs-and-the-iommu).


## When the process stops responding: tracing every call

Sometimes the process doesn't just hang, it stops responding entirely, for example after a serious hardware problem. `py-spy` can't attach to a process that doesn't respond. If the problem recurs, the alternative is to record every Python call *as it happens*, like `strace` does for system calls, so that after the freeze you have the last call each process made and everything before it.

### The standard library's `trace` module

Python's [`trace`](https://docs.python.org/3/library/trace.html) module can print every line as it executes. Combined with a `Tee` that writes stdout to the terminal and to a per-process file, each rank leaves a log that ends exactly where it stopped. Here is the pattern as a self-contained script **[ran]**:

```python
# trace_demo.py - log every line executed into a per-rank file with the stdlib `trace` module
import os, socket, sys, trace

class Tee:
    """Send everything written to stdout to the terminal AND to a file."""
    def __init__(self, filename):
        self.stdout = sys.stdout
        self.file = open(filename, "a")
    def __getattr__(self, attr):
        return getattr(self.stdout, attr)
    def write(self, msg):
        self.stdout.write(msg)
        self.file.write(msg)
        self.file.flush()               # flush every line: the last line must survive a hang
    def flush(self):
        self.stdout.flush()
        self.file.flush()

def step(i):
    x = i * 2
    return x + 1

def main():
    for i in range(2):
        step(i)

if __name__ == "__main__":
    if int(os.environ.get("TRACE", "0")):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        out = f"trace-{socket.gethostname().split('.')[0]}-{local_rank}-{os.getpid()}.txt"
        tracer = trace.Trace(ignoredirs=[sys.prefix, sys.exec_prefix, os.path.dirname(os.__file__)], trace=1, count=0, timing=True)
        sys.stdout = Tee(out)
        tracer.run("main()")
    else:
        main()
```

```
$ TRACE=1 python trace_demo.py
 --- modulename: trace_demo, funcname: <module>
0.00 <string>(1)
 --- modulename: trace_demo, funcname: main
0.00 trace_demo.py(24):      for i in range(2):
0.00 trace_demo.py(25):          step(i)
 --- modulename: trace_demo, funcname: step
0.00 trace_demo.py(20):      x = i * 2
0.00 trace_demo.py(21):      return x + 1
0.00 trace_demo.py(24):      for i in range(2):
0.00 trace_demo.py(25):          step(i)
 --- modulename: trace_demo, funcname: step
0.00 trace_demo.py(20):      x = i * 2
0.00 trace_demo.py(21):      return x + 1
0.00 trace_demo.py(24):      for i in range(2):
```

How to read it: `--- modulename: ..., funcname: ...` marks entry into a function; every other line is `<seconds since start> <file>(<line>): <source>`. The last line in the file is the last line the process executed. The file name carries hostname, local rank and PID, so after a multi-node freeze you have one file per process and can tell them apart.

What to know before using it on a real job:

- In a real training script, put this in the `if __name__ == "__main__":` block around your `main()`, and switch it on (`TRACE=1` here; an `if 0:`/`if 1:` toggle works too). `LOCAL_RANK` is set by `torchrun`.
- `ignoredirs` lists directories whose code is not traced. `sys.prefix` and `sys.exec_prefix` cover installed packages, but inside a virtual environment they point at the venv, not at the standard library, so `os.path.dirname(os.__file__)` is added to skip the standard library too (see [the trace trap in Debugging PyTorch Programs](debugging-pytorch.md)). Without these, every Python call is logged: expect many GBs of data, multiplied by hundreds of GPUs.
- You don't have to start from `main`. If you suspect a specific area, start tracing there; it's much faster and saves less data.
- `trace` can only be told which directories to *ignore*, not which packages to *follow*, which is harder to set up and gives much more data than needed.
- The code runs much slower under tracing, and slower still the more packages you trace.
- `count=1` additionally collects line-execution counts (the original version of this snippet used it); for hang hunting only `trace=1` matters.

### NicerTrace

Because `trace` proved of limited use when debugging a complex multi-node, multi-hour crash, Stas Bekman started an improved version, [NicerTrace](../debug/NicerTrace.py). It adds constructor flags, most importantly `packages_to_include`, which inverts `trace`'s ignore-list: list the top-level packages you want followed, and it traces those plus anything not installed under `site-packages` (your own code and `pip install -e .` packages). The file contains a full working example; run it with `python NicerTrace.py` (needs Pillow) **[ran]**:

```
        /Users/ibrahimansari/Desktop/ml-engineering/debug/NicerTrace.py:1 <module>
0:00:00 <string>:     1:         /Users/ibrahimansari/Desktop/ml-engineering/debug/NicerTrace.py:190 main
0:00:00 NicerTrace.py:   191:     img = Image.new("RGB", (4, 4))
        PIL.Image:3209 new
0:00:00 Image.py:  3228:     _check_size(size)
        PIL.Image:3190 _check_size
0:00:00 Image.py:  3198:     if not isinstance(size, (list, tuple)):
0:00:00 Image.py:  3201:     if len(size) != 2:
0:00:00 Image.py:  3204:     if size[0] < 0 or size[1] < 0:
```

How to read it: indented lines without a timestamp are function entries, `module:line function`; timestamped lines are executed source lines, `elapsed file: line: source`. The example sets

```
            packages_to_include=["PIL"],
```

so it follows into `PIL` (Pillow), which plain `trace` with `ignoredirs` would have skipped. To trace another package, add it to that list. It is a work in progress, built while chasing one hard crash; read the constructor's docstring for the other flags (`packages_to_include="uninstalled"`, `log_pids`).

### Working with per-rank trace files

Once every node and rank has written its trace file, a few one-liners summarize them. In the `find ... -exec sh -c '...' _ {} \;` commands below, `find` runs the quoted shell snippet once per matching file; the `_` fills the snippet's `$0` (its name, unused) so that the file name `{}` arrives as `$1`, and `\;` ends the `-exec` clause. On three trace files from the script above (two ranks, plus one earlier run) **[ran]**:

```bash
$ grep -n "x + 1" trace*        # a specific match, with file name and line number
trace-Ibrahims-MacBook-Pro-1-11380.txt:8:0.00 trace_demo.py(21):      return x + 1
trace-Ibrahims-MacBook-Pro-1-11380.txt:13:0.00 trace_demo.py(21):      return x + 1
trace-Ibrahims-MacBook-Pro-0-11377.txt:8:0.00 trace_demo.py(21):      return x + 1
...
$ find . -name "trace*" -exec sh -c 'echo "$1: $(tail -1 "$1")"' _ {} \;      # last line of each file
./trace-Ibrahims-MacBook-Pro-1-11380.txt: 0.00 trace_demo.py(24):      for i in range(2):
./trace-Ibrahims-MacBook-Pro-0-11379.txt: 0.00 trace_demo.py(24):      for i in range(2):
./trace-Ibrahims-MacBook-Pro-0-11377.txt: 0.00 trace_demo.py(24):      for i in range(2):
$ find . -name "trace*" -exec sh -c 'echo "$1: $(grep "x = i" $1 | wc -l)"' _ {} \;   # matches per file
./trace-Ibrahims-MacBook-Pro-1-11380.txt:        2
./trace-Ibrahims-MacBook-Pro-0-11379.txt:        2
./trace-Ibrahims-MacBook-Pro-0-11377.txt:        2
```

Two more forms: `tail -3` of every file, and the last 5 lines with the file name on its own line and a blank line between files, which is easier to read:

```bash
find . -name "trace*" -exec sh -c 'echo "$1: $(tail -3 "$1")"' _ {} \;
find . -name "trace*" -exec sh -c 'echo; echo $1; echo "$(tail -5 "$1")"' _ {} \;
```

How to read them on a real hang: all files ending on the same line means all processes are stuck in the same place (a collective, most likely); the file whose last line differs is your suspect. The per-file match count does the same for a repeated event such as `backward`: the rank that ran it one time fewer is the one that fell behind. These are the same tricks as for [per-rank NCCL logs](../network/measuring-and-debugging.md#nccl_debug_file).


## Printing from many ranks

### `print` with a rank prefix, and `printflock`

Once you know *where* a hang happens, you need to know *why*. A debugger is the ideal tool, but debugging multi-process, multi-node programs with one is often very hard. Good old `print` works: add prints before the calls that hang, with whatever helps you understand what led to the deadlock. For example, a missing `barrier` lets one or a few processes skip some code while the rest block, waiting for everyone to send data in an NCCL collective like `gather` or `reduce`.

Prefix each print with the rank so you can tell processes apart:

```python
import torch.distributed as dist
print(f"{dist.get_rank()}: passed stage 0")
```

With many GPUs these prints get interleaved, sometimes mid-line, and are hard to make sense of. The fix is to override `print` with a version that takes `flock` (an advisory file lock that only one process can hold at a time) around each call, so only one process writes to stdout at once. The helper is [printflock.py](../training/tools/printflock.py), and its core is six lines:

```python
import builtins, fcntl

def printflock(*args, **kwargs):
    with open(__file__, "r") as fh:                 # the lock file is printflock.py itself
        fcntl.flock(fh, fcntl.LOCK_EX)              # block until no other process holds it
        try:
            builtins.print(*args, **kwargs)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
```

To make every `print` in the module you're debugging non-interleaved, put this at its top:

```python
from printflock import printflock as print
```

or call it explicitly:

```python
from printflock import printflock
import torch.distributed as dist
printflock(f"{dist.get_rank()}: passed stage 0")
```

The order of output between ranks is still random; only the lines stay whole. The lock serializes printing across processes, so keep it for debug prints, not for logging in a hot loop. A sibling script, [multi-gpu-non-interleaved-print.py](../training/tools/multi-gpu-non-interleaved-print.py), contains the same wrapper plus a full example to run with `torchrun --nproc_per_node 2 multi-gpu-non-interleaved-print.py`. As its comments note, interleaving is typically a problem only among processes on the same node; output from processes on different nodes doesn't get interleaved this way.

On my laptop, 4 processes each printing 200 lines of 2,000 characters through a pipe produced no garbled lines either with or without the lock **[ran]**, so I can't show you the failure here. It depends on the OS, on how the launcher collects output, and on the number of writers; treat `printflock` as cheap insurance on multi-GPU nodes.

### Prefixing logs with `node:rank`

Every line of output, not only your prints, can be prefixed with where it came from. With `torchrun` (`torch.distributed.run`), add `--role $(hostname -s): --tee 3`:

```bash
torchrun --role $(hostname -s): --tee 3 --nnodes 1 --nproc_per_node 2 \
torch-distributed-gpu-test.py
```

`--tee 3` copies both stdout and stderr of every worker to the console with a prefix, and `--role` sets the first part of that prefix, so each line starts with `[hostname:rank]`. The colon at the end of the role is important: it is what separates host from rank in the prefix.

Under Slurm the command becomes **[untested]**:

```bash
srun --jobid $SLURM_JOBID bash -c 'torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
--role $(hostname -s): --tee 3 \
torch-distributed-gpu-test.py'
```

Adjust the environment variables to yours. Note the **single quotes** around the command passed to `bash -c`: they delay `$(hostname -s)` until it runs on each node. With double quotes, `hostname -s` would be expanded once, on the node that launches the job, and every node would get the same prefix. If you need double quotes, escape what must be expanded per node:

```bash
srun --jobid $SLURM_JOBID bash -c "torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank \$SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
--role \$(hostname -s): --tee 3 \
torch-distributed-gpu-test.py"
```

`$SLURM_PROCID` is escaped too, since it is specific to each node and unknown when the command string is built on the first node, so this version has two `\$` escapes. Getting this wrong for `SLURM_PROCID` makes the job hang; see [`SLURM_PROCID` early interpolation](../orchestration/slurm/from-zero.md#slurm_procid-early-interpolation).

The prefix is most valuable when a distributed program crashes and every rank prints the same traceback at once. Without prefixes you get (quoted from the original source):

```
  File "/path/to/training/dataset.py", line 785, in __init__
  File "/path/to/training/dataset.py", line 785, in __init__
    if self.dataset_proba.sum() != 1:
AttributeError: 'list' object has no attribute 'sum'
    if self.dataset_proba.sum() != 1:
  File "/path/to/training/dataset.py", line 785, in __init__
  File "/path/to/training/dataset.py", line 785, in __init__
    if self.dataset_proba.sum() != 1:
    if self.dataset_proba.sum() != 1:
AttributeError: 'list' object has no attribute 'sum'
AttributeError: 'list' object has no attribute 'sum'
AttributeError: 'list' object has no attribute 'sum'
```

and with dozens of frames over 8 nodes that can't be untangled. With `--role` and `--tee`:

```
[host1:0]  File "/path/to/training/dataset.py", line 785, in __init__
[host1:1]  File "/path/to/training/dataset.py", line 785, in __init__
[host1:0]    if self.dataset_proba.sum() != 1:
[host1:0]AttributeError: 'list' object has no attribute 'sum'
[host1:1]    if self.dataset_proba.sum() != 1:
[host1:2]  File "/path/to/training/dataset.py", line 785, in __init__
[host1:3]  File "/path/to/training/dataset.py", line 785, in __init__
[host1:3]    if self.dataset_proba.sum() != 1:
[host1:2]    if self.dataset_proba.sum() != 1:
[host1:1]AttributeError: 'list' object has no attribute 'sum'
[host1:2]AttributeError: 'list' object has no attribute 'sum'
[host1:3]AttributeError: 'list' object has no attribute 'sum'
```

and grepping for one prefix reconstructs one rank's real traceback (`-F` makes `grep` treat the brackets literally):

```bash
$ grep -F '[host1:0]' log.txt
[host1:0]  File "/path/to/training/dataset.py", line 785, in __init__
[host1:0]    if self.dataset_proba.sum() != 1:
[host1:0]AttributeError: 'list' object has no attribute 'sum'
```

In a real job there can easily be hundreds to thousands of interleaved traceback lines. Recent `torchrun` versions also prefix worker tracebacks with `[rankN]:` on their own, as in the `gloo` run [above](#the-bug-we-will-diagnose), but without the host name.

More tips:

- On a single node, `--tee 3` alone is enough; `--role` adds nothing.
- If hostnames are long but numbered, like `[really-really-really-long-hostname-5:0]`, keep only the digits with `hostname -s | tr -dc '0-9'`, which gives `[5:0]`, `[5:1]`, `[5:2]`.
- For your own debug prints, combine this with [`printflock`](#print-with-a-rank-prefix-and-printflock).

The same feature in other launchers:

- `srun` in Slurm: add `--label`.
- Open MPI's `mpirun`: add `--tag-output`.
- `accelerate`: pass the same `--tee` and `--role` flags as to `torchrun`.


## Native hangs: core files

If a process hangs inside non-Python code and `py-spy --native` isn't enough, make it write a core file (a snapshot of the process's memory and threads) and inspect that with `gdb`. Either of these works **[untested]**:

```bash
gcore <pid>
kill -ABRT <pid>
```

`gcore` comes with `gdb`; it writes a core file of a running process and leaves it running. `SIGABRT` makes the process abort and dump core on its way out. Then load the core file in `gdb` as shown in the segfaults and core files section of [Debugging PyTorch programs](debugging-pytorch.md), with `thread apply all bt` to get every thread's stack. On a multi-node job, run the `gcore` through `pdsh` or `srun --jobid ... --gres=gpu:0` exactly like the `py-spy` commands [above](#every-node-via-srun), with the `pgrep` expression choosing the PIDs.

If no core file appears, you need to allow core dumps and decide where they go. In bash (other shells may use a different command):

```bash
ulimit -c unlimited
```

To make this persistent:

```bash
echo '* soft core unlimited' >> /etc/security/limits.conf
```

On some systems, like Ubuntu, core files are sent to `apport`, which may save them to `/var/lib/apport/coredump` or `/var/crash`. Check `/proc/sys/kernel/core_pattern` to see where they go, and override it with:

```bash
sudo sysctl -w kernel.core_pattern=/tmp/core-%e.%p.%h.%t
```

The `%` templates expand to the executable name (`%e`), PID (`%p`), hostname (`%h`) and time (`%t`); `man core` lists them all. Including `%h` matters on a multi-node job if the directory is on a shared filesystem, so cores from different nodes don't collide. Change the directory if you like, but the user the program runs as must be able to write there. To make it permanent, add (or edit) `kernel.core_pattern=/tmp/core-%e.%p.%h.%t` in `/etc/sysctl.conf`.

A quick test that your setup can produce a core file:

```bash
sleep 10 &
kill -SEGV $!
```

`$!` is the PID of the `sleep` just put in the background, so only that process is signalled. `SIGSEGV` is fine for this test, but don't use it on a real hanging program: it is likely to run the program's own signal handler instead of dumping core cleanly.


## Getting nodes to talk to each other

Before you can debug a multi-node job, the nodes must be able to reach each other. Once you scale past one node (for example to make DDP faster), every training step ends with GPUs on different nodes exchanging data through a communication library like [NCCL](https://github.com/NVIDIA/nccl); in DDP, an `all_reduce` that synchronizes the gradients across all ranks. For the NCCL layer itself see [Measuring and debugging the network](../network/measuring-and-debugging.md#how-to-diagnose-nccl-multi-gpu-and-multi-node-connectivity-issues); this section is the PyTorch-level checklist.

Take the simplest case: 2 nodes with 8 GPUs each, at IP addresses `10.0.0.1` and `10.0.0.2`. It extends to any number of nodes.

### Pick a port and check it is open

Rendezvous needs a TCP port (the background is in [ports and sockets](../foundations/networking.md#ports-and-sockets)). There are 65,536 ports; the first 1,024 are reserved for well-known services, so that any computer knows in advance where to connect. Port 22 is SSH, so `ssh example.com` really connects to `example.com:22`. Thousands of other services use whatever ports they like, but a fresh cloud or HPC Linux box rarely runs many services on high ports, so most are free. Let's use 6000, giving `10.0.0.1:6000` and `10.0.0.2:6000`.

The port must be open for incoming and outgoing connections on both nodes. It may be already; otherwise follow your platform's instructions for opening a port. Several tools test it, all available from your package manager:

```bash
telnet localhost 6000
nmap -p 6000 localhost
nc -zv localhost 6000
curl -v telnet://localhost:6000
```

With `nmap` (quoted output):

```bash
$ nmap -p 22 localhost
[...]
PORT   STATE SERVICE
22/tcp open  ssh
$ nmap -p 6000 localhost
[...]
PORT     STATE  SERVICE
6000/tcp closed X11
```

How to read it: `open` means something is listening and reachable; `closed` means reachable but nothing listening (fine for a port you are about to use, as long as no firewall blocks it); `filtered` (not shown) means a firewall dropped the probe. The `SERVICE` column is just `nmap`'s guess from the port number, which is why 6000 shows `X11`.

Now test each node from the other. In terminal A, from the first node to the second:

```bash
ssh 10.0.0.1
nmap -p 6000 10.0.0.2
```

and in terminal B, the reverse:

```bash
ssh 10.0.0.2
nmap -p 6000 10.0.0.1
```

If both are reachable you can use the port. If not, open it; most clouds have their own mechanism, so search for "open port" plus your provider's name. [Firewalls and why `MASTER_PORT` gets blocked](../foundations/networking.md#firewalls-and-why-master_port-gets-blocked) has the background.

### Find the right network interface

Compute nodes usually have several network interface cards (NICs). List them with:

```bash
$ sudo ifconfig
```

One interface is typically for users: ssh, email, downloading data. It is often called `eth0` (Ethernet), but names vary. Then there are the inter-node interfaces, InfiniBand, EFA, OPA, HPE Slingshot and others ([inter-node networking](../network/interconnects.md#inter-node-networking)), anywhere from one to dozens of them. Two examples, trimmed:

```bash
$ sudo ifconfig
enp5s0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 10.0.0.23  netmask 255.255.255.0  broadcast 10.0.0.255
        [...]
```

The key information is the address after `inet`: `10.0.0.23` is the IP of interface `enp5s0`. Another node on the same network will probably be `10.0.0.24` or `10.0.0.21` or similar, differing in the last number.

```bash
$ sudo ifconfig
ib0     Link encap:UNSPEC  HWaddr 00-00-00-00-00-00-00-00-00-00-00-00-00-00-00-00
        inet addr:172.0.0.50  Bcast: 172.0.0.255  Mask:255.255.255.0
        [...]
```

`ib` typically means InfiniBand, but other vendors use it too; [Omni-Path](../network/interconnects.md#omni-path) does, for example. Again `inet addr` gives the IP, `172.0.0.50`. These IPs are what you test `ip:port` against.

### Run the test script across nodes

Back to `10.0.0.1:6000` and `10.0.0.2:6000`. Run an `all_reduce` test from two terminals with [torch-distributed-gpu-test.py](../debug/torch-distributed-gpu-test.py), choosing `10.0.0.1` as the host that coordinates the others **[GPU, untested]**. Terminal A:

```bash
$ ssh 10.0.0.1
$ torchrun --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --rdzv_backend c10d --rdzv_endpoint 10.0.0.1:6000 torch-distributed-gpu-test.py
```

Terminal B:

```bash
$ ssh 10.0.0.2
$ torchrun --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --rdzv_backend c10d --rdzv_endpoint 10.0.0.1:6000 torch-distributed-gpu-test.py
```

Both sides use the same `--rdzv_endpoint 10.0.0.1:6000`, the port we checked, on the coordinating host. With `c10d` rendezvous, ranks are assigned automatically, so `--node_rank`, `--master_addr` and `--master_port` are not needed.

Running things by hand on each node is painful; these tools run the same command on many nodes.

**pdsh** folds the two commands into one; for more nodes, add them to `-w`:

```bash
PDSH_RCMD_TYPE=ssh pdsh -w 10.0.0.1,10.0.0.2 \
'torchrun --role $(hostname -s): --tee 3 --nnodes 2 --nproc_per_node 8 \
 --rdzv_backend c10d --rdzv_endpoint 10.0.0.1:6000 torch-distributed-gpu-test.py'
```

The single quotes matter: `$(hostname -s)` must expand on each remote host, not on the launching one.

**Slurm**: whoever set up the cluster almost certainly opened the ports, so it should just work. If it doesn't, the steps above help debug it. A batch script for 2 nodes **[untested]**:

```bash
#!/bin/bash
#SBATCH --job-name=test-nodes        # name
#SBATCH --nodes=2                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --time 0:05:00               # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#
export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
#
srun --jobid $SLURM_JOBID bash -c 'torchrun \
--nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
--master_addr $MASTER_ADDR --master_port $MASTER_PORT \
torch-distributed-gpu-test.py'
```

`--ntasks-per-node=1` is crucial: Slurm starts one `torchrun` per node, and `torchrun` starts the 8 per-GPU processes. `MASTER_ADDR` is the first host of the allocation. For more than 2 nodes, change `--nodes` and nothing else.

**MPI**: [Message Passing Interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface) launchers are another common way, with several open-source implementations. Write a hostfile with the target nodes and how many processes to start on each; for our 2 nodes with 8 GPUs:

```bash
$ cat hostfile
10.0.0.1:8
10.0.0.2:8
```

This `host:N` syntax is the MPICH/Hydra (and Intel MPI / MVAPICH) form. Hostfiles are not standardized across MPI implementations; pick the line for your `mpirun`:

| Implementation                    | Per-host count syntax   |
| --------------------------------- | ----------------------- |
| Open MPI                          | `10.0.0.1 slots=8`      |
| MPICH / Hydra, Intel MPI, MVAPICH | `10.0.0.1:8`            |
| MS-MPI                            | `10.0.0.1 8`            |

Then run:

```bash
$ mpirun --hostfile hostfile -np 16 -map-by ppr:8:node python my-program.py
```

`-np 16` is the total number of processes and `-map-by ppr:8:node` ("processes per resource") places 8 on each node. This uses `my-program.py` because [torch-distributed-gpu-test.py](../debug/torch-distributed-gpu-test.py) was written for `torchrun`. Under `mpirun`, check which environment variable your implementation uses to pass the local rank and use it in place of `LOCAL_RANK`; the rest stays mostly the same.

Nuances:

- You may have to tell Open MPI which interface to use: `--mca btl_tcp_if_include 10.0.0.0/24` for our example. `--mca` sets an Open MPI component parameter, here for its TCP transport (the `btl`), and `10.0.0.0/24` means every address `10.0.0.*`. With many interfaces it may otherwise pick one that is closed or simply wrong.
- Or the reverse, exclude interfaces: with `docker0` and `lo` present, add `--mca btl_tcp_if_exclude docker0,lo`.

`mpirun` has a great many flags, and different implementations use different ones; read its manpage.

### Case study: Ethernet works, InfiniBand doesn't

On Azure, two nodes on a shared subnet were given the 2-node NCCL test **[GPU, quoted from the original source]**:

```bash
NCCL_DEBUG=INFO python -u -m torch.distributed.run --nproc_per_node=1 --nnodes 2 --rdzv_endpoint 10.2.0.4:6000  --rdzv_backend c10d torch-distributed-gpu-test.py
```

The debug output showed the InfiniBand interfaces were detected:

```
node-2:5776:5898 [0] NCCL INFO NET/IB : Using [0]ibP111p0s0:1/IB [1]rdmaP1111p0s2:1/RoCE [RO]; OOB eth0:10.2.0.4<0>
```

but then the connection timed out with:

```
node-2:5776:5902 [0] transport/net_ib.cc:1296 NCCL WARN NET/IB : Got completion from peer 10.2.0.5<33092> with error 12, opcode 0, len
0, vendor err 129 (Recv)
node-2:5776:5902 [0] NCCL INFO transport/net.cc:1134 -> 6
node-2:5776:5902 [0] NCCL INFO proxy.cc:679 -> 6
node-2:5776:5902 [0] NCCL INFO proxy.cc:858 -> 6 [Proxy Thread]
```

and nothing worked. How to read it: the `NET/IB` line says NCCL found an IB port and a RoCE port, with `eth0` for out-of-band bootstrap; the bootstrap succeeded, so Ethernet between the nodes works. The `WARN` is an IB completion error from the peer, and the `-> 6` lines are the error code propagating up NCCL's call stack. So: Ethernet works, the IB fabric does not.

There are many possible reasons. The most likely on a cloud is that the two nodes were not provisioned so that their IB is connected: Ethernet works but is far too slow for training. The fix is to re-provision the nodes so they are allocated together; on Azure, inside an [availability set](https://learn.microsoft.com/en-us/azure/virtual-machines/availability-set-overview?source=recommendations). Once the nodes were deleted and recreated in an availability set, the test worked out of the box.

The general lesson: individual cloud nodes are often not meant for inter-node communication. Clouds have a concept of clusters (placement groups, availability sets and so on) for allocating nodes as a group that is already wired and configured to work together. Use it.


## A debugger on one rank

### `dist.breakpoint()` and `ForkedPdb`

Since PyTorch 2.2 you can stop one rank in `pdb` (Python's debugger) while the others wait:

```python
import torch.distributed as dist
[...]

def mycode(...):

   dist.breakpoint(0)

```

This breaks on the rank you choose, rank 0 here. When the breakpoint hits, type `up;;n` right away to get into your own code: `up` moves one frame up, out of the `breakpoint` helper into your function, `;;` separates two pdb commands, and `n` executes the next line.

Why a special helper: worker processes started by a launcher don't have the terminal as their stdin, so plain `pdb.set_trace()` can't read your keystrokes. Underneath, `dist.breakpoint` does essentially this:

```python
import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


def mycode():

    if dist.get_rank() == 0:
        ForkedPdb().set_trace()
    dist.barrier()

```

`interaction` reopens `/dev/stdin` (the controlling terminal) for the duration of the pdb session, and the `barrier` keeps the other ranks waiting while rank 0 is stopped. You can write it yourself, and use `ForkedPdb` for ordinary forked (multiprocessing) programs too, minus the `dist` calls.

### Attaching pdb to an already-running process

`dist.breakpoint()` and `ForkedPdb` both require planting the breakpoint in the source before the run. Python 3.14 adds the missing case: attaching to a process that is *already running* and that you did not prepare, given only its PID ([PEP 768](https://peps.python.org/pep-0768/)):

```bash
python -m pdb -p PID
```

Both the target and the interpreter you attach with must be Python 3.14+, since it relies on the new safe remote-debugging hook. It drops you at a `(Pdb)` prompt controlling the target wherever it happens to be (quoted from the original source):

```
$ python -m pdb -p 2916
> /tmp/target.py(6)<module>()
-> time.sleep(0.2)
(Pdb) p counter
8
(Pdb) c
```

From there the usual commands work: `p`/`pp` to print values, `bt` for the stack, `w` for the current frame. `c` resumes the process; closing the session detaches without killing it (in the run above the target continued and finished normally). There is no `detach` command; use `c` or quit the debugger.

This is the standard-library counterpart to `py-spy dump`. `py-spy` samples the stack read-only and never stops the target; `python -m pdb -p PID` stops the process and lets you evaluate expressions inside it. For a wedged rank in a multi-GPU job, that means you can inspect live state even though you never planted a `dist.breakpoint()`. Keep `dist.breakpoint(rank)` for when you *can* edit the source and want a clean, all-ranks-aware stop.

Caveats: you can normally attach only to your own processes; systems that restrict `ptrace` need the same `kernel.yama.ptrace_scope` change as [`py-spy`](#one-process); the target can opt out entirely with `-X disable-remote-debug` or `PYTHON_DISABLE_REMOTE_DEBUG=1`; and a rank blocked inside a C call will not break until that call returns, which for a rank stuck in an NCCL wait may be never.

Primary reference: [Python 3.14 pdb](https://docs.python.org/3.14/library/pdb.html).


## Measuring durations

The second half of this chapter is about speed. Faster ML software means less money spent on hardware (even on hardware you own, you pay for the electricity), and results sooner. In a race to train a model, a competitor whose framework trains faster can make your better idea irrelevant before your run finishes. Making software faster starts with measuring durations, then optimizing, then measuring again. [Profiling and optimization](../training/profiling-and-optimization.md) covers the methodology and a full training loop; this section covers the general tools.

### `time`

The simplest measurement is two timestamps:

```bash
$ python -c 'import time; start=time.time(); time.sleep(1); print(f"duration: {time.time()-start} secs")'
duration: 1.0000815391540527 secs
```

The shell's `time` measures a whole command, and reports three kinds of time (quoted from the original source, a 192-core Linux node):

```bash
$ time python -c 'import torch'
real    0m1.399s
user    0m8.222s
sys     0m0.209s
```

- `real` is wall clock time: what a stopwatch would show from launch to exit.
- `user` is CPU time spent in user space, that is, in your program.
- `sys` is CPU time spent in system calls (in the kernel): IO, networking, memory.

Here `user+sys` (`8.222+0.209=8.431`) is six times `real` (`1.399`). That's not a contradiction: `user` and `sys` are summed over every thread that ran. This node has 192 cores and importing `torch` starts a thread pool across them, so CPU-seconds pile up much faster than wall-clock seconds. Restrict it to one thread and the CPU time collapses while the wall-clock time stays put:

```bash
$ time OMP_NUM_THREADS=1 python -c 'import torch'
real    0m1.377s
user    0m1.099s
sys     0m0.258s
```

Now `user+sys` (`1.099+0.258=1.357`) does add up to `real` (`1.377`), and the 7 extra CPU-seconds of the first run bought no wall-clock time at all.

`real` can also be *bigger* than `user+sys`: that happens when the program waits instead of computing. If you build PyTorch with `make -j` on all cores and run another program at the same time, the latter reports a bigger `real` and smaller `user+sys`, because it fights dozens of `gcc` processes for CPU time and spends a lot of it waiting. Slow or blocking IO (a shared NFS filesystem, say) causes the same gap. Unless your software competes with other software, `real` is the number to care about; `user` and `sys` matter to people optimizing low-level systems.

### hyperfine

[hyperfine](https://github.com/sharkdp/hyperfine) runs a command many times and reports the mean, standard deviation, and fastest and slowest runs. It can also compare commands. Install it with `sudo apt-get install -y hyperfine` (or your package manager). On the same import (quoted from the original source):

```bash
$ hyperfine 'python -c "import torch"'
Benchmark 1: python -c "import torch"
  Time (mean ± σ):      1.412 s ±  0.024 s    [User: 8.160 s, System: 0.290 s]
  Range (min … max):    1.378 s …  1.454 s    10 runs
```

It ran the command 10 times, the default; `-r N` sets the count. The `± 0.024 s` is what `time` can't tell you: repeated runs differ by about 24 ms, so a change that saves 10 ms hasn't been shown to save anything.

To compare two commands, pass both; `-w 2` warms up by running each twice before measuring:

```bash
$ hyperfine -w 2 'python -c "import torch"' 'python -c "import transformers"'
Benchmark 1: python -c "import torch"
  Time (mean ± σ):      1.415 s ±  0.026 s    [User: 8.226 s, System: 0.224 s]
  Range (min … max):    1.379 s …  1.455 s    10 runs

Benchmark 2: python -c "import transformers"
  Time (mean ± σ):      1.211 s ±  0.017 s    [User: 8.024 s, System: 0.245 s]
  Range (min … max):    1.186 s …  1.233 s    10 runs

Summary
  python -c "import transformers" ran
    1.17 ± 0.03 times faster than python -c "import torch"
```

Importing `transformers` is faster than importing `torch`, even though `transformers` needs `torch`, because it imports it lazily:

```bash
$ python -c 'import transformers, sys; print("torch" in sys.modules)'
False
```

**Cache effects.** All the runs above most likely read the imported files from the kernel's [page cache](../foundations/linux-and-os.md#the-page-cache-why-the-second-read-is-fast), filled by earlier runs. On a new node, after a fresh install, on the first run of the day, or after the kernel reclaims that memory, they are read from disk instead. `--prepare` (`-p`) runs a command before each timed run; use it to drop the caches **[untested here: needs Linux and sudo]**:

```bash
$ hyperfine -p 'sync; echo 3 | sudo tee /proc/sys/vm/drop_caches' 'python -c "import torch"'
Benchmark 1: python -c "import torch"
  Time (mean ± σ):      2.103 s ±  0.054 s    [User: 8.319 s, System: 0.348 s]
  Range (min … max):    2.018 s …  2.191 s    10 runs
```

The same import takes 1.5x longer with nothing cached. On a networked filesystem it's far worse. Here is the `torch` package copied onto a local NVMe drive and onto a shared Lustre mount, then imported from each (from the original source):

| package on | from cache     | from the filesystem |
| :--------- | -------------: | ------------------: |
| local NVMe | 1.491s ± 0.178 |      2.134s ± 0.035 |
| Lustre     | 1.681s ± 0.494 |     10.589s ± 0.104 |

With nothing cached, Lustre takes 5.0x as long as local NVMe, and 7.1x as long as local NVMe from cache. This is why installing packages on a shared filesystem and importing them from there is so much slower than using a local disk. Writing is punished even harder: copying these same files took 2.026 s onto the local drive and 2 min 35.880 s onto Lustre. The cached Lustre number is unreliable too: those runs ranged from 1.441 s to 2.912 s, because the Lustre client doesn't keep a package that size cached for long. `User` time stays at about 8.3 s in all of these runs, so the extra wall-clock time is time spent waiting for IO. What an import actually reads, and how to build a filesystem benchmark from timings like these, is in [Usability perception IO benchmarks](../foundations/storage-and-data.md#usability-perception-io-benchmarks).

So when you compare two timings, take both the same way: both cached or both not, from the same filesystem. Otherwise you're measuring the cache and the filesystem, not your change.

### Device events

Wall-clock time does not measure GPU work correctly, because kernel launches are asynchronous; [Timing GPU code correctly](../training/profiling-and-optimization.md#timing-gpu-code-correctly) shows by how much it lies, and the right patterns. The precise tool is device events: **[GPU, quoted from the original source]**

```python
import torch
start_event = torch.cuda.Event(enable_timing=True)
end_event   = torch.cuda.Event(enable_timing=True)
x = torch.ones(2**10,2**10, device="cuda")
start_event.record()
x@x.T
end_event.record()
torch.cuda.synchronize()
duration = start_event.elapsed_time(end_event) / 1000
print(f"It took {duration} seconds to run this")
```

```
It took 0.07701897430419923 seconds to run this
```

The code creates two timing events, records the first, runs a matmul of a tensor with its own transpose, records the second, waits for all queued work to finish, and asks for the time between the events (`elapsed_time` returns milliseconds, hence the `/1000`; it can be called any time after the sync). Note that this is a single call without warmup, so the 77 ms most likely includes one-time startup costs (such as cuBLAS initialization) on top of the matmul itself; warm up first when you time for real. The part that matters is the exact order:

```python
start_event.record()
<the code execution of which you want to time goes here>
end_event.record()
torch.cuda.synchronize()
```

Other devices have similar or identical APIs:

- `torch.cuda.Event` - NVIDIA CUDA and AMD ROCm
- `ht.hpu.Event(enable_timing)` - Intel Gaudi (`ht` being `import habana_frameworks.torch as ht`)
- `torch.xpu.Event(enable_timing)` - Intel dGPUs
- Apple MPS now has `torch.mps.Event(enable_timing=True)` (used in [timing.py](../training/profiling-and-optimization.md#why-the-naive-timer-lies)); older code emulated the API, as in [this version of mamf-finder.py](https://github.com/stas00/ml-engineering/blob/0099885db36a8f06556efe1faf552518852cb1e0/compute/accelerator/benchmarks/mamf-finder.py#L159).

Practical examples of timing with device events in this repository: [mamf-finder.py](../compute/accelerator/benchmarks/mamf-finder.py) and [torch-dist-bench.py](../network/benchmarks/torch-dist-bench.py) (explained in [torch-dist-bench](../network/measuring-and-debugging.md#torch-dist-bench)). For a deeper treatment, see [How to Accurately Time CUDA Kernels in Pytorch](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch).


## Profilers

General Python profilers (for example `cProfile` from the standard library; [Python Profilers](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md#profilers) is a good tour) tell you where a Python program spends its time. PyTorch has profilers designed for its APIs that also see GPUs, TPUs and other hardware, with small overhead and access to fine-grained events Python profilers can't see. Since PyTorch runs device work asynchronously, a Python profiler like `cProfile` doesn't know how long a kernel actually took; it reports what it can see happen on the CPU.

### `torch.profiler` in one shot

`torch.profiler` correctly measures operations that run asynchronously, where the Python dispatcher pushes a kernel launch into a queue and the device runs it when it gets to it. [Profiling and optimization](../training/profiling-and-optimization.md#torchprofiler-end-to-end) shows the scheduled form for a training loop; here is the one-shot form around a single call **[GPU, quoted from the original source, H100]**:

```python
# torch-profile-linear-example.py
import torch
from torch.profiler import profile, ProfilerActivity

linear = torch.nn.Linear(1024, 1024).to("cuda")
x = torch.randn(256, 512, 1024, device="cuda")

# warmup
out = linear(x)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CUDA], with_stack=True) as prof:
    out = linear(x)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=27))
```

It runs a Linear layer twice: once as warmup, which is ignored, and once under the profiler, then prints the report:

```bash
$ python torch-profile-linear-example.py
---------------------------  ----------  ---------  -----------  ---------  ------------  ---------  -----------  ----------  -------------  ----------
                       Name  Self CPU %   Self CPU  CPU total %  CPU total  CPU time avg  Self CUDA  Self CUDA %  CUDA total  CUDA time avg  # of Calls
---------------------------  ----------  ---------  -----------  ---------  ------------  ---------  -----------  ----------  -------------  ----------
sm90_xmma_gemm_f32f32_tf...       0.00%    0.000us        0.00%    0.000us       0.000us  687.139us       99.80%   687.139us      687.139us           1
            Memset (Device)       0.00%    0.000us        0.00%    0.000us       0.000us    1.344us        0.20%     1.344us        1.344us           1
    Activity Buffer Request      69.98%    2.193ms       69.98%    2.193ms       2.193ms    0.000us        0.00%     0.000us        0.000us           1
      cudaStreamIsCapturing       0.26%    8.203us        0.26%    8.203us       8.203us    0.000us        0.00%     0.000us        0.000us           1
                 cudaMalloc      10.75%  337.029us       10.75%  337.029us     337.029us    0.000us        0.00%     0.000us        0.000us           1
            cudaMemsetAsync       0.61%   19.135us        0.61%   19.135us      19.135us    0.000us        0.00%     0.000us        0.000us           1
      cudaFuncGetAttributes       0.28%    8.769us        0.28%    8.769us       4.385us    0.000us        0.00%     0.000us        0.000us           2
        cudaLaunchKernelExC       1.23%   38.672us        1.23%   38.672us      38.672us    0.000us        0.00%     0.000us        0.000us           1
      cudaDeviceSynchronize      16.88%  529.079us       16.88%  529.079us     264.540us    0.000us        0.00%     0.000us        0.000us           2
---------------------------  ----------  ---------  -----------  ---------  ------------  ---------  -----------  ----------  -------------  ----------
Self CPU time total: 3.134ms
Self CUDA time total: 688.483us
```

How to read this table: each row is one operator, kernel or CUDA runtime call. "Self" columns count time spent in that row itself; "total" columns include everything it called. CPU columns are host-side time; CUDA columns are time the GPU spent executing kernels (more in [Reading `key_averages()`](../training/profiling-and-optimization.md#reading-key_averages)). Rows with CUDA time are GPU kernels, the `cuda*` rows are CUDA runtime API calls made by the CPU, and `Activity Buffer Request` is the profiler's own setup.

The code spends almost all its GPU time in one GEMM, as a linear layer should: 687.139 us, 99.80% of the CUDA time. The other rows are the CUDA calls made while launching it. The kernel name `sm90_xmma_gemm_f32f32_tf...` is a cuBLAS kernel: `sm90` means it targets Hopper, `f32f32` gives the input and output dtypes, and the part cut off by the column width encodes the math mode, tile shape and layouts. Kernel names are long because they include dtypes and shapes; to see them in full, widen the column:

```diff
- print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=27))
+ print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=450))
```

The last two lines are the end-to-end totals: about 3.1 ms on the CPU and 0.7 ms on the GPU. Once you profile a whole model rather than one layer, you see many more kernels, and that's where it gets interesting.

The same example on CPU (`.to("cuda")` removed, a smaller batch `x = torch.randn(16, 512, 1024)`, `ProfilerActivity.CPU`, sorted by `cpu_time_total`) **[ran]**:

```
--------------------  ------------  ------------  ------------  ------------  ------------  ------------  
                Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls  
--------------------  ------------  ------------  ------------  ------------  ------------  ------------  
        aten::linear         0.12%      18.435us       100.00%      15.642ms      15.642ms             1  
         aten::addmm        86.36%      13.508ms        99.76%      15.604ms      15.604ms             1  
         aten::copy_        13.39%       2.094ms        13.39%       2.094ms       2.094ms             1  
             aten::t         0.03%       5.375us         0.05%       7.934us       7.934us             1  
aten::_unsafe_vie...         0.04%       6.000us         0.04%       6.000us       6.000us             1  
          aten::view         0.04%       5.708us         0.04%       5.708us       5.708us             1  
--------------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 15.642ms
```

This one shows self vs total clearly: `aten::linear` has 100% total but 0.12% self, because it just calls `aten::addmm` (the fused bias-add plus matmul), which does 86% of the work itself.

For more options see the [`torch.profiler` docs](https://docs.pytorch.org/docs/stable/profiler.html), and [an excellent introduction to `torch.profiler` from the Hugging Face team](https://huggingface.co/blog/torch-profiler).

### When `torch.profiler` isn't enough: a cProfile case study

`cProfile` is the wrong profiler for the GPU part of PyTorch code, but there are cases where it is exactly the right one: when the time goes somewhere that is *not* a PyTorch operator.

Stas Bekman was diagnosing a strange ~1 s overhead in `forward` and `backward` calls. `torch.profiler` measured the forward at about 100 ms, but the wall-clock timer said about 1 s, and `torch.profiler` gave no hint why. `cProfile` showed it immediately: a Triton kernel (Triton is a Python-embedded language for writing GPU kernels, compiled at run time) was being recompiled, taking about 1 s each time. The input was a padded 2D batch flattened into an unpadded 1D one, so the final length differed on many steps, and each new length triggered a recompilation of a kernel specialized to a specific input length.

Reproducing it needs a CUDA GPU, a recent PyTorch (see [PyTorch's install page](https://pytorch.org/get-started/locally/)) and [Liger Kernel](https://github.com/linkedin/Liger-Kernel), LinkedIn's collection of Triton kernels for LLM training (RMSNorm, RoPE, SwiGLU, fused cross-entropy and so on). Its `AutoLigerKernelForCausalLM` swaps them into a Hugging Face Transformers model in place of the stock PyTorch modules:

```bash
pip install liger-kernel
```

First, a context manager that switches between `cProfile`, `torch.profiler` and no profiler without changing the code being profiled:

```python
# profilers.py
from contextlib import nullcontext
from pstats import Stats
from torch.profiler import profile, record_function, ProfilerActivity
import cProfile
import pstats
import torch

# customize the precision of cProfile to give 6 decimals
pstats.f8 = lambda x: f"{x:3.6f}"


class ProfilerContext:
    """
    A proxy Profiler context manager class that can quickly choose between cProfile, torch.profiler and no-profiler w/o changing the end user code (other than changing the profiler type flag)

    Example:

    prof_fwd = ProfilerContext(type="c", name="some context")
    with prof_fwd():
        x = 1
    prof_fwd.report()
    """

    def __init__(self, type="none", name=None):
        """
        Args:
        - type: "torch": torch.profiler, "c": cProfile, "none": none
        - name: some context string for the reports
        """
        self.torch = False
        self.c = False
        if type == "torch":
            self.torch = True
        elif type == "c":
            self.c = True
        elif type == "none":
            pass
        else:
            raise ValueError(f"the `type` can be one of torch|c|none but got {type}")

        self.name = name if name is not None else "unknown"

        if self.torch:
            self.ctx = profile(activities=[ProfilerActivity.CUDA], record_shapes=False, with_stack=True)
        elif self.c:
            self.ctx = cProfile.Profile()
        else:
            self.ctx = nullcontext()

    def __call__(self):
        if self.torch and torch.cuda.is_available():
            torch.cuda.synchronize()

        return self.ctx

    def report(self):
        if self.torch:
            print(f"*** torch.profiler {self.name} ***")
            print(self.ctx.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        elif self.c:
            print(f"*** cProfile {self.name} ***")
            stats = Stats(self.ctx)
            stats.sort_stats("tottime").print_stats(20)
            # cumulative report is useful to understand where some of the large internal time overheads
            # come from - because it shows you the stack of calls leading to the slow call. So
            # `tottime` shows candidates to study and `cumulative` for finding context for those calls
            stats.sort_stats('cumulative').print_stats(50)
        else:
            pass # report nothing
```

(`cProfile.Profile` objects are context managers themselves since Python 3.8, which is why `with prof():` works for all three types.) Now profile 2 forward calls of Qwen3-0.6B with Liger's kernels swapped in **[GPU, quoted from the original source]**:

```python
# liger-kernel-varlen-recompile.py
import torch
import time
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from profilers import ProfilerContext

#PROFILER_TYPE = "c"
PROFILER_TYPE = "torch"
#PROFILER_TYPE = "none"

model = AutoLigerKernelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B").to("cuda")

batch_size = 5
seq_len = 512
vocab_size = model.config.vocab_size

input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=model.device)
attention_mask = torch.ones_like(input_ids)

for i in range(2):
    prof = ProfilerContext(type=PROFILER_TYPE, name=f"FWD {i}")
    start = time.perf_counter()
    with prof(): outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    prof.report()
    print(f"wallclock duration: {(time.perf_counter()-start)*1000:0.3f} msecs")
```

Running it shows the discrepancy (most of the profiler table trimmed):

```bash
$ python liger-kernel-varlen-recompile.py
*** torch.profiler FWD 0 ***
[...]
Self CPU time total: 119.385ms
Self CUDA time total: 25.041ms

wallclock duration: 1003.727 msecs
```

On the second step the gap is much smaller:

```
*** torch.profiler FWD 1 ***
[...]
Self CPU time total: 59.184ms
Self CUDA time total: 25.011ms

wallclock duration: 228.978 msecs
```

How to read it: on the first call, the profiler accounts for 119 ms of CPU and 25 ms of GPU time, but the wall clock says 1,004 ms. Something took over 800 ms that `torch.profiler` did not attribute to any PyTorch operator. So switch to `cProfile` by editing two lines:

```
PROFILER_TYPE = "c"
#PROFILER_TYPE = "torch"
```

and the first forward call shows the problem:

```
*** cProfile FWD 0 ***
         113616 function calls (109641 primitive calls) in 3.290 seconds

   Ordered by: internal time
   List reduced from 947 to 20 due to restriction <20>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
       28 1.991778 0.071135 1.991778 0.071135 {built-in method torch._C._nn.scaled_dot_product_attention}
       57 0.427335 0.007497 0.427335 0.007497 {built-in method torch.cat}
      398 0.268139 0.000674 0.268139 0.000674 {method 'update' of '_hashlib.HASH' objects}
      197 0.134879 0.000685 0.134879 0.000685 {built-in method torch._C._nn.linear}
       28 0.107747 0.003848 0.107747 0.003848 {built-in method torch.empty_like}
      475 0.091337 0.000192 0.091337 0.000192 {method 'read' of '_io.BufferedReader' objects}
        1 0.050481 0.050481 0.055220 0.055220 python3.12/site-packages/transformers/models/qwen3/modeling_qwen3.py:319(forward)
       80 0.035789 0.000447 0.036184 0.000452 {built-in method _io.open}
      115 0.025036 0.000218 0.025036 0.000218 {method '__exit__' of '_io._IOBase' objects}
    144/7 0.008807 0.000061 0.037330 0.005333 {built-in method builtins.exec}
        6 0.005818 0.000970 0.005818 0.000970 {built-in method _imp.create_dynamic}
        1 0.005175 0.005175 0.005175 0.005175 {method 'all' of 'torch._C.TensorBase' objects}
      376 0.004812 0.000013 0.004812 0.000013 {built-in method posix.stat}
```

**How to read a cProfile table.** `ncalls` is how many times the function was called (`144/7` means 144 calls in total, of which 7 were primitive, i.e. not recursive). `tottime` is the time spent inside the function itself, excluding the functions it called; `cumtime` includes them. Each `percall` is the preceding column divided by the number of calls. Entries in `{...}` are built-in functions implemented in C. The header says how many calls were recorded in total and how the list was cut down (`List reduced from 947 to 20`).

Also note that `cProfile` makes the program slower (3.29 s here vs 1.0 s unprofiled) and attributes asynchronous GPU work to whichever CPU call happened to wait for it, so the large `scaled_dot_product_attention` and `torch.cat` numbers are not to be trusted as GPU costs. What *is* trustworthy is the presence of calls that interface with the operating system: `read`, `_io.open`, `exec`, `posix.stat`, `_imp.create_dynamic` (loading a compiled extension module), and SHA hashing. Those have no business running during a fast `forward`/`backward`.

The wrapper prints the data twice. Sorting by `tottime` shows which functions are slow in themselves, but not where they were called from. The second dump, sorted by `cumtime`, shows the chain of callers:

```
         113616 function calls (109641 primitive calls) in 3.290 seconds

   Ordered by: cumulative time
   List reduced from 947 to 50 due to restriction <50>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
[...]]
      141 0.000894 0.000006 0.620426 0.004400 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/liger_kernel/ops/utils.py:33(wrapper)
      169 0.000906 0.000005 0.524757 0.003105 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/triton/runtime/jit.py:370(<lambda>)
      169 0.004586 0.000027 0.523851 0.003100 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/triton/runtime/jit.py:695(run)
      113 0.000454 0.000004 0.498234 0.004409 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/liger_kernel/transformers/rms_norm.py:37(forward)
      113 0.000670 0.000006 0.490840 0.004344 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/liger_kernel/ops/rms_norm.py:605(forward)
      113 0.002546 0.000023 0.490049 0.004337 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/liger_kernel/ops/rms_norm.py:409(rms_norm_forward)
        5 0.000087 0.000017 0.435603 0.087121 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/triton/runtime/jit.py:826(_do_compile)
        5 0.000169 0.000034 0.435442 0.087088 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/triton/compiler/compiler.py:226(compile)
       57 0.427335 0.007497 0.427335 0.007497 {built-in method torch.cat}
       28 0.000101 0.000004 0.426423 0.015229 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/transformers/cache_utils.py:742(update)
       28 0.000226 0.000008 0.426321 0.015226 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/transformers/cache_utils.py:98(update)
        5 0.000105 0.000021 0.368456 0.073691 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/triton/runtime/cache.py:307(get_cache_key)
        1 0.000693 0.000693 0.349386 0.349386 /home/yak/miniconda3/envs/dev/lib/python3.12/site-packages/triton/runtime/cache.py:271(triton_key)
      398 0.268139 0.000674 0.268139 0.000674 {method 'update' of '_hashlib.HASH' objects}
```

How to read it: follow `cumtime` down the chain. Liger's `rms_norm_forward` (0.49 s cumulative over 113 calls) goes through Triton's `jit.py run`, which calls `_do_compile` and `compiler.py compile` 5 times (0.44 s), and `cache.py get_cache_key`/`triton_key` (0.37 s), which is where the SHA hashing (`_hashlib.HASH update`, 0.27 s) comes from: Triton hashes its own source files to build a cache key. So the strange OS calls came from Liger Kernel's Triton RMSNorm kernel being compiled and cached.

A slower first call is normally fine: the first call often runs one-time work such as `torch.compile`. But older versions of liger-kernel had a bug that recompiled and re-cached the `RMSNorm` kernel for **every new sequence length**, which massively hurt end-to-end performance, twice over, since `forward` and `backward` are two different kernels. With `pip install liger-kernel==0.6.1` you see the problem whenever the sequence length changes from step to step; `liger-kernel>=0.8.0` fixes it.

Benchmarks normally need a warmup phase, with measurement starting after step 2 or later. Here the very first step was enough to demonstrate how `cProfile` helps when seemingly pure PyTorch code underperforms, but the same approach works from step 2 onwards, which is where a per-length recompilation would keep showing up.

### A laptop reproduction: recompiles under `torch.compile`

The same failure pattern is easy to reproduce without a GPU, because `torch.compile` also specializes on shapes. This compiles an RMSNorm with `dynamic=False`, so every new sequence length `T` triggers a recompile, and compares what `torch.profiler` and `cProfile` see **[ran]**:

```python
# recompile.py - a compiled function fed a new sequence length every step; cProfile vs torch.profiler
import cProfile, pstats, time, torch
from torch.profiler import profile, ProfilerActivity

pstats.f8 = lambda x: f"{x:3.6f}"          # 6 decimals instead of 3 in the cProfile table

@torch.compile(dynamic=False)               # dynamic=False: specialize on every new shape
def rmsnorm(x, w):                          # x: (B, T, C)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w

w = torch.ones(256)
rmsnorm(torch.randn(4, 100, 256), w)        # warmup: first compile

for step, T in enumerate([100, 117]):       # same T as warmup, then a new T
    x = torch.randn(4, T, 256)              # (B, T, C)
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        t0 = time.perf_counter(); rmsnorm(x, w); dt = time.perf_counter() - t0
    ops = sum(e.self_cpu_time_total for e in prof.key_averages() if e.key.startswith("aten::")) / 1e3
    print(f"step {step} T={T}: wall {dt*1e3:8.1f} ms, aten ops seen by torch.profiler {ops:6.2f} ms")

x = torch.randn(4, 131, 256)                # yet another T, this time under cProfile
pr = cProfile.Profile(); pr.enable(); rmsnorm(x, w); pr.disable()
st = pstats.Stats(pr)
st.sort_stats("tottime").print_stats(8)
st.sort_stats("cumulative").print_stats("inductor|subprocess", 8)  # only lines matching the regex, top 8
```

Run with Inductor's on-disk caches disabled, so every compile is a cold one (site-packages paths shortened):

```
$ TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python recompile.py
step 0 T=100: wall      0.1 ms, aten ops seen by torch.profiler   0.00 ms
step 1 T=117: wall    247.0 ms, aten ops seen by torch.profiler   1.07 ms
         241823 function calls (234055 primitive calls) in 0.289 seconds

   Ordered by: internal time
   List reduced from 4622 to 8 due to restriction <8>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1 0.195265 0.195265 0.195265 0.195265 {method 'read' of '_io.BufferedReader' objects}
        1 0.003949 0.003949 0.003956 0.003956 {built-in method torch._C._dynamo.guards.profile_guard_manager}
        1 0.003459 0.003459 0.003459 0.003459 {built-in method _posixsubprocess.fork_exec}
37425/37002 0.002670 0.000000 0.003150 0.000000 {built-in method builtins.isinstance}
        1 0.001606 0.001606 0.001606 0.001606 {built-in method posix.read}
12143/11970 0.001315 0.000000 0.001582 0.000000 {built-in method builtins.getattr}
        1 0.001227 0.001227 0.001229 0.001229 {built-in method gc.collect}
1243/1240 0.001204 0.000001 0.001925 0.000002 torch/utils/_config_module.py:571(__getattr__)


         241823 function calls (234055 primitive calls) in 0.289 seconds

   Ordered by: cumulative time
   List reduced from 4622 to 1461 due to restriction <'inductor|subprocess'>
   List reduced from 1461 to 8 due to restriction <8>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        1 0.000011 0.000011 0.266222 0.266222 torch/_inductor/compile_fx.py:3125(compile_fx)
        1 0.000043 0.000043 0.266206 0.266206 torch/_inductor/compile_fx.py:3291(_maybe_wrap_and_compile_fx_main)
        1 0.000021 0.000021 0.266069 0.266069 torch/_inductor/compile_fx.py:3341(_compile_fx_main)
        1 0.000025 0.000025 0.251472 0.251472 torch/_inductor/compile_fx.py:3389(fw_compiler_base)
        1 0.000019 0.000019 0.250315 0.250315 torch/_inductor/compile_fx.py:2831(compile_fx_forward)
        1 0.000024 0.000024 0.249971 0.249971 torch/_inductor/compile_fx.py:1032(compile_fx_inner)
        1 0.000059 0.000059 0.249360 0.249360 torch/_inductor/compile_fx.py:1113(_compile_fx_inner)
        1 0.000130 0.000130 0.249146 0.249146 torch/_inductor/compile_fx.py:2127(fx_codegen_and_compile)
```

How to read it:

- Step 0 reuses the shape from the warmup: 0.1 ms. Step 1 has a new `T`: 247 ms of wall clock, of which `torch.profiler`'s `aten::` operators account for about 1 ms. The same signature as the Liger case: the profiler is blind to the cost.
- The `tottime` table is topped by a single 195 ms `read` of a `BufferedReader` plus a `fork_exec`: the process forked a subprocess (the C++ compiler that Inductor, `torch.compile`'s backend, runs to build the CPU kernel) and waited to read its result.
- The `cumulative` table, filtered with a regex (`print_stats` accepts a regex and a count, applied in order), shows where that came from: Inductor's `compile_fx` with 266 ms cumulative. One recompilation per new shape.
- A second run *with* caches enabled took 18.8 ms at step 1, not 247 ms: the compiled kernel came from Inductor's on-disk cache, though Dynamo still re-traced the function. Caches hide this kind of bug on your machine and expose it on a fresh node; measure both ways.

Since the culprit in this case is `torch.compile`, you can also spot recompilations directly with `TORCH_LOGS=recompiles` (and graph breaks, where compiled code falls back to eager, with `TORCH_LOGS=graph_breaks`) **[ran]**:

```
$ TORCH_LOGS=recompiles python recompile.py
V1111 23:35:59.753000 11144 torch/_dynamo/guards.py:6374] [0/1] [__recompiles] Recompiling function rmsnorm in /private/tmp/absorb/run/debug2/recompile.py:7
V1111 23:35:59.753000 11144 torch/_dynamo/guards.py:6374] [0/1] [__recompiles]     triggered by the following guard failure(s):
V1111 23:35:59.753000 11144 torch/_dynamo/guards.py:6374] [0/1] [__recompiles]     - 0/0: tensor 'x' size mismatch at index 1. expected 100, actual 117
V1111 23:35:59.773000 11144 torch/_dynamo/guards.py:6374] [0/2] [__recompiles] Recompiling function rmsnorm in /private/tmp/absorb/run/debug2/recompile.py:7
V1111 23:35:59.773000 11144 torch/_dynamo/guards.py:6374] [0/2] [__recompiles]     triggered by the following guard failure(s):
V1111 23:35:59.773000 11144 torch/_dynamo/guards.py:6374] [0/2] [__recompiles]     - 0/1: tensor 'x' size mismatch at index 1. expected 117, actual 131
V1111 23:35:59.773000 11144 torch/_dynamo/guards.py:6374] [0/2] [__recompiles]     - 0/0: tensor 'x' size mismatch at index 1. expected 100, actual 131
```

How to read it: `[0/1]` is frame 0, compiled version 1 (the second). A *guard* is a check `torch.compile` inserts to decide whether a compiled version still applies; here the guard on `x.shape[1]` failed (expected 100, got 117), then again for 131, each time listing every existing version it tried. A recompile line on every training step is the bug. The fixes are to let shapes be dynamic (`dynamic=True` or `torch._dynamo.mark_dynamic`), or to pad or bucket lengths to a few fixed values. See [`torch.compile`](../training/profiling-and-optimization.md#torchcompile) and [PyTorch compiler troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html). This logging only covers `torch.compile`; a kernel library with its own JIT, like the Triton kernels in old Liger versions, recompiles silently, and `cProfile` is how you catch it.


## You can now read

- [Testing](testing.md) - the next chapter: how to keep these bugs from coming back, including tests that need several GPUs or processes.
- [Fault tolerance and reproducibility](../training/fault-tolerance-and-reproducibility.md) - turning the hangs and crashes from this chapter into automatic restarts, and emulating multi-node setups on one node to reproduce them.
- [Profiling and optimization](../training/profiling-and-optimization.md) - if you came here for the performance tools: scheduled profiling, traces, Nsight Systems and the optimization ladder.
- [Measuring and debugging the network](../network/measuring-and-debugging.md) - if the hang turned out to be below PyTorch, in NCCL or the fabric.

Parts of this chapter are adapted from Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (CC BY-SA 4.0).

## Further reading

- [ProcessGroupNCCL environment variables](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html) - the desync, flight recorder and watchdog controls.
- [`py-spy`](https://github.com/benfred/py-spy), and its note on [when you need sudo](https://github.com/benfred/py-spy/blob/master/README.md#when-do-you-need-to-run-as-sudo).
- [PEP 768](https://peps.python.org/pep-0768/) and the [Python 3.14 `pdb` docs](https://docs.python.org/3.14/library/pdb.html) - attaching a debugger to a running process.
- Python's [`trace` module](https://docs.python.org/3/library/trace.html); `man core` for core-file name templates.
- [NCCL troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html) and the [NCCL Issues](https://github.com/NVIDIA/nccl/issues).
- [hyperfine](https://github.com/sharkdp/hyperfine).
- [`torch.profiler` docs](https://docs.pytorch.org/docs/stable/profiler.html) and the [Hugging Face introduction to `torch.profiler`](https://huggingface.co/blog/torch-profiler).
- [How to Accurately Time CUDA Kernels in Pytorch](https://www.speechmatics.com/company/articles-and-news/timing-operations-in-pytorch).
- [Python Profilers](https://github.com/stas00/the-art-of-debugging/blob/master/python/README.md#profilers) in Stas Bekman's *The Art of Debugging*.
- [PyTorch compiler troubleshooting](https://docs.pytorch.org/docs/stable/torch.compiler_troubleshooting.html) - `TORCH_LOGS`, recompiles and graph breaks.
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel).
