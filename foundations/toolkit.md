# Foundations: The Working Toolkit

Every other chapter in this book quietly assumes a pile of small skills: typing a command into a terminal, making a Python environment that has `torch` in it, cloning a repository with git, reading a YAML file without being bitten by it, grepping a log with a regular expression, typing `make`, and reading a short C program with pointers in it. None of these is deep. All of them are things nobody teaches you if you came to ML through Python notebooks, and each one can cost an afternoon the first time.

This chapter teaches exactly that pile, from zero, and nothing more. It is deliberately shallow where a later chapter goes deep: the shell gets a proper treatment in [Linux and the OS](linux-and-os.md#2-the-shell), C++ for GPUs in [Writing kernels](../compute/accelerator/writing-kernels.md#just-enough-c-to-read-cuda), and Kubernetes-flavoured YAML in [Kubernetes from zero](../orchestration/kubernetes-from-zero.md#yaml-for-kubernetes).

Conventions:

- Every program and every command shown with output was run while writing, on an Apple M4 MacBook (macOS 15, arm64, i.e. an ARM-family CPU) with Python 3.12-3.14, PyTorch 2.15, Apple clang 17, GNU Make 3.81, jq 1.7 and git 2.52; the output blocks are pasted from those runs. Paths, timestamps, hashes and memory addresses will differ on your machine. A few commands shown *without* output are syntax illustrations that were not run (installers like `brew`/`apt`/`conda`, `git config`, `git switch --detach`, `CUDA_LAUNCH_BLOCKING=1`); they're standard usage, but untested here.
- Nothing here needs a GPU.

## Prerequisites

- Python (well), PyTorch (tensors, autograd, `nn.Module`, training loops) and deep-learning math. That's all. This is the first chapter; it assumes no shell, no git, no C.

Forward links: the shell in depth (pipes, quoting, exit codes, scripts, ssh, tmux) is [Linux and the OS, section 2](linux-and-os.md#2-the-shell); what a process, a file and memory actually are is the rest of that chapter; the C programs here get used for real in [Computer architecture](computer-architecture.md).



## The terminal, just enough

### What a terminal is

A **terminal** is a window where you type commands as text and read their output as text. The program inside it that reads your command, runs it and prints the result is the **shell**. On macOS the app is called Terminal (in Applications → Utilities; many people use iTerm2 or the terminal built into VS Code instead), and the default shell is `zsh`. On Linux it's usually called Terminal too, and the default shell is usually `bash`. Every remote GPU machine you will ever log into gives you a shell, and nothing else, so this is not optional.

When the shell is ready for input it prints a **prompt**, something like:

```
you@laptop project %
```

That is your user name, the machine name, the current directory and a `%` (zsh) or `$` (bash). Prompts are configurable and differ between machines, so this book doesn't print them. Instead it uses one convention:

```console
$ python3 --version
Python 3.14.5
```

**A line starting with `$ ` is a command you type** (without the `$ `). Lines that don't start with `$ ` are what the command printed. Sometimes a code block has no `$ ` lines at all; then every line in it is a command (or a file's contents, if a comment like `# train.py` names a file at the top). A `#` in a command line starts a comment that the shell ignores, so `$ ls -l   # long listing` runs `ls -l`. A word in angle brackets, like `<file>` or `<hash>`, is a placeholder: type your own value instead, without the brackets.

A command finishes when the prompt comes back. If it doesn't come back, the command is still running (or waiting for you to type input). **Ctrl-C** interrupts it. **Ctrl-D** on an empty line means "end of input", which also exits a Python REPL or a shell. Up-arrow recalls the previous command; Tab completes file names (type `cd Desk` then Tab).

### Anatomy of a command

```console
$ ls -l data
```

The first word (`ls`) is the program to run. The rest are **arguments**, separated by spaces. Arguments starting with `-` are **options** (also called flags) that change behaviour: `-l` means "long format". Single-letter options can be combined (`ls -la` = `ls -l -a`); long options use two dashes (`--version`). Some options take a value: `head -n 5 file` or `--epochs=3`. Every program defines its own options; `program --help` or `man program` (manual page; `q` to quit) lists them.

Arguments containing spaces must be quoted: `ls "My Documents"`. Without quotes, the shell splits it into two arguments, `My` and `Documents`. Quoting has more rules than this, and they cause real bugs; [Quoting](linux-and-os.md#quoting-the-source-of-many-bugs) has them.

### Moving around: paths, `pwd`, `cd`, `ls`, `cat`

Files live in a tree of directories (folders). A **path** names a place in the tree:

- An **absolute path** starts at the root `/`: `/tmp/tkdemo/project/notes.txt`.
- A **relative path** starts at the **current working directory**, the directory the shell is "in" right now: if you are in `/tmp/tkdemo`, then `project/notes.txt` means `/tmp/tkdemo/project/notes.txt`.
- `.` is the current directory, `..` is its parent, `~` is your home directory (`/Users/you` on macOS, `/home/you` on Linux).

The handful of commands you need on day one, run in an empty directory:

```console
$ pwd                                  # print working directory
/tmp/tkdemo
$ mkdir -p project/data                # make directories (-p: parents too, no error if they exist)
$ cd project                           # change directory
$ echo "hello from a file" > notes.txt # write a line into a file (> redirects output to a file)
$ ls                                   # list
data
notes.txt
$ ls -la                               # long format, including hidden files (names starting with .)
total 8
drwxr-xr-x  4 you            wheel  128 Nov  6 19:03 .
drwxr-xr-x  3 you            wheel   96 Nov  6 19:03 ..
drwxr-xr-x  2 you            wheel   64 Nov  6 19:03 data
-rw-r--r--  1 you            wheel   18 Nov  6 19:03 notes.txt
$ cat notes.txt                        # print a file
hello from a file
$ cd data; pwd                         # ; separates two commands on one line
/tmp/tkdemo/project/data
$ cd ..; pwd
/tmp/tkdemo/project
$ cd ~; pwd
/Users/you
```

In `ls -la` output, the first character is `d` for a directory and `-` for a file; the `rwx` letters are permissions (explained in [Permissions](linux-and-os.md#permissions-ownership-umask)); then the owner, group, size in bytes, modification time and name. The hidden entries `.` and `..` are the directory itself and its parent.

Other commands you will meet in the first few chapters, each worth trying once:

| Command | Does |
|---|---|
| `less file` | page through a long file (Space/b to move, `/word` to search, `q` to quit) |
| `head -n 20 file`, `tail -n 20 file` | first/last 20 lines; `tail -f file` keeps printing as the file grows (watching a training log) |
| `cp a b`, `mv a b`, `rm a` | copy, move/rename, delete. `rm` has no undo and no trash can; `rm -r dir` deletes a directory tree |
| `wc -l file` | count lines |
| `grep pattern file` | print lines containing `pattern` (much more in [Regular expressions](#regular-expressions)) |
| `which python3` | which program a command name runs (the full path) |
| `echo $HOME` | print a variable's value |
| `nano file` | a simple terminal text editor (Ctrl-O to save, Ctrl-X to exit). Or open files in your normal editor |

### Exit codes, pipes, redirection, variables: a first look

Four more ideas appear in nearly every shell snippet in the book. [Linux and the OS](linux-and-os.md#2-the-shell) teaches each properly; here is enough to read them.

Every command ends with an **exit code**: 0 means success, anything else means failure. The shell keeps the last one in `$?`:

```console
$ ls nosuchfile
ls: nosuchfile: No such file or directory
$ echo "exit code: $?"
exit code: 1
```

`a | b` (a **pipe**) feeds the output of `a` into `b` as input. `> file` writes output to a file (replacing it), `>> file` appends, `< file` feeds a file to the command as its input, `2>&1` sends error messages to the same place as normal output. `a && b` runs `b` only if `a` succeeded. A trailing `&` runs a command in the background so you get the prompt back:

```console
$ ls -l /usr/bin | wc -l                       # how many lines does the listing of /usr/bin have
     916
```

and the pattern you will use for every long-running job (here `train.py` stands for any script):

```console
$ python3 train.py > train.log 2>&1 &          # run in background, all output to a file
$ tail -f train.log                            # watch it; Ctrl-C stops watching, not the job
```

**Environment variables** are named strings that every program inherits from the shell that started it. `export NAME=value` sets one for this shell and everything it starts; `NAME=value command` sets it for one command only. ML tooling is configured this way all the time (`CUDA_VISIBLE_DEVICES=0`, `OMP_NUM_THREADS=1`, `HF_HOME=...`), and Python reads them with `os.environ` (`python3 -c '...'` runs the quoted text as a one-line program; [Running a script](#running-a-script) shows it again):

```console
$ export SEED=42
$ python3 -c 'import os; print(os.environ["SEED"])'
42
$ SEED=7 python3 -c 'import os; print(os.environ["SEED"])'
7
```

`PATH` is the variable that lists the directories the shell searches for programs; that is how `python3` becomes `/opt/homebrew/bin/python3`. When "command not found" happens, the program is not installed or its directory is not on `PATH` ([Environment variables and PATH](linux-and-os.md#environment-variables-and-path)).

### How to read this book's labels

The chapters are written to be run, and they're honest about which parts were. You will see:

- **Pasted output.** An output block after a `$ ` command is what the command printed when the author ran it, on the machine named at the top of that chapter. Your numbers (timings, addresses, hashes) will differ; the shape should not.
- **[torch]**: needs a Python environment with PyTorch (the [setup block](#one-setup-block-for-the-whole-book) below). Runs on a CPU-only laptop.
- **[GPU]**: needs an NVIDIA GPU. Most of these were not run while writing; they say so.
- **Untested**: the command or code was *not executed* while writing, usually because it needs Linux, a GPU, a cluster or root (administrator rights on the machine; `sudo cmd` runs one command as root). It follows the tool's documentation, but treat it as a strong hint, not a recipe. Check flags against `--help` on your system.
- **Sketched output**: the output block shows the *format* of what you will see (real column names, real structure), but the values were written by hand, not captured. Don't draw conclusions from the numbers in a sketched block.

When a chapter says "one run on one laptop", it means exactly that: one data point, an anecdote, not a benchmark.



## Running the book's code

### Getting Python

You already write Python, but it's worth knowing which Python a terminal runs, because a machine usually has several:

```console
$ which python3
/opt/homebrew/bin/python3
$ python3 --version
Python 3.14.5
```

Ways to get one:

- **macOS:** Apple ships an old `/usr/bin/python3` (3.9 on this machine) with the developer tools; don't use it for real work. Install a current one with [Homebrew](https://brew.sh) (the usual macOS package manager: `brew install X` downloads and installs program X; `brew install python`), from [python.org](https://www.python.org/downloads/), or let `uv` manage it (next sections).
- **Linux:** the distribution's `python3` (Ubuntu 24.04 ships 3.12) plus `sudo apt install python3-venv` is fine (`apt` is Ubuntu's package manager, and installing system packages needs `sudo`, i.e. root). Or `uv`.
- **Any OS:** `uv python install 3.12` downloads a standalone Python into your home directory without touching the system.

Use a version that PyTorch publishes **wheels** (prebuilt packages) for; the install page on pytorch.org lists them. 3.12 is a safe choice (this chapter also installed `torch` 2.15 on 3.14). A brand-new Python release often has no PyTorch build for its first few months.

### Why virtual environments

`pip install` puts packages into a `site-packages` directory belonging to one Python installation. If that is the system Python, you are modifying something the OS or Homebrew manages, and two projects needing different `torch` versions cannot coexist. Recent Pythons from Homebrew and Debian/Ubuntu therefore refuse:

```console
$ python3 -m pip install numpy
error: externally-managed-environment

× This environment is externally managed
╰─> To install Python packages system-wide, try brew install
    xyz, where xyz is the package you are trying to
    install.

    If you wish to install a Python library that isn't in Homebrew,
    use a virtual environment:

    python3 -m venv path/to/venv
    source path/to/venv/bin/activate
```

Don't fight it (the `--break-system-packages` flag exists and is a bad idea). A **virtual environment** ("venv") is a directory containing a link to a Python interpreter plus its own private, empty `site-packages`. You make one per project, install into it, and delete the directory when you're done. Nothing else on the machine changes.

### venv: the built-in way

```console
$ cd /tmp/tkdemo/project
$ python3 -m venv .venv                 # create the environment in the directory .venv
$ source .venv/bin/activate             # activate it in this shell
(.venv) $ which python
/private/tmp/tkdemo/project/.venv/bin/python
(.venv) $ pip install numpy
...
(.venv) $ python -c "import numpy; print(numpy.__version__, numpy.__file__)"
2.5.4 /private/tmp/tkdemo/project/.venv/lib/python3.14/site-packages/numpy/__init__.py
(.venv) $ pip list
Package Version
------- -------
numpy   2.5.4
pip     26.1.1
(.venv) $ deactivate
```

(`/tmp` on macOS is a link to `/private/tmp`, hence the path.)

What `activate` does is small: it puts `.venv/bin` at the front of `PATH`, so `python` and `pip` now mean the venv's copies, and it adds `(.venv)` to the prompt. It only affects the current terminal window; a new window starts un-activated. You can skip activation entirely by naming the interpreter: `.venv/bin/python train.py` always uses the venv. That's the most robust form for scripts and for anything you put in a job file, and it's what most of this book's setup blocks do.

Two habits that prevent most environment confusion:

- `python -m pip install ...` instead of `pip install ...`. It guarantees the `pip` belongs to the `python` you think it does.
- When an import fails or picks up the wrong version, print `import sys; print(sys.executable)` and `module.__file__`. That tells you which interpreter and which copy of the package are actually in use.

### uv: the fast way

[uv](https://docs.astral.sh/uv/) is a single-binary replacement for `pip`, `venv` and Python installers. It does the same things much faster (its own benchmarks claim 10-100x over pip), which matters when `torch` is a multi-gigabyte install. Install it once (`curl -LsSf https://astral.sh/uv/install.sh | sh`, or `brew install uv`; `curl URL` downloads a URL and prints it, and piping that into `sh` runs the downloaded install script), then either use it as a drop-in for the commands above:

```console
$ uv venv .venv                          # like python -m venv, but fast; --python 3.12 picks a version
$ uv pip install numpy                   # installs into ./.venv without activating
$ .venv/bin/python -c "import numpy"
```

or let it manage a whole project:

```console
$ uv init uvdemo
Initialized project `uvdemo` at `/private/tmp/tkdemo/uvdemo`
$ cd uvdemo; ls -a
.
..
.git
.gitignore
.python-version
README.md
main.py
pyproject.toml
$ uv add numpy pyyaml
Using CPython 3.13.12
Creating virtual environment at: .venv
Resolved 3 packages in 2.45s
Installed 2 packages in 89ms
 + numpy==2.5.4
 + pyyaml==6.0.3
$ cat pyproject.toml
[project]
name = "uvdemo"
version = "0.1.0"
description = "Add your description here"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "numpy>=2.5.4",
    "pyyaml>=6.0.3",
]
$ uv run main.py
Hello from uvdemo!
$ uv run python -c "import numpy, yaml; print(numpy.__version__, yaml.__version__)"
2.5.4 6.0.3
```

`uv add` records the dependency in `pyproject.toml` (a [TOML](#toml) file), pins exact versions in a `uv.lock` file, and installs into `.venv`. `uv run X` makes sure `.venv` matches the lock file and then runs `X` inside it, no activation needed. Many recent ML repositories (including some RL frameworks in [RL infrastructure](../platform/rl-infrastructure.md)) are launched as `uv run ...` for this reason.

**conda** is the third option you'll see, especially on clusters and in older repos: `conda create -n myenv python=3.12`, `conda activate myenv`. It manages non-Python libraries too (CUDA runtime, compilers), which was its big advantage before PyTorch wheels bundled their own CUDA libraries. If a repo's README says `conda env create -f environment.yml`, that's a YAML list of packages. Mixing `conda install` and `pip install` in one environment works but is a common source of confusion; prefer one.

### requirements files and version pins

A `requirements.txt` is a plain list of packages, one per line, optionally pinned:

```
torch==2.5.1
numpy>=2.0
pyyaml
```

`pip install -r requirements.txt` (or `uv pip install -r requirements.txt`) installs all of them. `==` pins an exact version, `>=` a minimum. `pip freeze > requirements.txt` writes out exactly what's installed now, which is how you make an environment reproducible (the book's [Reproducibility](../training/fault-tolerance-and-reproducibility.md#reproducibility) chapter covers why).

### One setup block for the whole book

Almost every runnable example in the book needs some subset of `torch`, `numpy`, `pyyaml` (the `yaml` module) and `safetensors`. A few chapters need extras (FastAPI and friends in the serving and platform chapters, `triton` and `ninja` in the kernel chapter); each of those lists its own `pip install` line. Make one environment for the book and keep it:

```console
$ mkdir -p ~/mle && cd ~/mle
$ uv venv .venv --python 3.12
$ uv pip install torch numpy pyyaml safetensors
$ .venv/bin/python check_env.py
```

(Without uv: `python3 -m venv .venv && .venv/bin/python -m pip install torch numpy pyyaml safetensors`.)

where `check_env.py` is:

```python
# check_env.py - prints what this book's examples will find
import sys, platform
print("python  ", sys.version.split()[0], "at", sys.executable)
print("machine ", platform.system(), platform.machine())
import numpy, yaml, safetensors, torch
print("numpy   ", numpy.__version__)
print("pyyaml  ", yaml.__version__)
print("safetens", safetensors.__version__)
print("torch   ", torch.__version__)
print("cuda?   ", torch.cuda.is_available())
print("mps?    ", torch.backends.mps.is_available())
x = torch.randn(1024, 1024)
print("matmul ok, trace(x x^T)/1024 ≈ 1024:", round((x @ x.T).trace().item() / 1024))
```

On this Mac (the environment was in `/tmp/tk`):

```
python   3.12.13 at /private/tmp/tk/.venv/bin/python
machine  Darwin arm64
numpy    2.5.4
pyyaml   6.0.3
safetens 0.8.0
torch    2.15.0
cuda?    False
mps?     True
matmul ok, trace(x x^T)/1024 ≈ 1024: 1025
```

`cuda? False` is expected on a Mac: Apple GPUs don't run CUDA. `mps? True` means PyTorch can use the Apple GPU through its MPS backend, which a few chapters use when available. On a Linux machine with an NVIDIA GPU you want `cuda? True`. If it says `False` there, you probably installed a CPU-only build of `torch` or the NVIDIA driver (the low-level software, installed system-wide, that lets programs use the GPU) is missing; [The GPU software stack](../compute/gpu-software-stack.md) explains the layers and how to check each one.

The plain `pip install torch` picks a sensible default build for your platform. When a chapter needs a specific CUDA build it shows the command, which looks like `uv pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124`: the `--index-url` points pip at PyTorch's own package server, and `cu124` means "built against CUDA 12.4".

### Running a script

You know how to run Python, but there are a few forms you'll see in the book:

```console
$ python train.py --lr 3e-4 --steps 100     # a script with arguments (read with argparse / sys.argv)
$ python -c 'print(1 + 1)'                  # a one-line program
$ python -m http.server 8000                # run a module as a script (this one serves the current directory at http://localhost:8000)
$ python                                    # the interactive REPL; Ctrl-D to leave
```

Many chapters give a program as a code block whose first line is a comment with a file name, like `# race.py`. Save the block into that file (in your editor, or with the shell's **heredoc**, which feeds the lines up to `EOF` into a file) and run it:

```console
$ cat > hello.py <<'EOF'
import sys
print("argv:", sys.argv)
EOF
$ python hello.py a b
argv: ['hello.py', 'a', 'b']
```

The quotes around `'EOF'` stop the shell from interpreting `$` and backticks inside the block, which you want when the block is code.

### macOS vs Linux

The book is written on a Mac and deployed on Linux. The differences that bite:

- **The command-line tools differ slightly.** The classic Unix tools exist in two families, BSD and GNU, with the same names and mostly the same flags. macOS ships BSD versions of `sed` (a stream editor; [Linux and the OS](linux-and-os.md#just-enough-awk-and-sed) shows its few useful forms), `grep`, `date`, `stat`, `find`; Linux has the GNU versions. Most flags are the same; some aren't (`sed -i` needs `sed -i ''` on macOS; `grep -P` doesn't exist in macOS grep). When a chapter's command fails on a Mac with "illegal option", this is usually why. `brew install coreutils gnu-sed grep` gets GNU versions as `gsed`, `ggrep` and so on.
- **Linux-only interfaces.** `/proc`, `/sys`, cgroups, `numactl`, `perf`, `systemd`, namespaces: these exist only on Linux, which is why parts of [Linux and the OS](linux-and-os.md) and [Containers](containers-and-virtualization.md) are marked untested or run in a Linux VM (virtual machine: a whole Linux computer emulated inside your Mac). A free option on a Mac is a small Linux VM (UTM, Lima, OrbStack) or a cheap cloud CPU instance.
- **CUDA is NVIDIA-on-Linux (or Windows).** A Mac can't run CUDA code at all. Anything marked **[GPU]** needs a rented machine.
- **Paths:** home is `/Users/you` on macOS and `/home/you` on Linux; `/tmp` on macOS is `/private/tmp`.

### Which parts need a GPU, and renting one briefly

You can read the whole book, and run most of its examples, on a laptop. The ones that need an NVIDIA GPU are concentrated in a few places: the CUDA kernels in [Writing kernels](../compute/accelerator/writing-kernels.md), the driver and NCCL (NVIDIA's library for moving data between GPUs) checks in [The GPU software stack](../compute/gpu-software-stack.md) and [Collectives internals](../network/collectives-internals.md), and the vLLM/SGLang (open-source LLM serving engines) runs in [Serving hands-on](../inference/serving-hands-on.md). Multi-node examples (Slurm and Kubernetes, which schedule work across many machines, and InfiniBand, a fast network between them) need a cluster and are mostly labelled untested.

For the single-GPU material, rent a cloud GPU by the hour. The pattern is the same everywhere: create an account, add a payment method and an SSH public key ([ssh](linux-and-os.md#working-remotely-ssh-scp-rsync) explains keys), pick a GPU type and a machine image with a recent NVIDIA driver and PyTorch preinstalled, start it, `ssh` in (`ssh user@host` gives you a shell on the remote machine), work, and **stop or delete it when you're done**. A forgotten instance bills all night. Hyperscalers (AWS, Google Cloud, Azure) and smaller GPU clouds both offer this; the smaller ones are usually simpler and cheaper for an hour of tinkering. This book doesn't recommend a provider.

Rough on-demand prices at the time of writing (late 2026), for orientation only - they vary by provider and region, spot/preemptible capacity is cheaper, and prices fall over time:

| GPU | Good for in this book | Approx. \$/GPU-hour |
|---|---|---|
| a consumer-class or L4/A10-class card (16-24 GB) | kernels, profiling, small serving runs | \$0.3-1 |
| A100 40/80 GB | everything single-GPU, larger serving runs | \$1-2 |
| H100 80 GB | Hopper-specific kernels (TMA, FP8: features new in this "Hopper" GPU generation), realistic serving numbers | \$2-4 |

A focused afternoon on the [GPU]-labelled examples of one chapter is a few dollars. Multi-GPU nodes (8x H100) cost 8x as much per hour, and you need them only for the collective-communication and multi-GPU serving parts.



## git from zero

**git** is a version-control system: it records snapshots of a directory of files over time, so you can see what changed, when, by whom and why, go back to any earlier state, and work on several variations in parallel. Nearly all code you'll touch in ML (PyTorch, vLLM, Megatron, NCCL, every paper's reference implementation) lives in git repositories, usually hosted on GitHub. For experiments, git is also the cheapest reproducibility tool there is: a run that logs the **commit hash** of the code it ran can be reconstructed exactly.

The vocabulary, in one go:

- A **repository** ("repo") is a directory whose history git tracks. The history lives in a hidden `.git/` subdirectory; delete that and it's just a folder again.
- A **commit** is one saved snapshot of the whole tree, plus a message, an author, a time, and a pointer to its parent commit(s). Each commit is named by a 40-hex-digit **hash** such as `fbf9f0ce74b0...`; the first 7 characters are usually enough to refer to it.
- The **working tree** is the files you see and edit. The **staging area** (or "index") is what will go into the next commit; `git add` puts changes there.
- A **branch** is a movable name pointing at a commit (`main`, `try-cosine`). Committing on a branch moves it forward. **`HEAD`** means "the commit you currently have checked out".
- A **remote** is another copy of the repo, usually on GitHub, conventionally named `origin`. `clone`, `pull` and `push` move commits between your copy and a remote.

### A first repository

Set your name and email once per machine (they're stamped on every commit): `git config --global user.name "Your Name"` and `git config --global user.email you@example.com`. Then, in an empty directory, create two small files (with your editor, or the heredoc form from [Running a script](#running-a-script)):

```python
# config.py
lr = 3e-4
steps = 100
```

```python
# train.py
from config import lr, steps
print(f"training for {steps} steps at lr={lr}")
```

```console
$ git init -b main                       # make this directory a repo, first branch called main
Initialized empty Git repository in /private/tmp/tkdemo/gitdemo/.git/
$ git status
On branch main

No commits yet

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	config.py
	train.py

nothing added to commit but untracked files present (use "git add" to track)
$ git add config.py train.py             # stage both files
$ git commit -m "First version of the training script"
[main (root-commit) 10c873f] First version of the training script
 2 files changed, 4 insertions(+)
 create mode 100644 config.py
 create mode 100644 train.py
```

`git status` is the command to run whenever you're unsure what state things are in; it also tells you what to type next. (In the commit output, `10c873f` is the new commit's short hash, and `create mode 100644` just records each new file's permission bits, here "an ordinary, non-executable file"; [Permissions](linux-and-os.md#permissions-ownership-umask) explains the numbers.) Now edit `config.py` (change `steps = 100` to `steps = 500`) and look:

```console
$ git status
On branch main
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
	modified:   config.py

no changes added to commit (use "git add" and/or "git commit -a")
$ git diff
diff --git a/config.py b/config.py
index db30fbc..ea030a2 100644
--- a/config.py
+++ b/config.py
@@ -1,2 +1,2 @@
 lr = 3e-4
-steps = 100
+steps = 500
```

That is a **diff** (also called a patch), and you'll read thousands of them in code review. `---` and `+++` name the old and new file; `@@ -1,2 +1,2 @@` says "this hunk covers 2 lines starting at line 1 in both versions"; lines starting with `-` were removed, `+` were added, a space means unchanged context. Commit it and look at history:

```console
$ git add config.py
$ git commit -m "Train for 500 steps"
$ git log
commit fbf9f0ce74b05e716a9459affde7210dc26739d3
Author: Ada Lovelace <ada@example.com>
Date:   Fri Nov 6 10:00:00 2026 -0500

    Train for 500 steps

commit 10c873f2285113df9a0b800fe41c7ac1dc523224
Author: Ada Lovelace <ada@example.com>
Date:   Fri Nov 6 10:00:00 2026 -0500

    First version of the training script
$ git log --oneline
fbf9f0c Train for 500 steps
10c873f First version of the training script
```

(Your hashes will differ: they depend on content, author and time.) `git commit -am "msg"` stages every modified tracked file and commits in one step; it does not pick up new files. `git show <hash>` prints one commit with its diff. `git restore file` throws away uncommitted edits to `file` (there's no undo for that).

### Branches

A branch lets you try something without disturbing the working version:

```console
$ git switch -c try-cosine               # create a branch and switch to it
Switched to a new branch 'try-cosine'
$ echo 'schedule = "cosine"' >> config.py
$ git commit -am "Add cosine schedule"
[try-cosine 57aea93] Add cosine schedule
 1 file changed, 1 insertion(+)
$ git switch main                        # back to main: config.py has no schedule line
$ cat config.py
lr = 3e-4
steps = 500
$ git merge try-cosine                   # bring the branch's commits into main
Updating fbf9f0c..57aea93
Fast-forward
 config.py | 1 +
 1 file changed, 1 insertion(+)
$ git log --oneline
57aea93 Add cosine schedule
fbf9f0c Train for 500 steps
10c873f First version of the training script
$ git branch                             # list branches; * marks the current one
* main
  try-cosine
```

"Fast-forward" means main had no new commits of its own, so git just moved the `main` label forward. If both branches had changed the same lines, `git merge` would stop with a **merge conflict** and mark the file with `<<<<<<<`, `=======`, `>>>>>>>` sections; you edit the file to the version you want, `git add` it, and `git commit`. (Older docs use `git checkout -b name` and `git checkout name` for what `git switch` does.)

### Files git should ignore

A file called `.gitignore` lists patterns git should never track: virtual environments, logs, checkpoints, data.

```
.venv/
__pycache__/
*.log
checkpoints/
*.safetensors
```

Never commit model weights, datasets or secrets (API keys, tokens) to git. Big binary files make every clone slow forever, and a committed secret stays in history even after you delete the file.

### Working with other people's repositories

```console
$ git clone https://github.com/karpathy/micrograd.git
Cloning into 'micrograd'...
$ cd micrograd
$ git remote -v                          # where it came from
origin	https://github.com/karpathy/micrograd.git (fetch)
origin	https://github.com/karpathy/micrograd.git (push)
$ git pull                               # fetch new commits from origin and merge them in
Already up to date.
$ git log --oneline -3
7bc720e link to microgpt
c911406 add setup.py to allow registering micrograd as package under pypi
c34adac license should not be markdown, fixing
```

`git clone URL` copies the whole repository including history. `git pull` updates it later. `git push` sends your commits to the remote; you need write access, which on someone else's project you don't have, so contributions go through a **fork** (your own copy on GitHub) and a **pull request** (a request that they merge your branch). `git clone --depth 1 URL` fetches only the latest snapshot, which is much faster for huge repos like PyTorch when you only want to read the code.

To get the exact version of a project that matches what you have installed, check out a **tag** (a permanent name for a release commit): `git tag` lists them, `git switch --detach v2.5.1` checks one out. Reading the `main` branch of a library while running last year's release is a classic way to be confused by code that doesn't match your behaviour.

### git grep, log -- path, blame

Three commands answer most "where is this / why is this" questions in a codebase:

```console
$ git grep -n "def backward"             # search tracked files; -n prints line numbers
micrograd/engine.py:54:    def backward(self):
$ git log --format='%h %ad %s' --date=short -- micrograd/engine.py | head -3   # history of one file
5bb6392 2020-04-18 small tweaks and bug fixes to docs
315a2cb 2020-04-17 add a number of more ops including -,/,**, and supporting unit tests
9d268f5 2020-04-14 add graphviz gracing utility which I think is cute
$ git blame -L 54,58 micrograd/engine.py # which commit last touched each line
9fd9cc02 (Andrej Karpathy 2020-04-14 00:25:30 -0700 54)     def backward(self):
9fd9cc02 (Andrej Karpathy 2020-04-14 00:25:30 -0700 55)
9fd9cc02 (Andrej Karpathy 2020-04-14 00:25:30 -0700 56)         # topological order all of the children in the graph
9fd9cc02 (Andrej Karpathy 2020-04-14 00:25:30 -0700 57)         topo = []
575bc173 (Andrej Karpathy 2020-04-14 01:21:57 -0700 58)         visited = set()
```

In the `git log` line, `--format='%h %ad %s'` prints the short hash, author date and subject (first line of the message) of each commit, `--date=short` shortens the date, and `--` separates options from the file path whose history you want. `git blame -L 54,58` limits the output to lines 54-58.

`git grep` is faster than plain `grep -r` on big repos and skips `.git/`, build outputs and anything ignored. `git blame` plus `git show <hash>` is how you find *why* a strange line exists: the commit message and the rest of that commit's diff usually explain it.

### Reading an open-source codebase

A method that works on a 200-line project and on PyTorch alike:

1. **README first.** It says what the project is, how to install it, the main entry points and how to run the tests. micrograd's README shows an example (`from micrograd.engine import Value`) and a "Running tests" section saying `python -m pytest`.
2. **Get the size and the layout.** `ls`, then count lines per file, to know what you're up against:

   ```console
   $ find . -name "*.py" -not -path "./.git/*" | xargs wc -l
         22 ./setup.py
         67 ./test/test_engine.py
         60 ./micrograd/nn.py
          0 ./micrograd/__init__.py
         94 ./micrograd/engine.py
        243 total
   ```

   (`find` lists files matching a name, where `*` is a wildcard for "any characters", so `"*.py"` is every name ending in `.py`; `-not -path "./.git/*"` skips git's own directory; `xargs` passes the found names as arguments to `wc -l`.) In a big repo, look for the directory that matches the README's import path, and for packaging metadata (`pyproject.toml`, `setup.py`) which lists the command-line entry points.
3. **Find the entry point.** For a library, it's the class or function the README example uses (`Value` in `engine.py`). For a training repo, it's the script the README tells you to run (`train.py`, or a `[project.scripts]` entry in `pyproject.toml`). Read it top to bottom, following calls into other files with `git grep -n "def name"`.
4. **Read the tests.** Tests are executable documentation: small, concrete examples of every behaviour the authors cared about, with the expected answer. micrograd's `test/test_engine.py` builds the same expression in micrograd and in PyTorch and asserts the gradients match, which tells you exactly what "correct" means for the engine.
5. **Run something small and change something.** Run the tests (`python -m pytest`, in a venv with the project's dependencies; not run here), then put a `print` or a breakpoint (`breakpoint()` drops into the Python debugger) in the code path you care about and run the README example. Ten minutes of watching real values flow beats an hour of reading.
6. **Use history for "why".** `git log -- path` and `git blame` on the confusing part.

For C/C++/CUDA projects (NCCL, nccl-tests), step 2 also means finding the build file, usually a `Makefile` or `CMakeLists.txt`; [Make](#make-and-makefiles) below covers enough to read one.


## JSON and jq

Configuration, logs, API requests and responses, profiler traces, Kubernetes objects and model metadata are all passed around as text in a handful of formats. Each maps onto Python's built-in types, and each has a few ways to surprise you. JSON first, because the others are defined relative to it.

### JSON

**JSON** (JavaScript Object Notation) is the lingua franca of APIs and logs. It has exactly six kinds of value:

| JSON | Python | Example |
|---|---|---|
| object | `dict` | `{"lr": 0.0003, "name": "adamw"}` |
| array | `list` | `[0.9, 0.95]` |
| string | `str` | `"tiny-gpt"` (double quotes only) |
| number | `int` / `float` | `384`, `3e-4` |
| `true` / `false` | `True` / `False` | |
| `null` | `None` | |

Rules worth memorizing because they're what break hand-edited JSON: keys must be strings in double quotes; no trailing commas; no comments; no single quotes. A config file:

```json
{
  "model": {"name": "tiny-gpt", "n_layer": 6, "d_model": 384, "tied_embeddings": true},
  "optimizer": {"name": "adamw", "lr": 3e-4, "betas": [0.9, 0.95], "weight_decay": 0.1},
  "data": ["shard-000.bin", "shard-001.bin"],
  "resume_from": null
}
```

From Python, `json.load(f)`/`json.loads(s)` parse a file/string, and `json.dump(obj, f)`/`json.dumps(obj)` write one. The surprises:

```python
# json_demo.py
import json, math
cfg = json.load(open("config.json"))           # file -> dict
print(type(cfg["optimizer"]["betas"]), cfg["optimizer"]["betas"], cfg["resume_from"])
s = json.dumps({"loss": float("nan"), "step": 400, "ok": True, "ids": (1, 2)})
print(s)                                        # Python happily writes NaN ...
try:
    json.dumps({"loss": float("nan")}, allow_nan=False)
except ValueError as e:
    print("strict:", e)                          # ... which is not valid JSON
print(json.loads('{"1": "a"}'), json.dumps({1: "a"}))   # keys are always strings
rows = [json.loads(line) for line in open("metrics.jsonl")]
print(len(rows), rows[-1]["loss"], math.isnan(rows[-1]["loss"]))
```

```console
$ python3 json_demo.py
<class 'list'> [0.9, 0.95] None
{"loss": NaN, "step": 400, "ok": true, "ids": [1, 2]}
strict: Out of range float values are not JSON compliant: nan
{'1': 'a'} {"1": "a"}
5 nan True
```

- **NaN and Infinity are not JSON.** Python writes them anyway (`allow_nan=True` is the default), and Python reads them back, but a strict parser in another language (or a dashboard) will reject the line. A diverged training run logging `NaN` is exactly when you need the logs to parse.
- **Tuples become lists**, and **integer keys become strings**: `{1: "a"}` round-trips to `{"1": "a"}`. A dict keyed by layer index comes back keyed by `"0"`, `"1"`, ...
- `json.dumps(obj, indent=2)` pretty-prints; `python3 -m json.tool file.json` does the same from the shell and doubles as a validator:

```console
$ echo '{"a": 1,}' | python3 -m json.tool
Illegal trailing comma before end of object: line 1 column 8 (char 7)
```

### jq: JSON on the command line

[jq](https://jqlang.org) is a small program for slicing JSON in the shell. It ships with recent macOS (`/usr/bin/jq`) and is `apt install jq` on Ubuntu. Its argument is a **filter**, a little expression describing what to pull out:

```console
$ jq '.optimizer.lr' config.json                  # .key walks into objects
0.0003
$ jq '.optimizer.betas[1]' config.json            # [i] indexes arrays
0.95
$ jq -r '.data[]' config.json                     # [] iterates; -r prints raw strings, no quotes
shard-000.bin
shard-001.bin
$ jq '.model | keys' config.json                   # | feeds one filter's output into the next
[
  "d_model",
  "n_layer",
  "name",
  "tied_embeddings"
]
$ jq -c '.optimizer | {lr, wd: .weight_decay}' config.json   # build a new object; -c = compact
{"lr":0.0003,"wd":0.1}
```

`jq . file` just pretty-prints, which is the first thing to do with any unreadable one-line JSON blob (`curl -s URL | jq .` is a very common idiom). jq really shines on logs with one JSON object per line ([JSONL](#csv-and-jsonl), below), because it applies the filter to each line separately:

```console
$ cat metrics.jsonl
{"step": 100, "loss": 4.21, "lr": 0.0003, "event": "train"}
{"step": 200, "loss": 3.87, "lr": 0.0003, "event": "train"}
{"step": 200, "val_loss": 3.95, "event": "eval"}
{"step": 300, "loss": 3.62, "lr": 0.00029, "event": "train"}
{"step": 400, "loss": NaN, "lr": 0.00028, "event": "train"}
$ jq -c 'select(.event == "eval")' metrics.jsonl                     # keep matching lines
{"step":200,"val_loss":3.95,"event":"eval"}
$ jq -r 'select(.event=="train") | "\(.step) \(.loss)"' metrics.jsonl # string interpolation \( )
100 4.21
200 3.87
300 3.62
400 null
$ jq -s 'map(select(.event=="train")) | length' metrics.jsonl       # -s: slurp all lines into one array
4
```

Note the last training line: jq 1.7 accepted Python's non-standard `NaN` and printed it as `null`, silently. Other tools reject the line outright. Either way, a NaN loss is the thing you were looking for, and jq hid it. Know your tools' edge cases before trusting them on the edge case.

The `select(...)`, `-c`, `-r` and `\( )` forms above cover most uses of jq in this book, for example filtering a structured log in [Control planes](../platform/control-planes.md). The [jq manual](https://jqlang.org/manual/) has the rest.



## YAML

**YAML** is the format of almost every hand-written config in ML infrastructure: Kubernetes manifests (config files for a system that runs programs across a cluster), Helm charts, GitHub Actions and other CI pipelines (continuous integration: servers that run your tests on every commit), Docker Compose files, conda environments, Hydra/OmegaConf training configs, DeepSpeed and Axolotl configs. It's JSON's data model (dicts, lists, strings, numbers, booleans, null) written with indentation instead of braces, plus comments and multi-line strings. That makes it pleasant to read and write, and the "guess the type of an unquoted value" rule makes it easy to get wrong. The examples below use PyYAML (`import yaml`, installed by the [setup block](#one-setup-block-for-the-whole-book)), which is what most Python ML code uses.

### Mappings, lists and scalars

```python
# yaml_basics.py
import yaml
doc = """
# a training config
run_name: baseline-lr3e-4
model:
  n_layer: 6
  d_model: 384
  dropout: 0.0
  tied_embeddings: true
optimizer: {name: adamw, betas: [0.9, 0.95]}   # inline (flow) style
datasets:
  - name: web
    weight: 0.8
  - name: code
    weight: 0.2
resume_from: null
"""
cfg = yaml.safe_load(doc)
print(cfg)
print(cfg["datasets"][1]["name"], type(cfg["model"]["dropout"]).__name__)
```

```console
$ python3 yaml_basics.py
{'run_name': 'baseline-lr3e-4', 'model': {'n_layer': 6, 'd_model': 384, 'dropout': 0.0, 'tied_embeddings': True}, 'optimizer': {'name': 'adamw', 'betas': [0.9, 0.95]}, 'datasets': [{'name': 'web', 'weight': 0.8}, {'name': 'code', 'weight': 0.2}], 'resume_from': None}
code float
```

Reading it line by line:

- **`key: value`** is one entry of a **mapping** (a dict). The space after the colon is required.
- **Indentation is nesting.** `n_layer` and `d_model` are indented under `model:`, so they're keys of the dict `cfg["model"]`. Any consistent number of spaces works (2 is conventional); siblings must line up exactly.
- **`- item`** is one element of a **list**. `datasets:` is a list of two dicts; `- name: web` starts a new dict, and `weight: 0.8` indented to line up with `name` belongs to the same dict.
- **Flow style**: `{a: 1, b: 2}` and `[x, y]` are JSON-like inline forms of a mapping and a list. In fact every JSON document is valid YAML.
- **Scalars** (single values) are unquoted unless they need to be: `adamw` is a string, `6` an int, `0.0` a float, `true` a bool, `null` (or `~`, or nothing at all) is `None`.
- `#` starts a comment, when preceded by a space or at the start of a line.

Always load with **`yaml.safe_load`**, never plain `yaml.load` with the full loader: the full loader can construct arbitrary Python objects named in the file, so loading an untrusted YAML file could execute code.

### Strings, quoting and multi-line text

```python
# yaml_strings.py
import yaml
doc = """
literal: |
  #!/bin/bash
  echo step 1
  echo step 2
folded: >
  This long description
  is folded into
  one line.
stripped: |-
  no trailing newline
plain: a string with spaces, no quotes needed
quoted: "tab\\there, and a # that is not a comment"
single: 'it''s literal: \\t stays as backslash-t'
"""
for k, v in yaml.safe_load(doc).items():
    print(f"{k:9s}{v!r}")
```

```console
$ python3 yaml_strings.py
literal  '#!/bin/bash\necho step 1\necho step 2\n'
folded   'This long description is folded into one line.\n'
stripped 'no trailing newline'
plain    'a string with spaces, no quotes needed'
quoted   'tab\there, and a # that is not a comment'
single   "it's literal: \\t stays as backslash-t"
```

(The doubled backslashes `\\t` are because the YAML is inside a Python string; in a `.yaml` file you'd write `\t`.)

- **`|` (literal block)** keeps the following indented lines exactly, newlines included. This is how a whole shell script or a Python program is embedded in a Kubernetes manifest or a CI file.
- **`>` (folded block)** joins the lines with spaces into one paragraph. Good for long descriptions.
- A **`-`** after either (`|-`, `>-`) strips the final newline. Without it you get one trailing `\n`, which occasionally matters (a token or a password read from a YAML file with a stray newline).
- **Double quotes** allow escapes (`\t`, `\n`, `\"`) and protect characters that would otherwise mean something (`#`, `: `, a leading `-`, `{`, `*`, `&`, `!`, `%`, `@`).
- **Single quotes** are fully literal; the only escape is `''` for a single quote.

### Several documents in one file: `---`

A line containing only `---` separates **documents**. Kubernetes manifests use this constantly to put several objects in one file:

```yaml
# two.yaml - two documents in one file, as in a Kubernetes manifest
kind: Service
name: web
---
kind: Deployment
name: web
replicas: 2
```

`safe_load` refuses a multi-document file; `safe_load_all` returns one object per document (in error output, a line `...` marks where the top of a long Python traceback was cut; the last lines are the error itself):

```console
$ python3 -c 'import yaml; [print(d) for d in yaml.safe_load_all(open("two.yaml"))]'
{'kind': 'Service', 'name': 'web'}
{'kind': 'Deployment', 'name': 'web', 'replicas': 2}
$ python3 -c 'import yaml; yaml.safe_load(open("two.yaml"))'
...
yaml.composer.ComposerError: expected a single document in the stream
  in "two.yaml", line 2, column 1
but found another document
  in "two.yaml", line 4, column 1
```

### Anchors and merge keys

Configs with repeated blocks sometimes use **anchors**: `&name` labels a node, `*name` refers back to it, and `<<: *name` merges a mapping into another (a YAML 1.1 feature that PyYAML and most tools support):

```python
# anchors.py
import yaml
doc = """
defaults: &defaults
  lr: 3.0e-4
  warmup: 100
small:
  <<: *defaults
  d_model: 384
large:
  <<: *defaults
  lr: 1.5e-4
  d_model: 1024
"""
cfg = yaml.safe_load(doc)
print(cfg["small"]); print(cfg["large"])
```

```console
$ python3 anchors.py
{'lr': 0.0003, 'warmup': 100, 'd_model': 384}
{'lr': 0.00015, 'warmup': 100, 'd_model': 1024}
```

Keys written after the merge override the merged ones (`large` gets its own `lr`).

### The traps: unquoted values that are not what you meant

YAML guesses the type of every unquoted scalar from its spelling. PyYAML implements YAML 1.1, whose guessing rules are generous. This is the most important part of this section:

```python
# yaml_traps.py
import yaml
doc = """
country: NO
answer: yes
switch: off
version: 1.10
python: 3.10
lr: 3e-4
lr_ok: 3.0e-4
time: 1:30
mode: 0755
date: 2026-11-06
sha: 012345
empty:
tilde: ~
"""
for k, v in yaml.safe_load(doc).items():
    print(f"{k:8s}{v!r:28s}{type(v).__name__}")
```

```console
$ python3 yaml_traps.py
country False                       bool
answer  True                        bool
switch  False                       bool
version 1.1                         float
python  3.1                         float
lr      '3e-4'                      str
lr_ok   0.0003                      float
time    90                          int
mode    493                         int
date    datetime.date(2026, 11, 6)  date
sha     5349                        int
empty   None                        NoneType
tilde   None                        NoneType
```

Every one of these has caused a real bug somewhere:

- **The Norway problem.** `NO` (Norway's country code) is the boolean `False`. So are `no`, `n`, `off`, `false`; `yes`, `y`, `on`, `true` are `True`, in any capitalization. A list of country codes or a `debug: off` that you meant as a string isn't one.
- **Versions become floats.** `1.10` is the float `1.1`; `python: 3.10` is `3.1`. Every version number, image tag and commit-like ID must be quoted: `"1.10"`, `"3.10"`.
- **Scientific notation without a dot is a string in PyYAML.** YAML 1.1 requires a `.` in a float, so `lr: 3e-4` loads as the *string* `'3e-4'`, and your optimizer gets `lr='3e-4'` (`torch.optim.AdamW(..., lr='3e-4')` raises `TypeError: '<=' not supported between instances of 'float' and 'str'`, which doesn't mention YAML at all; other code silently does string math). Write `3.0e-4`, or `0.0003`. This one is ML-specific and very common.
- **Sexagesimal (base 60).** `1:30` is the integer 90 (1*60 + 30). A time, or a port mapping like `22:22` in an older Docker Compose file, turns into a number.
- **Octal.** A leading zero means base 8: `0755` is 493, and a numeric ID like `012345` becomes 5349. File modes (Unix permission bits, see [Permissions](linux-and-os.md#permissions-ownership-umask)) in YAML should be written `0o755` or quoted, and IDs quoted.
- **Dates** like `2026-11-06` become `datetime.date` objects, which then fail `json.dumps`.
- An **empty value** (`empty:`) is `None`, not `""`.

YAML 1.2 (2009) fixed the boolean, octal and sexagesimal rules, and some parsers (for example `ruamel.yaml` in Python) follow it, while others, PyYAML included, still implement 1.1, and some mix the two. The same file can therefore mean different things to different tools, which is the strongest argument for the one rule that avoids all of it: **quote every string that isn't obviously a word, and every version, ID, tag and time.** `yaml.safe_dump` quotes for you when writing:

```console
$ python3 -c 'import yaml; print(yaml.safe_dump({"country": "NO", "version": "1.10", "lr": 3e-4, "steps": 100}))'
country: 'NO'
lr: 0.0003
steps: 100
version: '1.10'
```

### Tabs, indentation and other parse errors

YAML forbids tab characters for indentation. A tab anywhere in the indentation is an error:

```console
$ printf 'model:\n\tn_layer: 6\n' > tabs.yaml        # printf is echo that understands \n (newline) and \t (tab)
$ python3 -c 'import yaml; yaml.safe_load(open("tabs.yaml"))'
...
yaml.scanner.ScannerError: while scanning for the next token
found character '\t' that cannot start any token
  in "tabs.yaml", line 2, column 1
```

Configure your editor to insert spaces in YAML files. Other mistakes and what they produce:

`indent.yaml`, with `d_model` indented one space more than its sibling:

```yaml
model:
  n_layer: 6
   d_model: 384
```

gives `mapping values are not allowed here  in "indent.yaml", line 3, column 11`. That error message is YAML's way of saying "I found a `:` where I didn't expect a key", and it almost always means bad indentation on that line or the one above. And `-b` instead of `- b` (no space after the dash) gives `could not find expected ':'`, because `-b` is read as a plain string, not a list item.

Worse than errors are the mistakes that parse fine:

```console
$ cat dedent.yaml
model:
  n_layer: 6
d_model: 384
$ python3 -c 'import yaml; print(yaml.safe_load(open("dedent.yaml")))'
{'model': {'n_layer': 6}, 'd_model': 384}
$ cat dup.yaml
lr: 3.0e-4
steps: 100
lr: 1.0e-3
$ python3 -c 'import yaml; print(yaml.safe_load(open("dup.yaml")))'
{'lr': 0.001, 'steps': 100}
```

`d_model` slid out of `model` and became a top-level key; the code reading `cfg["model"]["d_model"]` gets a `KeyError` or, worse, a default. And a **duplicate key** is silently resolved to the last value by PyYAML (the YAML spec says duplicates are an error; Kubernetes' parser rejects them). In a 300-line config, the second `lr:` is invisible.

### A parse check you can run on anything

Before you submit a manifest or launch a run from a config, parse it. This one-liner checks every file you give it, handles multi-document files, and reports each failure without stopping:

```console
$ python3 -c '
import sys, yaml
for f in sys.argv[1:]:
    try:
        docs = list(yaml.safe_load_all(open(f)))
        print(f"ok    {f}: {len(docs)} document(s)")
    except yaml.YAMLError as e:
        print(f"FAIL  {f}: {e}".replace("\n", " "))
' *.yaml
FAIL  dash.yaml: while scanning a simple key   in "dash.yaml", line 3, column 1 could not find expected ':'   in "dash.yaml", line 4, column 1
ok    dedent.yaml: 1 document(s)
ok    dup.yaml: 1 document(s)
FAIL  indent.yaml: mapping values are not allowed here   in "indent.yaml", line 3, column 11
FAIL  tabs.yaml: while scanning for the next token found character '\t' that cannot start any token   in "tabs.yaml", line 2, column 1
ok    two.yaml: 2 document(s)
```

(`*.yaml` is expanded by the shell into every file name ending in `.yaml`.) Note what it can't tell you: `dedent.yaml` and `dup.yaml` are "ok". Parsing proves the file is YAML, not that it says what you meant. For that, print the loaded object (`python3 -c 'import yaml,sys,pprint; pprint.pprint(yaml.safe_load(open(sys.argv[1])))' config.yaml`) and look at it. Kubernetes-specific checks (`kubectl apply --dry-run`) are in [YAML for Kubernetes](../orchestration/kubernetes-from-zero.md#yaml-for-kubernetes).


## TOML

**TOML** is the config format of the Python packaging world (`pyproject.toml`, which every modern Python project has, and which `uv add` edited [above](#uv-the-fast-way)), Rust's `Cargo.toml`, and a growing number of training frameworks (some RL trainers take a `.toml` config, see [RL infrastructure](../platform/rl-infrastructure.md)). It has the same data model as JSON, looks like an old-style `.ini` file, and was designed to avoid YAML's guessing: **strings are always quoted**, so a bare `NO` is a syntax error rather than `False`.

```toml
# train.toml
run_name = "baseline"
seed = 1234
version = "1.10"          # strings are always quoted, so no guessing
country = "NO"

[model]                   # a table = a dict
n_layer = 6
d_model = 384
dropout = 0.0

[optimizer]
name = "adamw"
lr = 3e-4                 # a float in TOML
betas = [0.9, 0.95]

[[datasets]]              # [[...]] = one element of a list of tables
name = "web"
weight = 0.8

[[datasets]]
name = "code"
weight = 0.2
```

```python
# toml_demo.py
import tomllib, pprint                      # stdlib since Python 3.11; read-only
with open("train.toml", "rb") as f:         # tomllib wants binary mode
    cfg = tomllib.load(f)
pprint.pprint(cfg, sort_dicts=False)
print(type(cfg["optimizer"]["lr"]).__name__, cfg["datasets"][1]["name"])
```

```console
$ python3 toml_demo.py
{'run_name': 'baseline',
 'seed': 1234,
 'version': '1.10',
 'country': 'NO',
 'model': {'n_layer': 6, 'd_model': 384, 'dropout': 0.0},
 'optimizer': {'name': 'adamw', 'lr': 0.0003, 'betas': [0.9, 0.95]},
 'datasets': [{'name': 'web', 'weight': 0.8}, {'name': 'code', 'weight': 0.2}]}
float code
$ printf 'country = NO\n' > bad.toml
$ python3 -c 'import tomllib; tomllib.load(open("bad.toml","rb"))'
...
tomllib.TOMLDecodeError: Invalid value (at line 1, column 11)
```

The rules: `key = value` at the top level; `[name]` starts a **table** (all following keys belong to it until the next header); `[a.b]` is a nested table; `[[name]]` appends one table to a **list of tables**, the TOML way to write a list of dicts. `3e-4` is a float here, unlike in PyYAML. Python's standard library can read TOML but not write it; `pip install tomli-w` if you need to.

In a `pyproject.toml` you'll mostly read three tables: `[project]` (name, version, `dependencies = [...]`), `[project.scripts]` (command-line entry points, e.g. `train = "mypkg.train:main"` means the `train` command calls `main()` in `mypkg/train.py`, which is how you find a repo's entry point) and `[tool.something]` (settings for tools like `pytest`, `ruff` or `uv`).



## CSV and JSONL

Two formats for many records rather than one config.

### CSV

**CSV** (comma-separated values) is a table as text: a header line of column names, then one row per line. Spreadsheets, `nvidia-smi --format=csv` (`nvidia-smi` is NVIDIA's GPU status tool), profiler exports and many benchmark scripts produce it. Its one rule that matters: a field containing a comma, a quote or a newline is wrapped in double quotes. That rule is why you must parse CSV with a real parser and not `line.split(",")`:

```console
$ cat results.csv
run,lr,val_loss,notes
baseline,3e-4,3.95,first try
warmup-1k,3e-4,3.81,"warmup 1000 steps, cosine"
lr-1e-3,1e-3,,diverged at step 400
```

```python
# csv_demo.py
import csv
with open("results.csv", newline="") as f:
    rows = list(csv.DictReader(f))
for r in rows:
    print(r)
print(sorted(rows, key=lambda r: float(r["val_loss"] or "inf"))[0]["run"])
print(open("results.csv").read().splitlines()[2].split(","))   # naive split: wrong
```

```console
$ python3 csv_demo.py
{'run': 'baseline', 'lr': '3e-4', 'val_loss': '3.95', 'notes': 'first try'}
{'run': 'warmup-1k', 'lr': '3e-4', 'val_loss': '3.81', 'notes': 'warmup 1000 steps, cosine'}
{'run': 'lr-1e-3', 'lr': '1e-3', 'val_loss': '', 'notes': 'diverged at step 400'}
warmup-1k
['warmup-1k', '3e-4', '3.81', '"warmup 1000 steps', ' cosine"']
```

Everything comes back as a string (`'3.95'`, and `''` for a missing value), so convert explicitly and decide what an empty field means. For real analysis, `pandas.read_csv` does the type conversion for you. Quick looks from the shell are fine with `cut -d, -f1,3 results.csv` (columns 1 and 3) or `column -s, -t < results.csv` (align into a table) as long as no field contains a comma; on this file `column` splits the quoted note in two and shifts the empty `val_loss`, which is the same bug as the naive split.

### JSONL

**JSONL** (JSON Lines, also `.ndjson`) is one complete JSON object per line, nothing else. It's the standard format for training metrics logs, structured application logs, and most LLM datasets (one example per line, e.g. `{"prompt": ..., "completion": ...}`). Three properties make it the right default for anything you write incrementally:

- **Appending is trivial and cheap**: open in append mode, write one line. No closing bracket to maintain, unlike a JSON array.
- **Line-oriented tools work**: `wc -l` counts records, `head`/`tail` sample them, `grep` finds them, `jq` filters them, `split -l` shards them.
- **Damage is local.** If the process crashes mid-write, you lose one line, not the file:

```python
# jsonl_demo.py
import json
with open("log.jsonl", "a") as f:                  # append: one object per line
    for step in (1, 2, 3):
        f.write(json.dumps({"step": step, "loss": 1.0 / step}) + "\n")
with open("log.jsonl", "a") as f:                  # simulate a crash mid-write
    f.write('{"step": 4, "lo')
good = []
for i, line in enumerate(open("log.jsonl"), 1):
    try:
        good.append(json.loads(line))
    except json.JSONDecodeError as e:
        print(f"line {i} unreadable: {e}")
print(len(good), "records, last:", good[-1])
```

```console
$ python3 jsonl_demo.py
line 4 unreadable: Unterminated string starting at: line 1 column 13 (char 12)
3 records, last: {'step': 3, 'loss': 0.3333333333333333}
$ wc -l log.jsonl
       3 log.jsonl
```

`wc -l` counts newline characters, so it says 3 even though there are 4 lines' worth of text: the torn last line has no newline. Readers of JSONL should skip (and count) unparseable lines rather than crash, exactly as above. Whether the three good lines are actually on disk after a power loss is a different question, answered in [fsync and durability](linux-and-os.md#fsync-and-durability-why-checkpoints-get-lost).

For large numeric data (tensors, token arrays) none of these text formats are appropriate; [Storage and data](storage-and-data.md) covers the binary formats (safetensors, Parquet, memory-mapped arrays).


## Regular expressions

A **regular expression** (regex) is a small pattern language for matching text. You need it for two things constantly: searching code and logs with `grep`, and pulling numbers out of log lines in Python. Here is a training log to work on (`rank0`, `rank3` and so on name which process wrote the line, in a training job that runs one process per GPU; the processes are numbered from 0 and each number is called a **rank**):

```console
$ cat train.log
2026-11-06 10:00:01 INFO  rank0 step=100 loss=4.2100 lr=3.00e-04 tok/s=51234
2026-11-06 10:00:09 INFO  rank0 step=200 loss=3.8731 lr=3.00e-04 tok/s=50981
2026-11-06 10:00:12 WARN  rank3 grad norm 12.5 exceeds clip 1.0
2026-11-06 10:00:17 INFO  rank0 step=300 loss=3.6202 lr=2.90e-04 tok/s=51102
2026-11-06 10:00:25 INFO  rank0 step=400 loss=nan lr=2.80e-04 tok/s=50877
2026-11-06 10:00:31 ERROR rank5 CUDA error: an illegal memory access was encountered
```

### The syntax you actually need

Most characters match themselves: the pattern `loss` matches the text `loss`. A few characters are special:

| Pattern | Matches |
|---|---|
| `.` | any one character |
| `[abc]`, `[0-9]`, `[A-Z]` | one character from the set or range; `[^ ]` is any character *except* a space |
| `\d`, `\s`, `\w` | a digit, a whitespace character, a "word" character (letter, digit, `_`); capitals `\D \S \W` are the opposites |
| `*`, `+`, `?` | the previous item 0 or more times, 1 or more, 0 or 1 |
| `{3}`, `{2,5}` | exactly 3 times, 2 to 5 times |
| `^`, `$` | start, end of the line |
| `\b` | a word boundary (between a `\w` and a non-`\w` character) |
| `a\|b` | `a` or `b` |
| `( )` | group (and, in Python, capture what matched) |
| `\.`, `\(`, `\\` | a literal `.`, `(`, `\`: backslash removes the special meaning |

So `loss=[0-9]+\.[0-9]+` is "`loss=`, one or more digits, a literal dot, one or more digits", and `rank[1-9]` is "`rank` followed by a digit 1-9".

### grep

`grep PATTERN FILE` prints matching lines. Use `grep -E` ("extended" regex) so that `+ ? | ( ) {}` work as in the table; plain `grep` needs them backslashed. Useful flags: `-i` ignore case, `-v` invert (non-matching lines), `-c` count, `-n` line numbers, `-o` print only the matched part, `-r` recurse into directories, `-A 3`/`-B 3` include 3 lines after/before each match (context around an error).

```console
$ grep -E 'WARN|ERROR' train.log
2026-11-06 10:00:12 WARN  rank3 grad norm 12.5 exceeds clip 1.0
2026-11-06 10:00:31 ERROR rank5 CUDA error: an illegal memory access was encountered
$ grep -c INFO train.log
4
$ grep -E 'loss=[0-9]+\.[0-9]+' train.log | wc -l     # lines with a finite-looking loss
       3
$ grep -oE 'loss=[^ ]+' train.log
loss=4.2100
loss=3.8731
loss=3.6202
loss=nan
$ grep -nE '\bnan\b|\binf\b' train.log                # whole word nan or inf
5:2026-11-06 10:00:25 INFO  rank0 step=400 loss=nan lr=2.80e-04 tok/s=50877
$ grep -oE 'loss=[0-9.]+' train.log | cut -d= -f2 | sort -n | head -1   # best loss (cut: field 2 after "="; sort -n: numeric sort)
3.6202
$ grep -rn --include='*.py' 'import re' .              # search a source tree, Python files only
./parse_log.py:1:import re, math
```

Always put the pattern in **single quotes**, so the shell passes `|`, `*`, `$` and `\` through to grep untouched. In a git repository, `git grep -nE` takes the same patterns and is faster. (`rg`, ripgrep, is a popular faster grep with the same idea and saner defaults, if you want to install one more tool.)

### Python `re`

```python
# parse_log.py
import re, math
pat = re.compile(r"step=(\d+) loss=(\S+) lr=(\S+)")     # \d digit, \S non-space, ( ) capture
steps, losses = [], []
for line in open("train.log"):
    m = pat.search(line)                                 # search: anywhere in the line
    if m:
        step, loss, lr = int(m.group(1)), float(m.group(2)), float(m.group(3))
        steps.append(step); losses.append(loss)
print(list(zip(steps, losses)))
print("first non-finite at step", next(s for s, l in zip(steps, losses) if not math.isfinite(l)))

errors = [l for l in open("train.log") if re.search(r"\b(ERROR|WARN)\b", l)]
ranks = sorted({int(r) for r in re.findall(r"rank(\d+)", "".join(errors))})
print("ranks with problems:", ranks)

m = re.match(r"(?P<date>\S+) (?P<time>\S+) (?P<level>[A-Z]+)\s+(?P<rank>rank\d+) (?P<msg>.*)",
             open("train.log").readlines()[-1])
print(m.groupdict())
print(re.sub(r"\d+\.\d+e-\d+", "<LR>", "lr=3.00e-04 lr=2.90e-04"))
print(re.findall(r"loss=.*", "loss=4.21 lr=3e-4"), re.findall(r"loss=.*?(?= |$)", "loss=4.21 lr=3e-4"))
```

```console
$ python3 parse_log.py
[(100, 4.21), (200, 3.8731), (300, 3.6202), (400, nan)]
first non-finite at step 400
ranks with problems: [3, 5]
{'date': '2026-11-06', 'time': '10:00:31', 'level': 'ERROR', 'rank': 'rank5', 'msg': 'CUDA error: an illegal memory access was encountered'}
lr=<LR> lr=<LR>
['loss=4.21 lr=3e-4'] ['loss=4.21']
```

What each piece shows:

- **Raw strings `r"..."`.** Always write patterns as raw strings so Python doesn't interpret the backslashes first (`"\b"` in a normal string is a backspace character, not a word boundary).
- **`re.search`** finds a match anywhere in the string; **`re.match`** only at the start; **`re.fullmatch`** the whole string. All return a match object or `None`, so test before using it.
- **Groups.** `m.group(1)` is the text matched by the first `( )`. Named groups `(?P<name>...)` and `m.groupdict()` turn a log line into a dict, which is the start of every log parser.
- **`re.findall`** returns all matches (just the group if there is one); **`re.sub`** replaces them, handy for normalizing lines before comparing or counting them ("which error messages occur, ignoring the numbers in them?").
- **Greedy vs lazy.** `.*` grabs as much as possible, so `loss=.*` ran to the end of the line. `.*?` grabs as little as possible; with the lookahead `(?= |$)` ("followed by a space or the end, without consuming it") it stops at the value. Usually `\S+` (non-spaces) is the simpler fix.
- `float("nan")` parses, so the regex happily extracted a NaN. Regexes match text; deciding whether a value is sane is still your job.

If the log is JSONL, skip regexes and use `json.loads` or `jq`. Regexes are for logs written as text for humans, which is still most of them (NCCL, Slurm, the Linux kernel, CUDA).


## Reading C

PyTorch, NCCL, CUDA kernels, the Linux kernel, Python itself and every driver are written in C or C++. You don't need to write C to be an ML engineer, but you need to *read* it: a CUDA kernel (a function that runs on an NVIDIA GPU, written in C++), a short benchmark in [Computer architecture](computer-architecture.md), the `nccl-tests` source, a stack trace that ends in `libc10.so`. C is a small language, and the part you need is smaller. The key idea that Python hides from you: **data lives at numbered addresses in memory, and C lets you hold and do arithmetic on those addresses.** A tensor is exactly that: a pointer to the first byte of a buffer, plus shape and strides.

### Compiling and running

Python runs your source file directly. C must first be **compiled**: a compiler translates the source to a machine-code executable, which you then run. On a Mac, `xcode-select --install` installs Apple's `clang`; on Linux, `sudo apt install build-essential` gets `gcc` (and `clang` is a package away). The two accept the same flags for everything here.

```console
$ clang -Wall -O2 -o types types.c     # compile types.c into an executable called types
$ ./types                              # run it (./ = "in this directory")
```

`-o types` names the output; `-O2` turns on optimizations (what you want for timing anything); `-Wall` turns on warnings, which in C are often real bugs. `-g` adds debug information so debuggers and sanitizers can print file and line numbers. `gcc` and `clang++`/`g++` (for C++) take the same flags.

### Types, sizes and functions

```c
// types.c - C's basic types and their sizes on this machine
#include <stdio.h>     // printf
#include <stdint.h>    // fixed-width integers: int32_t, int64_t, uint8_t, ...

int square(int x) {    // return type, name, typed parameters
    return x * x;
}

int main(void) {
    int i = 7;                  // 32-bit signed integer
    long n = 1L << 40;          // 64-bit on Linux and macOS (but 32-bit on Windows)
    float f = 0.1f;             // 32-bit float (fp32); the f suffix makes the literal a float
    double d = 0.1;             // 64-bit float (fp64), Python's float
    char c = 'A';               // one byte; a character is just a small integer
    uint8_t byte = 255;
    byte = byte + 1;            // unsigned arithmetic wraps around: 255 + 1 == 0
    int q = 7 / 2;              // integer division truncates
    printf("sizeof: int=%zu long=%zu float=%zu double=%zu char=%zu int64_t=%zu\n",
           sizeof(int), sizeof(long), sizeof(float), sizeof(double), sizeof(char), sizeof(int64_t));
    printf("i=%d square=%d n=%ld f=%.10f d=%.10f c=%c (%d) byte=%u q=%d\n",
           i, square(i), n, f, d, c, c, byte, q);
    return 0;                   // exit code 0 = success
}
```

```console
$ clang -Wall -O2 -o types types.c && ./types; echo "exit code $?"
sizeof: int=4 long=8 float=4 double=8 char=1 int64_t=8
i=7 square=49 n=1099511627776 f=0.1000000015 d=0.1000000000 c=A (65) byte=0 q=3
exit code 0
```

The differences from Python that matter for reading:

- **Every variable has a fixed type and size.** `sizeof(T)` is its size in bytes. `float` is fp32 (so `0.1f` is stored as 0.1000000015), `double` is fp64. Integers don't grow: `uint8_t` wraps from 255 to 0, and a signed `int` overflowing past about 2.1 billion is **undefined behaviour** (C's term for "the language makes no promise": the compiler may assume it never happens, so the result can be a silently wrong value or a crash; a real bug class when indexing tensors with more than 2^31 elements, which is why CUDA code often uses `int64_t` indices). When the exact width matters, code uses `<stdint.h>` names: `int32_t`, `int64_t`, `uint8_t`.
- **Integer division truncates**: `7 / 2` is 3, like Python's `//` for positive numbers.
- **Functions declare their types**: `int square(int x)` takes an `int` and returns an `int`. `void` means "returns nothing" (or, as `(void)`, "takes nothing").
- **Statements end with `;`, blocks are `{ }`.** Indentation is for humans only. `for (int i = 0; i < n; i++)` is `for i in range(n)`: initialize, condition, step.
- **`main` is where the program starts**, and its return value is the process's [exit code](#exit-codes-pipes-redirection-variables-a-first-look).
- **`#include <stdio.h>`** pastes in a *header* that declares library functions (more [below](#headers-and-multiple-files)). `printf` is formatted printing: `%d` int, `%ld` long, `%zu` a size (`sizeof` result), `%f`/`%.10f` a float or double, `%c` a char, `%u` unsigned, `%p` a pointer (an address), `\n` a newline.

### Pointers and addresses

Every variable occupies some bytes at some **address** in memory. A **pointer** is a variable whose value is an address.

```c
// pointers.c - addresses, pointers, arrays and pointer arithmetic
#include <stdio.h>

void scale(float *x, int n, float a) {   // receives the address of the first element
    for (int i = 0; i < n; i++)
        x[i] *= a;                        // writes go to the caller's memory
}

int main(void) {
    float v = 2.5f;
    float *p = &v;                        // & = "address of"; p holds v's address
    printf("v=%.1f  &v=%p  p=%p  *p=%.1f\n", v, (void *)&v, (void *)p, *p);
    *p = 4.0f;                            // * = "the thing p points at": this changes v
    printf("after *p = 4: v=%.1f\n", v);

    float x[4] = {1, 2, 3, 4};            // 4 floats, contiguous, 16 bytes
    for (int i = 0; i < 4; i++)
        printf("x[%d]=%.0f at %p\n", i, x[i], (void *)&x[i]);   // addresses 4 bytes apart

    float *q = x;                         // an array name is the address of its first element
    printf("q+2 = %p, *(q+2) = %.0f, q[2] = %.0f\n", (void *)(q + 2), *(q + 2), q[2]);
    printf("(char*)(q+2) - (char*)q = %ld bytes\n", (char *)(q + 2) - (char *)q);

    scale(x, 4, 10.0f);                   // pass the address; scale edits x in place
    printf("after scale: %.0f %.0f %.0f %.0f\n", x[0], x[1], x[2], x[3]);
    scale(x + 2, 2, 0.5f);                // a pointer into the middle = a view of the last 2
    printf("after scale(x+2, 2): %.0f %.0f %.0f %.0f\n", x[0], x[1], x[2], x[3]);
    return 0;
}
```

```console
$ clang -Wall -O2 -o pointers pointers.c && ./pointers
v=2.5  &v=0x16b2fecfc  p=0x16b2fecfc  *p=2.5
after *p = 4: v=4.0
x[0]=1 at 0x16b2fed00
x[1]=2 at 0x16b2fed04
x[2]=3 at 0x16b2fed08
x[3]=4 at 0x16b2fed0c
q+2 = 0x16b2fed08, *(q+2) = 3, q[2] = 3
(char*)(q+2) - (char*)q = 8 bytes
after scale: 10 20 30 40
after scale(x+2, 2): 10 20 15 20
```

(Addresses are hexadecimal and differ on every run.) One piece of syntax first: a type in parentheses before a value, like `(void *)&v` or `(char *)q`, is a **cast**, which converts the value to that type. `void *` means "an address of anything"; `printf`'s `%p` expects one, hence the casts in the `printf` calls. Read the rest like this:

- **`float *p`** declares `p` as "pointer to float". **`&v`** is the address of `v`. **`*p`** is the float stored at that address ("dereferencing"); assigning to `*p` changes `v`. The two uses of `*` (in a declaration, and in an expression) are the most confusing bit of C syntax; a declaration's `*` says what the variable *is*, an expression's `*` follows the pointer.
- **Arrays are contiguous.** `float x[4]` is 16 bytes in a row; the addresses go up by `sizeof(float)` = 4. This is the property every tensor, every GPU kernel and every "memory bandwidth" argument in the book rests on.
- **Pointer arithmetic counts elements, not bytes.** `q + 2` is the address 2 *floats* past `q`, which is 8 bytes. `q[2]` is defined as `*(q + 2)`. Casting to `char *` (1-byte elements) shows the byte distance.
- **Passing a pointer lets a function modify the caller's data.** C passes arguments by copying them; `scale` gets a copy of the *address*, so its writes land in `x`. Passing `x + 2` hands it a view of the last two elements, just as `x[2:]` gives a view of a tensor that shares storage.
- **Nothing checks bounds.** `x[10]` compiles and reads whatever happens to be there ([below](#when-c-goes-wrong)).
- `const float *a` (in kernel signatures everywhere) means "pointer to floats this function promises not to modify". `NULL` is the null pointer, address 0, used to mean "no object".

### Structs, the heap and 2-D arrays

A **struct** groups named fields into one block of memory, like a dataclass without methods. And memory comes from two places:

- The **stack**: local variables like `float v` and `float x[4]` above. Allocation is free, and the memory is released automatically when the function returns. It's small (8 MB by default for the main thread on Linux and macOS; a thread is one flow of execution in a program, see [Concurrency](concurrency.md#programs-processes-and-threads)), so big arrays don't go here.
- The **heap**: memory you request with `malloc(nbytes)` and must release with `free(ptr)`. It lives until you free it, can be gigabytes, and is where every real buffer comes from. Forgetting `free` leaks memory; using memory after `free` (or freeing twice) corrupts it.

```c
// matrix.c - a struct, heap memory, and a 2-D array stored as 1-D (row-major)
#include <stdio.h>
#include <stdlib.h>    // malloc, free

typedef struct {       // a struct groups named fields; typedef lets us write just "Matrix"
    int rows, cols;
    float *data;       // pointer to rows*cols floats, allocated separately
} Matrix;

Matrix *matrix_new(int rows, int cols) {
    Matrix *m = malloc(sizeof(Matrix));                     // heap: lives until free()
    m->rows = rows;                                         // m->rows is (*m).rows
    m->cols = cols;
    m->data = malloc((size_t)rows * cols * sizeof(float));  // uninitialized memory
    if (m->data == NULL) { fprintf(stderr, "out of memory\n"); exit(1); }
    return m;
}

void matrix_free(Matrix *m) {
    free(m->data);     // free what was malloc'd, inner first
    free(m);
}

float *at(Matrix *m, int r, int c) {           // element (r, c) of a row-major matrix
    return &m->data[r * m->cols + c];          // row r starts r*cols elements in
}

int main(void) {
    Matrix *m = matrix_new(2, 3);
    for (int r = 0; r < 2; r++)
        for (int c = 0; c < 3; c++)
            *at(m, r, c) = 10 * r + c;
    printf("sizeof(Matrix)=%zu  rows=%d cols=%d\n", sizeof(Matrix), m->rows, m->cols);
    printf("flat memory:");
    for (int i = 0; i < 6; i++) printf(" %.0f", m->data[i]);
    printf("\nelement (1,2) = %.0f at offset %ld floats\n",
           *at(m, 1, 2), at(m, 1, 2) - m->data);
    int on_stack[3] = {1, 2, 3};   // stack: freed automatically when main returns
    printf("stack array at %p, heap Matrix at %p\n", (void *)on_stack, (void *)m);
    matrix_free(m);                // after this, m must not be used
    return 0;
}
```

```console
$ clang -Wall -O2 -o matrix matrix.c && ./matrix
sizeof(Matrix)=16  rows=2 cols=3
flat memory: 0 1 2 10 11 12
element (1,2) = 12 at offset 5 floats
stack array at 0x16af16d08, heap Matrix at 0x6000013ac020
```

- **`m->rows`** is shorthand for `(*m).rows`: a field through a pointer. With a struct value (not a pointer) you write `m.rows`.
- **`sizeof(Matrix)` is 16**: two 4-byte ints and an 8-byte pointer. The struct holds the *pointer* to the data, not the data, exactly like a tensor's metadata pointing at its storage.
- **A 2-D array is a 1-D array with an indexing rule.** Element `(r, c)` of a row-major `rows x cols` matrix is at offset `r * cols + c`. That is what PyTorch's **strides** describe. The same matrix in PyTorch:

```python
>>> import torch
>>> t = torch.tensor([[0., 1, 2], [10, 11, 12]])
>>> t.stride(), t.element_size(), t.flatten().tolist()
((3, 1), 4, [0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
>>> t.T.stride(), t.T.is_contiguous()
((1, 3), False)
```

Stride `(3, 1)` means "step 3 elements per row, 1 per column", so `t[r, c]` is at `r*3 + c`, the `at()` function above. `t.T` is the same memory with the strides swapped, which is why a transpose is free and why it's "not contiguous". When a CUDA kernel computes `A[row * K + k]`, this is all it's doing.

The rest of the program:

- **`typedef struct {...} Matrix;`** names the type so you can write `Matrix` instead of `struct Matrix`. `(size_t)rows * cols` converts to a 64-bit size before multiplying, so a large matrix doesn't overflow `int`.
- **`malloc` returns uninitialized memory** (whatever bytes were there), like `torch.empty`, and returns `NULL` on failure. `calloc` returns zeroed memory, like `torch.zeros`.
- **`fprintf(stderr, ...)`** prints to the error stream instead of normal output; `exit(1)` ends the program with exit code 1.

### Headers and multiple files

Real programs span many files. A **header** (`.h`) contains **declarations**: the names and types of functions and structs, without their bodies, so other files can call them. The matching `.c` file has the **definitions**. `#include "vec.h"` (quotes: look in this directory first) pastes the header's text in before compiling; `#include <stdio.h>` (angle brackets) looks in the system include directories.

```c
// vec.h - declarations: what exists, not how it works
#pragma once
float dot(const float *a, const float *b, int n);
```

```c
// vec.c - the definition
#include "vec.h"
float dot(const float *a, const float *b, int n) {
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}
```

```c
// main.c - uses dot() knowing only its declaration from vec.h
#include <stdio.h>
#include "vec.h"
int main(void) {
    float a[3] = {1, 2, 3}, b[3] = {4, 5, 6};
    printf("dot = %.1f\n", dot(a, b, 3));
    return 0;
}
```

```console
$ clang -Wall -O2 -o dotprog main.c vec.c          # compile both and link them together
$ ./dotprog
dot = 32.0
$ clang -Wall -O2 -o dotprog main.c                # forget vec.c
Undefined symbols for architecture arm64:
  "_dot", referenced from:
      _main in main-b47c74.o
ld: symbol(s) not found for architecture arm64
clang: error: linker command failed with exit code 1 (use -v to see invocation)
```

Building is two stages. **Compiling** turns each `.c` file separately into an **object file** (`.o`) of machine code (`clang -c vec.c` produces `vec.o`); `main.c` compiles fine on its own because the header promised `dot` exists. **Linking** (the program `ld`, the *linker*) combines the object files and libraries into one executable and connects each call to its definition. The error above is a *link* error: the promise was never kept. (A **symbol** is the name of a function or global variable in compiled code; macOS prefixes C names with `_`, hence `_dot`.) You'll see the same "undefined symbol" message when a PyTorch C++ extension is built against a different PyTorch version than the one importing it. Libraries such as `libnccl.so` are precompiled object code you link against with `-lnccl`; how the system finds them at run time is in [Shared libraries](linux-and-os.md#shared-libraries-and-ld_library_path).

`#pragma once` stops a header being pasted twice into one file. Lines starting with `#` (`#include`, `#define N 1024`, `#if`) are handled by the **preprocessor**, a text-substitution pass before compilation; `#define` constants and macros are just text replacement.

### When C goes wrong

C trusts you. This writes one element past the end of a 4-element heap array:

```c
// oob.c - reading one element past the end of an array
#include <stdio.h>
#include <stdlib.h>
int main(void) {
    float *x = malloc(4 * sizeof(float));
    for (int i = 0; i <= 4; i++)      // bug: <= should be <
        x[i] = i;
    printf("x[4] = %.0f\n", x[4]);
    free(x);
    return 0;
}
```

```console
$ clang -O0 -o oob oob.c && ./oob
x[4] = 4
$ clang -O0 -g -fsanitize=address -o oob_asan oob.c && ./oob_asan
=================================================================
==30035==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x6020000000c0 at pc 0x000100ffc88c bp 0x00016ee02cd0 sp 0x00016ee02cc8
WRITE of size 4 at 0x6020000000c0 thread T0
    #0 0x000100ffc888 in main oob.c:7
    #1 0x00018960ab94  (<unknown module>)

0x6020000000c0 is located 0 bytes after 16-byte region [0x6020000000b0,0x6020000000c0)
allocated by thread T0 here:
    #0 0x00010165d38c in malloc+0x78 (libclang_rt.asan_osx_dynamic.dylib:arm64e+0x3d38c)
    #1 0x000100ffc804 in main oob.c:5
...
```

(`-O0` turns optimizations off, so the compiler can't rearrange or remove the buggy code.) The plain build "works": it scribbled on 4 bytes it doesn't own and printed the value back. That's what memory bugs look like in C: silent until they corrupt something important, often far away, sometimes only on another machine. **AddressSanitizer** (`-fsanitize=address`) instruments every memory access and stops at the first bad one, naming the line of the bad write (`oob.c:7`) and where the buffer was allocated (`oob.c:5`). The GPU equivalent of this bug is CUDA's "an illegal memory access was encountered", and its sanitizer is `compute-sanitizer` ([Writing kernels](../compute/accelerator/writing-kernels.md)).

### The C++ you'll see in CUDA and PyTorch code

CUDA kernels and PyTorch extensions are C++, but mostly C-flavoured C++: the C above plus a handful of features. Recognize these and you can read them:

```cpp
// cpp_bits.cpp - the C++ features you'll meet in CUDA and PyTorch C++ code
#include <cstdio>
#include <vector>

namespace mylib {                         // names grouped like a module: mylib::add_one
void add_one(float &x) { x += 1.0f; }     // & in a parameter = reference: an alias, no copying
}

template <typename T, int N>              // generic over a type T and a compile-time int N
T sum(const T *x) {
    T s = 0;
    for (int i = 0; i < N; i++) s += x[i];
    return s;
}

struct Counter {                          // a struct with a member function
    int n = 0;
    void bump() { n++; }
};

int main() {
    float v = 1.0f;
    mylib::add_one(v);                    // v changes, though we didn't write &v
    printf("v=%.1f\n", v);

    int xi[3] = {1, 2, 3};
    double xd[2] = {0.5, 0.25};
    printf("sum<int,3>=%d  sum<double,2>=%.2f\n", sum<int, 3>(xi), sum<double, 2>(xd));

    auto *buf = new float[4]{1, 2, 3, 4}; // auto: compiler infers float*; new[] is C++'s malloc
    printf("buf[3]=%.0f\n", buf[3]);
    delete[] buf;                         // new[] pairs with delete[]

    std::vector<float> w(5, 0.5f);        // growable array that frees itself
    w.push_back(2.0f);
    printf("w.size()=%zu w.back()=%.1f data=%p\n", w.size(), w.back(), (void *)w.data());

    Counter c;
    c.bump(); c.bump();
    printf("c.n=%d\n", c.n);
    return 0;
}
```

```console
$ clang++ -std=c++17 -Wall -O2 -o cpp_bits cpp_bits.cpp && ./cpp_bits
v=2.0
sum<int,3>=6  sum<double,2>=0.75
buf[3]=4
w.size()=6 w.back()=2.0 data=0x6000017282d0
c.n=2
```

(`clang++` is the C++ compiler; `-std=c++17` selects the 2017 version of the C++ language.)

- **References** (`float &x`): a parameter that is an alias for the caller's variable. It does what a pointer does without the `&`/`*` at each use. `const torch::Tensor &t` in a PyTorch extension means "the caller's tensor, not copied, not modified".
- **Namespaces** (`a::b`): `torch::Tensor`, `std::vector`, `at::cuda::getCurrentCUDAStream()`. Read `::` like Python's `.` for modules.
- **Templates** (`template <typename T, int N>`): one source, compiled separately for each type or constant you use it with (`sum<int, 3>`). Kernels use them to make tile sizes compile-time constants.
- **`auto`** lets the compiler infer the type.
- **`new`/`delete`** are C++'s `malloc`/`free` (`new T[n]` pairs with `delete[]`). Modern code mostly uses containers like **`std::vector`** (a growable array that frees itself; `.data()` is the raw pointer to its contiguous elements) instead.
- **Structs and classes have member functions** (`c.bump()`), constructors and default field values.

That's enough for this book's C++. The CUDA-specific additions (`__global__`, `<<<grid, block>>>`, `threadIdx`) and the rest of the C++ that kernels use (`constexpr`, `reinterpret_cast`, `#pragma unroll`) are taught in [Just enough C++ to read CUDA](../compute/accelerator/writing-kernels.md#just-enough-c-to-read-cuda).



## Make and Makefiles

Typing `clang` commands by hand stops scaling at three files. **Make** is the classic build tool: you describe what each file is built from and how, and `make` runs only the commands whose inputs changed. C/CUDA projects you'll build in this book (nccl-tests, perftest) use it, and many Python repos use a `Makefile` simply as a list of named shortcuts (`make test`, `make lint`). It comes with the Mac developer tools and Linux's `build-essential`.

A `Makefile` for the three files from [Headers and multiple files](#headers-and-multiple-files):

```makefile
# Makefile
CC      = clang
CFLAGS  = -Wall -O2

# target: prerequisites
#     recipe lines (shell commands) must start with a TAB character
dotprog: main.o vec.o
	$(CC) -o $@ $^

%.o: %.c vec.h
	$(CC) $(CFLAGS) -c $<

.PHONY: run clean
run: dotprog
	./dotprog

clean:
	rm -f *.o dotprog
```

How to read it:

- **`NAME = value`** defines a variable; **`$(NAME)`** uses it. Variables can be overridden from the command line: `make CC=gcc CFLAGS=-O0`. Build instructions for real projects mostly consist of setting such variables (nccl-tests is built with `make MPI=1 CUDA_HOME=/usr/local/cuda`, for example).
- **A rule** is `target: prerequisites` followed by recipe lines. It means "to make `target`, first make sure the prerequisites are up to date; then, if any prerequisite is newer than the target, run the recipe."
- **`%.o: %.c vec.h`** is a pattern rule: any `X.o` is built from `X.c` (and `vec.h`). In recipes, **`$@`** is the target, **`$<`** the first prerequisite, **`$^`** all of them.
- **`.PHONY`** marks targets that are names for actions, not files.
- `make` with no argument builds the first target in the file; `make run` builds `run`.

```console
$ make
clang -Wall -O2 -c main.c
clang -Wall -O2 -c vec.c
clang -o dotprog main.o vec.o
$ make
make: `dotprog' is up to date.
$ touch vec.c                  # update vec.c's modification time, as if edited
$ make
clang -Wall -O2 -c vec.c
clang -o dotprog main.o vec.o
$ touch vec.h                  # both .o files depend on the header
$ make
clang -Wall -O2 -c main.c
clang -Wall -O2 -c vec.c
clang -o dotprog main.o vec.o
$ make run
./dotprog
dot = 32.0
$ make -n clean                # -n: print what would run, run nothing
rm -f *.o dotprog
```

Make decides by comparing file modification times, which is all "incremental build" means. Make echoes every command before running it, so a failing build shows you the exact compiler command that failed: copy it, run it by hand, add flags. `make -j8` runs up to 8 independent recipes in parallel; `make V=1` makes many projects print full commands.

The error everyone hits once: recipe lines must start with a **tab**, not spaces.

```console
$ make -f Makefile.spaces      # the same file with the tabs replaced by 4 spaces
Makefile.spaces:8: *** missing separator.  Stop.
```

Bigger C/C++ projects generate their Makefiles: `./configure && make` (autotools: `configure` inspects your system and writes a Makefile), or **CMake** (`cmake -B build && cmake --build build`, reading a `CMakeLists.txt`). PyTorch extensions are built through Python with **ninja**, a faster Make-like tool. You run all of these the same way: read the README's build section, set the variables it mentions (where CUDA is, which GPU architecture), and read the first error.



## Reading logs and tracebacks

Most of your time debugging is spent reading text a program printed when it failed. A few habits make it much faster.

### Python tracebacks: read from the bottom

```python
# train.py - a bug three calls deep
import torch, torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 64)
        self.proj = nn.Linear(32, 10)        # bug: should be nn.Linear(64, 10)

    def forward(self, ids):
        h = self.embed(ids)                  # (B, T, 64)
        return self.proj(h)                  # expects (..., 32)

def train_step(model, batch):
    logits = model(batch)
    return logits.mean()

def main():
    model = Model()
    batch = torch.randint(0, 100, (4, 16))  # (B, T)
    print("loss", train_step(model, batch).item())

if __name__ == "__main__":
    main()
```

```console
$ python train.py
Traceback (most recent call last):
  File "/private/tmp/tkdemo/tb/train.py", line 24, in <module>
    main()
  File "/private/tmp/tkdemo/tb/train.py", line 21, in main
    print("loss", train_step(model, batch).item())
                  ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/private/tmp/tkdemo/tb/train.py", line 15, in train_step
    logits = model(batch)
             ^^^^^^^^^^^^
  File "/private/tmp/tk/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/private/tmp/tk/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/private/tmp/tkdemo/tb/train.py", line 12, in forward
    return self.proj(h)                  # expects (..., 32)
           ^^^^^^^^^^^^
  File "/private/tmp/tk/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1787, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/private/tmp/tk/.venv/lib/python3.12/site-packages/torch/nn/modules/module.py", line 1798, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/private/tmp/tk/.venv/lib/python3.12/site-packages/torch/nn/modules/linear.py", line 134, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: addmm: mat1 and mat2 shapes cannot be multiplied (64x64 and 32x10)
```

1. **The last line is the error**: its type (`RuntimeError`) and message. Read that first.
2. **Frames are listed outermost first**, so the line that actually failed is at the bottom, and the call chain that led there reads downward. Each frame is a file, a line number, a function name and the source line.
3. **Find the lowest frame in *your* code** (here `train.py`, line 12, `self.proj(h)`). Frames inside `site-packages/` are library code; the bug is almost never there. The `module.py` frames are PyTorch's `nn.Module.__call__` machinery and appear twice per module call.
4. **Translate the message into your terms.** `addmm` is the matmul inside `nn.Linear`; `64x64` is the input flattened to $(B \cdot T, d) = (4 \cdot 16, 64)$; `32x10` is the transposed weight. So the layer expects 32 input features and got 64: the bug is in `__init__`, not on the line where it surfaced.

Two variations to know. **Chained exceptions** print several tracebacks separated by "During handling of the above exception, another exception occurred:" or "The above exception was the direct cause of the following exception:"; the first one is usually the root cause. And a **multi-process job** (DataLoader workers, or `torchrun` ranks: `torchrun` starts one copy of your training script per GPU, covered in [the Linux chapter](linux-and-os.md#how-torchrun-spawns-workers)) prints one traceback per failing process, interleaved; scroll up to the *first* error by time, since the others are often consequences ("connection reset" on ranks whose peer died).

### Errors that don't point at the bug

Some errors appear far from their cause:

- **Asynchronous GPU errors.** CUDA kernels run asynchronously ([Concurrency](concurrency.md) explains why), so "CUDA error: an illegal memory access was encountered" is reported at some later call, often an unrelated `.item()` or `print`. Rerun with `CUDA_LAUNCH_BLOCKING=1` so each kernel finishes before the next line runs, and the traceback points at the real launch. (Untested here: no NVIDIA GPU.)
- **Import shadowing.** A file in your directory with the name of a library wins over the library. Here `torch.py` is a forgotten scratch file containing `print("my experiment")`, and `run.py` is `import torch` followed by `x = torch.randn(3)`:

```console
$ ls
run.py  torch.py
$ python run.py
my experiment
Traceback (most recent call last):
  File "/private/tmp/tkdemo/shadow/run.py", line 2, in <module>
    x = torch.randn(3)
        ^^^^^^^^^^^
AttributeError: module 'torch' has no attribute 'randn'
$ python -c "import torch; print(torch.__file__)"
my experiment
/private/tmp/tkdemo/shadow/torch.py
```

`module.__file__` is the check. The same happens with a script called `random.py`, `json.py` or `datasets.py`.

- **Killed with no traceback.** A process that just prints `Killed` (exit code 137) was killed by the OS, usually the out-of-memory killer; Python never got a chance to print anything. See [the OOM killer](linux-and-os.md#overcommit-and-the-oom-killer).

### Logs

A training or serving log is thousands of lines, mostly noise. The workflow:

- **Find the first error, not the last.** `grep -n -m1 -iE 'error|exception|traceback' train.log` gives the line number of the first one (`-m1`: stop after one match); `sed -n '1200,1260p' train.log` (print only lines 1200-1260) or `less +1200 train.log` (open the pager at line 1200) shows the context around it. Later errors are usually consequences.
- **Look at the timestamps.** The gap before a failure (a hang for 30 minutes, then a timeout) is often the clue.
- **Separate by source.** In multi-rank logs, `grep 'rank3'` or the per-rank log files. Programs prefix lines with levels (`DEBUG`, `INFO`, `WARN`, `ERROR`); `grep -v INFO` removes the routine ones.
- **Turn up verbosity when the default says nothing.** Many libraries read an environment variable for it, e.g. `NCCL_DEBUG=INFO` or `TORCH_DISTRIBUTED_DEBUG=DETAIL`; each chapter names the relevant ones.

### Searching for an error

Strip what's specific to you (paths, numbers, tensor sizes, hostnames, PIDs, i.e. process IDs) and search the stable part in quotes: `"shapes cannot be multiplied"`, not the whole line. Then, in order:

1. **The library's own source.** The message is a string in the code, and the code around it tells you exactly what condition triggers it. With the library installed, grep its directory (`python -c "import torch; print(torch.__file__)"` tells you where it is):

   ```console
   $ cd .venv/lib/python3.12/site-packages/torch
   $ grep -n "cannot be multiplied" include/ATen/native/LinearAlgebraUtils.h
   51:      ": mat1 and mat2 shapes cannot be multiplied (",
   ```

   In a cloned repository, `git grep -n "the message"` does the same.
2. **The project's GitHub issues** (search open *and* closed). Someone has usually hit it, and the issue tells you which version fixed it.
3. **The general web**, adding the library name and version.

Before you ask anyone else, collect what they'll ask for: the full traceback (not a screenshot), the exact command, the versions (`python -m pip list | grep -iE 'torch|numpy'`, `python check_env.py` from [above](#one-setup-block-for-the-whole-book)), the OS and GPU, and the smallest script that reproduces it. Producing that smallest script finds the bug yourself about half the time.



## You can now read

With this chapter behind you, the rest of the book's everyday mechanics should not get in your way:

- shell sessions in every chapter: the `$ ` convention, paths, flags, pipes, `export`, background jobs, and the **untested** / **sketched** labels;
- the setup block at the top of a chapter, and why it creates a venv in `/tmp` or uses `uv`;
- `git clone ... && make` build recipes in [The GPU software stack](../compute/gpu-software-stack.md) and [Collectives internals](../network/collectives-internals.md);
- the C benchmarks in [Computer architecture](computer-architecture.md), and the C side of the CUDA code in [the GPU execution model](../compute/accelerator/execution-model.md) and [Writing kernels](../compute/accelerator/writing-kernels.md);
- Kubernetes manifests and Slurm-adjacent YAML in [Kubernetes from zero](../orchestration/kubernetes-from-zero.md) (with its [YAML for Kubernetes](../orchestration/kubernetes-from-zero.md#yaml-for-kubernetes) section), and `pyproject.toml`/`.toml` configs in [RL infrastructure](../platform/rl-infrastructure.md);
- JSONL structured logs filtered with `jq` in [Control planes](../platform/control-planes.md), and regex-based log parsing anywhere;
- Python tracebacks and C/CUDA error messages, well enough to find the frame that matters.

Next: [Computer architecture](computer-architecture.md), which puts the "data lives at addresses" picture from [Reading C](#reading-c) onto real hardware.



## Further reading

- [The Missing Semester of Your CS Education](https://missing.csail.mit.edu/) (MIT): the shell, editors, git, data wrangling with regex, and build systems, as short lecture videos. The closest thing to this chapter at greater length.
- [uv documentation](https://docs.astral.sh/uv/) and the [Python Packaging User Guide](https://packaging.python.org/) for environments and `pyproject.toml`.
- [Pro Git](https://git-scm.com/book) (free online), chapters 1-3, for git's model of commits and branches.
- [jq manual](https://jqlang.org/manual/); [YAML 1.2 spec](https://yaml.org/spec/1.2.2/) and [The YAML document from hell](https://ruudvanasseldonk.com/2023/01/11/the-yaml-document-from-hell) for more traps; [TOML spec](https://toml.io/).
- [Python `re` HOWTO](https://docs.python.org/3/howto/regex.html); [regex101](https://regex101.com/) to test a pattern interactively against sample text.
- [GNU Make manual](https://www.gnu.org/software/make/manual/), sections 2 ("An Introduction to Makefiles") and 10 (implicit rules).
- Kernighan and Ritchie, *The C Programming Language* (2nd ed.), chapters 1, 5 (pointers and arrays) and 6 (structs). Still the shortest complete treatment.
- [Linux and the OS](linux-and-os.md) in this book, for everything the terminal section only touched.
