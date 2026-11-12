# Writing and Running Tests

ML code fails quietly. A wrong mask, a scheduler stepped in the wrong order or a seed that isn't really fixed doesn't crash anything. It produces a slightly worse model three days later. Tests are how you catch those bugs in seconds instead of in a loss curve. This chapter has two halves. **Running tests** covers every `pytest` invocation you will need: selecting a subset, rerunning failures, looping on failure, running in parallel, shuffling and repeating to expose hidden coupling, and producing reports. **Writing tests** covers how to express hardware requirements ("needs 2 GPUs"), parametrize a test, get temporary directories that clean up after themselves, capture stdout, stderr and logs, fake environment variables, seed RNGs, compare tensors and test distributed code that `pytest` can't run directly. It ends with debugging a failing test and a hack that splits a `pytest` run into separate report files for CI.

Many examples use the helper module [testing_utils.py](../testing/testing_utils.py), which lives next to this book's sources in `testing/`. Stas Bekman developed most of it while at Hugging Face (it's a trimmed relative of `transformers.testing_utils`). [What the helper module provides](#what-the-helper-module-provides) lists what is actually in it, because the module has drifted from its documentation in several places.

**Conventions.** Blocks marked **[ran]** were executed on an Apple laptop with no CUDA GPU: Python 3.14.5, PyTorch 2.15.0 (CPU), pytest 9.2.0, pytest-xdist 3.8.0, pytest-instafail 0.5.0, pytest-random-order 1.2.0, pytest-repeat 0.9.4, pytest-flakefinder 1.1.0 and parameterized 0.9.0. The output shown is real, trimmed only where marked `[...]`. Everything that needs a GPU, or a plugin not in that list, is labelled **untested**. To follow along:

```bash
python3 -m venv /tmp/test-venv && source /tmp/test-venv/bin/activate
pip install torch numpy packaging setuptools pytest parameterized \
            pytest-xdist pytest-instafail pytest-random-order pytest-repeat pytest-flakefinder
mkdir -p /tmp/testdemo && cd /tmp/testdemo
cp ~/ml-engineering/testing/testing_utils.py .    # the tests below import it from the current dir
```

`setuptools` is in the list on purpose. See [the helper module's caveats](#caveats-before-you-rely-on-it).

## Prerequisites

- [The toolkit](../foundations/toolkit.md) for virtual environments, `pip` and `pyproject.toml`.
- [Linux and the OS](../foundations/linux-and-os.md): [environment variables and PATH](../foundations/linux-and-os.md#environment-variables-and-path) (aliases, `CUDA_VISIBLE_DEVICES`), [pipes and redirection](../foundations/linux-and-os.md#pipes-and-redirection) (stdout vs stderr, block buffering), [exit codes](../foundations/linux-and-os.md#exit-codes), [scripting essentials](../foundations/linux-and-os.md#scripting-essentials) (command substitution) and [how torchrun spawns workers](../foundations/linux-and-os.md#how-torchrun-spawns-workers).
- [Concurrency](../foundations/concurrency.md): [`asyncio`](../foundations/concurrency.md#asyncio), used by the subprocess helper for distributed tests.
- [Networking: collective communication](../foundations/networking.md#collective-communication-an-intuitive-introduction) for `all_reduce`, rank and world size, and the gloo backend.
- [Training arithmetic: mixed precision internals](../training/arithmetic.md#mixed-precision-internals) for why bf16 comparisons need tolerances.
- [Fault tolerance and reproducibility: achieving determinism](../training/fault-tolerance-and-reproducibility.md#achieve-determinism-in-randomness-based-software) for the full seeding recipe this chapter links to.
- [Debugging PyTorch programs](debugging-pytorch.md) and [Debugging distributed training](debugging-distributed.md), the two previous chapters. A failing test is often the first sign of one of their bugs.

## pytest and unittest in five minutes

Two test frameworks dominate Python. This chapter uses both, often in one suite.

- **`unittest`** is in the standard library. A test is a method whose name starts with `test` on a subclass of `unittest.TestCase`. Optional `setUp()` and `tearDown()` methods run before and after every test. **Teardown** is the cleanup after a test, and when it misses something, leftover state leaks into the next test.
- **`pytest`** is a third-party test runner (`pip install pytest`). It finds files named `test_*.py` or `*_test.py`, runs every function or method in them whose name starts with `test`, and treats a failing plain `assert` as a test failure (it rewrites `assert` so the failure message shows the values involved). It also runs `unittest.TestCase` classes, which is why the two can be mixed.
- A **plugin** is an extra package (`pytest-xdist`, `pytest-instafail`, ...) that adds command-line options or behaviour to `pytest` as soon as it's installed in the same environment. You don't configure anything to load it. That's convenient, and it's also a trap, see [random order](#run-tests-in-a-random-order).
- A **node id** is a test's full name, `path/to/file.py::Class::test_method` (or `file.py::test_function` with no class). You can pass node ids on the command line to run exactly those tests.
- A **fixture** is a function that `pytest` runs to give a test a resource. The test asks for it by naming it as an argument: `def test_x(tmp_path):` receives a fresh temporary directory, and `def test_y(capsys):` receives a handle on captured stdout and stderr. `unittest.TestCase` methods can't take such arguments, so most fixtures don't work in them. That's the main reason `testing_utils.py` exists.
- **`conftest.py`** is a file that `pytest` imports automatically before collecting the tests in its directory and below. Fixtures and **hooks** defined there apply to all those tests. Hooks are functions with reserved names that `pytest` calls at fixed points of a run, such as `pytest_addoption` (register extra command-line options) and `pytest_terminal_summary` (runs when the final summary is printed).
- **Config files.** `pytest` reads its settings from the first of `pytest.ini`, `pyproject.toml` (the `[tool.pytest.ini_options]` table), `tox.ini` or `setup.cfg` (the `[tool:pytest]` section) that it finds walking up from the test paths. The directory it was found in becomes the **rootdir**, printed at the top of every run.

The minimal suite used for most of the "Running tests" examples is a `unittest`-style file with two classes:

```python
# test_optimization.py -- a tiny unittest-style suite to practise test selection on
import unittest
import torch


class OptimizationTest(unittest.TestCase):
    def _one_step(self, opt_cls, **kw):
        w = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        opt = opt_cls([w], lr=0.1, **kw)
        (w ** 2).sum().backward()
        opt.step()
        return w.detach()

    def test_adam_w(self):
        w = self._one_step(torch.optim.AdamW, weight_decay=0.0)
        # Adam's first step moves every coordinate by ~lr against the sign of the gradient
        torch.testing.assert_close(w, torch.tensor([0.9, -1.9]))

    def test_adafactor(self):
        w = self._one_step(torch.optim.Adafactor)
        self.assertTrue((w.abs() < torch.tensor([1.0, 2.0])).all())

    def test_sgd(self):
        w = self._one_step(torch.optim.SGD)
        torch.testing.assert_close(w, torch.tensor([0.8, -1.6]))


class SchedulerTest(unittest.TestCase):
    def test_constant_schedule(self):
        w = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([w], lr=0.1)
        sched = torch.optim.lr_scheduler.ConstantLR(opt, factor=1.0)
        sched.step()
        self.assertEqual(opt.param_groups[0]["lr"], 0.1)
```

`test_constant_schedule` has a deliberate mistake: it calls `sched.step()` without ever calling `opt.step()`. PyTorch warns about this, and that warning comes back in [Debugging tests](#debugging-tests).

## What the helper module provides

[testing_utils.py](../testing/testing_utils.py) is a single file with no installation step. Copy it into your `tests/` directory (or anywhere on `sys.path`) and import from it. It needs `numpy`, `packaging` and `parameterized`, and `torch` if you want the torch helpers. Here's everything in it, grouped by purpose, with the section of this chapter that uses each piece:

| Purpose | Names | Section |
|---|---|---|
| Skip unless hardware or a dependency is present | `require_torch`, `require_torch_no_gpus`, `require_torch_gpu`, `require_torch_multi_gpu`, `require_torch_non_multi_gpu`, `require_torch_up_to_2_gpus`, `require_torch_bf16`, `require_deepspeed`, `require_bnb`, `require_bnb_non_decorator()` | [Hardware and dependency requirements](#hardware-and-dependency-requirements) |
| Probe the machine | `torch_device` (`"cuda"`, `"cpu"` or `None`), `get_gpu_count()`, `is_torch_available()`, `is_torch_bf16_available()`, `is_deepspeed_available()`, `is_bnb_available()` | same |
| Parse flags from the environment | `parse_flag_from_env(key, default)` (`yes`/`no`/`1`/`0`/... to bool), `parse_int_from_env(key, default)` | [Testing with environment variables](#testing-with-environment-variables) |
| Paths, temp dirs and env for subprocesses | `TestCasePlus`, `get_tests_dir()`, `ExtendSysPath` | [Files and directories](#files-and-directories) |
| Parametrized test names | `parameterized_custom_name_func_join_params` | [Parametrization](#parametrization) |
| Capture output | `CaptureStd`, `CaptureStdout`, `CaptureStderr`, `CaptureLogger`, `apply_print_resets()` | [Capturing outputs](#capturing-outputs) |
| Fake environment variables | `mockenv(**kw)`, `mockenv_context(*remove, **update)` | [Testing with environment variables](#testing-with-environment-variables) |
| Reproducibility and tensor comparison | `set_seed()`, `torch_assert_equal()`, `torch_assert_close()` | [Getting reproducible results](#getting-reproducible-results), [Comparing tensors](#comparing-tensors) |
| Distributed tests | `execute_subprocess_async()`, `get_xdist_worker_id()`, `get_unique_port_number()` | [Distributed tests](#distributed-tests) |
| Small file helpers | `write_file()`, `read_json_file()`, `replace_str_in_file()` | [Small file helpers](#small-file-helpers) |
| Split reports | `pytest_addoption_shared()`, `pytest_terminal_summary_main()` | [A massive hack to create multiple pytest reports](#a-massive-hack-to-create-multiple-pytest-reports) |

### Caveats before you rely on it

Running the module on Python 3.14 turned up these differences from what its docstrings (and older versions of this material) say. Each one was checked:

- **There is no `require_torch_tpu`.** Older docs list it, but the module never defines it. What the module has instead, and older docs don't mention, is `require_torch_no_gpus`, `require_torch_bf16`, `require_deepspeed` and `require_bnb`.
- **`from distutils.util import strtobool` at the top.** `distutils` was removed from the standard library in Python 3.12. The import only works because `setuptools` installs a `distutils` shim. Without `setuptools`, or with the shim disabled, the whole module fails to import **[ran]**:

  ```bash
  $ SETUPTOOLS_USE_DISTUTILS=stdlib python -c "import testing_utils"
  [...]
      from distutils.util import strtobool
  ModuleNotFoundError: No module named 'distutils'
  ```
- **`execute_subprocess_async` is broken on Python 3.11 and newer.** It passes bare coroutines to `asyncio.wait()` (a `TypeError` since 3.11) and calls `asyncio.get_event_loop()` with no loop running (a `RuntimeError` on 3.14). [Distributed tests](#distributed-tests) has a working replacement.
- **`TestCasePlus` looks for a repo root containing `m4/` and `tests/`.** `m4` was the Hugging Face project this copy came from. In any other repo it silently picks the directory three levels above the test file. [Files and directories](#files-and-directories) shows the consequence and the fix. It has a `data_dir` (`tests/test_data`) and no `examples_dir`.
- **`require_torch_no_gpus` imports `torch` before checking whether torch is installed**, so on a machine without torch it raises `ImportError` instead of skipping.
- `apply_print_resets` (used by the `Capture*` classes) calls `re.sub` with a positional `count`, which emits a `DeprecationWarning` on Python 3.13 and newer. It's harmless, but it shows up in your warnings summary.

## Running tests

### Run all tests

```bash
pytest
```

With no arguments, `pytest` collects tests starting from the current directory, or from the `testpaths` set in its config file.

A useful alias (a short name for a longer command, put in `~/.bashrc`, see [Environment variables and PATH](../foundations/linux-and-os.md#environment-variables-and-path)):

```bash
alias pyt="pytest --disable-warnings --instafail -rA"
```

It tells `pytest` to:

- `--disable-warnings`: not print the warnings summary.
- `--instafail`: show each failure as it happens, not at the end (needs `pip install pytest-instafail`).
- `-rA`: add a line per test to the "short test summary info" at the end. `-r` picks which outcomes get a line and `A` means all of them (passed, failed, skipped, xfailed, xpassed, errors). The default is `-rfE`, failures and errors only.

**[ran]**

```text
$ pytest --disable-warnings --instafail -rA test_optimization.py
============================= test session starts ==============================
platform darwin -- Python 3.14.5, pytest-9.2.0, pluggy-1.7.0
Test order randomisation NOT enabled. Enable with --random-order or --random-order-bucket=<bucket_type>
rootdir: /private/tmp/absorb-testing/demo
plugins: repeat-0.9.4, instafail-0.5.0, xdist-3.8.0, random-order-1.2.0, flakefinder-1.1.0, anyio-4.16.0
collected 4 items

test_optimization.py ....                                                [100%]

==================================== PASSES ====================================
=========================== short test summary info ============================
PASSED test_optimization.py::OptimizationTest::test_adafactor
PASSED test_optimization.py::OptimizationTest::test_adam_w
PASSED test_optimization.py::OptimizationTest::test_sgd
PASSED test_optimization.py::SchedulerTest::test_constant_schedule
========================= 4 passed, 1 warning in 0.92s =========================
```

How to read this output: the header shows the Python and pytest versions, the rootdir and **every installed plugin**. Check that line first when a run behaves strangely. Each `.` is one passing test (`F` failed, `s` skipped, `x` xfailed, `X` xpassed, `E` error). The last line counts everything. The "1 warning" is still counted even though `--disable-warnings` hides its text.

### Getting the list of all tests

`--collect-only` lists the tests without running them, and `-q` prints one node id per line. **[ran]**

```text
$ pytest --collect-only -q
test_optimization.py::OptimizationTest::test_adafactor
test_optimization.py::OptimizationTest::test_adam_w
test_optimization.py::OptimizationTest::test_sgd
test_optimization.py::SchedulerTest::test_constant_schedule

4 tests collected in 0.47s
```

For one file, `pytest tests/test_optimization.py --collect-only -q`. A handy alias:

```bash
alias pytc="pytest --disable-warnings --collect-only -q"
```

### Run a specific test module

```bash
pytest tests/utils/test_logging.py
```

### Run specific tests

With `unittest`-style tests you need the name of the class that holds the test. The node id has three parts:

```bash
pytest tests/test_optimization.py::OptimizationTest::test_adam_w
#      ^ the file with tests      ^ test class     ^ test method
```

To run every test in one class, stop after the class name. You can list what that would run with `--collect-only` **[ran]**:

```text
$ pytest test_optimization.py::OptimizationTest --collect-only -q
test_optimization.py::OptimizationTest::test_adafactor
test_optimization.py::OptimizationTest::test_adam_w
test_optimization.py::OptimizationTest::test_sgd

3 tests collected in 0.42s
```

**Keyword expressions.** `-k EXPR` keeps only the tests whose node id (file, class and function name) matches `EXPR` as a case-insensitive substring. `and`, `or`, `not` and parentheses combine substrings. **[ran]**, with `-q` output trimmed to the summary line or the selected ids:

```text
$ pytest -k adam -q test_optimization.py                    # names containing "adam"
1 passed, 3 deselected in 0.90s

$ pytest -k "not adam" -q test_optimization.py              # everything except those
3 passed, 1 deselected, 1 warning in 0.81s

$ pytest -k "ada and not adam" -q --collect-only test_optimization.py
test_optimization.py::OptimizationTest::test_adafactor

$ pytest -k "test_adafactor or test_adam_w" -q --collect-only test_optimization.py
test_optimization.py::OptimizationTest::test_adafactor
test_optimization.py::OptimizationTest::test_adam_w

$ pytest -k "test and ada" -q --collect-only test_optimization.py
test_optimization.py::OptimizationTest::test_adafactor
test_optimization.py::OptimizationTest::test_adam_w
```

Use `or` when either keyword should select a test (here, to get both `test_adafactor` and `test_adam_w`), and `and` when a test must contain both.

### Run only modified tests

[pytest-picked](https://github.com/anapaulagomes/pytest-picked) runs only the tests in files that Git reports as modified but not yet committed. It's a quick way to check your change didn't break anything nearby, without running tests for code you didn't touch. **Untested here.**

```bash
pip install pytest-picked
pytest --picked
```

### Clearing state

`pytest` keeps a cache in `.pytest_cache/` between runs. The most useful thing in it is the list of tests that failed last time, which `--lf` (`--last-failed`) reruns. **[ran]**, using a suite where one test fails:

```text
$ pytest -q -n 2 test_coupled.py
[...]
1 failed, 1 passed in 0.23s
$ cat .pytest_cache/v/cache/lastfailed
{
  "test_coupled.py::test_reads_cache": true
}
$ pytest -q --lf test_coupled.py
[...]
FAILED test_coupled.py::test_reads_cache - KeyError: 'vocab_size'
1 failed, 1 deselected in 0.01s
```

On CI, and whenever isolation matters more than speed, wipe the cache at the start of the run with `--cache-clear`:

```bash
pytest --cache-clear tests
```

With the cache cleared, `--lf` has nothing to go on and runs everything (`2 passed` in the same run). `-p no:cacheprovider` goes further and disables the cache entirely. Related flags: `-x` (`--exitfirst`) stops at the first failure, `--maxfail=N` after N, and `--sw` (`--stepwise`) stops at the first failure and, on the next run, resumes from that test.

### Automatically rerun failed tests on source modification

[pytest-xdist](https://github.com/pytest-dev/pytest-xdist) has a "loop on fail" mode. It runs the suite, then waits for files to change and reruns only the failing tests, again and again, until they all pass. You don't have to restart `pytest` after each fix. Once everything passes it does a full run again.

```bash
pip install pytest-xdist
pytest -f          # or: pytest --looponfail
```

**[ran]**: start the loop with a failing test, then fix the file from another terminal:

```text
$ pytest -f -q
F                                                                        [100%]
[...]
FAILED test_loop.py::test_math - assert (2 + 2) == 5
1 failed in 0.01s
################################ LOOPONFAILING #################################
test_loop.py::test_math
############################# waiting for changes ##############################
### Watching:   /private/tmp/absorb-testing/loop
# MODIFIED /private/tmp/absorb-testing/loop/test_loop.py
.                                                                        [100%]
1 passed in 0.00s
.                                                                        [100%]
1 passed in 0.00s
```

How to read it: the first `1 passed` is the rerun of the previously failing test, and the second is the full run that follows once everything passes.

Changes are detected by watching the `looponfailroots` directories recursively. The default is the current directory. That bit me: my first attempt redirected the output to a file inside the watched directory, so every write of the log counted as a modification and the loop reran forever. Set the roots in your config file to just your package and tests. In `setup.cfg`:

```ini
[tool:pytest]
looponfailroots = transformers tests
```

or in `pytest.ini`/`tox.ini`:

```ini
[pytest]
looponfailroots = transformers tests
```

Paths are relative to the config file's directory.

The feature is on its way out. pytest-xdist 3.8's source issues a `DeprecationWarning`: "The --looponfail command line argument and looponfailroots config variable are deprecated. The loop-on-fail feature will be removed in pytest-xdist 4.0." [pytest-watcher](https://github.com/olzhasar/pytest-watcher) is the maintained alternative, which reruns `pytest` when files change (untested here). The older `pytest-watch` is still widely linked but has had no commit since 2022.

### Skip a test module

To run all test modules except a few, pass an explicit list. For example, everything except `test_modeling_*.py`:

```bash
pytest $(ls -1 tests/*py | grep -v test_modeling)
```

`$(...)` is [command substitution](../foundations/linux-and-os.md#scripting-essentials): the shell runs the inner pipeline (list the test files one per line, drop those matching `test_modeling`) and pastes its output in as `pytest`'s arguments. `pytest --ignore=PATH` and `--deselect NODEID` do the same thing without the shell trick.

### Running tests in parallel

`pytest-xdist` runs tests in N worker processes with `-n N` (`-n auto` uses one per CPU core). If your project has a `make test` target, it most likely uses `-n`. Each worker is called `gw0`, `gw1`, ... and **imports the test modules separately**, so module globals are not shared between workers. **[ran]**, `-v` to see which worker ran what:

```text
$ pytest -n 2 -v test_optimization.py test_coupled.py
created: 2/2 workers
[gw1] [ 16%] PASSED test_optimization.py::OptimizationTest::test_sgd
[gw0] [ 33%] PASSED test_optimization.py::OptimizationTest::test_adafactor
[gw1] [ 50%] PASSED test_optimization.py::SchedulerTest::test_constant_schedule
[gw0] [ 66%] PASSED test_optimization.py::OptimizationTest::test_adam_w
[gw1] [ 83%] PASSED test_coupled.py::test_fill_cache
[gw0] [100%] FAILED test_coupled.py::test_reads_cache
==================== 1 failed, 5 passed, 1 warning in 1.64s ====================
```

`test_coupled.py` passes when run serially and fails here. It holds two tests that secretly share state:

```python
# test_coupled.py -- two tests that secretly share state through a module global
CACHE = {}


def test_fill_cache():
    CACHE["vocab_size"] = 32000
    assert CACHE["vocab_size"] == 32000


def test_reads_cache():
    # passes only if test_fill_cache already ran in this process
    assert CACHE["vocab_size"] == 32000
```

`test_fill_cache` ran on `gw1` and `test_reads_cache` on `gw0`, a different process with its own empty `CACHE`. The failure is a real bug in the tests, and parallel runs exposed it.

`--dist=` controls how tests are grouped onto workers. `--dist=loadfile` puts all the tests of one file on the same worker. **[ran]**:

```text
$ pytest -n 2 -v --dist=loadfile test_optimization.py test_coupled.py
[gw1] [ 16%] PASSED test_coupled.py::test_fill_cache
[gw1] [ 33%] PASSED test_coupled.py::test_reads_cache
[gw0] [ 50%] PASSED test_optimization.py::OptimizationTest::test_adafactor
[...]
========================= 6 passed, 1 warning in 1.45s =========================
```

`loadfile` makes the failure go away, but that's a workaround, not a fix. Use it for files whose tests legitimately share an expensive setup (a model loaded once per module), not to hide coupling. Other modes include `load` (the default with `-n`: send tests to whichever worker is free), `loadscope` (group by class or module) and `loadgroup` (group by `@pytest.mark.xdist_group(name=...)`).

Under xdist the execution order is different on every run. When a parallel run fails and serial runs don't, you have coupled tests, and you need the exact sequence each worker ran to reproduce it. [pytest-replay](https://github.com/ESSS/pytest-replay) records the per-worker order and replays it, so you can cut the failing sequence down to a minimum (untested here).

### Test order and repetition

Repeat tests several times, in sequence, in random order or in sets, to find inter-dependencies and state bugs such as a teardown that misses something. Plain repetition is also worth doing for DL code, because some bugs only appear for some draws of the RNG.

#### Repeat tests

[pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder) runs every test multiple times (50 by default):

```bash
pip install pytest-flakefinder
pytest --flake-finder --flake-runs=5 tests/test_failing_test.py
```

The older advice is that it doesn't work with `-n`. In my run with flakefinder 1.1.0 and xdist 3.8.0 it did. **[ran]**:

```text
$ pytest -v --flake-finder --flake-runs=3 -n 2 test_coupled.py
[gw1] [ 16%] PASSED test_coupled.py::test_fill_cache[2]
[gw0] [ 33%] PASSED test_coupled.py::test_fill_cache[0]
[gw1] [ 50%] PASSED test_coupled.py::test_reads_cache[0]
[...]
```

That's one small run, so don't depend on it. The plugin has also had no commit since 2022, so treat it as functional but unmaintained.

[pytest-repeat](https://github.com/pytest-dev/pytest-repeat) is actively maintained, with `--count=N`. But it **cannot repeat `unittest.TestCase` tests**. They run once regardless of `--count`. **[ran]**:

```text
$ pytest -q --count=5 test_optimization.py test_coupled.py
[...]
14 passed, 1 warning in 0.84s
```

How to read it: 4 `unittest` tests ×1 + 2 plain functions ×5 = 14. Nothing in the output tells you the 4 `unittest` tests weren't repeated (version 0.9.4 printed no warning about it). So the two plugins trade off: flakefinder handles `unittest` classes but is dormant, while pytest-repeat is current but skips them silently. If your suite is built on `unittest.TestCase`, as most `transformers`-derived suites are, use flakefinder.

#### Run tests in a random order

```bash
pip install pytest-random-order
```

Random order finds coupled tests, where one test's state affects another. **Older docs say that installing the plugin is enough to shuffle every run. That's no longer true.** Since version 1.0 it's opt-in, as its header line says on every run:

```text
Test order randomisation NOT enabled. Enable with --random-order or --random-order-bucket=<bucket_type>
```

Enable it with `--random-order`, any `--random-order-bucket=...` or a `--random-order-seed=...`. It prints the seed it used. **[ran]**, on the coupled file above, trying seeds until one fails:

```text
$ for s in 1 2 3 4 5 6; do echo "seed $s:"; pytest -q -p no:cacheprovider --random-order-seed=$s test_coupled.py | tail -1; done
seed 1:
2 passed in 0.00s
seed 2:
2 passed in 0.00s
seed 3:
2 passed in 0.00s
seed 4:
1 failed, 1 passed in 0.01s
[...]
$ pytest --random-order-seed=4 test_coupled.py
============================= test session starts ==============================
platform darwin -- Python 3.14.5, pytest-9.2.0, pluggy-1.7.0
Using --random-order-bucket=module
Using --random-order-seed=4
[...]
test_coupled.py F.                                                       [100%]
[...]
E       KeyError: 'vocab_size'
```

How to read it: `F.` means the first test to run (`test_reads_cache`) failed and the second passed. When a CI run fails with random order, copy the `Using --random-order-seed=...` line from its log and rerun with that seed to get the same order.

The seed reproduces the exact order only for the exact same list of tests (or no list at all). Once you start narrowing the list by hand, the seed no longer helps. List the tests in the order they failed and turn shuffling off with `--random-order-bucket=none`. `pytest` runs node ids in the order given **[ran]**:

```text
$ pytest -q --random-order-bucket=none test_coupled.py::test_reads_cache test_coupled.py::test_fill_cache
FAILED test_coupled.py::test_reads_cache - KeyError: 'vocab_size'
1 failed, 1 passed in 0.01s
```

`--random-order-bucket` selects the unit of shuffling: `module` (the default once enabled: shuffle tests within each module and shuffle the modules), `class`, `package`, `global` (everything together) or `none`. The full details are in its [documentation](https://github.com/pytest-dev/pytest-random-order).

An alternative is [pytest-randomly](https://github.com/pytest-dev/pytest-randomly), with a similar interface but no bucket modes. Unlike pytest-random-order it **does** shuffle as soon as it's installed (and also reseeds `random`, NumPy and others before each test). `pytest -p no:randomly` turns it off for one run (untested here).

### Look and feel variations

#### pytest-sugar

[pytest-sugar](https://github.com/Teemu/pytest-sugar) adds a progress bar and shows failing tests and their assert right away. It activates on installation. **Untested here.**

```bash
pip install pytest-sugar
pytest -p no:sugar     # run without it, or uninstall it
```

`-p no:NAME` disables any plugin by name for one run. It's the general way out of "a plugin I installed changed my output."

#### Report each sub-test name and its progress

For a single test file or a group of tests, after `pip install pytest-pspec` (**untested here**):

```bash
pytest --pspec tests/test_optimization.py
```

Built-in `pytest -v` gets you most of the way: one line per node id with its result.

#### Instantly shows failed tests

[pytest-instafail](https://github.com/pytest-dev/pytest-instafail) prints failures and errors as they happen instead of at the end of the session. That matters when a suite takes an hour and the first failure tells you to stop.

```bash
pip install pytest-instafail
pytest --instafail
```

#### Color control

To turn colour off (for example, yellow on a white background is unreadable):

```bash
pytest --color=no tests/utils/test_logging.py
```

### To GPU or not to GPU

On a machine with GPUs, `CUDA_VISIBLE_DEVICES` (taught in [Environment variables and PATH](../foundations/linux-and-os.md#environment-variables-and-path)) controls which GPUs a process can see. Empty means none, which forces CPU-only mode:

```bash
CUDA_VISIBLE_DEVICES="" pytest tests/utils/test_logging.py
```

To run only on the second GPU of two (`0` and `1`):

```bash
CUDA_VISIBLE_DEVICES="1" pytest tests/utils/test_logging.py
```

This lets you run different test jobs on different GPUs of one machine at the same time. Set it on the command line, not inside a test. `testing_utils.torch_device` and the `require_*` decorators are evaluated when the module is imported, and CUDA reads the variable once when it initializes. **Untested here** (no CUDA GPU).

### Output capture

During a run, `pytest` captures everything written to `stdout` and `stderr`. If a test fails, its captured output is printed with the traceback under "Captured stdout call" and "Captured stderr call". Passing tests show their output only with `-rA` (or `-rP`).

To disable capture and see output live, use `-s` (short for `--capture=no`):

```bash
pytest -s tests/utils/test_logging.py
```

`-s` is what you want for anything that prints progress or launches subprocesses, and for `pdb`.

**JUnit XML.** CI systems read a standard XML format with per-test results. To write it:

```bash
pytest tests --junitxml=result.xml        # older docs write py.test; same program
```

**[ran]**, on a suite with one failing test:

```xml
<?xml version="1.0" encoding="utf-8"?><testsuites name="pytest tests"><testsuite name="pytest" errors="0"
failures="1" skipped="0" tests="3" time="0.321" [...]><testcase classname="test_slow" name="test_sleepy"
time="0.303"><failure message="AssertionError: deliberately failing after 0.3s&#10;assert False">def test_sleepy():
[...]</failure></testcase><testcase classname="test_coupled" name="test_fill_cache" time="0.001" />[...]
```

(The line breaks were added for this page. The file is one long line.)

### Sending test report to online pastebin service

`--pastebin` uploads test output to a public paste service, [bpa.st](https://bpa.st), and prints a URL. It's handy for sharing a failure in an issue. **Don't use it on anything private.** The upload is public, and test output routinely includes paths, hostnames and sometimes tokens taken from the environment.

To create a URL for each failure:

```bash
pytest --pastebin=failed tests/utils/test_logging.py
```

Select tests as usual, or add `-x` to stop at, and send, only the first failure. To create a URL for the whole session log:

```bash
pytest --pastebin=all tests/utils/test_logging.py
```

In pytest 9.2 the flag still exists in core, but it is deprecated ("The --pastebin option is deprecated. The functionality is now available in an external plugin package, pytest-pastebin.") Install [`pytest-pastebin`](https://github.com/pytest-dev/pytest-pastebin) to keep it. I didn't upload anything, so this is **untested here**.

## Writing tests

### Mixing pytest and unittest

Combining `pytest` and `unittest` in one suite mostly just works. Which `pytest` features work on `unittest.TestCase` classes is listed [here](https://docs.pytest.org/en/stable/how-to/unittest.html). The important limitations are that most fixtures (`tmp_path`, `capsys`, `monkeypatch`, ...) and `@pytest.mark.parametrize` don't work on `TestCase` methods. The replacements are the `parameterized` package (below) and the helpers in `testing_utils.py`. Markers such as `@pytest.mark.skip` and `xfail`, and autouse fixtures, do work.

### Hardware and dependency requirements

Some tests must run CPU-only, others on CPU or GPU, others only with several GPUs or with an optional library. `testing_utils.py` provides **skip decorators** for this. Each one marks the test as skipped, not failed, when the machine doesn't meet the requirement, so a laptop and an 8-GPU CI runner can run the same suite. These are all of them:

| Decorator | Runs when | Skip message |
|---|---|---|
| `@require_torch` | `torch` imports | test requires PyTorch |
| `@require_torch_no_gpus` | torch sees 0 GPUs | test requires an environment w/o GPUs |
| `@require_torch_gpu` | `torch.cuda.is_available()` (≥1 GPU) | test requires CUDA |
| `@require_torch_multi_gpu` | ≥2 GPUs | test requires multiple GPUs |
| `@require_torch_non_multi_gpu` | 0 or 1 GPU | test requires 0 or 1 GPU |
| `@require_torch_up_to_2_gpus` | 0, 1 or 2 GPUs | test requires 0 or 1 or 2 GPUs |
| `@require_torch_bf16` | CUDA GPU with compute capability ≥8 (Ampere+), CUDA ≥11, torch ≥1.9 | test requires CUDA hardware supporting bf16 and PyTorch >= 1.9 |
| `@require_deepspeed` | `deepspeed` is importable | test requires deepspeed |
| `@require_bnb` | `bitsandbytes` is importable | test requires bitsandbytes ... |
| `require_bnb_non_decorator()` | call it inside a test. It raises `SkipTest` if `bitsandbytes` is missing | same |

As a GPU-count table:

| n gpus | decorator |
|--------|-----------|
| `== 0` | `@require_torch_no_gpus` |
| `>= 0` | `@require_torch` |
| `>= 1` | `@require_torch_gpu` |
| `>= 2` | `@require_torch_multi_gpu` |
| `< 2`  | `@require_torch_non_multi_gpu` |
| `< 3`  | `@require_torch_up_to_2_gpus` |

There's no TPU decorator. All GPU checks use `torch.cuda`, so they count CUDA (and ROCm builds of PyTorch, which reuse the `torch.cuda` API) devices only. The decision is made **once, at import time**, from what's visible then. That's why `CUDA_VISIBLE_DEVICES` must be set before `pytest` starts.

**[ran]**, every decorator on a CPU-only laptop:

```python
# test_helpers.py (first part)
from testing_utils import (require_bnb, require_deepspeed, require_torch, require_torch_bf16,
    require_torch_gpu, require_torch_multi_gpu, require_torch_no_gpus,
    require_torch_non_multi_gpu, require_torch_up_to_2_gpus, torch_device, get_gpu_count)

@require_torch
def test_require_torch(): pass

@require_torch_no_gpus
def test_require_torch_no_gpus(): pass

@require_torch_gpu
def test_require_torch_gpu(): pass

@require_torch_multi_gpu
def test_require_torch_multi_gpu(): pass

@require_torch_non_multi_gpu
def test_require_torch_non_multi_gpu(): pass

@require_torch_up_to_2_gpus
def test_require_torch_up_to_2_gpus(): pass

@require_torch_bf16
def test_require_torch_bf16(): pass

@require_deepspeed
def test_require_deepspeed(): pass

@require_bnb
def test_require_bnb(): pass


def test_device_and_count():
    print(f"torch_device={torch_device} get_gpu_count()={get_gpu_count()}")
```

```text
$ pytest -rA test_helpers.py
[...]
PASSED test_helpers.py::test_require_torch
PASSED test_helpers.py::test_require_torch_no_gpus
PASSED test_helpers.py::test_require_torch_non_multi_gpu
PASSED test_helpers.py::test_require_torch_up_to_2_gpus
PASSED test_helpers.py::test_device_and_count
[...]
SKIPPED [1] .../_pytest/unittest.py:521: test requires CUDA
SKIPPED [1] .../_pytest/unittest.py:521: test requires multiple GPUs
SKIPPED [1] .../_pytest/unittest.py:521: test requires CUDA hardware supporting bf16 and PyTorch >= 1.9
SKIPPED [1] .../_pytest/unittest.py:521: test requires deepspeed
SKIPPED [1] .../_pytest/unittest.py:521: test requires bitsandbytes from https://github.com/facebookresearch/bitsandbytes
$ pytest -q -s test_helpers.py -k device
torch_device=cpu get_gpu_count()=0
```

How to read it: every `SKIPPED` line carries the decorator's reason. The file path in it points at pytest's internals, not your test, because the decorators use `unittest.skip`. Use `-rs` to see the reasons in any run. A suite that skips everything on CI looks exactly like a suite that passes, so check the skip count. Inside a test, `get_gpu_count()` gives the number of GPUs and `torch_device` a ready-made device string.

To run only one category, put it in the test names and select with `-k`, e.g. `pytest -sv ./tests -k "multi_gpu"`. That's the convention the module's docstrings suggest.

**Stacking.** A `require_*` decorator stacks with other decorators:

```python no-style
from testing_utils import require_torch_gpu

@require_torch_gpu
@some_other_decorator
def test_example_slow_on_gpu():
    ...
```

**Order matters with `@parameterized.expand`.** `parameterized` (next section) rewrites a test into several new tests with new names. If a `require_*` decorator sits above it, it wraps the placeholder `parameterized` leaves behind, and the generated tests are not skipped. **List `require_*` last, right above the function.** **[ran]**, on a 0-GPU machine:

```python
# test_order.py -- does a skip decorator survive @parameterized.expand?
import unittest
from parameterized import parameterized
from testing_utils import require_torch_multi_gpu  # skips: this machine has 0 GPUs


class OrderTest(unittest.TestCase):
    @parameterized.expand([(1,), (2,)])
    @require_torch_multi_gpu            # require_* listed LAST (closest to the function)
    def test_right_order(self, x):
        pass

    @require_torch_multi_gpu            # require_* listed FIRST
    @parameterized.expand([(1,), (2,)])
    def test_wrong_order(self, x):
        pass
```

```text
$ pytest -rA test_order.py
[...]
PASSED test_order.py::OrderTest::test_wrong_order_0
PASSED test_order.py::OrderTest::test_wrong_order_1
SKIPPED [2] .../unittest/case.py:163: test requires multiple GPUs
SKIPPED [1] .:0: test requires multiple GPUs
========================= 2 passed, 3 skipped in 0.42s =========================
```

How to read it: the two `test_right_order_*` cases were correctly skipped (`SKIPPED [2]`). The two `test_wrong_order_*` cases **ran and passed on a machine with no GPUs**. On a real suite they would fail with CUDA errors, or worse, pass vacuously. The stray `SKIPPED [1] .:0:` is the leftover placeholder.

With `@pytest.mark.parametrize` on plain functions (not `TestCase` methods), the order doesn't matter. Both orders gave `4 skipped` in the same experiment.

### Parametrization

Often you want to run the same test several times with different arguments. A loop inside the test would do it, but then you can't run the test for just one set of arguments, and the first failing case hides the rest. Parametrization generates a separate test per argument set.

With the [parameterized](https://pypi.org/project/parameterized/) package, which works for both `unittest` and `pytest` tests:

```python
# test_this1.py
import math
import unittest
from parameterized import parameterized


class TestMathUnitTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("negative", -1.5, -2.0),
            ("integer", 1, 1.0),
            ("large fraction", 1.6, 1),
        ]
    )
    def test_floor(self, name, input, expected):
        self.assertEqual(math.floor(input), expected)
```

This runs three times, each time with the last three arguments of `test_floor` taken from one tuple. Each sub-test gets its own node id, built from the index and the first parameter. **[ran]**:

```text
$ pytest test_this1.py --collect-only -q
test_this1.py::TestMathUnitTest::test_floor_0_negative
test_this1.py::TestMathUnitTest::test_floor_1_integer
test_this1.py::TestMathUnitTest::test_floor_2_large_fraction

$ pytest -q -k "negative or integer" test_this1.py      # just those two sets
2 passed, 1 deselected in 0.00s
$ pytest -q -k "not negative" test_this1.py             # all but negative
2 passed, 1 deselected in 0.00s
$ pytest -q test_this1.py::TestMathUnitTest::test_floor_0_negative test_this1.py::TestMathUnitTest::test_floor_1_integer
2 passed in 0.00s
```

By default `parameterized` puts only the first parameter in the name, and for non-string first parameters just the index. `testing_utils.parameterized_custom_name_func_join_params` joins all parameters into the name instead. **[ran]**:

```python
# test_names.py -- default vs custom sub-test names with parameterized
import unittest
from parameterized import parameterized
from testing_utils import parameterized_custom_name_func_join_params

PARAMS = [(0, True), (0, False), (1, True)]


class NamesTest(unittest.TestCase):
    @parameterized.expand(PARAMS)
    def test_default_names(self, num_workers, pad_dataset):
        pass

    @parameterized.expand(PARAMS, name_func=parameterized_custom_name_func_join_params)
    def test_determinism_wrt_rank(self, num_workers, pad_dataset):
        pass
```

```text
$ pytest --collect-only -q test_names.py
test_names.py::NamesTest::test_default_names_0
test_names.py::NamesTest::test_default_names_1
test_names.py::NamesTest::test_default_names_2
test_names.py::NamesTest::test_determinism_wrt_rank_0_False
test_names.py::NamesTest::test_determinism_wrt_rank_0_True
test_names.py::NamesTest::test_determinism_wrt_rank_1_True
```

Now `-k "1_True"` selects exactly the `num_workers=1, pad_dataset=True` case. (The module's docstring claims lowercase `_0_true`. The real names keep Python's `True`/`False`.)

For tests that are not `unittest` classes you can use `pytest`'s own `parametrize` marker:

```python
# test_this2.py
import math
import pytest


@pytest.mark.parametrize(
    "name, input, expected",
    [
        ("negative", -1.5, -2.0),
        ("integer", 1, 1.0),
        ("large fraction", 1.6, 1),
    ],
)
def test_floor(name, input, expected):
    assert math.floor(input) == expected
```

It names sub-tests differently, by joining all parameter values with `-` in square brackets. **[ran]**:

```text
$ pytest test_this2.py --collect-only -q
test_this2.py::test_floor[negative--1.5--2.0]
test_this2.py::test_floor[integer-1-1.0]
test_this2.py::test_floor[large fraction-1.6-1]

$ pytest -q 'test_this2.py::test_floor[negative--1.5--2.0]' 'test_this2.py::test_floor[integer-1-1.0]'
2 passed in 0.00s
```

**Quote these ids.** In zsh (the macOS default) `[...]` is a filename pattern, and unquoted ids fail before `pytest` even starts **[ran]**:

```text
$ pytest -q test_this2.py::test_floor[integer-1-1.0]
zsh:1: no matches found: test_this2.py::test_floor[integer-1-1.0]
```

`-k` works here too, as in the previous example. Pass `ids=[...]` to `parametrize` for readable names.

### Files and directories

Tests often need paths relative to the test file, for example to load a fixture file from `tests/data`. That's harder than it looks, because the test can be invoked from different working directories and can sit in sub-directories at different depths. `testing_utils.TestCasePlus`, a subclass of `unittest.TestCase`, works out the basic paths in `setUp()` and exposes them as properties:

- `pathlib.Path` objects, all fully resolved:
  - `test_file_path`: the current test file, i.e. `__file__`
  - `test_file_dir`: the directory containing the current test file
  - `tests_dir`: the `tests` directory of the test suite
  - `data_dir`: `tests/test_data`
  - `repo_root_dir`: the repository root
  - `src_dir`: where the package sub-directory lives (here, the same as `repo_root_dir`)
- the same as strings: `test_file_path_str`, `test_file_dir_str`, `tests_dir_str`, `data_dir_str`, `repo_root_dir_str`, `src_dir_str`

To use them, make your test class a subclass of `TestCasePlus`:

```python
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_something_involving_local_locations(self):
        data_dir = self.tests_dir / "fixtures/tests_samples/wmt_en_ro"

    def test_something_involving_stringified_locations(self):
        tests_dir = self.tests_dir_str      # or str(self.tests_dir)
```

**How the repo root is found, and how it goes wrong.** `setUp()` walks up 1, 2 and 3 levels from the test file, looking for a directory containing both `m4/` and `tests/`. That's the layout of the project this copy came from. If it finds none, it doesn't raise. It keeps the last candidate, three levels up. **[ran]**, in a repo `/tmp/absorb-testing/proj` laid out as `tests/test_paths.py` + `mypkg/`:

```python
# tests/test_paths.py (first test)
from testing_utils import TestCasePlus


class PathExampleTest(TestCasePlus):
    def test_show_paths(self):
        for name in ["test_file_path", "test_file_dir", "tests_dir", "data_dir", "repo_root_dir", "src_dir"]:
            print(f"{name:15} {getattr(self, name)}")
        print("PYTHONPATH     ", self.get_env()["PYTHONPATH"])
```

```text
$ pytest -q -s tests/test_paths.py::PathExampleTest::test_show_paths
test_file_path  /private/tmp/absorb-testing/proj/tests/test_paths.py
test_file_dir   /private/tmp/absorb-testing/proj/tests
tests_dir       /private/tmp/tests
data_dir        /private/tmp/tests/test_data
repo_root_dir   /private/tmp
src_dir         /private/tmp
PYTHONPATH      /private/tmp:/private/tmp/tests:
```

`repo_root_dir` is `/tmp`, which is wrong. With an (empty) `m4/` directory added to the repo, the same test prints `repo_root_dir /private/tmp/absorb-testing/proj` and `tests_dir /private/tmp/absorb-testing/proj/tests`. When you adopt `TestCasePlus`, change the `"m4"` in `setUp()` to your package directory name (and `_src_dir` to `repo_root / "src"` if you use a `src/` layout). Then assert `repo_root_dir` once in a test, so a layout change can't silently move it.

#### Temporary files and directories

Parallel tests need unique temporary files and directories, or they overwrite each other's data. Tests should also delete what they create. The `tempfile` module covers both. When you're debugging a test, though, you want the opposite: a fixed, known path whose contents you can inspect after the test, and that starts empty so a previous run's leftovers don't fool you. `TestCasePlus.get_auto_remove_tmp_dir()` handles both cases:

```python
from testing_utils import TestCasePlus


class ExamplesTests(TestCasePlus):
    def test_whatever(self):
        tmp_dir = self.get_auto_remove_tmp_dir()
```

- **Unique temp dir**, `self.get_auto_remove_tmp_dir()`: creates a fresh directory (via `tempfile.mkdtemp()`) and returns its path. It's deleted at the end of the test.
- **Dir of your choice**, `self.get_auto_remove_tmp_dir("./xxx")`: emptied before the test starts if it exists, and left intact after the test. Use it when you want to watch a specific directory and be sure earlier tests left nothing in it.
- **Override either default** with the `before` and `after` arguments:
  - `before=True`: the directory is always emptied at the start of the test.
  - `before=False`: if it already exists, existing files stay.
  - `after=True`: the directory is always deleted at the end of the test.
  - `after=False`: the directory is always left intact at the end of the test.
- `self.get_auto_remove_tmp_dir_str(...)` returns a string instead of a `Path`.

Two safety rules. Deleting is `shutil.rmtree`, the equivalent of `rm -r`, so an explicit `tmp_dir` **must be a relative path starting with `./`**. Anything else raises, so a typo can't wipe `/tmp` or your home directory. Note that `./` is relative to the current working directory, not the repo, and the check is only that prefix: `./../..` would pass, so don't rely on it beyond typos. Second, a test can register several temporary directories, and all of them are removed in `tearDown()` unless requested otherwise. If you override `tearDown()`, call `super().tearDown()` or the directories leak.

**[ran]**:

```python
# tests/test_paths.py (continued: more methods of PathExampleTest; the file also does
# `import os` and `from pathlib import Path` at the top)
    def test_auto_tmp_dir(self):
        tmp_dir = self.get_auto_remove_tmp_dir()        # unique, deleted after the test
        (tmp_dir / "ckpt.bin").write_bytes(b"\0" * 10)
        print("auto:", tmp_dir, os.listdir(tmp_dir))
        Path("/tmp/absorb-testing/last_auto_tmp").write_text(str(tmp_dir))

    def test_debug_tmp_dir(self):
        tmp_dir = self.get_auto_remove_tmp_dir("./xxx")  # fixed, emptied first, kept after
        print("debug:", tmp_dir, "contents at start:", os.listdir(tmp_dir))
        (tmp_dir / "out.txt").write_text("look at me after the test")

    def test_refuses_absolute_path(self):
        with self.assertRaises(ValueError) as cm:
            self.get_auto_remove_tmp_dir("/tmp/important")
        print(cm.exception)
```

```text
$ pytest -q -s tests/test_paths.py
auto: /var/folders/l2/lk9_ptv52zjbqdh24y4c64s00000gn/T/tmp7wvdyujq ['ckpt.bin']
.debug: /private/tmp/absorb-testing/proj/xxx contents at start: []
.`tmp_dir` can only be a relative path, i.e. `./some/path`, but received `/tmp/important`
[...]
4 passed in 0.43s
$ test -e "$(cat /tmp/absorb-testing/last_auto_tmp)" && echo yes || echo no
no
$ ls xxx
out.txt
$ pytest -q -s tests/test_paths.py::PathExampleTest::test_debug_tmp_dir | head -1
debug: /private/tmp/absorb-testing/proj/xxx contents at start: []
```

How to read it: the auto directory existed during the test and was gone afterwards (`no`). `./xxx` survived the test with `out.txt` in it, and the second run found it empty again at the start. For plain `pytest` functions, the built-in `tmp_path` fixture gives the same unique-and-cleaned-up behaviour.

#### Temporary sys.path override

`sys.path` is the list of directories Python searches on `import`. To import from another test file or a helper directory for one block, without leaving the path modified for every other test, use the `ExtendSysPath` context manager. It inserts the path at the front and removes it on exit, even on an exception:

```python
import os
from testing_utils import ExtendSysPath

bindir = os.path.abspath(os.path.dirname(__file__))
with ExtendSysPath(f"{bindir}/.."):
    from test_trainer import TrainerIntegrationCommon  # noqa
```

**[ran]** (passes), with a `helpers/shared_checks.py` next to the test:

```python
from testing_utils import ExtendSysPath, get_tests_dir

def test_extend_sys_path():
    with ExtendSysPath(os.path.join(get_tests_dir(), "helpers")):
        from shared_checks import common_check
    assert common_check() == "imported from a sibling directory"
    assert not any(p.endswith("helpers") for p in sys.path)   # removed again on exit
```

`get_tests_dir(append_path=None)` returns the absolute directory of the *calling* file (it looks at the caller's `__file__`), optionally joined with `append_path`. It's the function-level version of `TestCasePlus.test_file_dir`. The imported module stays in `sys.modules` after the block, so later imports of the same name get it from there.

### Skipping tests

Skipping is useful when you have found a bug and written a test for it, but the bug isn't fixed yet. To commit the test to the main branch, it has to be skipped (or marked as expected to fail) during `make test`. There are two mechanisms:

- A **skip** means the test is expected to pass only under some conditions, otherwise it shouldn't run at all. Examples are Windows-only tests on Linux, or tests that need an external resource that's currently unavailable (a database, a GPU).
- An **xfail** means the test is expected to fail, for example a feature not yet implemented or a bug not yet fixed. If an xfail test passes anyway, it's reported as an **xpass**.

The important difference: `skip` doesn't run the test and `xfail` does. If the buggy code leaves bad state behind that affects other tests (a hung CUDA context, a half-written file, a modified global), use `skip`, not `xfail`.

Here is every variant in one file. **[ran]**:

```python
# test_skips.py -- every skip/xfail variant from this chapter, run on one machine
import sys
import unittest
import pytest

torch = pytest.importorskip("torch", minversion="2.0")   # skip the whole file if torch is missing/too old
torch_device = "cuda" if torch.cuda.is_available() else "cpu"


def has_something():
    return False


@unittest.skip("this bug needs to be fixed")                 # unconditional, unittest style
def test_skip_unittest_style():
    pass


@pytest.mark.skip(reason="this bug needs to be fixed")       # unconditional, pytest style
def test_skip_pytest_style():
    pass


@pytest.mark.xfail
def test_xfail_still_broken():
    assert 1 + 1 == 3


@pytest.mark.xfail
def test_xfail_but_passes():           # becomes XPASS
    assert 1 + 1 == 2


@pytest.mark.xfail(strict=True)
def test_xfail_strict_but_passes():    # strict: an unexpected pass is a FAILURE
    assert 1 + 1 == 2


def test_skip_from_inside():           # decide at run time, inside the test
    if not has_something():
        pytest.skip("unsupported configuration")


def test_xfail_from_inside():
    pytest.xfail("expected to fail until bug XYZ is fixed")


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_skipif_python():
    pass


@unittest.skipIf(torch_device == "cpu", "Can't do half precision")
def test_skipif_device():
    pass


@pytest.mark.skipif(sys.platform == "darwin", reason="does not run on macOS")
class TestClass:
    def test_feature_x(self):
        pass
```

```text
$ pytest -rA test_skips.py test_flagged.py
[...]
PASSED test_skips.py::test_skipif_python
SKIPPED [1] .../_pytest/unittest.py:521: this bug needs to be fixed
SKIPPED [1] test_skips.py:19: this bug needs to be fixed
SKIPPED [1] test_skips.py:41: unsupported configuration
SKIPPED [1] .../_pytest/unittest.py:521: Can't do half precision
SKIPPED [1] test_skips.py:60: does not run on macOS
SKIPPED [1] test_flagged.py:11: --custom-flag is missing, skipping tests
SKIPPED [1] test_flagged.py:15: --custom-flag is missing, skipping tests
XFAIL test_skips.py::test_xfail_still_broken
XFAIL test_skips.py::test_xfail_from_inside - expected to fail until bug XYZ is fixed
XPASS test_skips.py::test_xfail_but_passes
FAILED test_skips.py::test_xfail_strict_but_passes - [XPASS(strict)]
========= 1 failed, 1 passed, 7 skipped, 2 xfailed, 1 xpassed in 0.41s =========
```

How to read it:

- A plain `xfail` that starts passing is only an `XPASS` line, and the run stays green. Nobody notices that the bug got fixed and the marker should go. `strict=True` turns that into a failure. Set `xfail_strict = true` in your config to make it the default.
- `pytest.importorskip("torch", minversion="2.0")` at module level skips **every test in the file** if the import fails or the version is too low (`docutils = pytest.importorskip("docutils", minversion="0.3")` is the classic example). It also returns the module.
- `torch_device` in the `skipIf` example is the same string that `testing_utils.torch_device` provides.
- A `skipif` on a class skips every test in the class. To skip every test **in a file** with one condition, assign the marker to a module-level `pytestmark` variable. **[ran]**:

  ```python
  import sys
  import pytest

  pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="does not run on macOS")
  # ... every test below is skipped on macOS (2 skipped in the run)
  ```

**Skipping a whole module unless a command-line flag is given.** Older material used `pytest.config.getoption(...)` at module level. `pytest.config` was removed in pytest 4.0, so use an autouse fixture. `scope="module"` runs the fixture once per test file, `autouse=True` applies it to every test without them asking for it, and `pytestconfig` is a built-in fixture that gives access to command-line options. The option itself has to be registered with the `pytest_addoption` hook in a `conftest.py`:

```python
# conftest.py -- registers --custom-flag so test_flagged.py can check it
def pytest_addoption(parser):
    parser.addoption("--custom-flag", action="store_true", default=False, help="run the flagged tests")
```

```python
# test_flagged.py -- the whole file is skipped unless --custom-flag is given
import pytest


@pytest.fixture(scope="module", autouse=True)
def _require_custom_flag(pytestconfig):
    if not pytestconfig.getoption("--custom-flag"):
        pytest.skip("--custom-flag is missing, skipping tests")


def test_one():
    pass


def test_two():
    pass
```

Without the flag, both tests are skipped (the two `test_flagged.py` lines above). With `pytest -q --custom-flag test_flagged.py` you get `2 passed`. This is how "slow" or "needs-network" suites are gated. Hugging Face Transformers uses an environment variable for the same job, `RUN_SLOW=1`, read by its `@slow` decorator.

More details, examples and ways to skip are in the [pytest skipping docs](https://docs.pytest.org/en/stable/how-to/skipping.html).

### Capturing outputs

#### Capturing the stdout/stderr output

To test functions that write to `stdout` or `stderr`, a plain `pytest` test can request the [`capsys` fixture](https://docs.pytest.org/en/stable/how-to/capture-stdout-stderr.html) by naming it as an argument. All examples in this section are from one file. **[ran]** (`10 passed`):

```python
# test_capture.py (part 1)
import sys


def print_to_stdout(s):
    print(s)


def print_to_stderr(s):
    sys.stderr.write(s)


def test_capsys(capsys):
    msg = "Hello"
    print_to_stdout(msg)
    print_to_stderr(msg)
    out, err = capsys.readouterr()  # consume the captured output streams
    sys.stdout.write(out)           # optional: replay the consumed streams
    sys.stderr.write(err)
    assert msg in out
    assert msg in err
```

`readouterr()` returns what has been written so far and resets the capture, so a second call only sees new output.

Error text usually arrives as an exception rather than on `stderr`. Catch it and check the message. Write the test so that "no exception at all" also fails:

```python
def raise_exception(msg):
    raise ValueError(msg)


def test_exception_message():
    msg = "Not a good value"
    try:
        raise_exception(msg)
    except Exception as e:
        assert msg in str(e), f"{msg} is not in the exception:\n{e}"
    else:
        raise AssertionError("no exception raised")
```

The idiomatic `pytest` form is `with pytest.raises(ValueError, match="Not a good value"): raise_exception(msg)`. In `unittest`, use `with self.assertRaises(ValueError) as cm:`.

Another way to capture stdout, which works anywhere (including `unittest` methods), is `contextlib.redirect_stdout`:

```python
import sys
from io import StringIO
from contextlib import redirect_stdout


def test_result_and_stdout():
    msg = "Hello"
    buffer = StringIO()
    with redirect_stdout(buffer):
        print_to_stdout(msg)
    out = buffer.getvalue()
    sys.stdout.write(out)           # optional: replay the consumed stream
    assert msg in out
```

**Carriage returns.** Progress bars redraw a line by printing `\r`, which moves the cursor back to the start of the line on a terminal. The captured buffer isn't a terminal, so it contains **every** intermediate state. **[ran]**:

```python
import re


def progress_bar_then_result():
    for pct in (10, 50, 100):
        print(f"progress {pct}%", end="\r")
    print("final message")


def test_raw_capture_with_carriage_returns(capsys):
    progress_bar_then_result()
    out, _ = capsys.readouterr()
    print(repr(out), file=sys.stderr)
    assert re.sub(r"^.*\r", "", out, flags=re.M) == "final message\n"
```

```text
----------------------------- Captured stderr call -----------------------------
'progress 10%\rprogress 50%\rprogress 100%\rfinal message\n'
```

So `out == "final message\n"` would fail. The cleanup `re.sub(r"^.*\r", "", buf, flags=re.M)` removes everything up to the last `\r` on each line, which leaves what a terminal would show. (Older docs said the `\r`s only show up under `pytest -s`. In this run they were in the capture both with and without `-s`, so always clean.)

`testing_utils` wraps that cleanup and the stream swapping into context managers that work in `unittest` and `pytest` tests alike:

```python
from testing_utils import CaptureStdout

with CaptureStdout() as cs:
    function_that_writes_to_stdout()
print(cs.out)
```

**[ran]**:

```python
from testing_utils import CaptureLogger, CaptureStd, CaptureStderr, CaptureStdout


def test_capture_stdout_helper():
    with CaptureStdout() as cs:
        progress_bar_then_result()
    assert cs.out == "final message\n", f"captured: {cs.out!r}"   # \r states already removed


def test_capture_stderr_and_both():
    with CaptureStderr() as cs:
        print("Warning: disk 91% full", file=sys.stderr)
    assert "Warning" in cs.err
    with CaptureStd(replay=False) as cs:
        print("to stdout")
        print("to stderr", file=sys.stderr)
    assert cs.out == "to stdout\n" and cs.err == "to stderr\n"
    assert cs.combined == "to stdout\nto stderr\n"
```

- `CaptureStdout` captures `stdout` into `cs.out`, with `\r` cleanup.
- `CaptureStderr` captures `stderr` into `cs.err`. There's **no** `\r` cleanup on `err`, so apply `apply_print_resets(cs.err)` yourself if your progress bars write to `stderr`, as `tqdm` does by default.
- `CaptureStd` captures both. `CaptureStd(out=False)` or `(err=False)` turns one off, but the single-stream subclasses are clearer. Capturing only the stream you're testing makes it harder to miss things.
- `cs.combined` is `out` followed by `err`. Use it when a program sends the same message to `stdout` in one environment and to `stderr` in another. Note that it's concatenated, not interleaved in time.
- By default the captured streams are **replayed** to the real stream on exit, so the output still shows up in failure reports (and live under `-s`). Pass `replay=False` to keep it quiet.
- The attributes are only valid after the `with` block ends. Before that they hold an "unfinished yet, called too early" message.

#### Capturing logger stream

To check what a `logging` logger emitted, use `CaptureLogger`. It attaches a `StreamHandler` to that logger for the duration of the block. **[ran]**:

```python
import logging


def test_capture_logger():
    logger = logging.getLogger("mylib.trainer")
    logger.setLevel(logging.INFO)          # the logger must let INFO through, or nothing is captured
    msg = "Testing 1, 2, 3"
    with CaptureLogger(logger) as cl:
        logger.info(msg)
    assert cl.out == msg + "\n"
```

With Hugging Face Transformers the equivalent is `from transformers import logging`, `logging.set_verbosity_info()`, `logger = logging.get_logger("transformers.models.bart.tokenization_bart")`, then the same `with CaptureLogger(logger) as cl:` block. Plain `pytest` tests can also use the built-in `caplog` fixture.

### Testing with environment variables

To test how code reacts to an environment variable, set it for one test only and restore it afterwards. Otherwise one test's `NCCL_DEBUG=INFO` leaks into every test after it. `mockenv` is a decorator for that, a thin wrapper around `unittest.mock.patch.dict(os.environ, ...)`. `mockenv_context` is a context manager that can also **remove** variables. **[ran]**:

```python
import os
from testing_utils import mockenv, mockenv_context


@mockenv(MYLIB_VERBOSITY="error")
def test_mockenv_decorator():
    assert os.getenv("MYLIB_VERBOSITY") == "error"


def test_env_restored_afterwards():
    assert os.getenv("MYLIB_VERBOSITY") is None


def test_mockenv_context():
    os.environ["KEEP_ME"] = "1"
    with mockenv_context("KEEP_ME", NCCL_DEBUG="INFO"):      # positional = remove, keyword = set
        assert "KEEP_ME" not in os.environ and os.environ["NCCL_DEBUG"] == "INFO"
    assert os.environ["KEEP_ME"] == "1" and "NCCL_DEBUG" not in os.environ
    del os.environ["KEEP_ME"]
```

The same decorator works on `unittest` methods:

```python
class HfArgumentParserTest(unittest.TestCase):
    @mockenv(TRANSFORMERS_VERBOSITY="error")
    def test_env_override(self):
        env_level_str = os.getenv("TRANSFORMERS_VERBOSITY", None)
```

Values must be strings, because `os.environ` only holds strings. To read boolean or integer flags the way the suite does, `parse_flag_from_env("RUN_SLOW", default=False)` accepts `yes/no/true/false/1/0/on/off` (and raises on anything else), and `parse_int_from_env("N", default=None)` raises on non-integers.

**Environment for external programs.** Some tests call an external program (a launcher, a training script) that must import your package and test helpers from the checkout, not from some installed copy. `TestCasePlus.get_env()` returns a **copy** of `os.environ` (the original stays untouched) with `PYTHONPATH` set to `src_dir`, then `tests_dir`, then whatever `PYTHONPATH` was before:

```python
from testing_utils import TestCasePlus


class EnvExampleTest(TestCasePlus):
    def test_external_prog(self):
        env = self.get_env()
        # now call the external program, passing `env` to it
```

The `test_show_paths` run above shows the result, and also that it's only as right as `repo_root_dir`. The old docs describe a version that picks `tests` or `examples` depending on where the test file lives. This copy always uses `tests`.

### Distributed tests

`pytest` can't run distributed training directly. If a test calls something like `torch.multiprocessing.spawn` or launches workers in-process, the child processes can end up thinking they are `pytest` and start running the test suite themselves, in loops. What works is a test that starts an ordinary subprocess, `torchrun` (taught in [How torchrun spawns workers](../foundations/linux-and-os.md#how-torchrun-spawns-workers)), which spawns the workers, while the test manages the I/O pipes and checks the result.

Hugging Face Transformers tests built this way include [test_trainer_distributed.py](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/trainer/test_trainer_distributed.py) and [test_deepspeed.py](https://github.com/huggingface/transformers/blob/58e3d23e97078f361a533b9ec4a6a2de674ea52a/tests/deepspeed/test_deepspeed.py). To jump straight to where the processes are launched, search them for `execute_subprocess_async`, which is defined in [testing_utils.py](../testing/testing_utils.py). Those tests need at least 2 GPUs and are gated behind `RUN_SLOW` (**untested here**):

```bash
CUDA_VISIBLE_DEVICES=0,1 RUN_SLOW=1 pytest -sv tests/test_trainer_distributed.py
```

`RUN_SLOW` is an environment variable: tests marked with Transformers' `@slow` decorator are skipped unless `RUN_SLOW=1` is set. `-s` turns off output capture, so you see the workers live, and `-v` prints one line per test with its result.

`execute_subprocess_async(cmd, env=None, stdin=None, timeout=180, quiet=False, echo=True)` runs `cmd` (a list), streams each line of the child's stdout and stderr to the terminal as it arrives, prefixed with `stdout:` or `stderr:`, and keeps a copy of both. It raises `RuntimeError` if the return code is non-zero (including all of stderr, so the test failure shows *why* the workers died), and also if the child produced no output at all, which catches a launcher that silently did nothing. Live streaming matters because a hanging distributed test with buffered output shows you nothing.

**It's broken on current Python.** On Python 3.14 **[ran]**:

```text
>           raise RuntimeError('There is no current event loop in thread %r.'
                               % threading.current_thread().name)
E           RuntimeError: There is no current event loop in thread 'MainThread'.
```

and once that is worked around, the next line fails with `TypeError: Passing coroutines is forbidden, use tasks explicitly.` (enforced since Python 3.11). Here's a drop-in replacement with the same behaviour, written with `asyncio.run` and `asyncio.gather` (see [asyncio](../foundations/concurrency.md#asyncio)). It also enforces the timeout, which the original didn't ("XXX: the timeout doesn't seem to make any difference here", says its source), and kills the child on timeout so a hung collective fails the test instead of hanging CI:

```python
# run_tee.py -- execute_subprocess_async for Python 3.11+: run cmd, stream both pipes live, keep a copy
import asyncio
import sys


async def _tee(stream, sink, pipe, label):
    while line := await stream.readline():
        line = line.decode("utf-8").rstrip()
        sink.append(line)
        print(label, line, file=pipe)


async def _run(cmd, env, timeout):
    p = await asyncio.create_subprocess_exec(*cmd, env=env,
                                             stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    out, err = [], []
    try:
        await asyncio.wait_for(asyncio.gather(_tee(p.stdout, out, sys.stdout, "stdout:"),
                                              _tee(p.stderr, err, sys.stderr, "stderr:")), timeout)
    except TimeoutError:
        p.kill()            # a hung collective must fail the test, not hang CI
        raise
    return await p.wait(), out, err


def run_and_tee(cmd, env=None, timeout=180):
    print("\nRunning:", " ".join(cmd))
    returncode, out, err = asyncio.run(_run(cmd, env, timeout))
    if returncode != 0:
        raise RuntimeError(f"'{' '.join(cmd)}' failed with returncode {returncode}\n\n"
                           "The combined stderr from workers follows:\n" + "\n".join(err))
    if not out and not err:
        raise RuntimeError(f"'{' '.join(cmd)}' produced no output.")
    return out, err
```

The script under test does one `all_reduce` on 2 CPU ranks with the gloo backend (see [collective communication](../foundations/networking.md#collective-communication-an-intuitive-introduction)):

```python
# allreduce_script.py -- the program the distributed test launches (2 CPU processes, gloo backend)
import torch
import torch.distributed as dist

dist.init_process_group("gloo")               # rank, world size, master addr/port come from torchrun's env vars
rank = dist.get_rank()
t = torch.tensor([rank + 1.0])                # rank 0 holds 1.0, rank 1 holds 2.0
dist.all_reduce(t)                            # sum over ranks -> 3.0 everywhere
print(f"rank {rank}: all_reduce result {t.item()}")
assert t.item() == 3.0
dist.destroy_process_group()
```

and the test:

```python
# test_distributed.py -- pytest launches a normal process, that process launches the workers
import sys
from pathlib import Path

from run_tee import run_and_tee
from testing_utils import get_unique_port_number


def test_allreduce_two_ranks():
    script = Path(__file__).parent / "allreduce_script.py"
    cmd = [
        sys.executable, "-m", "torch.distributed.run",   # == torchrun
        "--nproc-per-node", "2",
        "--master-port", str(get_unique_port_number()),  # distinct port per xdist worker
        str(script),
    ]
    out, err = run_and_tee(cmd)
    # ranks share one pipe, so their lines can interleave mid-line: count in the joined text
    assert "\n".join(out).count("all_reduce result 3.0") == 2, out
```

**[ran]**:

```text
$ pytest -sv test_distributed.py
[...]
Running: /private/tmp/absorb-venv/bin/python3.14 -m torch.distributed.run --nproc-per-node 2 --master-port 10999 /private/tmp/absorb-testing/demo/allreduce_script.py
stderr: Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, [...]
stdout: rank 0: all_reduce result 3.0
stdout: rank 1: all_reduce result 3.0
PASSED
```

Two things this example taught the hard way:

- **Ranks share the pipe.** My first version asserted that two separate *lines* contained the result. It failed 2 times in 5 runs. The failure message showed why: `['rank 1: all_reduce result 3.0rank 0: all_reduce result 3.0', '']`. Both workers inherit the same stdout pipe from `torchrun`, and one rank's text landed before the other's newline. Count in the joined text, have each rank write to its own file, or check only rank 0's output. With the fix: 8 of 8 runs passed.
- **Ports collide under xdist.** Two distributed tests on two xdist workers would both try to bind the default rendezvous port. `get_unique_port_number()` returns `10999 + worker index`. The index comes from `get_xdist_worker_id()`, which parses `PYTEST_XDIST_WORKER` (`gw3` → 3) and returns 0 when not under xdist.

For GPU versions, wrap the test in `@require_torch_multi_gpu` and pass `env=self.get_env()` from a `TestCasePlus` so the workers import your checkout.

### Comparing tensors

`torch_assert_equal(actual, expected)` is `torch.testing.assert_close` with `rtol=0, atol=0`: bit-exact. `torch_assert_close(actual, expected)` is `assert_close` with its dtype-dependent defaults. Both also check shape, dtype, device and layout, and both accept `msg=...` to add your own text to the failure. `assert_close` passes when $|a - b| \le \text{atol} + \text{rtol} \cdot |b|$ elementwise. The defaults for each dtype are in the [`assert_close` table](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close), e.g. `rtol=1.6e-2, atol=1e-5` for bf16. **[ran]**:

```python
import torch
from testing_utils import torch_assert_close, torch_assert_equal


def test_assert_close_tolerances():
    x = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    y = x + torch.tensor([0.0, 0.015625], dtype=torch.bfloat16)  # 1 bf16 ulp at 2.0 (7 stored mantissa bits: 2 * 2**-7)
    torch_assert_close(x, y)                       # bf16 defaults: rtol=1.6e-2, atol=1e-5
    try:
        torch_assert_equal(x, y)
    except AssertionError as e:
        print(str(e).splitlines()[0])
    else:
        raise AssertionError("expected exact comparison to fail")
```

```text
Tensor-likes are not equal!
```

A one-ulp difference (the gap between adjacent representable numbers, see [mixed precision internals](../training/arithmetic.md#mixed-precision-internals)) passes the bf16 default tolerance (relative error $0.015625/2 \approx 0.78\%$, under $1.6\%$) and fails the exact check. My first attempt added `0.0078125`, half an ulp, which rounded away to nothing in bf16, so both checks passed and the test proved nothing. Print the actual difference before trusting a tolerance test. Use exact comparison for things that must be deterministic (the same seed on the same device), and tolerances for anything that crosses devices, kernels or dtypes.

### Getting reproducible results

`set_seed(seed=42)` seeds Python's `random`, NumPy and torch (CPU and all CUDA devices). That's enough for a CPU test to be repeatable. **[ran]** (passes):

```python
from testing_utils import set_seed, torch_assert_equal

def test_set_seed():
    set_seed(42); a = torch.randn(3)
    set_seed(42); b = torch.randn(3)
    torch_assert_equal(a, b)                       # rtol=atol=0: bit-identical
```

On CUDA, seeding alone is **not** enough. Some kernels are nondeterministic (atomics, cuBLAS workspace selection), so identical seeds can still give different results. The full recipe, `enforce_reproducibility(seed)` with `torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG` and DataLoader worker seeding, is in [Achieve determinism in randomness based software](../training/fault-tolerance-and-reproducibility.md#achieve-determinism-in-randomness-based-software). Call it from the test rather than keeping a second, incomplete copy in your suite. Remember too that a test with one fixed seed checks one draw of the RNG. [Repeat it](#repeat-tests) or parametrize over a few seeds before calling it stable.

### Small file helpers

`write_file(path, content)`, `read_json_file(path)` and `replace_str_in_file(path, old, new)` cover the common "write a config, tweak one value, read it back" steps in tests that drive scripts. **[ran]** (passes):

```python
from testing_utils import read_json_file, replace_str_in_file, write_file

def test_file_helpers(tmp_path):
    f = tmp_path / "config.json"
    write_file(f, '{"lr": 3e-4, "warmup": 100}')
    replace_str_in_file(f, "100", "2000")
    assert read_json_file(f) == {"lr": 3e-4, "warmup": 2000}
```

`replace_str_in_file` replaces **every** occurrence, so pick a search string that occurs only once.

## Debugging tests

To start a debugger right where a warning is raised:

```bash
pytest tests/utils/test_logging.py -W error::UserWarning --pdb
```

`-W error::UserWarning` turns every `UserWarning` into an exception, so the test fails at the line that raises the warning. `--pdb` then opens the Python debugger `pdb` at the point of failure, with capture turned off. **[ran]**, on the scheduler warning planted in `test_optimization.py`. The `pdb` commands were piped in: `up` moves one frame up from PyTorch's `warnings.warn` into the test, and `p` prints an expression.

```text
$ printf 'up\np opt.param_groups[0]["lr"]\nq\n' | pytest test_optimization.py -W error::UserWarning --pdb
[...]
E               UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. [...]

../../absorb-venv/lib/python3.14/site-packages/torch/optim/lr_scheduler.py:268: UserWarning
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> entering PDB >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

>>>>>>>>>>>>>>>>>> PDB post_mortem (IO-capturing turned off) >>>>>>>>>>>>>>>>>>>
> /private/tmp/absorb-venv/lib/python3.14/site-packages/torch/optim/lr_scheduler.py(268)step()
-> warnings.warn(
(Pdb) > /private/tmp/absorb-testing/demo/test_optimization.py(33)test_constant_schedule()
-> sched.step()
(Pdb) 0.1
(Pdb)

=========================== short test summary info ============================
FAILED test_optimization.py::SchedulerTest::test_constant_schedule - UserWarn...
!!!!!!!!!!!!!!!!!!! _pytest.outcomes.Exit: Quitting debugger !!!!!!!!!!!!!!!!!!!
========================= 1 failed, 3 passed in 0.87s ==========================
```

How to read it: post-mortem `pdb` starts in the frame that raised, which here is inside PyTorch. One `up` lands on our line 33 (`sched.step()`), and from there the test's locals are available. The warning that was easy to ignore in the summary is now a stopped program you can inspect.

Variations: `-W error` turns every warning into an error. `--trace` opens `pdb` at the start of each selected test. Put `breakpoint()` in the test or code and run with `-s` to stop at a specific line. For a failure that only happens on one rank of a multi-process run, see the rank-aware `pdb` recipes in [Debugging distributed training](debugging-distributed.md). For inspecting tensors and tracing crashes, see [Debugging PyTorch programs](debugging-pytorch.md).

## A massive hack to create multiple pytest reports

A `pytest` run produces one stream of output with everything mixed together: failures in three traceback styles, warnings, timings, the summary. On CI it's much easier to diagnose a run, and to view or download just the part you need, when each kind of report is in its own file. All this functionality already exists inside `pytest`, but there's no supported way to send each part somewhere else. Stas Bekman proposed adding it to `pytest` and the proposal wasn't accepted, so `testing_utils.py` does it by **monkey-patching**: replacing a library's functions or attributes at runtime from the outside. At the end of the run it repeatedly points the terminal reporter's writer at a different file and calls pytest's own summary functions.

To enable it, add to `tests/conftest.py` (create it if needed):

```python
# tests/conftest.py
import pytest

def pytest_addoption(parser):
    from testing_utils import pytest_addoption_shared

    pytest_addoption_shared(parser)


def pytest_terminal_summary(terminalreporter):
    from testing_utils import pytest_terminal_summary_main

    make_reports = terminalreporter.config.getoption("--make-reports")
    if make_reports:
        pytest_terminal_summary_main(terminalreporter, id=make_reports)
```

`pytest_addoption_shared` registers `--make-reports` only once per process. That lets two `conftest.py` files (say `tests/` and `examples/`) both call it without `pytest` failing on a duplicate option. Then run with a report id:

```bash
pytest --make-reports=mytests tests
```

**[ran]**, with pytest 9.2.0 on a suite with two failures (the coupled test run alone, and a test that sleeps 0.3 s and fails):

```text
$ pytest -p no:cacheprovider --rootdir=tests --make-reports=mytests tests/test_coupled.py::test_reads_cache tests/test_optimization.py tests/test_slow.py
[...]
=================== 2 failed, 4 passed, 3 warnings in 0.73s ====================
$ ls -1 reports/mytests/
durations.txt
errors.txt
failures_line.txt
failures_long.txt
failures_short.txt
stats.txt
summary_short.txt
warnings.txt
$ wc -l reports/mytests/*
       3 reports/mytests/durations.txt
       0 reports/mytests/errors.txt
      19 reports/mytests/failures_line.txt
      19 reports/mytests/failures_long.txt
      17 reports/mytests/failures_short.txt
       1 reports/mytests/stats.txt
       7 reports/mytests/summary_short.txt
       6 reports/mytests/warnings.txt
$ cat reports/mytests/durations.txt
slowest durations
0.41s call     test_optimization.py::OptimizationTest::test_adafactor
0.30s call     test_slow.py::test_sleepy
16 durations < 0.05 secs were omitted
$ cat reports/mytests/summary_short.txt
=========================== short test summary info ============================
PASSED tests/test_optimization.py::OptimizationTest::test_adafactor
PASSED tests/test_optimization.py::OptimizationTest::test_adam_w
PASSED tests/test_optimization.py::OptimizationTest::test_sgd
PASSED tests/test_optimization.py::SchedulerTest::test_constant_schedule
FAILED tests/test_coupled.py::test_reads_cache - KeyError: 'vocab_size'
FAILED tests/test_slow.py::test_sleepy - AssertionError: deliberately failing...
$ cat reports/mytests/stats.txt
==================== 2 failed, 4 passed, 1 warning in 0.73s ====================
```

What each file holds:

| File | Contents |
|---|---|
| `durations.txt` | every setup/call/teardown phase that took ≥0.05 s, slowest first (like `--durations=0` without having to pass it) |
| `errors.txt` | errors (failures *outside* the test body, e.g. in a fixture or setup). Empty here |
| `failures_long.txt` | full tracebacks |
| `failures_short.txt` | tracebacks cut down to the last frame, under a "FAILURES SHORT STACK" header |
| `failures_line.txt` | one `path:line: Error` line per failure. In pytest 9.2 these lines come **after** the full traceback, so `grep '^/' failures_line.txt` gives the one-line list |
| `warnings.txt` | the warnings summary |
| `summary_short.txt` | the `-rA` style one-line-per-test summary |
| `stats.txt` | the final counts line |

Use a different `--make-reports=` id for each group of tests in a CI job (`--make-reports=tests_torch`, `--make-reports=examples`) so the groups don't overwrite each other's files. The code also prepares a `passes` report but never writes it, because on Hugging Face's CI it took more than 5 minutes with no terminal output and tripped the 10-minute CircleCI timeout.

**Caveats.** It uses private `_pytest` APIs (`create_terminal_writer`, `tr._tw`, `_getfailureheadline`), so a pytest release can break it. It still worked on 9.2.0 here. It also calls the terminal reporter's default methods, which other plugins can take over. pytest-instafail does that, so the failure reports may come out empty or different when `--instafail` is on. With a bare `--junitxml` you get per-test results in one file, which is more robust but not split by kind.

## You can now read

This is the last chapter of the book's main text. With it you can:

- Read and extend the test suite of any PyTorch project, including Hugging Face Transformers' `tests/` (their `testing_utils.py` is a superset of the one here), and make your own research code testable with the "overfit one batch / tiny model" checks from [Debugging PyTorch programs](debugging-pytorch.md) written as fast `pytest` tests.
- Turn the distributed bugs from [Debugging distributed training](debugging-distributed.md) into regression tests launched through `torchrun`.
- Use [Further reading](../appendix/further-reading.md) for where to go from here.

Parts of this chapter are adapted from Stas Bekman's [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering) (CC BY-SA 4.0).

## Further reading

- pytest documentation: [How to invoke pytest](https://docs.pytest.org/en/stable/how-to/usage.html), [skipping and xfail](https://docs.pytest.org/en/stable/how-to/skipping.html), [capturing stdout/stderr](https://docs.pytest.org/en/stable/how-to/capture-stdout-stderr.html), [running unittest-based tests](https://docs.pytest.org/en/stable/how-to/unittest.html), [parametrizing](https://docs.pytest.org/en/stable/how-to/parametrize.html), [deprecations and removals](https://docs.pytest.org/en/stable/deprecations.html) (including `--pastebin`).
- Plugins used in this chapter: [pytest-xdist](https://github.com/pytest-dev/pytest-xdist), [pytest-instafail](https://github.com/pytest-dev/pytest-instafail), [pytest-random-order](https://github.com/pytest-dev/pytest-random-order), [pytest-randomly](https://github.com/pytest-dev/pytest-randomly), [pytest-flakefinder](https://github.com/dropbox/pytest-flakefinder), [pytest-repeat](https://github.com/pytest-dev/pytest-repeat), [pytest-replay](https://github.com/ESSS/pytest-replay), [pytest-picked](https://github.com/anapaulagomes/pytest-picked), [pytest-watcher](https://github.com/olzhasar/pytest-watcher), [pytest-sugar](https://github.com/Teemu/pytest-sugar), [pytest-pastebin](https://github.com/pytest-dev/pytest-pastebin), and the [parameterized](https://pypi.org/project/parameterized/) package.
- [`torch.testing.assert_close`](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close) and its per-dtype tolerance table.
- Hugging Face Transformers' [testing guide](https://huggingface.co/docs/transformers/testing) and its `src/transformers/testing_utils.py`, the larger relative of the module used here.
- Python docs: [`unittest`](https://docs.python.org/3/library/unittest.html), [`unittest.mock`](https://docs.python.org/3/library/unittest.mock.html) (`patch.dict` is what `mockenv` wraps), [`asyncio` subprocesses](https://docs.python.org/3/library/asyncio-subprocess.html).
