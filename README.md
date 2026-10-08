# Deploying a 176B Mixture-of-Experts Model on a 12 GB Consumer GPU as a Maintenance-Script Sub-Agent: A Test-Driven Field Study

**Author:** yifeifelix (deployment, operation and decisions), with Claude acting as orchestrator and reviewer; execution was delegated to sub-agents.
**Period of study:** 7–8 October 2026
**Platform:** NVIDIA RTX 4070 SUPER (12 GB), AMD Ryzen 5 5600X, 64 GB DDR4, Windows 11
**Software:** Strata v0.1.40.3; Swift-1.5 Qwen3.8-Flash-Next, IQ3_XXS quantisation
**Anonymisation:** user names, host names, MAC addresses and API keys have been removed. Private network addresses are written as `<PC_IP>` and `<MAC_IP>`.

---

## Abstract

This report documents an end-to-end deployment of a large language model on commodity hardware. We installed Qwen3.8-Flash-Next, a Mixture-of-Experts (MoE) model with about 176 billion parameters in total and about 6 billion active per token, on an ordinary Windows desktop with 12 GB of graphics memory and 64 GB of system memory. The aim was to use it as a *sub-agent* that writes PowerShell and bash maintenance scripts. A laptop on the same local network was to reach it through an OpenAI-compatible interface.

The deployment ran in five stages: preparation, installation and first start, network exposure with automatic start-up, client integration, and capability acceptance. It followed a strict test-driven discipline. For every step we first wrote acceptance tests and observed them fail. We then implemented the step and advanced only once the tests passed. An orchestrating model planned the work and reviewed every result. Sub-agents executed individual tasks. A human took every decision that involved a material problem or a privileged change.

**The engineering outcome was successful.** The service starts automatically, authenticates callers, is reachable only from the local subnet, and is supervised by a health check with self-healing. It reached a decode rate of **52.7 tokens per second**, against a prior estimate of 25–40. Every stage passed its acceptance suite, including a verification after a real reboot (10/10).

**The capability outcome fell short of the pre-registered threshold.** On a ten-item script-writing examination the best score was **11 out of 20**, against a pass mark of 14. We compared four configurations in controlled runs: reasoning disabled, reasoning enabled at the template's default effort, reasoning at low effort, and reasoning disabled with a domain-specific pitfall checklist. With reasoning enabled at the default effort (`xhigh`), the model entered reasoning loops and produced no answers. Neither low-effort reasoning nor the checklist gave a measurable improvement.

We analyse eleven classes of problem. For each we give the symptom, the diagnostic procedure, the root cause, the remedy and the transferable lesson. Three are the most instructive: a graphics-memory reserve that was too small and caused `cublasCreate` to fail; a chat template whose default reasoning effort produces non-terminating reasoning; and defects in the test code itself that produced false failures. On the evidence collected, we reposition the local model as a **first-draft generator** placed behind four quality gates, and we set out a staged plan for further work.

**Keywords:** local large language models; Mixture-of-Experts offloading; Strata; Qwen3.8-Flash-Next; test-driven deployment; sub-agents; systems administration; PowerShell

---

## 1 Introduction

### 1.1 Motivation

Routine maintenance of personal computing equipment involves a steady stream of small scripting tasks: disk inspection, backups, scheduled tasks, log housekeeping and cache accounting. These tasks are well bounded and their outcomes are easy to verify, so in principle they are well suited to a local language model. Delegating them locally would avoid the recurring cost of hosted models, keep data within the local network, and make help available on demand.

In late September 2026 the community released the Strata inference engine. Strata keeps the expert weights of an MoE model in system memory and uses the GPU only as a cache for the experts that are used most often. For the first time, this made the 176B Qwen3.8-Flash-Next usable on a single 12–24 GB consumer GPU. Community reports cite roughly 32 tokens per second on an RTX 3060 (12 GB) and more than 60 on an RTX 3090 (24 GB). A desktop machine could therefore plausibly serve as a local model server.

### 1.2 Objectives

1. Turn the desktop into a local-network inference server with an OpenAI-compatible interface, authentication, subnet-only exposure, automatic start-up and self-healing.
2. Let a laptop on the same network delegate scripting tasks through a command-line client, `askl`.
3. Measure the model's ability to write scripts on a ten-item examination with a pass mark of 14/20, set in advance.
4. Advance no stage without a passing acceptance test.

### 1.3 Contributions

- A complete, reproducible record of deploying Strata on a 12 GB GPU with 64 GB of system memory, including the faults met and how they were fixed.
- A small evaluation method for the generation of maintenance scripts, with comparative data for four reasoning configurations.
- A problem analysis structured as *symptom → diagnosis → root cause → remedy → lesson*.
- A repositioning of the model's role that rests on measurement rather than expectation.

---

## 2 Background

### 2.1 Strata and MoE expert offloading

A Mixture-of-Experts model activates only a few expert sub-networks for each generated token. Qwen3.8-Flash-Next has 176B parameters in total but activates about 6B per token. Strata exploits this sparsity as follows:

- The **bulk of the weights stays in system memory**. In our configuration this is about 40 GiB.
- The **GPU** holds an *expert cache*, pre-filled by routing frequency, and computes the dense components such as attention.
- **Experts missing from the cache** are either streamed to the GPU over PCIe or computed by a pool of CPU threads.
- **Multi-Token Prediction (MTP)** speculative decoding uses a lightweight draft head to propose several tokens, which the main model then verifies. When most proposals are accepted, throughput rises substantially.

The price of this design is that the engine is **single-stream**. It serves one request at a time and queues the others in first-in, first-out order. Community measurements show that with N concurrent clients the aggregate throughput stays roughly constant, so each client receives about 1/N of the single-stream rate.

### 2.2 Quantisation tiers

| Tier | Memory + graphics memory required | Download size | Quality (community description) |
| --- | --- | --- | --- |
| Q2_0 | 37.6 GB | 66.4 GB | good |
| IQ2_XS | 39.2 GB | 68.0 GB | better (officially recommended) |
| IQ3_XXS | 47.0 GB | 75.8 GB | very good, close to Q4 |
| IQ3_S | 54.8 GB | 83.6 GB | best |

We chose IQ3_XXS. Script generation favours quality, and 64 + 12 = 76 GB is enough to hold this tier.

### 2.3 Prior community knowledge

Before deployment we consulted a curated local corpus: about 780 forum threads, distilled into 23 thematic runbooks. The following points were relevant:

- Below 96 GB of system memory, Strata switches to a *low-RAM* tier, and prompt processing slows by an estimated 17–42%.
- The default MTP window causes a fall in decode speed at 32K tokens of context. Setting `--mtp-window 65536` reportedly removes it; the evidence came from AMD hardware.
- Versions earlier than v0.1.38 lack Host and Origin validation. Without an API key, a malicious web page could therefore invoke the local service across sites.
- Upgrades overwrite configuration files, so those files must be backed up first.
- Local models work best as single-task sub-agents with verifiable pass criteria. Given one concrete error, they tend to repair it. Given several problems at once, they tend to loop.

As later sections show, some of this guidance proved valuable and some had been overtaken by version drift.

---

## 3 System Design

### 3.1 Architecture

```
┌────────── Laptop (client) ───────────┐        ┌────────────── Desktop (inference server) ──────────────┐
│ Claude Code (orchestration, review)  │        │ Scheduled task Strata-Server (at logon, highest level) │
│   └─ askl (bash)                     │  HTTP  │   └─ start-strata.ps1 (kill stale processes; respawn)  │
│       ├─ key read from Keychain      │ ─────▶ │       └─ serve/server.py :8080 (OpenAI-compatible)     │
│       ├─ queues on /status busy      │  LAN   │           └─ strata.exe engine                         │
│       └─ reasoning off + checklist   │        │ Scheduled task Strata-HealthCheck (every 10 minutes)   │
└──────────────────────────────────────┘        │ Firewall: port 8080 open to Private + LocalSubnet only │
                                                └────────────────────────────────────────────────────────┘
```

**Division of labour.** A strong model plans, reviews and gives final approval. The local model receives single, well-defined tasks. Nothing it produces runs before review.

### 3.2 Hardware and storage allocation

| Component | Specification | Note |
| --- | --- | --- |
| GPU | RTX 4070 SUPER, 12 GB, driver 617.42 | also drives the desktop display |
| CPU | Ryzen 5 5600X (6 cores, AVX2) | — |
| System memory | 64 GB | below the 96 GB low-RAM threshold |
| Storage | C: Crucial P3 2 TB (NVMe, PCIe 3.0); E: Crucial P3 Plus 2 TB (NVMe, PCIe 4.0); D: 4 TB hard disk | — |

The choice of storage changed twice, and the history is worth recording.

1. The first draft placed everything on `D:`. Checking the hardware showed that **D: is a mechanical hard disk**. Under the low-RAM tier, expert weights may have to be read from storage at run time, which would make a hard disk unusable.
2. The installation target moved to the fastest drive (E:, PCIe 4.0).
3. For capacity and ease of management, the owner finally chose C: (PCIe 3.0, 671 GB free). This accepted slightly slower cold starts and shared I/O with the operating system. The measured load rate was 1.92 GiB/s, which was acceptable.

**Lesson.** Check the physical model and media type behind each drive letter with `Get-PhysicalDisk` before choosing an installation path. Do not rely on recollection.

### 3.3 Test-driven deployment

Every stage followed the same cycle:

1. **Write the acceptance tests first**, using Pester 6 on Windows and bats on macOS, so that they state what "done" means.
2. **Run them and confirm that they fail** (red), and keep the output as evidence.
3. **Implement the step.**
4. **Run the tests again until they all pass** (green), and keep that output too.
5. **Never advance on red.**

The cycle paid off in two concrete ways.

- **Tests made acceptance criteria precise.** "Expose the service to the local network" became checkable assertions: HTTP 401 without a key; HTTP 200 with a key; the LAN address answers; the firewall rule is limited to the Private profile and the local subnet; the network profile is Private.
- **Tests exposed apparent successes that were in fact failures.** In one case a privileged script that the operator believed had run had only been previewed with `-WhatIf` (§5.6).

### 3.4 Separating orchestration from execution

The orchestrator delegated each PowerShell or bash task to a sub-agent, together with exact acceptance criteria, and received an evidence-bearing report in return. The orchestrator reviewed every report and every privileged script itself. When a material problem arose, it stopped and referred the decision to the human.

Every privileged change was written as an **idempotent script that supports `-WhatIf`**. The operator ran these scripts in an elevated terminal, and tests then verified the outcome. The changes covered the firewall, the network profile, power settings, scheduled tasks and the antivirus exclusion.

### 3.5 Security model

- **Network exposure.** The service listens on `0.0.0.0:8080`. The firewall admits only the local subnet on the Private profile, and the router forwards no ports.
- **API key generation.** The key comes from a cryptographic random-number generator (24 bytes, hex-encoded). The access-control list on the key file is restricted to the owner, SYSTEM and Administrators.
- **Client-side storage.** On the laptop the key is kept in the macOS Keychain. It is passed to `curl` via `-H @<(...)`, so it never appears in the process arguments.
- **Model output.** Scripts produced by the local model are **never executed directly**. They must first pass the four gates described in §8.

---

## 4 Implementation

### 4.1 Stage 0: Preparation

The first run of the pre-flight suite exposed two failures: PowerShell 7 was absent, and only Python 3.11 was installed. It also exposed two latent risks: the Wi-Fi profile was Public, and the session was not elevated.

With the owner's consent, we installed the following:

| Component | Version | Source |
| --- | --- | --- |
| PowerShell | 7.6.6 | winget (`--scope user`, no elevation) |
| Python | 3.12.10 (installed alongside 3.11) | winget (`--scope user`) |
| Pester | 6.2.0 | PowerShell Gallery |
| PSScriptAnalyzer | 1.25.0 | PowerShell Gallery |

The suite then passed (8/8).

### 4.2 Stage 1: Installation and first start

**Stage 1a: clone and audit.** The repository was cloned but nothing in it was run. A static audit of the installer recorded:

- every interactive prompt and its non-interactive flag;
- every download host;
- every path written;
- every operation needing elevation;
- any suspicious construct.

The main installation path needed no elevation, and no remote telemetry was found. Engine binaries are checked with SHA-256, although the hashes come from the same repository as the binaries, so the check is of limited independence. Two discrepancies with our plan came to light: the launcher is named `START-HERE.bat` (with a hyphen), and the `--no-low-ram` flag does not exist.

**Stage 1b: installation.** We created the virtual environment by hand with `py -3.12` to pin the interpreter, because the installer prefers `py -3`, which can resolve to a newer version. We then ran the installer non-interactively:

```bat
START-HERE.bat --yes --no-start --no-browser --family swift --model IQ3_XXS ^
  --context 131072 --vision no --kv int8 --port 8080 --vram-reserve-mib 1500 ^
  --data-dir C:\AI\Strata-data
```

About 82 GB was downloaded: the 76 GB model, plus the MTP draft layer, the engine and the CUDA runtime libraries. The installer chose to stream the KV cache to system memory (1.8 GB), which leaves graphics memory free for the expert cache. Stage 1b passed (20/20).

**Stage 1c: first start.** The first start failed for lack of graphics memory (§5.4). After the reserve was raised from 1500 MiB to 3072 MiB, the suite passed (6/6):

| Metric | Measured |
| --- | --- |
| Decode rate (engine timing) | 52.7 tok/s |
| Decode rate (end-to-end, incl. HTTP and prefill) | 40.6 tok/s |
| MTP draft acceptance | 82% (65/79) |
| KV reads served from graphics memory | 98.6% |
| Expert-cache hit rate | 18.8% → 45.7% (still warming) |

### 4.3 Stage 2: Network exposure and automatic start-up

The orchestrator's side produced four items:

- the API key;
- the service configuration (`host 0.0.0.0`, `api_key`, and `model_name local-flash-next`);
- a self-healing launcher, which removes stale processes before each start and respawns the server 30 seconds after any exit;
- a privileged script for the network profile, firewall rule, power settings, scheduled task and optional antivirus exclusion.

Once the operator had run the privileged script, the suite passed (19/19).

**Reboot verification.** We wrote the reboot suite before rebooting. Its four post-boot assertions failed beforehand, as expected. After a real reboot, the scheduled task had started the service within 40 seconds of boot, and the model loaded. Two assertions failed on the first run, both because of defects in the tests themselves (§5.7). After correction the suite passed (10/10).

### 4.4 Stage 3: Client integration

We wrote the laptop's work as a self-contained brief and gave it to a separate Claude Code instance on the laptop, under the same test-driven rules. The operator, not the model, stored the key in the Keychain.

The bats suite passed (10/10). It covered connectivity, the 401/200 behaviour, fenced-code output, input from a pipe, exit code 2 when the server is offline, exit code 4 when the server is busy, and absence of key leakage. A defect in the test itself, a negative array index that bash 3.2 does not support, was found and corrected.

### 4.5 Operational hardening: busy-aware queuing and health supervision

During the evaluation it became clear that a single long request can monopolise the single-stream engine for minutes. Two measures were therefore added.

1. **`askl` checks `/status` before sending.** If the `busy` flag is set, it waits and re-checks, for at most 600 seconds by default. It then gives up with exit code 4 rather than letting requests accumulate.
2. **A health check, `strata-health.ps1`,** runs every ten minutes as a scheduled task. Its design is described in §7. Its acceptance suite passed (9/9). A manual trigger logged `healthy` and left the service undisturbed.

---

## 5 Problem Analysis

Each problem is set out as *symptom → diagnosis → root cause → remedy → lesson*.

### 5.1 Missing toolchain and a damaged PATH

- **Symptom.** `pwsh` was reported missing. After installation, neither `pwsh` nor `winget` could be resolved.
- **Diagnosis.** The user-level `PATH` in the registry lacked `%LOCALAPPDATA%\Microsoft\WindowsApps`. That is the only place where an MSIX-packaged PowerShell exposes its execution alias.
- **Root cause.** The user `PATH` on this machine had been edited at some earlier point, and the WindowsApps entry had been lost.
- **Remedy.** We left `PATH` untouched and called every tool by its full path, including from the scheduled task. A real reboot confirmed that Task Scheduler can start PowerShell through the alias.
- **Lesson.** Launching an MSIX alias from Task Scheduler works, but it should be confirmed with a genuine reboot rather than assumed.

### 5.2 Version drift between documentation and code

- **Symptoms.**
  - `START_HERE.bat` was "not recognised".
  - `--no-low-ram` could not be found anywhere in the source.
  - Launching the installer with `cmd /c` from a PowerShell session failed to find the file.
- **Diagnosis.** A line-by-line audit of the source recovered the actual file name, the actual flag table (`setup.py:4515–4612`) and the actual low-memory options. `cmd` failed because PowerShell's `Set-Location` is not inherited by child processes as their working directory.
- **Root cause.** Strata moved from v0.1.28 to v0.1.40 in about ten days. The forum threads and our corpus described earlier builds or a community fork; `--no-low-ram` originated in the NVFP4 fork.
- **Remedy.** We treated the current source as authoritative:
  - the launcher is `START-HERE.bat`;
  - the low-memory control is `--low-ram auto|on|off|resident|mmap`;
  - the installer is invoked as `cmd /c "cd /d <dir> && <dir>\START-HERE.bat ..."`.
- **Lesson.** For a fast-moving project, audit the source before deploying. An audit costs far less than repeated trial and error against stale instructions.

### 5.3 Insufficient system-memory headroom

- **Symptom.** The engine warned: `the expert arena needs 39.97 GiB of RAM but 37.20 GiB is available (2.77 GiB short)`.
- **Diagnosis.** Inspecting process working sets showed that browsers and other applications were holding a substantial share of memory at start-up.
- **Root cause.** The resident expert arena for IQ3_XXS needs about 40 GiB, and about 43 GB with the KV cache and the pack included. That leaves little margin on a 64 GB machine.
- **Remedy.** Close heavy applications before starting, so that at least 48 GB is free. On the second start, 50.9 GB was free.
- **Lesson.** On 64 GB, IQ3_XXS fits, but without much room. The alternatives are a disciplined start-up routine, the IQ2_XS tier, or memory-mapped experts via `--low-ram mmap`, which is slower and which we did not measure.

### 5.4 Critical fault: an insufficient graphics-memory reserve prevents cuBLAS initialisation

- **Symptom.** The model loaded and the engine reported `session is up`. It then terminated:

  ```
  strata serve: the prompt path borrows 2055 CUDA0 cache slots (3.37 GiB)
  strata serve: prefill gemm: cublasCreate: cuBLAS status 1
  RuntimeError: the engine exited before it was ready
  ```

- **Diagnosis.** Hypotheses were tested in order of cost, cheapest first.
  1. *Missing libraries?* `cublas64_13.dll`, `cublasLt64_13.dll` and `cudart64_13.dll` were present in the virtual environment. The configuration's `lib_dirs` pointed at them, and driver 617 supports CUDA 13. **Rejected.**
  2. *Graphics-memory exhaustion?* Adding up the budget from the log:
     - automatic sizing found 5.40 GiB free and, after the 1500 MiB reserve, arrived at **1,731** cache slots;
     - the expert profile then allocated **2,286** slots (3.75 GiB);
     - on top of that came a 32K-token resident KV region, an 881 MiB MTP head and a further 3.37 GiB borrowed for prompt processing.

     By the time the cuBLAS handle was created, which itself needs device memory, nothing was left. `CUBLAS_STATUS_NOT_INITIALIZED` (status 1) is also what cuBLAS returns when its allocation fails.
  3. *Precedent?* Searching the Strata repository for `cublasCreate` found a comment in a community benchmark driver. It describes the same failure: without an explicit reserve, the cache sizes itself against almost all free memory and `cublasCreate` then fails to allocate. The remedy used there was a reserve of 3072 MiB.
- **Root cause.** The expert cache was sized from the routing profile rather than from the space left after the reserve. Too little memory remained for the cuBLAS workspace. The 500–600 MiB used by the desktop display on the same card made the shortage worse.
- **Remedy.** A single variable was changed: `--vram-reserve-mib` from 1500 to 3072. The suite passed (6/6), and about 2.5 GB of graphics memory remained free at run time.
- **Method.** Only one variable was changed at a time. Further steps had been prepared in advance: `--draft-vocab en`, saving about 215 MiB, and a smaller `--kv-resident`. They were not needed.
- **Lessons.**
  1. On a 12 GB card that also drives a display, reserve at least 3 GB.
  2. When `cublasCreate` fails, add up the memory budget from the log first; the shortfall is usually obvious.
  3. Search the project's own benchmarks and issue history for the exact error string before experimenting.

### 5.5 False failure: locale-dependent command output

- **Symptom.** A power-settings assertion failed. It expected the text `Current AC Power Setting Index: 0x00000000`.
- **Diagnosis.** Running `powercfg /q` by hand showed localised (Chinese) output.
- **Root cause.** The test assumed English output.
- **Remedy.** The pattern was extended to accept both the English and the localised label. The test file was saved as UTF-8 with a byte-order mark so that Windows PowerShell 5.1 decodes it correctly.
- **Lesson.** Command output varies with the system locale. Use structured data, such as CIM classes and object properties, wherever possible instead of parsing text.

### 5.6 Preview mistaken for execution

- **Symptom.** The operator reported the privileged script as complete. The tests nevertheless showed no change to the network profile, firewall, scheduled task or power settings.
- **Diagnosis.** The output the operator pasted consisted entirely of `What if:` lines, and its summary read "would …".
- **Root cause.** Only the `-WhatIf` preview had been run.
- **Remedy.** The operator ran the command without `-WhatIf`, having been told what successful output looks like: "set to Private", "created", "registered".
- **Lesson.** Tests, not verbal confirmations, are the source of truth. Instructions to an operator should state what successful output looks like.

### 5.7 False failures caused by defects in the tests

The first post-reboot run reported two failures although the service was demonstrably healthy.

1. **Log timestamp assertion.**
   - *Cause:* `$lines = Get-Content … | Where-Object {…}` returns a scalar string, not an array, when exactly one line matches. `$lines[-1]` therefore returned the **last character** of that line, not the last line. This is a classic PowerShell pitfall: a pipeline that yields one object unwraps it.
   - *Remedy:* force an array with `@(...)`.
2. **"Exactly one server instance" assertion**, which failed twice for different reasons.
   - *Before the reboot it counted two processes.* On Windows a virtual environment's `python.exe` is a launcher that spawns the base interpreter, and the two processes share the same command line. *Remedy:* count only root processes, that is, those whose parent is not in the matched set.
   - *After the reboot it counted none.* The server had been started by a task running at the highest privilege level, and a non-elevated CIM query cannot read the `CommandLine` of such processes; it returned null. *Remedy:* assert that exactly one process listens on port 8080 and that exactly one `strata.exe` exists. Neither check needs elevation.
3. **On the laptop**, bash 3.2 does not support the negative index `${lines[-1]}`.

**Lesson.** When a test fails, ask first whether the subject or the test is at fault. Answer the question by checking the subject's real state independently, for example with `curl /health` or by inspecting the process tree. If the test is at fault, correct it and record what was changed and why. **A test must never be weakened merely to make it pass.** The revised assertion must still check the same property.

### 5.8 Deviations by sub-agents

- **A skipped red step.** On one occasion a sub-agent changed the code before writing the tests, so no red evidence existed. The orchestrator reported this openly. Later briefs required the red output to be saved before any implementation, and that requirement was then met.
- **A tool invoked incorrectly.** Piping text into `pwsh -File` does not bind standard input to a pipeline parameter. Multi-line text passed as a command-line argument is split. One generation round was wasted on this before the sub-agent found the cause, re-extracted the code and archived the invalid data separately. The correct form is `pwsh -Command "Get-Content -Raw f | & script.ps1 -OutFile out"`.
- **Lesson.** A brief should state what is forbidden and what evidence must be returned, and the orchestrator should read every report's section on deviations.

### 5.9 Interception by a safety hook

- **Symptom.** A `Copy-Item … -Force` onto a file in the root of drive `D:` was intercepted by a path-protection hook. The command did not execute at all.
- **Diagnosis.** The target was checked and found unchanged, with no partial write.
- **Remedy.** The file was overwritten with `[IO.File]::Copy(src, dst, $true)`.
- **Lesson.** An intercepted command has done nothing. Verify the state, then reach the same outcome by another route.

### 5.10 Side effects in a model-generated dry run

- **Symptom.** The script generated for examination item 4 (a robocopy backup) ran in list-only mode (`/L`) by default, yet it created the directory `D:\Backup\knowledge`.
- **Root cause.** The directory creation sat inside a `ShouldProcess` branch. Without `-WhatIf`, `ShouldProcess` returns true, so the dry run and the real run followed the same path.
- **Remedy.**
  - The empty directory was removed non-recursively, and the removal was reported.
  - The danger screen gained a rule against directory creation on `D:` outside the real-run branch.
  - The pitfall checklist gained a rule that a dry run must have no side effects.
  - A later script of the same kind was stopped before it ran.
- **Lesson.** The meaning of "dry run" must be defined explicitly, and tests should confirm that it has no side effects, for example by checking target paths before and after the run.

### 5.11 Non-terminating reasoning

- **Symptom.** With `enable_thinking` set, seven of eight items used the whole 16,384-token budget on reasoning and **produced no answer**. The reasoning traces ran to 63,000–67,000 characters, and `content` was null. Raising the budget to 49,152 tokens for item 1 still gave nothing after 13 minutes and about 132,000 characters.
- **Diagnosis.**
  1. *Insufficient budget?* No. Tripling the budget gave no answer, so the reasoning was failing to stop rather than running short.
  2. *Contrast case.* A trivial prompt (print the date) finished in 19 seconds with reasoning enabled. The fault therefore appeared only when a long system prompt met a complex task.
  3. *Source inspection.* Strata's chat template (`serve/chat_template.jinja:59`) contains `reasoning_effort|default('xhigh')`. Setting `enable_thinking` without naming an effort level therefore selects the **highest** level.
  4. *Community evidence.* Reports on Qwen3.8 note that `xhigh` loops in about 6% of cases and recommend medium or low effort for agentic use. The levels differ only in the text injected into the system prompt; they do not set different token budgets.
- **Controlled comparison.** We then used `reasoning_effort=low` together with a numerical cap ("think in at most about 30 words"), as the community recommends. Each item took 8–32 seconds and nothing looped. The score on the three hardest items, however, matched the reasoning-disabled baseline (1/6).
- **Root cause.** The `xhigh` instruction asks the model to verify its assumptions and consider alternatives, and on complex specifications the model fell into repeated verification. Low effort, conversely, is not enough to make up for the model's gaps in API knowledge.
- **Lessons.**
  1. Before enabling reasoning, check the template's default effort level.
  2. In automated pipelines, always set `max_tokens` and a request timeout, and report a truncated reasoning trace as a distinct outcome; we use exit code 3.
  3. On a single-stream engine, one looping request blocks every other client for minutes. Reasoning must therefore be combined with busy-aware queueing and time limits.

---

## 6 Capability Evaluation

### 6.1 Method

- **Items.** The examination had ten maintenance tasks.
  - Eight in PowerShell 7 on the desktop: disk health, port ownership, large directories, incremental robocopy backup, cache accounting, scheduled tasks, GPU sampling and a service health check.
  - Two in bash on the laptop: stale large files and a LaunchAgent that checks availability.
  - A further three new PowerShell items, written after the first round, tested generalisation: stopped automatic services, the 20 largest files, and reboot status with update history.
- **Protocol.** For each item:
  1. one generation at temperature 0.2;
  2. extraction of the fenced code;
  3. static analysis (PSScriptAnalyzer or shellcheck);
  4. a danger screen;
  5. review by the orchestrator;
  6. a sandboxed run with fixed arguments;
  7. comparison against measurable acceptance facts set in advance. One example is that the reported size of `C:\AI` must lie within 5% of its true recursive size; another is that the npm cache must be reported as about 13.4 GB.
- **Scoring.**
  - 2 points if the first run passes.
  - 1 point if the script passes after a single repair round. The repair prompt names **exactly one** concrete problem: a verbatim error line or a single factual discrepancy.
  - 0 otherwise.
- **Lenient and strict scoring.** Strict scoring also counts static-analysis warnings as a failure. Figures are lenient unless stated otherwise.

### 6.2 Results

| Configuration | Desktop items (8) | New items (3) | Laptop items (2) | Generation time per item |
| --- | --- | --- | --- | --- |
| Reasoning disabled | 9/16 (strict: 5/16) | — | 2/4 | 8–48 s |
| Disabled, with PowerShell pitfall checklist | 9/16 | 3/6 | — | 8–25 s |
| `reasoning_effort = low` (three hardest items) | 1/6 (disabled baseline: 1/6) | — | — | 8–32 s |
| Reasoning enabled (default `xhigh`) | **0/16** | — | — | 4–13 min, no answer |

**Best overall score: 9/16 + 2/4 = 11/20, below the pass mark of 14.**

Per-item outcomes with reasoning disabled:

| Item | Score | Defect in the first attempt |
| --- | --- | --- |
| 1 Disk health | 2 | none (temperature and wear reported N/A because they require elevation, an environmental limit) |
| 2 Port ownership | 1 | `"$procId: …"` parsed as a scope-qualified variable, giving a parse error |
| 3 Large directories | 0 | sizes summed only to depth 2; `C:\AI` reported as 1.08 GB against a true 80.22 GB; not fixed by the repair |
| 4 Robocopy backup | 0 | `/LOG+` split from its path (robocopy exit 16); `/XF` given a path pattern |
| 5 Cache accounting | 1 | detection function returned `$null -ne $null`, always false, so 15.2 GB of caches was missed |
| 6 Scheduled tasks | 1 | read the non-existent `NextRunTime` property under StrictMode |
| 7 GPU sampling | 2 | none |
| 8 Health check | 2 | none |
| 9 (laptop) Stale large files | 2 | none |
| 10 (laptop) LaunchAgent | 0 | danger screen matched `rm -f`; no plist produced; URL hard-coded |

### 6.3 Classification of errors

| Class | Examples | Can a checklist help? |
| --- | --- | --- |
| Language detail | `$var:` interpolation; scalar unwrapping; property access under StrictMode | Partly. Rule 1 was obeyed, but rule 8 was broken although it was stated explicitly |
| Wrong API or CLI fact | `npm cache dir` does not exist; `Win32_Service` has no `StartType`; robocopy argument syntax | Only one fact at a time; it does not generalise |
| Misread specification | the depth parameter used to limit the size sum; side effects in a dry run | Partly |
| Incomplete deliverable | item 10 had no plist and no bootstrap command | No |

### 6.4 Discussion

1. **A capability ceiling, not a prompting problem.** All four configurations scored about the same, or worse. The model mostly avoided the pitfalls that the checklist named, but it then erred on API facts that the checklist did not cover. This agrees with community observations that reasoning effort cannot make up for limits in the model itself, and that tasks should be routed by difficulty between local and hosted models.
2. **Read-only, single-step tasks fare best.** Every item that passed at the first attempt was read-only or single-step (items 1, 7, 8 and 9, and new item 2). Failure was concentrated in tasks that depend on external tool syntax (robocopy), multi-level logic (directory sizing) or a complete set of deliverables (the LaunchAgent).
3. **Repair works for a concrete error, not for a wrong result.** Items 2, 5, 6 and 7 were fixed when the model was given a verbatim error. Items 3 and 4 were not: told only that a result was wrong, the model made superficial changes and left the faulty logic in place. This matches the community observation recorded in §2.3.
4. **The safety gates are necessary.** Both hazardous outputs — the directory created during a dry run and the `rm -f` — were caught: one by testing after the fact, the other before it ran.

### 6.5 Threats to validity

- **Small samples.** Each item was generated once, and output at temperature 0.2 still varies from run to run. No repeated sampling was done.
- **Self-set criteria.** The same party wrote the items and the acceptance criteria. The criteria were framed as measurable facts, but the difficulty mix remains a subjective choice.
- **Possible overfitting.** The pitfall checklist was written after the failures had been seen, so it may be tuned to the original items. The three new items were added to test generalisation, but three is a small number.
- **A single quantisation tier.** Only IQ3_XXS was evaluated. IQ2_XS would probably score lower, and IQ3_S does not fit in 64 GB.

---

## 7 Operational Design

### 7.1 Engineering for a single-stream engine

We measured the `/status` endpoint while the server was idle and while it was generating.

- The `busy` flag was reliable: true in every sample taken during generation and false when idle.
- When idle, `phase`, `generated` and the other fields still hold the values from the previous request, **so only `busy` should be trusted**.
- During generation, `/health` and `/v1/models` still answered within 1–2 ms. A slow health check is therefore not evidence that the server is busy, and health checks can safely use short timeouts.

Two measures follow from this.

- **Client-side queueing in `askl`.** Before sending, `askl` checks `busy`. If the server is busy, it re-checks every 5 seconds for up to 600 seconds, then gives up with exit code 4. This turns concurrent callers into serial queueing on the client side.
- **Exit-code convention.**

| Exit code | Meaning |
| --- | --- |
| 0 | success |
| 1 | general error |
| 2 | server offline |
| 3 | reasoning truncated |
| 4 | queue timeout |

### 7.2 Health-check state machine

```
            ┌──────── /health unreachable / timeout / non-200 ────────┐
            │                                                          ▼
/health ──▶ 200 and loaded=true ──▶ /status busy? ──yes──▶ log "busy", exit 0 (never restart)
            │                          └──no──▶ log "healthy", exit 0
            └─ 200 and loaded=false ─▶ loading for how long?
                                         < 15 min ─▶ log "loading", exit 0
                                         ≥ 15 min ─▶ treat as unhealthy
unhealthy ─▶ 3 retries, 20 s apart ─▶ still unhealthy ─▶ Strata-Server started < 5 min ago?
                                                         yes ─▶ log "startup grace", exit 0
                                                         no  ─▶ restart the task (inside ShouldProcess)
```

Design principles:

1. **Never restart a busy server.** A restart would kill a long-running job.
2. **Allow a grace period while the model loads.** The model takes one to two minutes to load after logon. Treating `loaded=false` as a fault would restart the server mid-load, again and again, so it would never come up.
3. **Allow a grace period after start-up.** No restart is made within five minutes of the task starting, so the supervisor and the launcher's own respawn loop do not interfere with each other.
4. **Never treat a quiet log as a failure signal.** The community has reported endless restart cycles caused by doing so.
5. **Log correctly.** Log lines are written with `-WhatIf:$false`, so they appear even during previews, and the key is never written to the log.

Each branch was tested against a mock HTTP server: an `HttpListener` on a background thread that serves `loaded:false`. The suite passed (15/15).

---

## 8 Final Role: A First-Draft Generator Behind Four Gates

On the evidence of §6, we abandoned the premise that the local model could complete scripting tasks on its own. We adopted the following arrangement instead:

| Gate | Instrument | Intercepts |
| --- | --- | --- |
| 1 Static analysis | PSScriptAnalyzer (Warning and above); shellcheck (`-S warning`) | syntax errors, unused variables, `Write-Host` |
| 2 Danger screen | pattern search for `Remove-Item`/`rm`, `/MIR`, `/PURGE`, registry writes, `Stop-Process`, cache purges, and writes outside the real-run branch | destructive operations; side effects in dry runs |
| 3 Strong-model review | the orchestrator reads the whole script | logic errors, incorrect APIs, misread specifications |
| 4 Sandboxed execution | read-only scripts run directly; writing scripts run only with `-WhatIf` or `/L` until the owner approves | run-time errors, incorrect results |

**Operating rules.**

- A repair names one concrete problem and is allowed one round only. If the script still fails, the strong model rewrites it.
- Scripts that pass all four gates go into a version-controlled library and are reused rather than regenerated.
- The tasks that suit the local model best are **read-only inspection, reporting and monitoring**.

---

## 9 Further Work

### 9.1 Short term (one to two weeks)

1. **Throughput tuning.** A community report on an RTX 3080 Ti (12 GB) found three settings that matter:
   - `--pcie-frac`, with an optimum at 0.34 on an inverted-U curve;
   - `--spec-min-p 0.70`, against 0.5 here;
   - the number of CPU workers. The 5600X has only six cores, so that report's value of 9 does not carry over.

   Each setting should be tested on its own, against a fixed set of prompts.
2. **A larger checklist.** Add the API facts exposed by the evaluation, for example `npm config get cache`, `Win32_Service.StartMode`, and the difference between reporting depth and summation depth. Then validate on **new** items, not the original ones.
3. **Larger samples.** Generate each item three times and estimate how temperature affects the first-attempt pass rate.
4. **A laptop availability monitor.** Have the strong model write item 10 and install it only once the owner approves.
5. **A minor correction.** The idempotence check in the health-task registration should ignore the domain prefix in the user name. At present every re-run registers the task again; the result is identical, so this is harmless.

### 9.2 Medium term (one to two months)

1. **The Coder variant.** The installer offers `--family coder` (Flash-Next Coder, IQ1_M only, 58.4 GB). The precision is very low, but fine-tuning for code may help with recalling APIs, so the variant should be compared on the same examination.
2. **Medium effort with a server-side budget.** Only `xhigh` and `low` were tested. If Strata gains a server-side cap on reasoning tokens, comparable to SGLang's `SGLANG_MAX_THINK_TOKENS`, that cap would prevent loops.
3. **Retrieval.** An index of cmdlet and CLI signatures from official documentation, retrieved before each generation, would target the most common class of error: wrong API facts.
4. **Upgrade discipline.** Strata changes quickly. Back up the configuration before each upgrade, because upgrades overwrite it. Afterwards, re-run every acceptance suite and the examination, and roll back if any score falls.

### 9.3 Long term

1. **Routing by difficulty.** Send read-only and single-step tasks to the local model, and send work that spans several tools, needs multi-level logic or has several deliverables directly to a strong model. The routing rules can be tuned from the pass rates recorded in the script library.
2. **Hardware.** The most cost-effective upgrade is **96 GB or more of system memory**. That would leave the low-RAM tier, allow IQ3_S and restore headroom. With this engine the bottleneck is memory and PCIe bandwidth rather than the GPU itself, so a larger GPU would help less.
3. **Observability.** Feed `/status` and `/metrics` into a dashboard that records daily request counts, mean queueing time and the expert-cache hit rate.

---

## 10 Conclusion

The engineering objective was met. A desktop with a 12 GB GPU and 64 GB of system memory can serve a 176B MoE model to the local network reliably at more than 50 tokens per second, with automatic start-up, authentication, health supervision and self-healing.

Test-driven, orchestrated execution proved its worth. It exposed eleven classes of problem. Two of them, graphics-memory exhaustion and reasoning loops, were subtle and surfaced only through tests and log analysis. Others, such as a preview mistaken for execution or defects in the tests themselves, would very probably have gone unnoticed without tests.

The model's ability to write scripts, however, did not meet the threshold set in advance. Its best score was 11/20, and its first-attempt pass rate was about 50%. Neither reasoning modes nor a domain checklist gave a measurable gain. We accept this result and have redefined the model's role as a first-draft generator whose output is controlled by four quality gates.

For anyone attempting a similar deployment, our principal recommendation is this: **define measurable acceptance criteria before deploying, and let measurement rather than expectation decide what role the model plays.**

---

## Appendix A: Key Configuration

```jsonc
// strata-swift-iq3_xxs.json (excerpt; manual edits marked)
{
  "host": "0.0.0.0",                 // manual
  "port": 8080,
  "api_key": "<REDACTED>",           // manual
  "model_name": "local-flash-next",  // manual
  "args": [
    "--expert-cache", "auto", "--prefill", "auto",
    "--spec", "4", "--spec-min-p", "0.5",
    "--max-context", "131072",
    "--mtp-window", "65536",         // manual; lost if setup is re-run
    "--kv", "int8", "--kv-resident", "32768",
    "--vram-reserve-mib", "3072"     // raised from 1500 to resolve the cuBLAS failure
  ]
}
```

## Appendix B: Reproduction Checklist

1. Install on NVMe storage, and confirm the drive with `Get-PhysicalDisk`.
2. Audit the installer source after cloning, and prefer its flag names to those in forum posts.
3. Create the virtual environment yourself with the Python version you intend to use.
4. On a 12 GB GPU that also drives a display, set `--vram-reserve-mib` to at least 3072.
5. With 64 GB of memory and IQ3_XXS, make sure at least 48 GB is free before starting.
6. Reasoning defaults to `xhigh`. For automation, either disable it or name an effort level explicitly, with a token cap and a timeout.
7. On a single-stream engine, clients should check `/status` `busy` first, and the health check must never restart a server that is busy or still loading.
8. Back up the configuration, `run-*.bat` and `serve/server.py` before upgrading; upgrades overwrite them.
9. Pass every script generated by the model through all four gates before deciding whether to run it.

## Appendix C: Acceptance Suites

| Suite | Assertions | Result |
| --- | --- | --- |
| Stage 0 pre-flight | 8 | 8/8 |
| Stage 0b toolchain | 6 | 6/6 |
| Stage 1a clone and audit | 5 | 5/5 |
| Stage 1b installation | 21 | 20/20 + 1 inconclusive |
| Stage 1c first start | 6 | 6/6 |
| Stage 2 network and start-up | 19 | 19/19 |
| Reboot verification | 10 | 10/10 |
| Stage 4a `askl` client | 12 | 12/12 |
| Stage 4b/4c busy-awareness and health check | 15 | 15/15 |
| Health-check scheduled task | 9 | 9/9 |
| Laptop `stage3.bats` | 10 | 10/10 |

## References

- Strata repository: <https://github.com/Niko1221/Strata>
- Community discussions on lcz.me:
  - Strata deployment and parameters: threads #1987, #1988, #2002, #2017, #2031, #2032, #2035
  - Qwen3.8 reasoning-effort levels: #1173, #1181, #1226
  - single-GPU request queueing: #1711
  - RTX 3080 Ti 12 GB tuning: #2095
  - an agent data-loss incident and the resulting safe-deletion rules: #433
