# Beyond Transformers — Autonomous Research Program

## Mission

Design, implement, and validate architecture ideas that could improve capability-per-compute beyond standard dense transformer assumptions.

This repo should be treated as a **research harness**, not as evidence that the core ideas already work.

The near-term goal is not "beat frontier models." The near-term goal is:

1. make the experiment loop reproducible
2. fix the experimental design
3. rerun trustworthy baselines
4. determine which ideas deserve deeper investment

## Current Assessment

The existing results are **out of date and only partially trustworthy**.

Why:

- The task is tiny character-level language modeling on Tiny Shakespeare.
- The metric is `val_bpb`, which is useful for fast iteration but weakly aligned with the actual thesis of competing with frontier models through better computation.
- The time budget is fixed at 120 seconds on CPU, which makes loop throughput good but makes the results sensitive to optimizer and throughput changes.
- Experiments were mostly architecture and hyperparameter hill-climbing without stronger controls:
  - no repeated runs across seeds
  - no matched baseline transformer or simple RNN comparisons under the same budget
  - no explicit separation between "loop sanity check" and "research conclusion"

Because of that, the current experiment table should be treated as:

- useful for understanding what the code did before
- not sufficient for making strong research claims

## Rules for the Agent

1. **Only modify `train.py`** during autoresearch experiments unless a human explicitly changes the protocol.
2. **One change per experiment.**
3. **The immediate metric is `val_bpb`** on the current harness. Lower is better.
4. **Do not treat `val_bpb` improvements as proof of frontier-relevant progress.**
5. **The first priority is reproducibility and experimental cleanup.**
6. **If the environment is broken, fix the environment before interpreting results.**

## Phase Structure

## Phase 0 — Reproducibility and Environment

Goal:
Make the repo runnable and its outputs trustworthy enough to rerun baseline experiments.

Required outcomes:
- local Python environment exists and can run `train.py`
- dataset preparation works
- one clean baseline run completes
- results logging is functioning

Current blocker:
- on this machine, `python3 train.py` fails because required packages like `numpy` are not installed in a project environment

## Phase 1 — Rerun the Existing Harness Cleanly

Goal:
Rerun the initial experiments under a better protocol so we know what is actually true inside the current tiny-shakespeare harness.

Required changes in practice:
- rerun baseline from scratch
- rerun the best historical experiments
- repeat promising runs across multiple seeds
- distinguish throughput wins from genuine modeling wins

Minimum trustworthy outputs:
- baseline rerun
- top 3 historical settings rerun
- seed variance notes
- updated results table with a "trusted" subset

## Phase 2 — Stronger Experimental Design

Goal:
Decide whether this repo is testing the right thing.

Questions to answer:
- Is character-level Tiny Shakespeare the right first-stage harness for these architecture ideas?
- Which ideas are really being tested by this setup:
  - pole dynamics?
  - training stability?
  - throughput?
  - long-range modeling?
- What matched baselines are missing?

Likely requirements:
- add matched baseline architectures under the same parameter/time budget
- separate "sanity loop benchmark" from "research benchmark"

## Phase 3 — Align the Repo with the Strongest Ideas

The strongest ideas from the broader discussion were not "randomly mutate a pole model."
They were more like:

- conditional compute
- adaptive computation / routing
- hyperpolarization-style dynamic state
- predictive coding
- multi-timescale dynamics

This repo should eventually decide which of these it is actually for.

Right now it is closest to:
- **Laplace / multi-timescale dynamics**

It is **not yet** a good testbed for:
- RL-guided sparse routing
- orchestrator-based specialist composition
- real frontier-competitive task evaluation

## Workflow

### Loop A — Reproducibility Loop

Use first:

1. verify environment
2. run baseline
3. rerun historical best configurations
4. record whether old results reproduce

### Loop B — Research Loop

Use only after Loop A is stable:

1. read `train.py`, `results.tsv`, and git log
2. form one hypothesis
3. edit `train.py`
4. run `python loop.py --once --tag "<description>"`
5. keep or revert based on `val_bpb`
6. log the result

### Loop C — Redesign Loop

After enough reruns:

1. identify what the current harness does not measure
2. propose improved benchmark design
3. decide whether to keep this repo focused on pole dynamics or repurpose it

## Metric

### Immediate metric
- `val_bpb` — validation bits per byte on the current Tiny Shakespeare harness

### Important warning
- `val_bpb` is a **local loop metric**, not the final research metric for the overarching frontier-model agenda

### Secondary things to track
- parameter count
- steps completed in time budget
- training stability
- pole statistics
- reproducibility across seeds

## Current Architecture

The current repo implements a pole-parameterized recurrent sequence model:

- complex poles `(sigma + i * omega)`
- sigma constrained `< 0` for stability
- multi-head pole structure
- bilinear discretization
- FFT-based parallel convolution
- character-level language modeling on Tiny Shakespeare

This is an interesting architecture sandbox, but it is only one slice of the overall research agenda.

## What Has Been Tried

Historical experiments include:
- warmup and learning rate changes
- hidden size changes
- layer count changes
- feedforward expansion changes
- learned `dt`
- FFT parallel convolution
- multi-head poles with bilinear discretization

These results are historically useful, but they need rerunning before they are trusted.

## What To Do Next

Ordered priority:

1. **Bootstrap the environment**
   - create a project venv
   - install dependencies
   - verify `train.py` runs

2. **Rerun the baseline**
   - establish fresh baseline under the current codebase

3. **Rerun the best historical settings**
   - at least:
     - baseline
     - cosine lr decay
     - 2-layer reduced model
     - learned `dt`
     - FFT parallel convolution
     - current HEAD configuration

4. **Repeat across seeds**
   - determine whether gains are robust or noise

5. **Write an experiment validity memo**
   - what is reproducible
   - what is stale
   - what this harness can and cannot tell us

6. **Then decide whether to keep investing here**
   - continue with pole dynamics
   - or redesign the repo toward a better-aligned experiment harness

## Signals This Repo Is Healthy

- environment runs locally without manual rescue
- baseline reproduces
- top historical runs are directionally consistent
- seed variance is understood
- we can clearly explain what a win here means

## Signals This Repo Is Not Yet Healthy

- environment is broken
- results depend too heavily on random seed
- throughput improvements masquerade as model improvements
- no matched baseline exists
- we cannot explain how the harness connects to the broader thesis

## Constraints

- keep the current loop lightweight enough to iterate quickly
- do not claim frontier-relevant conclusions from `val_bpb` alone
- treat the current repo as a research sandbox until stronger evaluation is added
