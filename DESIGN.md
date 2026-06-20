# Chell — Design Document

**Context-Aware, Query-Driven Debugging of Logical Errors in Small Language Models for Python Data Analysis**

> Full research-reproduction design: working pipeline + curated dataset + real CAAM attention
> surgery + fine-tuning + four-metric evaluation harness.

---

## 1. Context & Motivation

Chell makes **Small Language Models (SLMs)** — Phi-2, Qwen2.5-Coder, StarCoder, TinyLlama,
Code Llama — fix *logical* (not syntactic) errors in Python data-analysis code (pandas / NumPy /
Matplotlib). The core insight from the papers: SLMs make **plausible but wrong** fixes because
they *guess* at ambiguous user intent. Chell inserts a **clarification step** before fixing.

**Four-step workflow** (paper Fig. 1):

```
Error Identification → Clarification Dialogue → Suggest Correction → Validation
                                  ↑__________________(iterate)__________________|
```

Backed by **three modules** (error detection, query-generation engine, assisted refinement),
**memory-augmented retrieval** (DPR + FAISS), and the novel **Context-Aware Attention Mechanism
(CAAM)** that injects a context term `C` into the attention logits:

```
raw_scores = (QKᵀ + C) / √dₖ
```

The CAAM diagram contrasts four attention families — **MHA** (Phi-2, TinyLlama), **MQA**
(StarCoder), **GQA** (Qwen2.5), and Chell's **context-aware** variant — which this design
realizes as per-architecture **adapters**.

**Evaluation** uses four metrics — error-resolution accuracy, interaction efficiency, execution
correctness, AST similarity — benchmarked against baseline SLMs, GPT-4, Gemini 1.5, PyLint, and
PandasAI.

**Scope decisions (locked):** full research reproduction · real attention surgery for CAAM ·
pluggable local-HuggingFace + API backends · full 150-case dataset curation.

This is a greenfield build. The original README's flat layout (`chell/ examples/ data/ notebook/
results/`) is a reasonable seed but too shallow for a research repro; this design deepens it into
a package with **one sub-package per module** and explicit OOP seams.

---

## 2. Repository Layout

```
chell/
├── chell/                          # main package
│   ├── config.py                   # dataclass configs + YAML loading
│   ├── core/
│   │   ├── types.py                # BugReport, ErrorDiagnosis, ClarificationQuery,
│   │   │                           #   UserResponse, Correction, ValidationResult, DebugSession
│   │   ├── pipeline.py             # ChellPipeline — 4-step orchestrator
│   │   └── responders.py           # UserResponder ABC: Interactive / Simulated (for eval)
│   ├── detection/                  # Module 1: error detection
│   │   ├── base.py                 # ErrorDetector ABC
│   │   ├── taxonomy.py             # ErrorType enum + category taxonomy
│   │   ├── static_detector.py      # AST/heuristic logical-error rules
│   │   └── llm_detector.py         # SLM-driven detection
│   ├── query/                      # Module 2: query-generation engine
│   │   ├── base.py                 # QueryGenerator ABC
│   │   ├── generator.py            # clarification-question generation
│   │   ├── retriever.py            # DPR retriever (question/context encoders)
│   │   └── ranking.py              # relevance ranking + dedup (anti-fatigue)
│   ├── refinement/
│   │   ├── base.py                 # Refiner ABC
│   │   └── refiner.py              # Module 3: assisted refinement
│   ├── validation/
│   │   ├── base.py                 # Validator ABC
│   │   ├── executor.py             # SandboxExecutor (subprocess, timeout/limits)
│   │   └── validators.py           # execution-correctness + AST-similarity checks
│   ├── memory/
│   │   ├── store.py                # MemoryStore — memory-augmented case store
│   │   └── faiss_index.py          # FaissIndex wrapper (IndexFlatIP / IVF)
│   ├── models/                     # pluggable backends
│   │   ├── base.py                 # ModelBackend ABC (.generate / .embed / .torch_model)
│   │   ├── hf_backend.py           # local HuggingFace transformers
│   │   ├── api_backend.py          # hosted API (Claude / OpenAI / Gemini)
│   │   └── registry.py             # factory: config → backend
│   ├── caam/                       # Context-Aware Attention Mechanism (novelty)
│   │   ├── attention.py            # ContextAwareAttention nn.Module
│   │   ├── context_encoder.py      # DebugSession → context tensor (C-source)
│   │   ├── patcher.py              # CAAMPatcher: inject into a loaded HF model
│   │   └── adapters/               # per-arch: phi2, qwen2 (GQA), gptbigcode (MQA), llama
│   ├── data/
│   │   ├── schema.py               # DebugCase dataclass + JSON Schema validation
│   │   ├── dataset.py              # ChellDataset loader + torch Dataset view
│   │   └── builder.py              # curation / augmentation / dedup helpers
│   ├── training/
│   │   ├── collator.py             # builds (buggy + query + response → fix) sequences
│   │   └── trainer.py              # ChellTrainer over HF Trainer (+ CAAM + LoRA)
│   ├── evaluation/
│   │   ├── metrics.py              # 4 paper metrics
│   │   ├── evaluator.py            # benchmark harness (uses SimulatedResponder)
│   │   └── baselines.py            # PyLint, PandasAI, GPT-4/Gemini, one-shot SLM
│   └── cli.py                      # subcommands: demo / curate / finetune / evaluate
├── data/{raw,curated,splits}/      # curated 150 cases + train/val/test
├── configs/                        # YAML: model, caam, train, retrieval, eval
├── scripts/                        # curate_dataset.py finetune.py evaluate.py run_demo.py
├── examples/  notebooks/  results/  tests/
├── main.py   pyproject.toml   README.md
```

---

## 3. Low-Level Design (OOP)

### 3.1 Data model — `core/types.py` (frozen dataclasses)

```
BugReport(code, task, libs)
  → ErrorDiagnosis(error_type, location, description, confidence, is_ambiguous, candidate_intents)
  → ClarificationQuery(question, options, rationale, retrieved_refs)
  → UserResponse(text, selection)
  → Correction(code, explanation, diff)
  → ValidationResult(executed_ok, output, ast_similarity, passed)
```

`DebugSession` accumulates conversation turns + clarification history. It is the **single source
of "context"** consumed by both the prompt and CAAM's `ContextEncoder`.

### 3.2 Patterns

| Pattern | Where | Why |
|---|---|---|
| **Strategy + ABC** | detection / query / refinement / validation | swap static-vs-LLM detector, query strategies, validators via config |
| **Factory / Registry** | `models/registry.py` | `build_backend(config) → ModelBackend` (local HF or API) |
| **Adapter** | `caam/adapters/` | one adapter per attention family (MHA / MQA / GQA) |
| **Template method** | `ChellPipeline.debug()` | fixed 4-step skeleton, pluggable steps |
| **Strategy (test seam)** | `core/responders.py` | `Interactive` (demo) vs `Simulated` (eval) answer source |

### 3.3 Pipeline orchestrator — `core/pipeline.py`

`ChellPipeline.debug(bug, responder)` composes `(detector, query_gen, refiner, validator,
memory, backend)` and runs:

```
detect → if ambiguous: generate queries → responder.answer() → refine → validate
       → repeat until valid or max_turns
```

`UserResponder` ABC decouples the answer source so the **eval harness reuses the exact
production pipeline** — `SimulatedResponder` feeds the dataset's `expected_user_response`.

### 3.4 Backends — `models/`

`ModelBackend` ABC exposes `.generate(prompt)`, `.embed(text)`, and `.torch_model` (so CAAM can
reach internals). `HFBackend` (local transformers) + `APIBackend` (hosted). API models default to
the latest Claude models for production, while still supporting GPT-4 / Gemini for the paper's
baseline comparison.

### 3.5 CAAM — the real attention surgery (the novelty)

- `ContextAwareAttention(nn.Module)` reproduces base self-attention, then adds bias
  `C = scale · (Q @ Wc(ctx)ᵀ)` to the `QKᵀ` logits **before softmax**, where `ctx` is the
  encoded clarification/memory context.
- `Wc` (the only new weights) is **zero-initialized** → the patched model is bit-for-bit
  identical to the pretrained model until fine-tuning teaches it to use context. This is critical
  for training stability and gives a clean ablation.
- `CAAMPatcher.patch(model)` swaps the attention forward per architecture via **adapters**
  (Phi-2 MHA, Qwen2 GQA, StarCoder GPTBigCode MQA, Llama/TinyLlama), handling K/V sharing and
  head grouping. Context is threaded into the fixed HF `forward` signature via a
  `model.caam_context` register set by the pipeline before each forward pass.
- `ContextEncoder` pools the `DebugSession` (clarification Q&A + retrieved cases) into the
  representation that feeds `Wc`.

### 3.6 Memory + retrieval — `memory/`, `query/retriever.py`

DPR-style dual encoders embed bug + diagnosis; `FaissIndex` retrieves similar prior cases /
clarification queries; `MemoryStore` persists embeddings + cases and grows across sessions
(memory-augmented learning). Retrieved context feeds both **query ranking** (avoid redundant
questions → reduce user fatigue) and **CAAM's C**.

### 3.7 Dataset — `data/`

`DebugCase` matches the paper schema: `id, buggy_code, error_type, clarification_query,
expected_user_response, corrected_code, explanation`. `schema.py` validates via JSON Schema.
Category targets: **pandas 50 · NumPy 40 · Matplotlib 30 · misc 30 = 150**. `DatasetBuilder`
curates from GitHub / Kaggle / forums + PyLint / PandasAI augmentation, dedups, and writes
train/val/test splits.

### 3.8 Training — `training/`

`ChellTrainer` wraps HF `Trainer`, applies `CAAMPatcher`, trains `Wc` (+ optional LoRA on base)
with paper hyperparams: **batch 8, lr 5e-5, 3 epochs, AdamW**. `ChellCollator` formats sequences
and routes the context span to `ContextEncoder`.

### 3.9 Evaluation — `evaluation/`

`metrics.py` implements the four paper metrics:

| Metric | Definition |
|---|---|
| Error-resolution accuracy | % of labeled logical errors successfully resolved |
| Interaction efficiency | # clarification turns before a correct fix |
| Execution correctness | corrected code runs error-free (via `SandboxExecutor`) and matches expected output |
| AST similarity | tree-edit distance (e.g. `zss`) between generated and reference `corrected_code` |

`Evaluator` runs the pipeline with `SimulatedResponder` over the test split; `baselines.py` wraps
PyLint, PandasAI, one-shot SLM (no clarification), and GPT-4 / Gemini to reproduce the comparison
tables into `results/`.

---

## 4. Build Plan (milestones → tasks)

### M0 — Scaffold & contracts
1. `pyproject.toml` (torch, transformers, faiss-cpu, datasets, jsonschema, zss, pylint,
   pandas/numpy/matplotlib, anthropic/openai, pytest), package skeleton, `configs/` YAMLs.
2. `core/types.py` dataclasses + all module ABCs (empty contracts) + `MockModelBackend` so every
   layer is testable without a real model.

### M1 — Pipeline framework (end-to-end on mocks/API)
3. `models/`: `ModelBackend` ABC, `HFBackend`, `APIBackend`, `registry`.
4. `detection/`: taxonomy + `static_detector` + `llm_detector`.
5. `query/`: `generator` + `ranking` (retriever stubbed first).
6. `refinement/refiner.py`; `validation/` `SandboxExecutor` + validators.
7. `core/pipeline.py` + `responders.py`; wire `cli.py demo` and `main.py`.
8. Unit tests per module against `MockModelBackend`; one integration test runs the full loop.

### M2 — Dataset (full 150-case curation)
9. `data/schema.py` + JSON Schema; `data/dataset.py` loader + splits.
10. `data/builder.py` curation/augmentation; curate all 150 cases (pandas 50 / NumPy 40 /
    Matplotlib 30 / misc 30) into `data/curated/`; test asserting schema + counts.

### M3 — Memory & retrieval
11. `memory/faiss_index.py` + `MemoryStore`; `query/retriever.py` DPR encoders; plug retrieval
    into query ranking and the memory-augmented loop. Tests for index round-trip + recall.

### M4 — CAAM (attention surgery)
12. `caam/attention.py` `ContextAwareAttention` + `context_encoder.py`.
13. `caam/patcher.py` + per-arch `adapters/` (phi2 / qwen2 / gptbigcode / llama).
14. **Zero-init equivalence test**: patched output == base output when `Wc=0`. Tests for
    MHA/MQA/GQA shape handling.

### M5 — Fine-tuning
15. `training/collator.py` + `training/trainer.py`; `scripts/finetune.py`. Smoke-train on a
    small subset to confirm loss decreases and `Wc` moves off zero.

### M6 — Evaluation & benchmarking
16. `evaluation/metrics.py` (4 metrics) + tests on synthetic pairs.
17. `evaluation/evaluator.py` + `baselines.py`; `scripts/evaluate.py`. Produce comparison tables
    (Chell vs one-shot SLM vs PyLint/PandasAI vs GPT-4/Gemini) into `results/`.

### M7 — Polish
18. Rewrite `README.md` with the new architecture + usage; example notebooks; `examples/` sample
    bugs; final CLI subcommands wired (`curate / finetune / evaluate`).

---

## 5. Verification

- **Unit/integration** — `pytest tests/`: module contracts on `MockModelBackend`, dataset schema
  + count assertions, FAISS round-trip, metric correctness on synthetic pairs.
- **CAAM correctness** — the **zero-init equivalence test** is the gate: if patched ≠ base at
  `Wc=0`, the surgery is wrong. Verify per architecture (Phi-2 / Qwen / StarCoder / Llama).
- **Pipeline demo** — `python main.py demo --file examples/<buggy>.py` runs the interactive
  4-step loop against an API backend and produces a validated fix.
- **Fine-tune smoke** — `scripts/finetune.py --subset 20` shows decreasing loss and non-zero `Wc`.
- **Benchmark** — `scripts/evaluate.py --split test` reproduces the four-metric comparison tables
  in `results/`, confirming Chell (with clarification) beats one-shot SLM on intent-ambiguous
  cases — the paper's core hypothesis.

## 6. Suggested execution order

Start at **M0/M1** (scaffold + pipeline on the pluggable backend) for an early runnable
detect→clarify→fix→validate loop, then layer **dataset → memory → CAAM → fine-tune → eval**.
CAAM and fine-tuning depend on M0's `ModelBackend.torch_model` seam and M2's dataset, so this
ordering keeps every milestone independently testable.
