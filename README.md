# Chell: Context-Aware, Query-Driven Debugging for Small Language Models

Chell is a research reproduction of *"Context-Aware, Query-Driven Debugging of Logical Errors in Small Language Models for Python Data Analysis"*. It implements an interactive debugging system that helps Small Language Models (SLMs) fix logical errors in Python data-analysis code by asking targeted clarification questions before proposing a fix.

## Architecture

```
BugReport
    │
    ▼
ErrorDetector ──── taxonomy (17 error types)
    │                └── StaticDetector + LLMDetector
    │
    ▼
QueryGenerator ─── DPRRetriever ──── MemoryStore (FAISS)
    │                └── sentence-transformers
    │
    ▼
  [clarification loop — up to max_turns]
    │   UserResponder (interactive or simulated)
    │
    ▼
Refiner ──────────── LLMRefiner (unified diff)
    │
    ▼
Validator ─────────── SandboxExecutor + ASTValidator
    │
    ▼
  DebugSession  →  ValidationResult  →  4 paper metrics
```

CAAM (Context-Aware Attention Modification) patches the SLM's attention layers during fine-tuning, adding a learned context bias `C = scale · (Q @ Wc(context)ᵀ)` to the QKᵀ logits. `Wc` is zero-initialized so the patched model starts identically to the base model.

## Quick Start

```bash
git clone <repo>
cd chell
pip install -e .
```

### Interactive demo

```bash
python main.py demo --file examples/buggy_pandas.py
```

### Fine-tune (smoke test on 20 cases, no real GPU needed for mocks)

```bash
python scripts/finetune.py --subset 20 --no-lora
```

### Evaluate on the test split

```bash
python scripts/evaluate.py --split test --chell-only
```

## Dataset

150 hand-annotated Python debugging cases across 4 categories:

| Category    | Count | Example errors                              |
|-------------|-------|---------------------------------------------|
| pandas      | 50    | wrong groupby key, wrong merge key          |
| numpy       | 40    | wrong axis, broadcasting error              |
| matplotlib  | 30    | wrong plot type, missing labels             |
| misc        | 30    | off-by-one, logic inversion                 |

Cases are split into train (113) / val (19) / test (18) under `data/splits/`.

## Evaluation Metrics (paper §4)

| Metric                     | Description                                              |
|----------------------------|----------------------------------------------------------|
| Error Resolution Accuracy  | Fraction of cases where the fix passed validation        |
| Interaction Efficiency     | Mean clarification turns for resolved cases (↓ better)  |
| Execution Correctness      | Fraction where corrected code executed and matched output|
| AST Similarity Score       | Mean Jaccard similarity of AST node types               |

Baselines: PyLint (static), one-shot SLM (no clarification), GPT-4o, Gemini 1.5 Pro.

## CLI Reference

```bash
python main.py demo       --file <path>          # interactive debug session
python main.py curate     --out data/curated/    # add dataset cases
python main.py finetune   --subset N             # fine-tune on N cases
python main.py evaluate   --split test           # run eval harness
```

## Project Layout

```
chell/
├── core/          # types, pipeline, responders
├── detection/     # taxonomy (17 ErrorTypes), StaticDetector, LLMDetector
├── query/         # LLMQueryGenerator, DPRRetriever, QueryRanker
├── refinement/    # LLMRefiner (difflib unified diff)
├── validation/    # SandboxExecutor, ExecutionValidator, ASTValidator
├── memory/        # FaissIndex, MemoryStore
├── caam/          # ContextAwareAttention, ContextEncoder, CAAMPatcher
│   └── adapters/  # phi2, qwen2, gptbigcode, llama
├── data/          # DebugCase schema, ChellDataset, DatasetBuilder
├── training/      # ChellCollator, ChellTrainer (LoRA + CAAM)
├── evaluation/    # 4 paper metrics, Evaluator, baselines
└── models/        # APIBackend (Claude/OpenAI/Gemini), HFBackend, MockModelBackend
configs/           # model.yaml, caam.yaml, train.yaml, retrieval.yaml, eval.yaml
data/curated/      # 150 JSON cases (case_001–case_150)
data/splits/       # train/val/test split indices
scripts/           # finetune.py, evaluate.py, curate_dataset.py, run_demo.py
tests/             # 116 tests (pytest)
results/           # evaluation output (JSON)
examples/          # buggy_pandas.py and other sample inputs
```

## Configuration

Key configs in `configs/`:

- **`model.yaml`** — backend type (`api` / `hf` / `mock`), provider, model name
- **`caam.yaml`** — `context_dim`, `scale`, `arch` (`phi2`/`qwen2`/`gptbigcode`/`llama`)
- **`train.yaml`** — batch size, lr, epochs, LoRA `r`/`alpha`/`target_modules`
- **`eval.yaml`** — split, baselines to run, max_turns, output dir

## Model Backends

| Backend       | Use case                                | Notes                         |
|---------------|-----------------------------------------|-------------------------------|
| `api`         | Claude (primary), OpenAI, Gemini        | Set `ANTHROPIC_API_KEY` etc.  |
| `hf`          | Local SLMs (Phi-2, Qwen2, StarCoder)    | Requires GPU + model download |
| `mock`        | Tests, CI, offline development          | Deterministic, no network     |

Default API model: `claude-sonnet-4-6`.

## Running Tests

```bash
pytest tests/          # 116 tests, ~6s
pytest tests/ -k caam  # CAAM attention surgery tests only
pytest tests/ -k training  # fine-tuning tests only
```

## Paper

This codebase reproduces the Chell system described in:

> *Chell: Context-Aware, Query-Driven Debugging of Logical Errors in Small Language Models for Python Data Analysis*

See `DESIGN.md` for the full system specification and milestone plan.
