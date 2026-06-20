"""chell/cli.py

Click-based command-line interface for the Chell project.

Subcommands
-----------
chell demo     -- run an interactive debugging session on a Python file or snippet
chell curate   -- placeholder: prints dataset curation instructions
chell finetune -- run fine-tuning via ChellTrainer
chell evaluate -- run the evaluation harness
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import click
import yaml
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config(path: str) -> dict:
    """Load a YAML config file; return empty dict if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _default_config() -> dict:
    """Return a sensible default configuration when no config file is found."""
    return {
        "backend": "api",
        "api": {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "max_tokens": 2048,
            "temperature": 0.2,
        },
    }


def _print_correction(correction) -> None:
    """Pretty-print a Correction object using rich."""
    console.rule("[bold green]Corrected Code[/bold green]")
    syntax = Syntax(correction.code, "python", theme="monokai", line_numbers=True)
    console.print(syntax)

    console.rule("[bold cyan]Explanation[/bold cyan]")
    console.print(correction.explanation)

    if correction.diff:
        console.rule("[bold yellow]Diff[/bold yellow]")
        diff_syntax = Syntax(correction.diff, "diff", theme="monokai")
        console.print(diff_syntax)


def _print_validation(validation) -> None:
    """Pretty-print a ValidationResult object using rich."""
    table = Table(title="Validation Result", show_header=True, header_style="bold magenta")
    table.add_column("Field", style="cyan", no_wrap=True)
    table.add_column("Value", style="white")

    passed_str = "[green]PASS[/green]" if validation.passed else "[red]FAIL[/red]"
    exec_str = "[green]Yes[/green]" if validation.executed_ok else "[red]No[/red]"

    table.add_row("Status", passed_str)
    table.add_row("Executed OK", exec_str)
    table.add_row("AST Similarity", f"{validation.ast_similarity:.2%}")
    if validation.output:
        table.add_row("Output", validation.output[:200])
    if validation.error_message:
        table.add_row("Error", f"[red]{validation.error_message}[/red]")

    console.print(table)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="chell")
def cli() -> None:
    """Chell — Context-Aware, Query-Driven Debugging for Small Language Models."""


# ---------------------------------------------------------------------------
# demo subcommand
# ---------------------------------------------------------------------------


@cli.command("demo")
@click.option(
    "--file",
    "file_path",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    default=None,
    help="Path to a Python source file to debug.",
)
@click.option(
    "--code",
    "code_snippet",
    default=None,
    help="Inline Python code snippet to debug.",
)
@click.option(
    "--config",
    "config_path",
    default="configs/model.yaml",
    show_default=True,
    help="Path to the model configuration YAML.",
)
def demo(
    file_path: Optional[str],
    code_snippet: Optional[str],
    config_path: str,
) -> None:
    """Run an interactive debugging session on a Python file or code snippet."""
    if file_path is None and code_snippet is None:
        raise click.UsageError("Provide either --file <path> or --code <snippet>.")
    if file_path is not None and code_snippet is not None:
        raise click.UsageError("Provide only one of --file or --code, not both.")

    # 1. Load config --------------------------------------------------------
    config = _load_config(config_path)
    if not config:
        console.print(
            f"[yellow]Config file {config_path!r} not found — using defaults.[/yellow]"
        )
        config = _default_config()

    # 2. Read source code ---------------------------------------------------
    if file_path is not None:
        source_code = Path(file_path).read_text(encoding="utf-8")
        console.print(Panel(f"[bold]Debugging file:[/bold] {file_path}", style="blue"))
    else:
        source_code = code_snippet
        console.print(Panel("[bold]Debugging inline code snippet[/bold]", style="blue"))

    console.print(Syntax(source_code, "python", theme="monokai", line_numbers=True))

    # 3. Prompt for task description ----------------------------------------
    task = click.prompt("\nDescribe what this code is supposed to do")

    # 4. Build backend and pipeline -----------------------------------------
    try:
        from chell.models.registry import build_backend
        from chell.pipeline import ChellPipeline
    except ImportError as exc:
        console.print(f"[red]Import error: {exc}[/red]")
        console.print(
            "[yellow]Ensure the pipeline module exists at chell/pipeline.py[/yellow]"
        )
        sys.exit(1)

    console.print("\n[bold]Building backend...[/bold]")
    try:
        backend = build_backend(config)
    except Exception as exc:
        console.print(f"[red]Failed to build backend: {exc}[/red]")
        sys.exit(1)

    pipeline = ChellPipeline(backend=backend)

    # 5. Build BugReport and run pipeline -----------------------------------
    from chell.core.types import BugReport
    from chell.core.responders import InteractiveResponder

    # Detect imported libraries from source code heuristically
    import ast as _ast

    libs: list[str] = []
    try:
        tree = _ast.parse(source_code)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Import):
                for alias in node.names:
                    libs.append(alias.name.split(".")[0])
            elif isinstance(node, _ast.ImportFrom):
                if node.module:
                    libs.append(node.module.split(".")[0])
    except SyntaxError:
        pass
    libs = sorted(set(libs))

    bug_report = BugReport(code=source_code, task=task, libs=libs)
    responder = InteractiveResponder()

    console.rule("[bold]Starting Chell Debug Session[/bold]")

    try:
        session = pipeline.debug(bug_report, responder)
    except Exception as exc:
        console.print(f"[red]Pipeline error: {exc}[/red]")
        raise SystemExit(1) from exc

    # 6. Print results ------------------------------------------------------
    console.rule("[bold green]Session Complete[/bold green]")
    console.print(f"[dim]Clarification turns: {session.num_turns}[/dim]")

    if session.correction is not None:
        _print_correction(session.correction)
    else:
        console.print("[yellow]No correction was produced.[/yellow]")

    if session.validation is not None:
        _print_validation(session.validation)
    else:
        console.print("[dim]No validation result available.[/dim]")


# ---------------------------------------------------------------------------
# curate subcommand
# ---------------------------------------------------------------------------


@cli.command("curate")
@click.option(
    "--output",
    "output_dir",
    default="data/curated/",
    show_default=True,
    help="Directory where curated JSON case files are stored.",
)
def curate(output_dir: str) -> None:
    """Print dataset curation instructions and validate existing cases."""
    import json
    from pathlib import Path

    console.print(
        Panel(
            "[bold cyan]Chell Dataset Curation[/bold cyan]\n\n"
            "Each case is a JSON file saved in the output directory.\n"
            "Required fields per case:\n"
            "  • id                    — unique string, e.g. 'case_042'\n"
            "  • buggy_code            — Python source with the logical error\n"
            "  • task                  — natural-language description of intent\n"
            "  • libs                  — list of library names, e.g. [\"pandas\"]\n"
            "  • error_type            — value from the ErrorType taxonomy\n"
            "  • error_location        — e.g. 'line 3: df.groupby(...)'\n"
            "  • clarification_query   — question asked to resolve ambiguity\n"
            "  • clarification_options — at least two multiple-choice strings\n"
            "  • expected_user_response — ground-truth answer\n"
            "  • corrected_code        — fixed Python source\n"
            "  • explanation           — prose description of the fix\n"
            "  • difficulty (opt)      — 'easy' | 'medium' | 'hard'\n"
            "  • category   (opt)      — 'pandas' | 'numpy' | 'matplotlib' | 'misc'",
            title="Instructions",
        )
    )

    output_path = Path(output_dir)
    if not output_path.exists():
        console.print(f"[yellow]Output directory {output_dir!r} does not exist.[/yellow]")
        return

    json_files = sorted(output_path.glob("*.json"))
    if not json_files:
        console.print(f"[yellow]No case files found in {output_dir!r}.[/yellow]")
        return

    # Validate existing cases
    try:
        from chell.data.schema import validate_case
        import jsonschema
    except ImportError:
        console.print("[yellow]jsonschema not installed — skipping validation.[/yellow]")
        return

    table = Table(title=f"Cases in {output_dir}", header_style="bold magenta")
    table.add_column("File", style="cyan")
    table.add_column("ID", style="white")
    table.add_column("Status", style="white")

    ok_count = 0
    fail_count = 0
    for json_file in json_files:
        try:
            with json_file.open() as fh:
                data = json.load(fh)
            validate_case(data)
            table.add_row(json_file.name, data.get("id", "?"), "[green]OK[/green]")
            ok_count += 1
        except Exception as exc:
            table.add_row(json_file.name, "?", f"[red]FAIL: {exc}[/red]")
            fail_count += 1

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] {ok_count} valid, {fail_count} invalid "
        f"(total {ok_count + fail_count} cases)"
    )


# ---------------------------------------------------------------------------
# finetune subcommand
# ---------------------------------------------------------------------------


@cli.command("finetune")
@click.option(
    "--config",
    "config_path",
    default="configs/train.yaml",
    show_default=True,
    help="Path to the training configuration YAML.",
)
@click.option(
    "--subset",
    "subset",
    type=int,
    default=None,
    help="Use only the first N cases (useful for quick smoke tests).",
)
@click.option(
    "--cases-dir",
    "cases_dir",
    default="data/curated/",
    show_default=True,
    help="Directory with curated JSON case files.",
)
@click.option(
    "--model-config",
    "model_config_path",
    default="configs/model.yaml",
    show_default=True,
    help="Path to the model configuration YAML.",
)
def finetune(
    config_path: str,
    subset: Optional[int],
    cases_dir: str,
    model_config_path: str,
) -> None:
    """Fine-tune a local model using ChellTrainer."""
    train_config = _load_config(config_path)
    if not train_config:
        console.print(f"[red]Training config not found at {config_path!r}[/red]")
        sys.exit(1)

    model_config = _load_config(model_config_path)
    if not model_config:
        console.print(
            f"[yellow]Model config not found at {model_config_path!r} — using defaults.[/yellow]"
        )
        model_config = _default_config()

    backend_type = model_config.get("backend", "hf")
    if backend_type == "api":
        console.print(
            "[red]Fine-tuning requires a local HF backend, not an API backend.\n"
            "Set 'backend: hf' in your model config and specify a model under 'hf:'.[/red]"
        )
        sys.exit(1)

    try:
        from chell.models.registry import build_backend
        from chell.data.dataset import ChellDataset
        from chell.training.trainer import ChellTrainer
    except ImportError as exc:
        console.print(f"[red]Import error: {exc}[/red]")
        sys.exit(1)

    console.print("[bold]Loading dataset...[/bold]")
    dataset = ChellDataset(cases_dir=cases_dir, split="train")
    if subset is not None:
        from torch.utils.data import Subset
        dataset = Subset(dataset, list(range(min(subset, len(dataset)))))
        console.print(f"[dim]Using subset of {min(subset, len(dataset))} cases.[/dim]")
    else:
        console.print(f"[dim]Loaded {len(dataset)} training cases.[/dim]")

    console.print("[bold]Building backend...[/bold]")
    backend = build_backend(model_config)

    trainer = ChellTrainer(backend=backend, dataset=dataset, config=train_config)

    console.print("[bold green]Starting training...[/bold green]")
    try:
        metrics = trainer.train()
    except Exception as exc:
        console.print(f"[red]Training failed: {exc}[/red]")
        sys.exit(1)

    console.rule("[bold green]Training Complete[/bold green]")
    table = Table(title="Training Metrics", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    for k, v in metrics.items():
        table.add_row(str(k), str(v))
    console.print(table)


# ---------------------------------------------------------------------------
# evaluate subcommand
# ---------------------------------------------------------------------------


@cli.command("evaluate")
@click.option(
    "--config",
    "config_path",
    default="configs/eval.yaml",
    show_default=True,
    help="Path to the evaluation configuration YAML.",
)
@click.option(
    "--split",
    "split",
    default="test",
    show_default=True,
    help="Dataset split to evaluate on.",
)
@click.option(
    "--cases-dir",
    "cases_dir",
    default="data/curated/",
    show_default=True,
    help="Directory with curated JSON case files.",
)
@click.option(
    "--model-config",
    "model_config_path",
    default="configs/model.yaml",
    show_default=True,
    help="Path to the model configuration YAML.",
)
def evaluate(
    config_path: str,
    split: str,
    cases_dir: str,
    model_config_path: str,
) -> None:
    """Run the evaluation harness and write results to results/."""
    eval_config = _load_config(config_path)
    if not eval_config:
        console.print(
            f"[yellow]Eval config not found at {config_path!r} — using defaults.[/yellow]"
        )
        eval_config = {"split": split, "max_turns": 5, "output_dir": "results/"}

    model_config = _load_config(model_config_path)
    if not model_config:
        console.print(
            f"[yellow]Model config not found at {model_config_path!r} — using defaults.[/yellow]"
        )
        model_config = _default_config()

    try:
        from chell.models.registry import build_backend
        from chell.data.dataset import ChellDataset
        from chell.evaluation.evaluator import Evaluator
        from chell.pipeline import ChellPipeline
    except ImportError as exc:
        console.print(f"[red]Import error: {exc}[/red]")
        sys.exit(1)

    console.print("[bold]Loading dataset...[/bold]")
    dataset = ChellDataset(cases_dir=cases_dir, split=split)
    console.print(f"[dim]Loaded {len(dataset)} cases for split '{split}'.[/dim]")

    console.print("[bold]Building backend and pipeline...[/bold]")
    backend = build_backend(model_config)
    pipeline = ChellPipeline(backend=backend)

    max_turns: int = eval_config.get("max_turns", 5)
    evaluator = Evaluator(pipeline=pipeline, dataset=dataset, max_turns=max_turns)

    console.print(f"[bold green]Evaluating on '{split}' split...[/bold green]")
    try:
        metrics = evaluator.run(split=split)
    except Exception as exc:
        console.print(f"[red]Evaluation failed: {exc}[/red]")
        sys.exit(1)

    # Print metrics
    console.rule("[bold green]Evaluation Results[/bold green]")
    table = Table(title="Chell Metrics", header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    for k, v in metrics.items():
        if isinstance(v, float):
            table.add_row(str(k), f"{v:.4f}")
        else:
            table.add_row(str(k), str(v))
    console.print(table)

    # Write results to disk
    output_dir = Path(eval_config.get("output_dir", "results/"))
    output_dir.mkdir(parents=True, exist_ok=True)
    import json
    from datetime import datetime

    result_file = output_dir / f"eval_{split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with result_file.open("w", encoding="utf-8") as fh:
        json.dump({"split": split, "metrics": metrics}, fh, indent=2)

    console.print(f"\n[dim]Results written to {result_file}[/dim]")


if __name__ == "__main__":
    cli()
