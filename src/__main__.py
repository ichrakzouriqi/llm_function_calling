"""
__main__.py – Command-line entry point for the call-me-maybe project.

Run with:
    uv run python -m src
    uv run python -m src --help

Default paths:
    --functions_definition  data/input/functions_definition.json
    --input                 data/input/function_calling_tests.json
    --output                data/output/function_calling_results.json
    --model                 Qwen/Qwen3-0.6B

What this module does, in order:
    1. Parse and validate command-line arguments.
    2. Load + validate the functions definition file (via Pydantic).
    3. Load + validate the prompts file.
    4. Load the language model from the SDK.
    5. Run constrained decoding on every prompt.
    6. Write the results to the output file.

Every foreseeable error (missing file, bad JSON, wrong shape, decode
failure) is caught and reported with a clear message — the program
never crashes with a raw traceback shown to the user.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.decoder import decode_all
from src.schemas import FunctionCall, FunctionDefinition


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Return the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "Function calling for small LLMs via constrained decoding.\n"
            "Reads a list of callable functions and a list of prompts, "
            "then writes one function-call JSON object per prompt."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        default="data/input/functions_definition.json",
        metavar="PATH",
        help="Path to the JSON file describing callable functions "
             "(default: data/input/functions_definition.json).",
    )
    parser.add_argument(
        "--input",
        default="data/input/function_calling_tests.json",
        metavar="PATH",
        help="Path to the JSON file with prompts to process "
             "(default: data/input/function_calling_tests.json).",
    )
    parser.add_argument(
        "--output",
        default="data/output/function_calling_results.json",
        metavar="PATH",
        help="Path to write the results JSON file "
             "(default: data/output/function_calling_results.json).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-0.6B",
        metavar="MODEL_ID",
        help="Hugging Face model ID to use "
             "(default: Qwen/Qwen3-0.6B). "
             "First run downloads the weights; later runs use the cache.",
    )
    return parser


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path, label: str) -> Any:
    """
    Load a JSON file and return the parsed object.

    Raises SystemExit with a clear message on:
      - file not found
      - JSON syntax error
    """
    if not path.exists():
        print(
            f"ERROR: {label} file not found: {path}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        print(
            f"ERROR: {label} file is not valid JSON: {path}\n  {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


def _load_functions(path: Path) -> list[FunctionDefinition]:
    """
    Load and validate functions_definition.json.

    Returns a list of FunctionDefinition objects.
    Exits with a clear message if the file is missing, invalid JSON,
    or does not match the expected schema.
    """
    raw = _load_json(path, "functions definition")

    if not isinstance(raw, list):
        print(
            f"ERROR: functions definition must be a JSON array, got {type(raw).__name__}.",
            file=sys.stderr,
        )
        sys.exit(1)

    functions: list[FunctionDefinition] = []
    for i, item in enumerate(raw):
        try:
            functions.append(FunctionDefinition.model_validate(item))
        except ValidationError as exc:
            print(
                f"ERROR: functions definition entry {i} is invalid:\n{exc}",
                file=sys.stderr,
            )
            sys.exit(1)

    if not functions:
        print("ERROR: functions definition file is empty.", file=sys.stderr)
        sys.exit(1)

    return functions


def _load_prompts(path: Path) -> list[str]:
    """
    Load and validate function_calling_tests.json.

    Returns a list of prompt strings.
    Each entry in the file must be an object with a "prompt" key.
    """
    raw = _load_json(path, "prompts")

    if not isinstance(raw, list):
        print(
            f"ERROR: prompts file must be a JSON array, got {type(raw).__name__}.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompts: list[str] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict) or "prompt" not in item:
            print(
                f"ERROR: prompts entry {i} must be an object with a 'prompt' key.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not isinstance(item["prompt"], str):
            print(
                f"ERROR: prompts entry {i}: 'prompt' must be a string.",
                file=sys.stderr,
            )
            sys.exit(1)
        prompts.append(item["prompt"])

    if not prompts:
        print("ERROR: prompts file is empty.", file=sys.stderr)
        sys.exit(1)

    return prompts


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def _write_results(path: Path, results: list[FunctionCall]) -> None:
    """
    Serialize the list of FunctionCall objects to JSON and write to *path*.

    The output directory is created if it does not exist.
    Each entry has exactly three keys: prompt / name / parameters.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    output: list[dict[str, Any]] = [call.model_dump() for call in results]

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, ensure_ascii=False)

    print(
        f"Wrote {len(results)} function call(s) to {path}",
        file=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Parse arguments, load inputs, run decoding, write output."""
    parser = _build_parser()
    args = parser.parse_args()

    functions_path = Path(args.functions_definition)
    input_path = Path(args.input)
    output_path = Path(args.output)
    model_id: str = args.model

    # ── 1. Load and validate inputs ───────────────────────────────────
    functions = _load_functions(functions_path)
    prompts = _load_prompts(input_path)

    # ── 2. Load the language model ────────────────────────────────────
    # We import here (not at module level) so that the SDK's heavy
    # dependencies (torch, transformers) are only loaded when we actually
    # run the program — unit tests never trigger this import.
    print(f"Loading model {model_id} ...", file=sys.stderr)
    try:
        from llm_sdk import Small_LLM_Model
        model = Small_LLM_Model(model_id)
    except Exception as exc:
        print(f"ERROR: could not load model '{model_id}':\n  {exc}", file=sys.stderr)
        sys.exit(1)

    # ── 3. Run constrained decoding on every prompt ───────────────────
    results = decode_all(model, prompts, functions)

    # ── 4. Write results ──────────────────────────────────────────────
    try:
        _write_results(output_path, results)
    except OSError as exc:
        print(f"ERROR: could not write output file {output_path}:\n  {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
