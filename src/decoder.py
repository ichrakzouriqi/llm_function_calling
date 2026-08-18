"""
decoder.py – Constrained decoding for function-call generation.

Core idea
---------
We build the JSON answer token by token. At each step we ask the model
"which token comes next?" but we filter out any token that would make the
output invalid.  The model still chooses *what* to say; we only stop it
from going off the rails.

Two things the model must decide:
  1. Which function to call  (chosen by _choose)
  2. What each argument value is  (chosen by _string / _number / _value)

Everything else (braces, quotes, colons, parameter names) is written
directly in code — these are never decisions.

Public surface used here from the SDK:
  model.encode(text)               -> tensor  (token IDs for text)
  model.get_logits_from_input_ids  -> list[float]
  model.get_path_to_vocab_file()   -> str
"""

from __future__ import annotations

import re
import sys
from typing import Any, Callable

import numpy as np

from src.schemas import FunctionCall, FunctionDefinition
from src.tokenizer import Vocabulary


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class DecodeError(Exception):
    """Raised when no valid next token can be found."""


# ---------------------------------------------------------------------------
# Legality rule 1 — JSON number
# ---------------------------------------------------------------------------

def _number_ok(piece: str, allow_fraction: bool, end: str) -> bool:
    """
    Return True if *piece* is still on track to become a valid JSON number.

    piece     : text accumulated so far (may contain the terminating char)
    allow_fraction : True for type "number" (float), False for "integer"
    end       : the character that terminates the number ("," or "}")

    Rules (following the JSON spec exactly):
      - No leading zeros  ("06" is illegal; "0" alone is fine).
      - A dot must be followed by at least one digit  ("2." is illegal).
      - Integers may never contain a dot.
      - Once *end* appears inside *piece*, the part before it must be a
        *complete* number (not just a prefix).

    Examples:
      _number_ok("4",    False, ",") -> True   # valid integer prefix
      _number_ok("4,",   False, ",") -> True   # complete integer + end
      _number_ok("4.",   False, ",") -> False  # dot forbidden for int
      _number_ok("2.5}", True,  "}") -> True   # complete float + end
      _number_ok("06",   False, ",") -> False  # leading zero
      _number_ok("0",    True,  ",") -> True   # solo zero is fine
      _number_ok("2.,",  True,  ",") -> False  # dot needs a digit after it
    """
    # Empty piece is always a valid start (no token emitted yet).
    if not piece:
        return True

    if end in piece:
        # The end marker appeared – the part before it must be a complete number.
        number_part = piece[: piece.index(end)]
        if not number_part:
            return False  # empty is not a number
        if allow_fraction:
            # Complete float: optional minus, integer part, optional decimal.
            pattern = r"-?(?:0|[1-9]\d*)(?:\.\d+)?"
        else:
            # Complete integer: optional minus, no dot.
            pattern = r"-?(?:0|[1-9]\d*)"
        return bool(re.fullmatch(pattern, number_part))
    else:
        # No end marker yet – piece must be a valid *prefix* of a number.
        if allow_fraction:
            # Prefix can end in a dot (waiting for the decimal digits).
            pattern = r"-?(?:0|[1-9]\d*)(?:\.(?:\d+)?)?"
        else:
            pattern = r"-?(?:0|[1-9]\d*)"
        # Also allow a leading minus sign alone (e.g. still typing "-3").
        return bool(re.fullmatch(pattern, piece)) or piece == "-"


# ---------------------------------------------------------------------------
# Legality rule 2 — JSON string
# ---------------------------------------------------------------------------

def _string_close(piece: str) -> int:
    """
    Scan a JSON string starting from the opening quote at index 0 of *piece*.

    Returns:
       index >= 0 : index of the closing quote  (string is complete)
       -1         : string is still open and valid  (keep going)
       -2         : impossible – bad escape sequence detected

    Allowed escape sequences: \\\\ \\" \\/ \\b \\f \\n \\r \\t
    \\uXXXX is intentionally rejected (the prompts are ASCII; we don't
    need it and it complicates boundary detection).

    Examples:
      _string_close('"')            -> -1   (just the opening quote)
      _string_close('"hello')       -> -1   (still open)
      _string_close('"hello"')      -> 6    (closes at index 6)
      _string_close('"a\\\\nb"')    -> 5    (valid \\n escape, closes)
      _string_close('"a\\\\xb"')    -> -2   (\\x is not a valid escape)
      _string_close('"a\\\\u0041"') -> -2   (\\u intentionally disallowed)
      _string_close('"a", "b": ')  -> 2    (closes at 2, rest ignored)
    """
    # Valid single-char escapes after the backslash.
    valid_escapes: set[str] = set('\\"/ bfnrt')

    i = 1  # skip the opening quote at index 0
    while i < len(piece):
        ch = piece[i]
        if ch == "\\":
            if i + 1 >= len(piece):
                # Backslash at the very end – escape is incomplete.
                return -1
            next_ch = piece[i + 1]
            if next_ch == "u":
                return -2  # \uXXXX intentionally disallowed
            if next_ch not in valid_escapes:
                return -2  # unknown escape
            i += 2  # skip backslash + escape character
        elif ch == '"':
            return i  # found the closing quote
        else:
            i += 1  # ordinary character

    return -1  # reached the end without finding a closing quote


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(prompt: str, functions: list[FunctionDefinition]) -> str:
    """
    Build the natural-language preamble shown to the model before the answer.

    The preamble lists the available functions so the model can pick the
    right one.  The answer skeleton starts immediately after "Answer: " so
    the model's first token is already inside the JSON object.
    """
    lines = [
        "You are a function-calling assistant.",
        "Given a user request and the list of available functions below,",
        "output the function call as a JSON object.",
        "",
        "Available functions:",
    ]
    for fn in functions:
        params_desc = ", ".join(
            f"{pname}: {pspec.type}" for pname, pspec in fn.parameters.items()
        )
        lines.append(f"  - {fn.name}({params_desc}): {fn.description}")

    lines.append("")
    lines.append(f"User request: {prompt}")
    lines.append("")
    lines.append("Answer: ")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Constrained decoder
# ---------------------------------------------------------------------------

class ConstrainedDecoder:
    """
    Wraps an LLM and generates valid, schema-correct function-call JSON.

    Construction:
        decoder = ConstrainedDecoder(model)   # model is a Small_LLM_Model

    Usage:
        call = decoder.generate(prompt, functions)
        # call is a FunctionCall(prompt=..., name=..., parameters={...})
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        # Build the id -> real-text mapping once at startup.
        vocab_path = model.get_path_to_vocab_file()
        self._id_to_text: dict[int, str] = Vocabulary(vocab_path).id_to_text

    # ------------------------------------------------------------------
    # Core primitive: pick the best allowed token
    # ------------------------------------------------------------------

    def _best(self, context: str, allowed: Callable[[str], bool]) -> str:
        """
        Ask the model for next-token logits and return the highest-logit
        token whose text satisfies *allowed*.

        This *is* constrained decoding: illegal tokens are conceptually
        set to -∞; we pick the best legal one.

        context : full text seen by the model so far
        allowed : predicate  token_text -> bool
        """
        input_ids: list[int] = self._model.encode(context).tolist()[0]
        logits = np.asarray(
            self._model.get_logits_from_input_ids(input_ids)
        )
        # Iterate from highest logit to lowest.
        for token_id in np.argsort(logits)[::-1]:
            token_text = self._id_to_text.get(int(token_id))
            if token_text and allowed(token_text):
                return token_text
        raise DecodeError("no valid token found – vocabulary exhausted")

    # ------------------------------------------------------------------
    # Choose from a fixed set of options (function names, true/false)
    # ------------------------------------------------------------------

    def _choose(self, context: str, options: list[str]) -> str:
        """
        Decode one token at a time, keeping only tokens that keep the
        accumulated text a prefix of at least one option.

        Stop as soon as the text equals a known option.

        Example: options = ["fn_add_numbers", "fn_greet"]
          step 1: text=""  -> model must emit a prefix of one name
          step 2: text="fn_greet" -> done, return "fn_greet"
        """
        text = ""
        while text not in options:
            # Capture current `text` in the default argument so the lambda
            # does not close over the mutable variable (Python gotcha).
            def allowed(t: str, txt: str = text) -> bool:
                return any(o.startswith(txt + t) for o in options)

            text += self._best(context + text, allowed)
        return text

    # ------------------------------------------------------------------
    # Decode a JSON string value
    # ------------------------------------------------------------------

    def _string(self, context: str) -> tuple[str, str]:
        """
        Decode a JSON string value starting with the opening quote.

        Returns a pair:
          (python_str, json_repr)
          e.g. ("shrek", '"shrek"')

        The model generates tokens inside the string until it emits the
        closing quote.  We allow tokens that cross the boundary (e.g. a
        token that is  shrek"  closes the string) and cut at the quote.
        """
        piece = '"'  # opening quote — always written, never a model choice

        while True:
            close_idx = _string_close(piece)
            if close_idx >= 0:
                # We have the closing quote; extract content and JSON repr.
                python_value = piece[1:close_idx]          # strip both quotes
                json_repr = piece[: close_idx + 1]         # keep both quotes
                return python_value, json_repr

            def allowed(t: str, p: str = piece) -> bool:
                return _string_close(p + t) != -2

            # String still open – get the next token.
            piece += self._best(context + piece, allowed)

    # ------------------------------------------------------------------
    # Decode a JSON number value
    # ------------------------------------------------------------------

    def _number(self, context: str, allow_fraction: bool, end: str) -> str:
        """
        Decode a JSON number, stopping when the terminating character
        *end* ("," between params, "}" for the last one) appears.

        Returns the number text *without* the end marker.
        The model can emit tokens that cross the boundary (e.g. "42,");
        we cut at the first occurrence of *end*.
        """
        piece = ""
        while end not in piece:
            def allowed(t: str, p: str = piece) -> bool:
                return _number_ok(p + t, allow_fraction, end)

            piece += self._best(context + piece, allowed)
        # Cut the end marker off – the caller writes its own structural chars.
        return piece[: piece.index(end)]

    # ------------------------------------------------------------------
    # Dispatch to the right value decoder by JSON type
    # ------------------------------------------------------------------

    def _value(
        self, context: str, ptype: str, last: bool
    ) -> tuple[Any, str]:
        """
        Decode one parameter value and return (python_value, written_text).

        ptype : "string" | "number" | "integer" | "boolean"
        last  : True if this is the last parameter (next char must be "}")
        """
        end = "}" if last else ","

        if ptype == "string":
            return self._string(context)

        if ptype == "boolean":
            word = self._choose(context, ["true", "false"])
            return (word == "true"), word

        # "number" (float) or "integer"
        number_str = self._number(context, ptype == "number", end)

        if ptype == "number":
            return float(number_str), number_str
        else:
            return int(number_str), number_str

    # ------------------------------------------------------------------
    # Generate one complete function call
    # ------------------------------------------------------------------

    def generate(
        self, prompt: str, functions: list[FunctionDefinition]
    ) -> FunctionCall:
        """
        Given a natural-language *prompt* and the available *functions*,
        return a FunctionCall with the chosen name and decoded parameters.

        The JSON is built by construction — correct by design, not by
        parsing and hoping.

        Algorithm overview:
          1. Build the text preamble (functions list + user request).
          2. Let the model choose the function name token by token.
          3. For each parameter (in declaration order):
               - write the literal  "param_name":
               - let the model decode the value (string / number / boolean)
          4. Wrap everything in a FunctionCall object.
        """
        context = build_prompt(prompt, functions)
        by_name: dict[str, FunctionDefinition] = {fn.name: fn for fn in functions}
        fn_names: list[str] = list(by_name.keys())

        # ── Step 1: choose the function name ──────────────────────────
        # We write the fixed prefix  {"name": "  and let the model spell
        # exactly one of the real function names.
        name = self._choose(context + '{"name": "', fn_names)
        fn = by_name[name]

        # ── Step 2: decode each parameter value ───────────────────────
        # We write the structural skeleton and only call the model for
        # the actual values.
        text = context + f'{{"name": "{name}", "parameters": {{'
        params: dict[str, Any] = {}
        items = fn.ordered_parameters()

        for index, (pname, ptype) in enumerate(items):
            is_last = index == len(items) - 1
            separator = "" if index == 0 else ", "
            text += separator + f'"{pname}": '

            value, written = self._value(text, ptype, last=is_last)
            params[pname] = value
            text += written  # keep context growing for the next param

        return FunctionCall(prompt=prompt, name=name, parameters=params)


# ---------------------------------------------------------------------------
# Batch decoding
# ---------------------------------------------------------------------------

def decode_all(
    model: Any,
    prompts: list[str],
    functions: list[FunctionDefinition],
) -> list[FunctionCall]:
    """
    Decode every prompt in *prompts* and return one FunctionCall per prompt.

    A single decoder is reused across all prompts (the vocabulary is loaded
    once at construction time).

    If a single prompt fails (DecodeError or any other exception) a warning
    is printed to stderr and a placeholder entry is emitted so the output
    list stays aligned with the input list — the run never crashes.
    """
    decoder = ConstrainedDecoder(model)
    results: list[FunctionCall] = []
    total = len(prompts)

    for i, prompt_str in enumerate(prompts, start=1):
        try:
            call = decoder.generate(prompt_str, functions)
            results.append(call)
            print(
                f"[{i}/{total}] {call.name}({call.parameters})",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[{i}/{total}] WARNING: could not decode prompt – {exc}",
                file=sys.stderr,
            )
            # Placeholder: keeps output aligned with input.
            # Use the first function with zero-value parameters.
            fallback_fn = functions[0] if functions else None
            placeholder_params: dict[str, Any] = {}
            if fallback_fn:
                for pname, ptype in fallback_fn.ordered_parameters():
                    if ptype == "string":
                        placeholder_params[pname] = ""
                    elif ptype == "boolean":
                        placeholder_params[pname] = False
                    elif ptype == "number":
                        placeholder_params[pname] = 0.0
                    else:
                        placeholder_params[pname] = 0
            results.append(
                FunctionCall(
                    prompt=prompt_str,
                    name=fallback_fn.name if fallback_fn else "",
                    parameters=placeholder_params,
                )
            )

    return results
