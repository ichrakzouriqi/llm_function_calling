*This project has been created as part of the 42 curriculum by izouriqi.*

# call me maybe

> **Function calling for small LLMs using constrained decoding.**
> Given a plain-English question, produce the exact function call that would answer it — right name, right arguments, right types — every single time.

---

## What is this project about?

Imagine asking a small AI model (600 million parameters) to always respond with valid JSON. Left to its own devices, it fails around 50% of the time — it forgets a quote, adds extra text, or writes `2` where `2.0` is required. This project solves that problem using a technique called **constrained decoding**.

Instead of letting the model generate whatever it wants and hoping for the best, we intercept each generation step and filter out any token that would break the output. The model still decides *which* function to call and *what* values to use — we just make it impossible for it to produce invalid JSON.

**Example:**

```
Input  : "What is the sum of 2 and 3?"
Output : {"name": "fn_add_numbers", "parameters": {"a": 2.0, "b": 3.0}}
```

Note: the value `5` is never computed. We only produce the *call*, not the result.

---

## How does it work? (The big picture)

Every function call has this fixed shape:

```
{"name": "<function>", "parameters": {"<arg>": <value>, ...}}
```

The braces, quotes, colons, and parameter names are **never decisions** — they are always the same. Only two things require the model's judgment:

1. **Which function** to call
2. **What each argument value is**

So the strategy is simple: **write the fixed parts in code, ask the model only for the two real decisions.** This keeps things fast, readable, and correct by construction.

### Constrained decoding in plain English

At each step of generation, the model produces a score (called a *logit*) for every token in its vocabulary (~151,000 tokens). Normally you just pick the highest score. With constrained decoding, you first eliminate any token that would make the output invalid, then pick the best among what remains.

```
model logits  →  mask illegal tokens  →  pick best legal token  →  repeat
```

That's it. The model's intelligence is preserved — we just remove the bad options before it chooses.

---

## Project structure

```
call-me-maybe/
├── src/
│   ├── __init__.py       # package marker
│   ├── __main__.py       # CLI: load inputs → run decoding → write output
│   ├── schemas.py        # Pydantic models for input/output validation
│   ├── tokenizer.py      # custom byte-level BPE vocabulary loader
│   └── decoder.py        # constrained decoding engine
├── tests/
│   ├── test_decoder.py   # unit tests for number/string rules + mock model
│   └── test_tokenizer.py # unit tests for byte-level decoding
├── llm_sdk/              # provided SDK (read-only, wraps Qwen3-0.6B)
├── data/
│   └── input/            # example function definitions + test prompts
├── Makefile              # shortcuts: make run / make test / make lint
└── pyproject.toml        # dependencies: numpy, pydantic, llm-sdk
```

---

## Input / Output format

### Input 1 — `functions_definition.json`

A list of callable functions, each with a name, description, typed parameters, and return type:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": {"type": "number"},
      "b": {"type": "number"}
    },
    "returns": {"type": "number"}
  }
]
```

Supported types: `string`, `number` (float), `integer`, `boolean`.

### Input 2 — `function_calling_tests.json`

A list of natural-language prompts:

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Greet Alice"}
]
```

### Output — `function_calling_results.json`

One result per prompt, in order, with exactly three keys:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

> **Type precision matters.** The grader calls the real Python function with your arguments and asserts exact types: `2` (int) would fail a `number` parameter, but `2.0` (float) passes. The decoder coerces values to the schema type automatically.

---

## Module breakdown

### `schemas.py` — Data models

Uses **Pydantic** to validate inputs and outputs:

- `FunctionDefinition` — one entry from `functions_definition.json`
- `FunctionCall` — one output entry (`prompt` / `name` / `parameters`)
- `ordered_parameters()` — returns `(name, type)` pairs in declaration order (order matters for JSON output)

### `tokenizer.py` — Vocabulary loader

The LLM's vocabulary file (`vocab.json`) stores tokens as byte-level BPE encoded strings — not their real text. For example, a space is stored as `Ġ` (U+0120), not ` ` (0x20).

The `Vocabulary` class reverses this encoding using the standard GPT-2 bytes-to-unicode table, producing a direct `token_id → real_text` dictionary. This lets the decoder check, for any candidate token, what text it would actually add.

```python
vocab = Vocabulary("/path/to/vocab.json")
vocab.id_to_text[42]        # → " the"
vocab.decode([0, 1, 2])     # → " hello world!"
```

### `decoder.py` — Constrained decoding engine

The core of the project. Key components:

#### `_number_ok(piece, allow_fraction, end)` — number legality rule

Checks whether the accumulated text is still on track to form a valid JSON number:
- No leading zeros (`06` → rejected)
- A dot must be followed by at least one digit (`2.` → rejected)
- Integers may never contain a dot
- Once the terminating character (`,` or `}`) appears, the number before it must be complete

#### `_string_close(piece)` — string legality rule

Scans a JSON string from its opening quote to find where it closes:
- Returns the index of the closing quote if found
- Returns `-1` if the string is still open (valid prefix)
- Returns `-2` if an invalid escape sequence is detected (`\x`, `\u` are disallowed)

#### `ConstrainedDecoder._best(context, allowed)`

The fundamental primitive. Asks the model for logits, then scans from the highest score downward and returns the **first token whose text satisfies the `allowed` predicate**. This is constrained decoding in ~5 lines.

#### `ConstrainedDecoder._choose(context, options)`

Decodes one token at a time, keeping only tokens that keep the accumulated text a valid prefix of one of the known options (function names or `true`/`false`). Stops when the text matches an option exactly.

#### `ConstrainedDecoder._string(context)` / `_number(context, ...)` / `_value(...)`

Type-specific value decoders. They accept tokens that cross field boundaries (e.g. a token that simultaneously closes a value and starts the next field) and cut at the right position.

#### `ConstrainedDecoder.generate(prompt, functions)`

Orchestrates a full function call:
1. Builds the natural-language preamble
2. Lets the model choose the function name
3. Iterates over parameters in declaration order, decoding each value
4. Returns a `FunctionCall` object — correct by construction, never parsed back

#### `decode_all(model, prompts, functions)`

Batch runner: one decoder reused across all prompts. If a single prompt fails, a warning is printed and a placeholder is emitted — the run never crashes.

---

## How to run

```sh
# Install dependencies (first time only)
make install       # or: python -m uv sync

# Run on the public test set
make run

# Run unit tests (no model needed, fast)
make test

# Check code style
make lint
```

Custom paths:

```sh
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input                data/input/function_calling_tests.json \
    --output               data/output/function_calling_results.json \
    --model                Qwen/Qwen3-0.6B
```

The first run downloads the model weights from Hugging Face (~1.2 GB). Later runs use the local cache.

Expected output on stderr:

```
Loading model Qwen/Qwen3-0.6B ...
[1/11] fn_add_numbers({'a': 2.0, 'b': 3.0})
[2/11] fn_add_numbers({'a': 265.0, 'b': 345.0})
...
Wrote 11 function calls to data/output/function_calling_results.json
```

---

## Testing strategy

Tests run offline — no model, no network needed:

```sh
make test    # runs in ~3 seconds
```

- **`test_decoder.py`** — tests `_number_ok` and `_string_close` directly against known edge cases, then drives `ConstrainedDecoder` with a **mock model** that returns random logits. Whatever the random preferences, the output must always be a valid, schema-correct call with the right types.
- **`test_tokenizer.py`** — verifies that `Ġ` decodes to a space, that ASCII tokens map to themselves, and that `decode()` concatenates correctly.

---

## Design decisions

**Write structure, ask only for decisions.**
The JSON skeleton (`{"name": "`, `", "parameters": {`, etc.) is written directly in code. The model is only called where a real choice exists. This avoids the "generate then re-parse and hope" approach entirely.

**Build the result from choices, not from parsing.**
The `FunctionCall` object is assembled directly from the chosen name and decoded values — it is correct by construction.

**Nothing is hard-coded.**
Function names, parameter names, and types all come from the input file at runtime. The program works with any set of functions.

**Legality rules are small, pure, and testable.**
`_number_ok` and `_string_close` have no model dependency. They can be (and are) unit-tested directly, without loading the LLM.

**Types are enforced by the schema.**
`number → float`, `integer → int`. A dot is rejected for integers. This is what makes the grader's `isinstance(a, float)` assertion pass.

**Allowed imports only.**
`src` uses only `json`, `re`, `numpy`, `pydantic`, and the public SDK API. No `torch`, no `transformers`, no private attributes.

---

## Grading

The moulinette does **not** compare JSON text. For each result it:
1. Checks that the `prompt` matches the expected prompt
2. Looks up the real Python function named by `name`
3. **Calls that function** with the provided `parameters`
4. Compares the return value to the expected answer

Two exercise sets exist: **public** (5 functions / 11 prompts, provided) and **private** (6 different functions / 11 prompts, used at evaluation). The program never hard-codes either — it reads whatever files it is given.

---

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)
- [GPT-2 byte-level BPE encoding](https://github.com/openai/gpt-2/blob/master/src/encoder.py)
- [Outlines — constrained generation background](https://github.com/dottxt-ai/outlines)
