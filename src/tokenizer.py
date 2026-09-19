"""
tokenizer.py – Custom vocabulary loader for byte-level BPE models
(GPT-2 / Qwen3).

Why do we need this?
    The model's vocab.json stores tokens as *Unicode-escaped* strings, not as
    their true text.  For example, a leading space is stored as 'Ġ' (U+0120),
    not as ' ' (0x20).  To apply our legality rules we need the *real* text
    that each token ID would append to the context, so we reverse that encoding
    here and build a direct  id -> text  dictionary.

Byte-level BPE encoding (GPT-2 convention):
    Every byte value 0-255 is mapped to a unique printable Unicode
    character so that the vocabulary file never contains raw control
    bytes.  The mapping is:
      - Printable ASCII bytes (33-126) and a few extra (161-172,
        174-255) map to themselves (same code point).
      - The remaining bytes (0-32, 127-160, 173) map to code points 256, 257, …
        in the order they appear.

    We call this forward map  byte -> unicode_char  "_bytes_to_unicode".
    We need the reverse:  unicode_char -> byte.
"""

from __future__ import annotations

import json


# ---------------------------------------------------------------------------
# GPT-2 byte ↔ unicode table
# ---------------------------------------------------------------------------

def _bytes_to_unicode() -> dict[int, str]:
    """
    Return the canonical GPT-2 byte-to-unicode table.

    Keys   : byte values  0-255  (int)
    Values : printable Unicode characters (str of length 1)

    Printable ASCII (and a few Latin-1 printable) bytes map to themselves.
    All other byte values map to code points starting at U+0100 (256), in
    the order they appear in the range 0-255.
    """
    # Bytes that are already printable — they map to the same Unicode
    # code point.
    printable: list[int] = (
        list(range(ord("!"), ord("~") + 1))   # 33-126  (ASCII printable)
        + list(range(ord("¡"), ord("¬") + 1))  # 161-172 (Latin-1 printable)
        + list(range(ord("®"), ord("ÿ") + 1))  # 174-255 (Latin-1 printable)
    )
    chars: list[str] = [chr(b) for b in printable]

    # All remaining bytes get mapped to code points starting at 256.
    extra_codepoint = 256
    for byte_val in range(256):
        if byte_val not in printable:
            chars.append(chr(extra_codepoint))
            printable.append(byte_val)
            extra_codepoint += 1

    return dict(zip(printable, chars))


# ---------------------------------------------------------------------------
# Vocabulary class
# ---------------------------------------------------------------------------

class Vocabulary:
    """
    Load a vocab.json file and expose a  id_to_text  dictionary.

    Construction:
        vocab = Vocabulary("/path/to/vocab.json")

    The vocab.json format (Qwen3 / GPT-2 style):
        { "token_string": token_id, ... }

    After construction:
        vocab.id_to_text[token_id]  ->  real UTF-8 string
        vocab.decode([id1, id2, …]) ->  concatenated text (unknown ids skipped)
    """

    def __init__(self, vocab_path: str) -> None:
        # Build the reverse mapping: unicode_char -> byte value
        unicode_to_byte: dict[str, int] = {
            uc: b for b, uc in _bytes_to_unicode().items()
        }

        # Load the raw vocabulary: token_string -> token_id
        with open(vocab_path, encoding="utf-8") as fh:
            raw_vocab: dict[str, int] = json.load(fh)

        # Convert each token string to its real UTF-8 text.
        self.id_to_text: dict[int, str] = {}
        for token_str, token_id in raw_vocab.items():
            try:
                # Each character in the token_str is a Unicode symbol that
                # represents exactly one byte in the original byte stream.
                raw_bytes = bytes(unicode_to_byte[ch] for ch in token_str)
                # Decode the byte sequence as UTF-8 to get the real text.
                self.id_to_text[token_id] = raw_bytes.decode("utf-8")
            except (KeyError, UnicodeDecodeError):
                # KeyError  : character not in our mapping (should not happen
                #             for a well-formed vocab, but skip to be safe).
                # UnicodeDecodeError : the bytes do not form valid UTF-8 on
                #             their own (e.g. a partial multi-byte
                #             sequence).  Such tokens are never needed
                #             for ASCII JSON output so we simply omit
                #             them.
                pass

    def decode(self, token_ids: list[int]) -> str:
        """
        Convert a list of token IDs back to a human-readable string.

        Unknown IDs (not in id_to_text) contribute nothing — they are silently
        skipped so the caller never has to guard against KeyError.

        This helper is mainly useful for debugging; the decoder uses
        id_to_text directly for performance.
        """
        return "".join(self.id_to_text.get(tid, "") for tid in token_ids)
