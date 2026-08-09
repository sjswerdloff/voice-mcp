"""Word-level phoneme overrides for Piper TTS.

Loads a small JSON table (``phoneme_overrides.json`` at the repo root)
mapping lowercase words to explicit Piper phoneme sequences, and splices
those sequences into the normal espeak-driven phonemization output --
leaving every other word's phonemization untouched. This exists because
espeak mispronounces certain transliterated words (e.g. Hebrew loanwords
such as "Barukh" and "Melekh"); fixing that requires substituting phonemes
before synthesis, which is only possible through the Piper Python API.

To add a new override, add another ``"word": ["phoneme", "phoneme", ...]``
entry to phoneme_overrides.json. Words are matched case-insensitively
against whitespace-delimited tokens (surrounding punctuation is stripped
for matching and preserved from the original phonemization).

Fail-safe by design: any problem loading, parsing, or validating the
override table (missing file, invalid JSON, a phoneme not present in a
given model's phoneme_id_map) degrades to "speak without that override"
rather than raising out of the synthesis path. Losing the guttural
pronunciation is a nuisance; losing the ability to speak at all is an
outage. Each distinct cause of degradation is logged once per process
(see `warn_once`) so a regression is never silent, but never floods the
log either.
"""

from __future__ import annotations

import json
import logging
import string
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

DEFAULT_OVERRIDES_PATH = Path(__file__).resolve().parent / "phoneme_overrides.json"

# Punctuation characters that Piper's espeak phonemizer attaches directly to
# the end of a word's phoneme chunk (no intervening ' ' phoneme). Needed to
# strip these off, apply an override, and reattach them.
_TRAILING_PUNCTUATION = frozenset(string.punctuation)

_WARNED_CAUSES: set[tuple[str, ...]] = set()
_warned_lock = threading.Lock()


class PhonemizerLike(Protocol):
    """Structural type for anything with a Piper-compatible phonemize().

    Lets `phonemize_with_overrides` be exercised in tests against a minimal
    fake, without needing a real model loaded, while still being the exact
    interface `PiperVoice` satisfies structurally.
    """

    def phonemize(self, text: str) -> list[list[str]]: ...


class PhonemeAlignmentError(ValueError):
    """Raised when word-token count does not match phoneme-chunk count.

    This is the fail-fast guard for the word-by-word splicing approach: if
    the whitespace-tokenized text does not produce the same number of
    tokens as the phonemizer produced ' '-delimited phoneme chunks, overrides
    cannot be safely aligned to their phoneme spans, and we must not guess.
    Callers should catch this and fall back to unmodified phonemization
    rather than letting it abort synthesis entirely.
    """

    def __init__(self, text: str, chunk_count: int, token_count: int) -> None:
        super().__init__(
            f"Cannot align text to phonemes for {text!r}: "
            f"{chunk_count} phoneme chunk(s) vs {token_count} whitespace token(s)."
        )
        self.text = text
        self.chunk_count = chunk_count
        self.token_count = token_count


def warn_once(key: tuple[str, ...], message: str) -> None:
    """Log `message` at WARNING level the first time `key` is seen this process.

    Subsequent calls with the same key are no-ops. This exists so a
    persistent degradation (missing overrides file, a phoneme absent from a
    model's map, the CLI-fallback switch, an alignment guard trip) produces
    exactly one visible log line instead of flooding the log once per
    utterance, while still guaranteeing the first occurrence is loud.
    """
    with _warned_lock:
        if key in _WARNED_CAUSES:
            return
        _WARNED_CAUSES.add(key)
    logger.warning(message)


def reset_warned_causes() -> None:
    """Clear the once-per-process warning dedupe set. Intended for tests."""
    with _warned_lock:
        _WARNED_CAUSES.clear()


def load_raw_override_table(path: Path = DEFAULT_OVERRIDES_PATH) -> dict[str, list[str]]:
    """Load the raw word -> phoneme-list table from JSON.

    Fail-safe: returns an empty table (logging a warning once) for any I/O
    or parse problem, rather than raising. Callers proceed with zero
    overrides in that case -- synthesis must not be blocked by a broken
    override file.
    """
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        warn_once(
            ("overrides_missing", str(path)),
            f"Phoneme overrides file not found at {path}; proceeding without overrides.",
        )
        return {}
    except (OSError, json.JSONDecodeError):
        warn_once(
            ("overrides_malformed", str(path)),
            f"Phoneme overrides file at {path} is unreadable or malformed; proceeding without overrides.",
        )
        return {}

    if not isinstance(data, dict):
        warn_once(
            ("overrides_not_object", str(path)),
            f"Phoneme overrides file at {path} does not contain a JSON object; proceeding without overrides.",
        )
        return {}

    table: dict[str, list[str]] = {}
    for word, elements in data.items():
        if not isinstance(word, str) or not isinstance(elements, list) or not all(isinstance(e, str) for e in elements):
            warn_once(
                ("overrides_bad_entry", str(path), str(word)),
                f"Skipping malformed override entry for {word!r} in {path}.",
            )
            continue
        table[word.lower()] = elements
    return table


def validate_overrides(
    raw_table: Mapping[str, list[str]],
    phoneme_id_map: Mapping[str, list[int]],
    model_key: str = "<unknown model>",
) -> dict[str, list[str]]:
    """Validate every override word's phoneme elements against phoneme_id_map.

    Each element must either be a key in phoneme_id_map directly, or
    decompose into single Unicode code points that are all valid keys (this
    handles the case where an override is authored as e.g. "ˈɑ" -- a single
    Python string combining a stress mark and a vowel -- when the model's
    map keys phonemes by individual code point). Words that fail validation
    are dropped (with a once-per-process warning naming the word and the
    model) rather than aborting the whole table -- one bad entry must not
    silence every other override, let alone the voice itself. `model_key`
    (typically the model path) is only used to scope the warn-once key, so
    the same word failing on two different models is reported for each.
    """
    corrected: dict[str, list[str]] = {}
    for word, elements in raw_table.items():
        corrected_elements: list[str] = []
        valid = True
        for element in elements:
            if element in phoneme_id_map:
                corrected_elements.append(element)
                continue
            decomposition = list(element)
            if decomposition and all(c in phoneme_id_map for c in decomposition):
                corrected_elements.extend(decomposition)
                continue
            warn_once(
                ("phoneme_not_in_map", model_key, word, element),
                f"Override phoneme {element!r} for word {word!r} is not in the "
                f"phoneme_id_map for model {model_key}; dropping override for {word!r}.",
            )
            valid = False
            break
        if valid:
            corrected[word] = corrected_elements
    return corrected


def load_and_validate_overrides(
    phoneme_id_map: Mapping[str, list[int]],
    model_key: str,
    path: Path = DEFAULT_OVERRIDES_PATH,
) -> dict[str, list[str]]:
    """Convenience wrapper: load the raw table and validate it in one call."""
    raw_table = load_raw_override_table(path)
    if not raw_table:
        return {}
    return validate_overrides(raw_table, phoneme_id_map, model_key)


def _split_on_space_phoneme(clause: list[str]) -> list[list[str]]:
    """Split one phonemized clause into per-word chunks on the ' ' phoneme.

    Piper/espeak represents inter-word whitespace as a literal ' ' element
    in the phoneme list. Trailing punctuation stays attached to the
    preceding word's chunk with no separating ' '.
    """
    chunks: list[list[str]] = []
    current: list[str] = []
    for phoneme in clause:
        if phoneme == " ":
            chunks.append(current)
            current = []
        else:
            current.append(phoneme)
    if current:
        chunks.append(current)
    return chunks


def _join_with_space_phoneme(chunks: list[list[str]]) -> list[str]:
    """Inverse of `_split_on_space_phoneme`."""
    joined: list[str] = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            joined.append(" ")
        joined.extend(chunk)
    return joined


def _normalize_word(token: str) -> str:
    """Strip leading/trailing punctuation and lowercase, for table lookup."""
    return token.strip(string.punctuation).lower()


def _trailing_punctuation_phonemes(chunk: list[str]) -> list[str]:
    """Return the run of trailing punctuation phonemes at the end of a chunk."""
    trailing: list[str] = []
    for phoneme in reversed(chunk):
        if phoneme in _TRAILING_PUNCTUATION:
            trailing.append(phoneme)
        else:
            break
    trailing.reverse()
    return trailing


def phonemize_with_overrides(
    voice: PhonemizerLike, text: str, override_table: Mapping[str, list[str]]
) -> list[list[str]]:
    """Phonemize `text` normally, substituting override phonemes per word.

    Phonemizes the whole sentence exactly as `voice.phonemize` would (so
    unrelated words and inter-word spacing/stress are byte-for-byte whatever
    the normal pipeline produces), splits each clause into per-word chunks
    on the ' ' phoneme, aligns those chunks 1:1 against `text.split()`, and
    only replaces the chunk content for words present in `override_table` --
    preserving any trailing punctuation phoneme(s) taken from the original
    chunk. Words not in the table are passed through completely unmodified,
    so a sentence with zero override words reconstructs to be identical to
    `voice.phonemize(text)`.

    Raises:
        PhonemeAlignmentError: if the whitespace-token count doesn't match
            the phoneme-chunk count, so we refuse to guess an alignment.
            Callers should catch this and fall back to plain phonemization.
    """
    baseline_clauses = voice.phonemize(text)
    word_tokens = text.split()

    all_chunks: list[list[list[str]]] = [_split_on_space_phoneme(clause) for clause in baseline_clauses]
    total_chunks = sum(len(chunks) for chunks in all_chunks)

    if total_chunks != len(word_tokens):
        raise PhonemeAlignmentError(text, total_chunks, len(word_tokens))

    token_iter = iter(word_tokens)
    result_clauses: list[list[str]] = []
    for chunks in all_chunks:
        new_chunks: list[list[str]] = []
        for chunk in chunks:
            token = next(token_iter)
            key = _normalize_word(token)
            override = override_table.get(key)
            if override is None:
                new_chunks.append(chunk)
            else:
                trailing = _trailing_punctuation_phonemes(chunk)
                new_chunks.append(list(override) + trailing)
        result_clauses.append(_join_with_space_phoneme(new_chunks))

    return result_clauses
