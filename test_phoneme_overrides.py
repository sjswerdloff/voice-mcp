#!/usr/bin/env python3
"""Unit tests for phoneme_overrides.py.

Uses unittest (not pytest -- pytest is not installed in this project's venv
and must not be pip-installed into it). Exercises the splice/validate logic
against fakes only; no real Piper model is loaded here (see
test_phoneme_synthesis.py for the main.py-level integration tests, and the
manual verification script for real-model checks).
"""

from __future__ import annotations

import json
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import ClassVar

import phoneme_overrides as po


class FakeVoice:
    """Minimal PhonemizerLike fake: splits words on spaces into fake phonemes.

    Mimics Piper's real convention closely enough for splice-logic testing:
    each word becomes one phoneme per character plus a literal ' ' phoneme
    between words, with trailing punctuation as its own trailing phoneme(s).
    """

    def phonemize(self, text: str) -> list[list[str]]:
        clause: list[str] = []
        for i, word in enumerate(text.split()):
            if i > 0:
                clause.append(" ")
            clause.extend(list(word))
        return [clause]


class BrokenAlignmentVoice:
    """A fake whose phonemize() deliberately produces a chunk/token mismatch."""

    def phonemize(self, text: str) -> list[list[str]]:
        # Always returns a single fixed chunk regardless of input, so any
        # multi-word input misaligns against text.split().
        return [["a", "b", "c"]]


class PhonemizeWithOverridesTests(unittest.TestCase):
    def test_transparent_for_sentence_with_no_override_words(self) -> None:
        """Contract: zero override words -> output identical to plain phonemize."""
        voice = FakeVoice()
        text = "the quick brown fox"
        baseline = voice.phonemize(text)
        spliced = po.phonemize_with_overrides(voice, text, {"barukh": ["X"]})
        self.assertEqual(baseline, spliced)

    def test_applies_override_for_matched_word(self) -> None:
        voice = FakeVoice()
        text = "say barukh now"
        override_table = {"barukh": ["b", "ɑ", "x", "x"]}
        spliced = po.phonemize_with_overrides(voice, text, override_table)
        # chunks: "say" -> s,a,y ; "barukh" -> override ; "now" -> n,o,w
        expected = ["s", "a", "y", " ", "b", "ɑ", "x", "x", " ", "n", "o", "w"]
        self.assertEqual(spliced, [expected])

    def test_override_matching_is_case_insensitive_and_punctuation_stripped(self) -> None:
        voice = FakeVoice()
        text = "Barukh, said the rabbi."
        override_table = {"barukh": ["X", "X"]}
        spliced = po.phonemize_with_overrides(voice, text, override_table)
        # "Barukh," -> override phonemes + trailing ',' phoneme preserved
        self.assertEqual(spliced[0][:3], ["X", "X", ","])

    def test_raises_alignment_error_on_chunk_token_mismatch(self) -> None:
        voice = BrokenAlignmentVoice()
        with self.assertRaises(po.PhonemeAlignmentError):
            po.phonemize_with_overrides(voice, "two words", {})

    def test_words_not_in_table_pass_through_unmodified(self) -> None:
        voice = FakeVoice()
        text = "hello world"
        spliced = po.phonemize_with_overrides(voice, text, {"barukh": ["X"]})
        self.assertEqual(spliced, voice.phonemize(text))


class LoadRawOverrideTableTests(unittest.TestCase):
    def setUp(self) -> None:
        po.reset_warned_causes()

    def test_missing_file_returns_empty_table_and_warns_once(self) -> None:
        missing = Path("/nonexistent/does/not/exist/overrides.json")
        with self.assertLogs("phoneme_overrides", level="WARNING") as ctx:
            table = po.load_raw_override_table(missing)
        self.assertEqual(table, {})
        self.assertEqual(len(ctx.records), 1)
        self.assertIn("not found", ctx.records[0].message)

    def test_malformed_json_returns_empty_table(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "overrides.json"
            path.write_text("{not valid json", encoding="utf-8")
            with self.assertLogs("phoneme_overrides", level="WARNING"):
                table = po.load_raw_override_table(path)
            self.assertEqual(table, {})

    def test_unreadable_path_returns_empty_table(self) -> None:
        # A directory is not a readable JSON file; open() raises IsADirectoryError
        # (an OSError subclass), exercising the OSError branch distinctly from
        # the JSONDecodeError branch.
        with TemporaryDirectory() as tmp:
            directory_as_path = Path(tmp)
            with self.assertLogs("phoneme_overrides", level="WARNING"):
                table = po.load_raw_override_table(directory_as_path)
            self.assertEqual(table, {})

    def test_json_that_is_not_an_object_returns_empty_table(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "overrides.json"
            path.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
            with self.assertLogs("phoneme_overrides", level="WARNING"):
                table = po.load_raw_override_table(path)
            self.assertEqual(table, {})

    def test_malformed_entry_is_skipped_but_valid_entries_survive(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "overrides.json"
            path.write_text(
                json.dumps({"good": ["a", "b"], "bad": "not-a-list", "alsobad": [1, 2]}),
                encoding="utf-8",
            )
            with self.assertLogs("phoneme_overrides", level="WARNING"):
                table = po.load_raw_override_table(path)
            self.assertEqual(table, {"good": ["a", "b"]})

    def test_valid_file_loads_all_entries_lowercased(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "overrides.json"
            path.write_text(json.dumps({"Barukh": ["b", "x"]}), encoding="utf-8")
            table = po.load_raw_override_table(path)
            self.assertEqual(table, {"barukh": ["b", "x"]})

    def test_the_shipped_repo_file_loads_and_matches_approved_values(self) -> None:
        """The actual phoneme_overrides.json at repo root must load cleanly
        and its values must be exactly what was approved -- this test fails
        loudly if anyone alters the approved phoneme sequences."""
        table = po.load_raw_override_table(po.DEFAULT_OVERRIDES_PATH)
        self.assertEqual(
            table,
            {
                "barukh": ["b", "ɑ", "ː", "ɹ", "ˈ", "u", "x", "x"],
                "ata": ["ɐ", "t", "ˈ", "ɑ", "ː"],
                "eloheinu": ["ɪ", "l", "ə", "ʊ", "h", "ˈ", "e", "ɪ", "n", "ˌ", "u", "ː"],
                "melekh": ["m", "ˈ", "ɛ", "l", "ɛ", "x", "x"],
                # Added after the first four, on Vivian's finding that the table stopped
                # short: these two carry no guttural, so unlike barukh/melekh they are
                # voice-independent -- they use only phonemes English models actually
                # produce. Stress corrections only. Acoustically confirmed on alba.
                "adonai": ["ɐ", "d", "ə", "n", "ˈ", "a", "ɪ"],
                "ha'olam": ["h", "ɐ", "ə", "ʊ", "l", "ˈ", "a", "m"],
            },
        )


class ValidateOverridesTests(unittest.TestCase):
    PHONEME_ID_MAP: ClassVar[dict[str, list[int]]] = {
        "b": [1], "a": [2], "x": [3], "ˈ": [4], "ɑ": [5],
    }

    def setUp(self) -> None:
        po.reset_warned_causes()

    def test_valid_elements_kept_as_is(self) -> None:
        raw = {"word": ["b", "a", "x"]}
        result = po.validate_overrides(raw, self.PHONEME_ID_MAP, model_key="model-a")
        self.assertEqual(result, {"word": ["b", "a", "x"]})

    def test_multi_codepoint_element_decomposed_when_parts_valid(self) -> None:
        raw = {"word": ["ˈɑ", "b"]}  # single string combining two valid keys
        result = po.validate_overrides(raw, self.PHONEME_ID_MAP, model_key="model-a")
        self.assertEqual(result, {"word": ["ˈ", "ɑ", "b"]})

    def test_word_with_invalid_phoneme_is_dropped_but_others_survive(self) -> None:
        raw = {"good": ["b", "a"], "bad": ["b", "z"]}  # 'z' invalid, doesn't decompose
        with self.assertLogs("phoneme_overrides", level="WARNING") as ctx:
            result = po.validate_overrides(raw, self.PHONEME_ID_MAP, model_key="model-a")
        self.assertEqual(result, {"good": ["b", "a"]})
        self.assertNotIn("bad", result)
        self.assertTrue(any("bad" in r.message for r in ctx.records))

    def test_invalid_phoneme_warning_is_scoped_per_model(self) -> None:
        """Same bad word on two different models should each warn once (not
        be suppressed by the other model's already-fired warning)."""
        raw = {"bad": ["z"]}
        with self.assertLogs("phoneme_overrides", level="WARNING") as ctx:
            po.validate_overrides(raw, self.PHONEME_ID_MAP, model_key="model-a")
            po.validate_overrides(raw, self.PHONEME_ID_MAP, model_key="model-b")
        self.assertEqual(len(ctx.records), 2)

    def test_repeated_validation_same_model_warns_only_once(self) -> None:
        raw = {"bad": ["z"]}
        logger = logging.getLogger("phoneme_overrides")
        with self.assertLogs("phoneme_overrides", level="WARNING") as ctx:
            po.validate_overrides(raw, self.PHONEME_ID_MAP, model_key="model-a")  # logs once
            po.validate_overrides(raw, self.PHONEME_ID_MAP, model_key="model-a")  # suppressed
            logger.warning("sentinel")  # guarantees the context manager sees >=1 record
        self.assertEqual([r.message for r in ctx.records], [ctx.records[0].message, "sentinel"])
        self.assertEqual(len(ctx.records), 2)

    def test_multi_voice_different_maps_produce_different_corrected_tables(self) -> None:
        """Contract: validating the same raw table against two different
        models' phoneme_id_maps must not share state -- each model gets its
        own correct/incorrect determination."""
        raw = {"word": ["b", "x"]}
        map_with_x = {"b": [1], "x": [2]}
        map_without_x = {"b": [1]}

        result_a = po.validate_overrides(raw, map_with_x, model_key="voice-a")
        result_b = po.validate_overrides(raw, map_without_x, model_key="voice-b")

        self.assertEqual(result_a, {"word": ["b", "x"]})
        self.assertEqual(result_b, {})


class WarnOnceTests(unittest.TestCase):
    def setUp(self) -> None:
        po.reset_warned_causes()

    def test_same_key_logs_only_first_time(self) -> None:
        logger = logging.getLogger("phoneme_overrides")
        with self.assertLogs("phoneme_overrides", level="WARNING") as ctx:
            po.warn_once(("cause", "detail"), "first")
            po.warn_once(("cause", "detail"), "second")  # suppressed
            logger.warning("sentinel")  # guarantees at least one record even if both suppressed
        self.assertEqual([r.message for r in ctx.records], ["first", "sentinel"])

    def test_different_keys_each_log(self) -> None:
        with self.assertLogs("phoneme_overrides", level="WARNING") as ctx:
            po.warn_once(("cause-a",), "a")
            po.warn_once(("cause-b",), "b")
        self.assertEqual([r.message for r in ctx.records], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
