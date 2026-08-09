#!/usr/bin/env python3
"""Unit tests for the main.py synthesis dispatcher: switch, cache, fallback.

Uses unittest (not pytest -- pytest is not installed in this project's venv
and must not be pip-installed into it). No real Piper model is loaded and
no audio is played: PiperVoice and subprocess boundaries are mocked
throughout. See the manual verification run (outside this suite) for
checks against the real alba model and real sox playback to a WAV file.
"""

from __future__ import annotations

import os
import threading
import time
import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import main
import phoneme_overrides as po


class FakePiperConfig:
    def __init__(self, phoneme_id_map: dict[str, list[int]], sample_rate: int = 22050) -> None:
        self.phoneme_id_map = phoneme_id_map
        self.sample_rate = sample_rate


class FakePiperVoice:
    """Word-per-chunk fake matching Piper's ' '-delimited clause convention."""

    def __init__(self, phoneme_id_map: dict[str, list[int]] | None = None, sample_rate: int = 22050) -> None:
        self.config = FakePiperConfig(phoneme_id_map or {}, sample_rate)

    def phonemize(self, text: str) -> list[list[str]]:
        clause: list[str] = []
        for i, word in enumerate(text.split()):
            if i > 0:
                clause.append(" ")
            clause.extend(list(word))
        return [clause]

    def phonemes_to_ids(self, phonemes: list[str]) -> list[int]:
        return [max(ord(c) % 250, 1) for c in phonemes]

    def synthesize_ids_to_raw(self, phoneme_ids: list[int], **kwargs: Any) -> bytes:
        return bytes(phoneme_ids)


class BrokenAlignmentFakeVoice(FakePiperVoice):
    """phonemize() always returns one fixed chunk, misaligning multi-word text."""

    def phonemize(self, text: str) -> list[list[str]]:
        return [["a", "b", "c"]]


def _clear_caches() -> None:
    main._voice_cache.clear()
    main._override_cache.clear()
    po.reset_warned_causes()


class SwitchTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_caches()

    @patch("main.speak_cli")
    @patch("main._speak_inprocess")
    def test_switch_unset_defaults_to_inprocess(self, mock_inprocess: MagicMock, mock_cli: MagicMock) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VOICE_MCP_PHONEME_OVERRIDES", None)
            main.speak("hello", "alba", 1.0, 0)
        mock_inprocess.assert_called_once_with("hello", "alba", 1.0, 0)
        mock_cli.assert_not_called()

    @patch("main.speak_cli")
    @patch("main._speak_inprocess")
    def test_switch_zero_forces_cli_path(self, mock_inprocess: MagicMock, mock_cli: MagicMock) -> None:
        with patch.dict(os.environ, {"VOICE_MCP_PHONEME_OVERRIDES": "0"}):
            with self.assertLogs("phoneme_overrides", level="WARNING") as ctx:
                main.speak("hello", "alba", 1.0, 0)
        mock_cli.assert_called_once_with("hello", "alba", 1.0, 0)
        mock_inprocess.assert_not_called()
        self.assertEqual(len(ctx.records), 1)

    @patch("main.speak_cli")
    @patch("main._speak_inprocess")
    def test_switch_falsy_values_all_force_cli(self, mock_inprocess: MagicMock, mock_cli: MagicMock) -> None:
        for value in ("0", "false", "False", "NO", "off", " Off "):
            with self.subTest(value=value):
                mock_inprocess.reset_mock()
                mock_cli.reset_mock()
                with patch.dict(os.environ, {"VOICE_MCP_PHONEME_OVERRIDES": value}):
                    main.speak("hi", "alba", 1.0, 0)
                mock_cli.assert_called_once()
                mock_inprocess.assert_not_called()

    @patch("main.speak_cli")
    @patch("main._speak_inprocess")
    def test_switch_recognized_truthy_values_use_inprocess(self, mock_inprocess: MagicMock, mock_cli: MagicMock) -> None:
        for value in ("1", "true", "True", "YES", " on "):
            with self.subTest(value=value):
                mock_inprocess.reset_mock()
                mock_cli.reset_mock()
                with patch.dict(os.environ, {"VOICE_MCP_PHONEME_OVERRIDES": value}):
                    main.speak("hi", "alba", 1.0, 0)
                mock_inprocess.assert_called_once()
                mock_cli.assert_not_called()

    @patch("main.speak_cli")
    @patch("main._speak_inprocess")
    def test_switch_unrecognized_value_disables_and_warns_loudly(
        self, mock_inprocess: MagicMock, mock_cli: MagicMock
    ) -> None:
        """RULING: a typo in the switch must DISABLE in-process synthesis
        (fail toward the CLI path that has worked for a year), never enable
        it, and must name the bad value in a loud once-per-process warning."""
        with patch.dict(os.environ, {"VOICE_MCP_PHONEME_OVERRIDES": "fasle"}):
            with self.assertLogs("phoneme_overrides", level="WARNING") as ctx:
                main.speak("hi", "alba", 1.0, 0)

        mock_cli.assert_called_once()
        mock_inprocess.assert_not_called()
        self.assertEqual(len(ctx.records), 1)
        self.assertIn("fasle", ctx.records[0].message)

    @patch("main.speak_cli")
    @patch("main._speak_inprocess")
    def test_switch_unrecognized_values_all_force_cli(self, mock_inprocess: MagicMock, mock_cli: MagicMock) -> None:
        for value in ("2", "enabled", "yesplease", ""):
            with self.subTest(value=value):
                mock_inprocess.reset_mock()
                mock_cli.reset_mock()
                with patch.dict(os.environ, {"VOICE_MCP_PHONEME_OVERRIDES": value}):
                    main.speak("hi", "alba", 1.0, 0)
                mock_cli.assert_called_once()
                mock_inprocess.assert_not_called()

    @patch("main.subprocess.Popen")
    def test_cli_path_is_unreachable_only_via_switch_not_deleted(self, mock_popen: MagicMock) -> None:
        """speak_cli must remain a real, callable function (not removed) --
        this calls it directly, bypassing the dispatcher entirely, to prove
        the original code path still exists and still works standalone."""
        piper_proc = MagicMock()
        piper_proc.wait.return_value = 0
        piper_proc.stderr.read.return_value = b""
        sox_proc = MagicMock()
        sox_proc.returncode = 0
        sox_proc.stderr.read.return_value = b""
        mock_popen.side_effect = [piper_proc, sox_proc]

        main.speak_cli("hello", "alba", 1.0, 0)

        self.assertEqual(mock_popen.call_count, 2)


class PiperUnavailableTests(unittest.TestCase):
    """The BLOCKING finding: `from piper import PiperVoice` must not be able
    to take the whole server down. These tests simulate the import having
    failed at module load time (by patching the module-level sentinel that
    records it) and prove `speak` still starts and still produces audio via
    the CLI path -- unconditionally, regardless of the kill switch."""

    def setUp(self) -> None:
        _clear_caches()

    @patch("main.speak_cli")
    @patch("main._speak_inprocess")
    def test_piper_unavailable_forces_cli_and_warns_once(
        self, mock_inprocess: MagicMock, mock_cli: MagicMock
    ) -> None:
        fake_import_error = ImportError("No module named 'piper'")
        with patch.object(main, "_PIPER_IMPORT_ERROR", fake_import_error):
            with self.assertLogs("phoneme_overrides", level="WARNING") as ctx:
                main.speak("hello", "alba", 1.0, 0)
                main.speak("hello again", "alba", 1.0, 0)

        self.assertEqual(mock_cli.call_count, 2)
        mock_inprocess.assert_not_called()
        # Warned once for the whole process, not once per utterance.
        self.assertEqual(len(ctx.records), 1)
        self.assertIn("piper", ctx.records[0].message.lower())

    @patch("main.speak_cli")
    @patch("main._speak_inprocess")
    def test_piper_unavailable_overrides_an_explicitly_enabled_switch(
        self, mock_inprocess: MagicMock, mock_cli: MagicMock
    ) -> None:
        """Even VOICE_MCP_PHONEME_OVERRIDES=1 cannot revive in-process
        synthesis if piper itself never imported -- there is nothing to
        dispatch to."""
        fake_import_error = ImportError("No module named 'piper'")
        with (
            patch.object(main, "_PIPER_IMPORT_ERROR", fake_import_error),
            patch.dict(os.environ, {"VOICE_MCP_PHONEME_OVERRIDES": "1"}),
        ):
            main.speak("hello", "alba", 1.0, 0)

        mock_cli.assert_called_once()
        mock_inprocess.assert_not_called()

    def test_piper_available_in_this_test_environment(self) -> None:
        """Sanity check: in the real dev .venv (where these tests normally
        run), piper IS importable, so `_piper_available` reflects that."""
        self.assertIsNone(main._PIPER_IMPORT_ERROR)
        self.assertTrue(main._piper_available())


class SpeakCliRegressionTests(unittest.TestCase):
    """Freeze speak_cli's subprocess construction so it stays byte-for-byte
    the pre-existing behaviour (QE requirement: the old CLI path must
    survive unmodified)."""

    @patch("main.subprocess.Popen")
    def test_speak_cli_constructs_original_piper_and_sox_commands(self, mock_popen: MagicMock) -> None:
        piper_proc = MagicMock()
        piper_proc.wait.return_value = 0
        piper_proc.stderr.read.return_value = b""
        sox_proc = MagicMock()
        sox_proc.returncode = 0
        sox_proc.stderr.read.return_value = b""
        mock_popen.side_effect = [piper_proc, sox_proc]

        main.speak_cli("hello there", "alba", 1.25, -450)

        self.assertEqual(mock_popen.call_count, 2)
        piper_cmd = mock_popen.call_args_list[0].args[0]
        sox_cmd = mock_popen.call_args_list[1].args[0]

        self.assertIn("--model", piper_cmd)
        self.assertIn(main.VOICE_MAP["alba"], piper_cmd)
        self.assertIn("--length-scale", piper_cmd)
        self.assertIn("1.25", piper_cmd)
        self.assertIn("--output_raw", piper_cmd)

        self.assertIn("-r", sox_cmd)
        self.assertIn("22050", sox_cmd)  # unchanged: CLI path is untouched, still hardcoded
        self.assertIn("pitch", sox_cmd)
        self.assertIn("-450", sox_cmd)

        piper_proc.stdin.write.assert_called_once_with("hello there".encode("utf-8"))
        piper_proc.stdin.close.assert_called_once()


class VoiceCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_caches()

    @patch("main.PiperVoice")
    def test_failed_load_is_not_cached_and_retries_next_call(self, mock_cls: MagicMock) -> None:
        fake_voice = FakePiperVoice()
        mock_cls.load.side_effect = [RuntimeError("boom"), fake_voice]

        with self.assertRaises(RuntimeError):
            main._load_voice("model-path")
        self.assertNotIn("model-path", main._voice_cache)

        voice = main._load_voice("model-path")
        self.assertIs(voice, fake_voice)
        self.assertIn("model-path", main._voice_cache)
        self.assertEqual(mock_cls.load.call_count, 2)

    @patch("main.PiperVoice")
    def test_successful_load_is_cached_and_reused(self, mock_cls: MagicMock) -> None:
        fake_voice = FakePiperVoice()
        mock_cls.load.return_value = fake_voice

        first = main._load_voice("model-path")
        second = main._load_voice("model-path")

        self.assertIs(first, second)
        mock_cls.load.assert_called_once()

    @patch("main.PiperVoice")
    def test_multi_voice_caches_are_independent(self, mock_cls: MagicMock) -> None:
        voice_a = FakePiperVoice(phoneme_id_map={"b": [1], "x": [2]})
        voice_b = FakePiperVoice(phoneme_id_map={"b": [1]})  # no 'x'
        mock_cls.load.side_effect = lambda path: {"path-a": voice_a, "path-b": voice_b}[path]

        with patch.object(po, "load_raw_override_table", return_value={"word": ["b", "x"]}):
            got_a, overrides_a = main._get_voice_and_overrides("path-a")
            got_b, overrides_b = main._get_voice_and_overrides("path-b")

        self.assertIs(got_a, voice_a)
        self.assertIs(got_b, voice_b)
        self.assertEqual(overrides_a, {"word": ["b", "x"]})
        self.assertEqual(overrides_b, {})  # 'x' not in voice_b's map -> word dropped

    def test_concurrent_requests_for_same_model_load_exactly_once(self) -> None:
        fake_voice = FakePiperVoice()

        def slow_load(_path: str) -> FakePiperVoice:
            time.sleep(0.05)
            return fake_voice

        results: list[FakePiperVoice] = []
        results_lock = threading.Lock()

        def worker() -> None:
            voice = main._load_voice("shared-model-path")
            with results_lock:
                results.append(voice)

        with patch("main.PiperVoice") as mock_cls:
            mock_cls.load.side_effect = slow_load
            threads = [threading.Thread(target=worker) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            self.assertEqual(mock_cls.load.call_count, 1)

        self.assertEqual(len(results), 8)
        self.assertTrue(all(r is fake_voice for r in results))


class SpeakFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        _clear_caches()

    @patch("main.speak_cli")
    @patch("main.PiperVoice")
    def test_persistent_load_failure_falls_back_to_cli_on_every_call(
        self, mock_cls: MagicMock, mock_cli: MagicMock
    ) -> None:
        """QE requirement: 'speaks degraded' must hold on the 2nd and 3rd
        utterance after a failure, not only the first."""
        mock_cls.load.side_effect = RuntimeError("model load broken")

        with self.assertLogs("phoneme_overrides", level="WARNING") as ctx:
            for _ in range(3):
                main.speak("hello", "alba", 1.0, 0)

        self.assertEqual(mock_cli.call_count, 3)
        # Warned once per process for this cause, not once per utterance.
        self.assertEqual(len(ctx.records), 1)

    @patch("main._play_raw_audio")
    @patch("main.speak_cli")
    @patch("main.PiperVoice")
    def test_recovers_in_process_after_earlier_failure(
        self, mock_cls: MagicMock, mock_cli: MagicMock, mock_play: MagicMock
    ) -> None:
        fake_voice = FakePiperVoice()
        mock_cls.load.side_effect = [RuntimeError("boom"), fake_voice]

        with patch.object(po, "load_raw_override_table", return_value={}):
            main.speak("hello", "alba", 1.0, 0)  # load fails -> CLI fallback
            main.speak("hello", "alba", 1.0, 0)  # load succeeds -> in-process

        self.assertEqual(mock_cli.call_count, 1)
        self.assertEqual(mock_play.call_count, 1)

    @patch("main._play_raw_audio")
    @patch("main.PiperVoice")
    def test_inprocess_success_produces_nonempty_audio_with_override_applied(
        self, mock_cls: MagicMock, mock_play: MagicMock
    ) -> None:
        fake_voice = FakePiperVoice(phoneme_id_map={"x": [7]})
        mock_cls.load.return_value = fake_voice

        with patch.object(po, "load_raw_override_table", return_value={"barukh": ["x", "x"]}):
            main._speak_inprocess("barukh now", "alba", 1.0, 0)

        mock_play.assert_called_once()
        raw_audio, sample_rate, pitch_shift = mock_play.call_args.args
        self.assertGreater(len(raw_audio), 0)
        self.assertEqual(sample_rate, fake_voice.config.sample_rate)
        self.assertEqual(pitch_shift, 0)

    @patch("main._play_raw_audio")
    @patch("main.PiperVoice")
    def test_alignment_guard_falls_back_to_plain_phonemize_and_still_speaks(
        self, mock_cls: MagicMock, mock_play: MagicMock
    ) -> None:
        fake_voice = BrokenAlignmentFakeVoice(phoneme_id_map={"x": [1]})
        mock_cls.load.return_value = fake_voice

        with (
            patch.object(po, "load_raw_override_table", return_value={"anything": ["x"]}),
            self.assertLogs("phoneme_overrides", level="WARNING") as ctx,
        ):
            main._speak_inprocess("two words here", "alba", 1.0, 0)

        mock_play.assert_called_once()
        self.assertTrue(any("alignment" in r.message.lower() for r in ctx.records))

    @patch("main.speak_cli")
    @patch("main.PiperVoice")
    def test_missing_overrides_file_still_speaks_in_process(self, mock_cls: MagicMock, mock_cli: MagicMock) -> None:
        """End-to-end fail-safe: a missing/unreadable phoneme_overrides.json
        must not block synthesis -- it degrades to no overrides, still
        in-process (exercised here via `load_raw_override_table` returning
        {}, which is exactly what it returns for a missing file -- see
        LoadRawOverrideTableTests.test_missing_file_... in
        test_phoneme_overrides.py for that lower-level guarantee)."""
        fake_voice = FakePiperVoice(phoneme_id_map={})
        mock_cls.load.return_value = fake_voice

        with (
            patch.object(po, "load_raw_override_table", return_value={}),
            patch("main._play_raw_audio") as mock_play,
        ):
            main.speak("hello world", "alba", 1.0, 0)

        mock_cli.assert_not_called()  # missing overrides file degrades within in-process path, not to CLI
        mock_play.assert_called_once()


if __name__ == "__main__":
    unittest.main()
