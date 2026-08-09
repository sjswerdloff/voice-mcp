from __future__ import annotations

from fastmcp import FastMCP
import subprocess
import os
import time
import threading
from pathlib import Path
import logging
import math

import phoneme_overrides as po

# `piper` (piper-tts + onnxruntime) is required for the in-process synthesis
# path but NOT for the original CLI subprocess path, which only needs the
# `.venv/bin/piper` binary. Import it defensively: on any host that rebuilds
# its venv from a lockfile missing these packages, a module-scope import
# failure here would take the whole MCP server down before it could even
# start -- including the CLI path, which needs no Python piper at all, and
# the VOICE_MCP_PHONEME_OVERRIDES kill switch, which would never get a
# chance to be read. So: catch it, remember why, and let `speak()` force
# the CLI path (see `_piper_available`) rather than let the server die.
_PIPER_IMPORT_ERROR: Exception | None
try:
    from piper import PiperVoice
except ImportError as _piper_import_exc:
    PiperVoice = None  # type: ignore[assignment, misc]
    _PIPER_IMPORT_ERROR = _piper_import_exc
else:
    _PIPER_IMPORT_ERROR = None


# Initialize FastMCP server
mcp = FastMCP("voice-mcp")

# Set up logging (file only, no stdout to avoid interfering with MCP protocol)
log_file = Path(__file__).parent / "voice-mcp.log"
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler(log_file)]
)


# Load voice mapping from file
def load_voice_mapping():
    """Load voice name to model path mapping from voice_name_map.txt"""
    voice_map = {}
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    voice_file = script_dir / "voice_name_map.txt"

    if voice_file.exists():
        with open(voice_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and "\t" in line:
                    voice_name, model_path = line.split("\t", 1)

                    # Convert relative paths to absolute paths based on project root
                    if model_path.startswith("../"):
                        # Remove the '../' and make it relative to the parent of script_dir
                        relative_path = model_path[3:]  # Remove '../'
                        absolute_path = script_dir.parent / relative_path
                        model_path = str(absolute_path)
                    elif not Path(model_path).is_absolute():
                        # Handle other relative paths by making them relative to script_dir
                        absolute_path = script_dir / model_path
                        model_path = str(absolute_path)

                    voice_map[voice_name] = model_path
    return voice_map


# Get available voices
VOICE_MAP = load_voice_mapping()
AVAILABLE_VOICES = list(VOICE_MAP.keys())


@mcp.tool()
async def list_voices():
    """
    List all available voices that can be used with the use_voice tool.

    Returns:
        A list of available voice names
    """
    if not AVAILABLE_VOICES:
        return "No voices available. Check voice_name_map.txt file."

    return f"Available voices ({len(AVAILABLE_VOICES)}): {', '.join(sorted(AVAILABLE_VOICES))}"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for streaming playback."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


@mcp.tool()
async def use_voice(text: str, voice_name: str, length_scale: float = None, pitch_shift: int = None, echo_text: bool = False):
    """
    This is a basic voice tool that uses piper to say something to the user. Use only if it is asked to speak.
    ONLY USE IF ASKED. The tool receives what it is to be said. Be laconic in your speech.

    Args:
        text: Content of what the user will listen. Speech should be concise
        voice_name: Name of the voice to use. Available voices: {', '.join(AVAILABLE_VOICES)}
        length_scale: Speech speed control (higher = slower). Default 1.0 for normal speed,
                     except vctk which defaults to 2.0 for deliberate speech. Use 1.5 for slightly slower,
                     or values above 2.0 for even slower speech.
        pitch_shift: Pitch adjustment in cents (100 cents = 1 semitone). Negative values lower pitch,
                    positive values raise pitch. Default None (no pitch change). Useful for voices
                    that are too high-pitched (try -200 to -400 for lower pitch).
    """
    if voice_name not in VOICE_MAP:
        return f"Error: Voice '{voice_name}' not found. Available voices: {', '.join(AVAILABLE_VOICES)}"

    # Set default length_scale based on voice
    if length_scale is None or isinstance(length_scale, float) and math.isnan(length_scale):
        if voice_name == "vctk":
            length_scale = 2.0
        elif voice_name == "southern_english_female":
            length_scale = 1.5
        else:
            length_scale = 1.0

    # Set default pitch_shift based on voice (southern_english_female tends to be high-pitched)
    if pitch_shift is None:
        pitch_shift = -450 if voice_name == "southern_english_female" else 0

    st = time.time()
    first_chunk_time = None
    sentences = _split_sentences(text)

    for i, sentence in enumerate(sentences):
        speak(sentence, voice_name, length_scale, pitch_shift)
        if i == 0:
            first_chunk_time = time.time() - st

    params = f"length_scale={length_scale}"
    if pitch_shift != 0:
        params += f", pitch_shift={pitch_shift}"
    if echo_text:
        return text
    return f"Spoke {len(sentences)} chunks with voice '{voice_name}' ({params}), first in {first_chunk_time:.2f}s, total {time.time() - st:.2f}s"


def speak_cli(text: str, voice_name: str, length_scale: float = 1.0, pitch_shift: int = 0):
    """Original synthesis path: shell out to the piper CLI binary, pipe into sox.

    This is the pre-existing, byte-for-byte-unchanged implementation. It is
    kept as a live fallback for two reasons: (1) it is reachable without a
    code edit via the VOICE_MCP_PHONEME_OVERRIDES=0 kill switch (see
    `_in_process_enabled` / `speak`), so an in-process regression can be
    rolled back per-host by flipping one environment variable; and (2) the
    in-process pipeline falls back to this automatically if it fails (see
    `speak`), so a broken model load or ONNX runtime error never silences
    the assistant -- it just loses the phoneme overrides for that call.
    """
    model_path = VOICE_MAP[voice_name]

    # Get the project root directory (where main.py is located) and find piper in the venv
    project_root = Path(__file__).parent
    piper_path = project_root / ".venv" / "bin" / "piper"

    # Fallback to system piper if venv piper doesn't exist
    if not piper_path.exists():
        piper_path = "piper"

    logging.info(f"Starting piper with model: {model_path}, length_scale: {length_scale}")

    # Start Piper subprocess with required model parameter
    # Use configurable length-scale for speech speed control
    piper_proc = subprocess.Popen(
        [str(piper_path), "--model", model_path, "--length-scale", str(length_scale), "--output_raw"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Start Sox to play audio with optional pitch shifting
    # Try to find sox in common locations
    sox_paths = ["/opt/homebrew/bin/sox", "/usr/local/bin/sox", "/usr/bin/sox", "sox"]
    sox_path = "sox"  # default fallback

    for path in sox_paths:
        if Path(path).exists() or path == "sox":
            sox_path = path
            break

    sox_cmd = [sox_path, "-t", "raw", "-r", "22050", "-b", "16", "-e", "signed-integer", "-c", "1", "-", "-d"]
    if pitch_shift != 0:
        sox_cmd.extend(["pitch", str(pitch_shift)])

    logging.info(f"Starting sox with command: {' '.join(sox_cmd)}")
    logging.info(f"Using sox path: {sox_path}")

    sox_proc = subprocess.Popen(sox_cmd, stdin=piper_proc.stdout, stderr=subprocess.PIPE)

    # Send the text to Piper
    piper_proc.stdin.write(text.encode("utf-8"))
    piper_proc.stdin.close()

    # Wait for it to finish and capture any errors
    sox_proc.wait()
    piper_return_code = piper_proc.wait()

    # Log any errors
    if piper_return_code != 0:
        piper_stderr = piper_proc.stderr.read().decode("utf-8")
        logging.error(f"Piper error (code {piper_return_code}): {piper_stderr}")

    if sox_proc.returncode != 0:
        sox_stderr = sox_proc.stderr.read().decode("utf-8")
        logging.error(f"Sox error (code {sox_proc.returncode}): {sox_stderr}")

    logging.info(f"Piper return code: {piper_return_code}, Sox return code: {sox_proc.returncode}")


# --- In-process synthesis path (phoneme overrides) ------------------------
#
# Runtime kill switch: VOICE_MCP_PHONEME_OVERRIDES. Defaults to enabled (the
# new in-process path) when UNSET. Set to an explicit falsy value
# ("0"/"false"/"no"/"off", case-insensitive) on a given host to force the
# original CLI subprocess path -- byte-for-byte unchanged -- without
# touching code. This is the rollback lever: a problem specific to
# in-process synthesis is one environment variable away from being
# neutralized on that host, without redeploying or silencing every other
# host.
#
# Unrecognised values DISABLE the in-process path (fail toward the CLI path
# that has worked for a year), rather than enabling it. Rationale: this
# switch exists to be flipped during an outage by someone who may be
# stressed and may mistype it ("fasle", "0ff"); a typo silently being
# treated as "enabled" would mean the person believes they rolled back and
# they did not. The boring, previously-working path is the safe default for
# an unrecognised value, not the new one. An unrecognised value is always
# logged loudly (once) naming exactly what was read.
_TRUTHY_SWITCH_VALUES = {"1", "true", "yes", "on"}
_FALSY_SWITCH_VALUES = {"0", "false", "no", "off"}


def _in_process_enabled() -> bool:
    """Read the VOICE_MCP_PHONEME_OVERRIDES kill switch from the environment.

    Read fresh on every call (not cached at import time) so the switch can
    be exercised deterministically in tests via env var patching, and so a
    supervisor that restarts this process with a changed environment takes
    effect immediately rather than requiring a code path that notices.

    Truth table:
      - unset                    -> True  (approved default: in-process)
      - recognised truthy value  -> True
      - recognised falsy value   -> False
      - anything else (a typo)   -> False, logged loudly once
    """
    raw = os.environ.get("VOICE_MCP_PHONEME_OVERRIDES")
    if raw is None:
        return True

    value = raw.strip().lower()
    if value in _TRUTHY_SWITCH_VALUES:
        return True
    if value in _FALSY_SWITCH_VALUES:
        po.warn_once(
            ("switch_disabled",),
            f"VOICE_MCP_PHONEME_OVERRIDES={raw!r} disables in-process synthesis; "
            "using the CLI piper subprocess path (phoneme overrides are not applied).",
        )
        return False

    po.warn_once(
        ("switch_unrecognized_value", raw),
        f"VOICE_MCP_PHONEME_OVERRIDES={raw!r} is not a recognised value "
        f"(truthy: {sorted(_TRUTHY_SWITCH_VALUES)}, falsy: {sorted(_FALSY_SWITCH_VALUES)}); "
        "treating it as DISABLED (fail toward the CLI path) and using the CLI piper "
        "subprocess path (phoneme overrides are not applied) until this is corrected.",
    )
    return False


class InProcessSynthesisError(Exception):
    """Raised when the in-process phoneme-override pipeline cannot produce audio.

    `speak` catches this specific exception (never a bare Exception) and
    falls back to `speak_cli`, so a problem confined to the in-process
    pipeline -- a corrupted model file, an ONNX runtime error, an
    unexpected exception during phonemization -- degrades to "overrides
    lost for this call" rather than "the assistant is silent."
    """

    def __init__(self, voice_name: str, cause: Exception) -> None:
        super().__init__(f"In-process synthesis failed for voice {voice_name!r}: {cause}")
        self.voice_name = voice_name
        self.cause = cause


# Voice and validated-override-table caches, keyed by model path. Guarded by
# a single lock rather than one lock per model path: this trades a little
# parallelism (two different, not-yet-cached models loading concurrently
# will serialize behind each other) for an implementation simple enough to
# reason about correctly. Failure is never cached -- see `_load_voice`.
_voice_cache: dict[str, PiperVoice] = {}
_override_cache: dict[str, dict[str, list[str]]] = {}
_cache_lock = threading.Lock()


def _load_voice(model_path: str) -> PiperVoice:
    """Load (or return the cached) PiperVoice for `model_path`.

    Thread-safety: loading and inserting into `_voice_cache` is guarded by
    `_cache_lock`, so concurrent callers requesting the same not-yet-cached
    model block on a single load rather than racing to load it twice. Once
    cached, concurrent *use* of the same PiperVoice instance (calling
    `synthesize_ids_to_raw` from multiple threads) is safe without further
    locking: that method only reads `self.config` and calls
    `self.session.run(...)`, never mutating instance state, and
    onnxruntime's InferenceSession.run() is documented as safe to call
    concurrently from multiple threads. Verified by reading this piper
    version's `synthesize_ids_to_raw` source directly (no writes to
    `self.session` or `self.config`) rather than assumed.

    UNESTABLISHED, deliberately written down as such: whether FastMCP's
    stdio transport ever actually dispatches `speak()` from more than one
    thread concurrently. Nothing in this codebase was traced to confirm or
    rule that out. The lock is kept regardless -- it costs nothing when
    uncontended -- but its presence should not be read as proof concurrent
    calls happen, and its removal should not be treated as proven safe
    without first establishing what FastMCP actually does here.

    Failure is never cached: if `PiperVoice.load` raises, this function
    re-raises without touching `_voice_cache`, so the very next call
    attempts a fresh load instead of staying poisoned for the life of the
    process.
    """
    with _cache_lock:
        voice = _voice_cache.get(model_path)
        if voice is not None:
            return voice
        voice = PiperVoice.load(model_path)
        _voice_cache[model_path] = voice
        return voice


def _load_overrides_for_voice(voice: PiperVoice, model_path: str) -> dict[str, list[str]]:
    """Load and validate the override table against `voice`'s phoneme_id_map.

    Cached per model path, since different models can have different
    phoneme_id_maps and therefore different valid/invalid override subsets.
    Never raises -- `load_and_validate_overrides` is fail-safe internally.
    """
    with _cache_lock:
        cached = _override_cache.get(model_path)
        if cached is not None:
            return cached
        table = po.load_and_validate_overrides(voice.config.phoneme_id_map, model_key=model_path)
        _override_cache[model_path] = table
        return table


def _get_voice_and_overrides(model_path: str) -> tuple[PiperVoice, dict[str, list[str]]]:
    voice = _load_voice(model_path)
    overrides = _load_overrides_for_voice(voice, model_path)
    return voice, overrides


def _play_raw_audio(raw_audio: bytes, sample_rate: int, pitch_shift: int) -> None:
    """Pipe raw PCM16 mono audio into the existing sox subprocess.

    Sample rate is passed in per-call (from the loaded model's config)
    rather than hardcoded, since not every Piper model uses the same rate.
    Sox failures are logged, not raised -- matching `speak_cli`'s existing
    behaviour, and consistent with the fact that sox is shared infrastructure
    for both paths: if sox itself is broken, falling back to the CLI path
    would not help either.
    """
    sox_paths = ["/opt/homebrew/bin/sox", "/usr/local/bin/sox", "/usr/bin/sox", "sox"]
    sox_path = "sox"

    for path in sox_paths:
        if Path(path).exists() or path == "sox":
            sox_path = path
            break

    sox_cmd = [sox_path, "-t", "raw", "-r", str(sample_rate), "-b", "16", "-e", "signed-integer", "-c", "1", "-", "-d"]
    if pitch_shift != 0:
        sox_cmd.extend(["pitch", str(pitch_shift)])

    logging.info(f"Starting sox (in-process path) with command: {' '.join(sox_cmd)}")

    sox_proc = subprocess.Popen(sox_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    _, sox_stderr = sox_proc.communicate(input=raw_audio)

    if sox_proc.returncode != 0:
        logging.error(f"Sox error (code {sox_proc.returncode}): {sox_stderr.decode('utf-8')}")

    logging.info(f"Sox return code: {sox_proc.returncode}")


def _speak_inprocess(text: str, voice_name: str, length_scale: float, pitch_shift: int) -> None:
    """Synthesize in-process via the Piper Python API, applying phoneme overrides.

    Any failure in loading the model, phonemizing, converting to ids, or
    running inference is wrapped in `InProcessSynthesisError` and re-raised
    (fail-fast internally) so `speak` can catch that one specific type and
    fall back to `speak_cli`. Sox playback is deliberately outside that
    boundary -- see `_play_raw_audio`.
    """
    model_path = VOICE_MAP[voice_name]
    try:
        voice, override_table = _get_voice_and_overrides(model_path)

        if override_table:
            try:
                phoneme_clauses = po.phonemize_with_overrides(voice, text, override_table)
            except po.PhonemeAlignmentError:
                po.warn_once(
                    ("alignment_error", voice_name),
                    f"Phoneme alignment guard tripped for voice {voice_name!r}; "
                    "synthesizing without word-level overrides for the affected utterance(s).",
                )
                phoneme_clauses = voice.phonemize(text)
        else:
            phoneme_clauses = voice.phonemize(text)

        phoneme_ids: list[int] = []
        for clause in phoneme_clauses:
            phoneme_ids.extend(voice.phonemes_to_ids(clause))

        raw_audio = voice.synthesize_ids_to_raw(phoneme_ids, length_scale=length_scale)
        sample_rate = voice.config.sample_rate
    except Exception as exc:
        raise InProcessSynthesisError(voice_name, exc) from exc

    _play_raw_audio(raw_audio, sample_rate, pitch_shift)


def _piper_available() -> bool:
    """Whether `import piper` succeeded at module load time.

    If False, the in-process path cannot run under any switch setting --
    `speak` forces the CLI path unconditionally, so a host that rebuilt its
    venv from a lockfile missing piper-tts/onnxruntime still starts and
    still speaks, just without phoneme overrides.
    """
    return _PIPER_IMPORT_ERROR is None


def speak(text: str, voice_name: str, length_scale: float = 1.0, pitch_shift: int = 0) -> None:
    """Synthesize `text` in `voice_name`'s voice and play it via sox.

    Dispatches between the two synthesis paths:
      - `piper` importable AND VOICE_MCP_PHONEME_OVERRIDES enabled (default):
        in-process synthesis with phoneme overrides (`_speak_inprocess`),
        falling back to `speak_cli` if the in-process pipeline fails for any
        reason.
      - `piper` not importable, OR VOICE_MCP_PHONEME_OVERRIDES disabled (or
        set to an unrecognised value): `speak_cli` directly, i.e. the
        original, unmodified CLI subprocess path.
    """
    if not _piper_available():
        po.warn_once(
            ("piper_unimportable",),
            f"piper package is not importable ({_PIPER_IMPORT_ERROR!r}); using the CLI "
            "piper subprocess path for all calls (phoneme overrides are not applied). "
            "Declare piper-tts/piper-phonemize-cross/onnxruntime/numpy in pyproject.toml "
            "and regenerate uv.lock (`uv lock`) to restore in-process synthesis.",
        )
        speak_cli(text, voice_name, length_scale, pitch_shift)
        return

    if not _in_process_enabled():
        speak_cli(text, voice_name, length_scale, pitch_shift)
        return

    try:
        _speak_inprocess(text, voice_name, length_scale, pitch_shift)
    except InProcessSynthesisError as exc:
        po.warn_once(
            ("inprocess_fallback", voice_name),
            f"In-process synthesis is failing for voice {voice_name!r} ({exc.cause!r}); "
            "falling back to the CLI piper subprocess path (phoneme overrides are not applied "
            "for this call). Will retry in-process on the next call.",
        )
        speak_cli(text, voice_name, length_scale, pitch_shift)


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
