# Voice MCP

A FastMCP v2 server that provides text-to-speech capabilities using Piper TTS with configurable voice selection, speech speed, and pitch control.

## Features

- **9 Available Voices**: British English voices including alan, alba, aru, cori, jenny_dioco, northern_english_male, semaine, southern_english_female, and vctk
- **Configurable Speech Speed**: Control speech rate with `length_scale` parameter (higher = slower)
- **Pitch Control**: Adjust voice pitch with `pitch_shift` parameter (in cents, negative = lower pitch)
- **Smart Defaults**: Optimized settings for specific voices (vctk defaults to slower speech, southern_english_female gets pitch correction)
- **Dynamic Voice Loading**: Voices loaded from `voice_name_map.txt` file for easy expansion

## Tools

### `list_voices()`
Lists all available voices that can be used with the `use_voice` tool.

### `use_voice(text: str, voice_name: str, length_scale: float = None, pitch_shift: int = None)`
Speaks the provided text using the specified voice with optional speed and pitch adjustments.

**Parameters:**
- `text`: Content to be spoken (keep it concise)
- `voice_name`: Name of the voice to use (see `list_voices()` for options)
- `length_scale`: Speech speed control (default: 1.0 for most voices, 2.0 for vctk, 1.5 for southern_english_female)
- `pitch_shift`: Pitch adjustment in cents (default: 0 for most voices, -450 for southern_english_female)

## Installation

### Prerequisites

**macOS users**: Install Sox using Homebrew:
```bash
brew install sox
```

**Linux users**: Install Sox using your package manager:
```bash
# Ubuntu/Debian
sudo apt-get install sox

# CentOS/RHEL
sudo yum install sox
```

### Python Dependencies

1. Install the main dependencies:
```bash
uv pip install -r requirements.txt
```

2. Install Piper TTS (requires `--no-deps` to avoid dependency conflicts):
```bash
uv pip install piper-tts --no-deps piper-phonemize-cross onnxruntime numpy
```

Note: The `--no-deps` flag is critical for piper-tts installation to prevent dependency conflicts.

## Usage

### As an MCP Server

Add to your Claude Code MCP configuration:

```bash
claude mcp add voice-mcp uvx --from /path/to/voice-mcp python main.py
```

### Testing

Run the test script to verify installation:
```bash
python test_voice.py
```

## Voice Configuration

Voices are configured in `voice_name_map.txt` with the format:
```
voice_name	/path/to/voice/model.onnx
```

## Examples

```python
# List available voices
list_voices()

# Basic usage with default settings
use_voice("Hello world", "alan")

# Custom speech speed
use_voice("Slower speech", "cori", length_scale=1.5)

# Custom pitch (lower pitch)
use_voice("Deeper voice", "alan", pitch_shift=-300)

# Both speed and pitch
use_voice("Custom speech", "jenny_dioco", length_scale=1.3, pitch_shift=-200)
```
