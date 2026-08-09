from fastmcp import FastMCP
import subprocess, os
import time
from pathlib import Path

# Initialize FastMCP server
mcp = FastMCP("voice-mcp")

# Load voice mapping from file
def load_voice_mapping():
    """Load voice name to model path mapping from voice_name_map.txt"""
    voice_map = {}
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    voice_file = script_dir / "voice_name_map.txt"
    
    if voice_file.exists():
        with open(voice_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and '\t' in line:
                    voice_name, model_path = line.split('\t', 1)
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

@mcp.tool()
async def use_voice(text: str, voice_name: str, length_scale: float = None, pitch_shift: int = None):
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
    if length_scale is None:
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
    speak(text, voice_name, length_scale, pitch_shift)

    params = f"length_scale={length_scale}"
    if pitch_shift != 0:
        params += f", pitch_shift={pitch_shift}"
    
    return f"Speaking '{text}' with voice '{voice_name}' ({params}) took {time.time()-st:.2f} sec(s)"

def speak(text: str, voice_name: str, length_scale: float = 1.0, pitch_shift: int = 0):
    model_path = VOICE_MAP[voice_name]
    
    # Start Piper subprocess with required model parameter
    # Use configurable length-scale for speech speed control
    piper_proc = subprocess.Popen(
        ['piper', '--model', model_path, '--length-scale', str(length_scale), '--output_raw'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    # Start Sox to play audio with optional pitch shifting
    sox_cmd = ['sox', '-t', 'raw', '-r', '22050', '-b', '16', '-e', 'signed-integer', '-c', '1', '-', '-d']
    if pitch_shift != 0:
        sox_cmd.extend(['pitch', str(pitch_shift)])
    
    sox_proc = subprocess.Popen(
        sox_cmd,
        stdin=piper_proc.stdout
    )

    # Send the text to Piper
    piper_proc.stdin.write(text.encode('utf-8'))
    piper_proc.stdin.close()

    # Wait for it to finish
    sox_proc.wait()
    piper_proc.wait()

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
