from mcp.server.fastmcp import FastMCP
import subprocess, os
import time

# Initialize FastMCP server
mcp = FastMCP("voice-mcp")

@mcp.tool()
async def use_voice(text: str):
    """
    This is a basic voice tool that uses piper to say something to the user. Use only if it is asked to speak.
    ONLY USE IF ASKED. The tool receives what it is to be said. Be laconic in your speech.

    Args:
        text: Content of what the user will listen. Speech should be concise
    """
    st = time.time()
    speak(text)

    return f"Speaking took {time.time()-st:.2f} sec(s)"

def speak(text: str):
    # Start Piper subprocess
    piper_proc = subprocess.Popen(
        ['bash', 'piper', '--output_raw'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        cwd= os.path.join(os.getcwd(), 'scripts')
    )

    # Start Sox to play audio
    sox_proc = subprocess.Popen(
        ['sox', '-t', 'raw', '-r', '22050', '-b', '16', '-e', 'signed-integer', '-c', '1', '-', '-d'],
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
