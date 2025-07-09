#!/usr/bin/env python3
"""
Test script for the voice-mcp service.
Tests the vctk voice with a specific message.
"""

import asyncio
import sys
from main import speak, VOICE_MAP, AVAILABLE_VOICES

async def test_vctk_voice():
    """Test the vctk voice with the specified text."""
    test_text = "This is a test of the piper and sox voice-mcp service"
    voice_name = "vctk"
    
    print(f"Testing voice: {voice_name}")
    print(f"Text: {test_text}")
    print("-" * 50)
    
    # First, list available voices
    print("Available voices:")
    print(f"Available voices ({len(AVAILABLE_VOICES)}): {', '.join(sorted(AVAILABLE_VOICES))}")
    print()
    
    # Check if voice exists
    if voice_name not in VOICE_MAP:
        print(f"Error: Voice '{voice_name}' not found in voice mapping")
        return
    
    # Test the voice
    print("Testing voice...")
    print(f"Using model: {VOICE_MAP[voice_name]}")
    
    import time
    st = time.time()
    speak(test_text, voice_name)
    elapsed = time.time() - st
    
    print(f"Speaking '{test_text}' with voice '{voice_name}' took {elapsed:.2f} sec(s)")

if __name__ == "__main__":
    asyncio.run(test_vctk_voice())