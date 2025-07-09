#!/usr/bin/env python3
"""
Test script for the length_scale parameter.
Tests different speech speeds with the vctk voice.
"""

import asyncio
import time
from main import speak, VOICE_MAP, AVAILABLE_VOICES

async def test_length_scales():
    """Test different length_scale values with the vctk voice."""
    test_text = "Testing different speech speeds"
    voice_name = "vctk"
    
    # Test different length scales
    scales = [1.0, 1.5, 2.0, 2.5]
    
    print(f"Testing voice: {voice_name}")
    print(f"Text: {test_text}")
    print("-" * 50)
    
    for scale in scales:
        print(f"\nTesting length_scale = {scale}")
        st = time.time()
        speak(test_text, voice_name, scale)
        elapsed = time.time() - st
        print(f"Completed in {elapsed:.2f} seconds")
        
        # Wait a moment between tests
        await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(test_length_scales())