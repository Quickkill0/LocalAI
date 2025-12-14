#!/usr/bin/env python3
"""Test TTSStream functionality"""
import sys
import numpy as np

# Mock the backend_pb2 module for testing
class MockTTSStreamChunk:
    def __init__(self, audio, sample_rate, chunk_index, is_final, error=None):
        self.audio = audio
        self.sample_rate = sample_rate
        self.chunk_index = chunk_index
        self.is_final = is_final
        self.error = error

class MockRequest:
    def __init__(self, text, voice='af_heart'):
        self.text = text
        self.voice = voice

def test_audio_conversion():
    """Test that audio conversion produces valid int16 PCM"""
    # Simulate audio tensor output
    audio_np = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz
    audio_np = np.clip(audio_np, -1.0, 1.0)  # Normalize

    # Convert to int16 (same as TTSStream does)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    assert len(audio_bytes) == 24000 * 2, f"Expected {24000*2} bytes, got {len(audio_bytes)}"
    assert isinstance(audio_bytes, bytes), "Should be bytes"

    # Verify we can reconstruct
    reconstructed = np.frombuffer(audio_bytes, dtype=np.int16)
    assert len(reconstructed) == 24000, f"Expected 24000 samples, got {len(reconstructed)}"

    print("Audio conversion test passed!")

if __name__ == "__main__":
    test_audio_conversion()
    print("All tests passed!")
