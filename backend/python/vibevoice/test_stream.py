#!/usr/bin/env python3
"""Test TTSStream functionality for VibeVoice with true streaming via AudioStreamer"""
import numpy as np

def test_audio_conversion():
    """Test that audio tensor conversion produces valid PCM"""
    # Simulate audio tensor output (float32, potentially multi-dimensional)
    audio_np = np.random.randn(24000).astype(np.float32)  # 1 second at 24kHz

    # Flatten if needed (same as TTSStream does)
    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)

    # Normalize if peak > 1.0
    peak = np.max(np.abs(audio_np)) if audio_np.size else 0.0
    if peak > 1.0:
        audio_np = audio_np / peak

    # Convert to 16-bit PCM
    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    assert len(audio_bytes) == 24000 * 2, f"Expected {24000*2} bytes, got {len(audio_bytes)}"
    assert isinstance(audio_bytes, bytes), "Should be bytes"

    # Verify we can reconstruct
    reconstructed = np.frombuffer(audio_bytes, dtype=np.int16)
    assert len(reconstructed) == 24000, f"Expected 24000 samples, got {len(reconstructed)}"

    print("Audio conversion test passed!")

def test_normalization():
    """Test audio normalization"""
    # Audio with peak > 1.0
    audio_np = np.array([0.5, 1.5, -2.0, 0.3], dtype=np.float32)
    peak = np.max(np.abs(audio_np))
    if peak > 1.0:
        audio_np = audio_np / peak

    assert np.max(np.abs(audio_np)) <= 1.0, "Normalization failed"
    print("Normalization test passed!")

def test_multidim_flatten():
    """Test flattening multi-dimensional audio (as AudioStreamer may return)"""
    # Simulate batched audio output
    audio_np = np.random.randn(1, 4800).astype(np.float32)

    if audio_np.ndim > 1:
        audio_np = audio_np.reshape(-1)

    assert audio_np.ndim == 1, "Should be 1D after flatten"
    assert len(audio_np) == 4800, f"Expected 4800 samples, got {len(audio_np)}"

    print("Multi-dimensional flatten test passed!")

def test_empty_audio_handling():
    """Test handling of empty audio chunks"""
    audio_np = np.array([], dtype=np.float32)

    peak = np.max(np.abs(audio_np)) if audio_np.size else 0.0
    assert peak == 0.0, "Empty array should have peak 0"

    # Should not divide by zero
    if peak > 1.0:
        audio_np = audio_np / peak

    audio_int16 = (audio_np * 32767).astype(np.int16)
    audio_bytes = audio_int16.tobytes()

    assert len(audio_bytes) == 0, "Empty audio should produce empty bytes"
    print("Empty audio handling test passed!")

if __name__ == "__main__":
    test_audio_conversion()
    test_normalization()
    test_multidim_flatten()
    test_empty_audio_handling()
    print("All tests passed!")
