"""Tests for speech-shaped and speech-correlated noise generation."""

from __future__ import annotations

import numpy as np
import pytest

from triton.degrade.speech_noise import (
    speech_shaped_noise,
    speech_correlated_noise,
)


def test_speech_shaped_noise_basic():
    """Test basic speech-shaped noise generation."""
    duration = 1.0
    sr = 16000
    
    noise = speech_shaped_noise(duration, sr, spectrum="ltass")
    
    assert noise.shape == (sr,)
    assert noise.dtype == np.float32
    assert np.max(np.abs(noise)) <= 1.0


def test_speech_shaped_noise_flat():
    """Test flat spectrum (white noise) generation."""
    duration = 0.5
    sr = 16000
    
    noise = speech_shaped_noise(duration, sr, spectrum="flat")
    
    assert noise.shape == (int(duration * sr),)
    assert noise.dtype == np.float32


def test_speech_shaped_noise_invalid_spectrum():
    """Test invalid spectrum type raises error."""
    with pytest.raises(ValueError, match="Unknown spectrum type"):
        speech_shaped_noise(1.0, 16000, spectrum="invalid")


def test_speech_correlated_noise_spectrum_match():
    """Test spectrum-matched noise generation."""
    sr = 16000
    # Create simple synthetic speech-like signal
    duration = 0.5
    t = np.linspace(0, duration, int(duration * sr))
    speech = np.sin(2 * np.pi * 200 * t) + 0.5 * np.sin(2 * np.pi * 500 * t)
    speech = speech.astype(np.float32)
    
    noise = speech_correlated_noise(
        speech, sr, method="spectrum_match", frame_length=1024
    )
    
    assert noise.shape == speech.shape
    assert noise.dtype == np.float32
    assert np.max(np.abs(noise)) <= 1.0


def test_speech_correlated_noise_modulation():
    """Test modulation-matched noise generation."""
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(duration * sr))
    speech = np.sin(2 * np.pi * 200 * t) * (1 + 0.5 * np.sin(2 * np.pi * 4 * t))
    speech = speech.astype(np.float32)
    
    noise = speech_correlated_noise(
        speech, sr, method="modulation", frame_length=2048
    )
    
    assert noise.shape == speech.shape
    assert noise.dtype == np.float32


def test_speech_correlated_noise_invalid_method():
    """Test invalid method raises error."""
    speech = np.random.randn(16000).astype(np.float32)
    
    with pytest.raises(ValueError, match="Unknown method"):
        speech_correlated_noise(speech, 16000, method="invalid")


def test_speech_shaped_noise_no_normalize():
    """Test noise generation without normalization."""
    noise = speech_shaped_noise(1.0, 16000, spectrum="ltass", normalize=False)
    
    # Without normalization, amplitude may exceed 1.0
    assert noise.shape == (16000,)
    assert noise.dtype == np.float32


def test_speech_correlated_noise_custom_hop():
    """Test with custom hop length."""
    sr = 16000
    speech = np.random.randn(sr).astype(np.float32)
    
    noise = speech_correlated_noise(
        speech, sr, method="spectrum_match", frame_length=1024, hop_length=256
    )
    
    assert noise.shape == speech.shape
    assert noise.dtype == np.float32
