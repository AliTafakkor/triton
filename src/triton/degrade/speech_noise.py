"""Speech-shaped and speech-correlated noise generation.

References:
    Byrne, D., et al. (1994). "An international comparison of long-term average 
    speech spectra." The Journal of the Acoustical Society of America, 96(4), 
    2108-2120.
    
    Rhebergen, K. S., & Versfeld, N. J. (2005). "A speech intelligibility 
    index-based approach to predict the speech reception threshold for sentences 
    in fluctuating noise for normal-hearing listeners." The Journal of the 
    Acoustical Society of America, 117(4), 2181-2192.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sp_signal

from triton.core.io import normalize_peak


def speech_shaped_noise(
    duration: float,
    sr: int,
    *,
    spectrum: str = "ltass",
    normalize: bool = True,
) -> np.ndarray:
    """Generate speech-shaped noise matching long-term average speech spectrum.

    Creates noise filtered to match the spectral characteristics of speech,
    commonly used in speech-in-noise experiments.

    Args:
        duration: Duration in seconds.
        sr: Sample rate.
        spectrum: Spectrum type - "ltass" (default, Byrne et al. 1994) or "flat".
        normalize: Whether to normalize to peak amplitude 0.99.

    Returns:
        Speech-shaped noise waveform.

    References:
        Byrne, D., et al. (1994). "An international comparison of long-term 
        average speech spectra." JASA, 96(4), 2108-2120.
    """
    n_samples = int(duration * sr)
    noise = np.random.randn(n_samples).astype(np.float32)

    if spectrum == "flat":
        # Return unfiltered white noise
        return normalize_peak(noise) if normalize else noise

    if spectrum != "ltass":
        raise ValueError(f"Unknown spectrum type: {spectrum}")

    # Apply LTASS filter (Byrne et al., 1994)
    # The LTASS has characteristic slope: roughly +6 dB/octave up to ~500 Hz,
    # then -9 dB/octave above ~1000 Hz
    noise = _apply_ltass_filter(noise, sr)

    return normalize_peak(noise) if normalize else noise


def speech_correlated_noise(
    speech: np.ndarray,
    sr: int,
    *,
    method: str = "spectrum_match",
    frame_length: int = 2048,
    hop_length: int | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Generate noise correlated with speech signal's spectral characteristics.

    Creates noise that matches the time-varying spectral envelope of the speech
    signal, providing more realistic masking than stationary noise.

    Args:
        speech: Input speech waveform (mono).
        sr: Sample rate.
        method: "spectrum_match" (spectral envelope matching) or "modulation" 
            (amplitude modulation matching).
        frame_length: FFT frame length for spectral analysis.
        hop_length: Hop length for STFT (default: frame_length // 4).
        normalize: Whether to normalize to peak amplitude 0.99.

    Returns:
        Speech-correlated noise waveform.

    References:
        Rhebergen, K. S., & Versfeld, N. J. (2005). "A speech intelligibility 
        index-based approach to predict the speech reception threshold for 
        sentences in fluctuating noise." JASA, 117(4), 2181-2192.
    """
    speech = np.asarray(speech, dtype=np.float32)
    
    if method == "spectrum_match":
        noise = _spectrum_matched_noise(speech, sr, frame_length, hop_length)
    elif method == "modulation":
        noise = _modulation_matched_noise(speech, sr, frame_length, hop_length)
    else:
        raise ValueError(f"Unknown method: {method}")

    return normalize_peak(noise) if normalize else noise


def _apply_ltass_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """Apply long-term average speech spectrum filter (Byrne et al., 1994).

    The LTASS shows:
    - Rising slope up to ~500 Hz
    - Peak around 500-1000 Hz  
    - Falling slope above ~1000 Hz at approximately -9 dB/octave
    """
    # Frequency points from Byrne et al. (1994) - representative values
    # Frequencies in Hz
    freqs = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    
    # Relative levels in dB (normalized to peak at 1000 Hz)
    # Approximation based on Byrne et al. LTASS for conversational speech
    levels_db = np.array([-4, 0, 4, 0, -6, -18, -30])
    
    # Convert to linear scale
    gains = 10 ** (levels_db / 20)
    
    # Design filter using frequency sampling method
    nyquist = sr / 2
    
    # Interpolate gains across frequency range
    n_freqs = 512
    freq_range = np.linspace(0, nyquist, n_freqs)
    
    # Interpolate and extrapolate
    interp_gains = np.interp(
        freq_range,
        freqs,
        gains,
        left=gains[0],
        right=gains[-1] * 0.01  # Strong attenuation above 8 kHz
    )
    
    # Create minimum-phase filter
    # Mirror for symmetric frequency response
    spectrum = np.concatenate([interp_gains, interp_gains[-2:0:-1]])
    
    # Get minimum phase impulse response
    ir = np.fft.ifft(spectrum).real
    ir = np.fft.fftshift(ir)
    
    # Window to reasonable length
    n_taps = min(1024, len(ir))
    window = sp_signal.windows.hann(n_taps)
    ir = ir[:n_taps] * window
    
    # Apply filter
    return sp_signal.fftconvolve(audio, ir, mode='same').astype(np.float32)


def _spectrum_matched_noise(
    speech: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int | None,
) -> np.ndarray:
    """Generate noise matching time-varying spectral envelope of speech."""
    if hop_length is None:
        hop_length = frame_length // 4
    
    # Generate white noise with same length as speech
    noise = np.random.randn(len(speech)).astype(np.float32)
    
    # Compute STFT of both signals
    f_speech, t, speech_stft = sp_signal.stft(
        speech, 
        sr, 
        nperseg=frame_length,
        noverlap=frame_length - hop_length,
    )
    
    f_noise, _, noise_stft = sp_signal.stft(
        noise,
        sr,
        nperseg=frame_length,
        noverlap=frame_length - hop_length,
    )
    
    # Extract magnitude and phase
    speech_mag = np.abs(speech_stft)
    noise_phase = np.angle(noise_stft)
    
    # Create noise with speech magnitude and noise phase
    shaped_stft = speech_mag * np.exp(1j * noise_phase)
    
    # Inverse STFT
    _, shaped_noise = sp_signal.istft(
        shaped_stft,
        sr,
        nperseg=frame_length,
        noverlap=frame_length - hop_length,
    )
    
    # Match length
    if len(shaped_noise) > len(speech):
        shaped_noise = shaped_noise[:len(speech)]
    elif len(shaped_noise) < len(speech):
        shaped_noise = np.pad(shaped_noise, (0, len(speech) - len(shaped_noise)))
    
    return shaped_noise.astype(np.float32)


def _modulation_matched_noise(
    speech: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int | None,
) -> np.ndarray:
    """Generate noise matching amplitude modulation characteristics of speech."""
    if hop_length is None:
        hop_length = frame_length // 4
    
    # Start with speech-shaped noise (LTASS)
    noise = speech_shaped_noise(len(speech) / sr, sr, spectrum="ltass", normalize=False)
    
    # Extract amplitude envelopes using Hilbert transform
    speech_analytic = sp_signal.hilbert(speech)
    speech_envelope = np.abs(speech_analytic)
    
    noise_analytic = sp_signal.hilbert(noise)
    noise_carrier = noise / (np.abs(noise_analytic) + 1e-10)
    
    # Modulate noise carrier with speech envelope
    modulated = noise_carrier * speech_envelope
    
    return modulated.astype(np.float32)
