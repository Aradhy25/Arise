"""Audio deepfake / voice-clone forensic cues.

Analyzes waveform consistency, spectral flatness, and high-frequency energy.
Not a substitute for a trained audio anti-spoofing model (ASVspoof), but provides
a real signal path for voice-clone / TTS artifacts until research weights are added.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}


def _load_mono_pcm(path: Path) -> tuple[np.ndarray, int]:
    """Load audio as float32 mono. WAV via stdlib; other formats via soundfile/librosa if present."""
    ext = path.suffix.lower()
    if ext == ".wav":
        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n = wf.getnframes()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            raw = wf.readframes(n)
        if sw == 1:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0
        elif sw == 2:
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        else:
            data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        if ch > 1:
            data = data.reshape(-1, ch).mean(axis=1)
        data = data / (np.max(np.abs(data)) + 1e-8)
        return data, sr

    try:
        import soundfile as sf  # type: ignore

        data, sr = sf.read(str(path), always_2d=False)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data / (np.max(np.abs(data)) + 1e-8)
        return data, int(sr)
    except Exception as exc:
        raise ValueError(
            f"Could not decode {ext}. Convert to WAV, or install soundfile (`pip install soundfile`)."
        ) from exc


def analyze_audio(path: str | Path) -> dict:
    path = Path(path)
    audio, sr = _load_mono_pcm(path)
    if audio.size < sr * 0.3:
        raise ValueError("Audio too short (need at least ~0.3s)")

    # Frame-wise spectral features
    frame = 2048
    hop = 512
    flats = []
    hf_ratios = []
    zcrs = []
    for i in range(0, max(1, len(audio) - frame), hop):
        chunk = audio[i : i + frame]
        if chunk.size < frame:
            break
        # Zero-crossing rate
        zcr = float(np.mean(np.abs(np.diff(np.sign(chunk)))) / 2)
        zcrs.append(zcr)
        spectrum = np.abs(np.fft.rfft(chunk * np.hanning(frame))) + 1e-12
        # Spectral flatness (Wiener entropy) — synthetic/vocoded often flatter
        geo = np.exp(np.mean(np.log(spectrum)))
        arith = np.mean(spectrum)
        flats.append(float(geo / arith))
        freqs = np.fft.rfftfreq(frame, d=1.0 / sr)
        hf = spectrum[freqs > 4000].mean() if np.any(freqs > 4000) else 0.0
        lf = spectrum[freqs < 1000].mean()
        hf_ratios.append(float(hf / (lf + 1e-8)))

    flat = float(np.mean(flats)) if flats else 0.0
    hf_ratio = float(np.mean(hf_ratios)) if hf_ratios else 0.0
    zcr_mean = float(np.mean(zcrs)) if zcrs else 0.0
    zcr_std = float(np.std(zcrs)) if zcrs else 0.0

    # Heuristic scoring for TTS / voice-clone artifacts
    flat_score = float(np.clip((flat - 0.15) / 0.35, 0.0, 1.0))
    hf_score = float(np.clip((hf_ratio - 0.05) / 0.4, 0.0, 1.0))
    # Unnaturally stable ZCR can indicate synthetic speech
    zcr_score = float(np.clip((0.04 - zcr_std) / 0.04, 0.0, 1.0)) if zcr_mean > 0.01 else 0.2

    fake_prob = float(np.clip(0.4 * flat_score + 0.35 * hf_score + 0.25 * zcr_score, 0.05, 0.95))
    prediction = "FAKE" if fake_prob >= 0.5 else "REAL"
    confidence = fake_prob if prediction == "FAKE" else 1.0 - fake_prob

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "fake_probability": round(fake_prob, 4),
        "mode": "audio_forensics",
        "media_type": "audio",
        "signals": {
            "spectral_flatness": round(flat, 4),
            "high_freq_ratio": round(hf_ratio, 4),
            "zcr_mean": round(zcr_mean, 4),
            "zcr_std": round(zcr_std, 4),
            "flat_score": round(flat_score, 4),
            "hf_score": round(hf_score, 4),
            "zcr_score": round(zcr_score, 4),
            "sample_rate": sr,
            "duration_sec": round(len(audio) / sr, 2),
        },
    }
