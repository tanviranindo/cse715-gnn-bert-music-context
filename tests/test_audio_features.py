import numpy as np
import pytest
from src.audio_features import resample_audio, extract_log_mel, extract_chroma, segment_audio


def make_sine(duration_s, sr, freq=440.0):
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_resample_audio_changes_length_by_ratio():
    y = make_sine(2.0, 44100)
    y_rs = resample_audio(y, orig_sr=44100, target_sr=22050)
    assert abs(len(y_rs) - 44100) < 10  # ~half the samples, tolerance for resampler edge effects


def test_extract_log_mel_shape():
    y = make_sine(2.0, 22050)
    mel = extract_log_mel(y, sr=22050, n_mels=128)
    assert mel.shape[0] == 128
    assert mel.shape[1] > 0


def test_extract_chroma_shape():
    y = make_sine(2.0, 22050)
    chroma = extract_chroma(y, sr=22050, n_chroma=12)
    assert chroma.shape[0] == 12
    assert chroma.shape[1] > 0


def test_segment_audio_splits_into_fixed_windows():
    sr = 22050
    y = make_sine(25.0, sr)  # 25s -> two full 10s segments, remainder dropped
    segments = segment_audio(y, sr=sr, segment_seconds=10)
    assert len(segments) == 2
    assert all(len(s) == sr * 10 for s in segments)


def test_segment_audio_drops_short_final_segment():
    sr = 22050
    y = make_sine(9.0, sr)  # shorter than one segment
    segments = segment_audio(y, sr=sr, segment_seconds=10)
    assert segments == []
