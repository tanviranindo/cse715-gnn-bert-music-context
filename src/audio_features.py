"""Week 1: mel/chroma/segmentation."""

import numpy as np
import librosa


def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def extract_log_mel(y: np.ndarray, sr: int, n_mels: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def extract_chroma(y: np.ndarray, sr: int, n_chroma: int) -> np.ndarray:
    return librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)


def segment_audio(y: np.ndarray, sr: int, segment_seconds: float) -> list[np.ndarray]:
    window = int(sr * segment_seconds)
    n_full_segments = len(y) // window
    return [y[i * window:(i + 1) * window] for i in range(n_full_segments)]
