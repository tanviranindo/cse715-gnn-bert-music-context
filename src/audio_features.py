"""Week 1: mel/chroma/segmentation."""

import numpy as np
import librosa

# Some CPU/image combinations crash inside librosa's tuning estimation --
# chroma_stft -> estimate_tuning -> piptrack is a numba guvectorize kernel, and
# it takes the worker process down natively rather than raising. requirements.txt
# pins numba/llvmlite against one such combination; it is not sufficient on all
# of them. Setting LIBROSA_SKIP_TUNING=1 passes tuning=0.0, which skips the
# failing kernel at the cost of assuming concert pitch. Off by default, so the
# committed caches stay reproducible by the path that built them.
import os as _os
_SKIP_TUNING = _os.environ.get("LIBROSA_SKIP_TUNING") == "1"
_CHROMA_KW = {"tuning": 0.0} if _SKIP_TUNING else {}



def resample_audio(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return y
    return librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)


def extract_log_mel(y: np.ndarray, sr: int, n_mels: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=np.max)


def extract_chroma(y: np.ndarray, sr: int, n_chroma: int) -> np.ndarray:
    return librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma, **_CHROMA_KW)


def segment_audio(y: np.ndarray, sr: int, segment_seconds: float) -> list[np.ndarray]:
    window = int(sr * segment_seconds)
    n_full_segments = len(y) // window
    return [y[i * window:(i + 1) * window] for i in range(n_full_segments)]
