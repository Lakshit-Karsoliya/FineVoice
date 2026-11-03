"""Helpers for loading, saving and describing voice audio."""
import io
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

from .logger import get_logger
from .voice_recorder import VoiceRecorder

logger = get_logger("utils")

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_SAMPLE_RATE = 16_000
AUDIO_EXTENSIONS = [
    "wav", "mp3", "m4a", "aac", "flac", "ogg", "opus", "webm",
    "wma", "aiff", "aif", "amr", "mp4", "3gp",
]


def to_db(value: float) -> float:
    return 20 * np.log10(max(value, 1e-9))


def audio_stats(audio: np.ndarray, sample_rate: int) -> str:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    return (f"{audio.size / sample_rate:.2f} s | {audio.size} samples @ {sample_rate} Hz | "
            f"peak {to_db(peak):.1f} dBFS | rms {to_db(rms):.1f} dBFS")


def load_audio_file(path, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Decode any common audio format to mono float32 at `sample_rate`."""
    path = Path(path)
    logger.info("Loading file: %s (%.1f KB)", path.name, path.stat().st_size / 1024)
    if shutil.which("ffmpeg"):
        audio = _decode_with_ffmpeg(path, sample_rate)
    else:
        logger.warning("ffmpeg not found, using librosa (fewer formats supported)")
        audio = _decode_with_librosa(path, sample_rate)
    if audio.size == 0:
        raise ValueError(f"'{path.name}' contains no audio")
    logger.info("Loaded: %s", audio_stats(audio, sample_rate))
    return audio


def load_audio_bytes(data: bytes, suffix: str = ".wav", sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Same as load_audio_file, for uploaded or recorded bytes."""
    logger.info("Received %.1f KB of audio data (%s)", len(data) / 1024, suffix or "unknown format")
    with tempfile.NamedTemporaryFile(suffix=suffix or ".wav", delete=False) as f:
        f.write(data)
        temp_path = f.name
    try:
        return load_audio_file(temp_path, sample_rate)
    finally:
        os.unlink(temp_path)


def load_from_recorder(seconds: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> np.ndarray:
    """Record from the local microphone for `seconds`."""
    audio = VoiceRecorder(sample_rate).record(seconds)
    logger.info("Recorded: %s", audio_stats(audio, sample_rate))
    return audio


def save_wav(path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(path, np.clip(audio, -1.0, 1.0), sample_rate, subtype="PCM_16")


def to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(audio, -1.0, 1.0), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


@contextmanager
def working_dir(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _decode_with_ffmpeg(path: Path, sample_rate: int) -> np.ndarray:
    command = ["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
               "-ar", str(sample_rate), "-f", "f32le", "-"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg could not decode '{path.name}': {result.stderr.decode()[-300:]}")
    return np.frombuffer(result.stdout, dtype=np.float32).copy()


def _decode_with_librosa(path: Path, sample_rate: int) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    return audio.astype(np.float32)
