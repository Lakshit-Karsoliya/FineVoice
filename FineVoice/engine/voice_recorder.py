"""Records voice from the local microphone."""
import time

import numpy as np

from .logger import get_logger

logger = get_logger("recorder")


class VoiceRecorder:
    def __init__(self, sample_rate: int = 16_000):
        self.sample_rate = sample_rate
        self._chunks = []
        self._stream = None

    def start(self) -> None:
        import sounddevice as sd  # imported here so the package loads without PortAudio

        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32", callback=self._on_audio
        )
        self._stream.start()
        logger.info("Recording started | %d Hz mono | device: %s", self.sample_rate, sd.query_devices(kind="input")["name"])

    def stop(self) -> np.ndarray:
        if self._stream is None:
            raise RuntimeError("Recorder was not started")
        self._stream.stop()
        self._stream.close()
        self._stream = None
        audio = np.concatenate(self._chunks).flatten() if self._chunks else np.zeros(0, dtype=np.float32)
        logger.info("Recording stopped | %.2f s captured", audio.size / self.sample_rate)
        return audio

    def record(self, seconds: float) -> np.ndarray:
        self.start()
        time.sleep(seconds)
        return self.stop()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("Recorder status: %s", status)
        self._chunks.append(indata.copy())
