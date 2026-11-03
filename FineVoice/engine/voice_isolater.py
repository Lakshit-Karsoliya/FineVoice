"""Voice isolation: keeps the main speaker, removes noise and other voices."""
import sys
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from .logger import get_logger
from .voice_utils import MODELS_DIR, audio_stats

logger = get_logger("isolator")

REPO_ID = "Scicom-intl/voice-isolator-cpu"
SAMPLE_RATE = 16_000


class VoiceIsolator:
    def __init__(self, models_dir: Path = MODELS_DIR):
        self.model_dir = Path(models_dir) / "voice-isolator-cpu"
        self._ensure_model()
        self._isolate = self._load()

    def run(self, audio: np.ndarray) -> np.ndarray:
        """Isolate the main voice. Input and output: 16 kHz mono float32."""
        logger.info("Isolating voice | input: %s", audio_stats(audio, SAMPLE_RATE))
        start = time.perf_counter()
        output = self._isolate(str(self.model_dir), audio.astype(np.float32))
        output = np.asarray(output, dtype=np.float32).squeeze()
        elapsed = time.perf_counter() - start
        logger.info("Isolation done in %.2f s (%.1fx real-time) | output: %s",
                    elapsed, (audio.size / SAMPLE_RATE) / elapsed, audio_stats(output, SAMPLE_RATE))
        return output

    def _ensure_model(self) -> None:
        if any(self.model_dir.rglob("*.onnx")):
            logger.info("Model found in %s, skipping download", self.model_dir)
            return
        logger.info("Model not found, downloading %s to %s", REPO_ID, self.model_dir)
        snapshot_download(REPO_ID, local_dir=self.model_dir, ignore_patterns=["figures/*", "examples/*"])
        logger.info("Download complete")

    def _load(self):
        if str(self.model_dir) not in sys.path:
            sys.path.insert(0, str(self.model_dir))  # the model repo ships the voice_isolator package
        from voice_isolator import isolate

        logger.info("Isolator ready (ONNX Runtime, CPU)")
        return isolate
