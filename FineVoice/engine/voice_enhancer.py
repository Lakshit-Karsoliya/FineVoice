"""Voice enhancement: MossFormer2 speech super-resolution, 16 kHz -> 48 kHz."""
import os

# Lets PyTorch fall back to CPU for ops the Apple GPU (MPS) backend lacks; must be set before torch loads.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import tempfile
import time
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from .logger import get_logger
from .voice_utils import MODELS_DIR, audio_stats, save_wav, working_dir

logger = get_logger("enhancer")

MODEL_NAME = "MossFormer2_SR_48K"
REPO_ID = f"alibabasglab/{MODEL_NAME}"
INPUT_SAMPLE_RATE = 16_000
OUTPUT_SAMPLE_RATE = 48_000


class VoiceEnhancer:
    def __init__(self, models_dir: Path = MODELS_DIR, device: str = "auto"):
        """device: 'auto' uses the Apple GPU (MPS) when available, 'cpu' forces CPU."""
        self.models_dir = Path(models_dir)
        self.checkpoint_dir = self.models_dir / "checkpoints" / MODEL_NAME
        self._ensure_model()
        self._clearvoice, self.device = self._load(device)

    def run(self, audio: np.ndarray) -> np.ndarray:
        """Restore full-band speech. Input: 16 kHz mono. Output: 48 kHz mono."""
        logger.info("Enhancing voice on %s | input: %s", self.device, audio_stats(audio, INPUT_SAMPLE_RATE))
        start = time.perf_counter()
        try:
            output = self._enhance(audio)
        except Exception:
            if not self.device.startswith("mps"):
                raise
            logger.exception("Enhancement failed on MPS, retrying on CPU")
            self._clearvoice, self.device = self._load("cpu")
            output = self._enhance(audio)
        elapsed = time.perf_counter() - start
        logger.info("Enhancement done in %.2f s (%.1fx real-time) | output: %s",
                    elapsed, (audio.size / INPUT_SAMPLE_RATE) / elapsed, audio_stats(output, OUTPUT_SAMPLE_RATE))
        return output

    def _ensure_model(self) -> None:
        if (self.checkpoint_dir / "last_best_checkpoint").is_file():
            logger.info("Model found in %s, skipping download", self.checkpoint_dir)
            return
        logger.info("Model not found, downloading %s to %s", REPO_ID, self.checkpoint_dir)
        snapshot_download(REPO_ID, local_dir=self.checkpoint_dir)
        logger.info("Download complete")

    def _load(self, device: str):
        import torch
        from clearvoice import ClearVoice

        logger.info("Loading %s (requested device: %s)", MODEL_NAME, device)
        original_check = torch.backends.mps.is_available
        if device == "cpu":
            torch.backends.mps.is_available = lambda: False  # ClearVoice picks MPS through this check
        try:
            with working_dir(self.models_dir):  # ClearVoice resolves ./checkpoints/<model> from the cwd
                clearvoice = ClearVoice(task="speech_super_resolution", model_names=[MODEL_NAME])
        finally:
            torch.backends.mps.is_available = original_check
        used_device = str(clearvoice.models[0].device)
        logger.info("Enhancer ready on device: %s", used_device)
        return clearvoice, used_device

    def _enhance(self, audio: np.ndarray) -> np.ndarray:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "isolated.wav"
            save_wav(path, audio, INPUT_SAMPLE_RATE)  # ClearVoice reads from a file and resamples to 48 kHz
            output = self._clearvoice(input_path=str(path), online_write=False)
        if isinstance(output, dict):
            output = next(iter(output.values()))
        return np.asarray(output, dtype=np.float32).squeeze()
