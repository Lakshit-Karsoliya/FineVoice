"""Post-processing: remove wind/rumble and hiss, then normalize loudness."""
import time

import numpy as np
import pyloudnorm as pyln
from scipy.signal import butter, sosfiltfilt

from .logger import get_logger
from .voice_utils import audio_stats, to_db

logger = get_logger("postprocess")


class VoicePostProcessor:
    def __init__(self, highpass_hz: float = 80.0, lowpass_hz: float = 16_000.0,
                 target_lufs: float = -16.0, peak_db: float = -1.0):
        self.highpass_hz = highpass_hz  # cuts wind and low rumble
        self.lowpass_hz = lowpass_hz    # cuts high-frequency hiss
        self.target_lufs = target_lufs  # common loudness target for spoken voice
        self.peak_db = peak_db          # highest allowed peak

    def run(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        logger.info("Post-processing | input: %s", audio_stats(audio, sample_rate))
        start = time.perf_counter()
        audio = self._remove_dc(audio)
        audio = self._filter(audio, sample_rate, self.highpass_hz, "highpass", "wind/rumble")
        audio = self._filter(audio, sample_rate, self.lowpass_hz, "lowpass", "hiss")
        audio = self._normalize_loudness(audio, sample_rate)
        audio = self._limit_peak(audio)
        audio = audio.astype(np.float32)
        logger.info("Post-processing done in %.2f s | output: %s",
                    time.perf_counter() - start, audio_stats(audio, sample_rate))
        return audio

    def _remove_dc(self, audio: np.ndarray) -> np.ndarray:
        offset = float(np.mean(audio))
        logger.info("DC offset removed: %.6f", offset)
        return audio - offset

    def _filter(self, audio, sample_rate, cutoff_hz, kind, purpose) -> np.ndarray:
        if cutoff_hz >= sample_rate / 2:
            logger.info("%s skipped: %.0f Hz is above the Nyquist frequency (%.0f Hz)",
                        kind.capitalize(), cutoff_hz, sample_rate / 2)
            return audio
        sos = butter(4, cutoff_hz, btype=kind, fs=sample_rate, output="sos")
        logger.info("%s filter at %.0f Hz (%s)", kind.capitalize(), cutoff_hz, purpose)
        return sosfiltfilt(sos, audio)

    def _normalize_loudness(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            loudness = pyln.Meter(sample_rate).integrated_loudness(audio)
        except ValueError:
            logger.warning("Loudness normalization skipped: audio is too short to measure")
            return audio
        if not np.isfinite(loudness):
            logger.warning("Loudness normalization skipped: audio is silent")
            return audio
        gain_db = self.target_lufs - loudness
        logger.info("Loudness %.1f LUFS -> target %.1f LUFS (gain %+.1f dB)", loudness, self.target_lufs, gain_db)
        return audio * 10 ** (gain_db / 20)

    def _limit_peak(self, audio: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(audio)))
        ceiling = 10 ** (self.peak_db / 20)
        if peak <= ceiling:
            logger.info("Peak %.1f dBFS is within the %.1f dBFS limit", to_db(peak), self.peak_db)
            return audio
        logger.info("Peak %.1f dBFS reduced to %.1f dBFS", to_db(peak), self.peak_db)
        return audio * (ceiling / peak)
