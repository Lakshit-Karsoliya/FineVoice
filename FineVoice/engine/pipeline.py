"""The isolate -> enhance -> post-process chain, shared by the Streamlit UI and the folder batch."""
import time
from typing import Callable, Optional

import numpy as np

from .logger import get_logger
from .voice_enhancer import OUTPUT_SAMPLE_RATE
from .voice_isolater import SAMPLE_RATE as ISOLATOR_SAMPLE_RATE

logger = get_logger("pipeline")


def run_stages(audio: np.ndarray, isolator, enhancer=None, postprocessor=None,
               on_step: Optional[Callable[[str], None]] = None):
    """Run the models on decoded 16 kHz mono audio.

    Pass enhancer=None / postprocessor=None to skip those stages.
    Returns (stages, timings); each stage is (title, audio, sample_rate) and stages[-1] is the final result.
    """
    timings: dict = {}
    say = on_step or (lambda message: None)

    def timed(label, function, *args):
        start = time.perf_counter()
        result = function(*args)
        timings[label] = time.perf_counter() - start
        logger.info("Step '%s' finished in %.2f s", label, timings[label])
        return result

    stages = [("Input (16 kHz mono)", audio, ISOLATOR_SAMPLE_RATE)]

    say("Isolating voice…")
    audio = timed("isolate", isolator.run, audio)
    stages.append(("Isolated", audio, ISOLATOR_SAMPLE_RATE))
    sample_rate = ISOLATOR_SAMPLE_RATE

    if enhancer is not None:
        say("Enhancing to 48 kHz…")
        audio = timed("enhance", enhancer.run, audio)
        sample_rate = OUTPUT_SAMPLE_RATE
        stages.append((f"Enhanced (48 kHz, {enhancer.device})", audio, sample_rate))

    if postprocessor is not None:
        say("Post-processing…")
        audio = timed("post-process", postprocessor.run, audio, sample_rate)
        stages.append(("Post-processed (final)", audio, sample_rate))

    return stages, timings