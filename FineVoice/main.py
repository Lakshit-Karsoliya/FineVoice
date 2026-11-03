"""REST API for the voice enhance service.

Run from the project root:  uvicorn engine.main:app --host 0.0.0.0 --port 8000
Docs:                       http://localhost:8000/docs
"""
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from .logger import get_logger
from .voice_enhancer import OUTPUT_SAMPLE_RATE, VoiceEnhancer
from .voice_isolater import SAMPLE_RATE as ISOLATOR_SAMPLE_RATE
from .voice_isolater import VoiceIsolator
from .voice_postprocessing import VoicePostProcessor
from .voice_utils import AUDIO_EXTENSIONS, load_audio_bytes, to_wav_bytes

logger = get_logger("api")

models = {}
model_lock = threading.Lock()  # the models are not thread-safe, so requests run one at a time


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting API, loading models")
    models["isolator"] = VoiceIsolator()
    models["enhancer"] = VoiceEnhancer()
    models["postprocessor"] = VoicePostProcessor()
    logger.info("API ready")
    yield
    models.clear()
    logger.info("API stopped")


app = FastAPI(title="Voice Enhance API", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "enhancer_device": models["enhancer"].device,
        "supported_formats": AUDIO_EXTENSIONS,
    }


@app.post("/enhance")
def enhance_voice(file: UploadFile = File(...), isolate: bool = True, enhance: bool = True, postprocess: bool = True):
    """Upload an audio file, get back a WAV. Each stage can be switched off with its query flag."""
    start = time.perf_counter()
    filename = file.filename or "audio"
    suffix = Path(filename).suffix.lower()
    logger.info("Request: file=%s isolate=%s enhance=%s postprocess=%s", filename, isolate, enhance, postprocess)
    if suffix.lstrip(".") not in AUDIO_EXTENSIONS:
        logger.warning("Rejected unsupported format: %s", suffix or "none")
        raise HTTPException(415, f"Unsupported format '{suffix}'. Supported: {', '.join(AUDIO_EXTENSIONS)}")

    try:
        audio = load_audio_bytes(file.file.read(), suffix)
    except ValueError as error:
        logger.error("Could not decode audio: %s", error)
        raise HTTPException(400, str(error))

    sample_rate = ISOLATOR_SAMPLE_RATE
    with model_lock:
        if isolate:
            audio = models["isolator"].run(audio)
        if enhance:
            audio = models["enhancer"].run(audio)
            sample_rate = OUTPUT_SAMPLE_RATE
        if postprocess:
            audio = models["postprocessor"].run(audio, sample_rate)

    wav = to_wav_bytes(audio, sample_rate)
    logger.info("Request done in %.2f s | returning %.1f KB WAV @ %d Hz",
                time.perf_counter() - start, len(wav) / 1024, sample_rate)
    return Response(
        content=wav,
        media_type="audio/wav",
        headers={
            "Content-Disposition": f'attachment; filename="{Path(filename).stem}_enhanced.wav"',
            "X-Sample-Rate": str(sample_rate),
        },
    )
