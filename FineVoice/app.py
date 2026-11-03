"""Streamlit UI: record / upload -> isolate -> enhance -> post-process.

Run:  streamlit run app.py --server.fileWatcherType none
"""
import time
from pathlib import Path

import numpy as np
import streamlit as st

from engine.logger import get_logger
from engine.voice_enhancer import OUTPUT_SAMPLE_RATE, VoiceEnhancer
from engine.voice_isolater import SAMPLE_RATE as ISOLATOR_SAMPLE_RATE
from engine.voice_isolater import VoiceIsolator
from engine.voice_postprocessing import VoicePostProcessor
from engine.voice_utils import AUDIO_EXTENSIONS, load_audio_bytes, to_wav_bytes

logger = get_logger("app")

st.set_page_config(page_title="Voice Isolate + Enhance", page_icon="🎙️", layout="centered")


@st.cache_resource(show_spinner=False)
def load_isolator():
    return VoiceIsolator()


@st.cache_resource(show_spinner=False)
def load_enhancer(device: str):
    return VoiceEnhancer(device=device)


@st.cache_resource(show_spinner=False)
def load_postprocessor():
    return VoicePostProcessor()


def timed(timings: dict, label: str, function, *args):
    start = time.perf_counter()
    result = function(*args)
    timings[label] = time.perf_counter() - start
    logger.info("Step '%s' finished in %.2f s", label, timings[label])
    return result


def run_pipeline(audio_bytes: bytes, suffix: str, device: str, use_enhancer: bool, use_postprocessing: bool):
    """Returns (stages, timings); each stage is (title, audio, sample_rate)."""
    timings = {}
    with st.status("Processing…", expanded=True) as status:
        st.write("Decoding audio…")
        audio = timed(timings, "decode", load_audio_bytes, audio_bytes, suffix)
        stages = [("Input (16 kHz mono)", audio, ISOLATOR_SAMPLE_RATE)]

        st.write("Isolating voice (first run downloads the model)…")
        isolator = timed(timings, "load isolator", load_isolator)
        audio = timed(timings, "isolate", isolator.run, audio)
        stages.append(("Isolated", audio, ISOLATOR_SAMPLE_RATE))
        sample_rate = ISOLATOR_SAMPLE_RATE

        if use_enhancer:
            st.write("Enhancing to 48 kHz (first run downloads the model)…")
            enhancer = timed(timings, "load enhancer", load_enhancer, device)
            audio = timed(timings, "enhance", enhancer.run, audio)
            sample_rate = OUTPUT_SAMPLE_RATE
            stages.append((f"Enhanced (48 kHz, {enhancer.device})", audio, sample_rate))

        if use_postprocessing:
            st.write("Post-processing…")
            postprocessor = timed(timings, "load post-processor", load_postprocessor)
            audio = timed(timings, "post-process", postprocessor.run, audio, sample_rate)
            stages.append(("Post-processed (final)", audio, sample_rate))

        status.update(label="Done", state="complete", expanded=False)
    return stages, timings


def spectrogram_figure(stages):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(stages), 1, figsize=(8, 2.4 * len(stages)), sharex=True)
    for ax, (title, audio, sample_rate) in zip(np.atleast_1d(axes), stages):
        ax.specgram(audio, NFFT=1024, Fs=sample_rate, noverlap=768, cmap="magma", vmin=-130, vmax=-30)
        ax.set_ylim(0, 24_000)
        ax.set_ylabel("Hz")
        ax.set_title(title, fontsize=10, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("seconds")
    fig.tight_layout()
    return fig


st.title("🎙️ Voice Isolate + Enhance")
st.caption("Record or upload → isolate the main voice → enhance to 48 kHz → post-process.")

with st.sidebar:
    st.header("Settings")
    device_label = st.radio("Enhancer device", ["Auto (Apple GPU / MPS)", "CPU"])
    use_enhancer = st.checkbox("Enhancer", value=True)
    use_postprocessing = st.checkbox("Post-processing", value=True, help="Filters, loudness normalization, peak limit")
    show_spectrograms = st.checkbox("Show spectrograms", value=True)

source = st.radio("Input", ["🎤 Record", "📁 Upload"], horizontal=True, label_visibility="collapsed")

audio_bytes, audio_name = None, ""
if source.startswith("🎤"):
    recording = st.audio_input("Click the mic to start / stop recording")
    if recording is not None:
        audio_bytes, audio_name = recording.getvalue(), "recording.wav"
else:
    upload = st.file_uploader("Drop an audio file", type=AUDIO_EXTENSIONS)
    if upload is not None:
        audio_bytes, audio_name = upload.getvalue(), upload.name
        st.audio(audio_bytes)

if st.button("✨ Process", type="primary", disabled=audio_bytes is None):
    logger.info("Processing '%s' (%.1f KB)", audio_name, len(audio_bytes) / 1024)
    try:
        device = "cpu" if device_label == "CPU" else "auto"
        stages, timings = run_pipeline(audio_bytes, Path(audio_name).suffix, device, use_enhancer, use_postprocessing)
        st.session_state["result"] = dict(name=audio_name, stages=stages, timings=timings)
        logger.info("Finished '%s' in %.2f s total", audio_name, sum(timings.values()))
    except Exception as error:
        logger.exception("Processing failed")
        st.error(f"{type(error).__name__}: {error}")

result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader("Results")
    for number, (title, audio, sample_rate) in enumerate(result["stages"], start=1):
        st.markdown(f"**{number} · {title}**")
        st.audio(to_wav_bytes(audio, sample_rate), format="audio/wav")

    final_title, final_audio, final_rate = result["stages"][-1]
    st.download_button("⬇️ Download final WAV", to_wav_bytes(final_audio, final_rate),
                       file_name=f"{Path(result['name']).stem}_final.wav", mime="audio/wav")

    if show_spectrograms:
        st.pyplot(spectrogram_figure(result["stages"]))

    with st.expander("Timings"):
        for label, seconds in result["timings"].items():
            st.write(f"{label}: {seconds:.2f} s")
