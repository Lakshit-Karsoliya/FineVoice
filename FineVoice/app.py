"""Streamlit UI: record / upload / folder -> isolate -> enhance -> post-process.

Run:  streamlit run app.py --server.fileWatcherType none
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from engine.batch_processor import SUFFIX, SUPPORTED_EXTENSIONS, run_batch
from engine.logger import clear_logs, get_log_text, get_logger
from engine.pipeline import run_stages
from engine.voice_enhancer import VoiceEnhancer
from engine.voice_isolater import VoiceIsolator
from engine.voice_postprocessing import VoicePostProcessor
from engine.voice_utils import load_audio_bytes, to_wav_bytes

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


def load_models(device: str, use_enhancer: bool, use_postprocessing: bool):
    """Returns (isolator, enhancer or None, postprocessor or None) according to the sidebar settings."""
    return (
        load_isolator(),
        load_enhancer(device) if use_enhancer else None,
        load_postprocessor() if use_postprocessing else None,
    )


def timed(timings: dict, label: str, function, *args):
    start = time.perf_counter()
    result = function(*args)
    timings[label] = time.perf_counter() - start
    logger.info("Step '%s' finished in %.2f s", label, timings[label])
    return result


def process_bytes(audio_bytes: bytes, name: str, device: str, use_enhancer: bool, use_postprocessing: bool):
    """Decode + run the full pipeline for one recorded/uploaded file. Returns a result dict."""
    timings = {}
    logger.info("Processing '%s' (%.1f KB)", name, len(audio_bytes) / 1024)
    with st.status(f"Processing {name}…", expanded=True) as status:
        st.write("Decoding audio…")
        audio = timed(timings, "decode", load_audio_bytes, audio_bytes, Path(name).suffix)
        st.write("Loading models (first run downloads them)…")
        models = timed(timings, "load models", load_models, device, use_enhancer, use_postprocessing)
        stages, run_timings = run_stages(audio, *models, on_step=st.write)
        timings.update(run_timings)
        status.update(label=f"{name} done", state="complete", expanded=False)
    logger.info("Finished '%s' in %.2f s total", name, sum(timings.values()))
    return dict(name=name, stages=stages, timings=timings)


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


def render_result(result: dict, key: str, show_spectrograms: bool, expanded: bool = True):
    with st.expander(f"✅ {result['name']}", expanded=expanded):
        for number, (title, audio, sample_rate) in enumerate(result["stages"], start=1):
            st.markdown(f"**{number} · {title}**")
            st.audio(to_wav_bytes(audio, sample_rate), format="audio/wav")

        _, final_audio, final_rate = result["stages"][-1]
        st.download_button("⬇️ Download final WAV", to_wav_bytes(final_audio, final_rate),
                           file_name=f"{Path(result['name']).stem}{SUFFIX}.wav", mime="audio/wav",
                           key=f"download_{key}")
        if show_spectrograms:
            st.pyplot(spectrogram_figure(result["stages"]))
        st.caption("Timings: " + " · ".join(f"{label} {seconds:.2f}s" for label, seconds in result["timings"].items()))


st.title("🎙️ Voice Isolate + Enhance")
st.caption("Record, upload, or point to a folder → isolate the main voice → enhance to 48 kHz → post-process.")

with st.sidebar:
    st.header("Settings")
    device_label = st.radio("Enhancer device", ["Auto (Apple GPU / MPS)", "CPU"])
    use_enhancer = st.checkbox("Enhancer", value=True)
    use_postprocessing = st.checkbox("Post-processing", value=True, help="Filters, loudness normalization, peak limit")
    show_spectrograms = st.checkbox("Show spectrograms", value=True)
device = "cpu" if device_label == "CPU" else "auto"

# Layout: tabs on top, log toggle underneath. The empty placeholder lets logs refresh live during a batch.
main = st.container()
st.divider()
show_logs = st.toggle("Show logs", value=False, help="Same output as the terminal, without leaving the page")
log_area = st.empty()


def render_logs():
    if show_logs:
        log_area.code(get_log_text(), language="log")


with main:
    tab_record, tab_upload, tab_folder = st.tabs(["🎤 Record", "⬆️ Upload", "📁 Folder"])

    # ---------------- Record ----------------
    with tab_record:
        recording = st.audio_input("Click the mic to start / stop recording")
        if st.button("✨ Process recording", type="primary", disabled=recording is None):
            try:
                st.session_state["record_result"] = process_bytes(
                    recording.getvalue(), "recording.wav", device, use_enhancer, use_postprocessing)
            except Exception as error:
                logger.exception("Processing failed")
                st.error(f"{type(error).__name__}: {error}")
        if "record_result" in st.session_state:
            render_result(st.session_state["record_result"], "record", show_spectrograms)

    # ---------------- Upload ----------------
    with tab_upload:
        uploads = st.file_uploader("Drop audio or video files", type=SUPPORTED_EXTENSIONS, accept_multiple_files=True)
        if st.button("✨ Process uploads", type="primary", disabled=not uploads):
            st.session_state["upload_results"] = []
            for upload in uploads:
                try:
                    st.session_state["upload_results"].append(process_bytes(
                        upload.getvalue(), upload.name, device, use_enhancer, use_postprocessing))
                except Exception as error:
                    logger.exception("Processing failed for %s", upload.name)
                    st.error(f"{upload.name}: {type(error).__name__}: {error}")
        for index, result in enumerate(st.session_state.get("upload_results", [])):
            render_result(result, f"upload_{index}", show_spectrograms, expanded=index == 0)

    # ---------------- Folder ----------------
    with tab_folder:
        st.write(f"Enter a folder path **on the machine running Streamlit**. Each file gets a `name{SUFFIX}.wav` "
                 "next to it (`clip.mp4` → `clip_fine_voice.wav`).")
        folder_text = st.text_input("Folder path", placeholder="/Users/me/data")
        col_left, col_right = st.columns(2)
        with col_left:
            run_all = st.toggle("Run on ALL files (overwrite)", value=False,
                                help="Off = skip files that already have a _fine_voice.wav")
            recursive = st.toggle("Include sub-folders", value=False)
        with col_right:
            skip_bad = st.toggle("Skip corrupted files", value=True,
                                 help="On = log the bad file and continue. Off = stop at the first bad file")
        out_text = st.text_input("Output folder (optional)", placeholder="Empty = same folder as each source file")

        if st.button("✨ Process folder", type="primary"):
            folder = Path(folder_text).expanduser()
            if not folder_text.strip() or not folder.is_dir():
                st.error("Folder not found.")
                logger.error("Folder not found: %r", folder_text)
            else:
                bar = st.progress(0.0, text="Loading models…")
                try:
                    models = load_models(device, use_enhancer, use_postprocessing)

                    def on_progress(done: int, total: int, name: str):
                        bar.progress(done / total if total else 1.0, text=f"{done}/{total} · {name}")
                        render_logs()

                    results = run_batch(
                        folder, *models, overwrite=run_all, skip_corrupted=skip_bad, recursive=recursive,
                        out_dir=Path(out_text).expanduser() if out_text.strip() else None,
                        progress_cb=on_progress,
                    )
                    st.session_state["batch_results"] = pd.DataFrame([r.__dict__ for r in results])
                    bar.progress(1.0, text="Finished")
                except Exception as error:
                    logger.exception("Batch failed")
                    st.error(f"{type(error).__name__}: {error}")

        batch = st.session_state.get("batch_results")
        if batch is not None:
            if batch.empty:
                st.info("No supported audio/video files found in that folder.")
            else:
                counts = batch["status"].value_counts()
                st.success(" | ".join(f"{status}: {count}" for status, count in counts.items()))
                st.dataframe(batch, use_container_width=True, hide_index=True)

# ---------------- Logs ----------------
render_logs()
if show_logs:
    left, right = st.columns(2)
    if left.button("Clear logs"):
        clear_logs()
        st.rerun()
    right.download_button("Download logs", get_log_text(3000), "finevoice_logs.txt")