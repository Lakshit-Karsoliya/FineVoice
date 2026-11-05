"""Process every audio/video file in a folder: clip.mp4 -> clip_fine_voice.wav"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .logger import get_logger
from .pipeline import run_stages
from .voice_utils import AUDIO_EXTENSIONS, load_audio_file, save_wav

logger = get_logger("batch")

VIDEO_EXTENSIONS = ["mp4", "mov", "mkv", "avi", "webm", "flv", "wmv", "m4v", "ts", "mpg", "mpeg", "3gp"]
SUPPORTED_EXTENSIONS = sorted(set(AUDIO_EXTENSIONS) | set(VIDEO_EXTENSIONS))  # without the dot
_SUPPORTED_SUFFIXES = {f".{ext}" for ext in SUPPORTED_EXTENSIONS}
SUFFIX = "_fine_voice"


@dataclass
class BatchResult:
    file: str
    status: str  # done | skipped_existing | skipped_corrupted | failed
    detail: str = ""
    output: str = ""


def scan_folder(folder: Path, recursive: bool = False) -> list[Path]:
    """Supported media files, ignoring hidden files and our own *_fine_voice outputs."""
    folder = Path(folder)
    if recursive:
        listing = [(root, names) for root, _, names in os.walk(folder)]
    else:
        listing = [(str(folder), os.listdir(folder))]  # raises on permission errors instead of returning nothing
    found = []
    for root, names in listing:
        for name in names:
            path = Path(root) / name
            if (path.suffix.lower() in _SUPPORTED_SUFFIXES and not name.startswith(".")
                    and not path.stem.endswith(SUFFIX) and path.is_file()):
                found.append(path)
    if not found:
        logger.warning("No supported media files in %s (recursive=%s). Entries seen: %s",
                       folder, recursive, os.listdir(folder)[:20])
    return sorted(found)


def plan_outputs(files: list[Path], out_dir: Optional[Path]) -> dict[Path, Path]:
    """clip.mp4 -> clip_fine_voice.wav (if clip.mov also exists it gets clip_mov_fine_voice.wav)."""
    plan, taken = {}, set()
    for source in files:
        target_dir = out_dir or source.parent
        target = target_dir / f"{source.stem}{SUFFIX}.wav"
        if target in taken:
            target = target_dir / f"{source.stem}_{source.suffix.lstrip('.').lower()}{SUFFIX}.wav"
        taken.add(target)
        plan[source] = target
    return plan


def run_batch(
    folder: Path,
    isolator,
    enhancer=None,
    postprocessor=None,
    overwrite: bool = False,        # True: process every file. False: skip files that already have an output
    skip_corrupted: bool = True,    # True: log the bad file and continue. False: stop the batch at the first problem
    recursive: bool = False,
    out_dir: Optional[Path] = None,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> list[BatchResult]:
    files = scan_folder(folder, recursive)
    print("hello",files)
    plan = plan_outputs(files, out_dir)
    total = len(files)
    logger.info("Batch started: %s | %d media file(s) | recursive=%s | mode=%s | on error: %s",
                folder, total, recursive,
                "process ALL (overwrite)" if overwrite else "skip already processed",
                "skip and continue" if skip_corrupted else "stop")
    results: list[BatchResult] = []

    for number, source in enumerate(files, start=1):
        tag, target = f"[{number}/{total}]", plan[source]
        if progress_cb:
            progress_cb(number - 1, total, source.name)

        if target.exists() and not overwrite:
            logger.info("%s SKIP already processed: %s -> %s", tag, source.name, target.name)
            results.append(BatchResult(source.name, "skipped_existing", "output already exists", str(target)))
            continue

        try:  # decoding doubles as the corruption check (also fails for videos without audio)
            audio = load_audio_file(source)
        except Exception as error:
            reason = (str(error).splitlines() or [type(error).__name__])[0][:300]
            logger.warning("%s CORRUPTED/UNREADABLE: %s | %s", tag, source.name, reason)
            results.append(BatchResult(source.name, "skipped_corrupted", reason))
            if skip_corrupted:
                continue
            logger.error("Stopping batch: 'skip corrupted' is off")
            break

        temp = target.with_name(f".{target.stem}.tmp.wav")
        try:
            logger.info("%s Processing %s", tag, source.name)
            stages, _ = run_stages(audio, isolator, enhancer, postprocessor)
            _, final_audio, sample_rate = stages[-1]
            target.parent.mkdir(parents=True, exist_ok=True)
            save_wav(temp, final_audio, sample_rate)
            os.replace(temp, target)  # atomic: a half-written file is never mistaken for "already processed"
            logger.info("%s DONE -> %s", tag, target)
            results.append(BatchResult(source.name, "done", "", str(target)))
        except Exception as error:
            logger.exception("%s FAILED %s", tag, source.name)
            temp.unlink(missing_ok=True)
            results.append(BatchResult(source.name, "failed", f"{type(error).__name__}: {error}"[:300]))
            if not skip_corrupted:
                logger.error("Stopping batch: 'skip corrupted' is off")
                break

    if progress_cb:
        progress_cb(total, total, "finished")
    summary = {}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    logger.info("Batch finished: %s", summary or "nothing to do")
    return results