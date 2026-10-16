"""Headless face-swap core.

Same model family Rope uses (InsightFace ``inswapper_128`` + ``buffalo_l``
detector) but with no Qt / OpenGL / TensorRT / Windows dependencies, so it
runs on a Linux Colab GPU with just ``onnxruntime-gpu``.

All images are expected as BGR ``numpy`` arrays (OpenCV convention).
Gradio hands us RGB, so conversion happens in ``gradio_app.py``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Model locations
# ---------------------------------------------------------------------------

INSIGHTFACE_HOME = Path(os.environ.get("INSIGHTFACE_HOME", Path.home() / ".insightface"))
MODELS_DIR = INSIGHTFACE_HOME / "models"
INSWAPPER_PATH = MODELS_DIR / "inswapper_128.onnx"
BUFFALO_DIR = MODELS_DIR / "buffalo_l"

INSWAPPER_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"

_analyser = None
_swapper = None


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------

def has_gpu() -> bool:
    """True if onnxruntime sees a CUDA execution provider."""
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


def ensure_models() -> None:
    """Raise a helpful error if the model files are missing.

    The Colab notebook downloads these; we don't auto-download here so a
    missing model fails fast with a clear message instead of hanging.
    """
    missing = []
    if not INSWAPPER_PATH.exists():
        missing.append(f"{INSWAPPER_PATH} (download from {INSWAPPER_URL})")
    if not BUFFALO_DIR.exists():
        missing.append(f"{BUFFALO_DIR} (unzip {BUFFALO_URL} there)")
    if missing:
        raise FileNotFoundError(
            "Missing face-swap models:\n  - " + "\n  - ".join(missing)
        )


def get_analyser(det_size: tuple[int, int] = (640, 640)):
    """Lazily build the InsightFace FaceAnalysis detector."""
    global _analyser
    if _analyser is None:
        ensure_models()
        from insightface.app import FaceAnalysis

        _analyser = FaceAnalysis(name="buffalo_l", root=str(INSIGHTFACE_HOME))
        _analyser.prepare(ctx_id=0 if has_gpu() else -1, det_size=det_size)
    return _analyser


def get_swapper():
    """Lazily load the inswapper_128 ONNX model."""
    global _swapper
    if _swapper is None:
        ensure_models()
        import insightface

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if has_gpu()
            else ["CPUExecutionProvider"]
        )
        _swapper = insightface.model_zoo.get_model(
            str(INSWAPPER_PATH), providers=providers
        )
    return _swapper


# ---------------------------------------------------------------------------
# Face helpers
# ---------------------------------------------------------------------------

def _largest_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def get_source_face(source_bgr: np.ndarray):
    """Return the largest face in the source image (the identity to inject)."""
    faces = get_analyser().get(source_bgr)
    if not faces:
        raise ValueError("No face found in the source image. Try a clear front-facing photo.")
    return _largest_face(faces)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Image swapping
# ---------------------------------------------------------------------------

def swap_image(
    source_bgr: np.ndarray,
    target_bgr: np.ndarray,
    swap_all: bool = True,
) -> np.ndarray:
    """Swap face(s) in a single image. Returns BGR array."""
    analyser = get_analyser()
    swapper = get_swapper()

    source_face = get_source_face(source_bgr)
    target_faces = analyser.get(target_bgr)
    if not target_faces:
        raise ValueError("No face found in the target image.")

    out = target_bgr
    faces_to_swap = target_faces if swap_all else [_largest_face(target_faces)]
    for face in faces_to_swap:
        out = swapper.get(out, face, source_face, paste_back=True)
    return out


# ---------------------------------------------------------------------------
# Video swapping
# ---------------------------------------------------------------------------

def _mux_audio(silent_video: str, original_video: str, final_video: str) -> str:
    """Copy audio from the original onto the swapped (silent) video.

    Uses the imageio-ffmpeg bundled binary so no `apt install ffmpeg` is
    needed on Colab. If anything fails (e.g. source has no audio track),
    falls back to the silent video.
    """
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg, "-y",
            "-i", silent_video,
            "-i", original_video,
            "-c:v", "copy", "-c:a", "aac",
            "-map", "0:v:0", "-map", "1:a:0?",
            "-shortest",
            final_video,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return final_video
    except Exception as e:
        print(f"[swapper] audio mux skipped ({e}), returning silent video")
        shutil.copyfile(silent_video, final_video)
        return final_video


def swap_video(
    source_bgr: np.ndarray,
    target_video_path: str,
    output_path: str | None = None,
    swap_all: bool = True,
    reference_bgr: np.ndarray | None = None,
    similarity_threshold: float = 0.4,
    max_frames: int | None = None,
    progress_cb=None,
) -> str:
    """Swap faces across every frame of a video.

    Args:
        source_bgr: identity to inject (BGR array).
        target_video_path: video to modify.
        output_path: where to write the result (defaults next to input).
        swap_all: swap every detected face per frame.
        reference_bgr: optional image identifying *which* face to swap.
            Per frame, only faces with cosine similarity >= threshold to the
            reference face's embedding are swapped. Useful for multi-person
            videos. If None, every face is swapped.
        similarity_threshold: 0.3-0.6 is a sane range; higher = stricter.
        max_frames: for quick previews (None = whole video).
        progress_cb: callable(fraction: float) for UI progress bars.
    """
    analyser = get_analyser()
    swapper = get_swapper()
    source_face = get_source_face(source_bgr)

    ref_embedding = None
    if reference_bgr is not None:
        ref_faces = analyser.get(reference_bgr)
        if ref_faces:
            ref_embedding = _largest_face(ref_faces).normed_embedding

    cap = cv2.VideoCapture(target_video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {target_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if max_frames:
        total = min(total or max_frames, max_frames)

    if output_path is None:
        stem = Path(target_video_path).stem
        output_path = str(Path(target_video_path).with_name(f"{stem}_swapped.mp4"))

    tmp_fd, tmp_silent = tempfile.mkstemp(suffix="_silent.mp4")
    os.close(tmp_fd)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_silent, fourcc, fps, (width, height))

    frame_idx = 0
    try:
        while True:
            if max_frames is not None and frame_idx >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break

            faces = analyser.get(frame)
            if faces:
                if ref_embedding is not None:
                    faces = [
                        f for f in faces
                        if _cosine(f.normed_embedding, ref_embedding) >= similarity_threshold
                    ]
                elif not swap_all:
                    faces = [_largest_face(faces)]
                for face in faces:
                    frame = swapper.get(frame, face, source_face, paste_back=True)

            writer.write(frame)
            frame_idx += 1
            if progress_cb and total:
                progress_cb(min(frame_idx / total, 1.0))
    finally:
        cap.release()
        writer.release()

    if frame_idx == 0:
        raise ValueError("No frames could be read from the video.")

    final = _mux_audio(tmp_silent, target_video_path, output_path)
    try:
        os.remove(tmp_silent)
    except OSError:
        pass
    return final
