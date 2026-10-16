"""Gradio UI for video face-swapping on Colab's GPU.

Run locally for a smoke test:
    python app/gradio_app.py

On Colab the notebook calls `create_demo().launch(share=True)`.
Gradio gives RGB arrays; the swapper core expects BGR, hence conversions.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

from .swapper import has_gpu, swap_image, swap_video


def _rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Tab 1: image -> image
# ---------------------------------------------------------------------------

def ui_swap_image(source_rgb, target_rgb, swap_all: bool):
    if source_rgb is None or target_rgb is None:
        raise gr.Error("Upload both a source face and a target image.")
    out_bgr = swap_image(
        _rgb_to_bgr(source_rgb), _rgb_to_bgr(target_rgb), swap_all=swap_all
    )
    return _bgr_to_rgb(out_bgr)


# ---------------------------------------------------------------------------
# Tab 2: image -> video
# ---------------------------------------------------------------------------

def ui_swap_video(source_rgb, target_video_path, swap_all, ref_rgb, threshold, progress=gr.Progress()):
    if source_rgb is None:
        raise gr.Error("Upload a source face image first.")
    if not target_video_path:
        raise gr.Error("Upload a target video first.")
    ref_bgr = _rgb_to_bgr(ref_rgb) if ref_rgb is not None else None
    out_path = str(Path(tempfile.gettempdir()) / "swapped_output.mp4")
    result = swap_video(
        source_bgr=_rgb_to_bgr(source_rgb),
        target_video_path=target_video_path,
        output_path=out_path,
        swap_all=swap_all,
        reference_bgr=ref_bgr,
        similarity_threshold=float(threshold),
        progress_cb=lambda f: progress(f, desc="Swapping faces..."),
    )
    return result


def ui_preview_frame(target_video_path, frame_number: float):
    if not target_video_path:
        return None
    cap = cv2.VideoCapture(target_video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = min(int(frame_number), max(total - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return _bgr_to_rgb(frame)


# ---------------------------------------------------------------------------
# App assembly
# ---------------------------------------------------------------------------

def create_demo() -> gr.Blocks:
    gpu_label = "GPU (CUDA)" if has_gpu() else "CPU"
    with gr.Blocks(title="Video FaceSwap (Colab GPU)") as demo:
        gr.Markdown(
            "# Video FaceSwap — runs on Colab GPU\n"
            f"**Backend:** InsightFace `inswapper_128` (same model family as Rope) · "
            f"**Runtime:** {gpu_label}\n\n"
            "Use only faces you have consent to use."
        )
        with gr.Tab("Image → Image"):
            with gr.Row():
                src_img = gr.Image(label="Source face (identity to inject)", type="numpy")
                tgt_img = gr.Image(label="Target image (face to replace)", type="numpy")
            swap_all_img = gr.Checkbox(label="Swap all faces in target", value=True)
            btn_img = gr.Button("Swap photo", variant="primary")
            out_img = gr.Image(label="Result", type="numpy")
            btn_img.click(ui_swap_image, [src_img, tgt_img, swap_all_img], out_img)

        with gr.Tab("Image → Video"):
            with gr.Row():
                with gr.Column():
                    src_vid_face = gr.Image(label="Source face", type="numpy")
                    tgt_vid = gr.Video(label="Target video")
                    swap_all_vid = gr.Checkbox(label="Swap all faces per frame", value=True)
                    with gr.Accordion("Only swap one person (optional)", open=False):
                        gr.Markdown(
                            "Upload a clear crop of the person to target. "
                            "Other faces in the video will be left alone."
                        )
                        ref_face = gr.Image(label="Reference face (optional)", type="numpy")
                        threshold = gr.Slider(0.1, 0.8, value=0.4, step=0.05,
                                              label="Match threshold (higher = stricter)")
                    btn_vid = gr.Button("Swap video", variant="primary")
                with gr.Column():
                    out_vid = gr.Video(label="Swapped video")
                    gr.Markdown("**Preview a frame** (to check which face / threshold to use):")
                    frame_slider = gr.Slider(0, 500, value=0, step=1, label="Frame number")
                    frame_preview = gr.Image(label="Frame preview", type="numpy")
                    tgt_vid.upload(
                        lambda p: gr.update(maximum=max(int(cv2.VideoCapture(p).get(cv2.CAP_PROP_FRAME_COUNT) or 1) - 1, 0), value=0),
                        tgt_vid, frame_slider,
                    )
                    frame_slider.change(ui_preview_frame, [tgt_vid, frame_slider], frame_preview)
            btn_vid.click(ui_swap_video,
                          [src_vid_face, tgt_vid, swap_all_vid, ref_face, threshold],
                          out_vid)
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public gradio.live link (use on Colab)")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    create_demo().queue().launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
