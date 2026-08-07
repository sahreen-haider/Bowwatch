import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from wasr_t.data.transforms import PytorchHubNormalization
from wasr_t.inference import Predictor
from wasr_t.utils import Option
from predict_video import load_model, preprocess_frame, SEGMENTATION_COLORS, SIZE, HIST_LEN

WINDOW_NAME = "WaSR-T Live"


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR-T Live Streaming Inference")
    parser.add_argument("--video", type=str, required=True,
                         help="Path to a video file to replay as a simulated live feed, or a camera index (e.g. `0`) for a real webcam.")
    parser.add_argument("--weights", type=str, required=True,
                         help="Model weights file.")
    parser.add_argument("--output", type=str, default=None,
                         help="If set, also save the live session to this video file.")
    parser.add_argument("--hist-len", default=HIST_LEN, type=int,
                         help="Number of past frames to be considered in addition to the target frame (context length). Must match the value used in training.")
    parser.add_argument("--fp16", action='store_true',
                         help="Use half precision for inference.")
    parser.add_argument("--mobile", action='store_true',
                         help="Use smaller network for mobile inference.")
    parser.add_argument("--size", type=Option(int), default=SIZE, nargs='+',
                         help="Resize input frames to a specified size (width height). Use `none` for no resizing.")
    parser.add_argument("--overlay-alpha", type=float, default=0.5,
                         help="Blend weight of the segmentation mask over the original frame (0=original only, 1=mask only).")
    return parser.parse_args()


def open_capture(source):
    """Opens a video source. Numeric strings (e.g. "0") are treated as a camera index."""
    try:
        source = int(source)
    except ValueError:
        pass
    return cv2.VideoCapture(source)


def run_live(predictor, source, size, overlay_alpha, output_path):
    """Reads frames from `source`, runs sequential inference, and displays the overlay live.
    Frames are paced to the source's native frame rate to simulate a real-time feed.
    Press 'q' in the display window to stop early."""
    predictor.model.clear_state()
    normalize_t = PytorchHubNormalization()

    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_interval = 1.0 / source_fps

    writer = None
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("Warming up model on first frame (first CUDA call can take up to 30-60s on WSL2)...")

    prev_tic = time.time()
    frame_idx = 0
    try:
        while True:
            loop_start = time.time()

            ret, frame_bgr = cap.read()
            if not ret:
                break

            orig_h, orig_w = frame_bgr.shape[:2]
            batch = {'image': preprocess_frame(frame_bgr, size, normalize_t)}
            probs = predictor.predict_batch(batch)

            class_mask = probs[0].argmax(0).astype(np.uint8)
            color_mask = SEGMENTATION_COLORS[class_mask]
            color_mask = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

            overlay = cv2.addWeighted(frame_bgr, 1 - overlay_alpha, color_mask_bgr, overlay_alpha, 0)

            now = time.time()
            frame_time = now - prev_tic
            fps = 1.0 / frame_time if frame_time > 0 else 0.0
            prev_tic = now
            frame_idx += 1
            print(f"Frame {frame_idx}: {frame_time:.2f}s ({fps:.1f} FPS)")
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if writer is None and output_path is not None:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (orig_w, orig_h))
            if writer is not None:
                writer.write(overlay)

            cv2.imshow(WINDOW_NAME, overlay)

            # Pace playback to the source's native frame rate; 'q' quits early.
            elapsed = time.time() - loop_start
            wait_ms = max(1, int((frame_interval - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    args = get_arguments()
    print(args)

    model = load_model(args)
    predictor = Predictor(model, half_precision=args.fp16)

    size = None
    if args.size[0] is not None:
        size = tuple(args.size)

    run_live(predictor, args.video, size, args.overlay_alpha, args.output)


if __name__ == '__main__':
    main()
