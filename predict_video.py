import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

from wasr_t.data.transforms import PytorchHubNormalization
from wasr_t.inference import Predictor
from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.mobile_wasr_t import wasr_temporal_lraspp_mobilenetv3
from wasr_t.utils import load_weights, Option

# Colors corresponding to each segmentation class (obstacle, water, sky)
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

HIST_LEN = 5
SIZE = (512, 384)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR-T Video Inference")
    parser.add_argument("--video", type=str, required=True,
                         help="Path to the input video file.")
    parser.add_argument("--weights", type=str, required=True,
                         help="Model weights file.")
    parser.add_argument("--output", type=str, default="output/video_predictions.mp4",
                         help="Path to the output video file.")
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


def load_model(args):
    if args.mobile:
        model = wasr_temporal_lraspp_mobilenetv3(pretrained=False, hist_len=args.hist_len)
    else:
        model = wasr_temporal_resnet101(pretrained=False, hist_len=args.hist_len)

    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict)
    model = model.sequential()  # Enable sequential mode

    return model


def preprocess_frame(frame_bgr, size, normalize_t):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    if size is not None:
        frame_rgb = cv2.resize(frame_rgb, size, interpolation=cv2.INTER_LINEAR)

    tensor = normalize_t(frame_rgb)
    return tensor.unsqueeze(0)  # Add batch dimension


def predict_video(predictor, video_path, output_path, size, overlay_alpha):
    """Runs inference on a video file. Frames are processed sequentially (stateful)."""
    predictor.model.clear_state()

    normalize_t = PytorchHubNormalization()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    pbar = tqdm(total=total_frames, desc='Processing frames')
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        pbar.update(1)

        orig_h, orig_w = frame_bgr.shape[:2]
        batch = {'image': preprocess_frame(frame_bgr, size, normalize_t)}
        probs = predictor.predict_batch(batch)

        class_mask = probs[0].argmax(0).astype(np.uint8)
        color_mask = SEGMENTATION_COLORS[class_mask]
        color_mask = cv2.resize(color_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(frame_bgr, 1 - overlay_alpha, color_mask_bgr, overlay_alpha, 0)

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (orig_w, orig_h))

        writer.write(overlay)

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()


def main():
    args = get_arguments()
    print(args)

    model = load_model(args)
    predictor = Predictor(model, half_precision=args.fp16)

    size = None
    if args.size[0] is not None:
        size = tuple(args.size)

    output_path = Path(args.output)
    predict_video(predictor, args.video, output_path, size, args.overlay_alpha)

    print(f"Saved output video to {output_path}")


if __name__ == '__main__':
    main()
