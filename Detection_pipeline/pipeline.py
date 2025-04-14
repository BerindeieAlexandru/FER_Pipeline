import cv2
import numpy as np
import argparse
import os
import logging
import time
import json
from collections import deque

try:
    from face_detection import FaceDetectorFactory, FaceDetector
except ImportError:
    logging.error("FATAL: Failed to import from face_detection.py. Make sure it's in the same directory or Python path.")
    exit(1)
try:
    from fer import FEREnsemble
except ImportError:
    logging.error("FATAL: Failed to import from fer.py. Make sure it's in the same directory or Python path.")
    exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_FER_CONFIG = [
    ('Eva02_wide', 1, r'../Models/EVA02_cross_db/eva02_base_best.pth'),
    # ('ResEmoteNet', 0.25, r'Models/ResEmoteNet/ResEmoteNet_best.pth'),
    # ('EfficientNetV2', 0.25, r'Models/EfficientNet_V2M/efficientnet_v2m_best.pth'),
]
try:
    total_w = sum(w for _, w, _ in DEFAULT_FER_CONFIG)
    if abs(total_w) > 1e-6 and abs(total_w - 1.0) > 1e-6:
        logging.warning(f"Normalizing default FER config weights as they sum to {total_w}")
        DEFAULT_FER_CONFIG = [(name, w / total_w, path) for name, w, path in DEFAULT_FER_CONFIG]
except Exception as e:
    logging.error(f"Error processing default FER config weights: {e}")


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Face Detection and Emotion Recognition Pipeline",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--source", required=True, help="Input source: path to image, video, or camera ID.")
    parser.add_argument("--output_dir", default="output_pipeline", help="Directory to save output.")
    parser.add_argument("--save_output", action="store_true", help="Enable saving output.")
    parser.add_argument("--no_display", action="store_true", help="Do not display output window.")

    available_detectors = ['mediapipe', 'haar', 'deepface']
    parser.add_argument("--detector", default="mediapipe", choices=available_detectors, help="Face detector type.")

    detector_args_help = """JSON string containing arguments for the selected face detector's constructor.
Examples:
  - For 'deepface': '{"backend": "retinaface"}'
      (Common backends: 'opencv', 'ssd', 'dlib', 'mtcnn', 'retinaface', 'fastmtcnn')
  - For 'haar': '{"scaleFactor": 1.2, "minNeighbors": 5, "minSize": [30, 30]}'
      (minSize should be a list or tuple [width, height])
      (Also possible: '{"cascade_path": "/path/to/your/cascade.xml"}')
  - For 'mediapipe': '{"model_selection": 1, "min_detection_confidence": 0.7}'
      (model_selection: 0 for short-range, 1 for full-range)
Default: "{}" (uses detector's internal defaults)
"""
    parser.add_argument("--detector_args", type=str, default="{}",
                        help=detector_args_help)

    parser.add_argument("--smoothing", type=int, default=10, help="Frames for prediction smoothing (0=off).")

    args = parser.parse_args()

    try:
        json.loads(args.detector_args)
    except json.JSONDecodeError as e:
        parser.error(f"--detector_args must be valid JSON. Error: {e}")

    return args

def draw_probability_bars(frame, probabilities, labels, x_start, y_start, bar_width=100, bar_height=15, gap=5):
    """Draws probability bars at a FIXED location on the frame."""
    if probabilities is None or not isinstance(probabilities, np.ndarray) or probabilities.size == 0:
        return

    max_prob_index = np.argmax(probabilities)
    frame_h, frame_w = frame.shape[:2]

    overlay_x = x_start - 5
    overlay_y = y_start - 5
    overlay_h = len(labels) * (bar_height + gap) + 5
    overlay_w = bar_width + 55
    if overlay_x >= 0 and overlay_y >=0 and overlay_x + overlay_w < frame_w and overlay_y + overlay_h < frame_h:
        sub_img = frame[overlay_y : overlay_y + overlay_h, overlay_x : overlay_x + overlay_w]
        white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 50
        res = cv2.addWeighted(sub_img, 0.6, white_rect, 0.4, 1.0)
        frame[overlay_y : overlay_y + overlay_h, overlay_x : overlay_x + overlay_w] = res

    for i, (prob, label) in enumerate(zip(probabilities, labels)):
        current_y = y_start + i * (bar_height + gap)
        if current_y + bar_height > frame_h: break

        short_label = label[:4]
        text = f"{short_label}: {prob:.2f}"
        text_x = x_start
        text_y = current_y + bar_height - (gap//2)
        cv2.putText(frame, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        bar_x1 = x_start + 45
        bar_y1 = current_y
        bar_length = int(prob * bar_width)
        bar_x2 = min(frame_w -1, bar_x1 + bar_length)
        bar_y2 = current_y + bar_height

        if bar_x2 > bar_x1 and bar_y2 > bar_y1 :
            bar_color = (0, 255, 0) if i == max_prob_index else (0, 0, 255)
            cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), bar_color, -1)

def process_frame(frame: np.ndarray, detector: FaceDetector, fer_ensemble: FEREnsemble, probability_history: deque = None):
    """Detects faces, predicts emotions, and draws results with static bars and labels on boxes."""
    frame_copy = frame.copy()
    frame_h, frame_w = frame.shape[:2]
    first_face_probs = None

    try:
        bounding_boxes = detector.detect_faces(frame)
        detected_results = []

        for i, (x, y, w, h) in enumerate(bounding_boxes):
            if w <= 0 or h <= 0: continue
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
            roi_w, roi_h = x2 - x1, y2 - y1
            if roi_w <= 0 or roi_h <= 0: continue
            roi = frame[y1:y2, x1:x2]

            try:
                probabilities = fer_ensemble.predict_proba(roi)
                if probabilities is not None and probabilities.size == fer_ensemble.num_classes:
                     detected_results.append({'box': (x, y, w, h), 'probabilities': probabilities})

                # --- Update Smoothing History (if enabled) ---
                if probability_history is not None:
                    if probabilities is not None and probabilities.size == fer_ensemble.num_classes:
                        probability_history.append(probabilities)

                # --- Store Probs for Static Display (First Face Only) ---
                # Use the SMOOTHED value if available, otherwise the raw probability
                if i == 0:
                    if probability_history is not None and len(probability_history) > 0:
                        first_face_probs = np.mean(probability_history, axis=0)
                    elif probabilities is not None and probabilities.size == fer_ensemble.num_classes:
                        first_face_probs = probabilities

            except Exception as fer_err:
                logging.error(f"Error during FER prediction for an ROI: {fer_err}", exc_info=False)

        if first_face_probs is not None:
            static_bar_x = frame_w - 155
            static_bar_y = 10
            draw_probability_bars(frame_copy, first_face_probs, fer_ensemble.EMOTION_LABELS,
                                  static_bar_x, static_bar_y, bar_width=100)

        for result in detected_results:
            x, y, w, h = result['box']
            probabilities = result['probabilities']

            pred_index = np.argmax(probabilities)
            confidence = float(probabilities[pred_index])
            pred_label = fer_ensemble.EMOTION_LABELS[pred_index]
            label_text = f"{pred_label}: {confidence:.2f}"

            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            bg_y = max(y - text_h - baseline - 2, 0)
            cv2.rectangle(frame_copy, (x, bg_y), (x + text_w, y - baseline + 2),
                          (0, 255, 0), -1)
            cv2.putText(frame_copy, label_text, (x, y - baseline - 1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    except Exception as det_err:
        logging.error(f"Error during face detection or processing loop: {det_err}", exc_info=True)

    return frame_copy

def main():
    args = parse_arguments()
    start_time_global = time.time()
    logging.info("--- Initializing Pipeline Components ---")
    detector = None
    fer_ensemble = None

    # 1. Initialize Face Detector
    try:
        detector_kwargs = json.loads(args.detector_args)
        if 'minSize' in detector_kwargs and isinstance(detector_kwargs['minSize'], list):
             detector_kwargs['minSize'] = tuple(detector_kwargs['minSize'])

        logging.info(f"Initializing face detector: {args.detector} with args: {detector_kwargs}")

        detector = FaceDetectorFactory(args.detector, **detector_kwargs)

        logging.info(f"OK: Initialized face detector: {args.detector}")
    except ValueError as e:
        logging.error(f"FATAL: Failed to initialize face detector '{args.detector}': {e}", exc_info=False)
        exit(1)
    except TypeError as e:
         logging.error(f"FATAL: Type error initializing detector '{args.detector}' with args {detector_kwargs}. "
                       f"Check if args match detector's constructor in face_detection.py. Error: {e}", exc_info=True)
         exit(1)
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize face detector '{args.detector}': {e}", exc_info=True)
        exit(1)

    # 2. Initialize FER Ensemble
    try:
        fer_config = DEFAULT_FER_CONFIG
        if not fer_config: logging.error("FER config empty."); exit(1)
        logging.info(f"Initializing FEREnsemble with {len(fer_config)} models.")
        fer_ensemble = FEREnsemble(ensemble_config=fer_config)
        logging.info("OK: Initialized FEREnsemble.")
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize FEREnsemble: {e}", exc_info=True)
        exit(1)

    # 3. Setup Output Directory
    if args.save_output:
        try: os.makedirs(args.output_dir, exist_ok=True); logging.info(f"Output dir: {args.output_dir}")
        except OSError as e: logging.error(f"Cannot create output dir '{args.output_dir}': {e}"); args.save_output = False

    # --- Input Source Handling
    logging.info("--- Processing Input Source ---")
    source = args.source
    cap = None
    video_writer = None
    is_video_source = False

    try:
        if source.isdigit():
            cam_id = int(source)
            cap = cv2.VideoCapture(cam_id)
            if not cap.isOpened(): raise IOError(f"Cannot open webcam ID: {cam_id}")
            logging.info(f"Processing webcam feed (ID: {cam_id}). Press 'q' to quit.")
            is_video_source = True
        elif os.path.isfile(source):
            img = cv2.imread(source)
            if img is not None:
                logging.info(f"Processing image file: {source}")
                processed_frame = process_frame(img, detector, fer_ensemble, probability_history=None) # No smoothing for single image
                if args.save_output:
                    fname = os.path.splitext(os.path.basename(source))[0] + "_output.jpg"
                    out_path = os.path.join(args.output_dir, fname)
                    try: cv2.imwrite(out_path, processed_frame); logging.info(f"Saved image to {out_path}")
                    except Exception as e: logging.error(f"Failed to save image {out_path}: {e}")
                if not args.no_display:
                    cv2.imshow("FER Pipeline Output", processed_frame)
                    logging.info("Press any key to close.")
                    cv2.waitKey(0)
                logging.info("Image processing complete.")
                return
            else:
                cap = cv2.VideoCapture(source)
                if not cap.isOpened(): raise IOError(f"Cannot open source as video or image: {source}")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                if total_frames <= 0 or fps <= 0:
                    cap.release()
                    raise IOError(f"Invalid video file (zero frames or FPS): {source}")
                logging.info(f"Processing video file: {source} ({total_frames} frames, {fps:.2f} FPS)")
                is_video_source = True
        else:
            raise FileNotFoundError(f"Input source not found or invalid: {source}")

        if is_video_source and args.save_output:
            if cap and cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                output_fps = fps if 'fps' in locals() and fps > 0 else 30.0
                fname_base = f"webcam_{int(time.time())}" if source.isdigit() else os.path.splitext(os.path.basename(source))[0]
                output_filename = os.path.join(args.output_dir, f"{fname_base}_output.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_filename, fourcc, output_fps, (w, h))
                if not video_writer.isOpened():
                    logging.error(f"Could not open VideoWriter for {output_filename}. Check codec and permissions.")
                    video_writer = None
                else:
                    logging.info(f"Saving video to {output_filename}")
            else:
                 logging.error("Video capture not opened, cannot setup video writer.")
                 args.save_output = False

        # --- Video Processing Loop---
        if is_video_source and cap:
            frame_count = 0
            start_time_proc = time.time()
            probability_history = deque(maxlen=args.smoothing) if args.smoothing > 0 else None

            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.info("End of video stream or cannot read frame.")
                    break
                frame_count += 1

                processed_frame = process_frame(frame, detector, fer_ensemble, probability_history)

                if video_writer:
                    video_writer.write(processed_frame)

                if not args.no_display:
                    current_time = time.time()
                    elapsed_time = current_time - start_time_proc
                    fps_proc = frame_count / elapsed_time if elapsed_time > 0 else 0
                    cv2.putText(processed_frame, f"Proc FPS: {fps_proc:.1f}", (10, processed_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                    cv2.imshow("FER Pipeline Output", processed_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logging.info("Exit key 'q' pressed.")
                        break

    # --- Error Handling and Cleanup ---
    except (FileNotFoundError, IOError, TypeError, ValueError, RuntimeError) as e:
        logging.error(f"Pipeline Error: {e}", exc_info=False)
    except Exception as e:
        logging.exception("An unexpected error occurred in the main pipeline loop.")
    finally:
        if cap and cap.isOpened():
            cap.release()
            logging.info("Video capture released.")
        if video_writer:
            video_writer.release()
            logging.info("Video writer released.")
        cv2.destroyAllWindows()
        logging.info("Output windows closed.")
        end_time_global = time.time()
        logging.info(f"--- Pipeline finished. Total time: {end_time_global - start_time_global:.2f}s ---")

if __name__ == "__main__":
    main()