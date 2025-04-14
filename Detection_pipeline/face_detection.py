import abc
import cv2
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FaceDetector(abc.ABC):
    """Abstract base class for face detectors."""

    @abc.abstractmethod
    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        pass

    def extract_rois(self, image: np.ndarray) -> list[np.ndarray]:
        rois = []
        bounding_boxes = self.detect_faces(image)
        height, width = image.shape[:2]
        for (x, y, w, h) in bounding_boxes:
            # Ensure coordinates are within image bounds
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width, x + w)
            y2 = min(height, y + h)
            # Extract ROI only if valid dimensions
            if x2 > x1 and y2 > y1:
                roi = image[y1:y2, x1:x2]
                rois.append(roi)
        return rois

# 1. MediaPipe implementation
try:
    import mediapipe as mp

    class MediaPipeFaceDetector(FaceDetector):
        def __init__(self, min_detection_confidence=0.75, model_selection=1):
            """
                min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for face detection.
                model_selection: 0 for short-range model (<= 2m), 1 for full-range model (<= 5m).
            """
            self.use_task_api = False
            self.detector_task = None
            self.face_detection_legacy = None

            try:
                # Try to use the new Task API first
                if hasattr(mp.tasks, 'vision') and hasattr(mp.tasks.vision, 'FaceDetector'):
                    model_path = 'face_detector.task'
                    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
                    options = mp.tasks.vision.FaceDetectorOptions(
                        base_options=base_options,
                        min_detection_confidence=min_detection_confidence,
                        running_mode=mp.tasks.vision.RunningMode.IMAGE
                    )
                    self.detector_task = mp.tasks.vision.FaceDetector.create_from_options(options)
                    self.use_task_api = True
                    logging.info("MediaPipe Face Detector initialized using Task API (model: %s).", model_path)
                else:
                    raise RuntimeError("MediaPipe Task API components not found.")

            except Exception as e_task:
                logging.warning("Failed to initialize MediaPipe Task API (%s). Falling back to legacy Solutions API.", e_task)
                try:
                    self.face_detection_legacy = mp.solutions.face_detection.FaceDetection(
                        min_detection_confidence=min_detection_confidence,
                        model_selection=model_selection
                    )
                    self.use_task_api = False
                    logging.info("MediaPipe Face Detector initialized using legacy Solutions API.")
                except Exception as e_legacy:
                     logging.error("Failed to initialize MediaPipe Legacy API as well: %s", e_legacy)
                     raise RuntimeError("Could not initialize MediaPipe using either Task API or Legacy API.") from e_legacy


        def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            bounding_boxes = []
            img_h, img_w = image.shape[:2]

            if self.use_task_api and self.detector_task:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                try:
                    detection_result = self.detector_task.detect(mp_image)
                    if detection_result.detections:
                        for detection in detection_result.detections:
                            bbox = detection.bounding_box
                            # Convert origin_x, origin_y, width, height to x, y, w, h integer tuple
                            # Ensure coordinates are within image bounds after calculation
                            x = max(0, bbox.origin_x)
                            y = max(0, bbox.origin_y)
                            w = min(img_w - x, bbox.width) # Adjust width if it goes past edge
                            h = min(img_h - y, bbox.height) # Adjust height if it goes past edge
                            if w > 0 and h > 0: # Only add valid boxes
                                bbox_tuple = (int(x), int(y), int(w), int(h))
                                bounding_boxes.append(bbox_tuple)
                except Exception as e:
                    logging.error("Error during MediaPipe Task API detection: %s", e)

            elif not self.use_task_api and self.face_detection_legacy: # Legacy API
                try:
                    results = self.face_detection_legacy.process(img_rgb)
                    if results.detections:
                        for detection in results.detections:
                            bboxC = detection.location_data.relative_bounding_box
                            # Handle potential None values if detection is weak or out of bounds
                            if bboxC.xmin is None or bboxC.ymin is None or bboxC.width is None or bboxC.height is None:
                                continue
                            # Convert relative coordinates to absolute, clamping to image bounds
                            x = max(0, int(bboxC.xmin * img_w))
                            y = max(0, int(bboxC.ymin * img_h))
                            x2 = min(img_w, int((bboxC.xmin + bboxC.width) * img_w))
                            y2 = min(img_h, int((bboxC.ymin + bboxC.height) * img_h))
                            w = x2 - x
                            h = y2 - y
                            if w > 0 and h > 0: # Only add valid boxes
                                bounding_boxes.append((x, y, w, h))
                except Exception as e:
                    logging.error("Error during MediaPipe Legacy API detection: %s", e)
            else:
                 logging.error("MediaPipe detector was not properly initialized.")


            return bounding_boxes

        def __del__(self):
            # Safely close the Task API detector if it was successfully created
            # Check attributes exist before accessing them
            if hasattr(self, 'use_task_api') and self.use_task_api and hasattr(self, 'detector_task') and self.detector_task:
                 try:
                     self.detector_task.close()
                     logging.info("MediaPipe Task API Face Detector closed.")
                 except Exception as e:
                     logging.warning(f"Error closing MediaPipe Task API detector: {e}")
            elif hasattr(self, 'use_task_api') and not self.use_task_api and hasattr(self, 'face_detection_legacy') and self.face_detection_legacy:
                 # Legacy API doesn't have an explicit close usually needed for simple detection
                 logging.info("MediaPipe Legacy API Face Detector resources released (implicitly).")
            # No else needed: If initialization failed early, attributes might not exist, nothing to clean up.

except ImportError:
    logging.warning("MediaPipe library not found or failed to import. MediaPipeFaceDetector will not be available.")
    MediaPipeFaceDetector = None

# 2. Haar Cascade (OpenCV)
class HaarCascadeFaceDetector(FaceDetector):
    """Face detector using OpenCV's Haar Cascades."""
    def __init__(self, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)):
        haar_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
        if not os.path.exists(haar_cascade_path):
             logging.error(f"Haar Cascade file not found at expected location: {haar_cascade_path}. Please ensure OpenCV is installed correctly or provide the correct path.")
             pass

        self.face_cascade = cv2.CascadeClassifier(haar_cascade_path)
        if self.face_cascade.empty():
            raise IOError(f"Could not load Haar Cascade classifier from {haar_cascade_path}. Check OpenCV installation and file path.")

        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.minSize = minSize
        logging.info("Haar Cascade Face Detector initialized.")

    def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
        # Convert to grayscale if the image is not already
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 2:
            gray = image # Already grayscale
        else:
            logging.error("Unsupported image format for Haar Cascade (expected BGR or Grayscale).")
            return []

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize
        )
        return [tuple(face) for face in faces]

# 3. DeepFace
try:
    from deepface import DeepFace
    class DeepFaceDetector(FaceDetector):
        ALLOWED_BACKENDS = ['mtcnn', 'retinaface', 'ssd', 'dlib', 'fastmtcnn']

        def __init__(self, backend='dlib'):
            backend_lower = backend.lower()
            # Validate backend against the allowed list
            if backend_lower not in self.ALLOWED_BACKENDS:
                raise ValueError(f"Invalid backend '{backend}' specified for DeepFaceDetector in this configuration. "
                                 f"Allowed options are: {self.ALLOWED_BACKENDS}")

            self.backend = backend_lower
            try:
                 from deepface.commons import functions as deepface_functions
                 available_deepface_backends = deepface_functions.get_detector_backends()
                 if self.backend not in available_deepface_backends:
                     logging.warning(f"Backend '{self.backend}' is allowed by configuration but might not be supported by the installed DeepFace version. Available in DeepFace: {available_deepface_backends}")
            except ImportError:
                 logging.warning("Could not import DeepFace internal functions to verify backend availability.")
            except Exception as e:
                 logging.warning(f"Error checking DeepFace backend availability: {e}")
            logging.info(f"DeepFace Detector initialized with backend: '{self.backend}'. Models load on first use.")

        def detect_faces(self, image: np.ndarray) -> list[tuple[int, int, int, int]]:
            bounding_boxes = []
            try:
                face_objs = DeepFace.extract_faces(
                    img_path=image.copy(),
                    detector_backend=self.backend,
                    enforce_detection=False,
                    align=False
                )
                for face_obj in face_objs:
                    if isinstance(face_obj, dict) and 'facial_area' in face_obj:
                        area = face_obj['facial_area']
                        if all(k in area for k in ('x', 'y', 'w', 'h')):
                             # Basic sanity check for non-negative width/height
                             if area['w'] > 0 and area['h'] > 0:
                                 bounding_boxes.append((area['x'], area['y'], area['w'], area['h']))
                             else:
                                 logging.debug(f"DeepFace returned invalid bbox dimensions: {area}")
                        else:
                             logging.warning(f"DeepFace facial_area dict missing keys: {area}")
                    else:
                         logging.warning(f"Unexpected object type returned by DeepFace.extract_faces: {type(face_obj)}")


            except Exception as e:
                logging.error(f"DeepFace detection with backend '{self.backend}' failed: {e}")
                if "not found" in str(e).lower() or "download" in str(e).lower():
                     logging.warning(f"DeepFace might be attempting to download model files for backend '{self.backend}'. Check internet connection and permissions.")
            return bounding_boxes

except ImportError:
    logging.warning("deepface library not found. DeepFaceDetector will not be available.")
    DeepFaceDetector = None


def FaceDetectorFactory(detector_type: str = 'haar', **kwargs) -> FaceDetector:
    detector_map = {
        'mediapipe': MediaPipeFaceDetector,
        'haar': HaarCascadeFaceDetector,
        'deepface': DeepFaceDetector,
    }

    detector_class = detector_map.get(detector_type.lower())

    if detector_class is None:
         if detector_type.lower() in ['mediapipe', 'deepface'] :
              raise ValueError(f"Detector type '{detector_type}' requires a library that is not installed or failed to import. Please install 'mediapipe' or 'deepface'.")
         else:
              raise ValueError(f"Unknown detector type: '{detector_type}'. Available types: {list(detector_map.keys())}")

    try:
        return detector_class(**kwargs)
    except TypeError as e:
        logging.error(f"Failed to instantiate {detector_type} detector. Check arguments: {kwargs}. Error: {e}")
        raise ValueError(f"Incorrect arguments provided for detector type '{detector_type}'. Error: {e}") from e
    except ValueError as e:
        logging.error(f"Failed to instantiate {detector_type} detector due to invalid configuration: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during {detector_type} detector instantiation: {e}")
        raise


# if __name__ == "__main__":

#     # Choose the detector type: 'mediapipe', 'haar', 'deepface'
#     DETECTOR_TYPE = 'mediapipe'

#     # Optional arguments for the chosen detector
#     detector_args = {}
#     if DETECTOR_TYPE == 'haar':
#         detector_args = {'scaleFactor': 1.1, 'minNeighbors': 5}
#     elif DETECTOR_TYPE == 'mediapipe':
#          detector_args = {'min_detection_confidence': 0.75}
#     elif DETECTOR_TYPE == 'deepface':
#         # Choose a backend from: 'mtcnn', 'retinaface', 'ssd', 'dlib', 'fastmtcnn'
#         detector_args = {'backend': 'mtcnn'}

#     try:
#         detector = FaceDetectorFactory(DETECTOR_TYPE, **detector_args)
#     except ValueError as e:
#         logging.error(f"Error creating detector: {e}")
#         exit()
#     except Exception as e:
#         logging.error(f"An unexpected error occurred during detector initialization: {e}")
#         exit()


    # --- Processing ---
    # Option 1: Process a single image (Uncomment to use)
    # image_path = r"D:\Alex\Desktop\test2.jpeg" # <--- Change to your image path
    # if os.path.exists(image_path):
    #     img = cv2.imread(image_path)
    #     if img is None:
    #         logging.error(f"Failed to load image: {image_path}")
    #     else:
    #         logging.info(f"Processing image: {image_path}")
    #         bounding_boxes = detector.detect_faces(img)
    #         rois = detector.extract_rois(img) # Alternatively, get ROIs directly

    #         logging.info(f"Found {len(bounding_boxes)} face(s).")

    #         # Draw bounding boxes on the image
    #         img_copy = img.copy() # Draw on a copy
    #         for i, (x, y, w, h) in enumerate(bounding_boxes):
    #             cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #             # Display the extracted ROI (optional)
    #             # if i < len(rois) and rois[i] is not None and rois[i].size > 0:
    #             #      try:
    #             #          cv2.imshow(f"ROI_{i+1}", rois[i])
    #             #      except cv2.error as e:
    #             #          logging.warning(f"Could not display ROI_{i+1}: {e}")


    #         cv2.imshow(f"Face Detection ({DETECTOR_TYPE})", img_copy)
    #         logging.info("Press any key to close image windows...")
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    # else:
    #     logging.warning(f"Test image not found at: {image_path}. Skipping image processing example.")


    # Option 2: Process video stream (webcam) - Default enabled
    # logging.info("Starting webcam processing. Press 'q' to quit.")
    # cap = cv2.VideoCapture(0) # 0 is typically the default webcam

    # if not cap.isOpened():
    #     logging.error("Cannot open webcam")
    #     exit()

    # frame_count = 0
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         logging.error("Can't receive frame (stream end?). Exiting ...")
    #         break

    #     frame_count += 1
    #     # --- Face Detection ---
    #     # Optional: Skip frames for performance (e.g., process every 3rd frame)
    #     # if frame_count % 3 != 0:
    #     #     # Draw previous boxes if needed, or just skip detection
    #     #     pass # For now, just process every frame
    #     # else:
    #     #     bounding_boxes = detector.detect_faces(frame)

    #     # Process every frame
    #     bounding_boxes = detector.detect_faces(frame)

    #     # --- Visualization ---
    #     frame_copy = frame.copy()
    #     for i, (x, y, w, h) in enumerate(bounding_boxes):
    #         # Basic check for valid coordinates before drawing
    #         if w > 0 and h > 0:
    #             cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

    #     # Display info
    #     detector_info = DETECTOR_TYPE
    #     if DETECTOR_TYPE == 'deepface':
    #         detector_info += f" ({detector_args.get('backend', 'N/A')})"

    #     cv2.putText(frame_copy, f"Detector: {detector_info}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    #     cv2.putText(frame_copy, f"Faces: {len(bounding_boxes)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    #     cv2.imshow('Real-time Face Detection - Press Q to Quit', frame_copy)

    #     # --- Exit Condition ---
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         logging.info("Quit key pressed. Exiting webcam loop.")
    #         break
    #     # Add other key controls if needed

    # # --- Cleanup ---
    # cap.release()
    # cv2.destroyAllWindows()
    # logging.info("Webcam and windows released.")