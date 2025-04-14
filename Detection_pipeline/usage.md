# Fer Pipeline

This pipeline uses various face detection algorithms and an ensemble of deep learning models to perform real-time or offline facial emotion recognition (FER).

## Features

* **Multiple Input Sources:** Process single images, video files, or live webcam streams.
* **Configurable Face Detectors:** Choose between different face detection algorithms:
  * `mediapipe`: Efficient and robust detector from Google's MediaPipe framework.
  * `haar`: Classic detector based on Haar Cascades (requires OpenCV).
  * `deepface`: Wrapper around various modern face detection models (e.g., RetinaFace, MTCNN, Dlib) via the DeepFace library.
* **Customizable Detector Settings:** Fine-tune parameters for each detector (e.g., backend model for DeepFace, confidence threshold for MediaPipe, scale factor for Haar).
* **Ensemble Emotion Recognition:** Uses `FEREnsemble` (defined in `fer.py`) to combine predictions from multiple Facial Emotion Recognition (FER) models for potentially improved accuracy and robustness. (Configured within the script).
* **Real-time Display:** Shows the processed video/image with bounding boxes, predicted emotions, confidence scores, and probability bars for the primary face.
* **Output Saving:** Option to save the processed output as an image (for image input) or video file (for video/webcam input).
* **Prediction Smoothing:** Optional temporal smoothing of emotion predictions over multiple frames for more stable results in videos.

## Requirements

Ensure you have the necessary Python libraries installed. You might need libraries like:

* OpenCV (`opencv-python`)
* NumPy (`numpy`)
* PyTorch (`torch`, `torchvision`)
* MediaPipe (`mediapipe`)
* DeepFace (`deepface`, `dlib`, `mtcnn`, `facenet-pytorch`)
* TIMM (`timm`)

It's recommended to use a virtual environment and install using requirements.txt file.

## Command-Line Arguments

Below is a detailed explanation of the available command-line arguments:

## Required Argument

* **`--source <path_or_id>`**  
  Specifies the input.
  * **Usage Options:**
    * **Image file:** Provide a path to an image file (e.g., `images/person.jpg`)
    * **Video file:** Provide a path to a video file (e.g., `videos/interview.mp4`)
    * **Camera ID:** Specify a camera ID (e.g., `0` for the default webcam, `1` for the next available device, etc.)

## Optional Arguments

* **`--output_dir <directory>`**  
  Directory where output files will be saved if `--save_output` is used.
  * **Default:** `./output_pipeline` (the directory will be created if it does not exist)

* **`--save_output`**  
  If present, saves the processed output.
  * **Output Naming:**
    * For image input, saves as `<original_name>_output.jpg`
    * For video/webcam input, saves as `<original_name_or_timestamp>_output.mp4`

* **`--no_display`**  
  If present, suppresses the OpenCV display window showing the processed output.
  * **Use Case:** Useful for batch processing or running without a GUI

* **`--detector <name>`**  
  Selects the face detection algorithm.
  * **Choices:**
    * `mediapipe` (default)
    * `haar`
    * `deepface`

* **`--detector_args <json_string>`**  
  (Advanced) Allows passing specific configuration parameters to the chosen face detector as a JSON string.
  * **Default:** `{}` (uses the detector's default settings)

* **`--smoothing <integer>`**  
  Number of past frames to average emotion predictions over for smoothing.
  * **Notes:**
    * Set to `0` to disable smoothing.
    * **Default:** `10`

## Configuring Detectors (`--detector_args`)

This argument allows you to fine-tune the selected face detector. You must provide a valid JSON string enclosed in single quotes (`' '`) in most shells to avoid issues with double quotes inside the JSON.

### DeepFace

Select a specific backend model.

**Common backends:**  
`opencv`, `ssd`, `dlib`, `mtcnn`, `retinaface`, `mediapipe`, `fastmtcnn`

**Commands:**

```bash
python pipeline.py --source 0 --detector deepface --detector_args '{"backend": "retinaface"}'
```

```bash
python pipeline.py --source video.mp4 --detector deepface --detector_args '{"backend": "ssd"}'
```

### Haar Cascade

Adjust detection parameters or specify a custom cascade file.

**scaleFactor:** Parameter specifying how much the image size is reduced at each image scale. (e.g., 1.1, 1.2). Must be > 1.0.

**minNeighbors:** Parameter specifying how many neighbors each candidate rectangle should have to retain it. Higher values result in fewer detections but higher quality (e.g., 3, 5, 6).

**minSize:** Minimum possible object size. Objects smaller than this are ignored. Provide as a list [width, height] in the JSON string (it will be converted to a tuple internally; e.g., [30, 30], [50, 50]).

```bash
python pipeline.py --source 0 --detector haar --detector_args '{"scaleFactor": 1.2, "minNeighbors": 6, "minSize": [40, 40]}'
```

## MediaPipe

Configure model selection and detection confidence.
**model_selection:** 0 for a short-range model (best for faces within 2 meters), 1 for a full-range model (best for faces within 5 meters).
**min_detection_confidence:** Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful (e.g., 0.5, 0.7).

```bash
python pipeline.py --source 0 --detector mediapipe --detector_args '{"model_selection": 1, "min_detection_confidence": 0.75}'
```

## Usage examples

1. Process a single image using the default MediaPipe detector:

```bash
python pipeline.py --source path/to/your/image.jpg
```

2. Process a video using the DeepFace detector with the mtcnn backend and save the output:
   
```bash
python pipeline.py --source path/to/your/video.mp4 --detector deepface --detector_args '{"backend": "mtcnn"}' --save_output
```

1. Run on webcam 0 using the Haar detector with custom parameters, disable smoothing, and save the output:

```bash
python pipeline.py --source 0 --detector haar --detector_args '{"scaleFactor": 1.3, "minNeighbors": 5, "minSize": [50, 50]}' --smoothing 0 --save_output
```

4. Process a video using MediaPipe's full-range model without displaying the output window:

```bash
python pipeline.py --source path/to/your/video.mp4 --detector mediapipe --detector_args '{"model_selection": 1}' --no_display --save_output
```
 ## Ensemble configuration

 The Facial Emotion Recognition (FER) models are configured directly within the pipeline script using the DEFAULT_FER_CONFIG list near the top of the file.

 ```python
DEFAULT_FER_CONFIG = [
    ('Eva02_wide', 0.5, r'../Models/EVA02_cross_db/eva02_base_best.pth'),
    ('ResEmoteNet', 0.25, r'Models/ResEmoteNet/ResEmoteNet_best.pth'),
    ('EfficientNetV2', 0.25, r'Models/EfficientNet_V2M/efficientnet_v2m_best.pth'),
]
 ```

 The list of available models is the following:

 ```python
MODEL_REGISTRY = {
        'EfficientNetB0': {'loader': load_efficientnet_b0, 'size': 224},
        'EfficientNetV2': {'loader': load_efficientnetv2, 'size': 224},
        'Eva': {'loader': load_eva, 'size': 196},
        'Eva02': {'loader': load_eva02, 'size': 224},
        'Eva02_wide': {'loader': load_eva02_wide, 'size': 224},
        'FBNetV3B': {'loader': load_fbnetv3b, 'size': 224},
        'MobileNetV3': {'loader': load_mobilenetv3, 'size': 224},
        'MobileNetV4': {'loader': load_mobilenetv4, 'size': 224},
        'ResEmoteNet': {'loader': load_resemotenet, 'size': 64},
        'ResNeSt': {'loader': load_resnest, 'size': 224},
        'ResNeXt50': {'loader': load_resnext50_32x4d, 'size': 224},
        'VGG19': {'loader': load_vgg, 'size': 224}
    }
 ```

 Add them in a similar way, make sure path to checkpoints is correct and you adjust weights to be equal to 1 otherwise it will be normalized automatically.