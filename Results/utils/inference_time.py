import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import logging
from collections import defaultdict
from model_arhitecture import ResEmoteNet
from torchvision import transforms
from typing import Callable, Dict, Tuple

try:
    from fer import (
        FEREnsemble,
        remove_prefix_from_state_dict,
        load_efficientnet_b0, load_efficientnetv2, load_eva, load_eva02,
        load_eva02_wide, load_fbnetv3b, load_mobilenetv3, load_mobilenetv4,
        load_resemotenet, load_resnest, load_resnext50_32x4d, load_vgg,
    )
except ImportError as e:
    logging.error(f"FATAL: Failed to import components from fer.py: {e}. Ensure it's accessible.")
    exit(1)

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ALL_MODELS_CONFIG = {
    'EfficientNetB0': r'../Models/EfficientNet_B0/efficientnet_b0_best.pth',
    'EfficientNetV2': r'../Models/EfficientNet_V2M/efficientnet_v2m_best.pth',
    'Eva':            r'../Models/Eva/eva_best.pth',
    'Eva02':          r'../Models/Eva02_base/eva02_best.pth',
    'Eva02_wide':     r'../Models/EVA02_cross_db/eva02_base_best.pth',
    'FBNetV3B':       r'../Models/FBNetV3b/fbnetv3b_best.pth',
    'MobileNetV3':    r'../Models/MobileNetV3/mobilenetv3_best.pth',
    'MobileNetV4':    r'../Models/MobileNetV4/mobilenetv4_best.pth',
    'ResEmoteNet':    r'../Models/ResEmoteNet/ResEmoteNet_best.pth',
    'ResNeSt':        r'../Models/ResNeSt50_f+/resnest50_best.pth',
    'ResNeXt50':      r'../Models/ResNeXt50_32x4d/resnext50_32x4d_best.pth',
    'VGG19':          r'../Models/VGG/vgg19_best.pth',
}

# --- Ensemble to Benchmark ---
ENSEMBLE_CONFIG_TO_BENCHMARK = [
    ('Eva02', 0.3, ALL_MODELS_CONFIG['Eva02']), # Use path from above dict
    ('Eva', 0.3, ALL_MODELS_CONFIG['Eva']),
    ('ResEmoteNet', 0.20, ALL_MODELS_CONFIG['ResEmoteNet']),
    ('EfficientNetV2', 0.20, ALL_MODELS_CONFIG['EfficientNetV2']),
]
ENSEMBLE_NAME = "MyEnsemble_Eva02_Eva_Res_EffV2" # Give your ensemble a name for the CSV

# --- Benchmarking Parameters ---
IMAGE_PATH = r"D:\Alex\Desktop\happy.jpg"
NUM_WARMUP_RUNS = 5
NUM_TIMED_RUNS = 20
OUTPUT_CSV = "fer_inference_benchmark.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _preprocess_image(roi_image: np.ndarray, target_size: int) -> torch.Tensor:
    if roi_image is None or roi_image.size == 0:
         raise ValueError("Input ROI image is empty.")
    try:
        face = cv2.resize(roi_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_3channel = cv2.merge([face_gray, face_gray, face_gray])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        tensor = transform(face_3channel)
        return tensor
    except Exception as e:
        logging.exception(f"Error during preprocessing for size {target_size}: {e}")
        raise RuntimeError(f"Error applying transforms for size {target_size}: {e}")

# --- Main Benchmarking Logic ---
def main():
    logging.info(f"--- Starting FER Inference Benchmark ---")
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"Input image: {IMAGE_PATH}")
    logging.info(f"Warm-up runs: {NUM_WARMUP_RUNS}, Timed runs: {NUM_TIMED_RUNS}")

    if not os.path.exists(IMAGE_PATH):
        logging.error(f"Input image not found: {IMAGE_PATH}")
        return

    # 1. Load and Preprocess Image for all required sizes
    logging.info("Preprocessing input image for required sizes...")
    preprocessed_tensors = {}
    required_sizes = set(info['size'] for info in FEREnsemble.MODEL_REGISTRY.values()) # Get all unique sizes
    try:
        original_image = cv2.imread(IMAGE_PATH)
        if original_image is None:
            raise IOError(f"Failed to read image: {IMAGE_PATH}")
        for size in required_sizes:
            logging.info(f"Preprocessing for size: {size}x{size}")
            tensor = _preprocess_image(original_image, size)
            preprocessed_tensors[size] = tensor.unsqueeze(0).to(DEVICE)
        logging.info("Image preprocessing complete.")
    except Exception as e:
        logging.error(f"Failed during image loading or preprocessing: {e}", exc_info=True)
        return

    benchmark_results = []

    # 2. Benchmark Individual Models
    logging.info("\n--- Benchmarking Individual Models ---")
    for model_name, model_info in FEREnsemble.MODEL_REGISTRY.items():
        if model_name not in ALL_MODELS_CONFIG:
            logging.warning(f"Skipping benchmark for '{model_name}': Checkpoint path not defined in ALL_MODELS_CONFIG.")
            continue
        if ResEmoteNet is None and model_name == 'ResEmoteNet':
            logging.warning("Skipping ResEmoteNet benchmark: Class not available.")
            continue

        checkpoint_path = ALL_MODELS_CONFIG[model_name]
        loader_func = model_info['loader']
        input_size = model_info['size']
        model = None # Ensure model is reset

        logging.info(f"Benchmarking: {model_name} (Input: {input_size}x{input_size})")

        try:
            # Load model
            model = loader_func(checkpoint_path=checkpoint_path, num_classes=7) # Assuming 7 classes
            model.to(DEVICE)
            model.eval()

            # Get preprocessed input
            input_tensor = preprocessed_tensors[input_size]

            # Warm-up runs
            logging.info(f"  Warm-up ({NUM_WARMUP_RUNS} runs)...")
            with torch.no_grad():
                for _ in range(NUM_WARMUP_RUNS):
                    _ = model(input_tensor)
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()

            # Timed runs
            logging.info(f"  Timing ({NUM_TIMED_RUNS} runs)...")
            timings = []
            with torch.no_grad():
                for _ in range(NUM_TIMED_RUNS):
                    start_time = time.perf_counter()
                    if DEVICE.type == 'cuda': torch.cuda.synchronize() # Sync before inference
                    _ = model(input_tensor)
                    if DEVICE.type == 'cuda': torch.cuda.synchronize() # Sync after inference
                    end_time = time.perf_counter()
                    timings.append(end_time - start_time)

            avg_time_ms = (sum(timings) / NUM_TIMED_RUNS) * 1000 # Convert to milliseconds
            logging.info(f"  Average Inference Time: {avg_time_ms:.3f} ms")
            benchmark_results.append({'Model': model_name, 'Avg Inference Time (ms)': avg_time_ms})

        except FileNotFoundError:
            logging.error(f"  FAILED: Checkpoint not found at {checkpoint_path}")
        except Exception as e:
            logging.error(f"  FAILED: Error benchmarking {model_name}: {e}", exc_info=True)
        finally:
             if model is not None: del model
             if DEVICE.type == 'cuda': torch.cuda.empty_cache()


    # 3. Benchmark Ensemble
    logging.info("\n--- Benchmarking Ensemble ---")
    logging.info(f"Ensemble Name: {ENSEMBLE_NAME}")
    ensemble_models_loaded = {}
    try:
        # Load all models needed for the specific ensemble
        logging.info("  Loading ensemble models...")
        for model_name, _, checkpoint_path in ENSEMBLE_CONFIG_TO_BENCHMARK:
            if model_name in ensemble_models_loaded: continue # Already loaded
            if model_name not in FEREnsemble.MODEL_REGISTRY:
                logging.error(f"  Model '{model_name}' in ensemble not found in registry. Cannot benchmark ensemble."); break
            if ResEmoteNet is None and model_name == 'ResEmoteNet':
                logging.error("  ResEmoteNet required for ensemble but class not loaded. Cannot benchmark ensemble."); break

            model_info = FEREnsemble.MODEL_REGISTRY[model_name]
            loader_func = model_info['loader']
            model = loader_func(checkpoint_path=checkpoint_path, num_classes=7)
            model.to(DEVICE)
            model.eval()
            ensemble_models_loaded[model_name] = model
        else:
            logging.info("  Ensemble models loaded.")

            # Prepare inputs for ensemble models
            ensemble_inputs = {}
            for model_name, _, _ in ENSEMBLE_CONFIG_TO_BENCHMARK:
                 input_size = FEREnsemble.MODEL_REGISTRY[model_name]['size']
                 if input_size not in ensemble_inputs:
                      ensemble_inputs[input_size] = preprocessed_tensors[input_size]

            # Warm-up runs for ensemble prediction
            logging.info(f"  Warm-up ({NUM_WARMUP_RUNS} runs)...")
            with torch.no_grad():
                for _ in range(NUM_WARMUP_RUNS):
                    total_weighted_logits = torch.zeros(7, device=DEVICE, dtype=torch.float32) # Assuming 7 classes
                    total_weight = 0.0
                    for model_name, weight, _ in ENSEMBLE_CONFIG_TO_BENCHMARK:
                        model = ensemble_models_loaded[model_name]
                        input_size = FEREnsemble.MODEL_REGISTRY[model_name]['size']
                        input_tensor = ensemble_inputs[input_size]
                        logits = model(input_tensor).squeeze(0)
                        total_weighted_logits += logits * weight
                        total_weight += weight
                    if total_weight > 1e-6:
                        averaged_logits = total_weighted_logits / total_weight
                        _ = torch.softmax(averaged_logits, dim=0) # Final step
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()

            # Timed runs for ensemble prediction
            logging.info(f"  Timing ({NUM_TIMED_RUNS} runs)...")
            timings = []
            with torch.no_grad():
                for _ in range(NUM_TIMED_RUNS):
                    start_time = time.perf_counter()
                    if DEVICE.type == 'cuda': torch.cuda.synchronize() # Sync before

                    total_weighted_logits = torch.zeros(7, device=DEVICE, dtype=torch.float32)
                    total_weight = 0.0
                    for model_name, weight, _ in ENSEMBLE_CONFIG_TO_BENCHMARK:
                        model = ensemble_models_loaded[model_name]
                        input_size = FEREnsemble.MODEL_REGISTRY[model_name]['size']
                        input_tensor = ensemble_inputs[input_size]
                        logits = model(input_tensor).squeeze(0)
                        total_weighted_logits += logits * weight
                        total_weight += weight
                    if total_weight > 1e-6:
                        averaged_logits = total_weighted_logits / total_weight
                        _ = torch.softmax(averaged_logits, dim=0) # Final step

                    if DEVICE.type == 'cuda': torch.cuda.synchronize() # Sync after
                    end_time = time.perf_counter()
                    timings.append(end_time - start_time)

            avg_time_ms = (sum(timings) / NUM_TIMED_RUNS) * 1000
            logging.info(f"  Average Ensemble Inference Time: {avg_time_ms:.3f} ms")
            benchmark_results.append({'Model': ENSEMBLE_NAME, 'Avg Inference Time (ms)': avg_time_ms})

    except Exception as e:
        logging.error(f"Failed during ensemble benchmark: {e}", exc_info=True)
    finally:
        # Clean up ensemble models
        for model in ensemble_models_loaded.values():
             if model is not None: del model
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()


    # 4. Save Results to CSV
    if benchmark_results:
        df = pd.DataFrame(benchmark_results)
        try:
            df.to_csv(OUTPUT_CSV, index=False)
            logging.info(f"\nBenchmark results saved to: {OUTPUT_CSV}")
            print("\n--- Benchmark Summary ---")
            print(df.to_string(index=False))
            print("-----------------------")
        except Exception as e:
            logging.error(f"Failed to save results to CSV {OUTPUT_CSV}: {e}")
    else:
        logging.warning("No benchmark results were generated.")

    logging.info("--- Benchmark Complete ---")

if __name__ == "__main__":
    main()