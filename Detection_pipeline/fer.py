import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_v2_m,
    EfficientNet_V2_M_Weights,
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    resnext50_32x4d,
    ResNeXt50_32X4D_Weights,
    vgg19_bn,
    VGG19_BN_Weights,
)
from PIL import Image
import timm
from collections import defaultdict
import logging
import cv2
try:
    from model_arhitecture import ResEmoteNet
except ImportError:
    logging.warning("Could not import ResEmoteNet. Ensure the path is correct if needed.")
    ResEmoteNet = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    new_state_dict = {}
    if state_dict is None:
        return new_state_dict
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def load_efficientnet_b0(checkpoint_path, num_classes=7):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for EfficientNetB0: {e}")
    model.eval()
    return model

def load_efficientnetv2(checkpoint_path, num_classes=7):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for EfficientNetV2M: {e}")
    model.eval()
    return model

def load_eva(checkpoint_path, num_classes=7):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=False, num_classes=num_classes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for Eva: {e}")
    model.eval()
    return model

def load_eva02(checkpoint_path, num_classes=7):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=False, num_classes=num_classes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for Eva02: {e}")
    model.eval()
    return model

def load_eva02_wide(checkpoint_path, num_classes=7):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=False, num_classes=num_classes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        state_dict = remove_prefix_from_state_dict(state_dict, prefix="_orig_mod.")
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for Eva02 Wide: {e}")
    model.eval()
    return model

def load_fbnetv3b(checkpoint_path, num_classes=7):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = timm.create_model('fbnetv3_b.ra2_in1k', pretrained=False, num_classes=num_classes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for FBNetV3b: {e}")
    model.eval()
    return model

def load_mobilenetv3(checkpoint_path, num_classes=7):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for MobileNetV3: {e}")
    model.eval()
    return model

def load_mobilenetv4(checkpoint_path, num_classes=7):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = timm.create_model('mobilenetv4_hybrid_large.e600_r384_in1k', pretrained=False, num_classes=num_classes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for MobileNetV4: {e}")
    model.eval()
    return model

def load_resemotenet(checkpoint_path, num_classes=7):
    if ResEmoteNet is None:
        raise RuntimeError("ResEmoteNet class not available.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = ResEmoteNet()
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for ResEmoteNet: {e}")
    model.eval()
    return model

def load_resnest(checkpoint_path, num_classes=7):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    try:
        model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    except Exception as e:
        raise RuntimeError(f"Failed to load base ResNeSt model from torch hub: {e}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for ResNeSt: {e}")
    model.eval()
    return model

def load_resnext50_32x4d(checkpoint_path, num_classes=7):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for ResNeXt50: {e}")
    model.eval()
    return model

def load_vgg(checkpoint_path, num_classes=7):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, num_classes)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        raise IOError(f"Error loading checkpoint {checkpoint_path} for VGG19: {e}")
    model.eval()
    return model

class FEREnsemble:
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

    EMOTION_LABELS = ['Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Neutral']

    def __init__(self, ensemble_config, device=None, num_classes=7):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        logging.info(f"FEREnsemble using device: {self.device}")

        self.ensemble_models = []
        self.num_classes = num_classes
        self._load_models(ensemble_config)

    def _load_models(self, ensemble_config):
        total_weight = 0
        loaded_model_names = []
        for model_name, weight, checkpoint_path in ensemble_config:
            if model_name not in self.MODEL_REGISTRY:
                logging.warning(f"Model '{model_name}' not found in registry. Skipping.")
                continue

            model_info = self.MODEL_REGISTRY[model_name]
            loader_func = model_info['loader']
            input_size = model_info['size']

            logging.info(f"Loading model: {model_name} from {checkpoint_path} with weight {weight}")
            try:
                model = loader_func(checkpoint_path=checkpoint_path, num_classes=self.num_classes)
                model.to(self.device)
                model.eval()

                self.ensemble_models.append({'model': model, 'weight': weight, 'size': input_size, 'name': model_name})
                total_weight += weight
                loaded_model_names.append(model_name)

            except FileNotFoundError as e:
                logging.error(f"Failed to load {model_name}: Checkpoint file not found at {checkpoint_path}. Skipping.")
            except Exception as e:
                logging.error(f"Failed to load model '{model_name}' from '{checkpoint_path}': {e}. Skipping.")

        if not self.ensemble_models:
            raise ValueError("No models were successfully loaded for the ensemble.")
        else: logging.info(f"Successfully loaded models: {', '.join(loaded_model_names)}")
        if abs(total_weight) < 1e-6:
             logging.warning("Total weight of loaded models is zero. Predictions might be unreliable.")
        elif abs(total_weight - 1.0) > 1e-6:
             logging.warning(f"Sum of weights ({total_weight:.4f}) does not equal 1. Probabilities will be normalized.")

    def _preprocess_image(self, roi_image: np.ndarray, target_size: int) -> torch.Tensor:
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

    @torch.no_grad()
    def predict_proba(self, roi_image: np.ndarray) -> np.ndarray:
        if not self.ensemble_models:
            logging.error("Cannot predict: No models loaded in the ensemble.")
            return np.zeros(self.num_classes, dtype=np.float32)

        total_weighted_probs = torch.zeros(self.num_classes, device=self.device, dtype=torch.float32)
        total_weight = 0.0

        for model_info in self.ensemble_models:
            model = model_info['model']
            weight = model_info['weight']
            size = model_info['size']
            model_name = model_info['name'] # For logging

            try:
                input_tensor = self._preprocess_image(roi_image, size)
                input_tensor = input_tensor.unsqueeze(0).to(self.device)

                outputs = model(input_tensor)
                probs = torch.softmax(outputs.squeeze(0), dim=0)

                total_weighted_probs += probs * weight
                total_weight += weight

            except Exception as e:
                logging.error(f"Error during prediction with model '{model_name}': {e}")
                continue

        if total_weight > 1e-6 :
            final_probs = total_weighted_probs / total_weight
        else:
            logging.error("Prediction failed for all models or total weight is zero.")
            final_probs = torch.zeros(self.num_classes, device=self.device, dtype=torch.float32)


        # Return probabilities as a NumPy array on CPU
        return final_probs.cpu().numpy()

    def predict(self, roi_image: np.ndarray) -> tuple[str, float]:
        probabilities = self.predict_proba(roi_image)

        if probabilities is None or np.sum(probabilities) < 1e-6:
             return "Unknown", 0.0

        predicted_index = np.argmax(probabilities)
        confidence = float(probabilities[predicted_index])

        if predicted_index < len(self.EMOTION_LABELS):
            predicted_label = self.EMOTION_LABELS[predicted_index]
            return predicted_label, confidence
        else:
            logging.error(f"Predicted index {predicted_index} out of bounds for labels.")
            return "Unknown", 0.0

# if __name__ == "__main__":
#     import cv2

#     config = [
#         ('Eva02', 0.3, r'..\Models\Eva02_base\eva02_best.pth'),
#         ('Eva02_wide', 0.3, r'..\Models\EVA02_cross_db\eva02_base_best.pth'),
#         ('ResEmoteNet', 0.20, r'..\Models\ResEmoteNet\ResEmoteNet_best.pth'),
#         ('EfficientNetV2', 0.20, r'..\Models\EfficientNet_V2M\efficientnet_v2m_best.pth'),
#     ]

#     try:
#         # Initialize the ensemble
#         fer_ensemble = FEREnsemble(ensemble_config=config)

#         image_path = r"D:\Alex\Desktop\happy.jpg"
#         if os.path.exists(image_path):
#              logging.info(f"\n--- Predicting on real image: {image_path} ---")
#              real_roi_image = cv2.imread(image_path)
#              if real_roi_image is not None:
#                  pred_label_real, conf_real = fer_ensemble.predict(real_roi_image)
#                  print(f"Real Image Predicted Emotion: {pred_label_real}")
#                  print(f"Real Image Confidence: {conf_real:.4f}")
#              else:
#                  print(f"Failed to load real image: {image_path}")
#         else:
#              print(f"\nSkipping real image test, path not found: {image_path}")

#     except ValueError as ve:
#         logging.error(f"Initialization or Prediction Error: {ve}")
#     except Exception as e:
#         logging.error(f"An unexpected error occurred: {e}")