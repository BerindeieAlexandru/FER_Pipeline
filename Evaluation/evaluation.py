import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    vgg19_bn,
    VGG19_BN_Weights,
)
from PIL import Image
import timm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm
from collections import defaultdict
from Models.ResEmoteNet.model_arhitecture import ResEmoteNet
from data_processor import DataProcessor

def remove_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_state_dict[k[len(prefix):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def get_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_efficientnet_b0(checkpointname=r'Models\EfficientNet_B0\efficientnet_b0_best.pth'):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 7)
    model.load_state_dict(torch.load(checkpointname, weights_only=True)['model_state_dict'])
    model.eval()
    return model

def load_efficientnetv2(checkpointname=r'Models\EfficientNet_V2M\efficientnet_v2m_best.pth'):
    model = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.DEFAULT)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 7)
    model.load_state_dict(torch.load(checkpointname, weights_only=True)['model_state_dict'])
    model.eval()
    return model

def load_eva(checkpointname=r'Models\Eva\eva_best.pth'):
    model = timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=True, num_classes=7)
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_eva02(checkpointname=r'Models\Eva02_base\eva02_best.pth'):
    model = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=True, num_classes=7)
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_eva02(checkpointname=r'Models\Eva02_base\eva02_best.pth'):
    model = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=True, num_classes=7)
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_eva02_wide(checkpointname=r"Models\EVA02_cross_db\eva02_base_best.pth"):
    model = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=False, num_classes=7)
    checkpoint = torch.load(checkpointname, weights_only=True)
    state_dict = remove_prefix_from_state_dict(checkpoint['model_state_dict'], prefix="_orig_mod.")
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_fbnetv3b(checkpointname=r'Models\FBNetV3b\fbnetv3b_best.pth'):
    model = timm.create_model('fbnetv3_b.ra2_in1k', pretrained=True, num_classes=7)
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_mobilenetv3(checkpointname=r'Models\MobileNetV3\mobilenetv3_best.pth'):
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 7)
    model.load_state_dict(torch.load(checkpointname, weights_only=True)['model_state_dict'])
    model.eval()
    return model

def load_mobilenetv4(checkpointname=r'Models\MobileNetV4\mobilenetv4_best.pth'):
    model = timm.create_model('mobilenetv4_hybrid_large.e600_r384_in1k', pretrained=True, num_classes=7)
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_resemotenet(checkpointname=r'Models\ResEmoteNet\ResEmoteNet_best.pth'):
    model = ResEmoteNet()
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_resnest(checkpointname=r'Models\ResNeSt50_f+\resnest50_best.pth'):
    model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 7)
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_resnext50_32x4d(checkpointname=r'Models\ResNeXt50_32x4d\resnext50_32x4d_best.pth'):
    model = resnext50_32x4d(weights="ResNeXt50_32X4D_Weights.IMAGENET1K_V2")
    model.fc = nn.Linear(model.fc.in_features, 7)
    checkpoint = torch.load(checkpointname, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def load_vgg(checkpointname=r'Models\VGG\vgg19_best.pth'):
    model = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, 7)
    model.load_state_dict(torch.load(checkpointname, weights_only=True)['model_state_dict'])
    model.eval()
    return model

def evaluate_model(model, model_name, test_loader, criterion, device, num_classes=7):
    model.to(device)
    test_targets = []
    test_predictions = []
    test_running_loss = 0.0

    with torch.no_grad():
        for data in tqdm(test_loader, desc=f"Testing {model_name}"):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_targets.extend(labels.cpu().numpy())
            test_predictions.extend(predicted.cpu().numpy())

    test_loss = test_running_loss / len(test_loader)
    test_acc = accuracy_score(test_targets, test_predictions)
    macro_f1 = f1_score(test_targets, test_predictions, average='macro', zero_division=0)
    weighted_precision = precision_score(test_targets, test_predictions, average='weighted', zero_division=0)
    weighted_recall = recall_score(test_targets, test_predictions, average='weighted', zero_division=0)
    weighted_f1 = f1_score(test_targets, test_predictions, average='weighted', zero_division=0)
    class_report_str = classification_report(test_targets, test_predictions, zero_division=0)
    cm = confusion_matrix(test_targets, test_predictions)

    return test_loss, test_acc, macro_f1, weighted_precision, weighted_recall, weighted_f1, class_report_str, cm

def evaluate_ensemble(ensemble_name, ensemble_config, model_loaders, model_input_sizes, dataloaders_dict, device, num_classes=7):
    models_by_size = defaultdict(list)
    for model_name, weight in ensemble_config:
        size = model_input_sizes[model_name]
        model = model_loaders[model_name]().to(device)
        models_by_size[size].append((model, weight))
    
    required_input_sizes = list(models_by_size.keys())
    required_dataloaders = [dataloaders_dict[size] for size in required_input_sizes]
    
    test_targets = []
    test_predictions = []
    
    with torch.no_grad():
        for batches in zip(*required_dataloaders):
            labels = batches[0][1].to(device)
            total_weighted_probs = torch.zeros((len(labels), num_classes), device=device)
            
            for idx, size in enumerate(required_input_sizes):
                inputs = batches[idx][0].to(device)
                models_weights = models_by_size[size]
                for model, weight in models_weights:
                    outputs = model(inputs)
                    probs = torch.softmax(outputs, dim=1)
                    total_weighted_probs += probs * weight
            
            sum_weights = sum(weight for _, weight in ensemble_config)
            final_probs = total_weighted_probs / sum_weights
            _, predicted = torch.max(final_probs, 1)
            
            test_targets.extend(labels.cpu().numpy())
            test_predictions.extend(predicted.cpu().numpy())
    
    test_acc = accuracy_score(test_targets, test_predictions)
    weighted_precision = precision_score(test_targets, test_predictions, average='weighted', zero_division=0)
    weighted_recall = recall_score(test_targets, test_predictions, average='weighted', zero_division=0)
    weighted_f1 = f1_score(test_targets, test_predictions, average='weighted', zero_division=0)
    class_report_str = classification_report(test_targets, test_predictions, zero_division=0)
    cm = confusion_matrix(test_targets, test_predictions)
    
    return test_acc, weighted_precision, weighted_recall, weighted_f1, class_report_str, cm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transforms_dict = {
        '224': get_transform(224),
        '196': get_transform(196),
        '64': get_transform(64)
    }

    datasets_dict = {
        size: DataProcessor(
            csv_file=r"Datasets\RAFDB\test_labels.csv",
            img_dir=r"Datasets\RAFDB\test",
            transform=transform
        ) for size, transform in transforms_dict.items()
    }

    dataloaders_dict = {
        size: DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True
        ) for size, dataset in datasets_dict.items()
    }

    model_input_sizes = {
        'EfficientNetB0': '224',
        'EfficientNetV2': '224',
        'FBNetV3B': '224',
        'MobileNetV3': '224',
        'MobileNetV4': '224',
        'ResNeSt': '224',
        'ResNeXt50': '224',
        'VGG19': '224',
        'Eva': '196',
        'Eva02': '224',
        'Eva02_wide': '224',
        'ResEmoteNet': '64'
    }

    model_loaders = {
        'EfficientNetB0': lambda: load_efficientnet_b0(),
        'EfficientNetV2': lambda: load_efficientnetv2(),
        'Eva': lambda: load_eva(),
        'Eva02': lambda: load_eva02(),
        'Eva02_wide': lambda: load_eva02_wide(),
        'FBNetV3B': lambda: load_fbnetv3b(),
        'MobileNetV3': lambda: load_mobilenetv3(),
        'MobileNetV4': lambda: load_mobilenetv4(),
        'ResEmoteNet': lambda: load_resemotenet(),
        'ResNeSt': lambda: load_resnest(),
        'ResNeXt50': lambda: load_resnext50_32x4d(),
        'VGG19': lambda: load_vgg()
    }

    criterion = nn.CrossEntropyLoss()
    results = []

    # Evaluate individual models
    for model_name, load_func in model_loaders.items():
        print(f"\nEvaluating {model_name}")
        try:
            model = load_func()
            input_size = model_input_sizes[model_name]
            test_loader = dataloaders_dict[input_size]
            test_loss, test_acc, macro_f1, weighted_precision, weighted_recall, weighted_f1, class_report_str, cm = evaluate_model(
                model, model_name, test_loader, criterion, device
            )
            
            results.append({
                'model': model_name,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'macro_f1': macro_f1,
                'weighted_precision': weighted_precision,
                'weighted_recall': weighted_recall,
                'weighted_f1': weighted_f1,
                'classification_report': class_report_str,
                'confusion_matrix': cm,
            })
            
            print(f"{model_name} Test Loss: {test_loss:.4f}")
            print(f"{model_name} Test Accuracy: {test_acc:.4f}")
            print(f"{model_name} Macro F1-Score: {macro_f1:.4f}")
            print(f"{model_name} Weighted Precision: {weighted_precision:.4f}")
            print(f"{model_name} Weighted Recall: {weighted_recall:.4f}")
            print(f"{model_name} Weighted F1-Score: {weighted_f1:.4f}")
            print(f"{model_name} Classification Report:\n{class_report_str}")
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")

    # # Evaluate ensemble(s)
    # ensemble_configs = [
    #     {
    #         'name': 'Ensemble_1',
    #         'models': [
    #             ('Eva02', 0.4),
    #             ('Eva', 0.2),
    #             ('ResEmoteNet', 0.3),
    #             ('EfficientNetV2', 0.1)
    #         ]
    #     }
    # ]

    # for ensemble in ensemble_configs:
    #     print(f"\nEvaluating {ensemble['name']}")
    #     try:
    #         test_acc, weighted_precision, weighted_recall, weighted_f1, class_report_str, cm = evaluate_ensemble(
    #             ensemble['name'],
    #             ensemble['models'],
    #             model_loaders,
    #             model_input_sizes,
    #             dataloaders_dict,
    #             device
    #         )
            
    #         results.append({
    #             'model': ensemble['name'],
    #             'test_accuracy': test_acc,
    #             'weighted_precision': weighted_precision,
    #             'weighted_recall': weighted_recall,
    #             'weighted_f1': weighted_f1,
    #             'classification_report': class_report_str,
    #             'confusion_matrix': cm,
    #         })
            
    #         print(f"{ensemble['name']} Test Accuracy: {test_acc:.4f}")
    #         print(f"{ensemble['name']} Weighted Precision: {weighted_precision:.4f}")
    #         print(f"{ensemble['name']} Weighted Recall: {weighted_recall:.4f}")
    #         print(f"{ensemble['name']} Weighted F1-Score: {weighted_f1:.4f}")
    #         print(f"{ensemble['name']} Classification Report:\n{class_report_str}")
            
    #     except Exception as e:
    #         print(f"Error evaluating {ensemble['name']}: {str(e)}")

    # Create summary DataFrame with uniform metrics for all models
    summary_rows = []
    for r in results:
        summary_rows.append((
            r['model'],
            r.get('test_accuracy', None),
            r.get('weighted_precision', None),
            r.get('weighted_recall', None),
            r.get('weighted_f1', None)
        ))
    
    summary = pd.DataFrame(summary_rows, columns=['Model', 'Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1'])
    
    print("\nPerformance Summary:")
    print(summary.to_string(index=False))
    summary.to_csv('model_comparison.csv', index=False)
    
    best_model = max(results, key=lambda x: x['test_accuracy'])
    print(f"\nBest model based on accuracy: {best_model['model']} with accuracy {best_model['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
