import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
import torch.optim as optim
from data_processor import DataProcessor
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
import torch.nn.functional as F
import torch.nn as nn
import timm

def compute_class_weights(csv_file, num_classes):

    df = pd.read_csv(csv_file, header=None, names=["image", "label"])
    labels = df["label"].tolist()
    
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    
    class_weights = [1 - (class_counts.get(i, 0) / total_samples) for i in range(num_classes)]

    print(f"Class Weights: {class_weights}")
    return class_weights

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7

    # Tracked metrics
    metrics = {
        "train_accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "val_accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "val_precision": Precision(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "val_recall": Recall(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "val_f1_score": F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "val_auroc": AUROC(task="multiclass", num_classes=num_classes).to(device),
    }

    model = timm.create_model('eva_large_patch14_196.in22k_ft_in22k_in1k', pretrained=True, num_classes=num_classes).to(device)
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    transforms_imagenet_train = timm.data.create_transform(**data_config, is_training=True)
    transforms_imagenet_eval = timm.data.create_transform(**data_config, is_training=False)

    # Transform for train
    train_transform  = v2.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms_imagenet_train
    ])

    # Transform for eval
    val_transform  = v2.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms_imagenet_eval
    ])

    # Training Data
    fer_dataset_train = DataProcessor(csv_file='../Dataset/train_labels.csv', img_dir='../Dataset/train', transform=train_transform)
    train_loader = DataLoader(fer_dataset_train, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    # Validation Data
    val_dataset = DataProcessor(csv_file='../Dataset/val_labels.csv', img_dir='../Dataset/val', transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    # Test Data
    test_dataset = DataProcessor(csv_file='../Dataset/test_labels.csv', img_dir='../Dataset/test', transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    # Compute class weights for loss function
    class_weights = compute_class_weights(r'../Dataset/train_labels.csv', 7)
    fer_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    criterion = torch.nn.CrossEntropyLoss(weight=fer_weights)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.6, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)

    patience = 5
    best_val_acc = 0
    patience_counter = 0
    epoch_counter = 0
    num_epochs = 100

     # Store metrics for each epoch
    epoch_metrics = {
        "epoch": [],
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1_score": [],
        "val_auroc": [],
    }

    # Start training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        metrics["train_accuracy"].reset()

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            metrics["train_accuracy"].update(outputs, labels)

        train_loss = running_loss / len(train_loader)
        train_acc = metrics["train_accuracy"].compute().item()

        # Validation Starts
        model.eval()
        val_running_loss = 0.0
        for metric in metrics.values():
            metric.reset()
        val_targets = []
        val_predictions = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
        
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_targets.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

                for metric_name, metric in metrics.items():
                    if metric_name != "train_accuracy":
                        metric.update(outputs, labels)
                

        val_loss = val_running_loss / len(val_loader)
        val_acc = metrics["val_accuracy"].compute().item()
        val_precision = metrics["val_precision"].compute().item()
        val_recall = metrics["val_recall"].compute().item()
        val_f1 = metrics["val_f1_score"].compute().item()
        val_auroc = metrics["val_auroc"].compute().item()

        val_report = classification_report(
            val_targets,
            val_predictions,
            target_names=['Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Neutral'],
            zero_division=0,
            digits=4
        )

        epoch_metrics["epoch"].append(epoch + 1)
        epoch_metrics["train_loss"].append(train_loss)
        epoch_metrics["train_accuracy"].append(train_acc)
        epoch_metrics["val_loss"].append(val_loss)
        epoch_metrics["val_accuracy"].append(val_acc)
        epoch_metrics["val_precision"].append(val_precision)
        epoch_metrics["val_recall"].append(val_recall)
        epoch_metrics["val_f1_score"].append(val_f1)
        epoch_metrics["val_auroc"].append(val_auroc)

        # Print metrics
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
        print(f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1-Score: {val_f1:.4f} | AUROC: {val_auroc:.4f}")
        print("Validation Classification Report:")
        print(val_report)

        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr}\n")
        
        epoch_counter += 1
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 
            torch.save({
                'model_state_dict': model.state_dict(),
            }, 'eva_best.pth')
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epoch(s).")
        
        if patience_counter > patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break

        metrics_df = pd.DataFrame(epoch_metrics)
        metrics_df.to_csv("metrics_results.csv", index=False)
        print("Training metrics saved to metrics_results.csv")

    checkpoint = torch.load('eva_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final test with the best model
    model.eval()
    test_targets = []
    test_predictions = []
    test_running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing with Best Model"):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_targets.extend(labels.cpu().numpy())
            test_predictions.extend(predicted.cpu().numpy())

    test_loss = test_running_loss / len(test_loader)
    test_acc = accuracy_score(test_targets, test_predictions)

    print(f"Final Test Loss: {test_loss}")
    print(f"Final Test Accuracy: {test_acc}")
    print("Test Classification Report:")
    print(classification_report(test_targets, test_predictions, zero_division=0))

if __name__ == '__main__':
    main()


