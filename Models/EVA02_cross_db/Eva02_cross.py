import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC
import torch.nn.functional as F
import torch.nn as nn
import timm
import cv2
import os
from torchvision.transforms import v2
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt

# FocalCELoss
class FocalCELoss(nn.Module):  
    def __init__(self, α=0.25, γ=2.0, smooth=0.1):  
        super().__init__()  
        self.α = α  
        self.γ = γ  
        self.smooth = smooth  

    def forward(self, inputs, targets):  
        ce_loss = F.cross_entropy(
            inputs, targets,
            label_smoothing=self.smooth
        )
        pt = torch.exp(-ce_loss)
        return self.α * (1 - pt) ** self.γ * ce_loss

# Dataset definition
class DataProcessor(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = cv2.imread(img_name)
        label = self.labels.iloc[idx, 1]

        if self.transform:
            image = self.transform(image)

        return image, label

def compute_class_weights(csv_file, num_classes):
    df = pd.read_csv(csv_file, header=None, names=["image", "label"])
    labels = df["label"].tolist()
    
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    
    class_weights = [1 - (class_counts.get(i, 0) / total_samples) for i in range(num_classes)]
    print(f"Class Weights: {class_weights}")
    return class_weights

def get_weighted_sampler(dataset):
    labels = dataset.labels.iloc[:, 1].values
    class_counts = Counter(labels)
    num_samples = len(labels)
    class_weights = [num_samples / class_counts[i] for i in range(7)]
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples, replacement=True)
    return sampler

def run_lr_finder(model, optimizer, criterion, train_loader, device):

    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_loader, start_lr=0, end_lr=1, num_iter=100)
    rez = lr_finder.plot(log_lr=True)
    plt.savefig("lr_finder.png")
    lr_finder.reset()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7
    num_epochs = 30
    accumulation_steps = 4

    # Initialize tracked metrics
    metrics = {
        "train_accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "val_accuracy": Accuracy(task="multiclass", num_classes=num_classes).to(device),
        "val_precision": Precision(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "val_recall": Recall(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "val_f1_score": F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device),
        "val_auroc": AUROC(task="multiclass", num_classes=num_classes).to(device),
    }
    torch.set_float32_matmul_precision('high')
    
    model = timm.create_model('eva02_base_patch14_224.mim_in22k', pretrained=True, num_classes=num_classes).to(device)
    #model = torch.compile(model)
    
    checkpoint = torch.load(r"checkpoints/eva02_fth_ftf_best.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    train_transform = v2.Compose([
    v2.RandomHorizontalFlip(p=0.3),
    v2.RandomRotation(15),
    v2.Grayscale(num_output_channels=3),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224)),
    v2.Grayscale(num_output_channels=3),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load training data and set up sampler for balanced batches
    fer_dataset_train = DataProcessor(csv_file=r"train_labels.csv", img_dir=train_data", transform=train_transform)
    sampler = get_weighted_sampler(fer_dataset_train)
    train_loader = DataLoader(fer_dataset_train, batch_size=32, sampler=sampler, num_workers=8, persistent_workers=True, pin_memory=True)
    
    # Load validation data
    val_dataset = DataProcessor(csv_file=r"val_labels.csv", img_dir=r"val_data",transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    criterion = FocalCELoss(α=0.25, γ=2.0, smooth=0.1)
    
    # run_lr_finder(model, optimizer, criterion, train_loader, device)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
    scaler = torch.amp.GradScaler("cuda")

    patience = 4
    best_val_acc = 0
    patience_counter = 0
    epoch_counter = 0

    # Optionally resume from a checkpoint
    checkpoint_path = 'eva02_base_best.pth'
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_acc = checkpoint['best_val_acc']
        print(f"Resumed training from epoch {start_epoch} with best_val_acc {best_val_acc:.4f}")

    # Dictionary to store epoch metrics
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

    # Training loop
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        metrics["train_accuracy"].reset()
        optimizer.zero_grad()

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            # Step optimizer every accumulation_steps batches or at the end of an epoch
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            metrics["train_accuracy"].update(outputs, labels)

        train_loss = running_loss / len(train_loader)
        train_acc = metrics["train_accuracy"].compute().item()

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        for metric in metrics.values():
            metric.reset()
        val_targets = []
        val_predictions = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = inputs.to(device), labels.to(device)
                with torch.amp.autocast("cuda"):
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

        scheduler.step(val_loss)
        current_lr = scheduler.get_last_lr()[0]
        print(f"Learning Rate: {current_lr}\n")
        
        epoch_metrics["epoch"].append(epoch + 1)
        epoch_metrics["train_loss"].append(train_loss)
        epoch_metrics["train_accuracy"].append(train_acc)
        epoch_metrics["val_loss"].append(val_loss)
        epoch_metrics["val_accuracy"].append(val_acc)
        epoch_metrics["val_precision"].append(val_precision)
        epoch_metrics["val_recall"].append(val_recall)
        epoch_metrics["val_f1_score"].append(val_f1)
        epoch_metrics["val_auroc"].append(val_auroc)

        # Print current epoch metrics and classification report
        print(f"\nEpoch {epoch+1}:")
        print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")
        print(f"Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1-Score: {val_f1:.4f} | AUROC: {val_auroc:.4f}")
        print("Validation Classification Report:")
        print(val_report)
        
        epoch_counter += 1
        
        # Save checkpoint if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
            }, checkpoint_path)
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epoch(s).")
        
        if patience_counter > patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break

        # Save metrics to CSV after each epoch
        metrics_df = pd.DataFrame(epoch_metrics)
        metrics_df.to_csv("metrics_results.csv", index=False)
        print("Training metrics saved to metrics_results.csv")

    # # Test Data
    # test_dataset = DataProcessor(csv_file=r"test_labels.csv", img_dir=r"test", transform=val_transform)
    # test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    # # Final test with the best model
    # model.eval()
    # test_targets = []
    # test_predictions = []
    # test_running_loss = 0.0
    # with torch.no_grad():
    #     for data in tqdm(test_loader, desc="Testing with Best Model"):
    #         inputs, labels = data[0].to(device), data[1].to(device)
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         test_running_loss += loss.item()
    #         _, predicted = torch.max(outputs.data, 1)
    #         test_targets.extend(labels.cpu().numpy())
    #         test_predictions.extend(predicted.cpu().numpy())

    # test_loss = test_running_loss / len(test_loader)
    # test_acc = accuracy_score(test_targets, test_predictions)

    # print(f"Final Test Loss: {test_loss}")
    # print(f"Final Test Accuracy: {test_acc}")
    # print("Test Classification Report:")
    # print(classification_report(test_targets, test_predictions, zero_division=0))

if __name__ == '__main__':
    main()
