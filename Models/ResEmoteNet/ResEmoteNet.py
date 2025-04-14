import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
from data_processor import DataProcessor
from model_arhitecture import ResEmoteNet
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

def compute_class_weights(csv_file, num_classes):

    # Load the CSV file and extract labels
    df = pd.read_csv(csv_file, header=None, names=["image", "label"])
    labels = df["label"].tolist()
    
    # Count occurrences of each class
    class_counts = Counter(labels)
    total_samples = sum(class_counts.values())
    
    # Compute class weights: Inverse class frequency
    class_weights = [1 - (class_counts.get(i, 0) / total_samples) for i in range(num_classes)]
    
    # print(f"Class Counts: {dict(class_counts)}")
    print(f"Class Weights: {class_weights}")
    return class_weights

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Transform the dataset
    train = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Transform the dataset
    eval = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Training Data
    fer_dataset_train = DataProcessor(csv_file='fer/train_labels.csv', img_dir='fer/train', transform=train)
    data_train_loader = DataLoader(fer_dataset_train, batch_size=8, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    # Validation Data
    val_dataset = DataProcessor(csv_file='fer/val_labels.csv', img_dir='fer/val', transform=eval)
    data_val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)

    # Testing Data
    fer_dataset_test = DataProcessor(csv_file='fer/test_labels.csv', img_dir='fer/test', transform=eval)
    data_test_loader = DataLoader(fer_dataset_test, batch_size=8, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    class_weights = compute_class_weights(r'fer/train_labels.csv', 7)
    fer_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    
    # Load the model
    model = ResEmoteNet()

    model.to(device)

    # Hyperparameters
    criterion = torch.nn.CrossEntropyLoss(weight=fer_weights)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.6, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=4, verbose=True)

    patience = 6
    best_val_acc = 0
    patience_counter = 0
    epoch_counter = 0

    num_epochs = 150

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Start training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total, correct = 0, 0

        for data in tqdm(data_train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(data_train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation Starts
        model.eval()
        val_running_loss = 0.0
        val_targets = []
        val_predictions = []
        with torch.no_grad():
            for data in tqdm(data_val_loader, desc=f"Validation Epoch {epoch+1}/{num_epochs}"):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_targets.extend(labels.cpu().numpy())
                val_predictions.extend(predicted.cpu().numpy())

        val_loss = val_running_loss / len(data_val_loader)
        val_losses.append(val_loss)
        val_acc = accuracy_score(val_targets, val_predictions)
        val_accuracies.append(val_acc)

        # Print metrics
        print(f"\nEpoch {epoch+1}: ")
        print(f"\nTrain Loss: {train_loss}")
        print(f"\nTrain Accuracy: {train_acc}")
        print(f"\nValidation Loss: {val_loss}")
        print(f"\nValidation Accuracy: {val_acc}")
        print("\nValidation Classification Report:")
        print(classification_report(val_targets, val_predictions, zero_division=0))

        # Adjust learning rate based on validation accuracy
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nLearning Rate: {current_lr}")
        
        epoch_counter += 1
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0 
            torch.save({
                'model_state_dict': model.state_dict(),
            }, 'fer_model.pth')
        else:
            patience_counter += 1
            print(f"No improvement in validation accuracy for {patience_counter} epoch(s).")
        
        if patience_counter > patience:
            print("Stopping early due to lack of improvement in validation accuracy.")
            break
    df = pd.DataFrame({
        'Epoch': range(1, epoch_counter+1),
        'Train Loss': train_losses,
        'Validation Loss': val_losses,
        'Train Accuracy': train_accuracies,
        'Validation Accuracy': val_accuracies,
    })
    df.to_csv('result_loss.csv', index=False)

    # Load the best model before testing
    checkpoint = torch.load('fer_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final test with the best model
    model.eval()
    test_targets = []
    test_predictions = []
    test_running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(data_test_loader, desc="Testing with Best Model"):
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_targets.extend(labels.cpu().numpy())
            test_predictions.extend(predicted.cpu().numpy())

    test_loss = test_running_loss / len(data_test_loader)
    test_acc = accuracy_score(test_targets, test_predictions)

    print(f"Final Test Loss: {test_loss}")
    print(f"Final Test Accuracy: {test_acc}")
    print("Test Classification Report:")
    print(classification_report(test_targets, test_predictions, zero_division=0))

if __name__ == '__main__':
    main()