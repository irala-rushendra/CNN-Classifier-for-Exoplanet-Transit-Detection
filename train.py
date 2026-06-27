import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from src.model import ExoplanetDualBranchCNN

def main():
    parser = argparse.ArgumentParser(description="Train ExoplanetDualBranchCNN")
    parser.add_argument("--dataset", default="data/processed/training_dataset.pt",
                        help="Path to training dataset .pt file")
    parser.add_argument("--weights", default="weights/best_model.pth",
                        help="Path to save best model weights")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    DATASET_PATH = args.dataset
    WEIGHTS_SAVE_PATH = args.weights
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr

    logging.info("Pipeline has started...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Found : {device}")
    logging.info(f"Training with {device}")

    if not os.path.exists(DATASET_PATH):
        print(f'Required file does not exist or is corrupted')
        logging.critical("Path does not exist")
        return

    logging.info(f"Loading from {DATASET_PATH}")
    X_global, X_local, Y = torch.load(DATASET_PATH)
    
    dataset = TensorDataset(X_global, X_local, Y)
    total_samples = len(dataset)
    
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    logging.info(f"Dataset split: {train_size}[Train], {val_size}[Validation]")

    model = ExoplanetDualBranchCNN().to(device)

    class_counts = torch.bincount(Y)
    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    logging.info(f"Class weights: {class_weights.tolist()}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        
        model.train() 
        running_train_loss = 0.0
        correct_train = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False)
        for batch_global, batch_local, batch_labels in train_pbar:
            
            batch_global = batch_global.to(device)
            batch_local = batch_local.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            
            predictions = model(batch_global, batch_local)
            
            loss = criterion(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            _, predicted_classes = torch.max(predictions, 1)
            correct_train += (predicted_classes == batch_labels).sum().item()
            
            train_pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        train_acc = correct_train / train_size
        avg_train_loss = running_train_loss / len(train_loader)

        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        
        with torch.no_grad():
            for batch_global, batch_local, batch_labels in val_loader:
                
                batch_global = batch_global.to(device)
                batch_local = batch_local.to(device)
                batch_labels = batch_labels.to(device)
                
                predictions = model(batch_global, batch_local)
                loss = criterion(predictions, batch_labels)
                
                running_val_loss += loss.item()
                _, predicted_classes = torch.max(predictions, 1)
                correct_val += (predicted_classes == batch_labels).sum().item()

        val_acc = correct_val / val_size
        avg_val_loss = running_val_loss / len(val_loader)

        logging.info(f"Epoch {epoch+1:02d} | Train Acc: {train_acc*100:5.2f}% | Val Acc: {val_acc*100:5.2f}% | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(WEIGHTS_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
            print(f"Model weights saved to '{WEIGHTS_SAVE_PATH}'. Check training_logs for details")

    logging.info("Training is completed")
    print("\nTraining has completed")

if __name__ == "__main__":
    main()
