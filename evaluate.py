import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
import os
import argparse
import numpy as np

from src.model import ExoplanetDualBranchCNN

CLASSES = {
    0: "Planet Candidate (PC)",
    1: "False Positive (FP)",
    2: "Confirmed Planet (CP)",
    3: "Kepler Candidate (KP)",
    4: "Astrophysical Candidate (APC)",
    5: "False Alarm (FA)"
}

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate ExoplanetDualBranchCNN")
    parser.add_argument("--dataset", default="data/processed/training_dataset.pt",
                        help="Path to evaluation dataset .pt file")
    parser.add_argument("--weights", default="weights/best_model.pth",
                        help="Path to model weights")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    DATASET_PATH = args.dataset
    WEIGHTS_PATH = args.weights
    BATCH_SIZE = args.batch_size

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logging.info("Evaluation started")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    if not os.path.exists(DATASET_PATH):
        logging.error(f"Dataset not found at {DATASET_PATH}")
        return

    X_global, X_local, Y = torch.load(DATASET_PATH, weights_only=True)
    dataset = TensorDataset(X_global, X_local, Y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    logging.info(f"Loaded {len(Y)} samples (classes: {sorted(set(Y.tolist()))})")

    model = ExoplanetDualBranchCNN().to(device)
    if not os.path.exists(WEIGHTS_PATH):
        logging.error(f"Weights not found at {WEIGHTS_PATH}")
        return
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    logging.info(f"Loaded weights from {WEIGHTS_PATH}")

    all_preds, all_labels = [], []
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    with torch.no_grad():
        for batch_global, batch_local, batch_labels in loader:
            batch_global = batch_global.to(device)
            batch_local = batch_local.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_global, batch_local)
            loss = criterion(outputs, batch_labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    logging.info(f"Test Loss: {running_loss / len(loader):.4f}")
    logging.info(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    present_labels = sorted(set(all_labels))
    target_names = [CLASSES[i] for i in present_labels]

    print("\nPer-Class Accuracy:")
    for label in present_labels:
        mask = np.array(all_labels) == label
        class_acc = np.mean(np.array(all_preds)[mask] == label)
        print(f"  {CLASSES[label]} (class {label}): {class_acc * 100:.2f}% ({mask.sum()} samples)")

    print("\nConfusion Matrix (rows=true, cols=predicted):")
    cm = np.zeros((len(present_labels), len(present_labels)), dtype=int)
    label_to_idx = {l: i for i, l in enumerate(present_labels)}
    for t, p in zip(all_labels, all_preds):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t]][label_to_idx[p]] += 1
    header = "     " + " ".join(f"{c:>8d}" for c in present_labels)
    print(header)
    for i, label in enumerate(present_labels):
        row = f"  {label:>2d}  " + " ".join(f"{cm[i][j]:>8d}" for j in range(len(present_labels)))
        print(row)

    logging.info("Evaluation completed")

if __name__ == "__main__":
    evaluate()
