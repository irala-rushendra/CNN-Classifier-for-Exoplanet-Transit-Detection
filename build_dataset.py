import numpy as np
import torch
import logging
import os
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

GLOBAL_LEN = 2001
LOCAL_LEN = 201

PHASE = np.linspace(-0.5, 0.5, GLOBAL_LEN)
LOCAL_PHASE = np.linspace(-0.5, 0.5, LOCAL_LEN)

SAMPLES_PER_CLASS = {
    0: 2000,
    1: 1000,
    2: 800,
    3: 600,
    4: 500,
    5: 200,
}

CLASS_LABELS = ["PC", "FP", "CP", "KP", "APC", "FA"]


def make_transit(dip_depth=0.01, ingress_width=0.01):
    flux = np.ones(GLOBAL_LEN)
    center = GLOBAL_LEN // 2
    half_dur = int(GLOBAL_LEN * 0.03)
    ingress = int(GLOBAL_LEN * ingress_width)
    for i in range(GLOBAL_LEN):
        dist = abs(i - center)
        if dist < half_dur - ingress:
            flux[i] = 1.0 - dip_depth
        elif dist < half_dur:
            t = (dist - (half_dur - ingress)) / ingress
            flux[i] = 1.0 - dip_depth * (1 - t)
    return flux


def make_v_shaped_dip(dip_depth=0.02):
    flux = np.ones(GLOBAL_LEN)
    center = GLOBAL_LEN // 2
    half_dur = int(GLOBAL_LEN * 0.04)
    for i in range(GLOBAL_LEN):
        dist = abs(i - center)
        if dist < half_dur:
            flux[i] = 1.0 - dip_depth * (1 - dist / half_dur)
    return flux


def generate_sample(class_id):
    rng = np.random.RandomState()

    if class_id == 0:
        flux = make_transit(dip_depth=rng.uniform(0.005, 0.02), ingress_width=rng.uniform(0.005, 0.015))
        noise = rng.normal(0, rng.uniform(0.0005, 0.0015), GLOBAL_LEN)

    elif class_id == 1:
        flux = make_v_shaped_dip(dip_depth=rng.uniform(0.01, 0.04))
        noise = rng.normal(0, rng.uniform(0.001, 0.003), GLOBAL_LEN)

    elif class_id == 2:
        flux = make_transit(dip_depth=rng.uniform(0.01, 0.03), ingress_width=rng.uniform(0.008, 0.02))
        noise = rng.normal(0, rng.uniform(0.0003, 0.0008), GLOBAL_LEN)

    elif class_id == 3:
        flux = make_transit(dip_depth=rng.uniform(0.005, 0.015), ingress_width=rng.uniform(0.008, 0.02))
        noise = rng.normal(0, rng.uniform(0.001, 0.002), GLOBAL_LEN)

    elif class_id == 4:
        flux = make_transit(dip_depth=rng.uniform(0.002, 0.008), ingress_width=rng.uniform(0.005, 0.015))
        noise = rng.normal(0, rng.uniform(0.002, 0.004), GLOBAL_LEN)

    else:
        flux = np.ones(GLOBAL_LEN)
        noise = rng.normal(0, rng.uniform(0.003, 0.006), GLOBAL_LEN)

    noisy = flux + noise
    noisy = (noisy - np.mean(noisy)) / np.std(noisy)

    global_view = noisy.astype(np.float32)
    center, half = GLOBAL_LEN // 2, LOCAL_LEN // 2
    local_view = global_view[center - half : center + half + 1]

    return (
        torch.tensor(global_view).view(1, 1, GLOBAL_LEN),
        torch.tensor(local_view).view(1, 1, LOCAL_LEN),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic training dataset")
    parser.add_argument("--save-dir", default="data/processed",
                        help="Directory to save dataset")
    parser.add_argument("--samples-per-class", type=int, nargs=6,
                        default=[2000, 1000, 800, 600, 500, 200],
                        help="Samples per class (PC FP CP KP APC FA)")
    args = parser.parse_args()

    SAVE_DIR = args.save_dir
    SAVE_PATH = os.path.join(SAVE_DIR, "training_dataset.pt")
    os.makedirs(SAVE_DIR, exist_ok=True)

    counts = args.samples_per_class

    X_global, X_local, Y = [], [], []
    for class_id, n in enumerate(counts):
        logging.info(f"Generating {n} samples for class {class_id} ({CLASS_LABELS[class_id]})")
        for _ in range(n):
            g, l = generate_sample(class_id)
            X_global.append(g)
            X_local.append(l)
            Y.append(class_id)

    X_global = torch.cat(X_global, dim=0)
    X_local = torch.cat(X_local, dim=0)
    Y = torch.tensor(Y, dtype=torch.long)

    logging.info(f"Dataset: {len(Y)} samples")
    logging.info(f"  X_global: {X_global.shape}")
    logging.info(f"  X_local:  {X_local.shape}")
    logging.info(f"  Y:        {Y.shape}")

    for cls in range(6):
        logging.info(f"  Class {cls} ({CLASS_LABELS[cls]}): {(Y == cls).sum().item()} samples")

    torch.save((X_global, X_local, Y), SAVE_PATH)
    logging.info(f"Saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
