import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import argparse
import warnings
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from src.model import ExoplanetDualBranchCNN

warnings.filterwarnings("ignore", category=UserWarning)

CLASSES = {
    0: "Planet Candidate (PC)",
    1: "False Positive (FP)",
    2: "Confirmed Planet (CP)",
    3: "Kepler Candidate (KP)",
    4: "Astrophysical Candidate (APC)",
    5: "False Alarm (FA)"
}

def load_brain():
    logging.info("Session has started")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ExoplanetDualBranchCNN().to(device)
    
    if os.path.exists(WEIGHTS_PATH):
        try:
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
            model.eval()
            logging.info("Classifier loaded successfully from weights checkpoint.")
            return model, device
        except Exception as e:
            print(f"Could not load weights ({e}). Running in evaluation-ready mode.")
            logging.info(f"Could not load weights ({e}). Running in evaluation-ready mode.")
            return model, device
    else:
        print(f"'{WEIGHTS_PATH}' not found. Initializing with random weights for demo.")
        logging.info(f"'{WEIGHTS_PATH}' not found. Initializing with random weights for demo.")
        model.eval()
        return model, device

def generate_synthetic_transit(global_len=2001, local_len=201, noise_level=0.002, dip_depth=0.015):
    logging.info("Creating Synthetic Data")
    
    time = np.linspace(-0.5, 0.5, global_len)
    
    flux = np.ones_like(time)
    transit_mask = (time > -0.04) & (time < 0.04)
    flux[transit_mask] = 1.0 - dip_depth
    
    noise = np.random.normal(0, noise_level, size=global_len)
    flux_noisy = flux + noise
    
    flux_norm = (flux_noisy - np.mean(flux_noisy)) / np.std(flux_noisy)
    
    phase_grid_global = np.linspace(np.min(time), np.max(time), global_len)
    interpolator = interp1d(time, flux_norm, kind='linear', fill_value="extrapolate")
    global_view = interpolator(phase_grid_global)
    
    center_idx = global_len // 2
    half_window = local_len // 2
    local_view = global_view[center_idx - half_window : center_idx + half_window + 1]
    
    return time, flux_norm, global_view, local_view, phase_grid_global

def run_synthetic_pipeline(model, device):
    time, flux_norm, global_view, local_view, phase_grid_global = generate_synthetic_transit()
    
    tensor_global = torch.tensor(global_view, dtype=torch.float32).view(1, 1, -1).to(device)
    tensor_local = torch.tensor(local_view, dtype=torch.float32).view(1, 1, -1).to(device)
    
    logging.info("Analyzing transit geometry...")
    with torch.no_grad():
        raw_outputs = model(tensor_global, tensor_local)
        probabilities = F.softmax(raw_outputs, dim=1).squeeze().cpu().numpy()
        
    pred_class_idx = np.argmax(probabilities)
    pred_label = CLASSES[pred_class_idx]
    confidence = probabilities[pred_class_idx] * 100
    
    plot_results(time, flux_norm, global_view, local_view, phase_grid_global, pred_label, confidence)

def plot_results(raw_time, raw_flux, global_view, local_view, phase_grid, label, confidence):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    is_planet = "Planet Candidate" in label or "Confirmed Planet" in label or "Kepler Candidate" in label
    title_color = '#1b5e20' if is_planet else '#b71c1c'
    
    fig.suptitle(f"Synthetic Engine Diagnostic View\nAI Pipeline Decision: {label} ({confidence:.2f}%)", 
                 fontsize=14, fontweight='bold', color=title_color)
    
    ax1.scatter(raw_time, raw_flux, color='gray', s=1.5, alpha=0.3, label="Folded Observations")
    ax1.plot(phase_grid, global_view, color='#1565c0', linewidth=1.5, label="1D Global View Vector")
    ax1.set_title("Global View (Full Phase Space)")
    ax1.set_xlabel("Phase")
    ax1.set_ylabel("Normalized Flux (Z-Score)")
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    local_phases = np.linspace(-0.05, 0.05, len(local_view))
    ax2.plot(local_phases, local_view, color='#e65100', linewidth=2.5, label="Deep Zoom Input")
    ax2.set_title("Local View (Transit Egress/Ingress Geometry)")
    ax2.set_xlabel("Relative Phase Window")
    ax2.set_ylabel("Normalized Flux (Z-Score)")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    #plt.savefig("prediction_report.png", dpi=150)
    #logging.info(f"Plot saved to prediction_report.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with ExoplanetDualBranchCNN")
    parser.add_argument("--weights", default="weights/best_model.pth",
                        help="Path to model weights")
    args = parser.parse_args()

    WEIGHTS_PATH = args.weights
    ai_model, compute_device = load_brain()
    run_synthetic_pipeline(ai_model, compute_device)
