CLASS_LABELS = {
    0: "Planet Candidate (PC)",
    1: "False Positive (FP)",
    2: "Confirmed Planet (CP)",
    3: "Kepler Candidate (KP)",
    4: "Astrophysical Candidate (APC)",
    5: "False Alarm (FA)",
}

CLASS_NAMES_SHORT = {
    0: "PC", 1: "FP", 2: "CP", 3: "KP", 4: "APC", 5: "FA",
}

def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
