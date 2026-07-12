import torch
import sys

# Append path to Restormer directory (Kaggle-specific path or modify for local)
sys.path.append("/kaggle/working/Restormer")

from basicsr.models.archs.restormer_arch import Restormer

def load_restormer_model(ckpt_path, device):
    model = Restormer()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model
