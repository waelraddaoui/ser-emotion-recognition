import torch
import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from app.ml import EmotionCNN

model_path = "models/best_model_gpu.pth"
model = EmotionCNN(n_classes=6)
state = torch.load(model_path, map_location="cpu")
model.load_state_dict(state)
model.eval()
print("Success loading model with 6 classes!")
