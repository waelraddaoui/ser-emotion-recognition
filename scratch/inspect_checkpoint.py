import torch

model_path = "models/best_model_gpu.pth"
state = torch.load(model_path, map_location="cpu")
for k, v in state.items():
    print(k, v.shape)
