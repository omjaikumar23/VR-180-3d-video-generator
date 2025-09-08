import torch
import numpy as np

class DepthEstimator:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
        self.model.eval()
        self.transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    def estimate_depth(self, pil_image):
        img_np = np.array(pil_image)
        input_image = self.transform(img_np).to(self.device)
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(input_image)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=pil_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        depth_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-6)
        return depth_norm

