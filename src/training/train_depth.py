import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoModelForDepthEstimation
from src.training.dataset import NYUPromptDataset

# Implements Scale-Invariant Logarithmic Loss (L_SI) [cite: 94-96]


class ScaleInvariantLogLoss(nn.Module):
    def __init__(self, lam=0.5, alpha=10.0):
        super().__init__()
        self.lam, self.alpha = lam, alpha

    def forward(self, p, t):
        p, t = torch.clamp(p, min=1e-4), torch.clamp(t, min=1e-4)
        mask = t > 0
        d = torch.log(p[mask]) - torch.log(t[mask])
        N = d.numel()
        if N == 0:
            return torch.tensor(0.0, requires_grad=True).to(p.device)
        return self.alpha * torch.sqrt(torch.mean(d**2) - (self.lam*(torch.sum(d)**2))/(N**2))


def train_depth_model(data_dir, epochs, out="models/checkpoints/trained/depth_metric.pth"):
    print("[*] Fine-tuning Depth Anything locally for absolute scale...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(NYUPromptDataset(data_dir, "depth"),
                        batch_size=4, shuffle=True)
    model = AutoModelForDepthEstimation.from_pretrained(
        "LiheYoung/depth-anything-small-hf").to(device)
    model.train()
    opt = optim.AdamW(model.parameters(), lr=1e-5)
    crit = ScaleInvariantLogLoss()

    for epoch in range(epochs):
        for b in loader:
            opt.zero_grad()
            p = model(pixel_values=b['pixel_values'].to(
                device)).predicted_depth
            p = torch.nn.functional.interpolate(p.unsqueeze(
                1), size=b['depth_labels'].shape[-2:], mode="bicubic").squeeze()
            loss = crit(p, b['depth_labels'].to(device))
            loss.backward()
            opt.step()
        print(f"Depth Epoch {epoch+1} complete.")
    torch.save(model.state_dict(), out)
