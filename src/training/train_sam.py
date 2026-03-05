import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from segment_anything import sam_model_registry
from src.training.dataset import NYUPromptDataset


def train_sam_model(data_dir, epochs, out="models/checkpoints/trained/sam_finetuned.pth"):
    print("[*] Fine-tuning SAM locally (Encoder frozen)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DataLoader(NYUPromptDataset(data_dir, "sam"),
                        batch_size=2, shuffle=True)
    model = sam_model_registry["vit_h"](
        checkpoint="models/checkpoints/sam_vit_h_4b8939.pth").to(device)

    # Freeze image encoder for local fine-tuning constraints
    for p in model.image_encoder.parameters():
        p.requires_grad = False
    for p in model.prompt_encoder.parameters():
        p.requires_grad = False

    opt = optim.Adam(model.mask_decoder.parameters(), lr=1e-5)
    bce = nn.BCELoss()

    for epoch in range(epochs):
        for b in loader:
            opt.zero_grad()
            with torch.no_grad():
                img_emb = model.image_encoder(b['image'].to(device))
                s_emb, d_emb = model.prompt_encoder(
                    points=None, boxes=b['boxes'].to(device), masks=None)
            masks, _ = model.mask_decoder(image_embeddings=img_emb, image_pe=model.prompt_encoder.get_dense_pe(
            ), sparse_prompt_embeddings=s_emb, dense_prompt_embeddings=d_emb, multimask_output=False)
            loss = bce(torch.sigmoid(masks), b['masks'].to(device))
            loss.backward()
            opt.step()
        print(f"SAM Epoch {epoch+1} complete.")
    torch.save(model.mask_decoder.state_dict(), out)
