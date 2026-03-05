import os, torch, numpy as np
from PIL import Image
from torch.utils.data import Dataset

class NYUPromptDataset(Dataset):
    def __init__(self, data_dir, task="depth"):
        self.data_dir = data_dir
        self.task = task
        self.images = sorted([f for f in os.listdir(os.path.join(data_dir, "images")) if f.endswith('.jpg')])

    def __len__(self): return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_tensor = torch.tensor(np.array(Image.open(f"{self.data_dir}/images/{img_name}").convert("RGB"))).permute(2,0,1).float()/255.0
        
        if self.task == "depth":
            depth = np.load(f"{self.data_dir}/depth_maps/{img_name.replace('.jpg', '.npy')}")
            return {"pixel_values": img_tensor, "depth_labels": torch.tensor(depth).float()}
        elif self.task == "sam":
            mask = np.array(Image.open(f"{self.data_dir}/masks/{img_name.replace('.jpg', '.png')}").convert("L"))
            return {"image": img_tensor, "masks": torch.tensor((mask>128).astype(np.float32)).unsqueeze(0), "boxes": torch.tensor([0,0,100,100])}