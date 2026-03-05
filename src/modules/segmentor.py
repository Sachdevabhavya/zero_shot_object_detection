import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

class ObjectSegmentor:
    def __init__(self, checkpoint_path, model_type="vit_h"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path).to(device)
        self.predictor = SamPredictor(sam)

    def segment(self, image_rgb, bbox):
        self.predictor.set_image(image_rgb)
        masks, _, _ = self.predictor.predict(box=bbox, multimask_output=False)
        return masks[0].astype(np.uint8)