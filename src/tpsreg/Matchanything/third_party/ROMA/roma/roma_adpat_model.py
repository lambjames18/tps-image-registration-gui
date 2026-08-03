import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from .models import roma_outdoor

class ROMA_Model(nn.Module):
    def __init__(self, MAX_MATCHES=5000, SAMPLE_THRESH=0.8, MATCH_THRESH=0.3) -> None:
        super().__init__()
        self.model = roma_outdoor(device=torch.device("cpu"))
        self.MAX_MATCHES = MAX_MATCHES
        self.MATCH_THRESH = MATCH_THRESH
        self.model.sample_thresh = SAMPLE_THRESH # Inner matcher
    
    def forward(self, data):
        img0, img1 = data['image0_rgb'][0], data['image1_rgb'][0] # unbatch, 3 * H * W

        H_A, W_A = img0.shape[-2:]
        H_B, W_B = img1.shape[-2:]
        warp, certainty = self.model.match(img0, img1) # 3 * H * W
        # Sample matches for estimation
        matches, certainty = self.model.sample(warp, certainty, num=self.MAX_MATCHES)

        mask = certainty > self.MATCH_THRESH
        kpts0, kpts1 = self.model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
        kpts0, kpts1, certainty = map(lambda x:x[mask], [kpts0, kpts1, certainty])
        data.update({'m_bids': torch.zeros_like(kpts0[:, 0]), "mkpts0_f": kpts0, "mkpts1_f": kpts1, "mconf": certainty})
        return data