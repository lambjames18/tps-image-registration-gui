from PIL import Image
import torch
import cv2
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from roma.roma_adpat_model import ROMA_Model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--im_A_path", default="assets/sacre_coeur_A.jpg", type=str)
    parser.add_argument("--im_B_path", default="assets/sacre_coeur_B.jpg", type=str)

    args, _ = parser.parse_known_args()
    im1_path = args.im_A_path
    im2_path = args.im_B_path
    # Create model
    model = ROMA_Model({"n_sample": 5000})


    W_A, H_A = Image.open(im1_path).size
    W_B, H_B = Image.open(im2_path).size

    # Match
    match_results = model({"image0_path": im1_path, "image1_path": im2_path})
    kpts1, kpts2 = match_results['mkpts0_f'], match_results['mkpts1_f']
    # Sample matches for estimation
    F, mask = cv2.findFundamentalMat(
        kpts1.cpu().numpy(), kpts2.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
    )