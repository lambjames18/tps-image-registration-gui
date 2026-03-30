from pathlib import Path
import numpy as np
from skimage import io, transform
import matplotlib.pyplot as plt
import tps

# path = "/Users/jameslamb/Documents/research/data/match_anything_datasets/CoNi-AM67_OM-SEM_Multiscale/eval_indexs/eval_CoNi-AM67_OM-SEM_Multiscale_0.npz"
# path = "/Users/jameslamb/Documents/research/data/match_anything_datasets/CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning/eval_indexs/eval_CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning_0.npz"
# path = "/Users/jameslamb/Documents/research/data/match_anything_datasets/CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning/eval_indexs/eval_CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning_1.npz"
# path = "/Users/jameslamb/Documents/research/data/match_anything_datasets/CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning/eval_indexs/eval_CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning_2.npz"
path = "/Users/jameslamb/Documents/research/data/match_anything_datasets/CoNi-AM90_OM-SEM_Multiscale/eval_indexs/eval_CoNi-AM90_OM-SEM_Multiscale_0.npz"
# path = "/Users/jameslamb/Documents/research/data/match_anything_datasets/CoNi-AM90_SEM-DIC_EBSD_SlipPartitioning/eval_indexs/eval_CoNi-AM90_SEM-DIC_EBSD_SlipPartitioning_0.npz"
# path = "/Users/jameslamb/Documents/research/data/match_anything_datasets/CoNi-AM90_SEM-EBSD_SameSlice/eval_indexs/eval_CoNi-AM90_SEM-EBSD_SameSlice_0.npz"
# path = "/Users/jameslamb/Documents/research/data/match_anything_datasets/Ta-AM-Spalled_SEM-BSE_EBSD_SameSliceSerialSectioning/eval_indexs/eval_Ta-AM-Spalled_SEM-BSE_EBSD_SameSliceSerialSectioning_0.npz"
# path = "/Users/jameslamb/Documents/research/data/match_anything_datasets/Ta-AM-Spalled_SEM-BSE_EBSD_SameSliceSerialSectioning/eval_indexs/eval_Ta-AM-Spalled_SEM-BSE_EBSD_SameSliceSerialSectioning_1.npz"
# path = "/Users/jameslamb/Documents/research/data/match_anything_datasets/Ta-AM-Spalled_SEM-BSE_EBSD_SameSliceSerialSectioning/eval_indexs/eval_Ta-AM-Spalled_SEM-BSE_EBSD_SameSliceSerialSectioning_2.npz"

data = np.load(path, allow_pickle=True)
# print(data)
for item in data:
    print(item)
    if isinstance(data[item], (np.ndarray, list)):
        for subitem in data[item]:
            if isinstance(subitem, (np.ndarray, list)):
                for subsubitem in subitem:
                    print("\t\t", subsubitem)
                print("-")
            elif isinstance(subitem, dict):
                for key in subitem:
                    print("\t\t", key, ":", subitem[key], type(subitem[key]))
                print("-")
            else:
                print("\t", subitem)
                print("-")
    elif isinstance(data[item], dict):
        for key in data[item]:
            print("\t", key, ":", data[item][key])
        print("-")
    else:
        print("\t", data[item])
        print("-")

pair_infos = data["pair_infos"]
gt_2D_matches = data["gt_2D_matches"]
image_paths = data["image_paths"]
folder = Path(path).parent.parent

for i in range(len(pair_infos)):
    im_target_idx = pair_infos[i][0][0]
    im_source_idx = pair_infos[i][0][1]
    im_source = io.imread(folder / image_paths[im_source_idx])
    im_target = io.imread(folder / image_paths[im_target_idx])
    matches = gt_2D_matches[i]
    target_pts = matches[:, :2]
    source_pts = matches[:, 2:]

    tform = tps.ThinPlateSplineTransform()
    tform.estimate(source_pts, target_pts, size=im_target.shape[:2])
    warped = transform.warp(im_source, tform, output_shape=im_target.shape[:2])

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(im_source, cmap="gray")
    ax[1].imshow(im_target, cmap="gray")
    ax[2].imshow(warped, cmap="gray")
    ax[0].scatter(source_pts[:, 0], source_pts[:, 1], c="r", s=5)
    ax[1].scatter(target_pts[:, 0], target_pts[:, 1], c="r", s=5)
    ax[0].set_title(f"Source image - Index {im_source_idx}")
    ax[1].set_title(f"Target image - Index {im_target_idx}")
    ax[2].set_title(f"Source warped to target")
    plt.show()
