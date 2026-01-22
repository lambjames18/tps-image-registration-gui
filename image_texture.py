import numpy as np
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from skimage import io


def get_energy_maps(im: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Compute the Law's texture energy maps for a given grayscale image.

    Parameters:
    im (numpy.ndarray): Grayscale input image.

    Returns:
    maps (numpy.ndarray): A stack of texture energy maps.
    """
    # zero-mean unit variance normalization
    im = im - np.mean(im)
    im = im / np.std(im)

    # Define Law's filters
    L5 = np.array([1, 4, 6, 4, 1])
    E5 = np.array([-1, -2, 0, 2, 1])
    S5 = np.array([-1, 0, 2, 0, -1])
    R5 = np.array([1, -4, 6, -4, 1])

    # Create 2D filters using outer products
    L5E5 = np.outer(L5, E5)
    E5L5 = np.outer(E5, L5)
    L5R5 = np.outer(L5, R5)
    R5L5 = np.outer(R5, L5)
    E5S5 = np.outer(E5, S5)
    S5E5 = np.outer(S5, E5)
    S5S5 = np.outer(S5, S5)
    R5R5 = np.outer(R5, R5)
    L5S5 = np.outer(L5, S5)
    S5L5 = np.outer(S5, L5)
    E5E5 = np.outer(E5, E5)
    E5R5 = np.outer(E5, R5)
    R5E5 = np.outer(R5, E5)
    S5R5 = np.outer(S5, R5)
    R5S5 = np.outer(R5, S5)

    # Convolve image with filters to get energy maps
    L5E5_E5L5_map = (
        convolve(im, L5E5, mode="reflect") + convolve(im, E5L5, mode="reflect")
    ) / 2.0
    L5R5_R5L5_map = (
        convolve(im, L5R5, mode="reflect") + convolve(im, R5L5, mode="reflect")
    ) / 2.0
    E5S5_S5E5_map = (
        convolve(im, E5S5, mode="reflect") + convolve(im, S5E5, mode="reflect")
    ) / 2.0
    S5S5_map = convolve(im, S5S5, mode="reflect")
    R5R5_map = convolve(im, R5R5, mode="reflect")
    L5S5_S5L5_map = (
        convolve(im, L5S5, mode="reflect") + convolve(im, S5L5, mode="reflect")
    ) / 2.0
    E5E5_map = convolve(im, E5E5, mode="reflect")
    E5R5_R5E5_map = (
        convolve(im, E5R5, mode="reflect") + convolve(im, R5E5, mode="reflect")
    ) / 2.0
    S5R5_R5S5_map = (
        convolve(im, S5R5, mode="reflect") + convolve(im, R5S5, mode="reflect")
    ) / 2.0

    # Convert response maps to energy maps using a 7x7 kernel of all ones on the absolute values
    k = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)
    maps = np.stack(
        [
            convolve(np.abs(L5E5_E5L5_map), k, mode="reflect"),  # 0
            convolve(np.abs(L5R5_R5L5_map), k, mode="reflect"),  # 1
            convolve(np.abs(E5S5_S5E5_map), k, mode="reflect"),  # 2
            convolve(np.abs(S5S5_map), k, mode="reflect"),  # 3
            convolve(np.abs(R5R5_map), k, mode="reflect"),  # 4
            convolve(np.abs(L5S5_S5L5_map), k, mode="reflect"),  # 5
            convolve(np.abs(E5E5_map), k, mode="reflect"),  # 6
            convolve(np.abs(E5R5_R5E5_map), k, mode="reflect"),  # 7
            convolve(np.abs(S5R5_R5S5_map), k, mode="reflect"),  # 8
        ],
        axis=0,
    )
    return np.abs(maps)


data = [
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning/EBSD_000_PRIAS.tiff",
        1.5,
        "67_EBSD_PRIAS_000",
        "#ea00ff",
        "<",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning/EBSD_000_IQ.tiff",
        1.5,
        "67_EBSD_IQ_000",
        "#0011ff",
        ">",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning/EBSD_000_CI.tiff",
        1.5,
        "67_EBSD_CI_000",
        "#00eeff",
        "^",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning/BSE_000.tif",
        0.520833333,
        "67_BSE_000",
        "#1eff00",
        "X",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/Ta-AM-Spalled_SEM-BSE_EBSD_SameSliceSerialSectioning/EBSD_001_PRIAS.tiff",
        1.5,
        "Ta_EBSD_PRIAS_001",
        "#ea00ff",
        "<",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/Ta-AM-Spalled_SEM-BSE_EBSD_SameSliceSerialSectioning/EBSD_001_IQ.tiff",
        1.5,
        "Ta_EBSD_IQ_001",
        "#0011ff",
        ">",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/Ta-AM-Spalled_SEM-BSE_EBSD_SameSliceSerialSectioning/EBSD_001_CI.tiff",
        1.5,
        "Ta_EBSD_CI_001",
        "#00eeff",
        "^",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/Ta-AM-Spalled_SEM-BSE_EBSD_SameSliceSerialSectioning/BSE_001.tif",
        0.27669271,
        "Ta_BSE_001",
        "#1eff00",
        "X",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM67_OM-SEM_Multiscale/CoNi67_BSE.tif",
        0.1038411,
        "67_BSE",
        "#1eff00",
        "X",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM67_OM-SEM_Multiscale/CoNi67_SE.tif",
        0.1038411,
        "67_SE",
        "#ffee00",
        "P",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM67_OM-SEM_Multiscale/CoNi67_high_OM.tif",
        0.113733,
        "67_high_OM",
        "#ff8800",
        "o",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM67_OM-SEM_Multiscale/CoNi67_mid_OM.tif",
        0.83682,
        "67_mid_OM",
        "#ff0000",
        "s",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM90_OM-SEM_Multiscale/CoNi90_BSE.tif",
        0.134766,
        "90_BSE",
        "#1eff00",
        "X",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM90_OM-SEM_Multiscale/CoNi90_SE.tif",
        0.134766,
        "90_SE",
        "#ffee00",
        "P",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM90_OM-SEM_Multiscale/CoNi90_high_OM.tif",
        0.113733,
        "90_high_OM",
        "#ff8800",
        "o",
    ),
    (
        "D:/Research/Datasets-Projects/MatchAnything_Datasets/CoNi-AM90_OM-SEM_Multiscale/CoNi90_mid_OM.tif",
        0.83682,
        "90_mid_OM",
        "#ff0000",
        "s",
    ),
]
all_res = [res for _, res, _, _, _ in data]

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].set_title("Mean of Texture Energy Maps")
ax[0].set_xlabel("Map Index")
ax[0].set_ylabel("Mean Value")
ax[1].set_title("Standard Deviation of Texture Energy Maps")
ax[1].set_xlabel("Map Index")
ax[1].set_ylabel("Standard Deviation")

distances = []
for path, res, name, color, marker in data:
    im = io.imread(path, as_gray=True).astype(np.float32)
    print(f"Loaded image {name} with shape {im.shape} and resolution {res} um/pixel")

    im = im[
        im.shape[0] // 3 : im.shape[0] * 2 // 3, im.shape[1] // 3 : im.shape[1] * 2 // 3
    ]

    energy_maps = get_energy_maps(im, kernel_size=7)  # * int(max(all_res) / res))

    mean_vals = [energy_maps[i].mean() for i in range(energy_maps.shape[0])]
    std_vals = [energy_maps[i].std() for i in range(energy_maps.shape[0])]
    distances.append((name, np.linalg.norm(mean_vals), np.linalg.norm(std_vals)))

    ax[0].scatter(
        np.arange(len(mean_vals)),
        mean_vals,
        s=50,
        label=name,
        marker=marker,
        color=color,
    )
    ax[1].scatter(
        np.arange(len(std_vals)),
        std_vals,
        s=50,
        label=name,
        marker=marker,
        color=color,
    )

    # fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    # for i in range(3):
    #     for j in range(3):
    #         a = axes[i, j]
    #         idx = i * 3 + j
    #         vmin, vmax = np.percentile(energy_maps[idx], (10, 90))
    #         a.imshow(energy_maps[idx], cmap="gray", vmin=vmin, vmax=vmax)
    #         a.set_title(f"Map {idx+1} - {name}")
    #         a.axis("off")
    # plt.tight_layout()


ax[0].legend()
ax[1].legend()
plt.tight_layout()

plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].set_title("Texture Energy Map Distance (Mean)")
ax[0].set_xlabel("Sample Name")
ax[0].set_ylabel("Euclidean Distance")
ax[1].set_title("Texture Energy Map Distance (Std Dev)")
ax[1].set_xlabel("Sample Name")
for i, (name, mean_dist, std_dist) in enumerate(distances):
    ax[0].bar(i, mean_dist, color=data[i][3])
    ax[1].bar(i, std_dist, color=data[i][3])

ax[0].set_xticks(range(len(distances)))
ax[0].set_xticklabels([name for name, _, _ in distances], rotation=45, ha="right")
ax[1].set_xticks(range(len(distances)))
ax[1].set_xticklabels([name for name, _, _ in distances], rotation=45, ha="right")
plt.tight_layout()
plt.show()
