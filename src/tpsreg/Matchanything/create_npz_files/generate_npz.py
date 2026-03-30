import os
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Union

import pickle
import numpy as np
from skimage import io
import tkinter as tk


save_folder_base = "/Users/jameslamb/Documents/research/data/match_anything_datasets/"


pick = 3

if pick == 0:
    json_paths = [
        "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/match_anything/CoNi-AM90_SEM-EBSD_SameSlice.json",
    ]
    parent_name = "CoNi-AM90_SEM-EBSD_SameSlice"
    same_scene = True
elif pick == 1:
    json_paths = [
        "/Users/jameslamb/Documents/research/data/CoNi-DIC-S1/match_anything/CoNi-AM90_SEM-DIC_EBSD_SlipPartitioning.json",
    ]
    parent_name = "CoNi-AM90_SEM-DIC_EBSD_SlipPartitioning"
    same_scene = True
elif pick == 2:
    json_paths = [
        "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/EBSD-SEM_001.json",
        "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/EBSD-SEM_450.json",
        "/Users/jameslamb/Documents/research/data/auto_reg_data/AMSpalledTa/EBSD-SEM_900.json",
    ]
    parent_name = "Ta-AM-Spalled_SEM-BSE_EBSD_SameSliceSerialSectioning"
    same_scene = False
elif pick == 3:
    json_paths = [
        "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/EBSD-SEM_000.json",
        "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/EBSD-SEM_101.json",
        "/Users/jameslamb/Documents/research/data/auto_reg_data/CoNi67/EBSD-SEM_202.json",
    ]
    parent_name = "CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning"
    same_scene = False
elif pick == 4:
    json_paths = [
        "/Users/jameslamb/Documents/research/data/CoNi-HREBSD-polish/67/CoNi67_high_OM-2-BSE.json",
        "/Users/jameslamb/Documents/research/data/CoNi-HREBSD-polish/67/CoNi67_high_OM-2-SE.json",
        "/Users/jameslamb/Documents/research/data/CoNi-HREBSD-polish/67/CoNi67_mid_OM-2-high_OM.json",
    ]
    parent_name = "CoNi-AM67_OM-SEM_Multiscale"
    same_scene = True
elif pick == 5:
    json_paths = [
        "/Users/jameslamb/Documents/research/data/CoNi-HREBSD-polish/90/CoNi90_mid_OM-2-BSE.json",
        "/Users/jameslamb/Documents/research/data/CoNi-HREBSD-polish/90/CoNi90_high_OM-2-BSE.json",
    ]
    parent_name = "CoNi-AM90_OM-SEM_Multiscale"
    same_scene = True


def read_ang_header(path):
    """Reads the header of an ang file"""
    col_names = None
    header = ""
    header_lines = 0
    with open(path, "r") as f:
        for line in f:
            header += line
            if "NCOLS_ODD" in line:
                ncols = int(line.split(": ")[1].strip())
            elif "NROWS" in line:
                nrows = int(line.split(": ")[1].strip())
            elif "COLUMN_HEADERS" in line:
                col_names = line.replace("\n", "").split(": ")[1].strip().split(", ")
            elif "XSTEP" in line:
                res = float(line.split(": ")[1].strip())
            elif "HEADER: End" in line:
                break
            header_lines += 1
    if col_names is None:
        col_names = ["phi1", "PHI", "phi2", "x", "y", "IQ", "CI", "Phase index"]
    return ncols, nrows, col_names, res, header, header_lines


def read_ang(path):
    """Reads an ang file into a numpy array"""
    ncols, nrows, col_names, res, header, header_lines = read_ang_header(path)
    raw_data = np.genfromtxt(path, skip_header=header_lines, dtype=float)
    n_entries = raw_data.shape[-1]
    if raw_data.shape[0] == ncols * nrows:
        data = raw_data.reshape((nrows, ncols, n_entries))
    elif raw_data.shape != ncols * nrows:
        raise ValueError(
            f"The number of data points ({raw_data.size}) does not match the expected grid ({nrows} rows, {ncols} cols, {ncols * nrows} total points). "
        )

    out = {col_names[i]: data[:, :, i] for i in range(n_entries)}
    out["EulerAngles"] = np.array([out["phi1"], out["PHI"], out["phi2"]]).T.astype(
        float
    )
    for key in out.keys():
        if key not in ["EulerAngles"]:
            out[key] = np.fliplr(np.rot90(out[key], k=3))
        if len(out[key].shape) == 2:
            out[key] = out[key].T
        else:
            out[key] = out[key].transpose((1, 0, 2))
        out[key] = out[key].reshape((1,) + out[key].shape)

    # Get the grain file if it exists
    dirname = os.path.dirname(path)
    basenames = [
        os.path.splitext(os.path.basename(path))[0] + "_Grain.txt",
        os.path.splitext(os.path.basename(path))[0] + "_grain.txt",
    ]
    grain_file_exists = [
        os.path.exists(os.path.join(dirname, basename)) for basename in basenames
    ]
    if any(grain_file_exists):
        grain_path = os.path.join(dirname, basenames[grain_file_exists.index(True)])
        grain_data = read_grainFile(grain_path)
        out["GrainIDs"] = grain_data.reshape((1,) + (nrows, ncols))
    return out, res


def read_grainFile(path):
    with open(path, "r") as f:
        for line in f:
            if line[0] != "#":
                break
            if "Grain ID" in line:
                column = int(
                    line.split(": ")[0].replace("#", "").replace("Column", "").strip()
                )
                break
    grain_data = np.genfromtxt(
        path, comments="#", delimiter="\n", skip_header=1, dtype=str
    )
    f = (
        lambda x: x.replace("      ", " ")
        .replace("     ", " ")
        .replace("    ", " ")
        .replace("   ", " ")
        .replace("  ", " ")
        .split(" ")
    )
    grainIDs = np.array([f(x)[column - 1] for x in grain_data]).reshape(-1).astype(int)
    grainIDs[grainIDs <= 0] = 0
    return grainIDs


class Image:
    def __init__(
        self,
        path: Union[str, Path],
        resolution: float,
        width: int,
        height: int,
        microscope: str,
        detector: str,
        derived: bool = False,
        derived_info: str = None,
        stitched: bool = False,
    ):
        self.path = Path(path) if isinstance(path, str) else path
        self.resolution = resolution
        self.width = width
        self.height = height
        self.microscope = microscope
        self.detector = detector
        self.derived = derived
        self.derived_info = derived_info
        self.stitched = stitched

    def __eq__(self, value):
        return (
            self.path == value.path
            and self.resolution == value.resolution
            and self.width == value.width
            and self.height == value.height
            and self.microscope == value.microscope
            and self.detector == value.detector
            and self.derived == value.derived
            and self.derived_info == value.derived_info
            and self.stitched == value.stitched
        )

    def __repr__(self):
        return f"Image(path={self.path.name}, resolution={self.resolution}, width={self.width}, height={self.height}, microscope={self.microscope}, detector={self.detector}, derived={self.derived}, derived_info={self.derived_info}, stitched={self.stitched})"

    @property
    def metadata(self):
        return {
            "Unique Images": self.path.name,
            "Physical Pixel Size [m]": float(self.resolution) / 1e6,
            "Resolution Width": self.width,
            "Resolution Height": self.height,
            "Microscope Type": self.microscope,
            "Detector/Imaging Mode": self.detector,
            "Derived Modality": self.derived,
            "Derived Modality Type": self.derived_info,
            "Stitching Indicator": self.stitched,
        }


class ImagePair:
    def __init__(self, source: Image, destination: Image, points: np.ndarray):
        self.source = source
        self.destination = destination
        self.points = points  # Nx4 array: [dest_x, dest_y, src_x, src_y]

    def __eq__(self, value):
        return (
            self.source == value.source
            and self.destination == value.destination
            and np.array_equal(self.points, value.points)
        )

    def __repr__(self):
        return f"ImagePair(source={self.source}, destination={self.destination}, points_shape={self.points.shape})"


class Project:
    def __init__(self, name, image_pairs: List[ImagePair] = None):
        self.name = name
        self.image_pairs = image_pairs if image_pairs is not None else []
        self.images = []
        self.image_pair_idxs = []
        self._get_images_from_pairs()

    def _get_images_from_pairs(self):
        images = []
        image_pair_idxs = []
        for pair in self.image_pairs:
            if pair.source not in images:
                images.append(pair.source)
                source_idx = len(images) - 1
            else:
                source_idx = images.index(pair.source)
            if pair.destination not in images:
                images.append(pair.destination)
                dest_idx = len(images) - 1
            else:
                dest_idx = images.index(pair.destination)
            image_pair_idxs.append((source_idx, dest_idx))
        self.images = images
        self.image_pair_idxs = image_pair_idxs

    def __eq__(self, value):
        return self.image_pairs == value.image_pairs

    def __repr__(self):
        return f"Project(image_pairs={self.image_pairs})"

    def __len__(self):
        return len(self.image_pairs)

    def add_pair(self, image_pair: ImagePair):
        self.image_pairs.append(image_pair)
        self._get_images_from_pairs()

    def save(self, path: Union[str, Path], scene_idx: int = 0):
        # Handle paths
        # Create the output folder if it doesn't exist
        # save_folder
        #     └── name
        #          ├── eval_indexs
        #          └── scenes

        sub_name = self.name + f"_{scene_idx}"
        path = Path(path) if isinstance(path, str) else path
        name_path = path / self.name
        scenes_path = name_path / "scenes"
        scene_path = scenes_path / sub_name
        eval_indexs_path = name_path / "eval_indexs"
        os.makedirs(scenes_path, exist_ok=True)
        os.makedirs(scene_path, exist_ok=True)
        os.makedirs(eval_indexs_path, exist_ok=True)

        # Copy images
        for img in self.images:
            # copy to scenes folder
            new_image_path = scene_path / img.path.name
            shutil.copy(img.path, new_image_path)

            # update image path in images list and image pairs
            for pair in self.image_pairs:
                if pair.source.path == img.path:
                    pair.source.path = new_image_path
                if pair.destination.path == img.path:
                    pair.destination.path = new_image_path
            img.path = new_image_path

        # Create metadata
        metadata = [img.metadata for img in self.images]

        # Save metadata as JSON
        metadata_path = name_path / f"{sub_name}_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Save the .npz file
        paths = [str(img.path.relative_to(name_path)) for img in self.images]
        pair_infos = [
            ([target_i, source_i], 1) for source_i, target_i in self.image_pair_idxs
        ]
        gt_2D_matches = [pair.points for pair in self.image_pairs]
        data = dict(
            dataset_name=self.name,  # string
            image_paths=paths,  # list of paths (strings)
            image_metadata=metadata,  # dictionary of metadata
            pair_infos=pair_infos,  # list of tuples: ([target_idx, source_idx], 1)
            gt_2D_matches=gt_2D_matches,  # list of Nx4 arrays, [target_x, target_y, source_x, source_y]
        )
        data_path = eval_indexs_path / f"eval_{sub_name}.npz"
        pickle.dump(data, open(data_path, "wb"))


class MetadataInputGUI:
    def __init__(self, image_name: str):
        self.image_name = image_name
        self.result = None

        self.microscope_lookup = {
            "SEM": "Scanning Electron Microscope",
            "LOM": "Light Optical Microscope",
            "STEM": "Scanning Transmission Electron Microscope",
        }

        self.detector_lookup = {
            "ETD": "Everhart-Thornley Detector",
            "BSE": "Backscattered Electron Detector",
            "EBSD": "Electron Backscatter Diffraction",
            "BF": "Bright-Field",
            "DF": "Dark-Field",
            "HAADF": "High-Angle Annular Dark-Field",
        }

        self.root = tk.Tk()
        self.root.title(f"Metadata for {image_name}")
        self._create_widgets()

    def _create_widgets(self):
        if "_ci" in self.image_name.lower():
            microscope = "SEM"
            detector = "EBSD"
            derived = True
            derived_info = "Confidence Index (Spherical Indexing)"
            stitched = False
        elif "_iq" in self.image_name.lower():
            microscope = "SEM"
            detector = "EBSD"
            derived = True
            derived_info = "Pattern Sharpness"
            stitched = False
        elif "_prias" in self.image_name.lower():
            microscope = "SEM"
            detector = "EBSD"
            derived = True
            derived_info = "PRIAS"
            stitched = False
        elif "om" in self.image_name.lower() or "lom" in self.image_name.lower():
            microscope = "LOM"
            detector = "BF"
            derived = False
            derived_info = ""
            stitched = False
        elif "bse" in self.image_name.lower():
            microscope = "SEM"
            detector = "BSE"
            derived = False
            derived_info = ""
            stitched = False
        elif "exx" in self.image_name.lower():
            microscope = "SEM"
            detector = "ETD"
            derived = True
            derived_info = (
                "High-Resolution Digital Image Correlation Logitudinal Strain Map"
            )
            stitched = True
        else:
            microscope = "SEM"
            detector = "ETD"
            derived = False
            derived_info = ""
            stitched = False

        # Image name label
        tk.Label(
            self.root, text=f"Image: {self.image_name}", font=("Arial", 12, "bold")
        ).grid(row=0, column=0, columnspan=2, pady=10, padx=10)

        # Microscope type
        tk.Label(self.root, text="Microscope Type:").grid(
            row=1, column=0, sticky="e", padx=10, pady=5
        )
        self.microscope_var = tk.StringVar(value=microscope)
        microscope_frame = tk.Frame(self.root)
        microscope_frame.grid(row=1, column=1, sticky="w", padx=10, pady=5)
        for idx, key in enumerate(self.microscope_lookup.keys()):
            tk.Radiobutton(
                microscope_frame, text=key, variable=self.microscope_var, value=key
            ).pack(side="left")

        # Detector type
        tk.Label(self.root, text="Detector Type:").grid(
            row=2, column=0, sticky="e", padx=10, pady=5
        )
        self.detector_var = tk.StringVar(value=detector)
        detector_frame = tk.Frame(self.root)
        detector_frame.grid(row=2, column=1, sticky="w", padx=10, pady=5)
        row1 = tk.Frame(detector_frame)
        row1.pack()
        row2 = tk.Frame(detector_frame)
        row2.pack()
        for idx, key in enumerate(self.detector_lookup.keys()):
            frame = row1 if idx < 3 else row2
            tk.Radiobutton(frame, text=key, variable=self.detector_var, value=key).pack(
                side="left"
            )

        # Derived image
        tk.Label(self.root, text="Derived Image:").grid(
            row=3, column=0, sticky="e", padx=10, pady=5
        )
        self.derived_var = tk.BooleanVar(value=derived)
        derived_frame = tk.Frame(self.root)
        derived_frame.grid(row=3, column=1, sticky="w", padx=10, pady=5)
        tk.Radiobutton(
            derived_frame,
            text="Yes",
            variable=self.derived_var,
            value=True,
            command=self._toggle_derived_info,
        ).pack(side="left")
        tk.Radiobutton(
            derived_frame,
            text="No",
            variable=self.derived_var,
            value=False,
            command=self._toggle_derived_info,
        ).pack(side="left")

        # Derived info (conditional)
        self.derived_info_label = tk.Label(self.root, text="Derived Info:")
        self.derived_info_entry = tk.Entry(self.root, width=30)
        if derived:
            self.derived_info_label.grid(row=4, column=0, sticky="e", padx=10, pady=5)
            self.derived_info_entry.grid(row=4, column=1, sticky="w", padx=10, pady=5)
            self.derived_info_entry.insert(0, derived_info)

        # Stitched image
        tk.Label(self.root, text="Stitched Image:").grid(
            row=5, column=0, sticky="e", padx=10, pady=5
        )
        self.stitched_var = tk.BooleanVar(value=stitched)
        stitched_frame = tk.Frame(self.root)
        stitched_frame.grid(row=5, column=1, sticky="w", padx=10, pady=5)
        tk.Radiobutton(
            stitched_frame, text="Yes", variable=self.stitched_var, value=True
        ).pack(side="left")
        tk.Radiobutton(
            stitched_frame, text="No", variable=self.stitched_var, value=False
        ).pack(side="left")

        # Submit button
        tk.Button(
            self.root,
            text="Submit",
            command=self._submit,
            bg="green",
            fg="white",
            font=("Arial", 10, "bold"),
        ).grid(row=6, column=0, columnspan=2, pady=20)

    def _toggle_derived_info(self):
        if self.derived_var.get():
            self.derived_info_label.grid(row=4, column=0, sticky="e", padx=10, pady=5)
            self.derived_info_entry.grid(row=4, column=1, sticky="w", padx=10, pady=5)
        else:
            self.derived_info_label.grid_remove()
            self.derived_info_entry.grid_remove()

    def _submit(self):
        microscope = self.microscope_lookup[self.microscope_var.get()]
        detector = self.detector_lookup[self.detector_var.get()]
        derived = self.derived_var.get()
        derived_info = self.derived_info_entry.get() if derived else None
        stitched = self.stitched_var.get()

        self.result = (
            microscope,
            detector,
            int(derived),
            str(derived_info),
            int(stitched),
        )
        self.root.destroy()

    def get_result(self):
        self.root.mainloop()
        return self.result


def normalize(img):
    img_min = np.min(img)
    img_max = np.max(img)
    return np.around(255 * ((img - img_min) / (img_max - img_min))).astype(np.uint8)


def get_metadata(image_name: str) -> Tuple[str, str, bool, str, bool]:
    """Get metadata for an image using a GUI."""
    gui = MetadataInputGUI(image_name)
    return gui.get_result()


if same_scene:
    # All JSON files belong to the same scene
    project = Project(parent_name)

    for idx in range(len(json_paths)):
        json_path = json_paths[idx]
        name = os.path.basename(json_path).replace(".json", "")
        print(f"Processing {name}...")

        # Get data from project
        with open(json_path, "r") as f:
            project_data = json.load(f)
        source_image_paths_dict = project_data["settings"]["source_paths"]
        target_image_paths_dict = project_data["settings"]["destination_paths"]
        source_resolution = project_data["settings"]["source_resolution"]
        target_resolution = project_data["settings"]["destination_resolution"]
        source_points = np.array(project_data["source_points"]["0"])
        target_points = np.array(project_data["destination_points"]["0"])
        points = np.hstack((target_points, source_points))

        # Parse source and destination image paths
        source_image_paths = [Path(k[0]) for k in source_image_paths_dict.values()]
        destination_image_paths = [Path(k[0]) for k in target_image_paths_dict.values()]

        # If an ang file is present, generate images from that
        for i in range(len(source_image_paths)):
            if source_image_paths[i].suffix == ".ang":
                path = source_image_paths.pop(i)
                ang_data = read_ang(path)[0]
                ang_data["IQ"][0][np.isnan(ang_data["IQ"][0])] = 0.0
                ang_data["CI"][0][np.isnan(ang_data["CI"][0])] = 0.0
                iq_path = path.with_stem(path.stem + "_IQ").with_suffix(".tiff")
                ci_path = path.with_stem(path.stem + "_CI").with_suffix(".tiff")
                prias_path = path.with_stem(path.stem + "_PRIAS").with_suffix(".tiff")
                io.imsave(iq_path, normalize(ang_data["IQ"][0]))
                io.imsave(ci_path, normalize(ang_data["CI"][0]))
                prias_img = normalize(
                    np.stack(
                        (
                            ang_data["PRIAS Bottom Strip"][0],
                            ang_data["PRIAS Center Square"][0],
                            ang_data["PRIAS Top Strip"][0],
                        ),
                        axis=2,
                    )
                )
                io.imsave(prias_path, prias_img)
                source_image_paths.append(iq_path)
                source_image_paths.append(ci_path)
                source_image_paths.append(prias_path)

        for i in range(len(destination_image_paths)):
            if destination_image_paths[i].suffix == ".ang":
                path = destination_image_paths.pop(i)
                ang_data = read_ang(path)[0]
                iq_path = path.with_stem(path.stem + "_IQ").with_suffix(".tiff")
                ci_path = path.with_stem(path.stem + "_CI").with_suffix(".tiff")
                prias_path = path.with_stem(path.stem + "_PRIAS").with_suffix(".tiff")
                io.imsave(iq_path, normalize(ang_data["IQ"][0]))
                io.imsave(ci_path, normalize(ang_data["CI"][0]))
                prias_img = normalize(
                    np.stack(
                        (
                            ang_data["PRIAS Bottom Strip"][0],
                            ang_data["PRIAS Center Square"][0],
                            ang_data["PRIAS Top Strip"][0],
                        ),
                        axis=2,
                    )
                )
                io.imsave(prias_path, prias_img)
                destination_image_paths.append(iq_path)
                destination_image_paths.append(ci_path)
                destination_image_paths.append(prias_path)

        print("Source images:", [p.name for p in source_image_paths])
        print("Destination images:", [p.name for p in destination_image_paths])

        source_images = []
        for path in source_image_paths:
            img = io.imread(path)
            microscope, detector, derived, derived_info, stitched = get_metadata(
                path.name
            )
            # microscope, detector, derived, derived_info, stitched = ("Scanning Electron Microscope", "Backscattered Electron Detector", False, None, False)
            img_obj = Image(
                path=path,
                resolution=source_resolution,
                width=img.shape[1],
                height=img.shape[0],
                microscope=microscope,
                detector=detector,
                derived=derived,
                derived_info=derived_info,
                stitched=stitched,
            )
            source_images.append(img_obj)

        destination_images = []
        for path in destination_image_paths:
            img = io.imread(path)
            microscope, detector, derived, derived_info, stitched = get_metadata(
                path.name
            )
            # microscope, detector, derived, derived_info, stitched = ("Scanning Electron Microscope", "Backscattered Electron Detector", False, None, False)
            img_obj = Image(
                path=path,
                resolution=target_resolution,
                width=img.shape[1],
                height=img.shape[0],
                microscope=microscope,
                detector=detector,
                derived=derived,
                derived_info=derived_info,
                stitched=stitched,
            )
            destination_images.append(img_obj)

        source_idx = np.arange(len(source_images))
        destination_idx = np.arange(len(destination_images))
        pair_idx = np.array(np.meshgrid(destination_idx, source_idx)).T.reshape(-1, 2)

        for di, si in pair_idx:
            image_pair = ImagePair(
                source=source_images[si],
                destination=destination_images[di],
                points=points,
            )
            project.add_pair(image_pair)

    # Save all pairs as a single scene
    project.save(save_folder_base, scene_idx=0)

else:
    # Each JSON file is a separate scene
    for idx in range(len(json_paths)):
        json_path = json_paths[idx]
        name = os.path.basename(json_path).replace(".json", "")
        print(f"Processing {name} as scene {idx}...")

        project = Project(parent_name)

        # Get data from project
        with open(json_path, "r") as f:
            project_data = json.load(f)
        source_image_paths_dict = project_data["settings"]["source_paths"]
        target_image_paths_dict = project_data["settings"]["destination_paths"]
        source_resolution = project_data["settings"]["source_resolution"]
        target_resolution = project_data["settings"]["destination_resolution"]
        source_points = np.array(project_data["source_points"]["0"])
        target_points = np.array(project_data["destination_points"]["0"])
        points = np.hstack((target_points, source_points))

        # Parse source and destination image paths
        source_image_paths = [Path(k[0]) for k in source_image_paths_dict.values()]
        destination_image_paths = [Path(k[0]) for k in target_image_paths_dict.values()]

        # If an ang file is present, generate images from that
        for i in range(len(source_image_paths)):
            if source_image_paths[i].suffix == ".ang":
                path = source_image_paths.pop(i)
                ang_data = read_ang(path)[0]
                ang_data["IQ"][0][np.isnan(ang_data["IQ"][0])] = 0.0
                ang_data["CI"][0][np.isnan(ang_data["CI"][0])] = 0.0
                iq_path = path.with_stem(path.stem + "_IQ").with_suffix(".tiff")
                ci_path = path.with_stem(path.stem + "_CI").with_suffix(".tiff")
                prias_path = path.with_stem(path.stem + "_PRIAS").with_suffix(".tiff")
                io.imsave(iq_path, normalize(ang_data["IQ"][0]))
                io.imsave(ci_path, normalize(ang_data["CI"][0]))
                prias_img = normalize(
                    np.stack(
                        (
                            ang_data["PRIAS Bottom Strip"][0],
                            ang_data["PRIAS Center Square"][0],
                            ang_data["PRIAS Top Strip"][0],
                        ),
                        axis=2,
                    )
                )
                io.imsave(prias_path, prias_img)
                source_image_paths.append(iq_path)
                source_image_paths.append(ci_path)
                source_image_paths.append(prias_path)

        for i in range(len(destination_image_paths)):
            if destination_image_paths[i].suffix == ".ang":
                path = destination_image_paths.pop(i)
                ang_data = read_ang(path)[0]
                iq_path = path.with_stem(path.stem + "_IQ").with_suffix(".tiff")
                ci_path = path.with_stem(path.stem + "_CI").with_suffix(".tiff")
                prias_path = path.with_stem(path.stem + "_PRIAS").with_suffix(".tiff")
                io.imsave(iq_path, normalize(ang_data["IQ"][0]))
                io.imsave(ci_path, normalize(ang_data["CI"][0]))
                prias_img = normalize(
                    np.stack(
                        (
                            ang_data["PRIAS Bottom Strip"][0],
                            ang_data["PRIAS Center Square"][0],
                            ang_data["PRIAS Top Strip"][0],
                        ),
                        axis=2,
                    )
                )
                io.imsave(prias_path, prias_img)
                destination_image_paths.append(iq_path)
                destination_image_paths.append(ci_path)
                destination_image_paths.append(prias_path)

        print("Source images:", [p.name for p in source_image_paths])
        print("Destination images:", [p.name for p in destination_image_paths])

        source_images = []
        for path in source_image_paths:
            img = io.imread(path)
            microscope, detector, derived, derived_info, stitched = get_metadata(
                path.name
            )
            # microscope, detector, derived, derived_info, stitched = ("Scanning Electron Microscope", "Backscattered Electron Detector", False, None, False)
            img_obj = Image(
                path=path,
                resolution=source_resolution,
                width=img.shape[1],
                height=img.shape[0],
                microscope=microscope,
                detector=detector,
                derived=derived,
                derived_info=derived_info,
                stitched=stitched,
            )
            source_images.append(img_obj)

        destination_images = []
        for path in destination_image_paths:
            img = io.imread(path)
            microscope, detector, derived, derived_info, stitched = get_metadata(
                path.name
            )
            # microscope, detector, derived, derived_info, stitched = ("Scanning Electron Microscope", "Backscattered Electron Detector", False, None, False)
            img_obj = Image(
                path=path,
                resolution=target_resolution,
                width=img.shape[1],
                height=img.shape[0],
                microscope=microscope,
                detector=detector,
                derived=derived,
                derived_info=derived_info,
                stitched=stitched,
            )
            destination_images.append(img_obj)

        source_idx = np.arange(len(source_images))
        destination_idx = np.arange(len(destination_images))
        pair_idx = np.array(np.meshgrid(destination_idx, source_idx)).T.reshape(-1, 2)

        for di, si in pair_idx:
            image_pair = ImagePair(
                source=source_images[si],
                destination=destination_images[di],
                points=points,
            )
            project.add_pair(image_pair)

        # Save this scene with its own index
        project.save(save_folder_base, scene_idx=idx)
