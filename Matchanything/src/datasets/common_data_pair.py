import os.path as osp
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from loguru import logger
from PIL import Image
from pathlib import Path
from src.utils.dataset import read_megadepth_gray

RGB_WEIGHTS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

def _to_float32_image(arr: np.ndarray) -> np.ndarray:
    """Convert input image array to float32 and normalise to [0, 1] when needed."""
    original_dtype = arr.dtype
    arr_float = arr.astype(np.float32, copy=False)
    if arr_float.size == 0:
        return arr_float
    if np.issubdtype(original_dtype, np.integer):
        data_min = float(arr_float.min())
        data_max = float(arr_float.max())
        if data_max > data_min:
            return (arr_float - data_min) / (data_max - data_min)
        return np.zeros_like(arr_float)
    if np.issubdtype(original_dtype, np.floating):
        finite_mask = np.isfinite(arr_float)
        if not np.any(finite_mask):
            return np.zeros_like(arr_float)
        finite_vals = arr_float[finite_mask]
        data_min = float(finite_vals.min())
        data_max = float(finite_vals.max())
        if data_max > data_min:
            if data_min < 0.0 or data_max > 1.0:
                return (arr_float - data_min) / (data_max - data_min)
            return arr_float
        return np.zeros_like(arr_float)
    return arr_float

def _rgb_to_gray(arr: np.ndarray) -> np.ndarray:

    """Collapse an RGB-like array to grayscale using luminance weights."""
    if arr.ndim == 2:
        return arr.astype(np.float32)
    if arr.shape[-1] == 1:
        return arr[..., 0].astype(np.float32)
    if arr.shape[-1] >= 3:
        return np.tensordot(arr[..., :3], RGB_WEIGHTS, axes=([-1], [0])).astype(np.float32)
    return arr.mean(axis=-1).astype(np.float32)

def _ensure_three_channels(arr: np.ndarray) -> np.ndarray:

    """Ensure the array has exactly three channels."""
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], repeats=3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, repeats=3, axis=-1)
    elif arr.shape[-1] > 3:
        arr = arr[..., :3]
    return arr

def _normalize(arr: np.ndarray) -> np.ndarray:
    """Stretch the dynamic range of the array to [0, 1]."""
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim == 2:
        finite_mask = np.isfinite(arr)
        if not np.any(finite_mask):
            return np.zeros_like(arr)
        finite_vals = arr[finite_mask]
        arr_min = float(finite_vals.min())
        arr_max = float(finite_vals.max())
        denom = arr_max - arr_min
        if denom > 1e-6:
            scaled = (arr - arr_min) / denom
        else:
            scaled = np.zeros_like(arr)
        scaled[~finite_mask] = 0.0
        return np.clip(scaled, 0.0, 1.0)
    flat = arr.reshape(-1, arr.shape[-1])
    finite_flat = np.where(np.isfinite(flat), flat, np.nan)
    with np.errstate(invalid="ignore"):
        ch_min = np.nanmin(finite_flat, axis=0)
        ch_max = np.nanmax(finite_flat, axis=0)
    denom = ch_max - ch_min
    valid = np.isfinite(ch_min) & np.isfinite(ch_max) & (denom > 1e-6)
    safe_denom = np.where(valid, denom, 1.0)
    norm = (arr - ch_min.reshape(1, 1, -1)) / safe_denom.reshape(1, 1, -1)
    if np.any(~valid):
        norm[..., ~valid] = 0.0
    norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(norm, 0.0, 1.0)

def _load_gray_tensor(path: Path, normalize: bool = False) -> torch.Tensor:

    with Image.open(path) as img:
        arr = np.array(img)
    arr = _to_float32_image(arr)
    arr = _rgb_to_gray(arr)
    if normalize:
        arr = _normalize(arr)
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr)[None]

def _load_rgb_tensor(path: Path, normalize: bool = False):

    with Image.open(path) as img:
        arr = np.array(img)
    arr = _to_float32_image(arr)
    arr = _ensure_three_channels(arr)
    if normalize:
        arr = _normalize(arr)
    arr = np.clip(arr, 0.0, 1.0)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1))
    rgb_uint8 = (arr * 255.0).round().clip(0, 255).astype(np.uint8)
    return tensor, rgb_uint8

class CommonDataset(Dataset):

    def __init__(self,
                 root_dir,
                 npz_path=None,
                 mode='train',
                 min_overlap_score=0.4,
                 img_resize=None,
                 df=None,
                 img_padding=False,
                 depth_padding=False,
                 augment_fn=None,
                 testNpairs=300,
                 fp16=False,
                 fix_bias=False,
                 sample_ratio=1.0,
                 dataset_name=None,
                 **kwargs):
        super().__init__()
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.mode = mode
        self.sample_ratio = sample_ratio
        self.fp16 = fp16
        self.fix_bias = fix_bias
        self.img_resize = img_resize
        self.df = df
        self.img_padding = img_padding
        self.augment_fn = augment_fn if mode == 'train' else None
        self.coarse_scale = kwargs.get("coarse_scale", 0.125)
        self.load_origin_rgb = kwargs.get("load_origin_rgb", False)
        self.read_gray = kwargs.get("read_gray", True)
        self.normalize_img = kwargs.get("normalize_img", False)
        self.resize_by_stretch = kwargs.get("resize_by_stretch", False)
        self.gt_matches_padding_n = kwargs.get("gt_matches_padding_n", 100)
        self.scene_info = None
        self.pair_infos = None
        self.dataset_name = dataset_name
        self.gt_matches = None
        self.gt_2D_warp = None
        self.gt_2D_matches = None
        self.intrins = None
        self.poses = None
        self.scene_id = None
        self.depth_max_size = 2000
        self._pairs_dataset = None
        if mode == 'train':
            assert img_resize is not None and depth_padding
        if self.fix_bias:
            self.df = 1
        if npz_path is not None:
            self._init_from_npz(npz_path, min_overlap_score, depth_padding, testNpairs)
        else:
            raise ValueError("Either npz_path or pairs_list must be provided for CommonDataset.")
    def _init_from_npz(self, npz_path, min_overlap_score, depth_padding, testNpairs):
        npz_path = str(npz_path)
        self.scene_id = Path(npz_path).stem
        if self.mode == 'test' and min_overlap_score > 0:
            logger.warning("You are using `min_overlap_score`!=0 in test mode. Set to 0.")
            min_overlap_score = -3.0
        self.scene_info = np.load(npz_path, allow_pickle=True)
        pair_infos = self.scene_info['pair_infos']
        if self.mode == 'test' and testNpairs:
            pair_infos = pair_infos[:testNpairs]
        self.pair_infos = pair_infos.copy()
        depth_max_size = 4000 if 'MTV_cross_modal_data' not in npz_path else 6000
        self.depth_max_size = depth_max_size if depth_padding else 2000
        dataset_name_meta = None
        if 'dataset_name' in self.scene_info:
            dataset_name_meta = self.scene_info['dataset_name']
            if isinstance(dataset_name_meta, np.ndarray):
                dataset_name_meta = dataset_name_meta.tolist()
            if isinstance(dataset_name_meta, (list, tuple)):
                dataset_name_meta = dataset_name_meta[0]
            if isinstance(dataset_name_meta, bytes):
                dataset_name_meta = dataset_name_meta.decode('utf-8')
        if not self.dataset_name:
            if dataset_name_meta:
                self.dataset_name = str(dataset_name_meta)
            elif self.root_dir is not None:
                try:
                    rel = Path(npz_path).resolve().relative_to(self.root_dir.resolve())
                    self.dataset_name = rel.parts[0] if rel.parts else self.scene_id
                except Exception:
                    self.dataset_name = self.scene_id
            else:
                self.dataset_name = self.scene_id
        self.gt_matches = self.scene_info['gt_matches'] if 'gt_matches' in self.scene_info else None
        self.gt_2D_warp = self.scene_info['gt_2D_transforms'] if 'gt_2D_transforms' in self.scene_info else None
        self.gt_2D_matches = self.scene_info['gt_2D_matches'] if 'gt_2D_matches' in self.scene_info else None
        self.image_metadata = self.scene_info['image_metadata'] if 'image_metadata' in self.scene_info else None
        self.intrins = self.scene_info['intrinsics'] if 'intrinsics' in self.scene_info else None
        self.poses = self.scene_info['poses'] if 'poses' in self.scene_info else None
        
    def _resolve_pair_path(self, path_like):
        path_obj = Path(path_like)
        if path_obj.is_absolute():
            return path_obj
        base = self.root_dir if self.root_dir is not None else Path('.')
        if self.dataset_name:
            return base / self.dataset_name / path_obj
        return base / path_obj
    def _resolve_image_path(self, path_like):
        path_obj = Path(path_like)
        if path_obj.is_absolute():
            return path_obj
        base = self.root_dir if self.root_dir is not None else Path('.')
        if self.dataset_name:
            return base / self.dataset_name / path_obj
        return base / path_obj
    def __len__(self):
        if self._pairs_dataset is not None:
            return len(self._pairs_dataset)
        return len(self.pair_infos)
    def __getitem__(self, idx):
        if self._pairs_dataset is not None:
            return self._pairs_dataset[idx]
        if isinstance(self.pair_infos[idx], np.ndarray):
            idx0, idx1 = self.pair_infos[idx][0], self.pair_infos[idx][1]
            img_path0, img_path1 = self.scene_info['image_paths'][idx0][0], self.scene_info['image_paths'][idx1][1]
            K_0 = torch.zeros((3,3), dtype=torch.float) if self.intrins is None else torch.from_numpy(self.intrins[idx0][0]).float()
            K_1 = torch.zeros((3,3), dtype=torch.float) if self.intrins is None else torch.from_numpy(self.intrins[idx1][1]).float()
        else:
            if len(self.pair_infos[idx]) == 3:
                (idx0, idx1), overlap_score, central_matches = self.pair_infos[idx]
            elif len(self.pair_infos[idx]) == 2:
                (idx0, idx1), overlap_score = self.pair_infos[idx]
            else:
                raise NotImplementedError
            img_path0, img_path1 = self.scene_info['image_paths'][idx0], self.scene_info['image_paths'][idx1]
            img_metadata0, img_metadata1 = self.scene_info['image_metadata'][idx0], self.scene_info['image_metadata'][idx1]
            K_0 = torch.zeros((3,3), dtype=torch.float) if self.intrins is None else torch.from_numpy(self.intrins[idx0]).float()
            K_1 = torch.zeros((3,3), dtype=torch.float) if self.intrins is None else torch.from_numpy(self.intrins[idx1]).float()
        # read grayscale image and mask. (1, h, w) and (h, w)
        img_name0 = self._resolve_image_path(img_path0)
        img_name1 = self._resolve_image_path(img_path1)
        # Note: should be pixel aligned with img0
        image0, mask0, scale0, origin_img_size0 = read_megadepth_gray(
            str(img_name0), self.img_resize, self.df, self.img_padding, None, read_gray=self.read_gray, normalize_img=self.normalize_img, resize_by_stretch=self.resize_by_stretch)
        image1, mask1, scale1, origin_img_size1 = read_megadepth_gray(
            str(img_name1), self.img_resize, self.df, self.img_padding, None, read_gray=self.read_gray, normalize_img=self.normalize_img, resize_by_stretch=self.resize_by_stretch)
        if self.gt_2D_warp is not None:
            gt_warp = np.concatenate([self.gt_2D_warp[idx], [[0,0,1]]]) # 3 * 3
        else:
            gt_warp = np.zeros((3, 3))
        depth0 = depth1 = torch.zeros([self.depth_max_size, self.depth_max_size], dtype=torch.float)
        homo_mask0 = torch.zeros((1, image0.shape[-2], image0.shape[-1]))
        homo_mask1 = torch.zeros((1, image1.shape[-2], image1.shape[-1]))
        gt_matches = torch.zeros((self.gt_matches_padding_n, 4), dtype=torch.float)
        if self.poses is None:
            T_0to1 = T_1to0 = torch.zeros((4,4), dtype=torch.float)  # (4, 4)
        else:
            # read and compute relative poses
            T0 = self.poses[idx0]
            T1 = self.poses[idx1]
            T_0to1 = torch.tensor(np.matmul(T1, np.linalg.inv(T0)), dtype=torch.float)[:4, :4]  # (4, 4)
            T_1to0 = T_0to1.inverse()
        if self.fp16:
            data = {
                'image0': image0.half(),  # (1, h, w)
                'depth0': depth0.half(),  # (h, w)
                'image1': image1.half(),
                'depth1': depth1.half(),
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'homo_mask0': homo_mask0,
                'homo_mask1': homo_mask1,
                'gt_matches': gt_matches,
                'gt_matches_mask': torch.zeros((1,), dtype=torch.bool),
                'homography': torch.from_numpy(gt_warp.astype(np.float32)),
                'norm_pixel_mat': torch.zeros((3,3), dtype=torch.float),
                'homo_sample_normed': torch.zeros((3,3), dtype=torch.float),
                'origin_img_size0': origin_img_size0,
                'origin_img_size1': origin_img_size1,
                'scale0': scale0.half(),  # [scale_w, scale_h]
                'scale1': scale1.half(),
                'dataset_name': self.dataset_name or 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (img_path0, img_path1),
                'image_metadata0': img_metadata0,
                'image_metadata1': img_metadata1
            }
        else:
            data = {
                'image0': image0,  # (1, h, w)
                'depth0': depth0,  # (h, w)
                'image1': image1,
                'depth1': depth1,
                'T_0to1': T_0to1,  # (4, 4)
                'T_1to0': T_1to0,
                'K0': K_0,  # (3, 3)
                'K1': K_1,
                'homo_mask0': homo_mask0,
                'homo_mask1': homo_mask1,
                'homography': torch.from_numpy(gt_warp.astype(np.float32)),
                'norm_pixel_mat': torch.zeros((3,3), dtype=torch.float),
                'homo_sample_normed': torch.zeros((3,3), dtype=torch.float),
                'gt_matches': gt_matches,
                'gt_matches_mask': torch.zeros((1,), dtype=torch.bool),
                'origin_img_size0': origin_img_size0, # H W
                'origin_img_size1': origin_img_size1,
                'scale0': scale0,  # [scale_w, scale_h]
                'scale1': scale1,
                'dataset_name': self.dataset_name or 'MegaDepth',
                'scene_id': self.scene_id,
                'pair_id': idx,
                'pair_names': (img_path0, img_path1),
                'rel_pair_names': (img_path0, img_path1),
                'image_metadata0': img_metadata0,
                'image_metadata1': img_metadata1
            }
        
        if self.gt_2D_matches is not None:
            data.update({'gt_2D_matches': torch.from_numpy(self.gt_2D_matches[idx]).to(torch.float)}) # N * 4
        if self.gt_matches is not None:
            gt_matches_ = self.gt_matches[idx]
            if isinstance(gt_matches_, str):
                gt_matches_ = np.load((Path(self.root_dir) / self.dataset_name / gt_matches_).as_posix(), allow_pickle=True) # removed , self.dataset_name, osp.join(self.root_dir, gt_matches_)
            gt_matches_ = torch.from_numpy(gt_matches_).to(torch.float) # N * 4: mkpts0, mkpts1
            # Warp mkpts1 by sampled homo:
            num = min(len(gt_matches_), self.gt_matches_padding_n)
            gt_matches[:num] = gt_matches_[:num]
            data.update({"gt_matches": gt_matches, 'gt_matches_mask': torch.ones((1,), dtype=torch.bool), 'norm_pixel_mat': torch.zeros((3,3), dtype=torch.float), "homo_sample_normed": torch.zeros((3,3), dtype=torch.float)})
        if mask0 is not None:  # img_padding is True
            if self.coarse_scale:
                if self.fix_bias:
                    [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                            size=((image0.shape[1]-1)//8+1, (image0.shape[2]-1)//8+1),
                                                            mode='nearest',
                                                            recompute_scale_factor=False)[0].bool()
                else:
                    [ts_mask_0, ts_mask_1] = F.interpolate(torch.stack([mask0, mask1], dim=0)[None].float(),
                                                            scale_factor=self.coarse_scale,
                                                            mode='nearest',
                                                            recompute_scale_factor=False)[0].bool()
            if self.fp16:
                data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
            else:
                data.update({'mask0': ts_mask_0, 'mask1': ts_mask_1})
        
        if self.load_origin_rgb:
            rgb0_tensor, rgb0_uint8 = _load_rgb_tensor(Path(img_name0), self.normalize_img)
            rgb1_tensor, rgb1_uint8 = _load_rgb_tensor(Path(img_name1), self.normalize_img)
            data.update({
                "image0_rgb_origin": rgb0_tensor,
                "image1_rgb_origin": rgb1_tensor,
                "image0_rgb_numpy": rgb0_uint8,
                "image1_rgb_numpy": rgb1_uint8,
            })
        return data
