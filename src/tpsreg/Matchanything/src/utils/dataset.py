import io
from loguru import logger

import cv2
import numpy as np
from pathlib import Path
import h5py
import torch
import re
import operator as op
from PIL import Image
from numpy.linalg import inv
from torchvision.transforms import Normalize
from .sample_homo import sample_homography_sap
from kornia.geometry import homography_warp, normalize_homography, normal_transform_pixel
OSS_FOLDER_PATH = '???'
PCACHE_FOLDER_PATH = '???'

import fsspec
from PIL import Image

# Initialize pcache
try:
    PCACHE_HOST = "???"
    PCACHE_PORT = 00000
    pcache_kwargs = {"host": PCACHE_HOST, "port": PCACHE_PORT}
    pcache_fs = fsspec.filesystem("pcache", pcache_kwargs=pcache_kwargs)
    root_dir='???'
except Exception as e:
    logger.error(f"Error captured:{e}")

#try:
    # for internel use only
    #from pcache_fileio import fileio
#except Exception:
#    MEGADEPTH_CLIENT = SCANNET_CLIENT = None

# --- DATA IO ---

def load_pfm(pfm_path):
    with open(pfm_path, 'rb') as fin:
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = str(fin.readline().decode('UTF-8')).rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', fin.readline().decode('UTF-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float((fin.readline().decode('UTF-8')).rstrip())
        if scale < 0:  # little-endian
            data_type = '<f'
        else:
            data_type = '>f'  # big-endian
        data_string = fin.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flip(data, 0)
    return data

def load_array_from_pcache(
    path, cv_type,
    use_h5py=False,
):

    filename = path.split(root_dir)[1]
    pcache_path = Path(root_dir) / filename
    try:
        if not use_h5py:
            load_failed = True
            failed_num = 0
            while load_failed:
                try:
                    with pcache_fs.open(str(pcache_path), 'rb') as f:
                        data = Image.open(f).convert("L")
                        data = np.array(data)
                    load_failed = False
                except:
                    failed_num += 1
                    if failed_num > 5000:
                        logger.error(f"Try to load: {pcache_path}, but failed {failed_num} times")
                    continue
        else:
            load_failed = True
            failed_num = 0
            while load_failed:
                try:
                    with pcache_fs.open(str(pcache_path), 'rb') as f:
                        data = np.array(h5py.File(io.BytesIO(f.read()), 'r')['/depth'])
                    load_failed = False
                except:
                    failed_num += 1
                    if failed_num > 5000:
                        logger.error(f"Try to load: {pcache_path}, but failed {failed_num} times")
                    continue

    except Exception as ex:
        print(f"==> Data loading failure: {path}")
        raise ex

    assert data is not None
    return data


def imread_gray(path, augment_fn=None, cv_type=None):
    if path.startswith('oss://'):
        path = path.replace(OSS_FOLDER_PATH, PCACHE_FOLDER_PATH)
    if path.startswith('pcache://'):
        path = path[:9] + path[9:].replace('////', '/').replace('///', '/').replace('//', '/') # remove all continuous '/'

    if cv_type is None:
        cv_type = cv2.IMREAD_GRAYSCALE if augment_fn is None \
                else cv2.IMREAD_COLOR
        # for Exx strain maps which are 32bit float images, use IMREAD_UNCHANGED is needed
        if op.contains(path, 'Exx') or op.contains(path, 'FeatureIDs'):
            cv_type = cv2.IMREAD_UNCHANGED
    if str(path).startswith('oss://') or str(path).startswith('pcache://'):
        image = load_array_from_pcache(str(path), cv_type)
    else:
        image = cv2.imread(str(path), cv_type)

    if augment_fn is not None:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = augment_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return image  # (h, w)

def imread_color(path, augment_fn=None):
    if path.startswith('oss://'):
        path = path.replace(OSS_FOLDER_PATH, PCACHE_FOLDER_PATH)
    if path.startswith('pcache://'):
        path = path[:9] + path[9:].replace('////', '/').replace('///', '/').replace('//', '/') # remove all continuous '/'

    if str(path).startswith('oss://') or str(path).startswith('pcache://'):
        filename = path.split(root_dir)[1]
        pcache_path = Path(root_dir) / filename
        load_failed = True
        failed_num = 0
        while load_failed:
            try:
                with pcache_fs.open(str(pcache_path), 'rb') as f:
                    pil_image = Image.open(f).convert("RGB")
                    load_failed = False
            except:
                failed_num += 1
                if failed_num > 5000:
                    logger.error(f"Try to load: {pcache_path}, but failed {failed_num} times")
                continue
    else:
        pil_image = Image.open(str(path)).convert("RGB")
    image = np.array(pil_image)

    if augment_fn is not None:
        image = augment_fn(image)
    return image  # (h, w)


def get_resized_wh(w, h, resize=None):
    if resize is not None:  # resize the longer edge
        scale = resize / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    else:
        w_new, h_new = w, h
    return w_new, h_new


def get_divisible_wh(w, h, df=None):
    if df is not None:
        w_new, h_new = map(lambda x: int(x // df * df), [w, h])
    else:
        w_new, h_new = w, h
    return w_new, h_new


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((inp.shape[0], pad_size, pad_size), dtype=inp.dtype)
        padded[:, :inp.shape[1], :inp.shape[2]] = inp
        if ret_mask:
            mask = np.zeros((inp.shape[0], pad_size, pad_size), dtype=bool)
            mask[:, :inp.shape[1], :inp.shape[2]] = True
        mask = mask[0]
    else:
        raise NotImplementedError()
    return padded, mask


# --- MEGADEPTH ---

def read_megadepth_gray(path, resize=None, df=None, padding=False, augment_fn=None, read_gray=True, normalize_img=False, resize_by_stretch=False):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    if read_gray:
        image = imread_gray(path, augment_fn)
    else:
        image = imread_color(path, augment_fn)

    # resize image
    try:
        w, h = image.shape[1], image.shape[0]
    except:
        logger.error(f"{path} not exist or read image error!")
    if resize_by_stretch:
        w_new, h_new = (resize, resize) if isinstance(resize, int) else (resize[1], resize[0])
    else:
        if resize:
            if not isinstance(resize, int): 
                assert resize[0] == resize[1]
                resize = resize[0]
            w_new, h_new = get_resized_wh(w, h, resize)
            w_new, h_new = get_divisible_wh(w_new, h_new, df)
        else:
            w_new, h_new = w, h

    # if the DIC_EBSD_Multimodal_In718 dataset is used, the int32 has to be cast to float32 to be resized
    if op.contains(path, 'FeatureIDs'):
        image = image.astype(np.float32)
    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)
    origin_img_size = torch.tensor([h, w], dtype=torch.float)

    if not read_gray:
        image = image.transpose(2,0,1)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    if len(image.shape) == 2:
        image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    else:
        image = torch.from_numpy(image).float() / 255  # (h, w) -> (1, h, w) and normalized
    if mask is not None:
        mask = torch.from_numpy(mask)
    
    if image.shape[0] == 3 and normalize_img:
        # Normalize image:
        image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image) # Input: 3*H*W

    return image, mask, scale, origin_img_size

def read_megadepth_gray_sample_homowarp(path, resize=None, df=None, padding=False, augment_fn=None, read_gray=True, normalize_img=False, resize_by_stretch=False):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read image
    if read_gray:
        image = imread_gray(path, augment_fn)
    else:
        image = imread_color(path, augment_fn)

    # resize image
    w, h = image.shape[1], image.shape[0]
    if resize_by_stretch:
        w_new, h_new = (resize, resize) if isinstance(resize, int) else (resize[1], resize[0])
    else:
        if not isinstance(resize, int): 
            assert resize[0] == resize[1]
            resize = resize[0]
        w_new, h_new = get_resized_wh(w, h, resize)
        w_new, h_new = get_divisible_wh(w_new, h_new, df)

    w_new, h_new = get_divisible_wh(w_new, h_new, df)
    
    origin_img_size = torch.tensor([h, w], dtype=torch.float)

    # Sample homography and warp:
    homo_sampled = sample_homography_sap(h, w) # 3*3
    homo_sampled_normed = normalize_homography(
        torch.from_numpy(homo_sampled[None]).to(torch.float32),
        (h, w),
        (h, w),
    )

    if len(image.shape) == 2:
        image = torch.from_numpy(image).float()[None, None] / 255 # B * C * H * W
    else:
        image = torch.from_numpy(image).float().permute(2,0,1)[None] / 255

    homo_warpped_image = homography_warp(
        image, # 1 * C * H * W
        torch.linalg.inv(homo_sampled_normed),
        (h, w),
    )
    image = (homo_warpped_image[0].permute(1,2,0).numpy() * 255).astype(np.uint8)
    norm_pixel_mat = normal_transform_pixel(h, w) # 1 * 3 * 3

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if not read_gray:
        image = image.transpose(2,0,1)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    if len(image.shape) == 2:
        image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    else:
        image = torch.from_numpy(image).float() / 255  # (h, w) -> (1, h, w) and normalized
    if mask is not None:
        mask = torch.from_numpy(mask)

    if image.shape[0] == 3 and normalize_img:
        # Normalize image:
        image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image) # Input: 3*H*W

    return image, mask, scale, origin_img_size, norm_pixel_mat[0], homo_sampled_normed[0]


def read_megadepth_depth_gray(path, resize=None, df=None, padding=False, augment_fn=None, read_gray=True, normalize_img=False, resize_by_stretch=False):
    """
    Args:
        resize (int, optional): the longer edge of resized images. None for no resize.
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    depth = read_megadepth_depth(path, return_tensor=False)

    # following controlnet  1-depth
    depth = depth.astype(np.float64)
    depth_non_zero = depth[depth!=0]
    vmin = np.percentile(depth_non_zero, 2)
    vmax = np.percentile(depth_non_zero, 85)
    depth -= vmin
    depth /= (vmax - vmin + 1e-4)
    depth = 1.0 - depth
    image = (depth * 255.0).clip(0, 255).astype(np.uint8)

    # resize image
    w, h = image.shape[1], image.shape[0]
    if resize_by_stretch:
        w_new, h_new = (resize, resize) if isinstance(resize, int) else (resize[1], resize[0])
    else:
        if not isinstance(resize, int): 
            assert resize[0] == resize[1]
            resize = resize[0]
        w_new, h_new = get_resized_wh(w, h, resize)
        w_new, h_new = get_divisible_wh(w_new, h_new, df)
    w_new, h_new = get_divisible_wh(w_new, h_new, df)
    origin_img_size = torch.tensor([h, w], dtype=torch.float)

    image = cv2.resize(image, (w_new, h_new))
    scale = torch.tensor([w/w_new, h/h_new], dtype=torch.float)

    if padding:  # padding
        pad_to = max(h_new, w_new)
        image, mask = pad_bottom_right(image, pad_to, ret_mask=True)
    else:
        mask = None

    if read_gray:
        image = torch.from_numpy(image).float()[None] / 255  # (h, w) -> (1, h, w) and normalized
    else:
        image = np.stack([image]*3) # 3 * H * W
        image = torch.from_numpy(image).float() / 255  # (h, w) -> (1, h, w) and normalized
    if mask is not None:
        mask = torch.from_numpy(mask)

    if image.shape[0] == 3 and normalize_img:
        # Normalize image:
        image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image) # Input: 3*H*W

    return image, mask, scale, origin_img_size

def read_megadepth_depth(path, pad_to=None, return_tensor=True):
    if path.startswith('oss://'):
        path = path.replace(OSS_FOLDER_PATH, PCACHE_FOLDER_PATH)
    if path.startswith('pcache://'):
        path = path[:9] + path[9:].replace('////', '/').replace('///', '/').replace('//', '/') # remove all continuous '/'

    load_failed = True
    failed_num = 0
    while load_failed:
        try:
            if '.png' in path:
                if 'scannet_plus' in path:
                    depth = imread_gray(path, cv_type=cv2.IMREAD_UNCHANGED).astype(np.float32)

                    with open(path, 'rb') as f:
                        # CO3D
                        depth = np.asarray(Image.open(f)).astype(np.float32)
                    depth = depth / 1000
            elif '.pfm' in path:
                # For BlendedMVS dataset (not support pcache):
                depth = load_pfm(path).copy()
            else:
                # For MegaDepth
                if str(path).startswith('oss://') or str(path).startswith('pcache://'):
                    depth = load_array_from_pcache(path, None, use_h5py=True)
                else:
                    depth = np.array(h5py.File(path, 'r')['depth'])
            load_failed = False
        except:
            failed_num += 1
            if failed_num > 5000:
                logger.error(f"Try to load: {path}, but failed {failed_num} times")
            continue

    if pad_to is not None:
        depth, _ = pad_bottom_right(depth, pad_to, ret_mask=False)
    if return_tensor:
        depth = torch.from_numpy(depth).float()  # (h, w)
    return depth


# --- ScanNet ---

def read_scannet_gray(path, resize=(640, 480), augment_fn=None):
    """
    Args:
        resize (tuple): align image to depthmap, in (w, h).
        augment_fn (callable, optional): augments images with pre-defined visual effects
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]        
    """
    # read and resize image
    image = imread_gray(path, augment_fn)
    image = cv2.resize(image, resize)

    # (h, w) -> (1, h, w) and normalized
    image = torch.from_numpy(image).float()[None] / 255
    return image


# def read_scannet_depth(path):
#     if str(path).startswith('s3://'):
#         depth = load_array_from_s3(str(path), SCANNET_CLIENT, cv2.IMREAD_UNCHANGED)
#     else:
#         depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
#     depth = depth / 1000
#     depth = torch.from_numpy(depth).float()  # (h, w)
#     return depth


def read_scannet_pose(path):
    """ Read ScanNet's Camera2World pose and transform it to World2Camera.
    
    Returns:
        pose_w2c (np.ndarray): (4, 4)
    """
    cam2world = np.loadtxt(path, delimiter=' ')
    world2cam = inv(cam2world)
    return world2cam


def read_scannet_intrinsic(path):
    """ Read ScanNet's intrinsic matrix and return the 3x3 matrix.
    """
    intrinsic = np.loadtxt(path, delimiter=' ')
    return intrinsic[:-1, :-1]

def dict_to_cuda(data_dict):
    data_dict_cuda = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict_cuda[k] = v.cuda()
        elif isinstance(v, dict):
            data_dict_cuda[k] = dict_to_cuda(v)
        elif isinstance(v, list):
            data_dict_cuda[k] = list_to_cuda(v)
        else:
            data_dict_cuda[k] = v
    return data_dict_cuda

def list_to_cuda(data_list):
    data_list_cuda = []
    for obj in data_list:
        if isinstance(obj, torch.Tensor):
            data_list_cuda.append(obj.cuda())
        elif isinstance(obj, dict):
            data_list_cuda.append(dict_to_cuda(obj))
        elif isinstance(obj, list):
            data_list_cuda.append(list_to_cuda(obj))
        else:
            data_list_cuda.append(obj)
    return data_list_cuda