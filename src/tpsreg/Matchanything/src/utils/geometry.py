from __future__ import division
import torch
import torch.nn.functional as F
import numpy as np
# from numba import jit

pixel_coords = None

def set_id_grid(depth):
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = torch.ones(1,h,w).type_as(depth)

    pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
    return pixel_coords

def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i,size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert(all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def pixel2cam(depth, intrinsics_inv):
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    pixel_coords = set_id_grid(depth)
    current_pixel_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w).reshape(b, 3, -1)  # [B, 3, H*W]
    cam_coords = (intrinsics_inv.float() @ current_pixel_coords.float()).reshape(b, 3, h, w)
    return cam_coords * depth.unsqueeze(1)

def cam2pixel_depth(cam_coords, proj_c2p_rot, proj_c2p_tr):
    """Transform coordinates in the camera frame to the pixel frame and get depth map.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        tensor of [-1,1] coordinates -- [B, 2, H, W]
        depth map -- [B, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)  # [B, H*W] min_depth = 1 mm

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b,h,w,2), Z.reshape(b, h, w)
    

def cam2pixel(cam_coords, proj_c2p_rot, proj_c2p_tr):
    """Transform coordinates in the camera frame to the pixel frame.
    Args:
        cam_coords: pixel coordinates defined in the first camera coordinates system -- [B, 3, H, W]
        proj_c2p_rot: rotation matrix of cameras -- [B, 3, 4]
        proj_c2p_tr: translation vectors of cameras -- [B, 3, 1]
    Returns:
        array of [-1,1] coordinates -- [B, 2, H, W]
    """
    b, _, h, w = cam_coords.size()
    cam_coords_flat = cam_coords.reshape(b, 3, -1)  # [B, 3, H*W]
    if proj_c2p_rot is not None:
        pcoords = proj_c2p_rot @ cam_coords_flat
    else:
        pcoords = cam_coords_flat

    if proj_c2p_tr is not None:
        pcoords = pcoords + proj_c2p_tr  # [B, 3, H*W]
    X = pcoords[:, 0]
    Y = pcoords[:, 1]
    Z = pcoords[:, 2].clamp(min=1e-3)  # [B, H*W] min_depth = 1 mm

    X_norm = 2*(X / Z)/(w-1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
    Y_norm = 2*(Y / Z)/(h-1) - 1  # Idem [B, H*W]

    pixel_coords = torch.stack([X_norm, Y_norm], dim=2)  # [B, H*W, 2]
    return pixel_coords.reshape(b,h,w,2)


def reproject_kpts(dim0_idxs, kpts, depth, rel_pose, K0, K1):
    """ Reproject keypoints with depth, relative pose and camera intrinsics
    Args:
        dim0_idxs (torch.LoneTensor): (B*max_kpts, )
        kpts (torch.LongTensor): (B, max_kpts, 2) - <x,y>
        depth (torch.Tensor): (B, H, W)
        rel_pose (torch.Tensor): (B, 3, 4) relative transfomation from target to source (T_0to1) -- 
        K0: (torch.Tensor): (N, 3, 3) - (K_0)
        K1: (torch.Tensor): (N, 3, 3) - (K_1)    
    Returns:
        (torch.Tensor): (B, max_kpts, 2) the reprojected kpts
    """
    # pixel to camera
    device = kpts.device
    B, max_kpts, _ = kpts.shape

    kpts = kpts.reshape(-1, 2)  # (B*K, 2)
    kpts_depth = depth[dim0_idxs, kpts[:, 1], kpts[:, 0]]  # (B*K, )
    kpts = torch.cat([kpts.float(), 
                    torch.ones((kpts.shape[0], 1), dtype=torch.float32, device=device)], -1)  # (B*K, 3)
    pixel_coords = (kpts * kpts_depth[:, None]).reshape(B, max_kpts, 3).permute(0, 2, 1)  # (B, 3, K)

    cam_coords = K0.inverse() @ pixel_coords  # (N, 3, max_kpts)
    # camera1 to camera 2
    rel_pose_R = rel_pose[:, :, :-1]  # (B, 3, 3)
    rel_pose_t = rel_pose[:, :, -1][..., None]  # (B, 3, 1)
    cam2_coords = rel_pose_R @ cam_coords + rel_pose_t  # (B, 3, max_kpts)
    # projection
    pixel2_coords = K1 @ cam2_coords  # (B, 3, max_kpts)
    reproj_kpts = pixel2_coords[:, :-1, :] / pixel2_coords[:, -1, :][:, None].expand(-1, 2, -1)
    return reproj_kpts.permute(0, 2, 1)


def check_depth_consistency(b_idxs, kpts0, depth0, kpts1, depth1, T_0to1, K0, K1, 
                            atol=0.1, rtol=0.0):
    """
    Args:
        b_idxs (torch.LongTensor): (n_kpts, ) the batch indices which each keypoints pairs belong to
        kpts0 (torch.LongTensor): (n_kpts, 2) - <x, y>
        depth0 (torch.Tensor): (B, H, W)
        kpts1 (torch.LongTensor): (n_kpts, 2)
        depth1 (torch.Tensor): (B, H, W)
        T_0to1 (torch.Tensor): (B, 3, 4)
        K0: (torch.Tensor): (N, 3, 3) - (K_0)
        K1: (torch.Tensor): (N, 3, 3) - (K_1)    
        atol (float): the absolute tolerance for depth consistency check
        rtol (float): the relative tolerance for depth consistency check
    Returns:
        valid_mask (torch.Tensor): (n_kpts, )
    Notes:
        The two corresponding keypoints are depth consistent if the following equation is held:
            abs(kpt_0to1_depth - kpt1_depth) <= (atol + rtol * abs(kpt1_depth))
        * In the initial reimplementation, `atol=0.1, rtol=0` is used, and the result is better with 
          `atol=1.0, rtol=0` (which nearly ignore the depth consistency check).
        * However, the author suggests using `atol=0.0, rtol=0.1` as in https://github.com/magicleap/SuperGluePretrainedNetwork/issues/31#issuecomment-681866054
    """
    device = kpts0.device
    n_kpts = kpts0.shape[0]
    
    kpts0_depth = depth0[b_idxs, kpts0[:, 1], kpts0[:, 0]]  # (n_kpts, )
    kpts1_depth = depth1[b_idxs, kpts1[:, 1], kpts1[:, 0]]  # (n_kpts, )
    kpts0 = torch.cat([kpts0.float(),
                      torch.ones((n_kpts, 1), dtype=torch.float32, device=device)], -1)  # (n_kpts, 3)
    pixel_coords = (kpts0 * kpts0_depth[:, None])[..., None]  # (n_kpts, 3, 1)
    
    # indexing from T_0to1 and K - treat all kpts as a batch
    K0 = K0[b_idxs, :, :]  # (n_kpts, 3, 3)
    T_0to1 = T_0to1[b_idxs, :, :]  # (n_kpts, 3, 4)
    cam_coords = K0.inverse() @ pixel_coords  # (n_kpts, 3, 1)
    
    # camera1 to camera2
    R_0to1 = T_0to1[:, :, :-1]  # (n_kpts, 3, 3)
    t_0to1 = T_0to1[:, :, -1][..., None]  # (n_kpts, 3, 1)
    cam1_coords = R_0to1 @ cam_coords + t_0to1  # (n_kpts, 3, 1)
    K1 = K1[b_idxs, :, :]  # (n_kpts, 3, 3)
    pixel1_coords = K1 @ cam1_coords  # (n_kpts, 3, 1)
    kpts_0to1_depth  = pixel1_coords[:, -1, 0]  # (n_kpts, )
    return (kpts_0to1_depth - kpts1_depth).abs() <= atol + rtol * kpts1_depth.abs()


def inverse_warp(img, depth, pose, intrinsics, mode='bilinear', padding_mode='zeros'):
    """
    Inverse warp a source image to the target image plane.

    Args:
        img: the source image (where to sample pixels) -- [B, 3, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: relative transfomation from target to source -- [B, 3, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        projected_img: Source image warped to the target image plane
        valid_points: Boolean array indicating point validity
    """
    # check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'BHW')
#     check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')

    batch_size, _, img_height, img_width = img.size()

    cam_coords = pixel2cam(depth, intrinsics.inverse())  # [B,3,H,W]

    pose_mat = pose  # (B, 3, 4)

    # Get projection matrix for target camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsics @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:]
    src_pixel_coords = cam2pixel(cam_coords, rot, tr)  # [B,H,W,2]
    projected_img = F.grid_sample(img, src_pixel_coords, mode=mode, 
                                  padding_mode=padding_mode, align_corners=True)

    valid_points = src_pixel_coords.abs().max(dim=-1)[0] <= 1

    return projected_img, valid_points

def depth_inverse_warp(depth_source, depth, pose, intrinsic_source, intrinsic, mode='nearest', padding_mode='zeros'):
    """
    1. Inversely warp a source depth map to the target image plane (warped depth map still in source frame)
    2. Transform the target depth map to the source image frame
    Args:
        depth_source: the source image (where to sample pixels) -- [B, H, W]
        depth: depth map of the target image -- [B, H, W]
        pose: relative transfomation from target to source -- [B, 3, 4]
        intrinsics: camera intrinsic matrix -- [B, 3, 3]
    Returns:
        warped_depth: Source depth warped to the target image plane -- [B, H, W]
        projected_depth: Target depth projected to the source image frame -- [B, H, W]
        valid_points: Boolean array indicating point validity -- [B, H, W]
    """
    check_sizes(depth_source, 'depth', 'BHW')
    check_sizes(depth, 'depth', 'BHW')
    check_sizes(intrinsic_source, 'intrinsics', 'B33')

    b, h, w = depth.size()

    cam_coords = pixel2cam(depth, intrinsic.inverse())  # [B,3,H,W]

    pose_mat = pose  # (B, 3, 4)

    # Get projection matrix from target camera frame to source pixel frame
    proj_cam_to_src_pixel = intrinsic_source @ pose_mat  # [B, 3, 4]

    rot, tr = proj_cam_to_src_pixel[:,:,:3], proj_cam_to_src_pixel[:,:,-1:]
    src_pixel_coords, depth_target2src = cam2pixel_depth(cam_coords, rot, tr)  # [B,H,W,2]
    warped_depth = F.grid_sample(depth_source[:, None], src_pixel_coords, mode=mode, 
                                 padding_mode=padding_mode, align_corners=True)  # [B, 1, H, W]

    valid_points = (src_pixel_coords.abs().max(dim=-1)[0] <= 1) &\
                    (depth > 0.0) & (warped_depth[:, 0] > 0.0) # [B, H, W]
    return warped_depth[:, 0], depth_target2src, valid_points

def to_skew(t):
    """ Transform the translation vector t to skew-symmetric matrix.
    Args:
        t (torch.Tensor): (B, 3)
    """
    t_skew = t.new_ones((t.shape[0], 3, 3))
    t_skew[:, 0, 1] = -t[:, 2]
    t_skew[:, 1, 0] = t[:, 2]
    t_skew[:, 0, 2] = t[:, 1]
    t_skew[:, 2, 0] = -t[:, 1]
    t_skew[:, 1, 2] = -t[:, 0]
    t_skew[:, 2, 1] = t[:, 0]
    return t_skew  # (B, 3, 3)


def to_homogeneous(pts):
    """
    Args:
        pts (torch.Tensor): (B, K, 2)
    """
    return torch.cat([pts, torch.ones_like(pts[..., :1])], -1)  # (B, K, 3)


def pix2img(pts, K):
    """
    Args:
        pts (torch.Tensor): (B, K, 2)
        K (torch.Tensor): (B, 3, 3)
    """
    return (pts - K[:, [0, 1], [2, 2]][:, None]) / K[:, [0, 1], [0, 1]][:, None]


def weighted_blind_sed(kpts0, kpts1, weights, E, K0, K1):
    """ Calculate the squared weighted blind symmetric epipolar distance, which is the sed between 
    all possible keypoints pairs.
    Args:
        kpts0 (torch.Tensor): (B, K0, 2)
        ktps1 (torch.Tensor): (B, K1, 2)
        weights (torch.Tensor): (B, K0, K1)
        E (torch.Tensor): (B, 3, 3) - the essential matrix
        K0 (torch.Tensor): (B, 3, 3)
        K1 (torch.Tensor): (B, 3, 3)
    Returns:
        w_sed (torch.Tensor): (B, K0, K1)
    """
    M, N = kpts0.shape[1], kpts1.shape[1]
    
    kpts0 = to_homogeneous(pix2img(kpts0, K0))
    kpts1 = to_homogeneous(pix2img(kpts1, K1))  # (B, K1, 3)
    
    R = kpts0 @ E.transpose(1, 2) @ kpts1.transpose(1, 2)  # (B, K0, K1)
    # w_R = weights * R  # (B, K0, K1)
    
    Ep0 = kpts0 @ E.transpose(1, 2)  # (B, K0, 3)
    Etp1 = kpts1 @ E  # (B, K1, 3)
    d = R**2 * (1.0 / (Ep0[..., 0]**2 + Ep0[..., 1]**2)[..., None].expand(-1, -1, N)
              + 1.0 / (Etp1[..., 0]**2 + Etp1[..., 1]**2)[:, None].expand(-1, M, -1)) * weights  # (B, K0, K1)
    return d

def weighted_blind_sampson(kpts0, kpts1, weights, E, K0, K1):
    """ Calculate the squared weighted blind sampson distance, which is the sampson distance between 
    all possible keypoints pairs weighted by the given weights.
    """
    M, N = kpts0.shape[1], kpts1.shape[1]
    
    kpts0 = to_homogeneous(pix2img(kpts0, K0))
    kpts1 = to_homogeneous(pix2img(kpts1, K1))  # (B, K1, 3)
    
    R = kpts0 @ E.transpose(1, 2) @ kpts1.transpose(1, 2)  # (B, K0, K1)
    # w_R = weights * R  # (B, K0, K1)
    
    Ep0 = kpts0 @ E.transpose(1, 2)  # (B, K0, 3)
    Etp1 = kpts1 @ E  # (B, K1, 3)
    d = R**2 * (1.0 / ((Ep0[..., 0]**2 + Ep0[..., 1]**2)[..., None].expand(-1, -1, N)
                     + (Etp1[..., 0]**2 + Etp1[..., 1]**2)[:, None].expand(-1, M, -1))) * weights  # (B, K0, K1)
    return d


def angular_rel_rot(T_0to1):
    """
    Args:
        T0_to_1 (np.ndarray): (4, 4)
    """
    cos = (np.trace(T_0to1[:-1, :-1]) - 1) / 2
    if cos < -1:
        cos = -1.0
    if cos > 1:
        cos = 1.0
    angle_error_rot = np.rad2deg(np.abs(np.arccos(cos)))
    
    return angle_error_rot

def angular_rel_pose(T0, T1):
    """
    Args:
        T0 (np.ndarray): (4, 4)
        T1 (np.ndarray): (4, 4)
        
    """
    cos = (np.trace(T0[:-1, :-1].T @ T1[:-1, :-1]) - 1) / 2
    if cos < -1:
        cos = -1.0
    if cos > 1:
        cos = 1.0
    angle_error_rot = np.rad2deg(np.abs(np.arccos(cos)))
    
    # calculate angular translation error
    n = np.linalg.norm(T0[:-1, -1]) * np.linalg.norm(T1[:-1, -1])
    cos = np.dot(T0[:-1, -1], T1[:-1, -1]) / n
    if cos < -1:
        cos = -1.0
    if cos > 1:
        cos = 1.0
    angle_error_trans = np.rad2deg(np.arccos(cos))
    
    return angle_error_rot, angle_error_trans