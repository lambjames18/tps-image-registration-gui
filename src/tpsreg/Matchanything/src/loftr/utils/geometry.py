import torch
from Matchanything.src.utils.homography_utils import warp_points_torch


def get_unique_indices(input_tensor):
    if input_tensor.shape[0] > 1:
        unique, inverse = torch.unique(
            input_tensor, sorted=True, return_inverse=True, dim=0
        )
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    else:
        perm = torch.zeros(
            (input_tensor.shape[0],), dtype=torch.long, device=input_tensor.device
        )
    return perm


@torch.no_grad()
def warp_kpts(
    kpts0,
    depth0,
    depth1,
    T_0to1,
    K0,
    K1,
    consistency_thr=0.2,
    cycle_proj_distance_thr=3.0,
):
    """Warp kpts0 from I0 to I1 with depth, K and Rt
    Also check covisibility and depth consistency.
    Depth is consistent if relative error < 0.2 (hard-coded).

    Args:
        kpts0 (torch.Tensor): [N, L, 2] - <x, y>,
        depth0 (torch.Tensor): [N, H, W],
        depth1 (torch.Tensor): [N, H, W],
        T_0to1 (torch.Tensor): [N, 3, 4],
        K0 (torch.Tensor): [N, 3, 3],
        K1 (torch.Tensor): [N, 3, 3],
    Returns:
        calculable_mask (torch.Tensor): [N, L]
        warped_keypoints0 (torch.Tensor): [N, L, 2] <x0_hat, y1_hat>
    """
    kpts0_long = kpts0.round().long()

    # Sample depth, get calculable_mask on depth != 0
    kpts0_depth = torch.stack(
        [
            depth0[i, kpts0_long[i, :, 1], kpts0_long[i, :, 0]]
            for i in range(kpts0.shape[0])
        ],
        dim=0,
    )  # (N, L)
    nonzero_mask = kpts0_depth != 0

    # Unproject
    kpts0_h = (
        torch.cat([kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1)
        * kpts0_depth[..., None]
    )  # (N, L, 3)
    kpts0_cam = K0.inverse() @ kpts0_h.transpose(2, 1)  # (N, 3, L)

    # Rigid Transform
    w_kpts0_cam = T_0to1[:, :3, :3] @ kpts0_cam + T_0to1[:, :3, [3]]  # (N, 3, L)
    w_kpts0_depth_computed = w_kpts0_cam[:, 2, :]

    # Project
    w_kpts0_h = (K1 @ w_kpts0_cam).transpose(2, 1)  # (N, L, 3)
    w_kpts0 = w_kpts0_h[:, :, :2] / (
        w_kpts0_h[:, :, [2]] + 1e-4
    )  # (N, L, 2), +1e-4 to avoid zero depth

    # Covisible Check
    h, w = depth1.shape[1:3]
    covisible_mask = (
        (w_kpts0[:, :, 0] > 0)
        * (w_kpts0[:, :, 0] < w - 1)
        * (w_kpts0[:, :, 1] > 0)
        * (w_kpts0[:, :, 1] < h - 1)
    )
    w_kpts0_long = w_kpts0.long()
    w_kpts0_long[~covisible_mask, :] = 0

    w_kpts0_depth = torch.stack(
        [
            depth1[i, w_kpts0_long[i, :, 1], w_kpts0_long[i, :, 0]]
            for i in range(w_kpts0_long.shape[0])
        ],
        dim=0,
    )  # (N, L)
    consistent_mask = (
        (w_kpts0_depth - w_kpts0_depth_computed) / w_kpts0_depth
    ).abs() < consistency_thr

    # Cycle Consistency Check
    dst_pts_h = (
        torch.cat(
            [w_kpts0, torch.ones_like(w_kpts0[..., [0]], device=w_kpts0.device)], dim=-1
        )
        * w_kpts0_depth[..., None]
    )  # B * N_dst * N_pts * 3
    dst_pts_cam = K1.inverse() @ dst_pts_h.transpose(2, 1)  # (N, 3, L)
    dst_pose = T_0to1.inverse()
    world_points_cycle_back = dst_pose[:, :3, :3] @ dst_pts_cam + dst_pose[:, :3, [3]]
    src_warp_back_h = (K0 @ world_points_cycle_back).transpose(2, 1)  # (N, L, 3)
    src_back_proj_pts = src_warp_back_h[..., :2] / (src_warp_back_h[..., [2]] + 1e-4)
    cycle_reproj_distance_mask = (
        torch.linalg.norm(src_back_proj_pts - kpts0[:, None], dim=-1)
        < cycle_proj_distance_thr
    )

    valid_mask = (
        nonzero_mask * covisible_mask * consistent_mask * cycle_reproj_distance_mask
    )

    return valid_mask, w_kpts0


@torch.no_grad()
def warp_kpts_by_sparse_gt_matches_batches(kpts0, gt_matches, dist_thr):
    B, n_pts = kpts0.shape[0], kpts0.shape[1]
    if n_pts > 20 * 10000:
        all_kpts_valid_mask, all_kpts_warpped = [], []
        for b_id in range(B):
            kpts_valid_mask, kpts_warpped = warp_kpts_by_sparse_gt_matches(
                kpts0[[b_id]], gt_matches[[b_id]], dist_thr[[b_id]]
            )
            all_kpts_valid_mask.append(kpts_valid_mask)
            all_kpts_warpped.append(kpts_warpped)
        return torch.cat(all_kpts_valid_mask, dim=0), torch.cat(all_kpts_warpped, dim=0)
    else:
        return warp_kpts_by_sparse_gt_matches(kpts0, gt_matches, dist_thr)


@torch.no_grad()
def warp_kpts_by_sparse_gt_matches(kpts0, gt_matches, dist_thr):
    kpts_warpped = torch.zeros_like(kpts0)
    kpts_valid_mask = torch.zeros_like(kpts0[..., 0], dtype=torch.bool)
    gt_matches_non_padding_mask = gt_matches.sum(-1) > 0

    dist_matrix = torch.cdist(kpts0, gt_matches[..., :2])  # B * N * M
    if dist_thr is not None:
        mask = dist_matrix < dist_thr[:, None, None]
    else:
        mask = torch.ones_like(dist_matrix, dtype=torch.bool)
    # Mutual-Nearest check:
    mask = (
        mask
        * (dist_matrix == dist_matrix.min(dim=2, keepdim=True)[0])
        * (dist_matrix == dist_matrix.min(dim=1, keepdim=True)[0])
    )

    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]

    j_uq_indices = get_unique_indices(torch.stack([b_ids, j_ids], dim=-1))
    b_ids, i_ids, j_ids = map(lambda x: x[j_uq_indices], [b_ids, i_ids, j_ids])

    i_uq_indices = get_unique_indices(torch.stack([b_ids, i_ids], dim=-1))
    b_ids, i_ids, j_ids = map(lambda x: x[i_uq_indices], [b_ids, i_ids, j_ids])

    kpts_valid_mask[b_ids, i_ids] = gt_matches_non_padding_mask[b_ids, j_ids]
    kpts_warpped[b_ids, i_ids] = gt_matches[..., 2:][b_ids, j_ids]

    return kpts_valid_mask, kpts_warpped


@torch.no_grad()
def warp_kpts_by_sparse_gt_matches_fine_chunks(kpts0, gt_matches, dist_thr):
    B, n_pts = kpts0.shape[0], kpts0.shape[1]
    chunk_n = 500
    all_kpts_valid_mask, all_kpts_warpped = [], []
    for b_id in range(0, B, chunk_n):
        kpts_valid_mask, kpts_warpped = warp_kpts_by_sparse_gt_matches_fine(
            kpts0[b_id : b_id + chunk_n], gt_matches, dist_thr
        )
        all_kpts_valid_mask.append(kpts_valid_mask)
        all_kpts_warpped.append(kpts_warpped)
    return torch.cat(all_kpts_valid_mask, dim=0), torch.cat(all_kpts_warpped, dim=0)


@torch.no_grad()
def warp_kpts_by_sparse_gt_matches_fine(kpts0, gt_matches, dist_thr):
    """
    Only support single batch
    Input:
    kpts0: N * ww * 2
    gt_matches: M * 2
    """
    B = kpts0.shape[0]  # B is the fine matches in a single pair
    assert gt_matches.shape[0] == 1
    kpts_warpped = torch.zeros_like(kpts0)
    kpts_valid_mask = torch.zeros_like(kpts0[..., 0], dtype=torch.bool)
    gt_matches_non_padding_mask = gt_matches.sum(-1) > 0

    dist_matrix = torch.cdist(kpts0, gt_matches[..., :2])  # B * N * M
    if dist_thr is not None:
        mask = dist_matrix < dist_thr[:, None, None]
    else:
        mask = torch.ones_like(dist_matrix, dtype=torch.bool)
    # Mutual-Nearest check:
    mask = (
        mask
        * (dist_matrix == dist_matrix.min(dim=2, keepdim=True)[0])
        * (dist_matrix == dist_matrix.min(dim=1, keepdim=True)[0])
    )

    mask_v, all_j_ids = mask.max(dim=2)
    b_ids, i_ids = torch.where(mask_v)
    j_ids = all_j_ids[b_ids, i_ids]

    j_uq_indices = get_unique_indices(torch.stack([b_ids, j_ids], dim=-1))
    b_ids, i_ids, j_ids = map(lambda x: x[j_uq_indices], [b_ids, i_ids, j_ids])

    i_uq_indices = get_unique_indices(torch.stack([b_ids, i_ids], dim=-1))
    b_ids, i_ids, j_ids = map(lambda x: x[i_uq_indices], [b_ids, i_ids, j_ids])

    kpts_valid_mask[b_ids, i_ids] = gt_matches_non_padding_mask[0, j_ids]
    kpts_warpped[b_ids, i_ids] = gt_matches[..., 2:][0, j_ids]

    return kpts_valid_mask, kpts_warpped


@torch.no_grad()
def warp_kpts_by_sparse_gt_matches_fast(
    kpts0, gt_matches, scale0, current_h, current_w
):
    B, n_gt_pts = gt_matches.shape[0], gt_matches.shape[1]
    kpts_warpped = torch.zeros_like(kpts0)
    kpts_valid_mask = torch.zeros_like(kpts0[..., 0], dtype=torch.bool)
    gt_matches_non_padding_mask = gt_matches.sum(-1) > 0

    all_j_idxs = torch.arange(
        gt_matches.shape[-2], device=gt_matches.device, dtype=torch.long
    )[None].expand(B, n_gt_pts)
    all_b_idxs = torch.arange(B, device=gt_matches.device, dtype=torch.long)[
        :, None
    ].expand(B, n_gt_pts)
    gt_matches_rescale = (
        gt_matches[..., :2] / scale0
    )  # From original img scale to resized scale
    in_boundary_mask = (
        (gt_matches_rescale[..., 0] <= current_w - 1)
        & (gt_matches_rescale[..., 0] >= 0)
        & (gt_matches_rescale[..., 1] <= current_h - 1)
        & (gt_matches_rescale[..., 1] >= 0)
    )

    gt_matches_rescale = gt_matches_rescale.round().to(torch.long)
    all_i_idxs = (
        gt_matches_rescale[..., 1] * current_w + gt_matches_rescale[..., 0]
    )  # idx = y * w + x

    # Filter:
    b_ids, i_ids, j_ids = map(
        lambda x: x[gt_matches_non_padding_mask & in_boundary_mask],
        [all_b_idxs, all_i_idxs, all_j_idxs],
    )

    j_uq_indices = get_unique_indices(torch.stack([b_ids, j_ids], dim=-1))
    b_ids, i_ids, j_ids = map(lambda x: x[j_uq_indices], [b_ids, i_ids, j_ids])

    i_uq_indices = get_unique_indices(torch.stack([b_ids, i_ids], dim=-1))
    b_ids, i_ids, j_ids = map(lambda x: x[i_uq_indices], [b_ids, i_ids, j_ids])

    kpts_valid_mask[b_ids, i_ids] = gt_matches_non_padding_mask[b_ids, j_ids]
    kpts_warpped[b_ids, i_ids] = gt_matches[..., 2:][b_ids, j_ids]

    return kpts_valid_mask, kpts_warpped


@torch.no_grad()
def homo_warp_kpts(
    kpts0, norm_pixel_mat, homo_sample_normed, original_size0=None, original_size1=None
):
    """
    original_size1: N * 2, (h, w)
    """
    normed_kpts0_h = norm_pixel_mat @ torch.cat(
        [kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1
    ).transpose(
        2, 1
    )  # (N * 3 * L)
    kpts_warpped_h = (
        torch.linalg.inv(norm_pixel_mat) @ homo_sample_normed @ normed_kpts0_h
    ).transpose(
        2, 1
    )  # (N * L * 3)
    kpts_warpped = kpts_warpped_h[..., :2] / kpts_warpped_h[..., [2]]  # N * L * 2
    valid_mask = (
        (kpts_warpped[..., 0] > 0)
        & (kpts_warpped[..., 0] < original_size1[:, [1]])
        & (kpts_warpped[..., 1] > 0)
        & (kpts_warpped[..., 1] < original_size1[:, [0]])
    )  # N * L
    if original_size0 is not None:
        valid_mask *= (
            (kpts0[..., 0] > 0)
            & (kpts0[..., 0] < original_size0[:, [1]])
            & (kpts0[..., 1] > 0)
            & (kpts0[..., 1] < original_size0[:, [0]])
        )  # N * L

    return valid_mask, kpts_warpped


@torch.no_grad()
# if using mask in homo warp(for coarse supervision)
def homo_warp_kpts_with_mask(
    kpts0,
    scale,
    depth_mask,
    norm_pixel_mat,
    homo_sample_normed,
    original_size0=None,
    original_size1=None,
):
    """
    original_size1: N * 2, (h, w)
    """
    normed_kpts0_h = norm_pixel_mat @ torch.cat(
        [kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1
    ).transpose(
        2, 1
    )  # (N * 3 * L)
    kpts_warpped_h = (
        torch.linalg.inv(norm_pixel_mat) @ homo_sample_normed @ normed_kpts0_h
    ).transpose(
        2, 1
    )  # (N * L * 3)
    kpts_warpped = kpts_warpped_h[..., :2] / kpts_warpped_h[..., [2]]  # N * L * 2
    # get coarse-level depth_mask
    depth_mask_coarse = depth_mask[:, :, ::scale, ::scale]
    depth_mask_coarse = depth_mask_coarse.reshape(depth_mask.shape[0], -1)

    valid_mask = (
        (kpts_warpped[..., 0] > 0)
        & (kpts_warpped[..., 0] < original_size1[:, [1]])
        & (kpts_warpped[..., 1] > 0)
        & (kpts_warpped[..., 1] < original_size1[:, [0]])
        & (depth_mask_coarse != 0)
    )  # N * L
    if original_size0 is not None:
        valid_mask *= (
            (kpts0[..., 0] > 0)
            & (kpts0[..., 0] < original_size0[:, [1]])
            & (kpts0[..., 1] > 0)
            & (kpts0[..., 1] < original_size0[:, [0]])
            & (depth_mask_coarse != 0)
        )  # N * L

    return valid_mask, kpts_warpped


@torch.no_grad()
# if using mask in homo warp(for fine supervision)
def homo_warp_kpts_with_mask_f(
    kpts0,
    depth_mask,
    norm_pixel_mat,
    homo_sample_normed,
    original_size0=None,
    original_size1=None,
):
    """
    original_size1: N * 2, (h, w)
    """
    normed_kpts0_h = norm_pixel_mat @ torch.cat(
        [kpts0, torch.ones_like(kpts0[:, :, [0]])], dim=-1
    ).transpose(
        2, 1
    )  # (N * 3 * L)
    kpts_warpped_h = (
        torch.linalg.inv(norm_pixel_mat) @ homo_sample_normed @ normed_kpts0_h
    ).transpose(
        2, 1
    )  # (N * L * 3)
    kpts_warpped = kpts_warpped_h[..., :2] / kpts_warpped_h[..., [2]]  # N * L * 2
    valid_mask = (
        (kpts_warpped[..., 0] > 0)
        & (kpts_warpped[..., 0] < original_size1[:, [1]])
        & (kpts_warpped[..., 1] > 0)
        & (kpts_warpped[..., 1] < original_size1[:, [0]])
        & (depth_mask != 0)
    )  # N * L
    if original_size0 is not None:
        valid_mask *= (
            (kpts0[..., 0] > 0)
            & (kpts0[..., 0] < original_size0[:, [1]])
            & (kpts0[..., 1] > 0)
            & (kpts0[..., 1] < original_size0[:, [0]])
            & (depth_mask != 0)
        )  # N * L

    return valid_mask, kpts_warpped


@torch.no_grad()
def homo_warp_kpts_glue(kpts0, homo, original_size0=None, original_size1=None):
    """
    original_size1: N * 2, (h, w)
    """
    kpts_warpped = warp_points_torch(kpts0, homo, inverse=False)
    valid_mask = (
        (kpts_warpped[..., 0] > 0)
        & (kpts_warpped[..., 0] < original_size1[:, [1]])
        & (kpts_warpped[..., 1] > 0)
        & (kpts_warpped[..., 1] < original_size1[:, [0]])
    )  # N * L
    if original_size0 is not None:
        valid_mask *= (
            (kpts0[..., 0] > 0)
            & (kpts0[..., 0] < original_size0[:, [1]])
            & (kpts0[..., 1] > 0)
            & (kpts0[..., 1] < original_size0[:, [0]])
        )  # N * L
    return valid_mask, kpts_warpped


@torch.no_grad()
# if using mask in homo warp(for coarse supervision)
def homo_warp_kpts_glue_with_mask(
    kpts0, scale, depth_mask, homo, original_size0=None, original_size1=None
):
    """
    original_size1: N * 2, (h, w)
    """
    kpts_warpped = warp_points_torch(kpts0, homo, inverse=False)
    # get coarse-level depth_mask
    depth_mask_coarse = depth_mask[:, :, ::scale, ::scale]
    depth_mask_coarse = depth_mask_coarse.reshape(depth_mask.shape[0], -1)

    valid_mask = (
        (kpts_warpped[..., 0] > 0)
        & (kpts_warpped[..., 0] < original_size1[:, [1]])
        & (kpts_warpped[..., 1] > 0)
        & (kpts_warpped[..., 1] < original_size1[:, [0]])
        & (depth_mask_coarse != 0)
    )  # N * L
    if original_size0 is not None:
        valid_mask *= (
            (kpts0[..., 0] > 0)
            & (kpts0[..., 0] < original_size0[:, [1]])
            & (kpts0[..., 1] > 0)
            & (kpts0[..., 1] < original_size0[:, [0]])
            & (depth_mask_coarse != 0)
        )  # N * L
    return valid_mask, kpts_warpped


@torch.no_grad()
# if using mask in homo warp(for fine supervision)
def homo_warp_kpts_glue_with_mask_f(
    kpts0, depth_mask, homo, original_size0=None, original_size1=None
):
    """
    original_size1: N * 2, (h, w)
    """
    kpts_warpped = warp_points_torch(kpts0, homo, inverse=False)
    valid_mask = (
        (kpts_warpped[..., 0] > 0)
        & (kpts_warpped[..., 0] < original_size1[:, [1]])
        & (kpts_warpped[..., 1] > 0)
        & (kpts_warpped[..., 1] < original_size1[:, [0]])
        & (depth_mask != 0)
    )  # N * L
    if original_size0 is not None:
        valid_mask *= (
            (kpts0[..., 0] > 0)
            & (kpts0[..., 0] < original_size0[:, [1]])
            & (kpts0[..., 1] > 0)
            & (kpts0[..., 1] < original_size0[:, [0]])
            & (depth_mask != 0)
        )  # N * L
    return valid_mask, kpts_warpped
