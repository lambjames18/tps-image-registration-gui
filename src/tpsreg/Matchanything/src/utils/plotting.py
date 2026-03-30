import bisect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import torch

def _compute_conf_thresh(data):
    dataset_name = data['dataset_name'][0].lower()
    if dataset_name == 'scannet':
        thr = 5e-4
    elif dataset_name == 'megadepth':
        thr = 1e-4
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    return thr


# --- VISUALIZATION --- #
def make_matching_figure(
        img0, img1, mkpts0, mkpts1, color,
        kpts0=None, kpts1=None, text=[], dpi=75, path=None):
    # draw image pair
    assert mkpts0.shape[0] == mkpts1.shape[0], f'mkpts0: {mkpts0.shape[0]} v.s. mkpts1: {mkpts1.shape[0]}'
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=dpi)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)
    
    if kpts0 is not None:
        assert kpts1 is not None
        axes[0].scatter(kpts0[:, 0], kpts0[:, 1], c='w', s=2)
        axes[1].scatter(kpts1[:, 0], kpts1[:, 1], c='w', s=2)

    # draw matches
    if mkpts0.shape[0] != 0 and mkpts1.shape[0] != 0:
        fig.canvas.draw()
        transFigure = fig.transFigure.inverted()
        fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
        fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
        fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]),
                                            (fkpts0[i, 1], fkpts1[i, 1]),
                                            transform=fig.transFigure, c=color[i], linewidth=1)
                                        for i in range(len(mkpts0))]
        
        axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
        axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def _make_evaluation_figure(data, b_id, alpha='dynamic', use_m_bids_f=False):
    if use_m_bids_f:
        b_mask = (data['m_bids_f'] == b_id) if 'm_bids_f' in data else (data['m_bids'] == b_id)
    else:
        b_mask = data['m_bids'] == b_id
    conf_thr = _compute_conf_thresh(data)
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    kpts0 = data['mkpts0_f'][b_mask].cpu().numpy()
    kpts1 = data['mkpts1_f'][b_mask].clone().detach().cpu().numpy()
    
    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()

    epi_errs = data['epi_errs'][b_mask].cpu().numpy()
    correct_mask = epi_errs < conf_thr
    precision = np.mean(correct_mask) if len(correct_mask) > 0 else 0
    n_correct = np.sum(correct_mask)
    n_gt_matches = int(data['conf_matrix_gt'][b_id].sum().cpu()) if 'conf_matrix_gt' in data else data['gt'][1]['gt_prob'].sum()
    recall = 0 if n_gt_matches == 0 else n_correct / (n_gt_matches)
    # recall might be larger than 1, since the calculation of conf_matrix_gt
    # uses groundtruth depths and camera poses, but epipolar distance is used here.

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(correct_mask))
    color = error_colormap(epi_errs, conf_thr, alpha=alpha)
    
    text = [
        f'#Matches {len(kpts0)}',
        f'Precision({conf_thr:.2e}) ({100 * precision:.1f}%): {n_correct}/{len(kpts0)}',
        f'Recall({conf_thr:.2e}) ({100 * recall:.1f}%): {n_correct}/{n_gt_matches}'
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text)
    return figure

def _make_confidence_figure(data, b_id):
    raise NotImplementedError()

def _make_gt_figure(data, b_id, alpha='dynamic', use_m_bids_f=False, mode='gt_fine'):
    if 'fine' in mode:
        mkpts0_key, mkpts1_key = 'mkpts0_f_gt', 'mkpts1_f_gt'
    else:
        mkpts0_key, mkpts1_key = 'mkpts0_c_gt', 'mkpts1_c_gt'

    if data['image0'].shape[0] == 1:
        b_mask = torch.tensor([True]*data[mkpts0_key].shape[0], device = data[mkpts0_key].device)
    else:
        raise NotImplementedError
    
    conf_thr = _compute_conf_thresh(data)
    
    img0 = (data['image0'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    img1 = (data['image1'][b_id][0].cpu().numpy() * 255).round().astype(np.int32)
    try:
        kpts0 = data[mkpts0_key][b_mask].cpu().numpy()
        kpts1 = data[mkpts1_key][b_mask].cpu().numpy()
    except:
        kpts0, kpts1 = np.ones((0, 2)), np.ones((0, 2))
    
    # for megadepth, we visualize matches on the resized image
    if 'scale0' in data:
        kpts0 = kpts0 / data['scale0'][b_id].cpu().numpy()
        kpts1 = kpts1 / data['scale1'][b_id].cpu().numpy()

    # matching info
    if alpha == 'dynamic':
        alpha = dynamic_alpha(len(kpts0))
    color = error_colormap(np.full((kpts0.shape[0]), conf_thr), conf_thr, alpha=0.1)
    
    text = [
        f'#Matches {len(kpts0)}',
    ]
    
    # make the figure
    figure = make_matching_figure(img0, img1, kpts0, kpts1,
                                  color, text=text)
    return figure

def make_matching_figures(data, config, mode='evaluation'):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            fig = _make_evaluation_figure(
                data, b_id,
                alpha=config.TRAINER.PLOT_MATCHES_ALPHA,
                use_m_bids_f=config.LOFTR.FINE.MTD_SPVS)
        elif mode == 'confidence':
            fig = _make_confidence_figure(data, b_id)
        elif 'gt' in mode:
            fig = _make_gt_figure(data, b_id, use_m_bids_f=config.LOFTR.FINE.MTD_SPVS, mode=mode)
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures

def make_scores_figures(data, config, mode='evaluation'):
    """ Make matching figures for a batch.
    
    Args:
        data (Dict): a batch updated by PL_LoFTR.
        config (Dict): matcher config
    Returns:
        figures (Dict[str, List[plt.figure]]
    """
    assert mode in ['evaluation', 'confidence', 'gt']  # 'confidence'
    figures = {mode: []}
    for b_id in range(data['image0'].size(0)):
        if mode == 'evaluation':
            if config.LOFTR.MATCH_COARSE.SKIP_SOFTMAX and config.LOFTR.MATCH_COARSE.PLOT_ORIGIN_SCORES:
                plots = [data['histc_skipmn_in_softmax'][b_id].reshape(-1)] # [-30, 70] scores
                if 'histc_skipmn_in_softmax_gt' in data:
                    plots.append(data['histc_skipmn_in_softmax_gt'][b_id].reshape(-1))
            elif config.LOFTR.MATCH_COARSE.PLOT_ORIGIN_SCORES:
                pass
            else:
                pass
            print(plots[0], plots[-1])
            group = len(plots)
            start, end = 0, 100
            bins=100
            width = (end//bins-1)/group
            fig, ax = plt.subplots()
            for i, hist in enumerate(plots):
                ax.set_yscale('log')
                x = range(start, end, end//bins)
                x = [t + i*width for t in x]
                ax.bar(x, hist.cpu(), align='edge', width=width)
            
        elif mode == 'confidence':
            raise NotImplementedError()
        elif mode == 'gt':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown plot mode: {mode}')
        figures[mode].append(fig)
    return figures


def dynamic_alpha(n_matches,
                  milestones=[0, 300, 1000, 2000],
                  alphas=[1.0, 0.8, 0.4, 0.2]):
    if n_matches == 0:
        return 1.0
    ranges = list(zip(alphas, alphas[1:] + [None]))
    loc = bisect.bisect_right(milestones, n_matches) - 1
    _range = ranges[loc]
    if _range[1] is None:
        return _range[0]
    return _range[1] + (milestones[loc + 1] - n_matches) / (
        milestones[loc + 1] - milestones[loc]) * (_range[0] - _range[1])


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    if thr is not None:
        large_error_mask = err > (thr * 2)
        x = np.clip((err - thr) / (thr * 2), 0, 1)
    else:
        large_error_mask = np.zeros_like(err, dtype=bool)
        x = np.clip(err, 0.1, 1)

    cm_ = matplotlib.colormaps['RdYlGn_r']
    color = cm_(x, bytes=False)
    color[:, 3] = alpha
    color[:, 3][large_error_mask] = alpha * 0.6
    return color