import os
#os.chdir("..")
import torch
import cv2
from time import time
from loguru import logger
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from notebooks.notebooks_utils import make_matching_figure, show_image_pair
import PIL
import torch.nn.functional as F
import pydegensac
from roma.roma_adpat_model import ROMA_Model

def extract_geo_model_inliers(mkpts0, mkpts1, mconfs,
                              geo_model, ransac_method, pixel_thr, max_iters, conf_thr,
                              K0=None, K1=None):
    if geo_model == 'E':
        f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
        pixel_thr = pixel_thr / f_mean

        mkpts0, mkpts1 = map(lambda x: normalize_ketpoints(*x), [(mkpts0, K0), (mkpts1, K1)])
    
    if ransac_method == 'RANSAC':
        if geo_model == 'E':
            E, mask = cv2.findEssentialMat(mkpts0, 
                                           mkpts1,
                                           np.eye(3),
                                           threshold=pixel_thr, 
                                           prob=conf_thr, 
                                           method=cv2.RANSAC)
        elif geo_model == 'F':
            F, mask = cv2.findFundamentalMat(mkpts0,
                                             mkpts1,
                                             method=cv2.FM_RANSAC,
                                             ransacReprojThreshold=pixel_thr,
                                             confidence=conf_thr,
                                             maxIters=max_iters)
    elif ransac_method == 'DEGENSAC':
        assert geo_model == 'F'
        F, mask = pydegensac.findFundamentalMatrix(mkpts0,
                                                   mkpts1,
                                                   px_th=pixel_thr,
                                                   conf=conf_thr,
                                                   max_iters=max_iters)
    elif ransac_method == 'MAGSAC':
        params = cv2.UsacParams()
        # params.threshold = pixel_thr
        # params.confidence = conf_thr
        # params.maxIterations = max_iters
        # params.randomGeneratorState = 0
        # params.
        # F, mask = cv2.findFundamentalMat(mkpts0,
        #                                 mkpts1,
        #                                 method=cv2.USAC_MAGSAC,
        # )
        F, mask = cv2.findFundamentalMat(mkpts0,
                                        mkpts1,
                                        method=cv2.USAC_MAGSAC,
                                        ransacReprojThreshold=pixel_thr,
                                        confidence=conf_thr,
                                        maxIters=max_iters)
    else:
        raise ValueError()
    
    if mask is not None:
        mask = mask.astype(bool).flatten()
    else:
        mask = np.full_like(mconfs, True, dtype=np.bool)
    return mask

def extract_inliers(data, args):
    """extract inlier matches assume bs==1.
    NOTE: If no inliers found, keep all matches.
    """
    mkpts0, mkpts1, mconfs= extract_preds(data)
    K0 = data['K0'][0].cpu().numpy() if args.geo_model == 'E' else None
    K1 = data['K1'][0].cpu().numpy() if args.geo_model == 'E' else None
    if len(mkpts0) >=8 :
        inliers = extract_geo_model_inliers(mkpts0, mkpts1, mconfs,
                                            args.geo_model, args.ransac_method, args.pixel_thr, args.max_iters, args.conf_thr,
                                            K0=K0, K1=K1)
        mkpts0, mkpts1, mconfs = map(lambda x: x[inliers], [mkpts0, mkpts1, mconfs, detector_kpts_mask])

# The default config uses dual-softmax.
# The outdoor and indoor models share the same config.
# You can change the default values like thr and coarse_match_type.
if __name__ == "__main__":
    # matching_method = 'SuperPoint+SuperGlue'
    matching_method = 'ROMA'
    # enable_geometric_verify = False
    enable_geometric_verify = True
    loftr_cfg_path = "configs/loftr/matchanything/exps/loftr_ds_dense_PAN_M2D_noalign_repvgg_fpn_fp16_nf_conly_inter_clip0_dense_skipsoft_match_sparse_spv.py"
    loftr_model_path = "logs/tb_logs/megadepth_trainval_1024_with_depth_modal_with_glddepthwarp_with_thermaltest@-@loftr_ds_dense_PAN_M2D_noalign_repvgg_fpn_fp16_nf_conly_inter_clip0_dense_skipsoft_match_sparse_spv-bs12/version_0/checkpoints/last.ckpt"
    pixel_thr = 2.0
    img_resize = 840
    img_warp_back = True
    if matching_method == 'SuperPoint+SuperGlue':
        matcher = SPPSPG()
        matcher = matcher.eval().cuda()
    elif matching_method == 'LoFTR':
        config = get_cfg_defaults()
        config.merge_from_file(loftr_cfg_path)
        config = lower_config(config)
        matcher = LoFTR(config=config['loftr'])
        # matcher = LoFTR(config=default_ot_cfg)
        ckpt = torch.load(
            loftr_model_path, map_location="cpu"
        )["state_dict"]
        for k in list(ckpt.keys()):
            if 'matcher' in k:
                newk = k[k.find("matcher")+len('matcher')+1:]
                ckpt[newk] = ckpt[k]
                ckpt.pop(k)
        matcher.load_state_dict(ckpt)
        matcher = matcher.eval().cuda()
    elif matching_method == 'ROMA':
        # matcher = ROMA_Model({"n_sample": 5000, "load_img_in_model": False})
        matcher = ROMA_Model({"n_sample": 5000, "load_img_in_model": True})

    # rotation_degree = -90
    # rotation_degree = 30
    # rotation_degree = -90
    # rotation_degree = -45
    # rotation_degree = 45
    # rotation_degree = 15

    # scene_name = 'thermal'
    # # img0_pth = "assets/rgb_daytime.jpg"
    # # img0_pth = "/data/hexingyi/code/LoFTR/assets/rgb_daytime_6446.jpg"
    # img0_pth = "/data/hexingyi/code/LoFTR/assets/rgb_daytime.jpg"
    # img1_pth = "/data/hexingyi/code/LoFTR/assets/thermal_daytime1.jpg"
    # rotation_degree = 0
    # pixel_thr = 1.0

    # scene_name = 'satellite'
    # img0_pth = "/data/hexingyi/code/LoFTR/assets/satellite.jpg"
    # img1_pth = "/data/hexingyi/code/LoFTR/assets/airplane.jpg"
    # rotation_degree = 60
    # pixel_thr = 4.0

    # scene_name = 'satellite4'
    # img0_pth = "/data/common_dataset/uva_localization_data/cropped_map_images/214_115.9440317_115.9540317_40.367160000000005_40.37716.png"
    # img1_pth = "/data/hexingyi/code/UAV_Loc/0.png"
    # # img1_pth = "/data/hexingyi/code/LoFTR/assets/airplane2.png"
    # rotation_degree = -30
    # pixel_thr = 4.0

    # scene_name = 'satellite2'
    # img0_pth = "/data/hexingyi/code/LoFTR/assets/satellite.jpg"
    # img1_pth = "/data/hexingyi/code/LoFTR/assets/airplane2_cropped.jpeg"
    # # img1_pth = "/data/hexingyi/code/LoFTR/assets/airplane2.png"
    # rotation_degree = 0
    # pixel_thr = 2.0

    # scene_name = 'satellite3'
    # # img0_pth = "/data/hexingyi/code/LoFTR/assets/airplane3_cropped.jpeg"
    # img0_pth = "/data/hexingyi/code/LoFTR/assets/airplane3_squere.jpg"
    # img1_pth = "/data/hexingyi/code/LoFTR/assets/satellite_squere.jpg"
    # # img1_pth = "/data/hexingyi/code/LoFTR/assets/airplane2.png"
    # pixel_thr = 2.0

    # scene_name = 'yanshen_demo'
    # # img0_pth = "/data/hexingyi/code/LoFTR/assets/airplane3_cropped.jpeg"
    # img0_pth = "/data/hexingyi/code/LoFTR/assets/view3_new.png"
    # img1_pth = "/data/hexingyi/code/LoFTR/assets/view1.png"
    # # img1_pth = "/data/hexingyi/code/LoFTR/assets/airplane2.png"
    # rotation_degree = 0
    # pixel_thr = 2.0

    # scene_name = 'map'
    # img0_pth = "/data/hexingyi/code/LoFTR/assets/pair76_1.jpg"
    # img1_pth = "/data/hexingyi/code/LoFTR/assets/pair76_2.jpg"
    # pixel_thr = 4.0
    # rotation_degree = 0

    scene_name = 'sar'
    img0_pth = "/data/hexingyi/code/LoFTR/assets/rgb_pair_24_1.jpg"
    img0_pth_ = "/data/hexingyi/code/LoFTR/assets/rgb_pair_24_1_edited.jpg"
    img1_pth = "/data/hexingyi/code/LoFTR/assets/sar_pair24_2.jpg"
    img1_pth_ = "/data/hexingyi/code/LoFTR/assets/sar_pair24_2_edited.jpg"
    pixel_thr = 4.0
    rotation_degree = 0


    # scene_name = 'sar2'
    # img0_pth = "/data/hexingyi/code/LoFTR/assets/pair183_1.jpg"
    # img1_pth = "/data/hexingyi/code/LoFTR/assets/pair183_2.jpg"
    # img1_pth_ = "/data/hexingyi/code/LoFTR/assets/pair183_2_edited.jpg"
    # pixel_thr = 4.0
    # rotation_degree = 0

    # scene_name = 'medacine'
    # img0_pth = "/data/hexingyi/code/LoFTR/assets/ct.png"
    # img1_pth = "/data/hexingyi/code/LoFTR/assets/mri.png"
    # rotation_degree = 0
    # pixel_thr = 0.8

    # scene_name = 'medacine2'
    # img0_pth = "/data/hexingyi/code/LoFTR/assets/ct2.png"
    # img1_pth = "/data/hexingyi/code/LoFTR/assets/mri2.png"
    # rotation_degree = 0
    # # pixel_thr = 0.8

    # scene_name = 'deepsea'
    # img0_pth = "/data/hexingyi/code/LoFTR/assets/deepsea540.png"
    # # img1_pth = "/data/hexingyi/code/LoFTR/assets/deepsea700.png"
    # img1_pth = "/data/hexingyi/code/LoFTR/assets/deepsea789.png"
    # rotation_degree = 0

    img0_raw = cv2.imread(img0_pth, cv2.IMREAD_GRAYSCALE)
    img1_raw = cv2.imread(img1_pth, cv2.IMREAD_GRAYSCALE)

    try:
        img0_origin = cv2.cvtColor(cv2.imread(img0_pth_), cv2.COLOR_BGR2RGB)
    except:
        img0_origin = cv2.cvtColor(cv2.imread(img0_pth), cv2.COLOR_BGR2RGB)
    # img0_origin = cv2.rotate(img0_origin, cv2.cv2.ROTATE_90_CLOCKWISE)
    if not img_warp_back:
        img0_origin, warp_matrix = rotate_image(img0_origin, rotation_degree, preserve_full_img=False)

    try:
        img1_origin = cv2.cvtColor(cv2.imread(img1_pth_),cv2.COLOR_BGR2RGB)
    except:
        img1_origin = cv2.cvtColor(cv2.imread(img1_pth),cv2.COLOR_BGR2RGB)

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        batch = matcher({"image0_path": img0_pth, "image1_path": img1_pth})
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        mconf = batch['mconf'].cpu().numpy()

        kpts0 = batch['keypoints0'][0].cpu().numpy() if "keypoints0" in batch else None
        kpts1 = batch['keypoints1'][0].cpu().numpy() if "keypoints1" in batch else None

    if enable_geometric_verify and mkpts0.shape[0] >= 8:
        t0 = time()
        # inliers = extract_geo_model_inliers(mkpts0, mkpts1, mconf,
        #                                     geo_model="F", ransac_method='MAGSAC', pixel_thr=1.0, max_iters=10000, conf_thr=0.99999,
        #                                     K0=None, K1=None)

        inliers = extract_geo_model_inliers(mkpts0, mkpts1, mconf,
                                            # geo_model="F", ransac_method='MAGSAC', pixel_thr=pixel_thr, max_iters=10000, conf_thr=0.99999,
                                            geo_model="F", ransac_method='DEGENSAC', pixel_thr=pixel_thr, max_iters=10000, conf_thr=0.99999,
                                            K0=None, K1=None)
        t1 = time()
        mkpts0, mkpts1, mconf = map(lambda x: x[inliers], [mkpts0, mkpts1, mconf])
        print(f"Ransac takes:{t1-t0}, num inlier:{mkpts0.shape[0]}")
    else:
        logger.info("Geometry Verify is not Performed.")

    # Draw
    alpha = 0.5 if matching_method == 'SuperPoint+SuperGlue' else 0.15
    color = cm.jet(mconf, alpha=alpha)
    text = [
        matching_method,
        'Number of Matches: {}'.format(len(mkpts0)),
    ]

    vertical = True
    #fig = make_matching_figure(img0_raw, img1_raw, mkpts0, mkpts1, color, text)
    # fig = make_matching_figure(img1_raw, img1_raw, mkpts0, mkpts1, color, text, path="/home/hexingyi/code/LoFTR/matching_vertical.jpg", vertical=False)
    if kpts0 is not None and kpts1 is not None: 
        text=[]
        fig = make_matching_figure(img0_origin, img1_origin, mkpts0, mkpts1, color, kpts0=kpts0, kpts1=kpts1,text=text, draw_detection=True, draw_match_type=None, path=f"matching_horizontal_{matching_method}_{scene_name}_detection.jpg", vertical=vertical, plot_size_factor=3 if matching_method == 'SuperPoint+SuperGlue' else 1)
    # fig = make_matching_figure(img0_origin, img1_origin, mkpts0, mkpts1, color, text=text, path=f"matching_horizontal_{matching_method}_{scene_name}.jpg", vertical=False)

    text=[]
    # draw_match_type = "color"
    draw_match_type = "corres"
    # fig = make_matching_figure(img0_origin, img1_origin, mkpts0, mkpts1, color, text=text, path=f"{scene_name}_{matching_method}_matching{'_ransac' if enable_geometric_verify else ''}.jpg", vertical=vertical, plot_size_factor= 3 if matching_method == 'SuperPoint+SuperGlue' else 1)
    # fig = make_matching_figure(img0_origin, img1_origin, mkpts0, mkpts1, color, text=text, path=f"{scene_name}_{matching_method}_matching{'_ransac' if enable_geometric_verify else ''}_{draw_match_type}.jpg", vertical=False, plot_size_factor= 3 if matching_method == 'SuperPoint+SuperGlue' else 1, draw_match_type=draw_match_type, r_normalize_factor=0.4)
    fig = make_matching_figure(img0_origin, img1_origin, mkpts0, mkpts1, color, text=text, path=f"{scene_name}_{matching_method}_matching{'_ransac' if enable_geometric_verify else ''}_{draw_match_type}.jpg", vertical=True, plot_size_factor= 3 if matching_method == 'SuperPoint+SuperGlue' else 1, draw_match_type=draw_match_type, r_normalize_factor=0.4)
    # fig = make_matching_figure(img0_origin, img1_origin, mkpts0, mkpts1, color, text=text, path=f"{scene_name}_{matching_method}_matching{'_ransac' if enable_geometric_verify else ''}_{draw_match_type}.jpg", vertical=False, plot_size_factor= 3 if matching_method == 'SuperPoint+SuperGlue' else 1, draw_match_type=draw_match_type, r_normalize_factor=0.4, use_position_color=True)

    # # visualize pca
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=3 ,svd_solver='arpack')

    # # visualize pca for backbone feature
    # # feat: h*w*c
    # feat0 = feat_c0
    # feat1 = feat_c1

    # h,w,c = feat0.shape
    # feat = np.concatenate([feat0.reshape(-1,c), feat1.reshape(-1, c)], axis=0)
    # test_pca = np.random.rand(*feat.shape)
    # feat_pca = pca.fit_transform(feat)

    # feat_pca0, feat_pca1 = feat_pca[:h*w].reshape(h,w,3), feat_pca[h*w:].reshape(h,w,3)
    # feat_pca_cv2 = cv2.normalize(np.concatenate([feat_pca0,feat_pca1], axis=1), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
    # feat_pca_cv2_resize = cv2.resize(feat_pca_cv2,(w*2*8, h*8), interpolation=cv2.INTER_LINEAR)

    # feat_pca_resize0, feat_pca_resize1 = feat_pca_cv2_resize[:,:w*8,:], feat_pca_cv2_resize[:,w*8:,:]
    # feat_map_gapped = np.hstack((feat_pca_resize0, np.ones((h*8, 10, 3),dtype=np.uint8)*255, feat_pca_resize1))

    # # draw backbone feature pca
    # fig, axes = plt.subplots(1,1,dpi=100)
    # axes.imshow(feat_map_gapped)
    # axes.get_yaxis().set_ticks([])
    # axes.get_xaxis().set_ticks([])
    # plt.tight_layout(pad=.5)
    # plt.savefig('/home/hexingyi/code/LoFTR/backbone_feature.jpg')

    # # visualize pca for loftr coarse feature
    # # feat: hw*c
    # feat0 = loftr_c0
    # feat1 = loftr_c1

    # h,w = feat_c0.shape[:2]
    # c = loftr_c0.shape[-1]
    # feat = np.concatenate([feat0, feat1], axis=0)
    # feat_pca = pca.fit_transform(feat)
    # feat_pca0, feat_pca1 = feat_pca[:h*w].reshape(h,w,3), feat_pca[h*w:].reshape(h,w,3)
    # feat_pca_cv2 = cv2.normalize(np.concatenate([feat_pca0,feat_pca1], axis=1), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC3)
    # feat_pca_cv2_resize = cv2.resize(feat_pca_cv2,(w*2*8, h*8), interpolation=cv2.INTER_LINEAR)

    # feat_pca_resize0, feat_pca_resize1 = feat_pca_cv2_resize[:,:w*8,:], feat_pca_cv2_resize[:,w*8:,:]
    # feat_map_gapped = np.hstack((feat_pca_resize0, np.ones((h*8, 10, 3),dtype=np.uint8)*255, feat_pca_resize1))

    # # draw patches
    # fig, axes = plt.subplots(1,dpi=100)
    # axes.imshow(feat_map_gapped)
    # axes.get_yaxis().set_ticks([])
    # axes.get_xaxis().set_ticks([])
    # plt.tight_layout(pad=.5)
    # plt.savefig('/home/hexingyi/code/LoFTR/loftr_coarse_feature.jpg')