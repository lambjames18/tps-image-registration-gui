from .resnet_fpn import ResNetFPN_8_2, ResNetFPN_16_4, ResNetFPN_8_1, ResNetFPN_8_2_align, ResNetFPN_8_1_align, ResNetFPN_8_2_fix, ResNet_8_1_align, VGG_8_1_align, RepVGG_8_1_align, \
    RepVGGnfpn_8_1_align, RepVGG_8_2_fix, s2dnet_8_1_align

def build_backbone(config):
    if config['backbone_type'] == 'ResNetFPN':
        if config['align_corner'] is None or config['align_corner'] is True:
            if config['resolution'] == (8, 2):
                return ResNetFPN_8_2(config['resnetfpn'])
            elif config['resolution'] == (16, 4):
                return ResNetFPN_16_4(config['resnetfpn'])
            elif config['resolution'] == (8, 1):
                return ResNetFPN_8_1(config['resnetfpn'])
        elif config['align_corner'] is False:
            if config['resolution'] == (8, 2):
                return ResNetFPN_8_2_align(config['resnetfpn'])
            elif config['resolution'] == (16, 4):
                return ResNetFPN_16_4(config['resnetfpn'])
            elif config['resolution'] == (8, 1):
                return ResNetFPN_8_1_align(config['resnetfpn'])
    elif config['backbone_type'] == 'ResNetFPNFIX':
        if config['align_corner'] is None or config['align_corner'] is True:
            if config['resolution'] == (8, 2):
                return ResNetFPN_8_2_fix(config['resnetfpn'])
    elif config['backbone_type'] == 'ResNet':
        if config['align_corner'] is None or config['align_corner'] is True:
            raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
        elif config['align_corner'] is False:
            if config['resolution'] == (8, 1):
                return ResNet_8_1_align(config['resnetfpn'])
    elif config['backbone_type'] == 'VGG':
        if config['align_corner'] is None or config['align_corner'] is True:
            raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
        elif config['align_corner'] is False:
            if config['resolution'] == (8, 1):
                return VGG_8_1_align(config['resnetfpn'])
    elif config['backbone_type'] == 'RepVGG':
        if config['align_corner'] is None or config['align_corner'] is True:
            raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
        elif config['align_corner'] is False:
            if config['resolution'] == (8, 1):
                return RepVGG_8_1_align(config['resnetfpn'])
    elif config['backbone_type'] == 'RepVGGNFPN':
        if config['align_corner'] is None or config['align_corner'] is True:
            raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
        elif config['align_corner'] is False:
            if config['resolution'] == (8, 1):
                return RepVGGnfpn_8_1_align(config['resnetfpn'])
    elif config['backbone_type'] == 'RepVGGFPNFIX':
        if config['align_corner'] is None or config['align_corner'] is True:
            if config['resolution'] == (8, 2):
                return RepVGG_8_2_fix(config['resnetfpn'])
        elif config['align_corner'] is False:
            raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
    elif config['backbone_type'] == 's2dnet':
        if config['align_corner'] is None or config['align_corner'] is True:
            raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
        elif config['align_corner'] is False:
            if config['resolution'] == (8, 1):
                return s2dnet_8_1_align(config['resnetfpn'])
    else:
        raise ValueError(f"LOFTR.BACKBONE_TYPE {config['backbone_type']} not supported.")
