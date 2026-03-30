import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision import models
from typing import List, Dict

# VGG-16 Layer Names and Channels
vgg16_layers = {
    "conv1_1": 64,
    "relu1_1": 64,
    "conv1_2": 64,
    "relu1_2": 64,
    "pool1": 64,
    "conv2_1": 128,
    "relu2_1": 128,
    "conv2_2": 128,
    "relu2_2": 128,
    "pool2": 128,
    "conv3_1": 256,
    "relu3_1": 256,
    "conv3_2": 256,
    "relu3_2": 256,
    "conv3_3": 256,
    "relu3_3": 256,
    "pool3": 256,
    "conv4_1": 512,
    "relu4_1": 512,
    "conv4_2": 512,
    "relu4_2": 512,
    "conv4_3": 512,
    "relu4_3": 512,
    "pool4": 512,
    "conv5_1": 512,
    "relu5_1": 512,
    "conv5_2": 512,
    "relu5_2": 512,
    "conv5_3": 512,
    "relu5_3": 512,
    "pool5": 512,
}

class AdapLayers(nn.Module):
    """Small adaptation layers.
    """

    def __init__(self, hypercolumn_layers: List[str], output_dim: int = 128):
        """Initialize one adaptation layer for every extraction point.

        Args:
            hypercolumn_layers: The list of the hypercolumn layer names.
            output_dim: The output channel dimension.
        """
        super(AdapLayers, self).__init__()
        self.layers = []
        channel_sizes = [vgg16_layers[name] for name in hypercolumn_layers]
        for i, l in enumerate(channel_sizes):
            layer = nn.Sequential(
                nn.Conv2d(l, 64, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, output_dim, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(output_dim),
            )
            self.layers.append(layer)
            self.add_module("adap_layer_{}".format(i), layer)

    def forward(self, features: List[torch.tensor]):
        """Apply adaptation layers.
        """
        for i, _ in enumerate(features):
            features[i] = getattr(self, "adap_layer_{}".format(i))(features[i])
        return features

class S2DNet(nn.Module):
    """The S2DNet model
    """

    def __init__(
        self,
        # hypercolumn_layers: List[str] = ["conv2_2", "conv3_3", "relu4_3"],
        hypercolumn_layers: List[str] = ["conv1_2", "conv3_3", "conv5_3"],
        checkpoint_path: str = None,
    ):
        """Initialize S2DNet.

        Args:
            device: The torch device to put the model on
            hypercolumn_layers: Names of the layers to extract features from
            checkpoint_path: Path to the pre-trained model.
        """
        super(S2DNet, self).__init__()
        self._checkpoint_path = checkpoint_path
        self.layer_to_index = dict((k, v) for v, k in enumerate(vgg16_layers.keys()))
        self._hypercolumn_layers = hypercolumn_layers

        # Initialize architecture
        vgg16 = models.vgg16(pretrained=False)
        # layers = list(vgg16.features.children())[:-2]
        layers = list(vgg16.features.children())[:-1]
        # layers = list(vgg16.features.children())[:23] # relu4_3
        self.encoder = nn.Sequential(*layers)
        self.adaptation_layers = AdapLayers(self._hypercolumn_layers) # .to(self._device)
        self.eval()

        # Restore params from checkpoint
        if checkpoint_path:
            print(">> Loading weights from {}".format(checkpoint_path))
            self._checkpoint = torch.load(checkpoint_path)
            self._hypercolumn_layers = self._checkpoint["hypercolumn_layers"]
            self.load_state_dict(self._checkpoint["state_dict"])

    def forward(self, image_tensor: torch.FloatTensor):
        """Compute intermediate feature maps at the provided extraction levels.

        Args:
            image_tensor: The [N x 3 x H x Ws] input image tensor.
        Returns:
            feature_maps: The list of output feature maps.
        """
        feature_maps, j = [], 0
        feature_map = image_tensor.repeat(1,3,1,1)
        layer_list = list(self.encoder.modules())[0]
        for i, layer in enumerate(layer_list):
            feature_map = layer(feature_map)
            if j < len(self._hypercolumn_layers):
                next_extraction_index = self.layer_to_index[self._hypercolumn_layers[j]]
                if i == next_extraction_index:
                    feature_maps.append(feature_map)
                    j += 1
        feature_maps = self.adaptation_layers(feature_maps)
        return feature_maps
