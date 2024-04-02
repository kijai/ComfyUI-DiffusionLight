import numpy as np

import torch

class TonemapHDR(object):
    """
        Tonemap HDR image globally. First, we find alpha that maps the (max(tensor_img) * percentile) to max_mapping.
        Then, we calculate I_out = alpha * I_in ^ (1/gamma)
        input : torch.Tensor batch of images : [H, W, C]
        output : torch.Tensor batch of images : [H, W, C]
    """

    def __init__(self, gamma=2.4, percentile=50, max_mapping=0.5):
        self.gamma = gamma
        self.percentile = percentile
        self.max_mapping = max_mapping  # the value to which alpha will map the (max(tensor_img) * percentile) to

    def __call__(self, tensor_img, clip=True, alpha=None, gamma=True):
        if gamma:
            power_tensor_img = torch.pow(tensor_img, 1 / self.gamma)
        else:
            power_tensor_img = tensor_img
        non_zero = power_tensor_img > 0
        if non_zero.any():
            r_percentile = torch.quantile(power_tensor_img[non_zero], self.percentile / 100.0)
        else:
            r_percentile = torch.quantile(power_tensor_img, self.percentile / 100.0)
        if alpha is None:
            alpha = self.max_mapping / (r_percentile + 1e-10)
        tonemapped_img = alpha * power_tensor_img

        if clip:
            tonemapped_img_clip = torch.clamp(tonemapped_img, 0, 1)
        else:
            tonemapped_img_clip = tonemapped_img

        return tonemapped_img_clip.float(), alpha, tonemapped_img