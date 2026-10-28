import torch
import matplotlib
import numpy as np
import cv2, os
from torchvision.transforms import Compose
import torch.nn.functional as F
from tqdm import tqdm
import kornia
# # V1
# from VM.Depth_Anything.depth_anything.dpt import DepthAnything
# from VM.Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
# # V2
from basicsr.depth.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2


@torch.no_grad()
class _depth_v2_interface():
    def __init__(self, root, device, encoder='vits'):

        assert encoder in ('vits', 'vitb', 'vitl'), f'encoder must be vits, vitb or vitg, but got {encoder}'

        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }

        model = DepthAnythingV2(**model_configs[encoder])
        model.load_state_dict(torch.load(os.path.join(root, f'depth_anything_v2_{encoder}.pth'), map_location='cpu'))
        for param in model.parameters():
            param.requires_grad = False
        self.model = model.to(device).eval()
        self.device = device

    def __call__(self, x, blur=False, size=21):
        b, c, h, w = x.shape
        x = self._normalize(x)
        x = self.check_image_size(x, padder_size=14)
        depth, features = self.model.forward(x)
        depth = depth[:, :, :h, :w]
        # depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = self.minmax_depth(depth, blur=blur, size=size)
        # depth = (depth - depth.amin(dim=(1, 2, 3), keepdim=True)) / (
        #             depth.amax(dim=(1, 2, 3), keepdim=True) - depth.amin(dim=(1, 2, 3), keepdim=True) + 1e-6)
        return depth#.unsqueeze(1)  # B 1 H W

    def post_process(self, x, blur=False, size=21, uint8=False, grayscale=False) :
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        b, c, h, w = x.shape
        depth = self.__call__(x, blur=blur, size=size)
        # depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)[0, 0]  # H W
        depth = depth.data.cpu().numpy()[0, 0]
        if uint8:
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        elif grayscale:
            depth = depth   # np.repeat(depth[..., np.newaxis], 3, axis=-1)
        else:
            raise NotImplementedError

        return depth

    def _normalize(self, features, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):  # noqa
        mean = torch.tensor(mean, dtype=features.dtype, device=features.device).reshape(1, 3, 1, 1)
        std = torch.tensor(std, dtype=features.dtype, device=features.device).reshape(1, 3, 1, 1)

        # features = features.sub_(mean).div_(std)
        features = features.sub(mean).div(std)
        return features

    def check_image_size(self, x, padder_size=14):
        _, _, h, w = x.size()
        mod_pad_h = (padder_size - h % padder_size) % padder_size
        mod_pad_w = (padder_size - w % padder_size) % padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def minmax_depth(self, depth, blur=False, size=21):
        # D4
        n, c, h, w = depth.shape
        # depth = F.avg_pool2d(depth,kernel_size=5)

        if blur:  # use median filter to filter out peak values for visualization.
            # depth = F.pad(depth, [4, 4, 4, 4], 'reflect')
            depth = kornia.filters.blur.box_blur(depth, kernel_size=(size, size), separable=False)  # [B, 1, H, W]
            # depth = depth[:, :, 3:h - 3, 3:w - 3]
        depth = (depth - depth.amin(dim=(1, 2, 3), keepdim=True)) / (
                    depth.amax(dim=(1, 2, 3), keepdim=True) - depth.amin(dim=(1, 2, 3), keepdim=True) + 1e-6)
        # if channel_avg:
        #     depth = (depth - depth.amin(dim=(1, 2, 3), keepdim=True)) / (depth.amax(dim=(1, 2, 3), keepdim=True) - depth.amin(dim=(1, 2, 3), keepdim=True) + 1e-6)
        # else:
        #     D_max = torch.max(depth.reshape(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        #     D_min = torch.min(depth.reshape(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        #     depth = (depth - D_min) / (D_max - D_min)
        return depth
