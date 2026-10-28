import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
from typing import List, Optional, Dict, Callable

from dinov3.hub.backbones import (
    dinov3_convnext_base,
    dinov3_convnext_large,
    dinov3_convnext_small,
    dinov3_convnext_tiny,
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vith16plus,
    dinov3_vitl16,
    dinov3_vitl16plus,
    dinov3_vits16,
    dinov3_vits16plus,
)
from dinov3.hub.classifiers import dinov3_vit7b16_lc
from dinov3.hub.detectors import dinov3_vit7b16_de
from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l
# from dinov3.hub.segmentors import dinov3_vit7b16_ms

from dinov3.hub.depthers import dinov3_vit7b16_dd

from dinov3.hub.backbones import Weights


"""
https://github.com/little51/dinov3-train/tree/main
https://github.com/little51/dinov3-samples
模型型号介绍: https://blog.csdn.net/qq_53644346/article/details/151717501
以下是按照 DINOv3 论文中描述的方法训练的 Vision Transformer 和 ConvNeXt 模型所提供的 12 种模型变体：

10 个模型在 Web 数据上预训练（LVD-1689M 数据集）
    从头开始训练的ViT-7B
    从 ViT-7B 中蒸馏出来的 ViT-S/S+/B/L/H+
    从 ViT-7B 中提炼而来的 ConvNeXt-{T/S/B/L} 型号
2 个在卫星数据上预训练的模型（SAT-493M 数据集）
    从头开始训练的ViT-7B
    从 ViT-7B 中蒸馏出来的ViT-L
"""


model_cards = {
    "dinov3_convnext_base": dinov3_convnext_base,
    "dinov3_convnext_large": dinov3_convnext_large,
    "dinov3_convnext_small": dinov3_convnext_small,
    "dinov3_convnext_tiny": dinov3_convnext_tiny,
    "dinov3_vit7b16": dinov3_vit7b16,
    "dinov3_vitb16": dinov3_vitb16,
    "dinov3_vith16plus": dinov3_vith16plus,
    "dinov3_vitl16": dinov3_vitl16,
    "dinov3_vitl16plus": dinov3_vitl16plus,
    "dinov3_vits16": dinov3_vits16,
    "dinov3_vits16plus": dinov3_vits16plus
}

weight_cards = {
    "dinov3_convnext_base": r"",
    "dinov3_convnext_large": r"",
    "dinov3_convnext_small": r"",
    "dinov3_convnext_tiny": r"",
    "dinov3_vit7b16": r"",
    "dinov3_vitb16": r"",
    "dinov3_vith16plus": r"",
    "dinov3_vitl16": r"",
    "dinov3_vitl16plus": r"",
    "dinov3_vits16": r"",
    "dinov3_vits16plus": r""
}


def _get_model(model_name: str):
    if model_name in model_cards:
        return model_cards[model_name](
            pretrained=True,
            weights=Weights.LVD1689M,
            check_hash=False,
        )
    else:
        raise ValueError(f"Model {model_name} not found")

"""
cls_token
storage_tokens
mask_token
patch_embed.proj.weight
patch_embed.proj.bias
blocks.0.norm1.weight
blocks.0.norm1.bias
blocks.0.attn.qkv.weight
blocks.0.attn.qkv.bias
blocks.0.attn.proj.weight
blocks.0.attn.proj.bias
blocks.0.ls1.gamma
blocks.0.norm2.weight
blocks.0.norm2.bias
blocks.0.mlp.fc1.weight
blocks.0.mlp.fc1.bias
blocks.0.mlp.fc2.weight
blocks.0.mlp.fc2.bias
blocks.0.ls2.gamma
norm.weight
norm.bias
"""

class DINOV3FeatureExtractor(nn.Module):

    def __init__(self,
                 extractor_type: str = "dinov3_vitb16",
                 layer_indexs: List[int] = [1, 4, 6, 11],
                 feature_transform: Optional[Callable] = None,
                 use_input_norm: bool = True
                 ):
        super().__init__()
        self.extractor = _get_model(extractor_type).eval()
        for k, v in self.extractor.named_parameters():
            v.requires_grad = False

        self.layer_indexs = layer_indexs
        self.feature_transform = feature_transform
        self.use_input_norm = use_input_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def make_transform(self, resize_size: int | list[int] = 768):
        to_tensor = v2.ToImage()
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return v2.Compose([to_tensor, resize, to_float, normalize])

    def _resize_16(self, img: torch.Tensor, base: int = 16) -> torch.Tensor:
        def _nearest_multiple(n: int, base: int = 16) -> int:
            # 保证至少为 base
            m = max(base, int(round(n / base)) * base)
            # 避免为 0
            return m if m > 0 else base
        h, w = img.shape[2:]
        if h % base != 0 or w % base != 0:
            h_new = _nearest_multiple(h, base)
            w_new = _nearest_multiple(w, base)
            img = F.interpolate(img, size=(h_new, w_new), mode='bicubic', align_corners=False, antialias=True)
        return img

    def _forward_features(self, x):
        features = self.extractor.get_intermediate_layers(
            x,
            n=self.layer_indexs,  # Layers or n last layers to take
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        return features

    @torch.no_grad()
    def forward(self, input_img: torch.Tensor) -> torch.Tensor:
        """
        input_img, target_img: (B,C,H,W), 范围 [0,1]
        返回：标量损失
        """
        # 1) 直接用 extractor 得到特征
        input_img = self._resize_16(input_img)
        if self.use_input_norm:
            input_img = (input_img - self.mean) / self.std
        feats_in = self._forward_features(input_img)

        return feats_in


class DINOV3PerceptualLoss(nn.Module):

    def __init__(self,
                 extractor_type: str = "dinov3_vitb16",
                 layer_indexs: List[int] = [1, 4, 6, 11],
                 layer_weights: Optional[Dict[str, float]] = None,
                 feature_transform: Optional[Callable] = None,
                 distance: str = 'mse',
                 use_input_norm: bool = True
                 ):
        super().__init__()
        self.extractor = _get_model(extractor_type).eval()
        for k, v in self.extractor.named_parameters():
            v.requires_grad = False

        self.layer_indexs = layer_indexs
        self.layer_weights = layer_weights
        self.feature_transform = feature_transform
        self.distance = distance
        self.use_input_norm = use_input_norm

        if self.use_input_norm:
            # the mean is for image with range [0, 1]
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            # the std is for image with range [0, 1]
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _distance(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.distance == 'mse':
            d = F.mse_loss(a, b, reduction='mean')
        elif self.distance == 'l1':
            d = F.l1_loss(a, b, reduction='mean')
        elif self.distance == 'cos':
            # 1 - cosine_similarity
            d = 1 - F.cosine_similarity(a.flatten(1), b.flatten(1), dim=1)
        else:
            raise ValueError('unsupported distance')

        return d

    def make_transform(self, resize_size: int | list[int] = 768):
        to_tensor = v2.ToImage()
        resize = v2.Resize((resize_size, resize_size), antialias=True)
        to_float = v2.ToDtype(torch.float32, scale=True)
        normalize = v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return v2.Compose([to_tensor, resize, to_float, normalize])

    def _resize_16(self, img: torch.Tensor, base: int = 16) -> torch.Tensor:
        def _nearest_multiple(n: int, base: int = 16) -> int:
            # 保证至少为 base
            m = max(base, int(round(n / base)) * base)
            # 避免为 0
            return m if m > 0 else base
        h, w = img.shape[2:]
        if h % base != 0 or w % base != 0:
            h_new = _nearest_multiple(h, base)
            w_new = _nearest_multiple(w, base)
            img = F.interpolate(img, size=(h_new, w_new), mode='bicubic', align_corners=False, antialias=True)
        return img

    def _forward_features(self, x):
        features = self.extractor.get_intermediate_layers(
            x,
            n=self.layer_indexs,  # Layers or n last layers to take
            reshape=True,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        return features

    # @torch.no_grad()
    def forward(self, input_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        """
        input_img, target_img: (B,C,H,W), 范围 [0,1]
        返回：标量损失
        """
        # 1) 直接用 extractor 得到特征
        input_img = self._resize_16(input_img)
        target_img = self._resize_16(target_img)
        if self.use_input_norm:
            input_img = (input_img - self.mean) / self.std
            target_img = (target_img - self.mean) / self.std
        feats_in = self._forward_features(input_img)
        feats_tar = self._forward_features(target_img)

        # 2) 对每个层计算距离并加权
        losses = []
        for f_in, f_tar, weight in zip(feats_in, feats_tar, self.layer_weights):

            if self.feature_transform is not None:
                f_in = self.feature_transform(f_in)
                f_tar = self.feature_transform(f_tar)
            # print(f_in.shape, f_tar.shape)
            d = self._distance(f_in, f_tar) * weight
            losses.append(d)

        if not losses:
            # 回退：若 extractor 未捕获到任何层，则直接在像素域计算
            return F.mse_loss(input_img, target_img)

        return sum(losses)



if __name__ == '__main__':
    """
    dinov3_vits16: 0 - 11  [1, 384, 14, 14]
    dinov3_vits16plus: 0 - 11  [1, 384, 14, 14]
    dinov3_vitl16: 0 -23  [1, 1024, 14, 14]
    dinov3_vitb16: 0 - 11  [1, 768, 14, 14]  
    """

    device = torch.device("cuda")

    model = _get_model("dinov3_vitb16").cuda().eval()
    for name, param in model.named_parameters():
        print(name)

    vgg = DINOV3PerceptualLoss(
        extractor_type="dinov3_vitb16",
        layer_indexs=[3, 5, 7, 9, 11],
        layer_weights=[0.1, 0.1, 1, 1, 1],
        feature_transform=None,
        distance='l1',
        use_input_norm=True,
    ).to(device)

    x = torch.randn(2, 3, 224, 224).to(device)
    x2 = torch.randn(2, 3, 224, 224).to(device)

    with torch.inference_mode():
        loss = vgg(x, x2)
        print(loss)


    # vgg = DINOV3PerceptualLoss(
    #     model,
    #     layer_weights=[1, 1, 1, 1],
    #     distance='mse',
    #     use_input_norm=True,
    # ).cuda()
    #
    # x = torch.randn(2, 3, 224, 224).cuda()
    # x2 = torch.randn(2, 3, 224, 224).cuda()
    #
    # with torch.inference_mode():
    #     features = model.get_intermediate_layers(
    #         x,
    #         n=(1, 4, 6, 11),  # Layers or n last layers to take
    #         reshape=True,
    #         return_class_token=False,
    #         return_extra_tokens=False,
    #         norm=True,
    #     )
    #     print(f"feature len: {len(features)}")
    #     for i in range(len(features)):
    #         print(f"feature {i} shape: {features[i].shape}")
    #
    #     loss = vgg(x, x2)
    #     print(loss)