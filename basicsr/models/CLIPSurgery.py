import os.path

import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
import clip

def uint2tensor(img):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
    return img


class CLIPModel():
    def __init__(self,
                 device,
                 model_path=r"G:\PhD\Codes\Foundation\clip\ViT-B-16.pt"
                 ):
        super(CLIPModel, self).__init__()

        self.device = device
        self.model_path = model_path

        self._init_clip_model()


    @torch.no_grad()
    def _init_clip_model(self):
        self.clip_model, _ = clip.load("ViT-B/16", device=self.device, model_path=self.model_path)
        self.preprocess = Compose(
            [
                Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                # ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]
        )
        self.clip_model.eval()

    @torch.no_grad()
    def _visualize_clip_similarity_map(self, image, texts):
        similarity_map = self.__call__(image, texts)
        image_np = tensor2uint8(image)

        for b in range(similarity_map.shape[0]):
            for n in range(similarity_map.shape[-1]):
                vis = (similarity_map[b, :, :, n].cpu().numpy() * 255).astype('uint8')
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                vis = image_np * 0.4 + vis * 0.6
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                print('CLIP:', texts[n])
                plt.tight_layout()
                plt.axis('off')
                plt.imshow(vis)
                # plt.savefig(f'CLIP_{base}.png', dpi=600, bbox_inches='tight', pad_inches=0, transparent=True)

                plt.show()

    @torch.no_grad()
    def __call__(self, image, texts):
        b, c, h, w = image.shape

        image = self.preprocess(image)

        image_features = self.clip_model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Prompt ensemble for text features with normalization
        text_features = clip.encode_text_with_prompt_ensemble(self.clip_model, texts, self.device)

        # Similarity map from image tokens with min-max norm and resize, B,H,W,N
        features = image_features @ text_features.t()
        similarity_map = clip.get_similarity_map(features[:, 1:, :], (h, w))

        return similarity_map


class CLIPSurgeryModel():
    def __init__(self,
                 device,
                 model_path=r"G:\PhD\Codes\Foundation\CLIP_Surgery-master\weights\ViT-B-16.pt"
                 ):
        super(CLIPSurgeryModel, self).__init__()

        self.device = device
        self.model_path = model_path

        self._init_clip_surgery_model()

    @torch.no_grad()
    def _init_clip_surgery_model(self):
        self.model, _ = clip.load("CS-ViT-B/16", device=self.device, model_path=self.model_path)
        self.preprocess = Compose(
            [
                Resize((512, 512), interpolation=InterpolationMode.BICUBIC),
                # ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
            ]
        )
        self.model.eval()

    @torch.no_grad()
    def _visualize_similarity_map(self, image, texts):
        similarity_map = self.__call__(image, texts)
        image_np = tensor2uint8(image)

        for b in range(similarity_map.shape[0]):
            for n in range(similarity_map.shape[-1]):
                heatmap = similarity_map[b, :, :, n].cpu().numpy()
                vis = (heatmap * 255).astype('uint8')
                # vis = cv2.GaussianBlur(vis, ksize=(21, 21), sigmaX=0)
                vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                vis = image_np * 0.4 + vis * 0.6
                vis = cv2.cvtColor(vis.astype('uint8'), cv2.COLOR_BGR2RGB)
                print('CLIP Surgery for a single text:', texts[n])

                fig, ax = plt.subplots(figsize=(5, 5))
                ax.imshow(vis)
                ax.axis('off')

                # colorbar 绑定 heatmap，而不是 vis
                mappable = plt.cm.ScalarMappable(
                    cmap='jet',
                    # norm=plt.Normalize(vmin=heatmap.min(), vmax=heatmap.max())
                    norm=plt.Normalize(vmin=0., vmax=1.)
                )
                mappable.set_array([])

                cbar = fig.colorbar(
                    mappable,
                    ax=ax,
                    fraction=0.046,
                    pad=0.04
                )

                # cbar.set_label('Similarity Score', fontsize=10)
                # cbar.ax.tick_params(labelsize=8)
                cbar.set_ticks([])

                # # 刻度字体：Arial
                # from matplotlib.font_manager import FontProperties
                # arial_font = FontProperties(family='Arial', size=8)
                # for tick in cbar.ax.get_yticklabels():
                #     tick.set_fontproperties(arial_font)

                fig.tight_layout()
                # plt.show()

                plt.savefig(
                    f'CLIP_Surgery_colorbar.png',
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0,
                    transparent=True
                )


    @torch.no_grad()
    def __call__(self, image, texts):
        b, c, h, w = image.shape

        image = self.preprocess(image)
        # [1, 3, 512, 512] -> [1, 1025, 512]
        image_features = self.model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Prompt ensemble for text features with normalization [2, 512]
        text_features = clip.encode_text_with_prompt_ensemble(self.model, texts, self.device)

        # Extract redundant features from an empty string [1, 512]
        redundant_features = clip.encode_text_with_prompt_ensemble(self.model, [""], self.device)

        # Apply feature surgery for single text: [1, 1025, 2] -> [1, 2, 32, 32] -interploate-> [1, 512, 512, 2]
        similarity = clip.clip_feature_surgery(image_features, text_features, redundant_features)
        similarity_map = clip.get_similarity_map(similarity[:, 1:, :], (h, w))


        return similarity_map