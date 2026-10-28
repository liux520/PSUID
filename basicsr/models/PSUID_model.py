import os.path
import random
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from basicsr.depth.depth_interface import _depth_v2_interface
from basicsr.depth.dehaze_utils import TransmissionEstimator
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from .sr_model import SRModel
from torchvision.utils import make_grid, save_image
from basicsr.utils import DiffJPEG, USMSharp
from scripts.self_ensemble import test_selfensemble
from basicsr.depth.fix_tools import wavelet_decomposition, wavelet_blur
from basicsr.models.PSUID_base_model import PSUIDBaseModel
from torchvision import transforms
import kornia.augmentation as K
ROOT = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))


def _init_dinov3(
        extractor_type="dinov3_vitb16",
        layer_indexs=[3, 5, 7, 9, 11],
        layer_weights=[0.1, 0.1, 1, 1, 1]
        ):
    from dinov3.feature_extractors import DINOV3PerceptualLoss
    vgg = DINOV3PerceptualLoss(
        extractor_type=extractor_type,
        layer_indexs=layer_indexs,
        layer_weights=layer_weights,
        feature_transform=None,
        distance='l1',
        use_input_norm=True,
    )
    return vgg


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)

        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

@MODEL_REGISTRY.register()
class PSUIDInheritModel(PSUIDBaseModel):
    def __init__(self, opt):
        super(PSUIDInheritModel, self).__init__(opt)

        self.cri_dinov3 = _init_dinov3(
            extractor_type="dinov3_vitb16",
            layer_indexs=[3, 5, 7, 9, 11],
            layer_weights=[0.1, 0.1, 1, 1, 1]
        ).to(self.device)

        self.net_m.requires_grad_(False)
        self.net_m.eval()

        # from basicsr.models.A_CLIPLOSS import CLIPLOSS
        # self.cri_clip = CLIPLOSS(device=self.device)
        from basicsr.models.CLIPSurgery import CLIPSurgeryModel
        self.cri_clip = CLIPSurgeryModel(
            device=self.device,
            model_path=self.opt['train']['CLIPSurgery']
        )

        ###
        # from CLIP_CoA import clip
        # from CLIP_CoA.clip_score_ import L_clip_from_feature
        #
        # foundation_path = r'/data1/sr617/codes/liux/Foundation'
        # clip_path = os.path.join(foundation_path, 'clip', 'ViT-B-32.pt')  # ViT-B/32
        # resnet_path = os.path.join(foundation_path, 'clip', 'RN101.pt')  # RN101
        # haze_prompt_path = os.path.join(foundation_path, 'clip', 'haze_prompt.pth')
        #
        # clip_model, _ = clip.load(clip_path, device=torch.device("cpu"))
        # clip_model.to(self.device).eval()
        # for param in clip_model.parameters():
        #     param.requires_grad = False
        # res_model, _ = clip.load(resnet_path, device=torch.device("cpu"))
        # res_model.to(self.device).eval()
        # for param in res_model.parameters():
        #     param.requires_grad = False
        #
        # data = torch.load(haze_prompt_path)
        # new_state_dict = OrderedDict({k.replace('module.', ''): v for k, v in data.items()})
        # embedding_prompt = nn.Parameter(new_state_dict['embedding_prompt']).to(self.device)
        # embedding_prompt.requires_grad = False
        #
        # text_encoder = TextEncoder(clip_model)
        # tokenized_prompts = torch.cat([clip.tokenize(p) for p in [" ".join(["X"] * 16)]])
        # self.text_features = text_encoder(embedding_prompt, tokenized_prompts)
        #
        # self.clip_loss = L_clip_from_feature()
        ###

    def _model_set(self):
        self.net_g.train()
        # self.net_m.train()
        # self.net_d.train()
        self.net_dehaze_d.train()
        self.netF.train()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'lq_strong' in data:
            self.lq_strong = data['lq_strong'].to(self.device)

    def optimize_parameters(self, current_iter):
        # for p in self.net_d.parameters():
        #     p.requires_grad = False
        # for p in self.net_dehaze_d.parameters():
        #     p.requires_grad = False
        self.optimizer_g.zero_grad()
        # self.optimizer_m.zero_grad()
        self.optimizer_F.zero_grad()

        # # int(self.opt['train']['beta_mul']) # 3
        # now, set apply_gamma=True, apply_noise=True, apply_jpeg=True, beta_mul=2
        lq, beta, A, t, depth = self._syn_haze(
            clean=self.gt, depth=None, kernel=None, beta_range=(0.6, 1.8), A_range=(0.7, 1.0), beta_mul=2,
            color_range=(-0.025, 0.025), noise_range=(2, 25), jpeg_range=(30, 95), A_dim=1, A_color=0.25,
            apply_gamma=True, apply_noise=True, apply_jpeg=True, apply_A_random=True,
            given_A=self._random_hazy_style() if torch.rand(1) < 0.25 else None
        )
        syn_hq = self.net_g(lq)
        syn_hq_2 = self.net_g(self.lq)

        with torch.no_grad():
            syn_hq_2_teacher = torch.ones_like(self.lq)
            for i in range(syn_hq_2_teacher.shape[0]):
                syn_hq_2_teacher[i, :, :, :] = test_selfensemble(self.lq[i, :, :, :].unsqueeze(0), self.net_g_ema).detach()
        syn_hq_2_student = self.net_g(self.lq_strong)

        # content_feat, deg_embedding, out
        with torch.no_grad():
            _, _, syn_lq_1 = self.net_m(self.gt, self.lq)

        syn_hq_1 = self.net_g(syn_lq_1.detach())  #

        # [0, 1] -> [1, 1.5]
        mask_syn_hq = self.cri_clip(syn_hq, ['hazy']) / 2.0 + 1.0  # ['haze', 'fog', 'mist']
        mask_syn_hq_1 = self.cri_clip(syn_hq_1, ['hazy']) / 2.0 + 1.0
        mask_syn_hq_2_student = self.cri_clip(syn_hq_2_student, ['hazy']) / 2.0 + 1.0

        mask_syn_hq = mask_syn_hq[0, :, :, 0].unsqueeze(0).unsqueeze(0)
        mask_syn_hq_1 = mask_syn_hq_1[0, :, :, 0].unsqueeze(0).unsqueeze(0)
        mask_syn_hq_2_student = mask_syn_hq_2_student[0, :, :, 0].unsqueeze(0).unsqueeze(0)

        if self.opt['train']['save_flag'] and current_iter % self.opt['train']['save_freq'] == 0:
            self._save_images(self.gt, current_iter, '0_gt_online_syn')
            self._save_images(lq, current_iter, '1_lq_online_syn')
            self._save_images(syn_hq, current_iter, '2_hq_online_syn')
            self._save_images((1 - depth), current_iter, '3_depth_online_syn')
            self._save_images(self.lq, current_iter, '4_real_lq')
            self._save_images(syn_hq_2, current_iter, '5_real_lq_dehaze')
            # self._save_images(syn_lq_2, current_iter, '6_real_lq_rehaze')
            self._save_images(syn_lq_1, current_iter, '7_gt_lq_syn_new_lq')
            self._save_images(syn_hq_1, current_iter, '8_new_lq_dehaze')
            self._save_images(syn_hq_2_teacher, current_iter, '9_syn_hq_2_teacher')
            self._save_images(syn_hq_2_student, current_iter, '10_syn_hq_2_student')

        l_g_total = 0.
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            loss_NCE = self.calculate_NCE_loss(self.lq, syn_hq_2) * self.opt['train']['NCE_loss_weight']
            # loss_NCE = self.calculate_NCE_loss(syn_hq_2_student, syn_hq_2_teacher)
            l_g_total += loss_NCE
            loss_dict['loss_NCE'] = loss_NCE

            # # clip loss
            # loss_clip = self.clip_loss(syn_hq_2, self.text_features) * self.opt['train']['clip_loss_weight']
            # l_g_total += loss_NCE
            # loss_dict['loss_clip'] = loss_clip

            # For G, PSNR Loss
            if self.cri_pix:
                l_g_pix1 = self.cri_pix(self.gt, syn_hq, mask_syn_hq)
                l_g_pix2 = self.cri_pix(self.gt, syn_hq_1, mask_syn_hq_1)
                l_g_pix3 = self.cri_pix(syn_hq_2_teacher, syn_hq_2_student, mask_syn_hq_2_student)

                l_g_total += (l_g_pix1 + l_g_pix2 + l_g_pix3)
                loss_dict['l_g_pix1'] = l_g_pix1
                loss_dict['l_g_pix2'] = l_g_pix2
                loss_dict['l_g_pix3'] = l_g_pix3

            if self.cri_l1:
                h3, l3 = self._get_low_high(self.gt)
                h4, l4 = self._get_low_high(syn_hq)
                h5, l5 = self._get_low_high(syn_hq_1)
                l_g_l2 = self.cri_l1(h3, h4)
                l_g_l3 = self.cri_l1(h3, h5)
                l_g_total += (l_g_l2 + l_g_l3 )
                loss_dict['l_g_l2'] = l_g_l2
                loss_dict['l_g_l3'] = l_g_l3

            if self.cri_perceptual:

                l_g_percep2 = self.cri_dinov3(self.gt, syn_hq_1) * 0.1
                l_g_percep3 = self.cri_dinov3(self.gt, syn_hq) * 0.1
                l_g_percep4 = self.cri_dinov3(syn_hq_2_teacher, syn_hq_2_student) * 0.1
                loss_dict['l_m_percep2'] = l_g_percep2
                loss_dict['l_m_percep3'] = l_g_percep3
                loss_dict['l_m_percep4'] = l_g_percep4
                l_g_total += (l_g_percep2 + l_g_percep3 + l_g_percep4)

            l_g_total.backward()  # retain_graph=True
            # l_m_total.backward(retain_graph=True)
            self.optimizer_g.step()
            self.optimizer_F.step()
            # self.optimizer_m.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(self.net_g, self.net_g_ema, decay=self.ema_decay)

    def _syn_haze(self, clean, depth=None, kernel=None, beta_range=(0.6, 1.8), A_range=(0.7, 1.0), given_A=None,
                  color_range=(-0.025, 0.025), noise_range=(2, 25), jpeg_range=(30, 95), A_dim=3, A_color=0.25, beta_mul=1.0,
                  apply_gamma=False, apply_noise=False, apply_jpeg=False, apply_A_random=False, depth_blur=False, size=21):
        """
        # RIDCP: beta_range: [0.3, 1.5], A_range: [0.25, 1.0], A random: 1.0
        img:   (B, C, H, W), float, [0, 1]
        depth: (B, 1, H, W), float, [0, 1]
        kernel: (B, k, k), float, [0, 1]
        """
        B, C, H, W = clean.shape
        # --- Beta and A random factors ---
        beta = torch.empty([B, 1, 1, 1], device=self.device).uniform_(*beta_range)

        if given_A is None:
            # A = torch.empty([B, 3, 1, 1], device=self.device).uniform_(*A_range)
            A = torch.empty([B, A_dim, 1, 1], device=self.device).uniform_(*A_range)
            # --- Beta and A random factors ---
            if torch.rand(1) < A_color and apply_A_random:
                A_random = torch.empty([1, C, 1, 1], device=self.device).uniform_(*color_range)
                A = A + A_random  # broadcast to (B, C, 1, 1)
        else:
            A = given_A

        # --- Initialize img_lq as clean ---
        img_lq = clean.clone()
        # --- Random gamma adjustment ---
        if torch.rand(1) < 0.5 and apply_gamma:
            img_lq = torch.pow(img_lq, torch.rand(1).item() * 1.5 + 1.5)
        # --- Add Gaussian noise ---
        if torch.rand(1) < 0.5 and apply_noise:
            noise_level = random.randint(*noise_range)
            sigma = noise_level / 255.0
            rnum = torch.rand(1).item()
            if rnum > 0.6:  # add color Gaussian noise
                noise = torch.randn_like(img_lq) * sigma
            elif rnum < 0.4:  # add grayscale Gaussian noise
                gray_noise = torch.randn(B, 1, H, W, device=self.device) * sigma
                noise = gray_noise.expand(-1, C, -1, -1)
            else:  # add multivariate Gaussian noise
                L = noise_range[1] / 255.0
                noise = torch.zeros_like(img_lq)
                for b in range(B):
                    D = torch.diag(torch.rand(C, device=self.device))
                    u, _, v = torch.linalg.svd(torch.rand(C, C, device=self.device), full_matrices=False)
                    U = torch.mm(u, v)
                    # U = orth_torch(torch.rand(C, C, device=self.device))
                    cov = U.T @ D @ U
                    # Sample multivariate normal per pixel
                    flat_noise = torch.distributions.MultivariateNormal(
                        loc=torch.zeros(C, device=self.device),
                        covariance_matrix=torch.abs(L ** 2 * cov)
                    ).sample((H * W,))
                    noise_sample = flat_noise.T.view(C, H, W)
                    noise[b] = noise_sample
            img_lq = (img_lq + noise).clamp(0.0, 1.0)

        depth = self._depth_estimate(clean, blur=depth_blur, size=size) if depth is None else depth
        # depth = filter2D(depth, kernel)

        t = torch.exp(-beta * beta_mul * (1 - depth))
        # t = torch.exp(-beta * min((1 - depth), 0.7))

        # I(x) = J(x)t(x) + A(1-t(x))
        hazy = img_lq * t + A * (1 - t)

        if torch.rand(1) < 0.5 and apply_jpeg:
            jpeg_p = torch.zeros(hazy.shape[0], device=hazy.device, dtype=hazy.dtype).uniform_(*jpeg_range)
            hazy = self.jpeger(hazy.clamp(0, 1), quality=jpeg_p)

        return hazy, beta, A, t, depth

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        # if hasattr(self, 'net_m_ema'):
        #     self.save_network([self.net_m, self.net_m_ema], 'net_m', current_iter, param_key=['params', 'params_ema'])
        # else:
        #     self.save_network(self.net_m, 'net_m', current_iter)

        # self.save_network(self.net_d, 'net_d', current_iter)
        self.save_network(self.netF, 'net_F', current_iter)
        # self.save_network(self.net_dehaze_d, 'net_dehaze_d', current_iter)
        # # self.save_training_state(epoch, current_iter)

    def _random_hazy_style(self, jitter=(-0.025, 0.025)):
        haze_color_map = {
            # 基础色调
            'neutral': (1.00, 1.00, 1.00),  # 清晨无色薄雾，最常见

            # 暖色系
            'warm_yellow': (1.00, 0.95, 0.85),  # 日出、秋季山林、晨雾
            'orange_dust': (0.95, 0.85, 0.70),  # 沙尘天气、干旱区黄沙
            'urban_smog': (0.90, 0.85, 0.80),  # 城市污染雾霾、尾气残留
            'wildfire': (1.00, 0.75, 0.55),  # 山火烟雾、火灾附近区域

            # 冷色系
            'cool_blue': (0.85, 0.90, 1.00),  # 清晨森林/山谷水汽雾
            'night_blue': (0.70, 0.75, 1.00),  # 夜间/黎明低温湿雾（偏冷）
            'glacier_blue': (0.80, 0.90, 0.95),  # 冰川湖区、雪地反射雾气
            'sea_mist': (0.88, 0.93, 0.97),  # 海边晨雾，含海盐颗粒，偏淡蓝

            # 特殊色调
            'green_pollen': (0.88, 1.00, 0.88),  # 春季植物大量繁殖，粉尘偏绿
            'brown_industrial': (0.70, 0.65, 0.60),  # 工业区废气排放型雾霾
            'volcanic_ash': (0.65, 0.60, 0.55),  # 火山喷发后的雾霾混合物

            # 极端天气
            'toxic_purple': (0.85, 0.80, 0.90),  # 模拟科幻/极端污染（影视类）
            'sunset_pink': (1.00, 0.80, 0.85),  # 黄昏薄雾叠加晚霞，偏红/粉
        }
        random_key = random.choice(list(haze_color_map.keys()))
        random_tuple = haze_color_map[random_key]
        random_tuple = torch.tensor(random_tuple, device=self.device).reshape(1, 3, 1, 1)

        if torch.rand(1) < 0.5:
            A_random = torch.empty([1, 3, 1, 1], device=self.device).uniform_(*jitter)
            random_tuple = random_tuple + A_random  # broadcast to (B, C, 1, 1)

        return random_tuple