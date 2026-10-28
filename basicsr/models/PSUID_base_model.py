import os.path
import random
import numpy as np
import torch
from collections import OrderedDict
import torch.nn.functional as F
import kornia
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
ROOT = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir))


@MODEL_REGISTRY.register()
class PSUIDBaseModel(SRModel):
    """SRGAN model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        # For derain
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        # self.print_network(self.net_g)

        # For degradation migration
        self.net_m = build_network(opt['network_m'])
        self.net_m = self.model_to_device(self.net_m)

        # For migration discriminator
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)

        # For dehazing discriminator
        self.net_dehaze_d = build_network(self.opt['net_dehaze_d'])
        self.net_dehaze_d = self.model_to_device(self.net_dehaze_d)

        # For, F
        from basicsr.archs.patchsamplef import PatchSampleF
        self.netF = PatchSampleF(use_mlp=True, init_type='xavier', init_gain=0.02, gpu_ids=[0], nc=256)
        print(f'F: ======================== netF init')
        gt_size = self.opt['datasets']['train']['gt_size']
        bs = self.opt['datasets']['train']['batch_size_per_gpu']
        # with torch.inference_mode():
        encoder_feats = self.net_g(torch.zeros((bs, 3, gt_size, gt_size), device=self.device), encoder_only=True)
        feat_k_pool, sample_ids = self.netF(encoder_feats, 256, None)
        print(f'F: ======================== {self.netF.state_dict().keys()}')


        # encoder_feats = [
        #     torch.randn((1, 32, gt_size, gt_size), device=self.device),
        #     torch.randn((1, 64, gt_size // 2, gt_size // 2), device=self.device),
        #     torch.randn((1, 128, gt_size // 4, gt_size // 4), device=self.device),
        #     torch.randn((1, 256, gt_size // 8, gt_size // 8), device=self.device),
        #     torch.randn((1, 512, gt_size // 16, gt_size // 16), device=self.device),
        # ]
        """
        torch.Size([1, 32, 512, 512])
        torch.Size([1, 64, 256, 256])
        torch.Size([1, 128, 128, 128])
        torch.Size([1, 256, 64, 64])
        torch.Size([1, 512, 32, 32])
        """

        # load pretrained models
        load_path_g = self.opt['path'].get('pretrain_network_g', None)
        print(f'G: ======================== {load_path_g}')
        if load_path_g is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path_g, self.opt['path'].get('strict_load_g', True), param_key)

        load_path_m = self.opt['path'].get('pretrain_network_m', None)
        print(f'M: ======================== {load_path_m}')
        if load_path_m is not None:
            param_key = self.opt['path'].get('param_key_m', 'params')
            self.load_network(self.net_m, load_path_m, self.opt['path'].get('strict_load_m', True), param_key)

        load_path_d = self.opt['path'].get('pretrain_network_d', None)
        if load_path_d is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path_d, self.opt['path'].get('strict_load_d', True), param_key)

        load_path_F = self.opt['path'].get('pretrain_network_F', None)
        if load_path_F is not None:
            param_key = self.opt['path'].get('param_key_F', 'params')
            self.load_network(self.netF, load_path_F, self.opt['path'].get('strict_load_F', True), param_key)

        load_path_dehaze_d = self.opt['path'].get('pretrain_network_dehaze_d', None)
        if load_path_dehaze_d is not None:
            param_key = self.opt['path'].get('param_key_dehaze_d', 'params')
            self.load_network(self.net_dehaze_d, load_path_dehaze_d, self.opt['path'].get('strict_load_dehaze_d', True), param_key)

        if self.is_train:
            self.init_training_settings()

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.float).reshape(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.float).reshape(3, 1, 1)
        self.depthv2 = _depth_v2_interface(
            root=self.opt['train']['depthv2_path'],
            device=self.device,
            encoder='vitl'  # 'vits', 'vitb', 'vitl'
        )
        self.dcp = TransmissionEstimator().to(self.device)
        self.jpeger = DiffJPEG(differentiable=False).to(self.device)

    def model_ema(self, net, net_ema, decay=0.999):
        net_g = self.get_bare_model(net)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(net_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def _model_set(self):
        self.net_g.train()
        self.net_m.train()
        self.net_d.train()
        self.net_dehaze_d.train()
        self.netF.train()

    def setup_loss(self, train_opt):
        # define losses
        if train_opt.get('chroma_opt'):
            self.cri_chroma = build_loss(train_opt['chroma_opt']).to(self.device)
        else:
            self.cri_chroma = None

        if train_opt.get('ssim_opt'):
            self.cri_ssim = build_loss(train_opt['ssim_opt']).to(self.device)
        else:
            self.cri_ssim = None

        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('l2_opt'):
            self.cri_l2 = build_loss(train_opt['l2_opt']).to(self.device)
        else:
            self.cri_l2 = None

        if train_opt.get('constrast_opt'):
            self.cri_constrast = build_loss(train_opt['constrast_opt']).to(self.device)
        else:
            self.cri_constrast = None

        if train_opt.get('fft_opt'):
            self.cri_fft = build_loss(train_opt['fft_opt']).to(self.device)
        else:
            self.cri_fft = None

        if train_opt.get('l1_opt'):
            self.cri_l1 = build_loss(train_opt['l1_opt']).to(self.device)
        else:
            self.cri_l1 = None

        if train_opt.get('color_opt'):
            self.cri_color = build_loss(train_opt['color_opt']).to(self.device)
        else:
            self.cri_color = None

        if train_opt.get('range_opt'):
            self.cri_range = build_loss(train_opt['range_opt']).to(self.device)
        else:
            self.cri_range = None

        if train_opt.get('ldl_opt'):
            self.cri_ldl = build_loss(train_opt['ldl_opt']).to(self.device)
        else:
            self.cri_ldl = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('style_opt'):
            self.cri_style = build_loss(train_opt['style_opt']).to(self.device)
        else:
            self.cri_style = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)
        else:
            self.cri_gan = None

        if train_opt.get('bgm_opt'):
            self.cri_bgm = build_loss(train_opt['bgm_opt']).to(self.device)
        else:
            self.cri_bgm = None

        from basicsr.losses.patchnce import PatchNCELoss
        self.criterionNCE = [PatchNCELoss(self.opt['datasets']['train']['batch_size_per_gpu']) for i in range(5)]

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # self.net_m_ema = build_network(self.opt['network_m']).to(self.device)

            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(self.net_g, self.net_g_ema, 0)  # copy net_g weight
            self.net_g_ema.eval()

            # load_path = self.opt['path'].get('pretrain_network_m', None)
            # if load_path is not None:
            #     self.load_network(self.net_m_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            # else:
            #     self.model_ema(self.net_m, self.net_m_ema, 0)  # copy net_g weight
            # self.net_m_ema.eval()

        self._model_set()
        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_loss(train_opt)
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        """
        https://github.com/YaN9-Y/D4/blob/main/src/models.py
        self.optimizer = optim.Adam(
                [
                    {'params': self.net_c2h.parameters()},
                    {'params': self.net_h2c.parameters()},

                ],

                lr=float(config.LR),
                betas=(config.BETA1, config.BETA2),
                weight_decay=config.WEIGHT_DECAY
            )
        optimizer = torch.optim.Adam([
            {'params': netA.parameters(), 'lr': 1e-4},
            {'params': netB.parameters(), 'lr': 5e-5}
        ], betas=(0.9, 0.999))
        optimizer = torch.optim.Adam([
            {
                'params': netA.parameters(),
                'lr': 1e-4,
                'betas': (0.9, 0.999),
                'weight_decay': 1e-4
            },
            {
                'params': netB.parameters(),
                'lr': 5e-5,
                'betas': (0.5, 0.999),
                'weight_decay': 0
            }
        ])
        # 修改第一个 param_group（netA）的学习率
        optimizer.param_groups[0]['lr'] = 1e-5

        # 修改第二个 param_group（netB）的 weight_decay
        optimizer.param_groups[1]['weight_decay'] = 1e-6

        https://github.com/Xiaofeng-life/ICDehazing/blob/master/task_ICDehazing/train_ICDehazing.py
        optimizer_G = optim.Adam(itertools.chain(generator_x2y.parameters(), generator_y2x.parameters()),
                             lr=config.g_lr, betas=(config.beta1, config.beta2))
        optimizer_D = optim.Adam(itertools.chain(discriminator_x.parameters(), discriminator_y.parameters()),
                             lr=config.d_lr, betas=(config.beta1, config.beta2))
        """
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer migration
        optim_type = train_opt['optim_m'].pop('type')
        self.optimizer_m = self.get_optimizer(optim_type, self.net_m.parameters(), **train_opt['optim_m'])
        self.optimizers.append(self.optimizer_m)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)
        # optimizer dehaze d
        optim_type = train_opt['optim_dehaze_d'].pop('type')
        self.optimizer_dehaze_d = self.get_optimizer(optim_type, self.net_dehaze_d.parameters(), **train_opt['optim_dehaze_d'])
        self.optimizers.append(self.optimizer_dehaze_d)
        # optimizer F
        optim_type = train_opt['optim_F'].pop('type')
        self.optimizer_F = self.get_optimizer(optim_type, self.netF.parameters(), **train_opt['optim_F'])
        self.optimizers.append(self.optimizer_F)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'lq_strong' in data:
            self.lq_strong = data['lq_strong'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()

        lq, beta, A, t, depth = self._syn_haze(
            clean=self.gt, depth=None, kernel=None, beta_range=(0.6, 1.8), A_range=(0.7, 1.0),
            color_range=(-0.025, 0.025), noise_range=(2, 25), jpeg_range=(30, 95), A_dim=1,  # 3
            apply_gamma=False, apply_noise=False, apply_jpeg=False, apply_A_random=False
        )
        syn_hq = self.net_g(lq)

        if self.opt['train']['save_flag'] and current_iter % self.opt['train']['save_freq'] == 0:
            self._save_images(self.gt, current_iter, '0_gt')
            self._save_images(lq, current_iter, '1_lq_online_syn')
            self._save_images(syn_hq, current_iter, '2_hq_online_syn')

        l_g_total = 0.
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_g_l11 = self.cri_pix(self.gt, syn_hq)
            l_g_total += l_g_l11
            loss_dict['l_g_l1'] = l_g_l11

        l_g_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(self.net_g, self.net_g_ema, decay=self.ema_decay)

    def _syn_haze(self, clean, depth=None, kernel=None, beta_range=(0.6, 1.8), A_range=(0.7, 1.0), given_A=None,
                  color_range=(-0.025, 0.025), noise_range=(2, 25), jpeg_range=(30, 95), A_dim=3, A_color=0.25,
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
        else:
            A = given_A
        # --- Beta and A random factors ---
        if torch.rand(1) < A_color and apply_A_random:
            A_random = torch.empty([1, C, 1, 1], device=self.device).uniform_(*color_range)
            A = A + A_random  # broadcast to (B, C, 1, 1)
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
        t = torch.exp(-beta * (1 - depth))
        # t = torch.exp(-beta * min((1 - depth), 0.7))

        # I(x) = J(x)t(x) + A(1-t(x))
        hazy = img_lq * t + A * (1 - t)

        if torch.rand(1) < 0.5 and apply_jpeg:
            jpeg_p = torch.zeros(hazy.shape[0], device=hazy.device, dtype=hazy.dtype).uniform_(*jpeg_range)
            hazy = self.jpeger(hazy.clamp(0, 1), quality=jpeg_p)

        return hazy, beta, A, t, depth

    def _get_dcp(self, img):
        dcp_result = self.dcp.get_radiance_all(img)
        return dcp_result

    def _depth_estimate(self, clean, **kwargs):
        depth = self.depthv2(clean, **kwargs)  # Bx1xHxW
        return depth

    def _get_channel(self, hazy, clean):
        max_hazy, _ = hazy.max(dim=1, keepdim=True)
        min_hazy, _ = hazy.min(dim=1, keepdim=True)
        max_clean, _ = clean.max(dim=1, keepdim=True)
        min_clean, _ = clean.min(dim=1, keepdim=True)
        range_rain = max_hazy - min_hazy
        range_clean = max_clean - min_clean

    def _save_images(self, img, iter, suffix='lq'):
        save_path = self.opt['path']['visualization']
        # os.makedirs(save_path, exist_ok=True)
        img = img.detach().cpu()
        save_image(img, os.path.join(save_path, f'{iter:07d}_{suffix}.png'))

    def _get_fft_amp_pha(self, image):
        fft = torch.fft.rfft2(image)
        amp = torch.abs(fft)
        pha = torch.angle(fft)
        return amp, pha

    def _swap_fft_amp_pha(self, lq, hq):
        # https://github.com/wangchx67/FourLLIE/blob/main/models/archs/FourLLIE.py
        # https://github.com/manman1995/pansharpening/blob/main/model/SFITNET-ECCV.py
        # https://github.com/pkumivision/FFC/blob/39dcc992c373b0705f64d0204c4c04669381b8ea/model_zoo/ffc.py#L38
        _, _, H, W = lq.shape

        amp_lq, pha_lq = self._get_fft_amp_pha(lq)
        amp_hq, pha_hq = self._get_fft_amp_pha(hq)

        amp_lq_pha_hq = torch.fft.ifft2(
            torch.complex(amp_lq * torch.cos(pha_hq), amp_lq * torch.sin(pha_hq)),
            s=(H, W), norm='forward'
        ).real  # ortho (normalize by ``1/sqrt(n)``) | forward (no normalization) | backward (normalize by ``1/n``)
        amp_hq_pha_lq = torch.fft.ifft2(
            torch.complex(amp_hq * torch.cos(pha_lq), amp_hq * torch.sin(pha_lq)),
            s=(H, W), norm='forward'
        ).real
        return amp_lq_pha_hq, amp_hq_pha_lq

    def _get_fft_amp_pha_new(self, image):
        fft = torch.fft.fft2(image)
        amp = torch.abs(fft)
        pha = torch.angle(fft)
        return amp, pha

    def _swap_fft_amp_pha_new(self, lq, hq):
        # https://github.com/wangchx67/FourLLIE/blob/main/models/archs/FourLLIE.py
        # https://github.com/manman1995/pansharpening/blob/main/model/SFITNET-ECCV.py
        # https://github.com/pkumivision/FFC/blob/39dcc992c373b0705f64d0204c4c04669381b8ea/model_zoo/ffc.py#L38
        def normalize_batch(img):
            img_min = img.amin(dim=(-2, -1), keepdim=True)
            img_max = img.amax(dim=(-2, -1), keepdim=True)
            return (img - img_min) / (img_max - img_min + 1e-8)

        _, _, H, W = lq.shape

        amp_lq, pha_lq = self._get_fft_amp_pha_new(lq)
        amp_hq, pha_hq = self._get_fft_amp_pha_new(hq)

        amp_lq_pha_hq = amp_lq * torch.exp(1j * pha_hq)
        amp_hq_pha_lq = amp_hq * torch.exp(1j * pha_lq)
        amp_lq_pha_hq = torch.fft.ifft2(amp_lq_pha_hq, dim=(-2, -1)).real
        amp_hq_pha_lq = torch.fft.ifft2(amp_hq_pha_lq, dim=(-2, -1)).real

        amp_lq_pha_hq = normalize_batch(amp_lq_pha_hq).clamp(0, 1)
        amp_hq_pha_lq = normalize_batch(amp_hq_pha_lq).clamp(0, 1)
        return amp_lq_pha_hq, amp_hq_pha_lq

    def _mixup(self, lq, hq, factor=0.5):
        factor = torch.FloatTensor([factor]).to(self.device)
        out = lq * factor + hq * (1 - factor)
        return out

    def save(self, epoch, current_iter):
        if self.opt['stage'] == 'first':
            if hasattr(self, 'net_g_ema'):
                self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
            else:
                self.save_network(self.net_g, 'net_g', current_iter)

        elif self.opt['stage'] == 'second':
            if hasattr(self, 'net_g_ema'):
                self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
            else:
                self.save_network(self.net_g, 'net_g', current_iter)
            # if hasattr(self, 'net_m_ema'):
            #     self.save_network([self.net_m, self.net_m_ema], 'net_m', current_iter, param_key=['params', 'params_ema'])
            # else:
            #     self.save_network(self.net_m, 'net_m', current_iter)
            #
            # self.save_network(self.net_d, 'net_d', current_iter)
            self.save_network(self.netF, 'net_F', current_iter)
            self.save_network(self.net_dehaze_d, 'net_dehaze_d', current_iter)
        # self.save_training_state(epoch, current_iter)

        elif self.opt['stage'] == 'third':
            if hasattr(self, 'net_g_ema'):
                self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
            else:
                self.save_network(self.net_g, 'net_g', current_iter)

        else:
            NotImplementedError(f'Not implemented stage: {self.opt["stage"]}')

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            self.output = self.net_g(self.lq)
            self.output = self.output[0] if isinstance(self.output, (tuple, list)) else self.output
        self.net_g.train()

    def _get_low_high(self, img):
        high_freq, low_freq = wavelet_decomposition(img)
        return high_freq, low_freq

    def minmax_depth(self, depth, blur=True):
        # D4
        n, c, h, w = depth.shape
        # depth = F.avg_pool2d(depth,kernel_size=5)

        if blur:  # use median filter to filter out peak values for visualization.
            depth = F.pad(depth, [4, 4, 4, 4], 'reflect')
            depth = kornia.median_blur(depth, (9, 9))
            depth = depth[:, :, 3:h - 3, 3:w - 3]

        D_max = torch.max(depth.reshape(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        D_min = torch.min(depth.reshape(n, c, -1), dim=2, keepdim=True)[0].unsqueeze(3)
        depth = (depth - D_min) / (D_max - D_min)
        return depth
