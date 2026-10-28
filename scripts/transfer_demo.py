import random
import shutil
from natsort import os_sorted
from basicsr.utils.misc import set_random_seed
import cv2
import torch
import numpy as np
import os
from PIL import Image
from scripts.utils_inference import load, uint2tensor, tensor2uint8
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from scripts.utils_func import AverageMeter
device = torch.device("cuda")

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


inp = {
    'SOTS_Outdoor': [
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\Outdoor\input',
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\Outdoor\target',
    ],
    'SOTS_Indoor': [
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\Indoor\input',
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\Indoor\target',
    ],
    'HSTS_syn': [
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\HSTS\synthetic',
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\HSTS\original',
    ],
    'NYU_Seg_Haze': [
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\D_Haze\NYU_Hazy',
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\D_Haze\NYU_GT',
    ],
    'I_haze': [
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\I_haze\Input',
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\I_haze\GT',
    ],
    'O_haze': [
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\O_haze\Input',
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\O_haze\GT',
    ],
    'NH_haze': [
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\NH_haze\input_down2',
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\NH_haze\target_down2',
    ],
    'Smoke': [
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\Smoke\hazy',
        r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy\Smoke\clean',
    ],
}


def _init_dgfd():
    from basicsr.archs.DGFDNet_arch import build_net
    net_dgfd = build_net(
        weights=r'G:\PhD\Paper\Ideal\Dehaze\Dehazing\weights\oknet\DGFDNet_ckpt\ots_best.pth'
    ).to(device).eval()
    net_dgfd.requires_grad_(False)
    return net_dgfd


def _init_dlm(weight, key='params'):
    from basicsr.archs.DLM_arch import DLMNet
    net_dlm = DLMNet().to(device).eval()
    net_dlm.load_state_dict(
        torch.load(weight, map_location=device)[key]
    )
    return net_dlm


def _init_csud(weight, key='params'):
    from basicsr.archs.csud_arch import _NetG_DOWN_init
    net_csud = _NetG_DOWN_init().cuda()
    net_csud.load_state_dict(
        torch.load(weight, map_location=device)[key]
    )
    return net_csud


def _scan_dir(gt1_path, lq1_path, gt2_path):
    if os.path.isdir(gt1_path):
        gt1_path = [os.path.join(gt1_path, _) for _ in os.listdir(gt1_path)]
        lq1_path = [os.path.join(lq1_path, _) for _ in os.listdir(lq1_path)]
        gt2_path = [os.path.join(gt2_path, _) for _ in os.listdir(gt2_path)]
        gt1_path = os_sorted(gt1_path)
        lq1_path = os_sorted(lq1_path)
        random.shuffle(gt2_path)
    else:
        gt1_path = [gt1_path]
        lq1_path = [lq1_path]
        gt2_path = [gt2_path]
    return gt1_path, lq1_path, gt2_path


@torch.inference_mode()
def _transfer_demo(gt1_path, lq1_path, gt2_path, size=(512, 512), save_path=''):
    gt1_path, lq1_path, gt2_path = _scan_dir(gt1_path, lq1_path, gt2_path)

    os.makedirs(save_path, exist_ok=True)

    # dlm_save_path = os.path.join(save_path, 'dlm_transfer')
    # csud_save_path = os.path.join(save_path, 'csud_transfer')
    # os.makedirs(dlm_save_path, exist_ok=True)
    # os.makedirs(csud_save_path, exist_ok=True)

    for i, (gt1, lq1, gt2) in enumerate(zip(gt1_path, lq1_path, gt2_path)):
        base_gt1, ext_gt1 = os.path.splitext(os.path.basename(gt1))
        base_lq1, ext_lq1 = os.path.splitext(os.path.basename(lq1))
        base_gt2, ext_gt2 = os.path.splitext(os.path.basename(gt2))

        gt1_uint8 = np.array(Image.open(gt1).convert('RGB').resize(size, Image.Resampling.BICUBIC))
        gt1_tensor = uint2tensor(gt1_uint8).to(device)
        lq1_uint8 = np.array(Image.open(lq1).convert('RGB').resize(size, Image.Resampling.BICUBIC))
        lq1_tensor = uint2tensor(lq1_uint8).to(device)
        gt2_uint8 = np.array(Image.open(gt2).convert('RGB').resize(size, Image.Resampling.BICUBIC))
        gt2_tensor = uint2tensor(gt2_uint8).to(device)

        _, _, dlm_gt1_gt1 = net_dlm(gt1_tensor, gt1_tensor)
        _, _, dlm_lq1_lq1 = net_dlm(lq1_tensor, lq1_tensor)
        _, _, dlm_gt1_lq1 = net_dlm(gt1_tensor, lq1_tensor)
        _, _, dlm_gt2_lq1 = net_dlm(gt2_tensor, lq1_tensor)

        csud_gt1_gt1 = net_csud(gt1_tensor, gt1_tensor)
        csud_lq1_lq1 = net_csud(lq1_tensor, lq1_tensor)
        csud_gt1_lq1 = net_csud(gt1_tensor, lq1_tensor)
        csud_gt2_lq1 = net_csud(gt2_tensor, lq1_tensor)

        dlm_show = np.concatenate([gt2_uint8, lq1_uint8, tensor2uint8(dlm_gt2_lq1)], axis=1)
        csud_show = np.concatenate([gt2_uint8, lq1_uint8, tensor2uint8(csud_gt2_lq1)], axis=1)

        cv2.imwrite(os.path.join(save_path, f'{i}_dlm_gt2_lq1_{base_gt2}_{base_lq1}{ext_gt2}'), dlm_show[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, f'{i}_csud_gt2_lq1_{base_gt2}_{base_lq1}{ext_gt2}'), csud_show[:, :, ::-1])

        # cv2.imwrite(os.path.join(dlm_save_path, f'{i}_gt1_gt1_{base_gt1}_{base_gt1}{ext_gt1}'), tensor2uint8(dlm_gt1_gt1)[:, :, ::-1])
        # cv2.imwrite(os.path.join(dlm_save_path, f'{i}_lq1_lq1_{base_lq1}_{base_lq1}{ext_lq1}'), tensor2uint8(dlm_lq1_lq1)[:, :, ::-1])
        # cv2.imwrite(os.path.join(dlm_save_path, f'{i}_gt1_lq1_{base_gt1}_{base_lq1}{ext_gt1}'), tensor2uint8(dlm_gt1_lq1)[:, :, ::-1])
        # cv2.imwrite(os.path.join(dlm_save_path, f'{i}_gt2_lq1_{base_gt2}_{base_lq1}{ext_gt2}'), tensor2uint8(dlm_gt2_lq1)[:, :, ::-1])
        #
        # cv2.imwrite(os.path.join(csud_save_path, f'{i}_gt1_gt1_{base_gt1}_{base_gt1}{ext_gt1}'), tensor2uint8(csud_gt1_gt1)[:, :, ::-1])
        # cv2.imwrite(os.path.join(csud_save_path, f'{i}_lq1_lq1_{base_lq1}_{base_lq1}{ext_lq1}'), tensor2uint8(csud_lq1_lq1)[:, :, ::-1])
        # cv2.imwrite(os.path.join(csud_save_path, f'{i}_gt1_lq1_{base_gt1}_{base_lq1}{ext_gt1}'), tensor2uint8(csud_gt1_lq1)[:, :, ::-1])
        # cv2.imwrite(os.path.join(csud_save_path, f'{i}_gt2_lq1_{base_gt2}_{base_lq1}{ext_gt2}'), tensor2uint8(csud_gt2_lq1)[:, :, ::-1])


if __name__ == '__main__':
    net_dlm = _init_dlm(
        weight=r'../weights/DegradationTransfer_PSUID.pth',
        key='params'
    ).to(device).eval()

    net_csud = _init_csud(
        weight=r'../weights/DegradationTransfer_CSUD.pth',
        key='params'
    ).to(device).eval()

    _transfer_demo(
        gt1_path=r'../pics/gts',
        lq1_path=r'../pics/lqs',
        gt2_path=r'../pics/gts',
        save_path=r'../pics/transfer',
        size=(512, 512),
    )