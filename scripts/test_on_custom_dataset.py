import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from tqdm import tqdm
import numpy as np
from natsort import os_sorted
from basicsr.metrics.psnr_ssim_compute import calculate_psnr_ssim, calculate_psnr_ssim_pt, calculate_psnr_ssim_auto
from scripts.utils_func import (check_dir, AverageMeter, uint2tensor, tensor2uint8, image2patch, expand2square,
                             patch2image, get_time, splitimage, mergeimage)
from scripts.utils_inference import load, _get_paths_from_images
from torch.utils.data import Dataset, DataLoader
import warnings
import torchvision
from torchvision.transforms.functional import InterpolationMode
from scripts.self_ensemble import test_selfensemble
warnings.filterwarnings("ignore")


def _init_nr_metric():
    from pyiqa.api_helpers import create_metric

    # root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    root = r'G:\PhD\Codes\Foundation\IQA'  # /data1/sr617/codes/liux/Foundation/IQA
    musiq = create_metric(metric_name='musiq', device=device,
                          pretrained_model_path=os.path.join(root, r'musiq_koniq_ckpt-e95806b9.pth'))
    maniqa = create_metric(metric_name='maniqa', device=device, weight_keys=None,
                           pretrained_model_path=os.path.join(root, r'ckpt_koniq10k.pt'))
    clipiqa = create_metric(metric_name='clipiqa+_vitL14_512', device=device,
                            pretrained_model_path=os.path.join(root, r'CLIPIQA+_ViTL14_512-e66488f2.pth'))
    topiq = create_metric(metric_name='topiq_nr', device=device,
                          pretrained_model_path=os.path.join(root, r'cfanet_nr_koniq_res50-9a73138b.pth'))

    return musiq, maniqa, clipiqa, topiq


class DehazeDataset(Dataset):
    def __init__(self, root_dir, name='SOTS_Indoor'):
        super(DehazeDataset, self).__init__()
        self.name = name

        if name == 'SOTS_Indoor':
            path = os.path.join(root_dir, f'Indoor/input')
            self.lqs = _get_paths_from_images(path)
        elif name == 'SOTS_Outdoor':
            path = os.path.join(root_dir, f'Outdoor/input')
            self.lqs = _get_paths_from_images(path)
        elif name == 'HSTS_syn':
            self.lqs = _get_paths_from_images(os.path.join(root_dir, f'HSTS/synthetic'))
            self.gts = _get_paths_from_images(os.path.join(root_dir, f'HSTS/original'))
        elif name == 'I_haze':
            self.lqs = _get_paths_from_images(os.path.join(root_dir, f'I_Haze/Input'))
            self.gts = _get_paths_from_images(os.path.join(root_dir, f'I_Haze/GT'))
        elif name == 'O_haze':
            self.lqs = _get_paths_from_images(os.path.join(root_dir, f'O_Haze/Input'))
            self.gts = _get_paths_from_images(os.path.join(root_dir, f'O_Haze/GT'))
        elif name == 'RTTS':  # no gt
            self.lqs = _get_paths_from_images(os.path.join(root_dir, f'RTTS'))
            self.gts = _get_paths_from_images(os.path.join(root_dir, f'RTTS'))
        elif name == 'NYU_Seg_Haze':  # no gt
            self.lqs = _get_paths_from_images(os.path.join(root_dir, f'D_Haze/NYU_Hazy'))
            self.gts = _get_paths_from_images(os.path.join(root_dir, f'D_Haze/NYU_GT'))
        elif name == 'NH_haze':  # no gt
            self.lqs = _get_paths_from_images(os.path.join(root_dir, f'NH_haze/input_down2'))
            self.gts = _get_paths_from_images(os.path.join(root_dir, f'NH_haze/target_down2'))
        elif name == 'Smoke':  # no gt
            self.lqs = _get_paths_from_images(os.path.join(root_dir, f'Smoke/hazy'))
            self.gts = _get_paths_from_images(os.path.join(root_dir, f'Smoke/clean'))
        elif name == 'Fattal':  # no gt
            self.lqs = _get_paths_from_images(os.path.join(root_dir, f'Fattal'))
            self.gts = _get_paths_from_images(os.path.join(root_dir, f'Fattal'))
        elif name == 'HSTS_Real':  # no gt
            self.lqs = _get_paths_from_images(os.path.join(root_dir, f'HSTS/real-world'))
            self.gts = _get_paths_from_images(os.path.join(root_dir, f'HSTS/real-world'))
        elif name == 'URHI':  # no gt
            self.lqs = _get_paths_from_images(os.path.join(root_dir, f'URHI'))
            self.gts = _get_paths_from_images(os.path.join(root_dir, f'URHI'))

    def __getitem__(self, item):
        lq_path = self.lqs[item]
        if self.name == 'SOTS_Indoor':
            # lq: RESIDE\SOTS\indoor\input\1400_1.png
            # gt: RESIDE\SOTS\indoor\target\1400.png
            dir, (base, ext) = os.path.dirname(lq_path), os.path.splitext(os.path.basename(lq_path))
            gt_path = os.path.join(dir.replace('input', 'target'), f'{base.split("_")[0]}{ext}')
        elif self.name == 'SOTS_Outdoor':
            # lq: RESIDE\SOTS\outdoor\input\0001_0.8_0.2.jpg
            # gt: RESIDE\SOTS\outdoor\target\0001.png
            dir, (base, ext) = os.path.dirname(lq_path), os.path.splitext(os.path.basename(lq_path))
            gt_path = os.path.join(dir.replace('input', 'target'), f'{base.split("_")[0]}.png')
        else:
            gt_path = self.gts[item]

        gt = cv2.imread(gt_path)[:, :, ::-1].astype(np.float32) / 255.
        lq = cv2.imread(lq_path)[:, :, ::-1].astype(np.float32) / 255.
        return self.np2tensor(lq), self.np2tensor(gt), os.path.basename(lq_path)

    def __len__(self):
        return len(self.lqs)

    def np2tensor(self, img):
        return torch.from_numpy(img.transpose(2, 0, 1)).float()


@torch.no_grad()
def test_demo(model, device, input_path, save_path, suffix=''):
    model.to(device)
    model.eval()

    if os.path.isdir(input_path):
        lr_imgs = _get_paths_from_images(input_path)
    else:
        lr_imgs = input_path if isinstance(input_path, (tuple, list)) else [input_path]

    for im in tqdm(lr_imgs):
        base, ext = os.path.splitext(os.path.basename(im))
        lr = uint2tensor(cv2.imread(im)[:, :, ::-1]).to(device)
        output = model(lr)
        sr = tensor2uint8(output)
        cv2.imwrite(os.path.join(save_path, f'{base}{suffix}{ext}'), sr[:, :, ::-1])


def check_image_size(x, padder_size=4):
    _, _, h, w = x.size()
    mod_pad_h = (padder_size - h % padder_size) % padder_size
    mod_pad_w = (padder_size - w % padder_size) % padder_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
    return x


@torch.inference_mode()
def test_on_custom_dataset_dl(
        dataloader,
        model,
        device,
        crop_border=0,
        test_y_channel=True,
        save=False,
        save_path='',
        dataset_name='Set5',
        save_suffix='',
        fr_metric=True,
        nr_metric=False,
        resize_size=None,
        ensemble=False
):
    if save:
        save_path_ = os.path.join(save_path, f'{dataset_name}')
        check_dir(save_path_)

    model.to(device)
    model.eval()

    psnr, ssim = AverageMeter(), AverageMeter()
    len_dl = len(dataloader)

    musiq_meter, maniqa_meter, clipiqa_meter, topiq_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    # if resize_size is not None:
    #     resize = torchvision.transforms.Resize(resize_size, InterpolationMode.BICUBIC)  # (256, 256) (2160 // 8, 3840 // 8)
    # else:
    #     resize = nn.Identity()

    for i, (lr_img, hr_img, basename) in enumerate(dataloader):
    # for i, (lr_img, hr_img, basename) in tqdm(enumerate(dataloader)):
        base, ext = os.path.splitext(basename[0])
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)
        # lr_img, hr_img = resize(lr_img), resize(hr_img)

        if ensemble:
            output = test_selfensemble(lr_img, model)
        else:
            output = model(lr_img)

        output = output[0] if isinstance(output, tuple) else output

        if fr_metric:
            psnr_temp, ssim_temp = calculate_psnr_ssim_auto(output, hr_img, crop_border=crop_border,
                                                            test_y=test_y_channel,
                                                            input_order='HWC', color_order='RGB', mode='np')
            psnr.update(psnr_temp)
            ssim.update(ssim_temp)

            # print(f'Processing {i}/{len_dl}: LR:{base}{ext} | HR:{hr_img} | PSNR/SSIM:{psnr_temp:.4f}/{ssim_temp:.4f}')
            # print(f'Processing {i}/{len_dl}: LR:{base}{ext} | PSNR/SSIM:{psnr_temp:.4f}/{ssim_temp:.4f}')

        if nr_metric:
            output = output.clamp(0., 1.)
            musiq_score = musiq(output)
            maniqa_score = maniqa(output)
            clipiqa_score = clipiqa(output)
            topiq_score = topiq(output)
            musiq_meter.update(musiq_score)
            maniqa_meter.update(maniqa_score)
            clipiqa_meter.update(clipiqa_score)
            topiq_meter.update(topiq_score)

        if save:
            output_copy = tensor2uint8(output)
            cv2.imwrite(os.path.join(save_path_, f'{base}_{save_suffix}{ext}'), output_copy[:, :, ::-1])

        # with open(f'{model_name}_{dataset_name}_psnr.txt', 'a+') as f:
        #     f.write(f'{base} {psnr_temp:.4f}/{ssim_temp:.4f}\n')

    if fr_metric:
        avg_psnr = psnr.avg
        avg_ssim = ssim.avg
        print(f'{get_time()} {dataset_name} Avg PSNR:{avg_psnr} | Avg SSIM: {avg_ssim}')

    if nr_metric:
        avg_musiq = musiq_meter.avg
        avg_maniqa = maniqa_meter.avg
        avg_clipiqa = clipiqa_meter.avg
        avg_topiq = topiq_meter.avg
        print(f'{get_time()} {dataset_name} Avg MUSIQ:{avg_musiq} | Avg MANIQA: {avg_maniqa} | Avg CLPIQA: {avg_clipiqa} | Avg TOPIQ: {avg_topiq}')


if __name__ == '__main__':
    from basicsr.archs.NAFNet_arch import NAFNet

    root = r'G:\PhD\Paper\Ideal\Dehaze\Results\hazy'  # os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    musiq, maniqa, clipiqa, topiq = _init_nr_metric()

    model_name = 'PSUID'
    model = NAFNet(width=32, enc_blk_nums=[2, 2, 4, 8], middle_blk_num=12, dec_blk_nums=[ 2, 2, 2, 2 ]).to(device)
    load(
        r'../weights/PSUID_bs4.pth',
        model, 'params', False
    )

    dataset_lists = [
        'SOTS_Outdoor',
        'SOTS_Indoor',
        'HSTS_syn',
        'NYU_Seg_Haze',
        'O_haze',
        'I_haze',
        'NH_haze',
        'Smoke',
        # 'RTTS',
        # 'Fattal',
        # 'HSTS_Real',
        # 'URHI'
    ]

    for d in dataset_lists:
        print(f'{get_time()}  Processing {d}')
        dataloader = DataLoader(dataset=DehazeDataset(root_dir=root, name=d), batch_size=1, shuffle=False, num_workers=4)

        test_on_custom_dataset_dl(
            # Custom LR/HR images dir path
            dataset_name=d,
            dataloader=dataloader,
            # Selected model
            model=model,
            device=device,
            # Test PSNR/SSIM configs
            crop_border=0,
            test_y_channel=False,
            # Save output or not
            save=False,
            save_path=r'',
            save_suffix=model_name,
            fr_metric=True,
            nr_metric=False,
            resize_size=None,
            ensemble=False
        )