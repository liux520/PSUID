from os import path as osp
import random, os, cv2
import numpy as np
from PIL import Image
import torch
from natsort import os_sorted
from torch.utils import data as data
from torchvision.transforms.functional import normalize
# import albumentations
from basicsr.data.data_util import paths_from_lmdb
from basicsr.utils import FileClient, imfrombytes, img2tensor, rgb2ycbcr, scandir, bgr2ycbcr
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils.registry import DATASET_REGISTRY
from PIL import Image
import torchvision.transforms as T
from typing import List, Dict, Tuple
from basicsr.data.randaugment import RandAugment
import torchvision.transforms as transforms


def collect_roots(opt: Dict, prefix: str) -> List[str]:
    """
    从配置中根据前缀(prefix)自动收集 dataroot_gt1, dataroot_gt2, ...
    或 dataroot_lq1, dataroot_lq2, ... 等字段。

    例如：
        prefix='dataroot_gt' 时，会收集 dataroot_gt, dataroot_gt1, dataroot_gt2, ...
        prefix='dataroot_lq' 时，会收集 dataroot_lq, dataroot_lq1, dataroot_lq2, ...
    """
    items: List[Tuple[int, str]] = []

    for k, v in opt.items():
        if not k.startswith(prefix):
            continue
        suffix = k[len(prefix):]  # 例如 'dataroot_gt2' -> '2'
        if suffix == "":          # 兼容 dataroot_gt 没有数字的情况
            idx = 1
        else:
            try:
                idx = int(suffix)
            except ValueError:
                # 如果不是数字结尾，直接跳过（也可以根据需要抛错）
                continue
        items.append((idx, v))

    # 按索引排序：dataroot_gt1, dataroot_gt2, ...
    items.sort(key=lambda x: x[0])

    return [path for _, path in items]


@DATASET_REGISTRY.register()
class UnpairedImageDataset(data.Dataset):

    def __init__(self, opt):
        super(UnpairedImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        # self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']

        self.lq_enlarge_ratio = opt['lq_enlarge_ratio'] if 'lq_enlarge_ratio' in opt else 1
        self.gt_enlarge_ratio = opt['gt_enlarge_ratio'] if 'gt_enlarge_ratio' in opt else 1

        # 1) 自动收集所有 dataroot_gt / dataroot_lq 根目录
        self.gt_folder = collect_roots(opt, "dataroot_gt")
        self.lq_folder = collect_roots(opt, "dataroot_lq")

        if len(self.gt_folder) == 0:
            raise ValueError("No dataroot_gt found in opt.")
        if len(self.lq_folder) == 0:
            raise ValueError("No dataroot_lq found in opt.")

        # 2) 遍历所有根目录，拼成一个大的列表
        self.gt_paths = []
        for gtf in self.gt_folder:
            self.gt_paths.extend(os_sorted(list(scandir(gtf, full_path=True))))
        random.shuffle(self.gt_paths)

        self.lq_paths = []
        for lqf in self.lq_folder:
            self.lq_paths.extend(os_sorted(list(scandir(lqf, full_path=True))))
        random.shuffle(self.lq_paths)

        if len(self.gt_paths) == 0:
            raise RuntimeError("No GT images found under given dataroot_gt*.")
        if len(self.lq_paths) == 0:
            raise RuntimeError("No LQ images found under given dataroot_lq*.")

        # self.lq_paths = sorted(
        #     list(scandir(opt['dataroot_lq1'], full_path=True))
        #     + list(scandir(opt['dataroot_lq2'], full_path=True)) * 200
        #     if opt['dataroot_lq2'] != ''
        #     else list(scandir(opt['dataroot_lq1'], full_path=True))
        # ) * int(self.lq_enlarge_ratio)  # [:5000]
        # random.shuffle(self.lq_paths)
        # self.gt_paths = sorted(
        #     list(scandir(opt['dataroot_gt1'], full_path=True))
        #     + list(scandir(opt['dataroot_gt2'], full_path=True)) * 200
        #     if opt['dataroot_gt2'] != ''
        #     else list(scandir(opt['dataroot_gt1'], full_path=True))
        # ) * int(self.gt_enlarge_ratio)  # [5000:10000]
        # random.shuffle(self.gt_paths)

        # self.lq_path_RTTS = sorted(list(scandir(opt['dataroot_lq_RTTS'], full_path=True)))[:900]
        # self.lq_path_URHI = sorted(list(scandir(opt['dataroot_lq_URHI'], full_path=True)))[:900]
        # self.lq_paths = self.lq_path_RTTS + self.lq_path_URHI
        # random.shuffle(self.lq_paths)
        #
        # self.gt_path_ITS = sorted(list(scandir(opt['dataroot_gt_ITS'], full_path=True)))[:900]
        # self.gt_path_OTS = sorted(list(scandir(opt['dataroot_gt_OTS'], full_path=True)))[:900]
        # self.gt_paths = self.gt_path_ITS + self.gt_path_OTS
        # random.shuffle(self.gt_paths)

        self.len_lqs, self.len_gts = len(self.lq_paths), len(self.gt_paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        lq_idx = index % self.len_lqs
        gt_idx = index % self.len_gts

        # load lq / gt image
        lq_path = self.lq_paths[lq_idx]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        gt_path = self.gt_paths[gt_idx]  # random.choice(self.gt_paths)
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train' or self.opt['phase'] == 'train1' or self.opt['phase'] == 'train2':
            gt_size = self.opt['gt_size']
            if self.opt['pad_or_resize_mode'] == 'pad':
                img_gt, img_lq = padding(img_gt, gt_size), padding(img_lq, gt_size)
            else:
                img_gt, img_lq = resize_and_crop(img_gt, (gt_size, gt_size), crop=False), resize_and_crop(img_lq, (gt_size, gt_size), crop=False)
            # random crop
            img_lq = _random_crop_patch(img_lq, gt_size)
            img_gt = _random_crop_patch(img_gt, gt_size)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            img_gt = augment([img_gt], self.opt['use_hflip'], self.opt['use_rot'])
            img_lq = augment([img_lq], self.opt['use_hflip'], self.opt['use_rot'])

            # # CLAHE
            # img_lq_clahe = (img_lq.clip(0., 1.) * 255.).round().astype(np.uint8)
            # img_lq_clahe = albumentations.CLAHE(clip_limit=1,
            #                                     tile_grid_size=(8, 8),
            #                                     always_apply=True,
            #                                     p=1)(image=img_lq_clahe)["image"]
            # img_lq_clahe = img_lq_clahe.astype(np.float32) / 255.

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        # img_lq_clahe = img2tensor(img_lq_clahe, bgr2rgb=True, float32=True)

        """Augment the image tensor."""
        strong_aug = transforms.Compose([
            # TO PIL
            transforms.ToPILImage(),
            RandAugment(2, 10),
            # TO TENSOR
            transforms.ToTensor()
        ])
        img_lq_strong = strong_aug(img_lq)
        # if len(img_lq_strong.shape) == 3:
        #     img_lq_strong = img_lq_strong.unsqueeze(0)



        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
            # normalize(img_lq_clahe, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'lq_path': lq_path, 'gt': img_gt, 'gt_path': gt_path, 'lq_strong': img_lq_strong}  # , 'lq_clahe': img_lq_clahe}

    def __len__(self):
        return self.len_lqs if self.opt['return_lqs_len'] else self.len_gts


@DATASET_REGISTRY.register()
class DehazeImageTestDataset(data.Dataset):

    def __init__(self, opt):
        super(DehazeImageTestDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        # self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.lq_folder = opt['dataroot_lq']
        # self.lq_enlarge_ratio = opt['lq_enlarge_ratio'] if 'lq_enlarge_ratio' in opt else 1
        self.lq_paths = sorted(list(scandir(self.lq_folder, full_path=True)))
        self.gt_suffix = opt['gt_suffix']

        self.len_lqs = len(self.lq_paths)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        lq_idx = index % self.len_lqs

        # load lq / gt image
        lq_path = self.lq_paths[lq_idx]
        img_bytes = self.file_client.get(lq_path, 'lq')
        img_lq = imfrombytes(img_bytes, float32=True)

        dir, (base, ext) = osp.dirname(lq_path), osp.splitext(osp.basename(lq_path))
        gt_path = os.path.join(dir.replace('input', 'target'), f'{base.split("_")[0]}{ext}')
        if not os.path.isfile(gt_path):
            gt_path = os.path.join(dir.replace('input', 'target'), f'{base.split("_")[0]}{self.gt_suffix}')
        img_bytes = self.file_client.get(gt_path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # augmentation for training
        if self.opt['phase'] == 'train' or self.opt['phase'] == 'train1' or self.opt['phase'] == 'train2':
            gt_size = self.opt['gt_size']
            # random crop
            img_lq = _random_crop_patch(img_lq, gt_size)
            img_gt = _random_crop_patch(img_gt, gt_size)
            # flip, rotation
            # img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_hflip'], self.opt['use_rot'])
            img_gt = augment([img_gt], self.opt['use_hflip'], self.opt['use_rot'])
            img_lq = augment([img_lq], self.opt['use_hflip'], self.opt['use_rot'])

        # color space transform
        if 'color' in self.opt and self.opt['color'] == 'y':
            img_lq = bgr2ycbcr(img_lq, y_only=True)[..., None]
            img_gt = bgr2ycbcr(img_gt, y_only=True)[..., None]

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True)
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            normalize(img_gt, self.mean, self.std, inplace=True)
        return {'lq': img_lq, 'lq_path': lq_path, 'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return self.len_lqs


def _random_crop_patch(img, patch_size):
    if isinstance(img, np.ndarray):
        h, w, c = img.shape
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img[h_space:h_space + patch_size, w_space:w_space + patch_size, :]
    elif isinstance(img, torch.Tensor):
        b, c, h, w = img.shape
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img[:, :, h_space:h_space + patch_size, w_space:w_space + patch_size]
    elif isinstance(img, Image.Image):
        w, h = img.size
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img.crop((w_space, h_space, w_space + patch_size, h_space + patch_size))
    else:
        raise TypeError("Unsupported image format. Supported formats: PIL, NumPy, Tensor.")


def padding(img_lq, gt_size):
    h, w, _ = img_lq.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)

    if h_pad == 0 and w_pad == 0:
        return img_lq

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    return img_lq


def resize_and_crop(image_np, crop_size, crop=False):
    """
    输入：
        image_np: 输入图像，numpy数组，HWC，范围[0, 1]
        crop_size: tuple (crop_h, crop_w)，裁剪尺寸
    输出：
        裁剪后的图像，numpy格式，HWC，范围[0, 1]
    """
    assert isinstance(image_np, np.ndarray), "输入必须是numpy数组"
    assert image_np.ndim == 3, "图像必须是HWC格式"
    assert image_np.dtype in [np.float32, np.float64], "图像值应为0-1范围"

    crop_h, crop_w = crop_size
    orig_h, orig_w = image_np.shape[:2]

    # 若任意一边小于裁剪尺寸，则等比放大
    scale_h = crop_h / orig_h if crop_h > orig_h else 1.0
    scale_w = crop_w / orig_w if crop_w > orig_w else 1.0
    scale = max(scale_h, scale_w)

    if scale > 1.0:
        # 将 numpy 图像转为 PIL
        image_pil = Image.fromarray((image_np * 255).astype(np.uint8))

        new_h = int(round(orig_h * scale))
        new_w = int(round(orig_w * scale))
        resize = T.Resize((new_h, new_w), interpolation=T.InterpolationMode.BICUBIC)
        image_pil = resize(image_pil)

        # 居中裁剪
        if crop:
            center_crop = T.CenterCrop((crop_h, crop_w))
            image_pil = center_crop(image_pil)

        # 转回 numpy 格式，0-1范围
        image_np_out = np.array(image_pil).astype(np.float32) / 255.0

        return image_np_out
    else:
        return image_np



if __name__ == '__main__':
    from torchvision.utils import make_grid, save_image
    # x = ['1', '2', '3'] * 3
    # print(x)
    # random.shuffle(x)
    # print(x)

    # lq_path = r'E:\Dataset\Restoration\Dehaze\RESIDE\SOTS\outdoor\input_all\0001_0.8_0.2.jpg'
    # dir, (base, ext) = osp.dirname(lq_path), osp.splitext(osp.basename(lq_path))
    # gt_path = os.path.join(dir.replace('input_all', 'target'), f'{base.split("_")[0]}{ext}')
    # print(gt_path)

    # root = r'G:\PhD\Paper\Ideal\SCDeRain\DeRain'
    # opt = {
    #     'dataroot_gt': os.path.join(root, r'datasets/SR/DF2K/DF2K_HR_train_p512'),
    #     'dataroot_lq': os.path.join(root, r'datasets/Derain/Rain100/Rain100L/train/input'),
    #     'lq_enlarge_ratio': 10,
    #     'io_backend': {'type': 'scandir'},
    #     'gt_size': 256,
    #     'use_hflip': True,
    #     'use_rot': False,
    #     'phase': 'train'
    # }

    root = r'G:\PhD\Paper\Ideal\Dehaze\Dehazing'
    opt = {
        'dataroot_gt1': os.path.join(root, r'datasets/Dehaze/RESIDE/ITS/train/target'),
        'dataroot_lq1': os.path.join(root, r'datasets/Dehaze/RESIDE/ITS/train/haze'),
        'dataroot_gt2': '',
        'dataroot_lq2': '',
        'lq_enlarge_ratio': 1,
        'io_backend': {'type': 'disk'},
        'gt_size': 256,
        'use_hflip': True,
        'use_rot': False,
        'phase': 'train',
        'return_lqs_len': False
    }

    dataset = UnpairedImageDataset(opt)
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    print(dataset.len_gts, dataset.len_lqs)
    print(len(dataset))

    # for i, data in enumerate(dataloader):
    #     print(f'{i}', end='\t')
    #     # if i < 10:
    #     #     save_image(
    #     #         data['gt'], rf'G:\PhD\Paper\Ideal\Dehaze\Dehazing\basicsr\data\temp\{i}.png'
    #     #     )
    #     if i < 10:
    #         print(data['lq'].shape, data['gt'].shape)
    #         print(data['lq_path'], data['gt_path'])
    #     else:
    #         break
