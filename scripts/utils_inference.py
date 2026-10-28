import torch
import os
import cv2
from tqdm import tqdm
import numpy as np
from natsort import os_sorted
import torch.nn.functional as F
import time
import random


def timeit_decorator(func):
    timeit = AverageMeter()

    def wrapper(*args, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        begin = time.time()
        result = func(*args, **kwargs)
        over = time.time()
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        timeit.update((over - begin))
        print(f"Function {func.__name__} took {(timeit.avg):.6f} seconds")
        return result, timeit.avg

    return wrapper


class AverageMeter():
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _get_paths_from_images(path, suffix=''):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname) and suffix in fname:
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return os_sorted(images)


IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def uint2tensor(img):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
    return img


def single2uint(image):
    '''Numpy:(0,1) (h,w,c) float32 -> Numpy:(0,255) (h,w,c) uint8'''
    return np.uint8((image.clip(0., 1.) * 255.).round())


def uint2single(image):
    '''Numpy:(0,255) (h,w,c) uint8 -> Numpy:(0,1) (h,w,c) float32'''
    return np.float32(image / 255.)


def load(path, model, key='state_dict', delete_module=False, delete_str='module.'):
    from collections import OrderedDict

    def delete_state_module(weights, delete_str='module.'):
        weights_dict = {}
        for k, v in weights.items():
            # new_k = k.replace('module.', '') if 'module' in k else k
            new_k = k.replace(delete_str, '') if delete_str in k else k
            weights_dict[new_k] = v
        return weights_dict

    ckpt = torch.load(path, map_location=lambda storage, loc: storage)  # ['state_dict']
    ckpt = ckpt.get(key, ckpt)

    if delete_module:
        ckpt = delete_state_module(ckpt, delete_str)

    overlap = OrderedDict()
    for key, value in ckpt.items():
        if key in model.state_dict().keys() and ckpt[key].shape == model.state_dict()[key].shape:
            overlap[key] = value
        else:
            print(f'Failed load: ckpt: {key}-{ckpt[key].shape} | model: {key}-{model.state_dict()[key].shape}')

    print(f'{(len(overlap) * 1.0 / len(ckpt) * 100):.4f}% weights is loaded!', end='\t')
    print(f'{(len(overlap) * 1.0 / len(model.state_dict()) * 100):.4f}% params is inited!')

    try:
        model.load_state_dict(overlap, strict=False)
        print(f'Loading weights from {path}.')
    except RuntimeError as e:
        print(f'RuntimeError: {e}')


def load_(path, model, key='state_dict', delete_module=False, delete_str='module.'):
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint = checkpoint if len(checkpoint) > 10 else checkpoint[key]

    model_dict = model.state_dict()
    if delete_module:
        checkpoint = delete_state_module(checkpoint, delete_str)
    overlap = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(overlap)
    print(f'{(len(overlap) * 1.0 / len(checkpoint) * 100):.4f}% weights is loaded!', end='\t')
    print(f'{(len(overlap) * 1.0 / len(model_dict) * 100):.4f}% params is init!')
    print(f'Drop Keys: {[k for k, v in checkpoint.items() if k not in model_dict]}')
    model.load_state_dict(model_dict)
    return model


def delete_state_module(weights, delete_str='module.'):
    weights_dict = {}
    for k, v in weights.items():
        # new_k = k.replace('module.', '') if 'module' in k else k
        new_k = k.replace(delete_str, '') if delete_str in k else k
        weights_dict[new_k] = v
    return weights_dict


def show_kv(path):
    ckpt = torch.load(path, map_location='cpu')['params_ema']
    for k, v in ckpt.items():
        print(f'{k}')


def patch_test(model, lr, scale=4, tile=320, tile_overlap=24):
    b, c, h, w = lr.size()
    tile = min(tile, h, w)

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
    w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
    E = torch.zeros(b, c, h * scale, w * scale).type_as(lr)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = lr[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
            out_patch = model(in_patch)
            if isinstance(out_patch, list):
                out_patch = out_patch[-1]
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx * scale:(h_idx + tile) * scale, w_idx * scale:(w_idx + tile) * scale].add_(
                out_patch)
            W[..., h_idx * scale:(h_idx + tile) * scale, w_idx * scale:(w_idx + tile) * scale].add_(
                out_patch_mask)
    res = E.div_(W)
    output = res  # [:, :, :h * 4, :w * 4]
    return output


def pad_input(img_lq, window_size):
    _, _, h_old, w_old = img_lq.size()
    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
    img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
    # output = test(img_lq, model, args, window_size)
    # output = output[..., :h_old * args.scale, :w_old * args.scale]
    return img_lq


def check_image_size(x, window_size):
    if len(x.shape) == 4:
        _, _, h, w = x.size()
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    else:
        _, h, w = x.size()
        mod_pad_h = (window_size - h % window_size) % window_size
        mod_pad_w = (window_size - w % window_size) % window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x


def _random_crop_patch(img, patch_size):
    if type(img) == np.ndarray:
        h, w, c = img.shape
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img[h_space:h_space + patch_size, w_space:w_space + patch_size, :]
    elif type(img) == torch.Tensor:
        b, c, h, w = img.shape
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img[:, :, h_space:h_space + patch_size, w_space:w_space + patch_size]
    else:
        w, h = img.size
        h_space = random.randint(0, h - patch_size)
        w_space = random.randint(0, w - patch_size)
        return img.crop((w_space, h_space, w_space + patch_size, h_space + patch_size))


if __name__ == '__main__':
    x = torch.randn(3, 31, 31)
    y = check_image_size(x, 4)
    print(y.size())
