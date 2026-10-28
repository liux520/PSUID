import os, time, matplotlib
from natsort import os_sorted
import cv2, random, torch
import numpy as np
# import albumentations
from basicsr.depth.depth_interface import _depth_v2_interface


def _scan_(path, suffix=''):
    if os.path.isdir(path):
        imgs = _get_paths_from_images(path, suffix)
    else:
        imgs = path if isinstance(path, (tuple, list)) else [path]
    return imgs


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


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


def add_haze(img, depth_map=None, beta_mul_range=(1.0, 2.0), beta_range=(1.0, 2.0), given_A=None, A_range=(0.6, 0.9), jitter=(-0.025, 0.025)):
    """
    雾气模拟：依赖深度图计算透射率，加入色温偏移模拟不同类型雾: I(x) = J(x)t(x) + A(1-t(x)), t(x) = e^(-beta * d(x))
    """

    if depth_map is None:
        depth = depthv2.post_process(
            x=img,
            grayscale=True,
            uint8=False
        )
    else:
        depth = cv2.imread(depth_map, flags=cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

    # 1. t(x) = e^(-beta * d(x))
    beta = np.random.rand(1) * (beta_range[1] - beta_range[0]) + beta_range[0]
    # depth_blur_kernel = random.choice([2 * v + 1 for v in range(3, 10)])  # 7 ~ 21
    # t = np.exp(
    #     -np.minimum(1 - cv2.blur(depth, (depth_blur_kernel, depth_blur_kernel)), 0.3) * beta
    # )
    beta_mul = np.random.rand(1) * (beta_mul_range[1] - beta_mul_range[0]) + beta_mul_range[0]
    t = np.exp(-beta * beta_mul * (1 - depth))
    t = np.repeat(t[..., np.newaxis], 3, axis=2)

    # 2. global atmosphere light A (* and +)
    if given_A is None:
        A_color = np.random.uniform(*A_range, size=(1, 1, 3))
        if np.random.rand() < 0.5:
            A_random = np.random.uniform(*jitter, size=(1, 1, 3))
            A_color = A_color + A_random
    else:
        A_color = given_A
        if np.random.rand() < 0.5:
            A_random = np.random.uniform(*jitter, size=(1, 1, 3))
            A_color = A_color + A_random

    print(f'beta: {beta} | beta_mul: {beta_mul} | A_color: {A_color}')

    # 3. I(x) = J(x)t(x) + A(1-t(x))
    hazy_image = np.clip(img * t + A_color * (1 - t), 0, 1)
    hazy_image = np.uint8(hazy_image * 255)
    return hazy_image, depth


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    depthv2 = _depth_v2_interface(
        root = r'G:\PhD\Codes\Foundation\DepthAnything',
        device = device,
        encoder='vitl',  # s/b/l
    )

    image_path = r'../pics/gts/0151.jpg'
    depth_path = r''
    save_path = r'../pics'
    save_depth = '../pics'
    os.makedirs(save_path, exist_ok=True)

    image_path_list = _scan_(image_path)
    depth_path_list = _scan_(depth_path) if depth_path != '' else [None] * len(image_path_list)

    random_key = random.choice(list(haze_color_map.keys()))
    random_tuple = haze_color_map[random_key]
    random_tuple = np.array(random_tuple).reshape(1, 1, 3)
    print(f"Select: {random_key}-{random_tuple}")

    for i, (image_p, depth_path) in enumerate(zip(image_path_list, depth_path_list)):
        base, ext = os.path.splitext(os.path.basename(image_p))
        _path_dict_ = {'image': image_p, 'depth': depth_path}
        image = cv2.imread(image_p)[:, :, ::-1].astype(np.float32) / 255.

        image, depth = add_haze(
            img=image,
            depth_map=depth_path,
            beta_range=(1.0, 2.0),
            A_range=(0.7, 1.0),
            given_A=random_tuple,
            jitter=(-0.025, 0.025),
            beta_mul_range=(1.0, 2.0),
        )
        print(image.shape, depth.shape)
        cv2.imwrite(os.path.join(save_path, f"{base}_hazy{ext}"), image[:, :, ::-1])

        if save_depth != '':
            depth_gray = (depth * 255.).astype(np.uint8)
            depth_gray = np.repeat(depth_gray[..., np.newaxis], 3, axis=-1)
            cv2.imwrite(os.path.join(save_depth, f"{base}_depth{ext}"), depth_gray)

            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth_color = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
            # depth_color = cv2.applyColorMap((depth * 255.).astype(np.uint8), cv2.COLORMAP_INFERNO)
            cv2.imwrite(os.path.join(save_depth, f"{base}_depth_color{ext}"), depth_color[:, :, ::-1])

            print(depth_gray.shape, depth_color.shape)
