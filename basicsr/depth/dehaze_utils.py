import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import reduce
import os

"""
https://github.com/zychen-ustc/PSD-Principled-Synthetic-to-Real-Dehazing-Guided-by-Physical-Priors/blob/main/PSD/losses/loss_functions.py
https://github.com/YaN9-Y/D4/blob/main/src/networks.py
https://github.com/DongLiangSXU/SLAdehazing/blob/main/train_test/train_stage1.py
"""


def get_dark_channel(I, w):
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    dc = maxpool(0 - I[:, :, :, :])
    return -dc


def get_bright_channel(I, w):
    _, _, H, W = I.shape
    maxpool = nn.MaxPool3d((3, w, w), stride=1, padding=(0, w // 2, w // 2))
    bc = maxpool(I[:, :, :, :])
    return bc


def get_atmosphere(I, dark_ch, p):
    B, _, H, W = dark_ch.shape
    num_pixel = int(p * H * W)
    flat_dc = dark_ch.resize(B, H * W)
    flat_I = I.resize(B, 3, H * W)
    index = torch.argsort(flat_dc, descending=True)[:, :num_pixel]
    A = torch.zeros((B, 3)).to('cuda')

    for i in range(B):
        A[i] = flat_I[i, :, index].mean((1, 2))

    return A


def get_atmosphere2(I, bright_ch, p):
    B, _, H, W = bright_ch.shape
    num_pixel = int(p * H * W)
    flat_bc = bright_ch.resize(B, H * W)
    flat_I = I.resize(B, 3, H * W)
    index = torch.argsort(flat_bc, descending=False)[:, :num_pixel]
    A = torch.zeros((B, 3)).to('cuda')

    for i in range(B):
        A[i] = flat_I[i, :, index].mean((1, 2))

    return A


def get_SV_from_HSV(img):
    if (img.max(1)[0] == 0).all():
        return 0, 0
    else:
        S = (img.max(1)[0] - img.min(1)[0]) / img.max(1)[0]
        V = img.max(1)[0]
        S[img.max(1)[0] == 0] = 0
        return S, V


def get_HSV(img):
    hue = torch.zeros((img.shape[0], img.shape[2], img.shape[3])).to(img.device)

    hue[img[:, 2] == img.max(1)[0]] = 4.0 + ((img[:, 0] - img[:, 1]) / (img.max(1)[0] - img.min(1)[0]))[
        img[:, 2] == img.max(1)[0]]
    hue[img[:, 1] == img.max(1)[0]] = 2.0 + ((img[:, 2] - img[:, 0]) / (img.max(1)[0] - img.min(1)[0]))[
        img[:, 1] == img.max(1)[0]]
    hue[img[:, 0] == img.max(1)[0]] = (0.0 + ((img[:, 1] - img[:, 2]) / (img.max(1)[0] - img.min(1)[0]))[
        img[:, 0] == img.max(1)[0]]) % 6

    hue[img.min(1)[0] == img.max(1)[0]] = 0.0
    hue[hue < 0] += 6

    hue = hue / 6
    '''    
    if (img.max(1)[0] == 0).all():
        return 0, 0
    else:
        S = (img.max(1)[0] - img.min(1)[0]) / img.max(1)[0]
        V = img.max(1)[0]

        S[img.max(1)[0] == 0] = 0
    '''
    return hue


class TransmissionEstimator(nn.Module):
    def __init__(self, width=15, ):
        super(TransmissionEstimator, self).__init__()
        self.width = width
        self.t_min = 0.2
        self.alpha = 2.5
        self.A_max = 220.0 / 255
        self.omega = 0.95
        self.p = 0.001
        self.max_pool = nn.MaxPool2d(kernel_size=width, stride=1)
        self.max_pool_with_index = nn.MaxPool2d(kernel_size=width, return_indices=True)
        self.guided_filter = GuidedFilter(r=40, eps=1e-3)

    def get_dark_channel(self, x):
        x = torch.min(x, dim=1, keepdim=True)[0]
        x = F.pad(x, (self.width // 2, self.width // 2, self.width // 2, self.width // 2), mode='constant', value=1)
        x = -(self.max_pool(-x))
        return x

    def get_atmosphere_light(self, I, dc):
        n, c, h, w = I.shape
        flat_I = I.view(n, c, -1)
        flat_dc = dc.view(n, 1, -1)
        searchidx = torch.argsort(flat_dc, dim=2, descending=True)[:, :, :int(h * w * self.p)]
        searchidx = searchidx.expand(-1, 3, -1)
        searched = torch.gather(flat_I, dim=2, index=searchidx)
        return torch.max(searched, dim=2, keepdim=True)[0].unsqueeze(3)

    def get_transmission(self, I, A):
        return 1 - self.omega * self.get_dark_channel(I / A)

    def get_refined_transmission(self, I, rawt):
        I_max = torch.max(I.contiguous().view(I.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        I_min = torch.min(I.contiguous().view(I.shape[0], -1), dim=1, keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        normI = (I - I_min) / (I_max - I_min)
        refinedT = self.guided_filter(normI, rawt)

        return refinedT

    def get_radiance(self, I, A, t):
        return (I - A) / t + A

    def get_radiance_all(self, I):
        A = self.get_atmosphere_light_new(I)
        t = self.get_transmission(I, A)
        t = self.get_refined_transmission(I, t)
        return (I - A) / t + A

    def get_depth(self, I):
        I_dark = self.get_dark_channel(I)

        A = self.get_atmosphere_light(I, I_dark)
        A[A > self.A_max] = self.A_max
        rawT = self.get_transmission(I, A)

        refinedT = self.get_refined_transmission(I, rawT)
        return refinedT

    def get_atmosphere_light_new(self, I):
        I_dark = self.get_dark_channel(I)
        A = self.get_atmosphere_light(I, I_dark)
        A[A > self.A_max] = self.A_max
        return A


class GuidedFilter(nn.Module):
    def __init__(self, r, eps=1e-8):
        super(GuidedFilter, self).__init__()

        self.r = r
        self.eps = eps
        self.boxfilter = BoxFilter(r)

    def forward(self, x, y):
        n_x, c_x, h_x, w_x = x.size()
        n_y, c_y, h_y, w_y = y.size()

        assert n_x == n_y
        # assert c_x == 1 or c_x == c_y
        assert h_x == h_y and w_x == w_y
        assert h_x > 2 * self.r + 1 and w_x > 2 * self.r + 1

        # N
        N = self.boxfilter(x.data.new().resize_((1, 1, h_x, w_x)).fill_(1.0))

        # mean_x
        mean_x = self.boxfilter(x) / N
        # mean_y
        mean_y = self.boxfilter(y) / N
        # cov_xy
        cov_xy = self.boxfilter(x * y) / N - mean_x * mean_y
        # var_x
        var_x = self.boxfilter(x * x) / N - mean_x * mean_x

        # A
        A = cov_xy / (var_x + self.eps)
        # b
        b = mean_y - A * mean_x

        # mean_A; mean_b
        mean_A = self.boxfilter(A) / N
        mean_b = self.boxfilter(b) / N
        return torch.mean(mean_A * x + mean_b, dim=1, keepdim=True)


def diff_x(input, r):
    assert input.dim() == 4

    left = input[:, :, r:2 * r + 1]
    middle = input[:, :, 2 * r + 1:] - input[:, :, :-2 * r - 1]
    right = input[:, :, -1:] - input[:, :, -2 * r - 1:    -r - 1]
    output = torch.cat([left, middle, right], dim=2)
    return output


def diff_y(input, r):
    assert input.dim() == 4

    left = input[:, :, :, r:2 * r + 1]
    middle = input[:, :, :, 2 * r + 1:] - input[:, :, :, :-2 * r - 1]
    right = input[:, :, :, -1:] - input[:, :, :, -2 * r - 1:    -r - 1]
    output = torch.cat([left, middle, right], dim=3)
    return output


class BoxFilter(nn.Module):
    def __init__(self, r):
        super(BoxFilter, self).__init__()
        self.r = r

    def forward(self, x):
        assert x.dim() == 4
        return diff_y(diff_x(x.cumsum(dim=2), self.r).cumsum(dim=3), self.r)


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma=3):
        super(GaussianFilter, self).__init__()

        self.gaussian_kernel = self.cal_kernel(kernel_size=kernel_size, sigma=sigma).expand(1, 1, -1, -1).cuda()

    def apply_gaussian_filter(self, x):
        # cal gaussian filter of N C H W in cuda
        n, c, h, w = x.shape
        gaussian = torch.nn.functional.conv2d(x, self.gaussian_kernel.expand(c, 1, -1, -1),
                                              padding=self.gaussian_kernel.shape[2] // 2, groups=c)

        return gaussian

    def cal_gaussian_kernel_at_ij(self, i, j, sigma):
        return (1. / (2 * math.pi * pow(sigma, 2))) * math.exp(-(pow(i, 2) + pow(j, 2)) / (2 * pow(sigma, 2)))

    def cal_kernel(self, kernel_size=3, sigma=1.):
        kernel = torch.ones((kernel_size, kernel_size)).float()
        for i in range(kernel_size):
            for j in range(kernel_size):
                kernel[i, j] = self.cal_gaussian_kernel_at_ij(-(kernel_size // 2) + j, (kernel_size // 2) - i,
                                                              sigma=sigma)

        kernel = kernel / torch.sum(kernel)
        return kernel
