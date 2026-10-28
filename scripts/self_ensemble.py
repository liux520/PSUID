import cv2
import numpy as np
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def uint2tensor(img):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
    return img


def test_selfensemble(lq, model=None):
    # TODO: to be tested
    # 8 augmentations
    # modified from https://github.com/thstkdgus35/EDSR-PyTorch

    def _transform(v, op):
        # if self.precision != 'single': v = v.float()
        v2np = v.data.cpu().numpy()
        if op == 'v':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'h':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 't':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).to(device)
        # if self.precision == 'half': ret = ret.half()

        return ret

    # prepare augmented data
    lq_list = [lq]
    for tf in 'v', 'h', 't':
        lq_list.extend([_transform(t, tf) for t in lq_list])

    # inference
    if model is None:
        out_list = [aug for aug in lq_list]
    else:
        out_list = [model(aug) for aug in lq_list]

    # merge results
    for i in range(len(out_list)):
        if i > 3:
            out_list[i] = _transform(out_list[i], 't')
        if i % 4 > 1:
            out_list[i] = _transform(out_list[i], 'h')
        if (i % 4) % 2 == 1:
            out_list[i] = _transform(out_list[i], 'v')
    output = torch.cat(out_list, dim=0)
    output = output.mean(dim=0, keepdim=True)

    return output


if __name__ == '__main__':
    path = r'E:\Dataset\Restoration\Denoise\CBSD68\25\0000.png'
    image = cv2.imread(path)
    image = uint2tensor(image).to(device)

    output = test_selfensemble(image)

    output = tensor2uint8(output)
    cv2.imwrite('out.png', output)
