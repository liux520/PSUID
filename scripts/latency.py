import torch
import time
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.models as models
from ptflops import get_model_complexity_info
import torch
from torchvision.models import resnet50
from thop import clever_format, profile


@torch.inference_mode()
def params_flops(model, input_tensor, use_cuda=True, model_name=''):
    model.eval()
    if use_cuda:
        model.cuda()
        dummy_input = input_tensor.cuda()
    macs, params = profile(model, inputs=(dummy_input,))

    # Format the numbers into a readable format (e.g., 4.14 GMac, 25.56 MParams)
    macs_readable, params_readable = clever_format([macs, params], "%.3f")

    print(f"{model_name}: Formatted MACs: {macs_readable}, Formatted Parameters: {params_readable}")
    # Expected output: Formatted MACs: 4.140G, Formatted Parameters: 25.557M


    # with torch.cuda.device(0):
    # macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True, backend='pytorch',
    #                                        print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    #     macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, backend='aten',
    #                                            print_per_layer_stat=True, verbose=True)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))


@torch.inference_mode()
def benchmark_latency(model, input_tensor, runs=100, use_cuda=True, model_name=''):
    model.eval()
    if use_cuda:
        model = model.cuda()
        input_tensor = input_tensor.cuda()

    # Warm up
    with torch.no_grad():
        for _ in range(20):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)

    torch.cuda.synchronize()
    total = time.time() - start

    print(f"{model_name} Average latency: {total / runs * 1000:.3f} ms")


if __name__ == '__main__':
    input_tensor = torch.randn(1, 3, 512, 512)

    # NAFNet
    from basicsr.archs.NAFNet_arch import NAFNet
    model = NAFNet(
        img_channel=3, width=32, middle_blk_num=12,
        enc_blk_nums=[ 2, 2, 4, 8 ], dec_blk_nums=[ 2, 2, 2, 2 ]
    )
    params_flops(model, input_tensor, model_name='NAFNet')
    benchmark_latency(model, input_tensor, model_name='NAFNet')
