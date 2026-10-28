import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table

def Analysis(x, model_name, model):
    model.eval()

    flops = FlopCountAnalysis(model, x)
    total = sum([param.nelement() for param in model.parameters()])
    print(f'{model_name}: Number of params: {total / 1e6:.2f}M, FLOPs: {flops.total()}')


if __name__ == '__main__':
    from basicsr.archs.swinir_arch import get_swinir
    from basicsr.archs.srformer_arch import get_srformer
    from basicsr.archs.HPINet_arch import get_hpinet
    from basicsr.archs.EIMN_arch import EIMN_L
    from basicsr.archs.dat_arch import get_dat
    from basicsr.archs.craft_arch import get_craft

    scale = 4

    model_dict = {
        'SwinIR': get_swinir(scale=scale).cuda(),
        'HPINet': get_hpinet(scale=scale).cuda(),
        'EIMN': EIMN_L(scale=scale).cuda(),
        'DAT': get_dat(scale=scale).cuda(),
        'CRAFT': get_craft(scale=scale).cuda(),
        'SRFormer': get_srformer(scale=scale).cuda(),
    }

    with torch.no_grad():
        h, w = 720 // scale, 1280 // scale  # 512//2,512//2
        x = torch.randn((1, 3, h, w)).cuda()

        for name, model in model_dict.items():
            Analysis(x, name, model)

"""
SwinIR:   Number of params: 0.93M, FLOPs: 63,593932800
HPINet:   Number of params: 0.90M, FLOPs: 136,862294144
EIMN:     Number of params: 1.00M, FLOPs: 54,728327168
DAT:      Number of params: 0.57M, FLOPs: 49,685227992
CRAFT:    Number of params: 0.75M, FLOPs: 52,400116928
SRFormer: Number of params: 0.87M, FLOPs: 62,819942400
"""