import os, sys
sys_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
sys.path.insert(0, sys_path)
# sys.path.append(sys_path)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch, time
import numpy as np
import csv
from collections import defaultdict
from PIL import Image


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _read_existing_models(csv_file: str):
    """Return a set of model names already in csv (excluding header)."""
    if not os.path.exists(csv_file):
        return set()
    try:
        with open(csv_file, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            rows = list(r)
        if len(rows) <= 1:
            return set()
        return {row[0] for row in rows[1:] if len(row) > 0 and row[0] != "model"}
    except Exception:
        return set()


def _append_row_csv(path: str, row):
    """Append one row to existing csv atomically (write to tmp then replace)."""
    # 读取已有全部行 -> 追加 -> 原子替换
    tmp = path + ".tmp"
    rows = []
    if os.path.exists(path):
        with open(path, "r", newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    rows.append(row)
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(rows)
    os.replace(tmp, path)


def uint2tensor(img):
    img = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1) / 255.).float().unsqueeze(0)
    return img


def tensor2uint8(img):
    img = img.detach().cpu().numpy().astype(np.float32).squeeze(0).transpose(1, 2, 0)
    img = np.uint8((img.clip(0., 1.) * 255.).round())
    return img


def _safe_float(x):
    """Convert metric output to python float if possible."""
    try:
        # torch scalar / numpy scalar / python number
        if hasattr(x, "item"):
            return float(x.item())
        return float(x)
    except Exception:
        return np.nan


def _atomic_write_csv(path: str, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in rows:
            w.writerow(r)
    os.replace(tmp, path)


def _init_dataset_csv_if_needed(csv_path: str, ds: str, model_name: list, metric_list: list):
    """
    不存在就创建（表头 + 占位行）。
    """
    os.makedirs(csv_path, exist_ok=True)
    out_path = os.path.join(csv_path, f"./{ds}.csv")
    if not os.path.exists(out_path):
        rows = [["model"] + metric_list]
        for m in model_name:
            rows.append([m] + [""] * len(metric_list))  # 空值占位
        _atomic_write_csv(out_path, rows)
    return out_path


def _append_dataset_csv(csv_path: str, ds: str, model_name: list, metric_list: list, sums, counts, current_model: str):
    """
    新逻辑：
    - ds.csv 不存在：创建（用你原来的 init）
    - ds.csv 已存在：在文件末尾追加一行本次 current_model 的结果，不覆盖旧内容
    """
    out_path = _init_dataset_csv_if_needed(csv_path, ds, model_name, metric_list)

    # 计算 current_model 在该 dataset 上的均值
    row = [current_model]
    for met in metric_list:
        c = counts.get((ds, current_model, met), 0)
        s = sums.get((ds, current_model, met), 0.0)
        row.append("" if c == 0 else (s / c))

    # 避免同名模型行重复：自动加后缀 _runK
    existing = _read_existing_models(out_path)
    if row[0] in existing:
        base = row[0]
        k = 2
        new_name = f"{base}_run{k}"
        while new_name in existing:
            k += 1
            new_name = f"{base}_run{k}"
        row[0] = new_name

    # 追加一行（原子写）
    _append_row_csv(out_path, row)
    return out_path


def _write_dataset_csv(csv_path: str, ds: str, model_name: list, metric_list: list, sums, counts):
    out_path = os.path.join(csv_path, f"{ds}.csv")
    rows = [["model"] + metric_list]
    for m in model_name:
        row = [m]
        for met in metric_list:
            c = counts.get((ds, m, met), 0)
            s = sums.get((ds, m, met), 0.0)
            row.append("" if c == 0 else (s / c))
        rows.append(row)
    _atomic_write_csv(out_path, rows)


def _init_nr_metric():
    from pyiqa.api_helpers import create_metric

    # root = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir))
    root = r'/data1/sr617/codes/liux/Foundation/IQA'  # r'G:\PhD\Codes\Foundation\IQA'
    niqe = create_metric(metric_name='niqe', device=device,
                         pretrained_model_path=os.path.join(root, r'niqe_modelparameters.mat'))
    brisque = create_metric(metric_name='brisque', device=device,
                            pretrained_model_path=os.path.join(root, r'brisque_svm_weights.pth'))
    nrqm = create_metric(metric_name='nrqm', device=device,
                         pretrained_model_path=os.path.join(root, r'NRQM_model.mat'))
    paq2piq = create_metric(metric_name='paq2piq', device=device,
                            pretrained_model_path=os.path.join(root, r'P2P_RoIPoolModel-fit.10.bs.120-ca69882e.pth'))
    nima = create_metric(metric_name='nima', device=device,
                         pretrained_model_path=os.path.join(root, r'NIMA_InceptionV2_ava-b0c77c00.pth'))
    piqe = create_metric(metric_name='piqe', device=device)
    entropy = create_metric(metric_name='entropy', device=device)

    musiq = create_metric(metric_name='musiq', device=device,
                          pretrained_model_path=os.path.join(root, r'musiq_koniq_ckpt-e95806b9.pth'))
    maniqa = create_metric(metric_name='maniqa', device=device, weight_keys=None,
                           pretrained_model_path=os.path.join(root, r'ckpt_koniq10k.pt'))
    clipiqa = create_metric(metric_name='clipiqa+_vitL14_512', device=device,
                            pretrained_model_path=os.path.join(root, r'CLIPIQA+_ViTL14_512-e66488f2.pth'))
    # qalign = create_metric(metric_name='qalign', device=device,
    #                        pretrained_model_path=os.path.join(root, r'qalign_svm_weights.pth'))
    topiq = create_metric(metric_name='topiq_nr', device=device,
                          pretrained_model_path=os.path.join(root, r'cfanet_nr_koniq_res50-9a73138b.pth'))

    return {
        'niqe': niqe,
        'brisque': brisque,
        'nima': nima,
        'nrqm': nrqm,
        'paq2piq': paq2piq,
        'piqe': piqe,
        'musiq': musiq,
        'maniqa': maniqa,
        'clipiqa': clipiqa,
        'topiq': topiq,
        'entropy': entropy,
    }


if __name__ == '__main__':
    root = r'/data1/sr617/codes/liux/Restoration/Dehazing/Results'  # r'G:\PhD\Paper\Ideal\Dehaze\Results'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metrics = _init_nr_metric()
    metric_list = list(metrics.keys())

    model_name = [
        # 'hazy',
        'Ours'
    ]
    dataset_name = [
        'Fattal',
        'HSTS_Real',
        'RTTS'
    ]
    csv_path = r'/data1/sr617/codes/liux/Restoration/Dehazing'

    # scores[(dataset, model, metric)] = [v1, v2, ...]
    # scores = defaultdict(list)
    # 全局累计：保证同一个 dataset.csv 能逐步填充多个模型的结果
    sums = defaultdict(float)
    counts = defaultdict(int)

    # 确保每个 dataset 的 csv 文件先创建好（即使中途崩，也能看到已完成的）
    for ds in dataset_name:
        _init_dataset_csv_if_needed(csv_path, ds, model_name, metric_list)

    for models in os.listdir(root):
        if models in model_name:
            current_processed_model = os.path.join(root, models)
            print(f'{get_time()} Current Processed Model: {current_processed_model}')

            for datasets in os.listdir(current_processed_model):
                if datasets in dataset_name:
                    current_processed_model_dataset = os.path.join(current_processed_model, datasets)
                    print(f'\t{get_time()} Current Processed Model Dataset: {current_processed_model_dataset}')

                    dataset_images = os.listdir(current_processed_model_dataset)
                    dataset_images_len = len(dataset_images)
                    for i, imgs in enumerate(dataset_images):
                        # print(f'\t\t{get_time()} [{i}/{dataset_images_len}] Current Processed Model Dataset Image: {imgs}')
                        absolute_path = os.path.join(current_processed_model_dataset, imgs)

                        try:
                            img_np = np.array(Image.open(absolute_path))
                            img_tensor = uint2tensor(img_np)

                            for metric_name, metric_fn in metrics.items():
                                with torch.inference_mode():
                                    per_score = metric_fn(img_tensor)

                                # scores[(datasets, models, metric_name)].append(_safe_float(per_score))

                                # with open('middle_record.txt', 'a+') as f:
                                #     f.write(f'{datasets}, {models}, {imgs}, {metric_name}: {_safe_float(per_score)}\n')

                                v = _safe_float(per_score)
                                if np.isfinite(v):
                                    key = (datasets, models, metric_name)
                                    sums[key] += v
                                    counts[key] += 1

                        except Exception as e:
                            print(f'{get_time()} [WARN] Skip {absolute_path} | {type(e).__name__}: {e}')

                    # # 这个 (model, dataset) 全部图片算完后，立刻把 dataset.csv 覆盖写一次
                    # _write_dataset_csv(csv_path, datasets, model_name, metric_list, sums, counts)
                    # print(f'\t{get_time()} Saved CSV for dataset={datasets} (updated with model={models})')

                    # 这个 (model, dataset) 全部图片算完后：追加一行到 dataset.csv（不覆盖旧内容）
                    out_file = _append_dataset_csv(csv_path, datasets, model_name, metric_list, sums, counts, current_model=models)
                    print(f'\t{get_time()} Appended CSV row -> {out_file} | dataset={datasets}, model={models}')

    # 可选：全部结束后再写一遍保险（不写也行）
    # for ds in dataset_name:
    #     _write_dataset_csv(csv_path, ds, model_name, metric_list, sums, counts)


    # # Write one CSV per dataset
    # for ds in dataset_name:
    #     out_path = os.path.join(csv_path, f"{ds}.csv")
    #     with open(out_path, "w", newline="", encoding="utf-8") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["model"] + metric_list)
    #
    #         for model in model_name:
    #             row = [model]
    #             for metric_name in metric_list:
    #                 v = float(np.mean(scores.get((ds, model, metric_name), [])))
    #                 row.append(v)
    #             writer.writerow(row)
