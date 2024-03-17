# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""AutoAnchor utils."""

import random

import numpy as np
import torch
import yaml
from tqdm import tqdm

from utils import TryExcept
from utils.general import LOGGER, TQDM_BAR_FORMAT, colorstr

PREFIX = colorstr("AutoAnchor: ")


def check_anchor_order(m):
    """Checks and corrects anchor order against stride in YOLOv5 Detect() module if necessary."""
    a = m.anchors.prod(-1).mean(-1).view(-1)  # mean anchor area per output layer
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da and (da.sign() != ds.sign()):  # same order
        LOGGER.info(f"{PREFIX}Reversing anchor order")
        m.anchors[:] = m.anchors.flip(0)


@TryExcept(f"{PREFIX}ERROR")
def check_anchors(dataset, model, thr=4.0, imgsz=640):
    """Evaluates anchor fit to dataset and adjusts if necessary, supporting customizable threshold and image size."""
    m = model.module.model[-1] if hasattr(model, "module") else model.model[-1]  # Detect()
    shapes = imgsz * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    scale = np.random.uniform(0.9, 1.1, size=(shapes.shape[0], 1))  # augment scale
    wh = torch.tensor(np.concatenate([l[:, 3:5] * s for s, l in zip(shapes * scale, dataset.labels)])).float()  # wh

    def metric(k):  # compute metric
        r = wh[:, None] / k[None]
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        best = x.max(1)[0]  # best_x
        aat = (x > 1 / thr).float().sum(1).mean()  # anchors above threshold
        bpr = (best > 1 / thr).float().mean()  # best possible recall
        return bpr, aat

    stride = m.stride.to(m.anchors.device).view(-1, 1, 1)  # model strides
    anchors = m.anchors.clone() * stride  # current anchors
    bpr, aat = metric(anchors.cpu().view(-1, 2))
    s = f"\n{PREFIX}{aat:.2f} anchors/target, {bpr:.3f} Best Possible Recall (BPR). "
    if bpr > 0.98:  # threshold to recompute
        LOGGER.info(f"{s}Current anchors are a good fit to dataset ✅")
    else:
        LOGGER.info(f"{s}Anchors are a poor fit to dataset ⚠️, attempting to improve...")
        na = m.anchors.numel() // 2  # number of anchors
        anchors = kmean_anchors(dataset, n=na, img_size=imgsz, thr=thr, gen=1000, verbose=False)
        new_bpr = metric(anchors)[0]
        if new_bpr > bpr:  # replace anchors
            anchors = torch.tensor(anchors, device=m.anchors.device).type_as(m.anchors)
            m.anchors[:] = anchors.clone().view_as(m.anchors)
            check_anchor_order(m)  # must be in pixel-space (not grid-space)
            m.anchors /= stride
            s = f"{PREFIX}Done ✅ (optional: update model *.yaml to use these anchors in the future)"
        else:
            s = f"{PREFIX}Done ⚠️ (original anchors better than new anchors, proceeding with original anchors)"
        LOGGER.info(s)


def kmean_anchors(dataset="./data/coco128.yaml", n=9, img_size=640, thr=4.0, gen=1000, verbose=True):
    """
    Creates kmeans-evolved anchors from training dataset.

    Arguments:
        dataset: path to data.yaml, or a loaded dataset
        n: number of anchors
        img_size: image size used for training
        thr: anchor-label wh ratio threshold hyperparameter hyp['anchor_t'] used for training, default=4.0
        gen: generations to evolve anchors using genetic algorithm
        verbose: print all results

    Return:
        k: kmeans evolved anchors

    Usage:
        from utils.autoanchor import *; _ = kmean_anchors()
    """
    from scipy.cluster.vq import kmeans

    npr = np.random
    thr = 1 / thr

    def metric(k, wh):  # compute metrics
        r = wh[:, None] / k[None]  # wh 扩展为列向量（[:, None]）和将 k 扩展为行向量（[None]），计算了每个标签框和预测框之间的比率
        x = torch.min(r, 1 / r).min(2)[0]  # ratio metric 沿着列向量取每行的最小值，其中这个最小值是有一个包含（最小值，对应索引张良）的数组，这些数组构成张量
        # x = wh_iou(wh, torch.tensor(k))  # iou metric
        return x, x.max(1)[0]  # x, best_x   x.max(1)是按照每列取最大值，x.max(1)[0]是每列最大值构成的张量

    def anchor_fitness(k):  # mutation fitness
        _, best = metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best > thr).float()).mean()  # fitness

    def print_results(k, verbose=True):  # 用于打印结果并返回排序后的k值
        k = k[np.argsort(k.prod(1))]  # sort small to large  排序方式是按照每行元素的乘积由小到大排列
        x, best = metric(k, wh0)  # 调用metric函数传入排序后的k值和wh_0参数，其中wh_0是包含标签宽度和高度的数组
        # 计算最佳召回率bpr和超过阈值的锚点数量aat，其中thr是阈值
        bpr, aat = (best > thr).float().mean(), (x > thr).float().mean() * n  # best possible recall, anch > thr
        s = (
            f"{PREFIX}thr={thr:.2f}: {bpr:.4f} best possible recall, {aat:.2f} anchors past thr\n"
            f"{PREFIX}n={n}, img_size={img_size}, metric_all={x.mean():.3f}/{best.mean():.3f}-mean/best, "
            f"past_thr={x[x > thr].mean():.3f}-mean: "
        )
        for x in k:
            s += "%i,%i, " % (round(x[0]), round(x[1]))
        if verbose:
            LOGGER.info(s[:-2])
        return k

    if isinstance(dataset, str):  # *.yaml file
        with open(dataset, errors="ignore") as f:
            data_dict = yaml.safe_load(f)  # model dict
        from utils.dataloaders import LoadImagesAndLabels

        dataset = LoadImagesAndLabels(data_dict["train"], augment=True, rect=True)

    # Get label wh
    # dataset.shapes每行代表一张图像的形状，img_size 乘以 dataset.shapes，这样做是为了将图像的实际形状转换为与 img_size 相同的尺寸空间
    # img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)：将每张图像的形状与其最大值进行归一化计算，得到相对于img_size 的比例
    shapes = img_size * dataset.shapes / dataset.shapes.max(1, keepdims=True)
    # 每张图像的形状和对应的标签一一配对，分别用s和l表示，然后将每个目标的宽度和高度
    # l[:, 3:5]：从标签中获取每个目标的第四列（通常是左上角坐标）到第五列（通常是右下角坐标），这样得到的是每个目标的宽度和高度。
    # 然后在*s，将标签也相应的缩放，得到一个含w和h的数组wh0
    wh0 = np.concatenate([l[:, 3:5] * s for s, l in zip(shapes, dataset.labels)])  # wh

    # Filter
    i = (wh0 < 3.0).any(1).sum()
    if i:
        LOGGER.info(f"{PREFIX}WARNING ⚠️ Extremely small objects found: {i} of {len(wh0)} labels are <3 pixels in size")
    wh = wh0[(wh0 >= 2.0).any(1)].astype(np.float32)  # filter > 2 pixels
    # wh = wh * (npr.rand(wh.shape[0], 1) * 0.9 + 0.1)  # multiply by random scale 0-1

    # Kmeans init
    try:
        LOGGER.info(f"{PREFIX}Running kmeans for {n} anchors on {len(wh)} points...")
        assert n <= len(wh)  # apply overdetermined constraint
        s = wh.std(0)  # sigmas for whitening
        k = kmeans(wh / s, n, iter=30)[0] * s  # points
        assert n == len(k)  # kmeans may return fewer points than requested if wh is insufficient or too similar
    except Exception:
        LOGGER.warning(f"{PREFIX}WARNING ⚠️ switching strategies from kmeans to random init")
        k = np.sort(npr.rand(n * 2)).reshape(n, 2) * img_size  # random init
    wh, wh0 = (torch.tensor(x, dtype=torch.float32) for x in (wh, wh0))
    k = print_results(k, verbose=False)

    # Plot
    # k, d = [None] * 20, [None] * 20
    # for i in tqdm(range(1, 21)):
    #     k[i-1], d[i-1] = kmeans(wh / s, i)  # points, mean distance
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7), tight_layout=True)
    # ax = ax.ravel()
    # ax[0].plot(np.arange(1, 21), np.array(d) ** 2, marker='.')
    # fig, ax = plt.subplots(1, 2, figsize=(14, 7))  # plot wh
    # ax[0].hist(wh[wh[:, 0]<100, 0],400)
    # ax[1].hist(wh[wh[:, 1]<100, 1],400)
    # fig.savefig('wh.png', dpi=200)

    # Evolve
    f, sh, mp, s = anchor_fitness(k), k.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
    pbar = tqdm(range(gen), bar_format=TQDM_BAR_FORMAT)  # progress bar
    for _ in pbar:
        v = np.ones(sh)
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)
        fg = anchor_fitness(kg)
        if fg > f:
            f, k = fg, kg.copy()
            pbar.desc = f"{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {f:.4f}"
            if verbose:
                print_results(k, verbose)

    return print_results(k).astype(np.float32)
