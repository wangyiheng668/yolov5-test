# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
)
from models.experimental import MixConv2d
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,  # 将卷积层和批量归一化层进行融合
    initialize_weights,  # 初始化模型的权重，通常在创建模型的实例后调用该函数
    model_info,  # 获取模型的信息
    profile,  # 对模型的性能分析，例如就算模型的前向传播时间
    scale_img,  # 调整输入图像的尺寸
    select_device,
    time_sync,  # 用于计算模型flops（浮点运算数）的工具，可用于评估模型的计算复杂度
)

try:
    import thop  # for FLOPs computation
except ImportError:
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build
    dynamic = False  # force grid reconstruction
    export = False  # export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # inplace是切片赋值的操作，即可以在原始存储空间直接修改变量而不是返回新的对象
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor  每个锚点的输出量（即类别+xywh和置信度）
        self.nl = len(anchors)  # number of detection layers  层的数量=锚点的元组长度
        self.na = len(anchors[0]) // 2  # number of anchors  每个检测层使用的锚点数目，anchors[0]代表第一个检测层，除以2是应为每个锚点有两个长度单位，即宽和高
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid  存储每个层的网格监测信息
        # 存储每个检测层的锚点网格信息的列表，即每个检测层的每个网格都有一个锚点信息
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid ...
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2) 将锚点转换为张量
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv  输出的卷积模块列表中每一个都是conv2d实例
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                if isinstance(self, Segment):  # (boxes + masks)  检查当前的模型是不是segment类或者子类的实例
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                else:  # Detect (boxes only)
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    y = torch.cat((xy, wh, conf), 4)
                z.append(y.view(bs, self.na * nx * ny, self.no))

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos
        self.detect = Detect.forward

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):
    """YOLOv5 base model."""

    # 这样将具体的计算过程封装到一个_私有方法(_forward_once)
    def forward(self, x, profile=False, visualize=False):
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    # 主要实现了前向传播的单次操作
    # 三个输入分别为输入数据、是否进行性能分析、是否进行特征可视化
    def _forward_once(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                # 若m.f为整型，则x之来源于一个层，输入为这个层的输出
                # 若m.f为列表，表示x为多层的输出作为输入数据，当j=-1时代表使用当前层的数据作为输入；当j=其它值时，使用第j层的输出y[j]作为输入，由此构成了一个输入数据列表
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:  # 调用_profile_one_layer对当前层进行性能分析
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output 保存这一层的输出，并添加到y中
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x # x是当前层的输出，将其返回传入到下一层的输入

    def _profile_one_layer(self, m, x, dt):
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        t = time_sync()
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")

    def fuse(self):  # 融合模型中的 Conv2d() 和 BatchNorm2d() 层，以提高推理速度。
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, "bn")  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, img_size=640):  # 打印模型的信息，包括各层的参数数量
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)

    def _apply(self, fn):  # 对模型的张量执行转换，只有当模型的最后一个组件是 Detect 或 Segment 类型时才会进行处理。
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg="yolov5s_ball.yaml", ch=3, nc=None, anchors=None):  # 这个3代表输入图像的通道，这里是3，可代表rgb三通道
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors."""
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg  # model dict
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value
        # 在这里创建模型，并使用deepcopy(self.yaml)将parse_model中的d设置为self.yaml
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases
        initialize_weights(self)
        self.info()
        LOGGER.info("")

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        if augment:
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train

    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales
        f = [None, 3, None]  # flips (2-ud, 3-lr)
        y = []  # outputs
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]  # forward
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        return torch.cat(y, 1), None  # augmented inference, train

    def _descale_pred(self, p, flips, scale, img_size):
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
        if self.inplace:
            p[..., :4] /= scale  # de-scale
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
        else:
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)
        return p

    def _clip_augmented(self, y):
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        g = sum(4**x for x in range(nl))  # grid points
        e = 1  # exclude layer count
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        y[0] = y[0][:, :-i]  # large
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        y[-1] = y[-1][:, i:]  # small
        return y

    def _initialize_biases(self, cf=None):
        """
        Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility


class SegmentationModel(DetectionModel):
    # YOLOv5 segmentation model
    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for channels, nc (int) for num classes, anchors (list)."""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel):
    # YOLOv5 classification model
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)

    def _from_detection_model(self, model, nc=1000, cutoff=10):
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification
        layer.
        """
        if isinstance(model, DetectMultiBackend):
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone仅保留前cutoff层，通常用于去掉检测头部分
        m = model.model[-1]  # last layer 截取模型的最后一层
        # 获取模型最后一层的输入通道数
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        # 创建了一个分类层，并传入通道数ch和类别数量nc
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type
        model.model[-1] = c  # replace
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc

    def _from_yaml(self, cfg):
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        self.model = None


# --------------------------------------------------------构建整个模型的连接------------------------------------------------------
def parse_model(d, ch):  # 这里的字典d是从yaml配置文件中导入，具体查看detectmodel类
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    # gd深度倍数、gw宽度倍数、 activation激活函数类型、ch_mul通道倍数
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
    if not ch_mul:  # 是否指定了通道数
        ch_mul = 6  # 此处由8改成了6
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)  模型的所有输出量（输出的维度）（锚点数量乘以（类别数量加上5））

    # layers 用于存储模型的各个层次，save 用于存储需要保存特征图的层次索引
    # 将当前层次的输出通道数初始化为输入通道列表ch的最后一个值，这样通过不断更新ch，确保了每个模块在构建过程中都以前一个模块的输出通道数作为输入通道数，从而保持通道数的连续性。
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    # 这里是将字典中的backbone列表和head列表进行合并，然后进行迭代遍历。其中i是枚举的索引，而 (f, n, m, args) 是枚举的值，分别表示接收那一层的数据、数量、模块和参数。
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args,其中f指的是当前层的输入来源来自于那一层
        m = eval(m) if isinstance(m, str) else m  # eval strings
        # 例如args=[3, 32, 6, 2, 2]，则其中分别代表输入通道、输出通道、卷积核大小、卷积步长、卷积填充值
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
        }:
            c1, c2 = ch[f], args[0]  # 这里依次将c1设置为ch的最后一个值，即确保通道的输入是上一层的输出
            if c2 != no:  # if not output
                # 即将输出展宽，这里指的是输出通道
                c2 = make_divisible(c2 * gw, ch_mul)  # 将给定的数字 c2 * gw 调整为可被 ch_mul 整除的最接近的大于等于c2 * gw 的值

            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
        else:
            c2 = ch[f]

        # 根据模块类型 m、参数 args 和重复次数 n 构建模块 m_。构建这一个同种类的连接。
        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        # 去掉m的前缀<class ' 以及后缀 ‘> 并保存到t中
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        # 计算即模型的参数量。一次迭代为（这一种块所构成的参数量），下次迭代通过sum函数进行相加
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2)
    # 将构建好的模块列表传递给 nn.Sequential 构建一个包含所有模块的神经网络模型。并保存列表进行排序
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
