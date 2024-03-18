# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Train a YOLOv5 model on a custom dataset. Models and datasets download automatically from the latest YOLOv5 release.

Usage - Single-GPU training:
    $ python train.py --data coco128.yaml --weights yolov5s.pt --img 640  # from pretrained (recommended)
    $ python train.py --data coco128.yaml --weights '' --cfg yolov5s.yaml --img 640  # from scratch

Usage - Multi-GPU DDP training:
    $ python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data coco128.yaml --weights yolov5s.pt --img 640 --device 0,1,2,3

Models:     https://github.com/ultralytics/yolov5/tree/master/models
Datasets:   https://github.com/ultralytics/yolov5/tree/master/data
Tutorial:   https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
"""

import argparse
import math
import os
import random
import subprocess # 用于创建和管理子进程的模块，例如获取其输出或发送输入。
import sys
import time
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path

try:
    import comet_ml  # must be imported before torch (if installed) ，应当在导入torch前导入此块
except ImportError:
    comet_ml = None
# comet_ml 允许跟踪、记录和可视化机器学习实验的关键指标和结果。
import numpy as np
import torch
import torch.distributed as dist  # 用于分布式训练的模块
import torch.nn as nn
import yaml  # 解析配置文件的模块
from torch.optim import lr_scheduler  # 用于学习率调整的模块
from tqdm import tqdm  # 用于在命令行中显示循环的进度

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory  MY DIRECTORY : E:\code\pycham\yolo\yolov5
print(ROOT)
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative 将yolov5根目录路径转化成与此脚本的相对路径'.'

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors  # 检查锚点(anchor)的有效性
from utils.autobatch import check_train_batch_size  # 自动调整训练批次
from utils.callbacks import Callbacks  # 包含定义训练过程的中回调函数的功能
from utils.dataloaders import create_dataloader  # 数据加载器
from utils.downloads import attempt_download, is_url  # 尝试下载文件，以及检查给定的字符串是否为URL
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import LOGGERS, Loggers
from utils.loggers.comet.comet_utils import check_comet_resume  # 这个函数可能用于检查是否可以从 Comet ML 日志服务中恢复实验
from utils.loss import ComputeLoss
from utils.metrics import fitness  # 用于评估模型性能的指标，通常用于训练中
from utils.plots import plot_evolve  # 可视化模型性能的工具，绘制图表
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)
# 从环境变量中获取名为 "LOCAL_RANK" 的值，如果找不到该变量，则默认值为 -1。
LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()


def train(hyp, opt, device, callbacks):
    """
    Trains YOLOv5 model with given hyperparameters, options, and device, managing datasets, model architecture, loss
    computation, and optimizer steps.

    `hyp` argument is path/to/hyp.yaml or hyp dictionary.
    """
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.evolve,
        opt.data,
        opt.cfg,
        opt.resume,
        opt.noval,  # 是否在训练过程中不进行验证
        opt.nosave,
        opt.workers,  # 工作进程数
        opt.freeze,
    )
    callbacks.run("on_pretrain_routine_start")  # 回调函数，用于在模型预训练阶段执行特定的操作，当执行train函数时，通过调用此函数可以触发与预训练相关的回调函数

    # Directories
    w = save_dir / "weights"  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters
    if isinstance(hyp, str):  # 如果参数类型是字符串则尝试从路径加载YAML格式的文件，并将超参数存储在hyp变量中
        with open(hyp, errors="ignore") as f:  # errors="ignore" 表示在遇到无法解码的字符时，将忽略这些错误而继续读取文件内容
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))  # info用于记录信息级别的日志
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints 复制一份到opt.hyp，以便于保留最初的超参数

    # Save run settings
    if not evolve:  # 当不处于评估模式时，将超参数和选项保存到指定的yaml文件中
        yaml_save(save_dir / "hyp.yaml", hyp)
        yaml_save(save_dir / "opt.yaml", vars(opt))  # 根据用户提供的选项添加一个后缀，以区分不同的文件或实验配置。

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        include_loggers = list(LOGGERS)
        if getattr(opt, "ndjson_console", False):
            include_loggers.append("ndjson_console")
        if getattr(opt, "ndjson_file", False):
            include_loggers.append("ndjson_file")

        loggers = Loggers(
            save_dir=save_dir,
            weights=weights,
            opt=opt,
            hyp=hyp,
            logger=LOGGER,
            include=tuple(include_loggers),
        )

        # Register actions
        for k in methods(loggers):
            print(k)
            # # 这里是将循环遍历 methods(loggers) 返回的方法列表，并将每个方法作为回调函数注册到 callbacks 对象的相应钩子上。
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size

    # Config
    plots = not evolve and not opt.noplots  # create plots 如果opt.noplots 的值为真（即非零），则不会创建图表
    cuda = device.type != "cpu"
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with torch_distributed_zero_first(LOCAL_RANK):  # LOCAL_RANK表示当前节点的本地片名，with是上下文管理器，在代码块执行后释放资源
        data_dict = data_dict or check_dataset(data)  # check if None 在进入上下文环境之前会执行，用于检查 data_dict 是否为 None，
        # 如果为 None，则调用 check_dataset(data) 函数来获取数据集字典。用于确保数据集字典存在
    train_path, val_path = data_dict["train"], data_dict["val"]  # 加载训练集和验证集路径
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes 如果是单类别数据集，则 nc 设置为 1；否则，将从数据集字典中获取类别数量。
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset
    print('is_coco:',is_coco)

    # Model
    check_suffix(weights, ".pt")  # check weights 检查权重文件的后缀名是否为pt
    pretrained = weights.endswith(".pt")
    if pretrained:  # 判断是否为预训练模型
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(weights)  # download if not found locally 如果采用预训练的权重文件，在本地未找到则利用此函数从GitHub上下载
        ckpt = torch.load(weights, map_location="cpu")  # load checkpoint to CPU to avoid CUDA memory leak  加载模型权重
        #   模型的配置、通道数、预测类别数、预定义锚点的参数
        model = Model(cfg or ckpt["model"].yaml, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create
        exclude = ["anchor"] if (cfg or hyp.get("anchors")) and not resume else []  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load 加载模型参数
        LOGGER.info(f"Transferred {len(csd)}/{len(model.state_dict())} items from {weights}")  # report
    else:
        model = Model(cfg, ch=3, nc=nc, anchors=hyp.get("anchors")).to(device)  # create 如果不是与训练模型则直接创建模型，不需要预训练权重文件
    amp = check_amp(model)  # check AMP 检查模型是否需要混合精度训练
    # Freeze
    # 设置冻结模型，使其模型的某些层被冻结，即保留某些层的参数，使这些层的参数在反向传播时不受影响。
    # 遍历模型中的参数，并且为模型中的参数前均加一个model.以确保与其它同名参数不重复，如果free是整数则遍历将freeze变成一个长度为freeze的范围序列
    freeze = [f"model.{x}." for x in (freeze if len(freeze) > 1 else range(freeze[0]))]  # layers to freeze
    # k代表模型的参数名称，v代表模型参数的值
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers  这里是将所有层的requires_grad均设置为true即表示为所有参数均可训练，无冻结层
        # v.register_hook(lambda x: torch.nan_to_num(x))  # NaN to 0 (commented for erratic training results)
        if any(x in k for x in freeze):  # 若x在k序列中，且为冻结层，设置requires_grad为false，即其参数不可更新
            LOGGER.info(f"freezing {k}")  # 将不可训练的层记录在日志中
            v.requires_grad = False

    # Image size
    # grid size (max stride) 这里取模型中最大的步长，如果其小于32，则将32作为模型的最大步长，以便后期对大尺寸的图像进行tiling处理（图像切片处理）。
    gs = max(int(model.stride.max()), 32)
    # 确保图像大小至少为 gs * 2。如果指定的 opt.imgsz 不符合要求，将会被调整为符合要求的大小。
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    # RANK=-1表示使用单gpu进行训练
    # 当 batch_size 被设置为 -1 时，通常表示使用整个数据集作为一个批次进行训练。即对所有输入数据进行一次参数更新
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)  # 自动选择新的batch——size
        loggers.on_params_update({"batch_size": batch_size})

    # Optimizer
    nbs = 64  # nominal batch size 作为一个预设的batch-size,通常用来作为训练时参考的批次大小
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing  决定了在进行一次参数更新前累积了多少个批次的梯度
    hyp["weight_decay"] *= batch_size * accumulate / nbs  # scale weight_decay 对权重衰减进行了调整，减小了模型的参数的大小
    # # 使用了给定的学习率、动量和权重衰减参数来初始化优化器
    optimizer = smart_optimizer(model, opt.optimizer, hyp["lr0"], hyp["momentum"], hyp["weight_decay"])

    # Scheduler
    if opt.cos_lr:  # 如果选择余弦退火学习率
        lf = one_cycle(1, hyp["lrf"], epochs)  # cosine 1->hyp['lrf'] 采用余弦退火更新学习率
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp["lrf"]) + hyp["lrf"]  # linear
    # lr_lambda得到当前学习率的缩放系数，其用于和初始化学习率相乘，从而得到当前的学习率.
    # scheduler 是一个 LambdaLR 类的实例，如果需要获取当前学习率的值，可以调用 scheduler.get_lr() 方法。
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None  # 创建一个移动平均模型。  RANK 的值为 -1 或 0 指当前的环境是单机训练。

    # Resume
    best_fitness, start_epoch = 0.0, 0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd   # 删除检查点（通常含有模型的权重、优化器状态、其它相关的训练信息）以及模型的状态字典

    # DP mode  如果有多个gup才会使用，否则不启用
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning(
            "WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.\n"
            "See Multi-GPU Tutorial at https://docs.ultralytics.com/yolov5/tutorials/multi_gpu_training to get started."
        )
        model = torch.nn.DataParallel(model)   # 应用于附加的硬件进行加载模型

    # SyncBatchNorm
    # 如果有多个gup才会使用，否则不启用
    # 是否启用同步批归一化，opt.sync_bn为true表示启用同步批归一化，rank！=-1表示当前进程不是主进程
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info("Using SyncBatchNorm()")

    # Trainloader
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,  # 每个gpu的批量大小，除以world_size是总进程数,用于分布式训练
        gs,  # 目标图像的大小
        single_cls,  # 是否为单类别检测
        hyp=hyp,  # 训练的超参数
        augment=True,  # 是否进行数据增强
        cache=None if opt.cache == "val" else opt.cache,  # 数据加载器是否使用缓存，如果不使用则按需从网络或者硬盘中调用，使用后从内存中的缓存调用
        rect=opt.rect,  # 是否使用矩阵图像训练
        rank=LOCAL_RANK,  # 本地进程的排名
        workers=workers,  # 数据加载器的工作进程数
        image_weights=opt.image_weights,  # 图像的权重
        quad=opt.quad,  # 是否使用四分之一缩放
        prefix=colorstr("train: "),  # 日志前缀，用于标识训练过程
        shuffle=True,  # 是否在每一个epoch开始时打乱数据集
        seed=opt.seed,
    )
    labels = np.concatenate(dataset.labels, 0)  # 这行代码将数据集中的所有标签连接成一个numpy数组
    mlc = int(labels[:, 0].max())  # max label class 第一列即索引值，获取所有标签的最大类别索引。
    # 检查最大索引数是否小于所有种类类别数，若大于则会引发断言错误
    # 这是一个断言语句，当前面的条件不成立时则断言失败，会提示后面的错误信息
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in {-1, 0}:  # 这里判断是否为单机训练的主进程或者在单机模式下运行，然后才创建验证数据加载器，是因为当不同进程并行运行时，可能要考虑验证数据的分配问题
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,  # 如果naval为真，则不缓存数据
            rect=True,  # 表示是否对图像进行矩形裁剪
            rank=-1,  # 表示在单机情况下运行
            workers=workers * 2,  # 工作线程的总数
            pad=0.5,  # 用于图像增强的填充
            prefix=colorstr("val: "),  # 一个前缀，用于标识验证数据加载器
        )[0]  # 这里的[0]表示其函数返回的元组中选择索引为0的元素（数据加载器本身），因为这个元组中还包含了其它的额外的配置参数，故此不仅仅是数据加载去本身。

        if not resume:  # resume表示是否从之前的训练中恢复模型和优化器的状态
            if not opt.noautoanchor:  # 是否启用了自动锚定功能
                check_anchors(dataset, model=model, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor 运行自动锚定算法
            model.half().float()  # pre-reduce anchor precision  half（）将模型的参数转换为半精度浮点数格式，然后用float（）转换为单精度浮点数格式，可以提高训练速度

        callbacks.run("on_pretrain_routine_end", labels, names)

    # DDP mode 分布式数据并行ddp模式的设置
    if cuda and RANK != -1:
        model = smart_DDP(model)

    # Model attributes
    # 以下代码中用于初始化模型训练过程中的超参数和属性，来确保正确的处理输入数据
    nl = de_parallel(model).model[-1].nl  # number of detection layers (to scale hyps) 获取了模型的最后一个模块（检测头）的层数
    hyp["box"] *= 3 / nl  # scale to layers 将检测框的超参数乘一个比例因子，使其调整到与其类别和层数相关的尺度
    hyp["cls"] *= nc / 80 * 3 / nl  # scale to classes and layers
    hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers 其中640是标准尺寸
    hyp["label_smoothing"] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    # 将类别权重（class weights）附加到模型，通常是用于在不平衡的数据集上进行训练
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights 得每个类别所占的权重
    model.names = names  # 将类别名称附加到模型

    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches ，当通过pbar = enumerate(train_loader)迭代器生效后，可以得到其长度即训练批次的大小
    # 这里是让学习率逐渐增加，若逐渐减小则选择下面一行注释部分代码
    # 其round是四舍五入取整，max确保预热阶段的总迭代次数不会小于100 所谓预热阶段即为在开始训练的若干周期内逐渐增加学习率的过程
    nw = max(round(hyp["warmup_epochs"] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    # nw = min(nw, (epochs - start_epoch) / 2 * nb)  # limit warmup to < 1/2 of training
    last_opt_step = -1  # 初始化最后一次优化步骤的索引为-1
    maps = np.zeros(nc)  # mAP per class 初始化一个长度为类别数量的数组，用于存储每个类别的map
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls) 包括精确度、召回率、map在阈值为0.5和0.95时的值、损失
    scheduler.last_epoch = start_epoch - 1  # do not move 将学习调度器上次更新学习率所处的训练周期数设置为这次训练起始周期的前一个周期 ...
    # ... 由于last_epoch的默认值为-1，学习率调度器会认为上一次更新实在-1周期，即在训练开始前，这意味着学习率调度器会认为自己已经错过了第一个周期，只会等到第二个周期才会更新学习率
    scaler = torch.cuda.amp.GradScaler(enabled=amp)  # 初始化梯度放大器（GradientScaler），用于混合精度训练
    # 初始化早停策略，用在训练过程中根据验证集的表现来提前终止训练，patience表示容忍验证集表现不再改善的epoch数
    stopper, stop = EarlyStopping(patience=opt.patience), False
    compute_loss = ComputeLoss(model)  # init loss class  #初始化计算训练过程中的损失函数
    callbacks.run("on_train_start")
    LOGGER.info(
        f'Image sizes {imgsz} train, {imgsz} val\n'
        f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
        f"Logging results to {colorstr('bold', save_dir)}\n"
        f'Starting training for {epochs} epochs...'
    )
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        callbacks.run("on_train_epoch_start")
        model.train()  # 将模型设为训练模式
        # Update image weights (optional, single-GPU only)
        if opt.image_weights:  # 若设置了图像的权重参数
            cw = model.class_weights.cpu().numpy() * (1 - maps) ** 2 / nc  # class weights  # 计算类别权重
            iw = labels_to_image_weights(dataset.labels, nc=nc, class_weights=cw)  # image weights 根据类别权重计算图像的权重
            dataset.indices = random.choices(range(dataset.n), weights=iw, k=dataset.n)  # rand weighted idx  根据图像的权重重新采样数据集的索引

        # Update mosaic border (optional)
        # b = int(random.uniform(0.25 * imgsz, 0.75 * imgsz + gs) // gs * gs)
        # dataset.mosaic_border = [b - imgsz, -b]  # height, width borders

        mloss = torch.zeros(3, device=device)  # mean losses  存储损失值
        if RANK != -1:
            train_loader.sampler.set_epoch(epoch)
        pbar = enumerate(train_loader)  # 在每一个循环中使用此数据加载器，数据加载器将返回一个带索引和数据的数组，其中索引以批次为单位
        LOGGER.info(("\n" + "%11s" * 7) % ("Epoch", "GPU_mem", "box_loss", "obj_loss", "cls_loss", "Instances", "Size"))
        if RANK in {-1, 0}:  # 表示唯一进程
            pbar = tqdm(pbar, total=nb, bar_format=TQDM_BAR_FORMAT)  # progress bar  # 用于创建一个进度条用来显示训练进度
        optimizer.zero_grad()  # 每个训练周期开始将优化器的梯度归零
        # 遍历数据加载器中每个批次的数据
        # i：当前周期内的批次索引。
        # imgs：图像批次。
        # targets：相应目标（标签）的批次。
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            callbacks.run("on_train_batch_start")
            # number integrated batches (since train start)  # 计算从开始到现在已经累计的批次数量，其中nb是每个周期的批次数量，epoch是周期数，i表示当前批次在周期中的索引
            ni = i + nb * epoch
            # uint8 to float32, 0-255 to 0.0-1.0  用于将图像数据转移到gpu，且将像素值转换到0.0-1.0
            imgs = imgs.to(device, non_blocking=True).float() / 255

            # Warmup  用于在训练初期调整学习率
            if ni <= nw:  # 若当前的训练批次小于指定的批次nw
                xi = [0, nw]  # x interp
                # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [hyp["warmup_bias_lr"] if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [hyp["warmup_momentum"], hyp["momentum"]])

            # Multi-scale
            # 使用多尺度训练，每个训练迭代中使用不同尺度大小的图像。
            if opt.multi_scale:
                # size 尺度变换在原始图像的0.5到1.5倍之间，并确保是gird size的倍数
                sz = random.randrange(int(imgsz * 0.5), int(imgsz * 1.5) + gs) // gs * gs
                # 新尺度与原始图像尺度的比率
                sf = sz / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    # 将每个原始图像的每个尺度按照上式sf缩放因子进行缩放
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                    # 并将原始图像利用双线性插值将图形插值到新的尺寸ns
                    imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)

            # Forward
            with torch.cuda.amp.autocast(amp):  # 采用混合精度计算，将浮点数转换成半浮点数，amp是一个布尔值，决定是否开启混合精度训练
                pred = model(imgs)  # forward  # 利用模型将图片作为输入进行前向传播
                loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size调用compute_loss计算损失
                if RANK != -1:
                    loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
                if opt.quad:  # 如果启用了四通道模式，则将计算得到的损失值乘以4，这样做更利于调整学习率
                    loss *= 4.0

            # Backward
            scaler.scale(loss).backward()  # 反向传播

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:  # 是否达到了累计梯度的步骤数，last_opt_step 表示上一次执行优化步骤的迭代步数
                scaler.unscale_(optimizer)  # unscale gradients
                # clip gradients  对模型的梯度进行裁剪，确保梯度的范数不超过给定的最大值（这里是10.0）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)  # optimizer.step  混合精度训练中，由于梯度被缩放过，所以需要用scaler来执行梯度更新操作。
                scaler.update()  # 下一次梯度更新时使用新的缩放因子。
                optimizer.zero_grad()  # 将优化器中的梯度置零，以便下一次迭代计算新的梯度。
                if ema:
                    ema.update(model)  # 在每一个训练批次中执行，用以确保指数移动平均模型的参数与当前模型的参数保持同步。
                last_opt_step = ni  # 更新 last_opt_step 为当前的迭代步数 ni

            # Log
            if RANK in {-1, 0}:
                mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
                mem = f"{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G"  # (GB)
                pbar.set_description(
                    ("%11s" * 2 + "%11.4g" * 5)
                    % (f"{epoch}/{epochs - 1}", mem, *mloss, targets.shape[0], imgs.shape[-1])
                )
                callbacks.run("on_train_batch_end", model, ni, imgs, targets, paths, list(mloss))
                if callbacks.stop_training:
                    return
            # end batch ------------------------------------------------------------------------------------------------

        # Scheduler
        lr = [x["lr"] for x in optimizer.param_groups]  # for loggers 将每个参数组的学习率存储在lr列表中
        scheduler.step()  # 修改学习率

        if RANK in {-1, 0}:
            # mAP
            callbacks.run("on_train_epoch_end", epoch=epoch)
            # 更新指数滑动平均模型的属性
            ema.update_attr(model, include=["yaml", "nc", "hyp", "names", "stride", "class_weights"])
            # 确定是否是最后一个epoch
            final_epoch = (epoch + 1 == epochs) or stopper.possible_stop
            # 检查是否需要进行验证，或者是否是最后一个epoch。
            if not noval or final_epoch:  # Calculate mAP
                # 运行验证过程，获取验证结果和mAP
                results, maps, _ = validate.run(
                    data_dict,
                    batch_size=batch_size // WORLD_SIZE * 2,
                    imgsz=imgsz,
                    half=amp,
                    model=ema.ema,
                    single_cls=single_cls,
                    dataloader=val_loader,
                    save_dir=save_dir,
                    plots=False,
                    callbacks=callbacks,
                    compute_loss=compute_loss,
                )

            # Update best mAP
            # 根据Precision、Recall和mAP等指标进行加权组合得到的综合分数。
            fi = fitness(np.array(results).reshape(1, -1))  # weighted combination of [P, R, mAP@.5, mAP@.5-.95]
            # 检查是否满足提前停止的条件。
            stop = stopper(epoch=epoch, fitness=fi)  # early stop check
            if fi > best_fitness:
                best_fitness = fi
            log_vals = list(mloss) + list(results) + lr
            callbacks.run("on_fit_epoch_end", log_vals, epoch, best_fitness, fi)

            # Save model
            if (not nosave) or (final_epoch and not evolve):  # if save
                ckpt = {
                    "epoch": epoch,
                    "best_fitness": best_fitness,
                    "model": deepcopy(de_parallel(model)).half(),
                    "ema": deepcopy(ema.ema).half(),
                    "updates": ema.updates,
                    "optimizer": optimizer.state_dict(),
                    "opt": vars(opt),
                    "git": GIT_INFO,  # {remote, branch, commit} if a git repo
                    "date": datetime.now().isoformat(),
                }

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fi:
                    torch.save(ckpt, best)
                # 当前epoch满足保存周期的条件，则保存模型参数到文件中，文件名包含了当前epoch数。
                if opt.save_period > 0 and epoch % opt.save_period == 0:
                    torch.save(ckpt, w / f"epoch{epoch}.pt")
                # 删除保存模型的临时变量，释放内存空间
                del ckpt
                callbacks.run("on_model_save", last, epoch, final_epoch, best_fitness, fi)

        # EarlyStopping
        if RANK != -1:  # if DDP training
            broadcast_list = [stop if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
            if RANK != 0:
                stop = broadcast_list[0]
        if stop:
            break  # must break all DDP ranks

        # end epoch ----------------------------------------------------------------------------------------------------
    # end training -----------------------------------------------------------------------------------------------------
    if RANK in {-1, 0}:
        # 输出训练已完成的epoch数以及训练所花费的时间，以小时为单位。
        LOGGER.info(f"\n{epoch - start_epoch + 1} epochs completed in {(time.time() - t0) / 3600:.3f} hours.")
        for f in last, best:
            if f.exists():
                # 移除优化器信息，保留模型参数。
                strip_optimizer(f)  # strip optimizers
                if f is best:
                    LOGGER.info(f"\nValidating {f}...")
                    results, _, _ = validate.run(
                        data_dict,
                        batch_size=batch_size // WORLD_SIZE * 2,
                        imgsz=imgsz,
                        # 进行验证，获取验证结果。验证时使用模型加载函数attempt_load加载模型参数
                        model=attempt_load(f, device).half(),
                        iou_thres=0.65 if is_coco else 0.60,  # best pycocotools at iou 0.65
                        single_cls=single_cls,
                        dataloader=val_loader,
                        save_dir=save_dir,
                        save_json=is_coco,
                        verbose=True,
                        plots=plots,
                        callbacks=callbacks,
                        compute_loss=compute_loss,
                    )  # val best model with plots
                    if is_coco:
                        callbacks.run("on_fit_epoch_end", list(mloss) + list(results) + lr, epoch, best_fitness, fi)

        callbacks.run("on_train_end", last, best, epoch, results)

    torch.cuda.empty_cache()  # 清空cuda缓存
    return results


def parse_opt(known=False):
    """Parses command-line arguments for YOLOv5 training, validation, and testing."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov5s.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument(
        "--evolve_population", type=str, default=ROOT / "data/hyps", help="location for loading population"
    )
    parser.add_argument("--resume_evolve", type=str, default=None, help="resume evolve from last generation")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # NDJSON logging
    parser.add_argument("--ndjson-console", action="store_true", help="Log ndjson to console")
    parser.add_argument("--ndjson-file", action="store_true", help="Log ndjson to file")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt, callbacks=Callbacks()):
    """Runs training or hyperparameter evolution with specified options and optional callbacks."""
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.evolve:
            if opt.project == str(ROOT / "runs/train"):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / "runs/evolve")
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv5 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo", timeout=timedelta(seconds=10800)
        )

    # Train
    if not opt.evolve:
        train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (including this hyperparameter True-False, lower_limit, upper_limit)
        meta = {
            "lr0": (False, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            "lrf": (False, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            "momentum": (False, 0.6, 0.98),  # SGD momentum/Adam beta1
            "weight_decay": (False, 0.0, 0.001),  # optimizer weight decay
            "warmup_epochs": (False, 0.0, 5.0),  # warmup epochs (fractions ok)
            "warmup_momentum": (False, 0.0, 0.95),  # warmup initial momentum
            "warmup_bias_lr": (False, 0.0, 0.2),  # warmup initial bias lr
            "box": (False, 0.02, 0.2),  # box loss gain
            "cls": (False, 0.2, 4.0),  # cls loss gain
            "cls_pw": (False, 0.5, 2.0),  # cls BCELoss positive_weight
            "obj": (False, 0.2, 4.0),  # obj loss gain (scale with pixels)
            "obj_pw": (False, 0.5, 2.0),  # obj BCELoss positive_weight
            "iou_t": (False, 0.1, 0.7),  # IoU training threshold
            "anchor_t": (False, 2.0, 8.0),  # anchor-multiple threshold
            "anchors": (False, 2.0, 10.0),  # anchors per output grid (0 to ignore)
            "fl_gamma": (False, 0.0, 2.0),  # focal loss gamma (efficientDet default gamma=1.5)
            "hsv_h": (True, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            "hsv_s": (True, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            "hsv_v": (True, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            "degrees": (True, 0.0, 45.0),  # image rotation (+/- deg)
            "translate": (True, 0.0, 0.9),  # image translation (+/- fraction)
            "scale": (True, 0.0, 0.9),  # image scale (+/- gain)
            "shear": (True, 0.0, 10.0),  # image shear (+/- deg)
            "perspective": (True, 0.0, 0.001),  # image perspective (+/- fraction), range 0-0.001
            "flipud": (True, 0.0, 1.0),  # image flip up-down (probability)
            "fliplr": (True, 0.0, 1.0),  # image flip left-right (probability)
            "mosaic": (True, 0.0, 1.0),  # image mixup (probability)
            "mixup": (True, 0.0, 1.0),  # image mixup (probability)
            "copy_paste": (True, 0.0, 1.0),
        }  # segment copy-paste (probability)

        # GA configs
        pop_size = 50
        mutation_rate_min = 0.01
        mutation_rate_max = 0.5
        crossover_rate_min = 0.5
        crossover_rate_max = 1
        min_elite_size = 2
        max_elite_size = 5
        tournament_size_min = 2
        tournament_size_max = 10

        with open(opt.hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            if "anchors" not in hyp:  # anchors commented in hyp.yaml
                hyp["anchors"] = 3
        if opt.noautoanchor:
            del hyp["anchors"], meta["anchors"]
        opt.noval, opt.nosave, save_dir = True, True, Path(opt.save_dir)  # only val/save final epoch
        # ei = [isinstance(x, (int, float)) for x in hyp.values()]  # evolvable indices
        evolve_yaml, evolve_csv = save_dir / "hyp_evolve.yaml", save_dir / "evolve.csv"
        if opt.bucket:
            # download evolve.csv if exists
            subprocess.run(
                [
                    "gsutil",
                    "cp",
                    f"gs://{opt.bucket}/evolve.csv",
                    str(evolve_csv),
                ]
            )

        # Delete the items in meta dictionary whose first value is False
        del_ = [item for item, value_ in meta.items() if value_[0] is False]
        hyp_GA = hyp.copy()  # Make a copy of hyp dictionary
        for item in del_:
            del meta[item]  # Remove the item from meta dictionary
            del hyp_GA[item]  # Remove the item from hyp_GA dictionary

        # Set lower_limit and upper_limit arrays to hold the search space boundaries
        lower_limit = np.array([meta[k][1] for k in hyp_GA.keys()])
        upper_limit = np.array([meta[k][2] for k in hyp_GA.keys()])

        # Create gene_ranges list to hold the range of values for each gene in the population
        gene_ranges = [(lower_limit[i], upper_limit[i]) for i in range(len(upper_limit))]

        # Initialize the population with initial_values or random values
        initial_values = []

        # If resuming evolution from a previous checkpoint
        if opt.resume_evolve is not None:
            assert os.path.isfile(ROOT / opt.resume_evolve), "evolve population path is wrong!"
            with open(ROOT / opt.resume_evolve, errors="ignore") as f:
                evolve_population = yaml.safe_load(f)
                for value in evolve_population.values():
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # If not resuming from a previous checkpoint, generate initial values from .yaml files in opt.evolve_population
        else:
            yaml_files = [f for f in os.listdir(opt.evolve_population) if f.endswith(".yaml")]
            for file_name in yaml_files:
                with open(os.path.join(opt.evolve_population, file_name)) as yaml_file:
                    value = yaml.safe_load(yaml_file)
                    value = np.array([value[k] for k in hyp_GA.keys()])
                    initial_values.append(list(value))

        # Generate random values within the search space for the rest of the population
        if initial_values is None:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size)]
        elif pop_size > 1:
            population = [generate_individual(gene_ranges, len(hyp_GA)) for _ in range(pop_size - len(initial_values))]
            for initial_value in initial_values:
                population = [initial_value] + population

        # Run the genetic algorithm for a fixed number of generations
        list_keys = list(hyp_GA.keys())
        for generation in range(opt.evolve):
            if generation >= 1:
                save_dict = {}
                for i in range(len(population)):
                    little_dict = {list_keys[j]: float(population[i][j]) for j in range(len(population[i]))}
                    save_dict[f"gen{str(generation)}number{str(i)}"] = little_dict

                with open(save_dir / "evolve_population.yaml", "w") as outfile:
                    yaml.dump(save_dict, outfile, default_flow_style=False)

            # Adaptive elite size
            elite_size = min_elite_size + int((max_elite_size - min_elite_size) * (generation / opt.evolve))
            # Evaluate the fitness of each individual in the population
            fitness_scores = []
            for individual in population:
                for key, value in zip(hyp_GA.keys(), individual):
                    hyp_GA[key] = value
                hyp.update(hyp_GA)
                results = train(hyp.copy(), opt, device, callbacks)
                callbacks = Callbacks()
                # Write mutation results
                keys = (
                    "metrics/precision",
                    "metrics/recall",
                    "metrics/mAP_0.5",
                    "metrics/mAP_0.5:0.95",
                    "val/box_loss",
                    "val/obj_loss",
                    "val/cls_loss",
                )
                print_mutation(keys, results, hyp.copy(), save_dir, opt.bucket)
                fitness_scores.append(results[2])

            # Select the fittest individuals for reproduction using adaptive tournament selection
            selected_indices = []
            for _ in range(pop_size - elite_size):
                # Adaptive tournament size
                tournament_size = max(
                    max(2, tournament_size_min),
                    int(min(tournament_size_max, pop_size) - (generation / (opt.evolve / 10))),
                )
                # Perform tournament selection to choose the best individual
                tournament_indices = random.sample(range(pop_size), tournament_size)
                tournament_fitness = [fitness_scores[j] for j in tournament_indices]
                winner_index = tournament_indices[tournament_fitness.index(max(tournament_fitness))]
                selected_indices.append(winner_index)

            # Add the elite individuals to the selected indices
            elite_indices = [i for i in range(pop_size) if fitness_scores[i] in sorted(fitness_scores)[-elite_size:]]
            selected_indices.extend(elite_indices)
            # Create the next generation through crossover and mutation
            next_generation = []
            for _ in range(pop_size):
                parent1_index = selected_indices[random.randint(0, pop_size - 1)]
                parent2_index = selected_indices[random.randint(0, pop_size - 1)]
                # Adaptive crossover rate
                crossover_rate = max(
                    crossover_rate_min, min(crossover_rate_max, crossover_rate_max - (generation / opt.evolve))
                )
                if random.uniform(0, 1) < crossover_rate:
                    crossover_point = random.randint(1, len(hyp_GA) - 1)
                    child = population[parent1_index][:crossover_point] + population[parent2_index][crossover_point:]
                else:
                    child = population[parent1_index]
                # Adaptive mutation rate
                mutation_rate = max(
                    mutation_rate_min, min(mutation_rate_max, mutation_rate_max - (generation / opt.evolve))
                )
                for j in range(len(hyp_GA)):
                    if random.uniform(0, 1) < mutation_rate:
                        child[j] += random.uniform(-0.1, 0.1)
                        child[j] = min(max(child[j], gene_ranges[j][0]), gene_ranges[j][1])
                next_generation.append(child)
            # Replace the old population with the new generation
            population = next_generation
        # Print the best solution found
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]  # 得到适应度最高的个体，此个体是一个向量，包含了所有参数
        print("Best solution found:", best_individual)
        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(
            f'Hyperparameter evolution finished {opt.evolve} generations\n'
            f"Results saved to {colorstr('bold', save_dir)}\n"
            f'Usage example: $ python train.py --hyp {evolve_yaml}'
        )


def generate_individual(input_ranges, individual_length):
    """Generates a list of random values within specified input ranges for each gene in the individual."""
    individual = []
    for i in range(individual_length):
        lower_bound, upper_bound = input_ranges[i]
        individual.append(random.uniform(lower_bound, upper_bound))
    return individual


def run(**kwargs):
    """
    Executes YOLOv5 training with given options, overriding with any kwargs provided.

    Example: import train; train.run(data='coco128.yaml', imgsz=320, weights='yolov5m.pt')
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
