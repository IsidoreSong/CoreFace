import argparse
import logging
import os
import time
import random
import numpy as np

# from pyinstrument import Profiler
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

# import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from config.config import config as cfg
from utils import losses
from utils.dataset import MXFaceDataset, DataLoaderX
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging
from backbones.softmaxnet import SoftmaxDropNet


def main(args):
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    init_seeds(cfg.seed + rank)

    if not os.path.exists(cfg.log_dir) and rank == 0:
        os.makedirs(cfg.log_dir)
    if not os.path.exists(cfg.model_dir) and rank == 0:
        os.makedirs(cfg.model_dir)
    else:
        time.sleep(2)
    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.log_dir)

    model = SoftmaxDropNet().to(local_rank)
    model = DistributedDataParallel(module=model, broadcast_buffers=False, device_ids=[local_rank])
    trainset = MXFaceDataset(root_dir=cfg.rec, local_rank=local_rank)

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, shuffle=True)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=trainset,
        batch_size=cfg.batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    writer = SummaryWriter(log_dir=cfg.log_dir)

    if cfg.global_step > 0:
        resume_model(model, cfg.global_step, local_rank, rank)

    opt = torch.optim.SGD(params=model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    # params=model.parameters(), lr=cfg.lr / 512 * cfg.batch_size * np.sqrt(world_size), momentum=0.9, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=cfg.lr_func)

    criterion = CrossEntropyLoss()

    start_epoch = 0
    epoch_step = int(len(trainset) / cfg.batch_size / world_size)
    total_step = epoch_step * cfg.num_epoch
    if cfg.stop_step > 0:
        total_step = cfg.global_step + cfg.stop_step
    if rank == 0:
        logging.info("Total Step is: %d" % total_step)
        logging.info(cfg)
        logging.info(f"{cfg.loss}")
        logging.info(f"{cfg.target_dir}")

    if cfg.global_step > 0:
        rem_steps = total_step - cfg.global_step
        cur_epoch = cfg.global_step // epoch_step
        start_epoch = cur_epoch
        scheduler.last_epoch = start_epoch
        opt.param_groups[0]["lr"] = scheduler.get_lr()[0]
        if rank == 0:
            logging.info(f"step per epoch:{epoch_step}")
            logging.info(f"resume from estimated epoch {cur_epoch}")
            logging.info(f"remaining steps {rem_steps}")
            logging.info(f"last learning rate: {opt.param_groups[0]['lr']}")

    callback_verification = CallBackVerification(
        cfg.eval_step, rank, local_rank, cfg.val_targets, cfg.rec, writer=writer
    )
    callback_logging = CallBackLogging(50, rank, total_step, cfg.global_step, cfg.batch_size, world_size, writer=writer)
    callback_checkpoint = CallBackModelCheckpoint(cfg.eval_step, rank, cfg.model_dir)

    loss = AverageMeter()

    global_step = cfg.global_step

    contrast_criterion = losses.ContraFace()

    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)

        if epoch >= cfg.shrink_epoch[0] - 1:
            model.module.backbone.dropout.p = cfg.dropout2

        for _, (img, label) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            loss_2, loss_contrast, loss_contrast2 = 0, 0, 0

            if (epoch >= cfg.shrink_epoch[0] - 1) and not cfg.single:
                x1, x2, theta1, theta2 = model(img, label, "double")
                loss_1 = criterion(theta1, label)
                loss_2 = criterion(theta2, label)
                loss_contrast = contrast_criterion(x1, x2, label)
                loss_contrast2 = contrast_criterion(x2, x1, label)
                loss_v = (
                    cfg.weight1 * loss_1
                    + cfg.weight2 * loss_2
                    + cfg.weightC * loss_contrast
                    + cfg.weightC2 * loss_contrast2
                )
            else:
                theta1 = model(img, label, "single")
                loss_1 = criterion(theta1, label)
                loss_v = loss_1

            opt.zero_grad()
            loss_v.backward()
            if rank == 0:
                writer.add_scalar("train_info/grad_B", model.module.header.kernel.grad.detach().mean(), global_step)

            clip_grad_norm_(model.module.backbone.parameters(), max_norm=5, norm_type=2)

            opt.step()

            loss.update(loss_v.item(), 1)

            dist.barrier()
            callback_logging(global_step, loss, loss_1, loss_2, loss_contrast, epoch)
            dist.barrier()
            callback_verification(global_step, model.module.backbone)
            callback_checkpoint(global_step, model)
            writer.add_scalar("train_info/lr", scheduler.get_last_lr()[0], global_step)
            writer.add_scalar("train_info/loss contrast", loss_contrast, global_step)

            if global_step // cfg.eval_step >= total_step // cfg.eval_step:
                exit(0)
            if cfg.stop_step > 0 and global_step - cfg.global_step >= cfg.stop_step:
                exit(0)
        scheduler.step()

    dist.destroy_process_group()
    logging.info(f"rank {rank} finished.")
    exit(0)


def resume_model(model, step, local_rank, rank, model_pth=None):
    try:
        if model_pth is None:
            model_pth = os.path.join(cfg.resume_dir, str(step) + "model.pth")
        model.module.load_state_dict(torch.load(model_pth, map_location=torch.device(local_rank)))
        if rank == 0:
            logging.info("model resume loaded successfully!")
    except (FileNotFoundError, KeyError, IndexError, RuntimeError) as e:
        if rank == 0:
            logging.info("load model resume init, failed!")
            logging.error(e)
        exit(1)


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch margin penalty loss  training")
    parser.add_argument(
        "--local_rank",
        type=int,
        default=os.getenv("LOCAL_RANK", -1),
        help="local_rank",
    )
    parser.add_argument("--resume", type=int, default=0, help="resume training")
    args_ = parser.parse_args()
    main(args_)
