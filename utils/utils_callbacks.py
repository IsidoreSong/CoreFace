import logging
import os
import time
from typing import List
import torch.distributed as dist

# from tqdm.notebook import tqdm

import torch

from eval import verification
from utils.utils_logging import AverageMeter


class CallBackVerification(object):
    def __init__(self, frequent, rank, local_rank, val_targets, rec_prefix, image_size=(112, 112), writer=None):
        self.frequent: int = frequent
        self.rank: int = rank
        self.local_rank = local_rank
        self.highest_acc: float = 0.0
        self.highest_acc_list: List[float] = [0.0] * len(val_targets)
        self.val_targets = val_targets
        self.ver_list: List[object] = []
        self.ver_name_list: List[str] = []
        self.writer = writer
        # if self.rank == 0:
        self.init_dataset(val_targets=val_targets, data_dir=rec_prefix, image_size=image_size)

    def ver_test(self, backbone: torch.nn.Module, global_step: int):
        results = []
        for i in range(len(self.ver_list)):
            if i % dist.get_world_size() == self.rank:
                acc1, std1, acc2, std2, xnorm, embeddings_list = verification.test(
                    self.ver_name_list[i],
                    self.ver_list[i],
                    backbone,
                    batch_size=10,
                    nfolds=10,
                    local_rank=self.local_rank,
                )
                if acc2 > self.highest_acc_list[i]:
                    self.highest_acc_list[i] = acc2
                if self.writer is not None:
                    self.writer.add_scalar("verification/" + self.ver_name_list[i], acc2, global_step)
                basic_info = f"[{global_step}][{self.rank}][{self.ver_name_list[i]}]"
                logging.info(basic_info + f"XNorm: {xnorm:f}")
                logging.info(basic_info + f"Accuracy-Flip: {acc2:1.5f}+-{std2:1.5f}")
                logging.info(basic_info + f"Accuracy-Highest: {self.highest_acc_list[i]:1.5f}")
                results.append(acc2)

    def init_dataset(self, val_targets, data_dir, image_size):
        logging.info(f"rank:{self.rank} - local_rank:{self.local_rank}")
        for name in val_targets:
            path = os.path.join(data_dir, name + ".bin")
            if os.path.exists(path):
                data_set = verification.load_bin(path, image_size, self.rank)
                self.ver_list.append(data_set)
                self.ver_name_list.append(name)

    def __call__(self, num_update, backbone: torch.nn.Module):
        if num_update > 0 and num_update % self.frequent == 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(
        self, frequent, rank, total_step, start_step, batch_size, world_size, writer=None, resume=0, rem_steps=None
    ):
        self.frequent: int = frequent
        self.rank: int = rank
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step = start_step
        self.batch_size: int = batch_size
        self.world_size: int = world_size
        self.writer = writer
        self.resume = resume
        self.rem_steps = rem_steps

        self.init = False
        self.tic = 0

    def __call__(self, global_step, loss: AverageMeter, loss1, loss2, loss_contrast, epoch: int):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float("inf")

                time_now = (time.time() - self.time_start) / 3600
                time_total = time_now / ((global_step - self.start_step + 1) / (self.total_step - self.start_step))
                time_for_end = time_total - time_now
                if self.writer is not None:
                    # self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar("train_info/speed", speed_total, global_step)
                    self.writer.add_scalar("train_info/loss", loss.avg, global_step)
                msg = (
                    f"[{global_step:d}]"
                    + f"[E{epoch:2d}]"
                    + f"[{speed_total:4.0f}/s]"
                    + f" -Loss {loss.avg:.2f}"
                    + f" -Loss1 {loss1:.2f}"
                    + f" -Loss2 {loss2:.2f}"
                    + f" -LossC {loss_contrast:.2f}"
                    + f" -Time: {time_for_end:2.1f} H"
                )

                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()


class CallBackModelCheckpoint(object):
    def __init__(self, frequent, rank, output="./"):
        self.rank: int = rank
        self.output: str = output
        # self.eval_step = eval_step
        self.frequent = frequent

    def __call__(self, global_step, model):
        if self.rank == 0 and global_step % self.frequent == 0:
            torch.save(model.module.state_dict(), os.path.join(self.output, str(global_step) + "model.pth"))
