from easydict import EasyDict as edict
import os

config = edict()
config.embedding_size = 512  # embedding size of model

config.loss = "ArcFace"

config.dataset = "emore"  # training dataset
# config.dataset = "webface"  # training dataset


config.batch_size = 128  # batch size per GPU
config.lr = 0.1
config.momentum = 0.9
config.weight_decay = 5e-4
config.SE = False  # SEModule


config.model_dir = "/public/home/fwang/Project/ContraFace/outcome"  # train model output folder
config.log_dir = "/public/home/fwang/Project/Logs"


if config.dataset == "emore":
    config.network = "iresnet100"
    config.resume_dir = os.path.join(config.model_dir, "epoch_ArcFace")
    config.global_step = 90976  # 90976  # 159208  # 238812

if config.dataset == "webface":
    config.network = "iresnet50"
    resume_loss = config.loss
    config.resume_dir = os.path.join(config.model_dir, resume_loss)
    config.global_step = 20118  # 20118  # 22034  # 26824


config.target_dir = config.loss

config.seed = 8
config.weight1 = 0.5
config.weight2 = 0.5
config.weightC = 0.05
config.weightC2 = 0  # test double-way N protocol
config.target_dir += "_Emore_d46_5505-W"
config.global_step = 0
config.single = False  # one channel training only
config.dropout = 0.4
config.dropout2 = 0.6

config.model_dir = os.path.join(config.model_dir, config.target_dir)
config.log_dir = os.path.join(config.log_dir, config.network + "_" + config.dataset, config.target_dir)
config.stop_step = 0
# config.resume_dir = config.model_dir

# config.s = 64.0
# config.m = 0.50


if config.dataset == "emore":

    config.rec = "/public/home/fwang/dataset/faces_emore2"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch = 24
    config.shrink_epoch = [8, 14, 20]
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]
    config.eval_step = 11372

    def lr_step_func(epoch):
        return (
            ((epoch + 1) / (4 + 1)) ** 2
            if epoch < -1
            else 0.1 ** len([m for m in config.shrink_epoch if m - 1 <= epoch])
        )

    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "/public/home/fwang/dataset/faces_webface_112x112"
    config.num_classes = 10572
    config.num_image = 490623  # 501195
    config.num_epoch = 40  # [22, 30, 35]
    config.shrink_epoch = [22, 30]
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]
    config.eval_step = 958  # 33350

    def lr_step_func(epoch):
        return (
            ((epoch + 1) / (4 + 1)) ** 2
            if epoch < config.warmup_epoch
            else 0.1 ** len([m for m in config.shrink_epoch if m - 1 <= epoch])
        )

    config.lr_func = lr_step_func
