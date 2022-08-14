from torch import nn
from utils import losses
from config.config import config as cfg
from backbones.iresnet import iresnet100, iresnet50

import torch.nn.functional as F


class SoftmaxNet(nn.Module):
    def __init__(self, num_features=None, num_classes=None, loss=cfg.loss):
        super(SoftmaxNet, self).__init__()
        self.backbone = get_backbone(num_features=num_features)
        self.header = get_header(in_features=num_features, out_features=num_classes, loss=loss)

    def forward(self, img, label, *args):
        features = F.normalize(self.backbone(img))
        thetas = self.header(features, label, *args)
        return thetas


class SoftmaxDropNet(nn.Module):
    def __init__(self, num_features=None, num_classes=None, loss=cfg.loss):
        super(SoftmaxDropNet, self).__init__()
        self.backbone = get_backbone(num_features=num_features)
        self.header = get_header(in_features=num_features, out_features=num_classes, loss=loss)

    def forward(self, img, label, mode, *args):
        if mode == "double":
            x1, x2 = self.backbone.double_forward(img)
            feature1 = F.normalize(x1)
            theta1 = self.header(feature1, label, *args)
            feature2 = F.normalize(x2)
            theta2 = self.header(feature2, label, *args)
            return x1, x2, theta1, theta2
        elif mode == "single":
            x = self.backbone.forward(img)
            feature = F.normalize(x)
            theta = self.header(feature, label, *args)
            return theta


def get_backbone(num_features=None):
    # load model
    if num_features is None:
        num_features = cfg.embedding_size

    if cfg.network == "iresnet100":
        backbone = iresnet100(dropout=cfg.dropout, num_features=num_features, use_se=cfg.SE)
    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=cfg.dropout, num_features=num_features, use_se=cfg.SE)
    # elif cfg.network == "PyramidNet":
    #     backbone = PyramidNet(50, alpha=128 - 64, dropout=cfg.dropout, num_features=num_features)
    else:
        exit()
    return backbone


def get_header(in_features=None, out_features=None, loss=cfg.loss):
    # get header
    if in_features is None:
        in_features = cfg.embedding_size
    if out_features is None:
        out_features = cfg.num_classes
    header = getattr(losses, loss)(in_features=in_features, out_features=out_features)
    return header
