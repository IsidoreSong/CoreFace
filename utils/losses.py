import torch
from torch import nn
from torch.nn import CrossEntropyLoss

# from config import config


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)

    return output


class MLLoss(nn.Module):
    def __init__(self, s=64.0):
        super(MLLoss, self).__init__()
        self.s = s

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        cos_theta.mul_(self.s)
        return cos_theta


class CosFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret


class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, embbedings, label):
        embbedings = l2_norm(embbedings, axis=1)
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(label != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1], device=cos_theta.device)
        m_hot.scatter_(1, label[index, None], self.m)
        cos_theta.acos_()
        cos_theta[index] += m_hot
        cos_theta.cos_().mul_(self.s)
        return cos_theta


class ContraFace(nn.Module):
    def __init__(self):
        super(ContraFace, self).__init__()
        self.cn = CrossEntropyLoss()
        self.s = 64
        self.m = 0

    def forward(self, feature1, feature2, label):
        # normalize features from different channel
        feature1 = l2_norm(feature1, axis=1)
        feature2 = l2_norm(feature2, axis=1)
        self.label = torch.arange(0, feature1.shape[0], device=feature1.device)
        # calculate the similarity matrix
        cos_theta = torch.mm(feature1, feature2.T)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        # check the number of samples for each class in this batch
        items, count = label.unique(return_counts=True)
        # eliminate the similarity scores describing a pair from the same class
        for item in items[count > 1]:
            idx = torch.where(label == item)[0]
            N_same = idx.shape[0]
            item_index = torch.arange(N_same, device=feature1.device)
            idx_matrix = idx.repeat(N_same, 1)
            idx_matrix[item_index, item_index] = torch.cat([idx, idx[0].view(1)])[1:]
            cos_theta[idx.view(-1, 1), idx_matrix] = 0
        # calculate the adaptive margin
        clone_theta = cos_theta.clone().detach()
        pos_target = clone_theta[self.label, self.label].clone()
        clone_theta[self.label, self.label] = 0
        neg_target2 = clone_theta.sort(1, True)[0][:, 0]
        # the margin in this batch
        m = (pos_target - neg_target2).mean().item()
        # the EMA margin
        self.m = 0.99 * m + (1 - 0.99) * self.m
        # add the margin the similarity of the positive pairs
        m_hot = torch.zeros_like(cos_theta)
        m_hot.scatter_(1, self.label.view(-1, 1), self.m)
        cos_theta = cos_theta - m_hot
        # scaled softmax
        cos_theta *= self.s
        loss = self.cn(cos_theta, self.label)
        return loss
