import torch.nn.functional as F


def dice_loss(input, target, dim=None, smooth=1.0):
    if dim is None:
        dim = ()

    i = (input * target).sum(dim)
    c = (input + target).sum(dim)

    dice = 2.0 * (i + smooth) / (c + smooth)
    loss = 1 - dice
    return loss


def iou_loss(input, target, dim=None, smooth=1.0):
    if dim is None:
        dim = ()

    i = (input * target).sum(dim)
    c = (input + target).sum(dim)
    u = c - i

    iou = (i + smooth) / (u + smooth)
    loss = 1 - iou
    return loss


def sigmoid_cross_entropy(input, target):
    loss = F.binary_cross_entropy_with_logits(input=input, target=target, reduction="none")

    return loss


def softmax_cross_entropy(input, target, dim=-1, keepdim=False):
    log_prob = input.log_softmax(dim)
    loss = -(target * log_prob).sum(dim, keepdim=keepdim)

    return loss


def sigmoid_focal_loss(input, target, gamma=2.0):
    prob = input.sigmoid()
    prob_true = prob * target + (1 - prob) * (1 - target)
    weight = (1 - prob_true) ** gamma

    loss = sigmoid_cross_entropy(input=input, target=target)
    loss = weight * loss

    return loss


def softmax_focal_loss(input, target, gamma=2.0, dim=1, keepdim=False):
    prob = input.softmax(dim)
    weight = (1 - prob) ** gamma

    log_prob = input.log_softmax(dim)
    loss = -(weight * target * log_prob).sum(dim, keepdim=keepdim)

    return loss
