import torch.nn.functional as F


def dice_loss(input, target, smooth=1., axis=None):
    intersection = (input * target).sum(axis)
    union = input.sum(axis) + target.sum(axis)
    dice = (2. * intersection + smooth) / (union + smooth)

    loss = 1 - dice

    return loss


def sigmoid_cross_entropy(input, target):
    loss = F.binary_cross_entropy_with_logits(input=input, target=target, reduction='none')

    return loss


def softmax_cross_entropy(input, target, axis=1, keepdim=False):
    log_prob = input.log_softmax(axis)
    loss = -(target * log_prob).sum(axis, keepdim=keepdim)

    return loss


def sigmoid_focal_loss(input, target, gamma=2.):
    prob = input.sigmoid()
    prob_true = prob * target + (1 - prob) * (1 - target)
    weight = (1 - prob_true)**gamma

    loss = sigmoid_cross_entropy(input=input, target=target)
    loss = weight * loss

    return loss


def softmax_focal_loss(input, target, gamma=2., axis=1, keepdim=False):
    prob = input.softmax(axis)
    weight = (1 - prob)**gamma

    log_prob = input.log_softmax(axis)
    loss = -(weight * target * log_prob).sum(axis, keepdim=keepdim)

    return loss
