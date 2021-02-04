import torch

from all_the_tools.torch.losses import dice_loss, iou_loss


def test_iou_loss():
    input = torch.empty((10, 20, 30)).uniform_()
    target = torch.distributions.Bernoulli(probs=0.5).sample((10, 20, 30))
    loss = iou_loss(input, target)
    assert loss.size() == ()
    loss = iou_loss(input, target, dim=1)
    assert loss.size() == (10, 30)
    loss = iou_loss(input, target, dim=(1, 2))
    assert loss.size() == (10,)


def test_dice_loss():
    input = torch.empty((10, 20, 30)).uniform_()
    target = torch.distributions.Bernoulli(probs=0.5).sample((10, 20, 30))
    loss = dice_loss(input, target)
    assert loss.size() == ()
    loss = dice_loss(input, target, dim=1)
    assert loss.size() == (10, 30)
    loss = dice_loss(input, target, dim=(1, 2))
    assert loss.size() == (10,)
