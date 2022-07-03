import torch
import numpy as np

from torch import nn

from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, JaccardLoss
from src.models.losses.focal_loss import FocalLoss
from src.models.losses.dice_loss import FocalTversky_loss


def torch_vision_criterion(inputs, target, apply_nonlin=None):

    losses = dict()
    for name, x in inputs.items():
        if apply_nonlin:
            x=apply_nonlin(x)
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

def torch_vision_criterion_dice(inputs, target, apply_nonlin=None):

    losses = dict()
    dice_loss = DiceLoss(mode='multiclass', ignore_index=255)
    for name, x in inputs.items():
        if apply_nonlin:
            x=apply_nonlin(x)
        losses[name] = dice_loss(x, target)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

def torch_vision_criterion_lovasz(inputs, target, apply_nonlin=None):

    losses = dict()
    lovasz_loss = LovaszLoss(mode='multiclass', ignore_index=255)
    for name, x in inputs.items():
        if apply_nonlin:
            x=apply_nonlin(x)
        losses[name] = lovasz_loss(x, target)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

def torch_vision_criterion_jaccard(inputs, target, apply_nonlin=None):

    losses = dict()
    ignore_index =255

    if ignore_index:
        target[target==ignore_index] = 0

    dice_loss = JaccardLoss(mode='multiclass')
    for name, x in inputs.items():
        if apply_nonlin:
            x=apply_nonlin(x)
        losses[name] = dice_loss(x, target)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

def torch_vision_criterion_focal(inputs, target, apply_nonlin=None):

    losses = dict()
    focal_loss = FocalLoss(apply_nonlin=apply_nonlin, alpha=0.5, gamma=2, smooth=1e-5)
    for name, x in inputs.items():
        losses[name] = focal_loss(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

def torch_vision_criterion_focal_tversky(inputs, target, apply_nonlin=None):

    losses = dict()
    focal_tversky_loss = FocalTversky_loss({'apply_nonlin': apply_nonlin, 'batch_dice': True, 'smooth': 1e-5, 'do_bg': False})
    for name, x in inputs.items():
        losses[name] = focal_tversky_loss(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

def u2net_criterion(inputs, target, apply_nonlin):

    if apply_nonlin:
        d0, d1, d2, d3, d4, d5, d6 = apply_nonlin(inputs[0]), apply_nonlin(inputs[1]), apply_nonlin(inputs[2]), apply_nonlin(inputs[3]), apply_nonlin(inputs[4]), apply_nonlin(inputs[5]), apply_nonlin(inputs[6])
    else:
        d0, d1, d2, d3, d4, d5, d6 = inputs

    # weights = np.array([1, 1.5, 1.5], dtype=np.float32)
    # weights = torch.from_numpy(weights).to(device)
    # loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)
    loss_CE = nn.functional.cross_entropy

    loss0 = loss_CE(d0, target, ignore_index=255)
    loss1 = loss_CE(d1, target, ignore_index=255)
    loss2 = loss_CE(d2, target, ignore_index=255)
    loss3 = loss_CE(d3, target, ignore_index=255)
    loss4 = loss_CE(d4, target, ignore_index=255)
    loss5 = loss_CE(d5, target, ignore_index=255)
    loss6 = loss_CE(d6, target, ignore_index=255)
    del d1, d2, d3, d4, d5, d6

    total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return total_loss

def u2net_criterion_focal(inputs, target, apply_nonlin):

    d0, d1, d2, d3, d4, d5, d6 = inputs

    # weights = np.array([1, 1.5, 1.5], dtype=np.float32)
    # weights = torch.from_numpy(weights).to(device)
    # loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)
    focal_loss = FocalLoss(apply_nonlin=apply_nonlin, alpha=0.5, gamma=2, smooth=1e-5)

    loss0 = focal_loss(d0, target, ignore_index=255)
    loss1 = focal_loss(d1, target, ignore_index=255)
    loss2 = focal_loss(d2, target, ignore_index=255)
    loss3 = focal_loss(d3, target, ignore_index=255)
    loss4 = focal_loss(d4, target, ignore_index=255)
    loss5 = focal_loss(d5, target, ignore_index=255)
    loss6 = focal_loss(d6, target, ignore_index=255)
    del d1, d2, d3, d4, d5, d6

    total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return total_loss


def u2net_criterion_focal_tversky(inputs, target, apply_nonlin):

    d0, d1, d2, d3, d4, d5, d6 = inputs

    # weights = np.array([1, 1.5, 1.5], dtype=np.float32)
    # weights = torch.from_numpy(weights).to(device)
    # loss_CE = nn.CrossEntropyLoss(weight=weights).to(device)
    focal_tversky_loss = FocalTversky_loss({'apply_nonlin': apply_nonlin, 'batch_dice': True, 'smooth': 1e-5, 'do_bg': False})

    loss0 = focal_tversky_loss(d0, target, ignore_index=255)
    loss1 = focal_tversky_loss(d1, target, ignore_index=255)
    loss2 = focal_tversky_loss(d2, target, ignore_index=255)
    loss3 = focal_tversky_loss(d3, target, ignore_index=255)
    loss4 = focal_tversky_loss(d4, target, ignore_index=255)
    loss5 = focal_tversky_loss(d5, target, ignore_index=255)
    loss6 = focal_tversky_loss(d6, target, ignore_index=255)
    del d1, d2, d3, d4, d5, d6

    total_loss = loss0 * 1.5 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return total_loss


def qubvel_criterion(inputs, target, apply_nonlin=None):

    if apply_nonlin:
        inputs=apply_nonlin(inputs)
    
    loss = nn.functional.cross_entropy(inputs, target, ignore_index=255)

    return loss


def qubvel_criterion_dice(inputs, target, apply_nonlin=None):

    # dice_loss = DiceLoss(mode=)
    
    loss = nn.functional.cross_entropy(inputs, target, ignore_index=255)

    return loss


def qubvel_criterion_focal(inputs, target, apply_nonlin):

    focal_loss = FocalLoss(apply_nonlin=apply_nonlin, alpha=0.5, gamma=2, smooth=1e-5)

    loss = focal_loss(inputs, target, ignore_index=255)

    return loss


def qubvel_criterion_focal_tversky(inputs, target, apply_nonlin):

    focal_tversky_loss = FocalTversky_loss({'apply_nonlin': apply_nonlin, 'batch_dice': True, 'smooth': 1e-5, 'do_bg': False})

    loss = focal_tversky_loss(inputs, target, ignore_index=255)

    return loss


LOSS_DICT = {
    "deeplabv3_ce": torch_vision_criterion,
    "deeplabv3_dice": torch_vision_criterion_dice,
    "deeplabv3_lovasz": torch_vision_criterion_lovasz,
    "deeplabv3_jaccard": torch_vision_criterion_jaccard,
    "deeplabv3_focal": torch_vision_criterion_focal,
    "deeplabv3_focal_tversky": torch_vision_criterion_focal_tversky,
    "u2net_ce": u2net_criterion,
    "u2net_focal": u2net_criterion_focal,
    "u2net_focal_tversky": u2net_criterion_focal_tversky,
    "qubvel_ce": qubvel_criterion,
    "qubvel_focal": qubvel_criterion_focal
}