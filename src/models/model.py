import timm
import torch

from torch import nn
from config import config

from src.models.losses.losses import LOSS_DICT


device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)

def create_model(num_classes, model_name='mobilenetv3_large_100', pretrained=True):
    """
    Create ML Network.
    :param num_classes: Number of classes.
    :param model_name: deeplabv3_resnet101 or fcn_resnet101.
    :param pretrained: if true, use pretrained model
    :param finetuning: if true, all paramters are trainable
    :return model: ML Network.
    """

    if config.LOSS_FUNC not in LOSS_DICT.keys():
        raise Exception(f"\n** Loss Function: {config.LOSS_FUNC} not supported |\n** Supported loss functions are: {list(LOSS_DICT.keys())} |")

    model = timm.create_model(model_name, pretrained=True, num_classes=2)

    params_to_optimize = model.parameters()
    model.to(device)

    return model, params_to_optimize, LOSS_DICT.get(config.LOSS_FUNC)