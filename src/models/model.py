import timm
import torch



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

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    params_to_optimize = model.parameters()
    model.to(device)

    return model, params_to_optimize