import torch
import torch.nn.functional as F

from tqdm import tqdm
from config import config


def evaluate(model, criterion, data_loader, device, num_classes, apply_nonlin=None):
    epoch_loss = list()
    model.eval()

    with torch.no_grad():
        for image, target in tqdm(data_loader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target, apply_nonlin=apply_nonlin)
            epoch_loss.append(loss.item())

            if config.MODEL_NAME in ['u2net']:
                output = F.log_softmax(output[0], dim=1)
            elif config.MODEL_NAME in ['dpn98', 'se_resnext101_32x4d']:
                output = output
            else:
                output = output['out']

    
    avg_loss = sum(epoch_loss)/len(epoch_loss) if len(epoch_loss) else 0.0

    return confmat, avg_loss

def handler():
    pass