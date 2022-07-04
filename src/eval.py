import torch
import torch.nn.functional as F

from tqdm import tqdm
from config import config


def evaluate(model, criterion, data_loader, device, num_classes):
    epoch_loss = list()
    model.eval()

    with torch.no_grad():
        for image, target in tqdm(data_loader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            epoch_loss.append(loss.item())
            
            output = output

    
    avg_loss = sum(epoch_loss)/len(epoch_loss) if len(epoch_loss) else 0.0

    return avg_loss

def handler():
    pass