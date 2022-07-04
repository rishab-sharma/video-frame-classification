import torch
from tqdm import tqdm


def evaluate(model, criterion, data_loader, device, num_classes):
    epoch_loss = list()
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for image, target in tqdm(data_loader):
            image, target = image.to(device), target.to(device)
            output = model(image)
            loss = criterion(output, target)
            epoch_loss.append(loss.item())
            
            output = output

            predicted = torch.argmax(output, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    
    avg_loss = sum(epoch_loss)/len(epoch_loss) if len(epoch_loss) else 0.0
    avg_accuracy = 100 * (correct // total)

    return avg_loss, avg_accuracy

def handler():
    pass