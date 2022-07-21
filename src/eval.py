import torch
from tqdm import tqdm
from sklearn.metrics import classification_report


def evaluate(model, criterion, data_loader, device, num_classes):
    classes = ["Not Stage 9", "Stage 9"]
    epoch_loss = list()
    model.eval()

    correct = 0
    total = 0
    
    y_test, y_pred = [], []

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
            
            y_test.extend([int(v) for v in target])
            y_pred.extend([int(v) for v in predicted])

    
    avg_loss = sum(epoch_loss)/len(epoch_loss) if len(epoch_loss) else 0.0
    avg_accuracy = 100 * (correct // total)
    print(classification_report(y_test, y_pred, target_names=classes))

    return avg_loss, avg_accuracy

def handler():
    pass