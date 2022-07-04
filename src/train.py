import os
import torch
import wandb
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from config import config
from src.eval import evaluate
from src.models.model import create_model
from src.data.dataloader import get_generator


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device):

    epoch_loss = list()
    model.train()

    correct = 0
    total = 0
    
    for image, target in tqdm(data_loader):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)
        epoch_loss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

        predicted = torch.argmax(output.cpu(), 1)
        correct += (predicted == target).sum().item()
        total += target.size(0)

    avg_loss = sum(epoch_loss)/len(epoch_loss) if len(epoch_loss) else 0.0
    avg_accuracy = 100 * (correct // total)

    return avg_loss, avg_accuracy


def handler(context):

    print("Initiating Wandb")
    wandb.init(project=config.PROJECT_DOMAIN, entity="kraken")
    wandb.config = {
        "learning_rate": config.LEARNING_RATE,
        "epochs": config.EPOCHS,
        "batch_size": config.BATCH_SIZE
        }

    print(f"Working in Project Directory - {config.PROJECT_DIR}")
    
    device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    
    print('Device Name: {}'.format(device_name))

    train_loader, val_loader = get_generator(context)

    num_classes = 2

    model, params_to_optimize = create_model(
        num_classes=num_classes,
        model_name=config.MODEL_NAME,
        pretrained=True)
    
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(train_loader) * config.EPOCHS)) ** 0.9)
    
    criterion = nn.CrossEntropyLoss()
    
    wandb.watch(model)

    for epoch in range(config.EPOCHS):
        model.to(device)
        
        average_epoch_train_loss, avg_train_accuracy = train_one_epoch(
            model, criterion, optimizer, train_loader, lr_scheduler, device)
        average_epoch_val_loss, avg_val_accuracy = evaluate(
            model, criterion, val_loader, device=device, num_classes=num_classes)


        with open(os.path.join(config.ABEJA_TRAINING_RESULT_DIR, "logs.txt"), "a+") as file:
            file.write(f"""
                ******\n
                Training Loss: {average_epoch_train_loss}\n
                Validation Loss: {average_epoch_val_loss}\n
                Training Accuracy: {avg_train_accuracy}\n
                Validation Accuracy: {avg_val_accuracy}\n
            """)
        
        print(f"Epoch {epoch} Training Loss =>", average_epoch_train_loss)
        print(f"Epoch {epoch} Validation Loss =>", average_epoch_val_loss)
        print(f"Epoch {epoch} Training Accuracy =>", avg_train_accuracy)
        print(f"Epoch {epoch} Validation Accuracy =>", avg_val_accuracy)
        
        wandb.log({
            "Training Loss": average_epoch_train_loss,
            "Validation Loss": average_epoch_val_loss,
            "Training Accuracy": avg_train_accuracy,
            "Validation Accuracy": avg_val_accuracy,
            "LR": lr_scheduler.get_last_lr()[0],            
            })
        
    # save final model
    torch.save(model.to('cpu').state_dict(), os.path.join(config.ABEJA_TRAINING_RESULT_DIR, f'best-model-{epoch}.pth'))

    