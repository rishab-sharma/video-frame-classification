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
from src.data.abeja_utils.pytorchtools import EarlyStopping
from src.data.abeja_utils import reports


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, num_classes, apply_nonlin=None):

    epoch_loss = list()
    model.train()
    confmat = reports.ConfusionMatrix(num_classes)
    
    for image, target in tqdm(data_loader):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target, apply_nonlin=apply_nonlin)
        epoch_loss.append(loss.item())
        
        output = output
        
        confmat.update(target.flatten(), output.argmax(1).flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_scheduler.step()

    confmat.reduce_from_all_processes()
    avg_loss = sum(epoch_loss)/len(epoch_loss) if len(epoch_loss) else 0.0

    return confmat, avg_loss


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

    train_loader, val_loader, num_classes = get_generator(context)

    model, params_to_optimize, criterion = create_model(
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
    early_stopping = EarlyStopping(ckp_dir=config.ABEJA_TRAINING_RESULT_DIR, metric_name='val_iou', patience=config.EARLY_STOPPING_PATIENCE, verbose=True)

    for epoch in range(config.EPOCHS):
        model.to(device)
        
        average_epoch_train_loss = train_one_epoch(
            model, criterion, optimizer, train_loader, lr_scheduler, device, epoch, num_classes)
        average_epoch_val_loss = evaluate(
            model, criterion, val_loader, device=device, num_classes=num_classes)


        with open(os.path.join(config.ABEJA_TRAINING_RESULT_DIR, "logs.txt"), "a+") as file:
            file.write(f"""
                ******\n
                Training Loss: {average_epoch_train_loss}\n
                Validation Loss: {average_epoch_val_loss}\n                
            """)
        
        print(f"Epoch {epoch} Training Loss =>", average_epoch_train_loss)
        print(f"Epoch {epoch} Validation Loss =>", average_epoch_val_loss)
        
        wandb.log({
            "Training Loss": average_epoch_train_loss,
            "Validation Loss": average_epoch_val_loss,
            "LR": lr_scheduler.get_last_lr()[0],            
            })

        early_stopping(average_epoch_val_loss, model, optimizer, epoch)

        if early_stopping.early_stop:
            print('--------------')
            print("Early stopping")
            break
    # save final model
    torch.save(model.to('cpu').state_dict(), os.path.join(config.ABEJA_TRAINING_RESULT_DIR, f'best-model-{epoch}.pth'))

    