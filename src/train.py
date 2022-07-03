import os
import torch
import wandb
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm

from config import config
from src.eval import evaluate
from src.predict import predict_on_samples
from src.models.model import create_model
from src.models.losses.dice_loss import softmax_helper
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
        pretrained=True,
        finetuning=False)
    
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=config.LEARNING_RATE, momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda x: (1 - x / (len(train_loader) * config.EPOCHS)) ** 0.9)
    
    wandb.watch(model)
    early_stopping = EarlyStopping(ckp_dir=config.ABEJA_TRAINING_RESULT_DIR, metric_name='val_iou', patience=config.EARLY_STOPPING_PATIENCE, verbose=True)
    
    if config.APPLY_NONLIN:
        apply_nonlin = softmax_helper # torch.nn.Softmax(dim=1)
    else:
        apply_nonlin = None

    for epoch in range(config.EPOCHS):
        model.to(device)
        
        train_confmat, average_epoch_train_loss = train_one_epoch(
                model, criterion, optimizer, train_loader, lr_scheduler, device, epoch, num_classes, apply_nonlin=apply_nonlin)
        eval_confmat, average_epoch_val_loss = evaluate(
            model, criterion, val_loader, device=device, num_classes=num_classes, apply_nonlin=apply_nonlin)

        average_epoch_train_acc, train_prec, train_recal, train_iu = train_confmat.compute()
        average_epoch_val_acc, val_prec, val_recal, val_iu = eval_confmat.compute()

        with open(os.path.join(config.ABEJA_TRAINING_RESULT_DIR, "logs.txt"), "a+") as file:
            file.write(f"""
                ******\n
                Training Loss: {average_epoch_train_loss}\n
                Validation Loss: {average_epoch_val_loss}\n
                Training Accuracy: {average_epoch_train_acc.item() * 100}\n
                Validation Accuracy: {average_epoch_val_acc.item() * 100}\n
                Training Precision: {['{:.1f}'.format(i) for i in (train_prec * 100).tolist()]}\n
                Validation Precision: {['{:.1f}'.format(i) for i in (val_prec * 100).tolist()]}\n
                Training Recall: {['{:.1f}'.format(i) for i in (train_recal * 100).tolist()]}\n
                Validation Recall: {['{:.1f}'.format(i) for i in (val_recal * 100).tolist()]}\n
                Training IoU: {['{:.1f}'.format(i) for i in (train_iu).tolist()]}\n
                Training Mean IoU: {train_iu.mean().item()}\n
                Validation IoU: {['{:.1f}'.format(i) for i in (val_iu).tolist()]}\n
                Validation Mean IoU: {val_iu.mean().item()}\n
            """)
        print(f"Epoch {epoch} Training Loss =>", average_epoch_train_loss)
        print(f"Epoch {epoch} Validation Loss =>", average_epoch_val_loss)
        print(f"Epoch {epoch} Training Accuracy =>", average_epoch_train_acc.item() * 100)
        print(f"Epoch {epoch} Validation Accuracy =>", average_epoch_val_acc.item() * 100)
        print(f"Epoch {epoch} Training Precision =>", ['{:.1f}'.format(i) for i in (train_prec * 100).tolist()])
        print(f"Epoch {epoch} Validation Precision =>", ['{:.1f}'.format(i) for i in (val_prec * 100).tolist()])
        print(f"Epoch {epoch} Training Mean Precision =>", (train_prec * 100).mean().item())
        print(f"Epoch {epoch} Validation Mean Precision =>", (val_prec * 100).mean().item())
        print(f"Epoch {epoch} Training Recall =>", ['{:.1f}'.format(i) for i in (train_recal * 100).tolist()])
        print(f"Epoch {epoch} Validation Recall =>", ['{:.1f}'.format(i) for i in (val_recal * 100).tolist()])
        print(f"Epoch {epoch} Training Mean Recall =>", (train_recal * 100).mean().item())
        print(f"Epoch {epoch} Validation Mean Recall =>", (val_recal * 100).mean().item())
        print(f"Epoch {epoch} Training IoU =>", ['{:.1f}'.format(i) for i in (train_iu).tolist()])
        print(f"Epoch {epoch} Training Mean IoU =>", train_iu.mean().item())
        print(f"Epoch {epoch} Validation IoU =>", ['{:.1f}'.format(i) for i in (val_iu).tolist()])
        print(f"Epoch {epoch} Validation Mean IoU =>", val_iu.mean().item())

        wandb.log({
            "Training Loss": average_epoch_train_loss,
            "Validation Loss": average_epoch_val_loss,
            "LR": lr_scheduler.get_last_lr()[0],
            "Training Accuracy": average_epoch_train_acc.item() * 100,
            "Validation Accuracy": average_epoch_val_acc.item() * 100,
            "Training Precision": wandb.Histogram((train_prec * 100).tolist()),
            "Validation Precision": wandb.Histogram((val_prec * 100).tolist()),
            "Training Mean Precision": (train_prec * 100).mean().item(),
            "Validation Mean Precision": (val_prec * 100).mean().item(),
            "Training Recall": wandb.Histogram((train_recal * 100).tolist()),
            "Validation Recall": wandb.Histogram((val_recal * 100).tolist()),
            "Training Mean Recall": (train_recal * 100).mean().item(),
            "Validation Mean Recall": (val_recal * 100).mean().item(),
            "Training IoU": wandb.Histogram((train_iu * 100).tolist()),
            "Validation IoU": wandb.Histogram((val_iu * 100).tolist()),
            "Training Mean IoU": train_iu.mean().item(),
            "Validation Mean IoU": val_iu.mean().item()
            })

        early_stopping(val_iu.mean().item(), model, optimizer, epoch)

        if early_stopping.early_stop:
            print('--------------')
            print("Early stopping")
            break
    # save final model
    torch.save(model.to('cpu').state_dict(), os.path.join(config.ABEJA_TRAINING_RESULT_DIR, f'best-model-{epoch}.pth'))

    