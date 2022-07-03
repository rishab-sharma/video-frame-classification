import random
import torch

from config import config
from src.data.dataset import CustomVideoDataset


def worker_init_fn(worker_id):
    random.seed(worker_id)


def get_generator(context):

    data_dir = context["datasets"]["frame_dir"]
    ann_file = context["datasets"]["ann_file"]

    train_dataset = CustomVideoDataset(data_dir=data_dir, ann_file=ann_file)
    val_dataset = CustomVideoDataset(data_dir=data_dir, ann_file=ann_file)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_DATA_LOAD_THREAD,
        worker_init_fn=worker_init_fn,    
        drop_last=True,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_DATA_LOAD_THREAD,
        worker_init_fn=worker_init_fn,        
        drop_last=True,
        shuffle=True)

    return train_loader, val_loader