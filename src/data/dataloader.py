from config import config
from src.data.dataset import CustomVideoDataset
from torch.utils.data import DataLoader


def get_generator(context):

    data_dir = context["datasets"]["frame_dir"]
    ann_file = context["datasets"]["ann_file"]

    train_dataset = CustomVideoDataset(data_dir=data_dir, ann_file=ann_file)
    val_dataset = CustomVideoDataset(data_dir=data_dir, ann_file=ann_file, val=True)
#     print(f"Train Dataset Files: {train_dataset.frame_dirs}")
#     print(f"Val Dataset Files: {val_dataset.frame_dirs}")

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, drop_last=True, shuffle=True)
    test_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, drop_last=True, shuffle=True)

    return train_dataloader, test_dataloader