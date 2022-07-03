import os
import cv2
from config import config
import random
from torchvision.datasets.vision import VisionDataset
from glob import glob


class CustomVideoDataset(VisionDataset):
    def __init__(self, data_dir, transform=None, target_transform=None, frame_batch=8):
        
        self.data_dir = data_dir
        self.video_files = glob(os.path.join(data_dir, "*.mp4"))
        self.ann_files = [os.path.join(data_dir, x.split('/')[-1][:-4]+".txt") for x in self.video_files]
        
        self.transform = transform
        self.target_transform = target_transform
        self.frame_batch=frame_batch

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        
        video_path = self.video_files[idx]
        ann_path = self.ann_files[idx]

        ## Getting video frames
        video = cv2.VideoCapture(video_path)
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        # fps = int(video.get(cv2.CAP_PROP_FPS))
        frame_no = random.randint(0,frames - 12)
        video.set(1,frame_no)
        images = []        
        for _ in range(self.frame_batch):
            _, image = video.read()
            images.append(image)

        ## Getting frame labels
        with open(ann_path, "r") as file:
            all_labels = file.readlines()
        labels = all_labels[frame_no: frame_no + self.frame_batch]
            
        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            labels = self.target_transform(labels)
        return images, labels
