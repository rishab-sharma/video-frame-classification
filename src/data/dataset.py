import os
import json
import random
from glob import glob

from torchvision.datasets.vision import VisionDataset
from torchvision.io import read_image



class CustomVideoDataset(VisionDataset):
    def __init__(self, data_dir, ann_file, transform=None, target_transform=None):
        
        self.data_dir = data_dir
        self.frame_dirs = glob(os.path.join(data_dir, "*"))        
   
        f = open(ann_file)
        self.ann_file = json.load(f)         
        
        self.transform = transform
        self.target_transform = target_transform        

    def __len__(self):
        return len(self.frame_dirs)

    def __getitem__(self, idx):
        
        frames_path = self.frame_dirs[idx]
        ann = self.ann_files[frames_path.split("/")[-1]]

        if idx%2 == 0:
            cat = "positive"
            img_id = random.randint(ann[cat][0][0], ann[cat][0][1])
            label = 1
        else:
            cat = "negative"
            if random.randint(0, 10)%2 ==0:
                img_id = random.randint(ann[cat][0][0], ann[cat][0][1])
            else:
                img_id = random.randint(ann[cat][1][0], ann[cat][1][1])
            label = 0
        
        img_path = os.path.join(frames_path, f"{img_id}.jpg")
        image = read_image(img_path)
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
