import os
import json
import random
from glob import glob

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_transform = transforms.Compose([transforms.Resize(255),
#                                 transforms.RandomResizedCrop(224),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(),                                
                                # transforms.ColorJitter(),
                                transforms.ToTensor(),
#                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
test_transform = transforms.Compose([
                                transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
#                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class CustomVideoDataset(Dataset):
    def __init__(self, data_dir, ann_file, target_transform=None, val=False):
        
        self.data_dir = data_dir
        dir_len = len(glob(os.path.join(data_dir, "*")))

        if not val:
            transform=train_transform
            self.frame_dirs = glob(os.path.join(data_dir, "*"))[:int(dir_len*0.8)]
            print(f'Train files: {[v.split("/")[-1] for v in self.frame_dirs]}')
        else:
            transform=test_transform
            self.frame_dirs = glob(os.path.join(data_dir, "*"))[int(dir_len*0.8):]
            print(f'Val files: {[v.split("/")[-1] for v in self.frame_dirs]}')
   
        f = open(ann_file)
        self.ann_file = json.load(f)         
        
        self.transform = transform
        self.target_transform = target_transform        

    def __len__(self):
        return len(self.frame_dirs)*100
    
    def get_random_image_id(self, idx, ann):
        if idx%3 == 0:
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
        return img_id, label

    def __getitem__(self, idx):
        frames_path = random.choice(self.frame_dirs)
        ann = self.ann_file[frames_path.split("/")[-1]]

        image = None
        while image is None:
            img_id, label = self.get_random_image_id(idx, ann)
            try:
                img_path = os.path.join(frames_path, f"{img_id}.jpg")
                image = Image.open(img_path)
            except:
                print(f"Image {img_path} not found for label {label}")
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
