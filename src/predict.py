import os
import cv2
import numpy as np
import json
import torch

from src.models.model import create_model
from config import config
from glob import glob
from tqdm import tqdm


model, params_to_optimize = create_model(
        num_classes=2,
        model_name=config.MODEL_NAME,
        pretrained=True)

f = open("/mnt/notebooks/2775883169583/ann.json")
ann_file = json.load(f)


vid_id = "20210402_002523B"

os.makedirs(f"/mnt/notebooks/2775883169583/outputs/{vid_id}", exist_ok = True)

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
org_ann = (50, 95)
fontScale = 0.6
color = (255, 0, 0)
color_alert = (0, 0, 255)
thickness = 2

device = torch.device('cuda')
model.eval()

with torch.no_grad():
    for img in tqdm(sorted(glob(f"/mnt/notebooks/2775883169583/images/{vid_id}/*.jpg"), key=lambda x: int(x.split('/')[-1][:-4]))[ann_file[vid_id]["positive"][0][0]-2000:ann_file[vid_id]["positive"][0][1]+2000]):
        image = cv2.imread(img)
        im_arr32 = image.astype(np.float32)
        im_tensor = torch.tensor(im_arr32)
        im_tensor = im_tensor.permute(2, 0, 1)

        inp = im_tensor.unsqueeze(0).to(device)
        output = model(inp)
        
        if np.argmax(output.cpu().numpy()) == 0:
            result = cv2.putText(image, 'pred: Not Stage 9', org, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            result = cv2.putText(image, 'pred: Stage 9', org, font, fontScale, color_alert, thickness, cv2.LINE_AA)
            
        if int(img.split('/')[-1][:-4]) not in range(ann_file[vid_id]["positive"][0][0], ann_file[vid_id]["positive"][0][1]):
            result = cv2.putText(result, 'label: Not Stage 9', org_ann, font, fontScale, color, thickness, cv2.LINE_AA)
        else:
            result = cv2.putText(result, 'label: Stage 9', org_ann, font, fontScale, color_alert, thickness, cv2.LINE_AA)
        cv2.imwrite(f"/mnt/notebooks/2775883169583/outputs/{vid_id}/{img.split('/')[-1]}", result)

img_array = []

for filename in tqdm(sorted(glob.glob(f"/mnt/notebooks/2775883169583/outputs/{vid_id}/*.jpg"), key=lambda x: int(x.split('/')[-1][:-4]))[ann_file[vid_id]["positive"][0][0]-2000:ann_file[vid_id]["positive"][0][1]+2000]):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

vidcap = cv2.VideoCapture(f"/mnt/notebooks/2775883169583/raw_videos_all/content/drive/MyDrive/vids/{vid_id}.mp4")
FPS = vidcap.get(cv2.CAP_PROP_FPS)
print("FPS ==> ", FPS)

#out = cv2.VideoWriter('/content/restore.webm', cv2.VideoWriter_fourcc('V','P','8','0'), FPS, size)
out = cv2.VideoWriter(f"/mnt/notebooks/2775883169583/results/{vid_id}.mp4", cv2.VideoWriter_fourcc(*'MP4V'), FPS, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()