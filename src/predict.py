import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from config import config
from PIL import Image


def predict_on_samples(model, device, pred_dataset, wandb, trainval_info, color_maps, epoch):

    # TODO: Works only when 1 validation set provided on first 12 samples - Make it general
    
    val_idx = trainval_info['val_dataset_ids'][0]
    color_map = np.array(color_maps[val_idx]["color_map"])
    
    model.eval()

    with torch.no_grad():
        idx = 0
        print("Predicting on 12 Images and logging Results")


        for src_img_tnf, target, src_img, tgt_img in pred_dataset:

            fig = plt.figure(figsize=(10,4))
            rows, columns = 1, 3
            
            image, target = src_img_tnf.unsqueeze(0).to(device), target.unsqueeze(0).to(device)
            output = model(image)

            if config.MODEL_NAME in ['u2net']:
                output = F.log_softmax(output[0], dim=1)
            elif config.MODEL_NAME in ['dpn98', 'se_resnext101_32x4d']:
                output = output
            else:
                output = output['out']

            om = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()

            fig.add_subplot(rows, columns, 1)
            plt.imshow(np.array(src_img.resize((300,300))))
            
            fig.add_subplot(rows, columns, 2)
            plt.imshow(np.array(Image.fromarray(color_map[om].astype(np.uint8)).resize((300,300))))
            
            fig.add_subplot(rows, columns, 3)
            plt.imshow(np.array(tgt_img.resize((300,300))))
            
            plt.savefig(os.path.join(config.TEMP_DIR, f'{idx}_result.png'))
            wandb.log({f'{idx}_result.png': wandb.Image(os.path.join(config.TEMP_DIR, f'{idx}_result.png'))}, step=epoch)
            plt.cla()

            if idx==11:
                break

            idx+=1
        
        plt.close(fig)

    return None

def handler():
    pass