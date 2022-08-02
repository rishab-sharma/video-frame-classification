import json

from config import config
from src.train import handler as train_handler
from src.eval import handler as eval_handler

def handler(context):

    print(f"\n\n*** RUN CONFIGURATION ***\n{json.dumps(config.__dict__, indent=4)}\n-----\n\n")
    
    if config.MODE == "train":
        train_handler(context)
    elif config.MODE == "eval":
        eval_handler(context)
    else:
        raise Exception("Invalid MODE provided")

if __name__ == "__main__":
    context = {}
    context['datasets'] = {
        'frame_dir': '/content/sample_data/images/',
        'ann_file': '/content/ann.json'
        }
    config.EPOCHS = 11
    config.BATCH_SIZE = 64
    handler(context)