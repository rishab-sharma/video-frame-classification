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
        'frame_dir': '/mnt/notebooks/2775883169583/images/',
        'ann_file': '/mnt/notebooks/2775883169583/ann.json'
        }
    handler(context)