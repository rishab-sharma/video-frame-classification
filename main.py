import json

from config import config
from src.train import handler as train_handler
from src.eval import handler as eval_handler
from src.predict import handler as predict_handler

def handler(context):

    print(f"\n\n*** RUN CONFIGURATION ***\n{json.dumps(config.__dict__, indent=4)}\n-----\n\n")
    
    if config.MODE == "train":
        train_handler(context)
    elif config.MODE == "eval":
        eval_handler(context)
    elif config.MODE == "predict":
        predict_handler(context)
    else:
        raise Exception("Invalid MODE provided")

if __name__ == "__main__":
    context = {}
    context['datasets'] = {'part_val': '2676439455897'}
    handler(context)