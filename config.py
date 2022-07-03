import os

from environs import Env
from pydantic import BaseModel
from pydantic.main import ModelMetaclass

from abejacli import config as abeja_config


class EnvironmentVariable(BaseModel):
    MODE: str = str(os.environ.get('MODE', 'train'))
    
    # TRAINING CONFIGS
    DATA: str = str(os.environ.get('DATA', 'abeja'))
    MODEL_NAME: str = str(os.environ.get('MODEL_NAME', 'deeplabv3_resnet101')) # deeplabv3_resnet101 u2net se_resnext101_32x4d dpn98
    LOSS_FUNC: str = str(os.environ.get('LOSS_FUNC', 'deeplabv3_ce')) # u2net_ce deeplabv3_ce qubvel_ce

    BATCH_SIZE: int = int(os.environ.get('BATCH_SIZE', '6'))
    EPOCHS: int = int(os.environ.get('EPOCHS', '100'))
    VAL_SIZE_RATIO: float = float(os.environ.get('VAL_SIZE_RATIO', '0.2'))
    NUM_TRAIN: int = int(os.environ.get('NUM_TRAIN', '2000'))
    NUM_VAL: int = int(os.environ.get('NUM_VAL', '200'))

    LEARNING_RATE: float = float(os.environ.get('LEARNING_RATE', '0.01'))
    MOMENTUM: float = float(os.environ.get('MOMENTUM','0.9'))
    WEIGHT_DECAY: float = float(os.environ.get('WEIGHT_DECAY','1e-4'))
    EARLY_STOPPING_PATIENCE: int = int(os.environ.get('EARLY_STOPPING_PATIENCE', '7'))
    NUM_DATA_LOAD_THREAD: int = int(os.environ.get('NUM_DATA_LOAD_THREAD', '1'))
    APPLY_NONLIN: bool = bool(os.environ.get('APPLY_NONLIN', 'false').lower() == 'true')
    DATA_TRANSFORM: str = str(os.environ.get('DATA_TRANSFORM', 'v1'))


    # EVALUATION CONFIGS
    EVAL_DATA: str = str(os.environ.get('EVAL_DATA', 'abeja'))


    # PREDICTION CONFIGS
    SRC_IMAGE: str = "./temp/test.jpg"



class LocalConfig(BaseModel):
    pass


class ProjectConfig:
    PROJECT_DOMAIN: str = "kkc-roadmarking-segmentation"
    PROJECT_DIR: str = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "opt")
    TEMP_DIR: str = os.path.join(PROJECT_DIR, "temp")

    ORG_ID: str = "2103353894345"
    CREDENTIAL: dict = {
        'user_id': abeja_config.config.user,
        'personal_access_token': abeja_config.config.token
    }
    NEW_CAT_JSON: str = os.path.join(PROJECT_DIR, "src/data/abeja_utils", "categories_new.json")
    OLD_CAT_JSON: str = os.path.join(PROJECT_DIR, "src/data/abeja_utils", "categories_old.json")
    ABEJA_DATASET_VERSION: bool = bool(os.environ.get('DATASET_VERSION', 'True').lower() == 'true')
    ABEJA_TRAINING_RESULT_DIR: str = str(os.environ.get('ABEJA_TRAINING_RESULT_DIR', '.'))
    

class Config(ProjectConfig, LocalConfig, EnvironmentVariable):
    def __init__(self, *args, **kwargs):
        env = Env(eager=False)
        # env.read_env(os.path.join(self.PROJECT_DIR, ".envs/.pre.env"))

        os.environ['WANDB_API_KEY']="aea5370244b8ebb66c0893809bd04594a3fc0957"
        os.environ['WANDB_NAME'] = str(os.environ.get('EXP', 'random-run'))
        os.environ['WANDB_NOTES']=str(os.environ.get('EXP_NOTES', 'Random experiment'))

        get_env_var = lambda var_type, var_value: getattr(env, var_type)(var_value)
        env_var = {}
        for var_value, var_type in EnvironmentVariable.__annotations__.items():
            # pydantic object or list of objects
            if type(var_type) == ModelMetaclass:
                data_type = "json"
            # primitive datatypes
            else:
                data_type: str = var_type.__name__
            env_var_value = get_env_var(data_type, var_value)
            if env_var_value:
                env_var[var_value] = env_var_value
        super().__init__(**env_var)


config = Config()

if not os.path.exists(config.TEMP_DIR):
    os.makedirs(config.TEMP_DIR)
