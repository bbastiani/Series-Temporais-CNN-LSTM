READ_DATASET_FROM_PICKLE = True
DATASET_WINTER = True
DATASET_SUMMER = False
#
_DATASET_NAME = "dataset_offset.pkl"
DATASET_PATH = f"datasets/{_DATASET_NAME}"
_SCALER_NAME = f"standard_scaler_{'winter' if DATASET_WINTER else 'summer'}.pkl"
SCALER_PATH = f"scalers/{_SCALER_NAME}"
MODEL_PATH = f"models_{'winter' if DATASET_WINTER else 'summer'}"
LOG_PATH = "logs"
IMAGES_PATH = "images"
# Features
_FEATURE_LIST = ["power", "temp", "wspd", "weekday", "weekend"]
_LABEL_LISTS = ["power"]

#
POINTS_IN_DAY = 24
WINDOW_WIDTH_DAYS = 7
PREDICT_POINTS = POINTS_IN_DAY
INPUT_POINTS = POINTS_IN_DAY * WINDOW_WIDTH_DAYS
# 
TAG = ""
START_DATE = "01/01/2019 00:00:00"
END_DATE = "01/01/2022 00:00:00"
#
SCALER = "standard"
# Hiperparams
RESAMPLE_FREQ = "1H" # 30T
TRAIN_TEST_SPLIT = 0.05
MAX_EPOCS = 100
PATIENCE = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 128
# model signature
from mlflow.types.schema import Schema, ColSpec
input_schema = Schema([ColSpec("double", feature) for feature in _FEATURE_LIST])
output_schema = Schema([ColSpec("double", label) for label in _LABEL_LISTS])
