# Files
DATASET_WINTER = False
DATASET_SUMMER = True
#
_SCALER_NAME = f"standard_scaler_{'winter' if DATASET_WINTER else 'summer'}.pkl"
SCALER_PATH = f"scalers/{_SCALER_NAME}"
MODEL_PATH = f"models_{'winter' if DATASET_WINTER else 'summer'}"
MODEL_XGB_PATH = f"xgb_model_{'winter' if DATASET_WINTER else 'summer'}.json"
LOG_PATH = "logs"
# Features
_FEATURE_LIST = ["power", "temp", "wspd", "weekend", "weekday"]
_LABEL_LISTS = ["power"]
# 
TAG = ""
#
POINTS_IN_DAY = 24
WINDOW_WIDTH_DAYS = 7 
PREDICT_POINTS = POINTS_IN_DAY
INPUT_POINTS = POINTS_IN_DAY * WINDOW_WIDTH_DAYS
#
WEATHER_STATION_ID = "86218"
#
DATE_FORMAT = "%d/%m/%Y %H:%M:%S"
