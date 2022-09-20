import datetime
import pytz
import numpy as np
import pandas as pd
import pickle
import lib.config as config
import tensorflow as tf
from pytz import timezone
from matplotlib import pyplot as plt

import matplotlib
matplotlib.use('qtagg')

def validate_date(date_text):
    try:
        datetime.datetime.strptime(date_text, '%d/%m/%Y %H:%M:%S')
    except ValueError:
        raise ValueError("Incorrect data format, should be YYYY-MM-DD")

def split_data_by_daylight_saving_time(dataset, summer_data = None, winter_data = None):
    # get daylight_saving_time start and end date
    tz = timezone("America/Asuncion")
    transition_times = tz._utc_transition_times[60:]
    transition_times_by_year = {}
    for start_time, stop_time in zip(transition_times[::2], transition_times[1::2]):
        year = start_time.year
        transition_times_by_year[year] = [pytz.utc.localize(start_time) - datetime.timedelta(hours=start_time.hour), pytz.utc.localize(stop_time) - datetime.timedelta(hours=stop_time.hour)]

    date = dataset.index
    summer = []
    for d in date:
        stop_dst, start_dst = transition_times_by_year[d.year]
        if (start_dst.timestamp() <= d.timestamp()) or (d.timestamp() <= stop_dst.timestamp()):
            summer.append(d)
        else:
            pass
    # TODO: checar porque o dataset de verão está retornando alguns valores nulas (NA)
    if summer_data:
        return dataset[dataset.index.isin(summer)].dropna()
    
    if winter_data:
        return dataset[~dataset.index.isin(summer)].dropna()

    return dataset[dataset.index.isin(summer)].dropna(), dataset[~dataset.index.isin(summer)].dropna() # summer, winter

def plot_results(model, x_test, y_test, timestamp_nn, filename):
    x_test = x_test[::config.PREDICT_POINTS]
    y_test = y_test[::config.PREDICT_POINTS]
    predict_results = model.predict(tf.convert_to_tensor(x_test))

    font = {
        "family": "Calibri",
        "weight": "normal",
        "size": 16
    }

    plt.rc('font',**font)
    timestamp = np.array(timestamp_nn[::config.PREDICT_POINTS]).flatten()

    fig = plt.figure(figsize=(18,6))
    y = np.array(y_test).flatten()
    y = invTransform(y, config._LABEL_LISTS[0], config._FEATURE_LIST)
    plt.plot(timestamp,y, color="#218ED3", linewidth=2)
    y = np.array(predict_results).flatten()
    y = invTransform(y, config._LABEL_LISTS[0], config._FEATURE_LIST)
    plt.plot(timestamp,y, color="#ff7f0e", linewidth=2)
    plt.xlabel("Amostras")
    plt.ylabel("Intercâmbio SIN-PY [MW]")
    plt.legend(["Valor Observado", "Valor Previsto"], frameon=False)
    plt.xticks(rotation = 45)
    plt.show()
    fig.savefig(filename)

def invTransform(data, colName, colNames):
    scaler = pickle.load(open(config.SCALER_PATH, 'rb')) 
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values