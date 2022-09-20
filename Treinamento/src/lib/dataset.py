#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import tempfile
import os, pickle
import lib.config as config
from datetime import timedelta
from lib.utils import validate_date, split_data_by_daylight_saving_time
from pathlib import Path
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, x, y, timestamp):
        self.x = x
        self.y = y
        self.timestamp = timestamp
    
class Dataset():
    def __init__(self, dataset_path = None, scaler_path = None):
        self._temp_path = tempfile.mkdtemp()
        self._dataset_path = dataset_path if dataset_path else self._temp_path
        self._scaler_path = scaler_path if dataset_path else self._temp_path

        self.raw_dataset = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def _read_data_pickle(self, filename):
        df = pd.read_pickle(filename)
        df = df.resample(config.RESAMPLE_FREQ).mean()
        return df.interpolate('linear')

    def _download_data(self, pi_tag: str, start_date: str, end_date: str):
        import PIconnect as PI
        # validate data format
        validate_date(start_date)
        validate_date(end_date)
        # 
        df = pd.DataFrame()
        time_range = pd.date_range(start=start_date, end=end_date, freq='M')
        with PI.PIServer() as server:
            for _start, _end in zip(time_range[:-1], time_range[1:]):
                points = server.search(pi_tag)
                data = points[0].recorded_values(str(_start), str(_end))
                df = df.append(pd.DataFrame(data))
        # resample data
        df = df.resample('H').mean()
        df = df.interpolate('linear')
        filepath = config.DATASET_PATH
        df.columns = ["power"]
        df.index = df.index - timedelta(hours=3) 
        df.to_csv(filepath)
        df.dropna(inplace=True)
        self.raw_dataset = df.copy()
        return filepath

    def _filter_data(self, dataset, feature_list):
        return dataset.drop(dataset.columns.difference(feature_list), 1)

    def _scale_data(self, dataset, scaler = None):
        filepath = config.SCALER_PATH
        if Path(filepath).is_file():
            scaler = pickle.load(open(filepath, 'rb'))
        else:
            scaler.fit(dataset)
            pickle.dump(scaler, open(filepath, 'wb'))
        
        normalized_dataset = scaler.transform(dataset)
        df = pd.DataFrame(normalized_dataset, columns = dataset.columns, index = dataset.index)
        return df

    def _moving_window(self, dataset, input_points, predict_points):
        x = []
        y = []
        timestamp = []
        for i in range(len(dataset)):
            sliding_window_idx = i+input_points
            predict_idx = i + input_points + predict_points
            if predict_idx == (len(dataset)-1):
                break

            # x.append(dataset.to_numpy()[i:sliding_window_idx].flatten())
            x.append(dataset.to_numpy()[i:sliding_window_idx])
            y.append(dataset[config._LABEL_LISTS].to_numpy()[sliding_window_idx : predict_idx].flatten())
            timestamp.append(dataset[sliding_window_idx : predict_idx].index)

        x = np.array(x)
        y = np.array(y)

        # return x.reshape((x.shape[0], x.shape[1], 1)), y.reshape((y.shape[0], y.shape[1], 1)), timestamp
        return x, y, timestamp

    def _split_data(self, dataset, summer_data, winter_data):
        return split_data_by_daylight_saving_time(dataset, summer_data=summer_data, winter_data=winter_data)

    def _load_dataset(self, read_pickle_data = False):
        if not read_pickle_data:
            self.raw_dataset = pd.read_csv(config.DATASET_PATH)
        else:    
            self.raw_dataset = self._read_data_pickle(config.DATASET_PATH)

    def preprocessing_data(self, args):
        dataset = self._filter_data(self.raw_dataset, config._FEATURE_LIST)
        dataset = self._split_data(dataset, args["summer_data"], args["winter_data"])
        dataset = self._scale_data(dataset, args["scaler"])
        x, y, timestamp = self._moving_window(dataset, args["input_points"], args["predict_points"])
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = args["train_test_split"], shuffle = False) 

        train_dataset = Data(self.x_train, self.y_train, timestamp=timestamp[:self.x_train.shape[0]])
        test_dataset = Data(self.x_test, self.y_test, timestamp=timestamp[self.x_train.shape[0]:])
        return train_dataset, test_dataset

    def download_data(self, read_pickle_data = False):
        if not read_pickle_data:
            self._download_data(config.TAG, config.START_DATE, config.END_DATE)
        else:
            self._load_dataset(read_pickle_data)
        