import lib.config as config
import tempfile
import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

import tensorflow.python.keras as keras

class NeuralNetwork():
    def __init__(self, output_path = None):
        self._output_path = output_path if output_path else tempfile.mkdtemp()
        self.model = None
        self.history = None

    def _generate_model(self, shape):
        # input_shape = tf.keras.layers.Input(shape=shape)
        # # LSTM-layers
        # lstm1 = tf.keras.layers.LSTM(200, return_sequences=True)(input_shape)
        # lstm2 = tf.keras.layers.LSTM(200, return_sequences=False)(lstm1)
        # # CONV layers
        # conv1 = tf.keras.layers.Conv1D(64, activation='relu', kernel_size=(3))(input_shape)
        # max_pool1 = tf.keras.layers.MaxPool1D(2)(conv1)
        # conv2 = tf.keras.layers.Conv1D(64, activation='relu', kernel_size=(3))(max_pool1)
        # max_pool2 = tf.keras.layers.MaxPool1D(2)(conv2)
        # flatten_cnn = tf.keras.layers.Flatten()(max_pool2)
        # # merge layersW
        # merged = tf.keras.layers.concatenate([flatten_cnn, lstm2], axis=1)
        # merged = tf.keras.layers.Flatten()(merged)
        # # output layer
        # out = tf.keras.layers.Dense(200, activation='relu')(merged)
        # out_layer = tf.keras.layers.Dense(config.PREDICT_POINTS, kernel_initializer=tf.initializers.zeros())(out)
        # # model
        # model = tf.keras.Model(input_shape, out_layer)        
        input_shape = tf.keras.layers.Input(shape=shape)
        # LSTM-layers
        lstm1 = tf.keras.layers.LSTM(256, return_sequences=True)(input_shape)
        b4 = tf.keras.layers.BatchNormalization()(lstm1)
        lstm2 = tf.keras.layers.LSTM(256, return_sequences=False)(b4)
        b5 = tf.keras.layers.BatchNormalization()(lstm2)
        # CONV layers
        conv1 = tf.keras.layers.Conv1D(128, activation='relu', kernel_size=(6))(input_shape)
        b1 = tf.keras.layers.BatchNormalization()(conv1)
        max_pool1 = tf.keras.layers.MaxPool1D(2)(b1)
        conv2 = tf.keras.layers.Conv1D(128, activation='relu', kernel_size=(6))(max_pool1)
        b2 = tf.keras.layers.BatchNormalization()(conv2)
        max_pool2 = tf.keras.layers.MaxPool1D(2)(b2)
        flatten_cnn = tf.keras.layers.Flatten()(max_pool2)
        # merge layers
        merged = tf.keras.layers.concatenate([flatten_cnn, b5], axis=1)
        merged = tf.keras.layers.Flatten()(merged)
        # output layer
        out = tf.keras.layers.Dense(512, activation='relu')(merged)
        b3 = tf.keras.layers.BatchNormalization()(out)
        out_layer = tf.keras.layers.Dense(config.PREDICT_POINTS, kernel_initializer=tf.initializers.zeros())(b3)
        # model
        model = tf.keras.Model(input_shape, out_layer)
        return model

    def _get_params_and_metrics(self, args):
        param_metrics = {}
        param_metrics["metrics"] = {}
        param_metrics["params"] = {}
        # metrics
        history_dict = self.history.history
        param_metrics["metrics"]["traning loss"] = history_dict["loss"][-1]
        param_metrics["metrics"]["validation loss"] = history_dict["val_loss"][-1]
        m = self.model.evaluate(args["test_dataset"].x, args["test_dataset"].y)
        param_metrics["metrics"]["MSE"] = m[0]
        param_metrics["metrics"]["MAE"] = m[1]
        # params
        param_metrics["params"]["max_epocx"] = args["max_epocs"]
        param_metrics["params"]["batch_size"] = args["batch_size"]
        param_metrics["params"]["patience"] = args["patience"]
        
        return param_metrics        

    def train_model(self, args):
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=args["patience"],
                                                    mode='min')
        
        self.model = self._generate_model(shape=args["input_shape"])
        
        self.model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(learning_rate=args["learning_rate"]),
                    metrics=[tf.metrics.MeanAbsoluteError()])

        self.history = self.model.fit(args["train_dataset"].x, args["train_dataset"].y,
                    epochs=args["max_epocs"],
                    batch_size=args["batch_size"],
                    validation_data=(args["test_dataset"].x, args["test_dataset"].y),
                    callbacks=[early_stopping])
    
        return self._get_params_and_metrics(args)

    def save_model(self, save_dir = None):
        if save_dir is None:
            self.model.save(self._output_path)
            return self._output_path
        
        self.model.save(save_dir)
        return save_dir
