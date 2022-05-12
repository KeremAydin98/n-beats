import tensorflow as tf
import numpy as np

class DL_models:

    def baseline_model(self, horizon):

        # At first we will build a baseline dense model
        model_baseline = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(horizon)
        ])

        model_baseline.compile(loss=tf.keras.losses.mape,
                               optimizer=tf.keras.optimizers.Adam())

        return model_baseline

    def Conv1D_model(self, horizon,window_size):
        # We need an expanding layer to put our train data into a Conv1D model, since Conv1D model waits for 1d data with batches
        expanding_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))

        # Create a Conv1D model
        conv1d_model = tf.keras.Sequential([
            expanding_layer,
            tf.keras.layers.Conv1D(filters=128, kernel_size=window_size, padding="causal", activation="relu"),
            tf.keras.layers.GlobalMaxPooling1D(),
            tf.keras.layers.Dense(horizon)
        ])

        conv1d_model.compile(loss=tf.keras.losses.mape,
                             optimizer=tf.keras.optimizers.Adam())

        return conv1d_model

    def LSTM_model(self, horizon):

        expanding_layer = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))

        # Create an LSTM model
        lstm_model = tf.keras.Sequential([
            expanding_layer,
            tf.keras.layers.LSTM(128, activation="relu"),
            tf.keras.layers.Dense(horizon)
        ])

        lstm_model.compile(loss=tf.keras.losses.mape,
                           optimizer=tf.keras.optimizers.Adam())

        return lstm_model


    def ensemble_preds(self, models,test_dataset):

      ensemble_results = []

      for model in models:

        predictions = model.predict(test_dataset)

        ensemble_results.append(predictions)

      return np.median(ensemble_results, axis=0)


class NBeatsBlock(tf.keras.layers.Layer):

  def __init__(self, input_size:int, theta_size:int, n_neurons:int, horizon:int, n_layers:int, **kwargs):

    super().__init__(**kwargs)

    self.input_size = input_size
    self.theta_size = theta_size
    self.n_neurons = n_neurons
    self.horizon = horizon
    self.n_layers = n_layers

    self.hidden_layer =  [tf.keras.layers.Dense(n_neurons, activation="relu") for _ in range(n_layers)]

    self.theta_layer = tf.keras.layers.Dense(theta_size, activation="linear")

  def call(self, inputs):

    x = inputs

    for layer in self.hidden_layer:

      x = layer(x)

    theta = self.theta_layer(x)

    backcast, forecast = theta[:,:self.input_size], theta[:,-self.horizon:]

    return backcast, forecast
