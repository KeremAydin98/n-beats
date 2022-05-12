import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from models import DL_models, NBeatsBlock
from evaluation import evaluate_preds
from preprocess import Preprocess

gold_df = pd.read_csv("gold_price_data.csv",parse_dates=["Date"],index_col=["Date"])

gold_df.plot(figsize=(10,7))
plt.show()

### Configurations
WINDOW_SIZE = 7
HORIZON = 1

# Prices and timesteps
prices = gold_df["Value"].to_numpy()
timesteps = gold_df.index.to_numpy()

# Split data into train and test results to plot it
split_size = int(len(prices) * 0.8)
X_train, y_train = timesteps[:split_size], prices[:split_size]
X_test, y_test = timesteps[split_size:], prices[split_size:]

plt.figure(figsize=(10,10))
plt.plot(X_train, y_train, label="Train data")
plt.plot(X_test, y_test, label="Test data")
plt.legend()
plt.show()

preprocess = Preprocess()
# Split data into windows and targets
full_windows, full_labels = preprocess.get_full_windows(dataset=prices)

# Split data into train and test results
split_size = int(len(prices) * 0.8)
train_windows, train_labels = full_windows[:split_size],full_labels[:split_size]
test_windows, test_labels = full_windows[split_size:], full_labels[split_size:]

# Create dataset with tf.data.Dataset
x_train = tf.data.Dataset.from_tensor_slices(train_windows)
y_train = tf.data.Dataset.from_tensor_slices(train_labels)

train_dataset = tf.data.Dataset.zip((x_train, y_train))

x_test = tf.data.Dataset.from_tensor_slices(test_windows)
y_test = tf.data.Dataset.from_tensor_slices(test_labels)

test_dataset = tf.data.Dataset.zip((x_test, y_test))

train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# -----------------------------------
# Baseline model
dl_models = DL_models()
baseline_model = dl_models.baseline_model(horizon=HORIZON)

baseline_model.fit(train_dataset, epochs=100, validation_data=(test_dataset))

baseline_model.evaluate(test_dataset)

label_preds = baseline_model.predict(test_windows)
baseline_results = evaluate_preds(test_labels, label_preds)

# -----------------------------------
# Conv1D model

conv1d_model = dl_models.Conv1D_model(horizon=HORIZON, window_size=WINDOW_SIZE)

conv1d_model.fit(train_dataset, epochs=100, validation_data=(test_dataset))

conv1d_model.evaluate(test_dataset)

predictions = conv1d_model.predict(test_windows)
conv1d_model_results = evaluate_preds(test_labels, predictions)

# -----------------------------------
# LSTM model

lstm_model = dl_models.LSTM_model(horizon=HORIZON)

lstm_model.fit(train_dataset, epochs=100, validation_data=(test_dataset))

lstm_model.evaluate(test_dataset)

predictions = lstm_model.predict(test_windows)
lstm_results = evaluate_preds(test_labels, predictions)

# -----------------------------------
# NBeats model

# Add windowed columns
gold_df_nbeats = gold_df.copy()

for i in range(WINDOW_SIZE):

  gold_df_nbeats[f"Value+{i+1}"] = gold_df_nbeats["Value"].shift(periods=i+1)

windows = gold_df_nbeats.dropna().drop("Value",axis=1).astype(np.float32)
labels = gold_df_nbeats.dropna()["Value"]

split_size = int(0.8 * len(windows))

train_windows, train_labels = windows[:split_size], labels[:split_size]
test_windows, test_labels = windows[split_size:], labels[split_size:]

x_train = tf.data.Dataset.from_tensor_slices(train_windows)
y_train = tf.data.Dataset.from_tensor_slices(train_labels)

train_dataset = tf.data.Dataset.zip((x_train, y_train))

x_test = tf.data.Dataset.from_tensor_slices(test_windows)
y_test = tf.data.Dataset.from_tensor_slices(test_labels)

test_dataset = tf.data.Dataset.zip((x_test, y_test))

train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Hyperparameters
# Values from N-BEATS paper
N_EPOCHS = 5000
N_NEURONS = 512
N_LAYERS = 4
N_STACKS = 30

INPUT_SIZE = WINDOW_SIZE * HORIZON
THETA_SIZE = WINDOW_SIZE + HORIZON

# 1. Setup an instance of the N-beats block layer
n_beats_block_layer = NBeatsBlock(input_size=INPUT_SIZE,
                                  theta_size=THETA_SIZE,
                                  n_neurons=N_NEURONS,
                                  horizon=HORIZON,
                                  n_layers=N_LAYERS)

# 2. Create an input layer for the N-BEATS stack
stack_inputs = tf.keras.layers.Input(shape=INPUT_SIZE)

# 3. Make the initial backcast and forecasts for the model with the layer created in
residuals, forecasts = n_beats_block_layer(stack_inputs)

# 4. Use for loop to create stacks of block layers
for _ in range(N_STACKS-1):

  # 5. Use the "NBeatsBlock" class within for loop in (4) to create blocks which return backcasts and block-level forecasts
  backcasts, block_forecasts = n_beats_block_layer(residuals)

  # 6. Create the double residual stacking using subtract and add layers
  residuals = tf.keras.layers.subtract([residuals, backcasts])
  forecasts = tf.keras.layers.add([forecasts,block_forecasts])

# 7. Put the model inputs and outputs together using "tf.keras.Model()"
n_beats_model = tf.keras.models.Model(inputs=stack_inputs, outputs=forecasts)

# 8. Compile the model with MAE loss and Adam optimizer
n_beats_model.compile(loss=tf.keras.losses.mape,
                      optimizer=tf.keras.optimizers.Adam())

n_beats_model.fit(train_dataset, epochs=N_EPOCHS, validation_data=test_dataset,
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=200,restore_best_weights=True)])

n_beats_model.evaluate(test_dataset)

predictions = n_beats_model.predict(test_windows)
n_beats_results = evaluate_preds(test_labels, predictions)

# -----------------------------------
# Ensemble model

models = [baseline_model, conv1d_model, lstm_model, n_beats_model]
preds = dl_models.ensemble_preds(models, test_windows)

ensemble_results = evaluate_preds(test_labels, preds)

all_results = pd.DataFrame({"baseline_dense":baseline_results,
               "Conv1D":conv1d_model_results,
               "LSTM":lstm_results,
               "N-Beats":n_beats_results,
               "Ensemble":ensemble_results})
all_results = all_results.transpose()

all_results_without_mse_and_rmse = all_results.drop(["MAE","MSE","RMSE"],axis=1)
all_results_without_mse_and_rmse.plot(kind="bar")
plt.show()
