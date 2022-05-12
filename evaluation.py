import tensorflow as tf

def evaluate_preds(y_true, y_pred):

  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  mae = tf.reduce_mean(tf.abs(y_true - y_pred))
  mse = tf.reduce_mean(tf.abs(y_true**2 - y_pred**2))

  rmse = tf.sqrt(mse)

  mape = tf.reduce_mean(tf.abs(y_true - y_pred) / y_true) * 100
  mae_naive = tf.reduce_mean(tf.abs(y_true[1:]-y_true[:-1]))

  mase = mae / mae_naive

  evaluation_metrics = {"MAE":mae.numpy(),
                       "MSE":mse.numpy(),
                       "RMSE":rmse.numpy(),
                       "MAPE":mape.numpy(),
                       "MASE":mase.numpy()}

  return evaluation_metrics