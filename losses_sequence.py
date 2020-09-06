import tensorflow as tf

y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true, y_pred).numpy())

y_true = [[[0, 1, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1]]]
y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]]
cce = tf.keras.losses.CategoricalCrossentropy()
print(cce(y_true, y_pred).numpy())

y_true = [[0, 1, 0], [0, 0, 1]]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
cce = tf.keras.losses.MeanSquaredError()
print(cce(y_true, y_pred).numpy())

y_true = [[[0, 1, 0], [0, 0, 1]], [[0, 1, 0], [0, 0, 1]]]
y_pred = [[[0.05, 0.95, 0], [0.1, 0.8, 0.1]], [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]]
cce = tf.keras.losses.MeanSquaredError()
print(cce(y_true, y_pred).numpy())
