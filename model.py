import tensorflow as tf
from tensorflow import keras

def create_model(input_shape, num_chars):
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(inputs)
    x = keras.layers.Bidirectional(keras.layers.LSTM(256, return_sequences=True))(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_chars + 1, activation='softmax')(x)  # +1 for blank token

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model

def ctc_loss(y_true, y_pred):
    input_length = tf.math.reduce_sum(tf.ones_like(y_true[:, :, 0]), axis=-1)
    label_length = tf.math.reduce_sum(tf.ones_like(y_pred[:, :, 0]), axis=-1)
    return keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
