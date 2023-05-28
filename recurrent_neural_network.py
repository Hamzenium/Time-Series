import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from absl import logging
logging.set_verbosity(logging.ERROR)

dataset = []

def create_uncompiled_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis= -1), input_shape = [None]),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences= True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 10.0),
    ])
    return model


    
def adjust_learning():
    model = create_uncompiled_model(),
    learning_rate_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10**(epoch / 20)),
    optimzers = tf.keras.optimizers.SGD(lr=1e-8,momentum=0.9),
    model.compile(loss=tf.keras.losses.Huber(), optimzers=optimzers, metrics=['mae'])

    history = model.fit(dataset, epochs=100, callbacks=[learning_rate_schedule])

    return history


    
def create_model():

    tf.random.set_seed(51)
    
    model = create_uncompiled_model()

    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9),metrics=["mae"])
    

    return model