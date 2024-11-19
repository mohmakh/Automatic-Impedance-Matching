import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load data
df = pd.read_excel("/content/Impedance_data.xlsx")

# Split data into training and validation sets
train_df = df.sample(frac=0.8, random_state=0)
val_df = df.drop(train_df.index)

# Normalize input data
train_mean = train_df.mean()
train_std = train_df.std()
train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std

# Convert data to NumPy arrays
train_labels = train_df[['C1', 'C2', 'L']].to_numpy()
val_labels = val_df[['C1', 'C2', 'L']].to_numpy()
train_features = train_df['Frequency'].to_numpy()
val_features = val_df['Frequency'].to_numpy()

# Define model architecture
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[1]),
    layers.Dense(64, activation='relu'),
    layers.Dense(3)
])

# Compile model
model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.001),
    loss='mse'
)


# Train model
history = model.fit(
    train_features, train_labels,
    validation_data=(val_features, val_labels),
    batch_size=32,
    epochs=100,
    verbose=1
)

# Evaluate model
loss = model.evaluate(val_features, val_labels, verbose=0)
print("Mean squared error on validation set: {:.3f}".format(loss))

