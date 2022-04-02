

import pandas as pd
import tensorflow as tf


# This function normalizes the dataset using min max scaling.
def normalize_series(data, min, max):
    data = data - min
    data = data / max
    return data


# This function is used to map the time series dataset into windows of
# features and respective targets, to prepare it for training and validation.
# The first element of the first window will be the first element of
# the dataset.
#
# Consecutive windows are constructed by shifting the starting position
# of the first window forward, one at a time (indicated by shift=1).
#
# For a window of n_past number of observations of the time
# indexed variable in the dataset, the target for the window is the next
# n_future number of observations of the variable, after the
# end of the window.

def windowed_dataset(series, batch_size, n_past=10, n_future=10, shift=1):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=shift, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)


# This function loads the data from the CSV file, normalizes the data and
# splits the dataset into train and validation data. It also uses
# windowed_dataset() to split the data into windows of observations and
# targets. Finally it defines, compiles and trains a neural network. This
# function returns the final trained model.

def solution_model():
    # Reads the dataset.
    df = pd.read_csv('Gas Prices2.csv',
                     infer_datetime_format=True, index_col='Date', header=0)

    # Number of features in the dataset. We use all features as predictors to
    # predict all features of future time steps.
    N_FEATURES = len(df.columns)

    # Normalizes the data
    data = df.values
    data = normalize_series(data, data.min(axis=0), data.max(axis=0))

    # Splits the data into training and validation sets.
    SPLIT_TIME = int(len(data) * 0.8) # DO NOT CHANGE THIS
    x_train = data[:SPLIT_TIME]
    x_valid = data[SPLIT_TIME:]

    tf.keras.backend.clear_session()
    tf.random.set_seed(42)

    BATCH_SIZE = 32


    # Number of past time steps based on which future observations should be
    # predicted
    N_PAST = 10

    # Number of future time steps which are to be predicted.
    N_FUTURE = 10

    # By how many positions the window slides to create a new window
    # of observations.
    SHIFT = 1

    # Code to create windowed train and validation datasets.
    train_set = windowed_dataset(series=x_train, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)
    valid_set = windowed_dataset(series=x_valid, batch_size=BATCH_SIZE,
                                 n_past=N_PAST, n_future=N_FUTURE,
                                 shift=SHIFT)

    # Code to define model.
    model = tf.keras.models.Sequential([

        tf.keras.layers.Dense(128, activation="relu"),
        # layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(N_FEATURES)
    ])

    model.compile(loss="mae",
                    optimizer=tf.keras.optimizers.Adam(),
                    metrics=["mae"])  # we don't necessarily need this when the loss function is already MAE


    model.fit(train_set,
            epochs=100,
            verbose=1,
            batch_size=128,
            validation_data=(valid_set)
    )

    return model



if __name__ == '__main__':
    model = solution_model()
    model.save("c5q12.h5")

