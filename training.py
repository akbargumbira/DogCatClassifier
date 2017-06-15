# coding=utf-8
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np
from preprocess import load_dataset


def train(previous_model=None, epochs=50, initial_epoch=0):
    """Train the best network model found by hyperparameter optimization.

    :param previous_model: Path to the initial model weights to be trained.
    :type previous_model: str

    :param epochs: The number of epoch for training.
    :type epochs: int

    :param initial_epoch: The initial epoch number.
    :type initial_epoch: int
    """
    # Load training dataset
    dataset_path = os.path.join(os.curdir, 'model/sift_500/training_data.dat')
    if not os.path.exists(dataset_path):
        print 'Dataset: %s under model dir does not exist.' % dataset_path
    data, label = load_dataset(dataset_path)

    batch_size = 122
    n_feature = data[0].shape[0]

    # The best network model found by hyperparameter optimization
    model = Sequential()
    # 1st hidden layer
    model.add(Dense(900, input_dim=n_feature, activation='relu'))
    model.add(Dropout(0.7499998682813903))
    # 2nd hidden layer
    model.add(Dense(609, activation='relu'))
    model.add(Dropout(0.3780638891568126))
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
        loss='binary_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy'])

    if previous_model:
        model.load_weights(previous_model)

    # Checkpoint
    filepath = ('model/final_model_500/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5')
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Train
    model.fit(
        np.array(data), np.array(label),
        validation_split=0.3,
        batch_size=batch_size,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks_list,
        shuffle=True,
        verbose=True)

    model.save_weights('model/final_model_500/weights-last.hdf5')


# 1st training
# train(epochs=100)

# 2nd training
train(
    previous_model='model/final_model_500/weights-improvement-89-0.78.hdf5',
    epochs=200,
    initial_epoch=101
)
