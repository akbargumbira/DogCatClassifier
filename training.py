# coding=utf-8
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
import numpy as np
from preprocess import load_dataset


def train(cb='sift_250', previous_model=None, lr=0.001, epochs=50):
    # Load training dataset
    dataset_path = os.path.join(os.curdir, 'model/%s/training_data.dat' % cb)
    if not os.path.exists(dataset_path):
        print 'Dataset: %s under model dir does not exist.' % dataset_path
    data, label = load_dataset(dataset_path)

    batch_size = 1024
    n_feature = data[0].shape[0]

    model = Sequential()
    model.add(Dense(n_feature * 2, input_dim=n_feature, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_feature, input_dim=n_feature, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_feature / 2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    if previous_model:
        model.load_weights(previous_model)

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=lr),
                  metrics=['accuracy'])

    # Checkpoint
    filepath = ('model/%s/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5' % cb)
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    # Train
    model.fit(
        np.array(data), np.array(label),
        validation_split=0.3,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks_list,
        shuffle=True,
        verbose=True)

    model.save_weights('model/%s/weights-last.hdf5' % cb)


# -------- SIFT 250 -------------------------------------------
# 1st training
# train()

# 2nd training
# train(previous_model='model/sift_250/train_1/weights-last.hdf5', epochs=50)

# 3rd training
# train(previous_model='model/sift_250/train_2/weights-last.hdf5',
#       lr=1e-4, epochs=100)

# 4th training
# train(previous_model='model/sift_250/train_3/weights-last.hdf5',
#       lr=1e-5, epochs=10)

# -------- SIFT 500 -------------------------------------------
# 1st training
# train(cb='sift_500')

# 2nd training
# train(
#     cb='sift_500',
#     previous_model='model/sift_500/train_1/weights-last.hdf5',
#     epochs=50
# )

# 3rd training
train(
    cb='sift_500',
    previous_model='model/sift_500/train_2/weights-improvement-44-0.79.hdf5',
    epochs=100,
    lr=1e-5
)
