# coding=utf-8
import os

from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np

from preprocess import load_dataset

# Load training dataset
dataset_path = os.path.join(os.curdir, 'model/training_data.dat')
if not os.path.exists(dataset_path):
    print 'Dataset: %s under model dir does not exist.' % dataset_path
dataset, data_label = load_dataset(dataset_path)

batch_size = 1024
n_class = np.unique(data_label).size
n_feature = dataset[0].shape[0]

# Train with ANN
model = Sequential()
model.add(Dense(n_feature * 2, input_dim=n_feature, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_feature * 2, input_dim=n_feature, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_feature / 2, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_class, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Checkpoint
filepath = 'model/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(
    filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# Train
model.fit(
    dataset, data_label,
    validation_split=0.33,
    batch_size=batch_size,
    epochs=5000,
    callbacks=callbacks_list,
    verbose=1)

model_output = os.path.join(os.curdir, 'last_model.hdf5')
model.save(model_output)
