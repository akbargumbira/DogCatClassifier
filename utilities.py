# coding=utf-8
import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout


def serialize_object(obj, output_path):
    """Serialize object into the specified output file."""
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


def load_serialized_object(input_path):
    """Load serialized object from the specified path"""
    with open(input_path, "rb") as f:
        obj = pickle.load(f)
    return obj


def init_network_model(features_number):
    """Create network model.

    :param features_number: The number of features.
    :type features_number: int
    """
    # The best network model found by hyperparameter optimization
    model = Sequential()
    # 1st hidden layer
    model.add(Dense(900, input_dim=features_number, activation='relu'))
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

    return model
