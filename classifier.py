import os
import time
import sys
import numpy as np
import cv2
import argparse
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocess import load_dataset, get_bow_extractor, get_histogram
from keras.models import load_model
from codebook import load_codebook
from utilities import init_network_model


class DogCatClassifier(object):
    """Class DogCatClassifier."""
    def __init__(self, weights_path):
        """The constructor."""
        # Using sift 500, there are 500 features
        self._model = init_network_model(500)
        # # Load the weight
        self._model.load_weights(weights_path)

        # Image detector
        self._detector = cv2.xfeatures2d.SIFT_create()
        # Codebook path
        curdir = os.path.dirname(os.path.abspath(__file__))
        codebook_path = os.path.join(curdir, 'model/sift_500/codebook.pkl')
        self._codebook = load_codebook(codebook_path)
        # Bag of Words Extractor
        self._bow_extractor = get_bow_extractor(self._detector, self._codebook)

    def predict(self, image_path, verbose=False):
        """Predict an image.

        :param image_path: The path to the image.
        :type image_path: str

        :param verbose: Boolean flag whether to print the prediction to stdout
        :type verbose: bool
        """
        if not os.path.exists(image_path):
            print 'Could not find the image...'

        image = cv2.imread(image_path)
        img_histogram = get_histogram(
            self._detector, self._bow_extractor, image)

        predictions = self._model.predict(img_histogram)
        label = 1 if predictions >= 0.5 else 0
        if verbose:
            if label == 1:
                print 'It is a dog!'
            else:
                print 'It is a cat!'

        return label
