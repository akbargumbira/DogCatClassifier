# coding=utf-8
import cv2
from preprocess import get_cat_dog_data
from utilities import serialize_object


detector = cv2.xfeatures2d.SIFT_create()
codebook_path = '/home/agumbira/dev/python/DogCatClassifier/model/codebook.pkl'
training_dir = '/home/agumbira/dev/data/dog_cat_kaggle/train/'
id, data, labels = get_cat_dog_data(detector, codebook_path, training_dir)

training_output = '/home/agumbira/dev/python/DogCatClassifier/model/training_data.dat'
serialize_object((id, data, labels), training_output)
