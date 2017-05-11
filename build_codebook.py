# coding=utf-8
from codebook import build_codebook

build_codebook(
    '/home/agumbira/dev/data/dog_cat_kaggle/train',
    '/home/agumbira/dev/python/DogCatClassifier/model/codebook.pkl',
    'sift',
    500,
    verbose=True)
