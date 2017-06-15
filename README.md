# Classifying Dog or Cat using Bag Of Words Concept
This is a proof of concept to apply Bag of Words model in Natural Language 
Processing into classifying images, in this case to classify between dog or cat.

## Dataset
The dataset is obtained from [Kaggle: Dog vs Cat](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data) containing 25000 images of a 
dogs and cats. Each image has the label as the part of the filename.

## Building Codebook
A codebook (or dictionary) can be built from all the images in the dataset 
with some parameters below:
 * The full path to the root directory containing all the images
 * The output file full path
 * Descriptors algorithm to obtain descriptors of all images (either SIFT or 
 KAZE)
 * The number of vocabularies
 
Format: 

```python codebook.py -i [images_dir] -o [output_file] -a [sift|kaze] -s [vocab_size] --verbose(optional)```

Example:

```python codebook.py -i /home/agumbira/dev/data/dog_cat_kaggle/train -o /home/agumbira/dev/python/BOWImageClassifier/model/dog_cat_kaggle/codebook_kaze_200.pkl -a kaze -s 200 --verbose```
 
## Finding the best network model
The best network model is found by running ```hyperparameter_optimization
.py```. The search scope is defined in the ```space``` variable. 1000 network
 models with various parameters are run and the one with the highest ROC AUC 
 is selected. This model then used for training.

## Training
To train the network model (found by hyperparameter optimization), run ```python training.py```
