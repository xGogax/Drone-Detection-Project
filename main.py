import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras import layers

from keras.callbacks import EarlyStopping
from keras.losses import SparseCategoricalCrossentropy

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from keras.applications import MobileNetV2