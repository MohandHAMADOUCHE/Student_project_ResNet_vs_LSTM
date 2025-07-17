import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import tf2onnx
import onnx

# Import
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neural_network
import typing_extensions
import warnings
import xml.etree.ElementTree as ET
import datetime
import logging
import math
import sys
import re
import os
import seaborn as sns
import subprocess
import pickle
import tpot2
import networkx as nx
import sklearn
import sklearn.metrics
import sklearn.datasets
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
from paretoset import paretoset
import argparse
from tpot2 import *
from tpot2 import GraphPipeline
from sklearn.linear_model import SGDRegressor
from tpot2.builtin_modules import ZeroTransformer, OneTransformer
from tpot2.config.regressors import params_SGDRegressor
from pyDOE import lhs
from mpl_toolkits.mplot3d import axes3d
from sklearn import preprocessing, linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, accuracy_score, mean_squared_error, r2_score, make_scorer, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Lasso, ARDRegression, ElasticNet, ElasticNetCV, HuberRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from itertools import combinations
from typing_extensions import Required, TypedDict
from matplotlib import rcParams
from scipy.stats import pearsonr
from sklearn.neural_network import MLPClassifier
from keras.utils import to_categorical 
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from keras_tuner import HyperModel, RandomSearch
from keras_tuner.tuners import BayesianOptimization
from tensorflow.keras.layers import LocallyConnected1D
from tensorflow.keras.datasets import mnist

from sklearn.neural_network import MLPClassifier
from keras.utils import to_categorical 
from tensorflow import keras
from tensorflow.keras import layers
from rfga_libs.klib import *
from rfga_libs.CNN import *
from rfga_libs.MLP import *
from rfga_libs.RF import *
from rfga_libs.evaluation_CNN import evaluate_CNN
#from rfga_libs.evaluation_MLP import *
#from rfga_libs.evaluation_RF import *


import gc
from keras_tuner import HyperModel, RandomSearch
from keras_tuner.tuners import BayesianOptimization
from tensorflow.keras.layers import LocallyConnected1D
from tensorflow.keras.datasets import mnist
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, Dense, BatchNormalization, Activation, Input
from rfga_libs.optimizers import *
from rfga_libs.datasets import *
from rfga_libs.optimizers import keras_tuner_opti
from rfga_libs.estimators import *
from rfga_libs.estimators import estimation

import subprocess
import pandas as pd
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# Load and preprocess the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images to the range [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape the images to add a channel dimension
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# One-hot encode the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the neural network model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_split=0.2)

#sleep for 10 seconds
import time

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')


#input_signature = tf.TensorSpec(shape=(None, 28, 28), dtype=tf.float32)

#input_signature = [tf.TensorSpec([28,28], tf.float32, name='x')]

# Use from_function for tf functions
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)
onnx.save(onnx_model, "model_test.onnx")



# print input and output shape of onnx model
onnx_model = onnx.load("model_test.onnx")
print(onnx.checker.check_model(onnx_model))
print(onnx.helper.printable_graph(onnx_model.graph))


# Get model input shapes
print("Input shapes:")
for input_tensor in onnx_model.graph.input:
    input_name = input_tensor.name
    input_shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
    input_type = input_tensor.type.tensor_type.elem_type  # Extract the data type
    print(f"Input Name: {input_name}, Shape: {input_shape}, Type: {onnx.TensorProto.DataType.Name(input_type)}")


# Get model output shapes
print("\nOutput shapes:")
for output_tensor in onnx_model.graph.output:
    output_name = output_tensor.name
    output_shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
    output_type = output_tensor.type.tensor_type.elem_type  # Extract the data type
    print(f"Output Name: {output_name}, Shape: {output_shape}, Type: {onnx.TensorProto.DataType.Name(output_type)}")


