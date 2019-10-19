# Author: Nihesh Anderson
# Date	: 25 Sept, 2019

import numpy as np 

# Constants
EPS = np.finfo(np.float32).tiny

# Dataset parameters
ROOT = "./dataset/"
NUM_DATASETS = 0					# Use 0 here to use all the input points
DIM = [256, 1600]					# All the images have fixed size
TRAIN_TEST_SPLIT = 0.8

# Training parameters
EPOCHS = 10000
BATCH_SIZE = 1
NUM_WORKERS = 0
LEARNING_RATE = 0.001
ALPHA = 0.5
BETA = 0.5


# Evaluation parameters
PREDICTION_THRESHOLD = 0.5

# Model parameters
SAVE_PATH = "./models/"
KERNEL_SIZE = 3
NUM_CLASSES = 4
FEATURE_MAPS = [3, 30, 45, 60]
