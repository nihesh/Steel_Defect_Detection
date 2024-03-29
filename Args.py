# Author: Nihesh Anderson
# Date	: 25 Sept, 2019

import numpy as np 

# Constants
EPS = np.finfo(np.float32).tiny

# Dataset parameters
ROOT = "./dataset/"
NUM_DATASETS = 0
DIM = [256, 1600]					# All the images have fixed size
TRAIN_TEST_SPLIT = 0.8

# Training parameters
EPOCHS = 100
BATCH_SIZE = 3
NUM_WORKERS = 0
LEARNING_RATE = 0.01

# Evaluation parameters
PREDICTION_THRESHOLD = 0.5

# Model parameters
KERNEL_SIZE = 3
NUM_CLASSES = 4
FEATURE_MAPS = [3, 30, 45, 60]
