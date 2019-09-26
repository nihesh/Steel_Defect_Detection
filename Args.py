# Author: Nihesh Anderson
# Date	: 25 Sept, 2019

# Dataset parameters
ROOT = "./dataset/"
DIM = [256, 1600]					# All the images have fixed size

# Training parameters
EPOCHS = 100
BATCH_SIZE = 2
NUM_WORKERS = 0
LEARNING_RATE = 0.001

# Model parameters
KERNEL_SIZE = 3
NUM_CLASSES = 4
FEATURE_MAPS = [3, 10, 15, 20]