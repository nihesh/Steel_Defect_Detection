# Author: Nihesh Anderson
# Date	: 25 Sept, 2019

import torch
import numpy as np
from torch.utils.data import Dataset
from Args import NUM_DATASETS, TRAIN_TEST_SPLIT
import os
import pickle
import cv2
import random

class DatasetReader(Dataset):

	def __init__(self, root):

		self.root = root
		self.files = os.listdir(root + "images/")
		if(NUM_DATASETS != 0):
			self.files = self.files[:NUM_DATASETS]
		self.files.sort()
		random.Random(0).shuffle(self.files)

		split_point = int(TRAIN_TEST_SPLIT * len(self.files))
		self.train = self.files[:split_point]
		self.test = self.files[split_point:]

		# 1 for train and 0 for test
		self.train_mode = True

	def setTrainMode(self, train_mode):
		
		self.train_mode = train_mode

	def __len__(self):

		if(self.train_mode):
			return len(self.train)
		else:
			return len(self.test)

	def __getitem__(self, idx):

		if(self.train_mode):
			self.files = self.train
		else:
			self.files = self.test

		data = cv2.imread(self.root + "images/" + self.files[idx]).astype(float)
		file = open(self.root + "labels/" + self.files[idx] + ".dump", "rb")
		target = pickle.load(file).astype(float)

		# Convert to tensors and load it on the gpu
		data = torch.tensor(data).float().cuda()
		data = data.transpose(0, 2)
		data = data.transpose(1, 2)
		target = torch.tensor(target).float().cuda()

		return data, target

if(__name__ == "__main__"):

	pass
