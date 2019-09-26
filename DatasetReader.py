# Author: Nihesh Anderson
# Date	: 25 Sept, 2019

import torch
import numpy as np
from torch.utils.data import Dataset
import os
import pickle
import cv2

class DatasetReader(Dataset):

	def __init__(self, root):

		self.root = root
		self.files = os.listdir(root + "images/")

	def __len__(self):

		return len(self.files)

	def __getitem__(self, idx):

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

	trainset = DatasetReader("./dataset/train/")	
	print(trainset[0][0].shape, trainset[0][1].shape)