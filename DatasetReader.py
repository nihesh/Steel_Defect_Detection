# Author: Nihesh Anderson
# Date	: 25 Sept, 2019

import torch
import numpy as np
from torch.utils.data import Dataset
import os
import cv2

def DatasetReader(Dataset):

	def __init__(self, root):

		self.root = root
		self.files = os.listdir(root)

	def __len__(self):

		return len(self.files)

	def __getitem__(self, idx):

		data = cv2.imread(self.root + self.files[idx])

		return data, target

if(__name__ == "__main__"):

	DatasetReader("./dataset/")