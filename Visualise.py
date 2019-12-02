
import torch
from DatasetReader import DatasetReader
from Args import ROOT
import numpy as np
import cv2
import random

WEIGHT = 0.2

def showDefect(img, cls_data):

	if(np.max(cls_data) == 0):
		return False

	cls_of_interest = np.max(cls_data)

	img = np.transpose(img, (1, 2, 0))

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(cls_data[i][j] == cls_of_interest):
				img[i][j][0] = (1 - WEIGHT) * img[i][j][0] + WEIGHT * 255

	img = img.astype(np.uint8)
	cv2.imwrite("./Results/img.jpg", img)

	return True

if(__name__ == "__main__"):
	
	dataset = DatasetReader(ROOT + "train/")

	i = random.randint(0, len(dataset) - 1)
	while(not showDefect(dataset[i][0].cpu().detach().numpy(), dataset[i][1].cpu().detach().numpy())):
		i = random.randint(0, len(dataset) - 1)
