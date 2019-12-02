
import torch
import torch.nn as nn
import numpy as np
from Args import ROOT
from DatasetReader import DatasetReader
import cv2
import torch.nn.functional as F
import os

MODEL_PATH = "./models/"
SAVE_PATH = "./Results/"
MIN_DEFECTS = 1000
HIGHLIGHT_FACTOR = 1

if(__name__ == "__main__"):

	files = os.listdir(MODEL_PATH)
	files.sort()
	print("Using file", files[-1])
	MODEL_PATH = os.path.join(MODEL_PATH, files[-1])

	model = torch.load(MODEL_PATH).cuda()
	dataset = DatasetReader(ROOT + "train/")
	
	SAMPLE_POINT = 0
	init_defects = -1
	while(True):
		test_image, label = dataset[SAMPLE_POINT]
		label = torch.min(label, torch.tensor([1]).long().cuda())
		if(label.sum() >= MIN_DEFECTS):
			init_defects = label.sum()
			break
		SAMPLE_POINT += 1

	prediction = model(test_image.unsqueeze(0))
	print(prediction[0, :, 0, 0])
	prediction = F.softmax(prediction, dim = 1)
	prediction = torch.argmax(prediction, dim = 1).sum(dim = 0)
	prediction[prediction > 0] = 1
	print(prediction[0, 0])

	prediction = prediction.cpu().detach().numpy()
	label = label.cpu().detach().numpy()
	test_image = test_image.cpu().detach().numpy()
	mask1 = test_image.astype(float)
	mask2 = test_image.astype(float)

	def_cnt = 0
	pred_cnt = 0
	for i in range(mask1.shape[1]):
		for j in range(mask1.shape[2]):
			if(prediction[i][j] > 0):
				pred_cnt += 1
				mask1[0][i][j] = mask1[1][i][j] = 0
				mask1[2][i][j] = 255
			if(label[i][j] > 0):
				def_cnt += 1
				mask2[0][i][j] = mask2[1][i][j] = 0
				mask2[2][i][j] = 255

	print("Actual defects: {def_cnt} | Predicted defects: {pred_cnt}".format(
		def_cnt = def_cnt, 
		pred_cnt = pred_cnt
	))
	
	predicted_defect = (test_image * (1 - HIGHLIGHT_FACTOR) + mask1 * HIGHLIGHT_FACTOR)
	actual_defect = (test_image * (1 - HIGHLIGHT_FACTOR) + mask2 * HIGHLIGHT_FACTOR)
	predicted_defect[predicted_defect > 255] = 255
	predicted_defect[predicted_defect < 0] = 0
	predicted_defect = predicted_defect.astype(np.uint8)

	actual_defect[actual_defect > 255] = 255
	actual_defect[actual_defect < 0] = 0
	actual_defect = actual_defect.astype(np.uint8)

	predicted_defect = predicted_defect.transpose(1, 2, 0)
	actual_defect = actual_defect.transpose(1, 2, 0)

	cv2.imwrite(SAVE_PATH + "./predicted_defect.jpg", predicted_defect)
	cv2.imwrite(SAVE_PATH + "./actual_defect.jpg", actual_defect)
