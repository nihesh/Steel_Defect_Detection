# Author: Nihesh Anderson
# Date	: 27 Sept, 2019

import torch
from Args import PREDICTION_THRESHOLD, EPS

def MeanDiceCoefficient(prediction, target):

	global PREDICTION_THRESHOLD

	prediction = prediction >= PREDICTION_THRESHOLD
	prediction = prediction.byte()
	target = target.byte()

	intersection = (prediction & target).float()
	union = (prediction | target).float()

	intersection = 2 * intersection.sum(dim = 2).sum(dim = 1)
	union = prediction.float().sum(dim = 2).sum(dim = 1) + target.float().sum(dim = 2).sum(dim = 1)

	iou = ((intersection + torch.tensor([EPS]).float().cuda()) / (union + torch.tensor([EPS]).float().cuda())).view(-1)

	return torch.mean(iou) * 100


if(__name__ == "__main__"):

	pass
