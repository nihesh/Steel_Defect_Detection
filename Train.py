# Author: Nihesh Anderson
# Date	: 26 Sept, 2019

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from Args import DIM, ROOT, EPOCHS, BATCH_SIZE, NUM_WORKERS, LEARNING_RATE, ALPHA, BETA, NUM_CLASSES
from DatasetReader import DatasetReader
from model import UNet
import torch.optim as optim
import torch.functional as F
from copy import deepcopy
from Evaluation import MeanDiceCoefficient

def tversky_loss(true, logits, alpha, beta, eps=1e-7):

	num_classes = logits.shape[1]
	true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
	true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
	probas = torch.softmax(logits, dim=1)
	true_1_hot = true_1_hot.type(logits.type())
	dims = (0,) + tuple(range(2, true.ndimension()))
	intersection = torch.sum(probas * true_1_hot, dims)
	fps = torch.sum(probas * (1 - true_1_hot), dims)
	fns = torch.sum((1 - probas) * true_1_hot, dims)
	num = intersection
	denom = intersection + (alpha * fps) + (beta * fns)
	tversky_loss = (num / (denom + eps)).mean()
	return (1 - tversky_loss)

if(__name__ == "__main__"):

	model = UNet().cuda()
	loss_fn = nn.CrossEntropyLoss().cuda()
	optimiser = optim.Adam(model.parameters(), lr = LEARNING_RATE)

	trainset = DatasetReader(ROOT + "train/")
	testset = deepcopy(trainset)
	testset.setTrainMode(False)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
	testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

	for epoch in range(EPOCHS):

		# Training phase
		train_epoch_loss = 0
		train_dice = 0
		train_acc = 0
		
		for data, target in trainloader:
			
			batch_size = data.shape[0]
			output = model(data)

			# Tversky Loss
			loss = tversky_loss(target, output, ALPHA, BETA)

			# Multi class CE
			# loss = loss_fn(output, target)

			model.zero_grad()
			loss.backward()
			optimiser.step()

			train_epoch_loss += loss.item() * batch_size
			dice, accuracy = MeanDiceCoefficient(output, target)
			train_dice += dice.item() * batch_size
			train_acc += accuracy * batch_size

		# Testing phase
		test_epoch_loss = 0
		test_dice = 0
		test_acc = 0

		for data, target in testloader:

			batch_size = data.shape[0]
			output = model(data)

			loss = loss_fn(output, target)

			test_epoch_loss += loss.item() * batch_size
			dice, accuracy = MeanDiceCoefficient(output, target)

			test_dice += dice.item() * batch_size
			test_acc += accuracy * batch_size

		print("[ Epoch - {epoch} ] - Train Loss: {train_loss} | Train Dice: {train_dice} | Train Acc: {train_acc} | Test Loss: {test_loss} | Test Dice: {test_dice} | Test Acc: {test_acc}".format(
				epoch = epoch,
				train_loss = train_epoch_loss / len(trainset),
				test_loss = test_epoch_loss / len(testset),
				train_dice = train_dice / len(trainset),
				test_dice = test_dice / len(testset),
				train_acc = train_acc / len(trainset),
				test_acc = test_acc / len(testset)
			))
 
