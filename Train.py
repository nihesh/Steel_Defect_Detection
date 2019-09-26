# Author: Nihesh Anderson
# Date	: 26 Sept, 2019

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from Args import DIM, ROOT, EPOCHS, BATCH_SIZE, NUM_WORKERS, LEARNING_RATE
from DatasetReader import DatasetReader
from model import UNet
import torch.optim as optim
from copy import deepcopy
from Evaluation import MeanDiceCoefficient

if(__name__ == "__main__"):

	model = UNet().cuda()
	loss_fn = nn.BCELoss()
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
		
		for data, target in trainloader:

			batch_size = data.shape[0]
			output = model(data)

			loss = loss_fn(output, target)

			model.zero_grad()
			loss.backward()
			optimiser.step()

			train_epoch_loss += loss.item() * batch_size
			train_dice += MeanDiceCoefficient(output, target).item() * batch_size

		# Testing phase
		test_epoch_loss = 0
		test_dice = 0
		for data, target in testloader:

			batch_size = data.shape[0]
			output = model(data)

			loss = loss_fn(output, target)

			test_epoch_loss += loss.item() * batch_size
			test_dice += MeanDiceCoefficient(output, target).item() * batch_size

		print("[ Epoch - {epoch} ] - Train Loss: {train_loss} | Train Dice: {train_dice} | Test Loss: {test_loss} | Test Dice: {test_dice}".format(
				epoch = epoch,
				train_loss = train_epoch_loss / len(trainset),
				test_loss = test_epoch_loss / len(testset),
				train_dice = train_dice / len(trainset),
				test_dice = test_dice / len(testset)
			))
 
