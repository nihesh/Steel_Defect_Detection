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

if(__name__ == "__main__"):

	model = UNet().cuda()
	loss_fn = nn.BCELoss()
	optimiser = optim.Adam(model.parameters(), lr = LEARNING_RATE)

	trainset = DatasetReader(ROOT + "train/")

	trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers = NUM_WORKERS)


	for epoch in range(EPOCHS):

		epoch_loss = 0

		for data, target in trainloader:

			batch_size = data.shape[0]
			output = model(data)

			loss = loss_fn(output, target)

			model.zero_grad()
			loss.backward()
			optimiser.step()

			epoch_loss += loss.item() * batch_size

		print("[ Epoch - {epoch} ] - Loss: {loss}".format(
				epoch = epoch,
				loss = epoch_loss / len(trainloader)
			))