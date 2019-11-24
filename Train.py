# Author: Nihesh Anderson
# Date	: 26 Sept, 2019

import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from Args import DIM, ROOT, EPOCHS, BATCH_SIZE, NUM_WORKERS, LEARNING_RATE, ALPHA, BETA, NUM_CLASSES, SAVE_PATH, LOAD_PATH, GAMMA, FOCAL_WEIGHT, BCE_WEIGHT, TVERSKY_WEIGHT
from DatasetReader import DatasetReader
from unet_model import UNet		# Change unet_model to model for my implementation of unet
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from Evaluation import MeanDiceCoefficient
from matplotlib import pyplot as plt
from torch.autograd import Variable
from refinenet import rf101

# reference: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.contiguous().view(-1)
        target = target.long()

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target.view_as(logpt))
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

# reference: https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py
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

def one_hot(target):

	target_shape = target.shape
	target = target.view(-1)
	one_hot = torch.zeros([target.shape[0], NUM_CLASSES + 1]).cuda()
	one_hot[torch.arange(target.shape[0]), target] = 1
	one_hot = one_hot.view(target_shape[0], target_shape[1], target_shape[2], NUM_CLASSES + 1)
	target = one_hot.transpose(1, 3).transpose(2, 3)

	return target

if(__name__ == "__main__"):

	# model = UNet(3, NUM_CLASSES + 1).cuda()
	model = rf101(NUM_CLASSES + 1).cuda()
	start_epoch = 0

	if(LOAD_PATH != ""):
		model = torch.load(LOAD_PATH)
		for i in range(1000, -1, -1):
			if(LOAD_PATH.find("epoch_" + str(i)) != -1):
				start_epoch = i + 1
				break
	
	loss_fn = nn.CrossEntropyLoss().cuda()
	BCELoss = torch.nn.BCEWithLogitsLoss()
	focal = FocalLoss(gamma = GAMMA)
	optimiser = optim.Adam(model.parameters(), lr = LEARNING_RATE)

	trainset = DatasetReader(ROOT + "train/")
	testset = deepcopy(trainset)
	testset.setTrainMode(False)

	trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)
	testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = True, num_workers = NUM_WORKERS)

	epoch_axis = []
	loss_axis = []
	train_dice_axis = []
	test_dice_axis = []

	for epoch in range(start_epoch, EPOCHS):

		epoch_axis.append(epoch + 1)

		# Training phase
		train_epoch_loss = 0
		train_dice = 0
		train_acc = 0
		
		for data, target in trainloader:
			
			batch_size = data.shape[0]
			output = model(data)
			# Tversky Loss
			loss = TVERSKY_WEIGHT * tversky_loss(target, output, ALPHA, BETA)
			
			# BCE Loss
			one_hot_vector = one_hot(target)
			loss = loss + BCE_WEIGHT * BCELoss(output, one_hot_vector)
			loss = loss + FOCAL_WEIGHT * focal(output, one_hot_vector)

			# Multi class CE
			# loss = loss_fn(output, target)

			model.zero_grad()
			loss.backward()
			optimiser.step()

			train_epoch_loss += loss.item() * batch_size
			dice, accuracy = MeanDiceCoefficient(output, target)
			train_dice += dice.item() * batch_size
			train_acc += accuracy * batch_size

			loss = None
			output = None
			one_hot_vector = None

		# Testing phase
		test_epoch_loss = 0
		test_dice = 0
		test_acc = 0

		for data, target in testloader:

			batch_size = data.shape[0]
			output = model(data)
	
			# Tversky Loss
			loss = TVERSKY_WEIGHT * tversky_loss(target, output, ALPHA, BETA)
					
			# BCE Loss
			one_hot_vector = one_hot(target)
			loss = loss + BCE_WEIGHT * BCELoss(output, one_hot_vector)
			loss = loss + FOCAL_WEIGHT * focal(output, one_hot_vector)

			# Multi class CE
			# loss = loss_fn(output, target)

			test_epoch_loss += loss.item() * batch_size
			dice, accuracy = MeanDiceCoefficient(output, target)

			test_dice += dice.item() * batch_size
			test_acc += accuracy * batch_size

			output = None
			loss = None
			one_hot_vector = None

		print("[ Epoch - {epoch} ] - Train Loss: {train_loss} | Train Dice: {train_dice} | Train Acc: {train_acc} | Test Loss: {test_loss} | Test Dice: {test_dice} | Test Acc: {test_acc}".format(
				epoch = epoch,
				train_loss = train_epoch_loss / len(trainset),
				test_loss = test_epoch_loss / len(testset),
				train_dice = train_dice / len(trainset),
				test_dice = test_dice / len(testset),
				train_acc = train_acc / len(trainset),
				test_acc = test_acc / len(testset)
			))

		loss_axis.append(train_epoch_loss / len(trainset))
		train_dice_axis.append(train_dice / len(trainset))
		test_dice_axis.append(test_dice / len(testset))

		torch.save(model, SAVE_PATH + "epoch_" + str(epoch) + "_DICE_" + str(train_dice / len(trainset)) + ".pth")
 
	plt.clf()
	plt.plot(epoch_axis, loss_axis)
	plt.xlabel("Epoch")
	plt.ylabel("Tversky + BCE Loss")
	plt.title("Training loss curve")
	plt.savefig("./Results/loss.jpg")

	plt.clf()
	plt.plot(epoch_axis, np.asarray(train_dice_axis), label = "train_dice")
	plt.plot(epoch_axis, np.asarray(test_dice_axis), label = "test_dice")
	plt.legend()
	plt.xlabel("Epoch")
	plt.ylabel("Dice score")
	plt.title("Model evaluation scores")
	plt.savefig("./Results/dice.jpg")

