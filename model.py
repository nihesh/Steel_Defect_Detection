# Author : Nihesh Anderson
# Date	 : 26 Sept, 2019

import torch
import torch.nn as nn
from Args import KERNEL_SIZE, FEATURE_MAPS, NUM_CLASSES

class UNet(nn.Module):

	def __init__(self):

		super(UNet, self).__init__()

		self.conv1 = nn.Conv2d(FEATURE_MAPS[0], FEATURE_MAPS[1], KERNEL_SIZE)
		self.conv2 = nn.Conv2d(FEATURE_MAPS[1], FEATURE_MAPS[2], KERNEL_SIZE)
		self.conv3 = nn.Conv2d(FEATURE_MAPS[2], FEATURE_MAPS[3], KERNEL_SIZE)
		self.leaky_relu = nn.LeakyReLU(True)

		self.conv_layers = [self.conv1, self.conv2, self.conv3]
		self.conv_activation = [self.leaky_relu, self.leaky_relu, self.leaky_relu]

		self.deconv1 = nn.ConvTranspose2d(FEATURE_MAPS[3], FEATURE_MAPS[2], KERNEL_SIZE)
		self.deconv2 = nn.ConvTranspose2d(FEATURE_MAPS[2], FEATURE_MAPS[1], KERNEL_SIZE)
		self.deconv3 = nn.ConvTranspose2d(FEATURE_MAPS[1], NUM_CLASSES, KERNEL_SIZE)
		self.sigmoid = nn.Sigmoid()

		self.deconv_layers = [self.deconv1, self.deconv2, self.deconv3]
		self.deconv_activation = [self.leaky_relu, self.leaky_relu, self.sigmoid]

	def forward(self, x):

		for i in range(len(self.conv_layers)):

			x = self.conv_layers[i](x)
			x = self.conv_activation[i](x)

		for i in range(len(self.deconv_layers)):

			x = self.deconv_layers[i](x)
			x = self.deconv_activation[i](x)

		x = x.transpose(1, 3)
		x = x.transpose(1, 2)

		return x

if(__name__ == "__main__"):

	pass