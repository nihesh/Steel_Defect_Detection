# Author : Nihesh Anderson
# Date	 : 26 Sept, 2019

import torch
import torch.nn as nn
from Args import KERNEL_SIZE, FEATURE_MAPS, NUM_CLASSES

class UNet(nn.Module):

	def __init__(self, num_channels, num_classes):

		super(UNet, self).__init__()

		self.conv1 = nn.Conv2d(FEATURE_MAPS[0], FEATURE_MAPS[1], KERNEL_SIZE)
		self.conv2 = nn.Conv2d(FEATURE_MAPS[1], FEATURE_MAPS[2], KERNEL_SIZE)
		self.conv3 = nn.Conv2d(FEATURE_MAPS[2], FEATURE_MAPS[3], KERNEL_SIZE)
		self.leaky_relu = nn.LeakyReLU(True)

		self.conv_layers = [self.conv1, self.conv2, self.conv3]
		self.conv_activation = [self.leaky_relu, self.leaky_relu, self.leaky_relu]

		self.deconv1 = nn.ConvTranspose2d(FEATURE_MAPS[3], FEATURE_MAPS[2], KERNEL_SIZE)
		self.deconv2 = nn.ConvTranspose2d(FEATURE_MAPS[2], FEATURE_MAPS[1], KERNEL_SIZE)
		self.deconv3 = nn.ConvTranspose2d(FEATURE_MAPS[1], NUM_CLASSES + 1, KERNEL_SIZE)

		self.deconv_layers = [self.deconv1, self.deconv2, self.deconv3]
		self.deconv_activation = [self.leaky_relu, self.leaky_relu]

	def forward(self, x):

		x1 = self.conv_layers[0](x)
		x1 = self.conv_activation[0](x1)
		x2 = self.conv_layers[1](x1)
		x2 = self.conv_activation[1](x2)
		x3 = self.conv_layers[2](x2)
		x3 = self.conv_activation[2](x3)

		x = self.deconv_layers[0](x3)
		x = self.deconv_activation[0](x) + x2 		# Skip connection from conv layer 1
		x = self.deconv_layers[1](x)
		x = self.deconv_activation[1](x) + x1		# Skip connection from conv layer 0
		x = self.deconv_layers[2](x)

		return x

if(__name__ == "__main__"):

	pass
