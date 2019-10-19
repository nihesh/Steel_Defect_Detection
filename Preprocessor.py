# Author: Nihesh Anderson 
# Date	: 26 Sept, 2019

import os
import pandas
import math
import pickle
from Args import DIM
import numpy as np

IMAGE_SRC = "./dataset/train/images/"
LABEL_SRC = "./dataset/train.csv"
LABEL_TARGET = "./dataset/train/labels/"

def LoadCSV(root):

	data = pandas.read_csv(root)
	loaded_data = {}

	for i in range(len(data)):

		raw_name = data.loc[i, "ImageId_ClassId"]
		image_name = raw_name[ : raw_name.find("_")]
		image_class = int(raw_name[raw_name.find("_") + 1 : ]) - 1

		if(image_name not in loaded_data):
			loaded_data[image_name] = {}

		pixels = data.loc[i, "EncodedPixels"]
		
		if(type(pixels) == float and math.isnan(pixels)):
			pixels = []
		else:
			pixels = list(map(int, pixels.split(" ")))
		
		loaded_data[image_name][image_class] = pixels

	# Loaded data contains file name as dim 0, class id [0, 1, 2, 3] as dim 1 and the stored value is the encoded pixel information stored in an even size list

	return loaded_data

def WriteToDisk(target, LABEL_TARGET, file):

	file = open(LABEL_TARGET + file, "wb")
	pickle.dump(target, file)
	file.close()

def Column_Major_To_Spatial(pixel):

	global DIM

	pixel -= 1		# 0 based indexing

	out = np.zeros(2)
	out[0] = pixel % DIM[0]
	out[1] = pixel // DIM[0]

	return out.reshape(1, 2)

def EncodedPixels_To_Spatial_Coords(encoded_pixels):

	out = []
	for i in range(0, len(encoded_pixels), 2):
		for j in range(encoded_pixels[i + 1]):
			out.append(Column_Major_To_Spatial(encoded_pixels[i] + j))

	if(len(out) == 0):
		return np.empty([0, 2])

	return np.concatenate(out, axis = 0).astype(int)

def DecodePixels(data): 

	global DIM
	num_classes = len(data)

	target = np.zeros([DIM[0], DIM[1]]).astype(int)

	for class_id in data:

		pixel_set = EncodedPixels_To_Spatial_Coords(data[class_id])
		
		# Set the corresponding bit if pixel set is not empty
		if(pixel_set.shape[0]):
			target[pixel_set[:, 0], pixel_set[:, 1]] = class_id + 1

	return target

def Decode_Labels_CSV():

	global LABEL_TARGET, LABEL_SRC

	os.system("rm -rf " + LABEL_TARGET)
	os.mkdir(LABEL_TARGET)

	loaded_meta = LoadCSV(LABEL_SRC)

	cnt = 0

	for file in loaded_meta: 

		cnt += 1
		print("Processing", cnt)

		target = DecodePixels(loaded_meta[file])
		WriteToDisk(target, LABEL_TARGET, file + ".dump")

if(__name__ == "__main__"):

	Decode_Labels_CSV()
