
import torch
from Args import PREDICTION_THRESHOLD, EPS, NONE_THRESHOLD
import torch.nn.functional as F
from Args import NUM_CLASSES
from sklearn.metrics import accuracy_score

def MeanDiceCoefficient(prediction, target):

	global PREDICTION_THRESHOLD

	prediction = F.softmax(prediction, dim = 1)
	max_value, _ = torch.max(prediction, dim = 1, keepdim = True)
	prediction[prediction >= max_value] = 1
	prediction[prediction < 1] = 0
	prediction = prediction.byte()
	prediction_labels = torch.argmax(prediction, dim = 1)

	accuracy = accuracy_score(prediction_labels.view(-1).cpu().detach().numpy(), target.view(-1).cpu().detach().numpy())

	target_shape = target.shape
	target = target.view(-1)
	one_hot = torch.zeros([target.shape[0], NUM_CLASSES + 1]).cuda()
	one_hot[torch.arange(target.shape[0]), target] = 1
	one_hot = one_hot.view(target_shape[0], target_shape[1], target_shape[2], NUM_CLASSES + 1)
	target = one_hot.transpose(1, 3).transpose(2, 3)
	target = target.byte()

	# print(prediction.sum(dim = 3).sum(dim = 2)[:, 1:].sum(dim = 1), target.sum(dim = 3).sum(dim = 2)[:, 1:].sum(dim = 1))

	indices = prediction.sum(dim = 3).sum(dim = 2)[:, 1:].sum(dim = 1) <= NONE_THRESHOLD
	prediction[indices, 1:] = 0
	prediction[indices, 0] = 1


	intersection = (prediction & target).float()

	intersection = 2 * intersection.sum(dim = 3).sum(dim = 2)
	intersection = intersection[:, 1:]
	union = prediction.float().sum(dim = 3).sum(dim = 2) + target.float().sum(dim = 3).sum(dim = 2)
	union = union[:, 1:]
	
	iou = ((intersection + torch.tensor([EPS]).float().cuda()) / (union + torch.tensor([EPS]).float().cuda())).view(-1)

	return torch.mean(iou) * 100, accuracy * 100


if(__name__ == "__main__"):

	pass
