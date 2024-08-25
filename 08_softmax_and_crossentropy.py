import torch
import torch.nn as nn
import numpy as np

#
#        -> 2.0              -> 0.65  
# Linear -> 1.0  -> Softmax  -> 0.25   -> CrossEntropy(y, y_hat)
#        -> 0.1              -> 0.1                   
#
#     scores(logits)      probabilities
#                           sum = 1.0
#

# Softmax applies the exponential function to each element, and normalizes
# by dividing by the sum of all these exponentials
# -> squashes the output to be between 0 and 1 = probability
# sum of all probabilities is 1

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

x = np.array([2,1,0.1],dtype=np.float32)

y = softmax(x)
print(x,y)
x = torch.tensor([2,1,0.1],dtype=torch.float32)
y= torch.softmax(x,dim=0) # dim = 0 is same as we do in numpy for axis
print(x,y)

#Cross entropy
# true label : One-hot encoding - [1,0,0]
# Predicted : [0.7,0.2,0.1] 
# loss = -1 * sum(y * log(y_predicted))

def crossEntropy(y,yPred):
    return -np.sum(np.log(yPred) * y,axis=0)

Y = np.array([1, 0, 0])
yPredGood = np.array([0.7, 0.2, 0.1])
yPredBad = np.array([0.1, 0.3, 0.6])
l1 = crossEntropy(Y, yPredGood)
l2 = crossEntropy(Y, yPredBad)
print(l1,l2)

loss = nn.CrossEntropyLoss()

# target is of size nSamples = 1
# each element has class label: 0, 1, or 2
# Y (=target) contains class labels, not one-hot
Y = torch.tensor([0])

yPredGood = torch.tensor([[0.7, 0.2, 0.1]])
yPredBad = torch.tensor([[0.1, 0.3, 0.6]])
l1 = loss(yPredGood, Y)
l2 = loss(yPredBad, Y)
print(l1.item(),l2.item())

_, predictions1 = torch.max(yPredGood, 1)
_, predictions2 = torch.max(yPredBad, 1)
print(f'Actual class: {Y.item()}, Y_pred1: {predictions1.item()}, Y_pred2: {predictions2.item()}')
 