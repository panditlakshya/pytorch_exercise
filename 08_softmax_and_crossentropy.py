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

