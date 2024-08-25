import torch
import torch.nn as nn

# Binary classification
class Model(nn.Module):
    def __init__(self,inputSize,hiddenSize):
        super(Model,self).__init__()
        self.linear1 = nn.Linear(inputSize,hiddenSize)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hiddenSize,1)
        
    def forward(self,x):
        out= self.linear1(x) 
        out = self.relu(out)
        out = self.linear2(out)
        yPred = torch.sigmoid(out)
        return yPred
    
model = Model(input_size=28*28, hidden_size=5)
loss = nn.BCELoss()  

# Multi-class classification
class Model(nn.Module):
    def __init__(self,inputSize,hiddenSize,numClasses):
        super(Model,self).__init__()
        self.linear1 = nn.Linear(inputSize,hiddenSize)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hiddenSize,numClasses)
        
    def forward(self,x):
        out= self.linear1(x) 
        out = self.relu(out)
        out = self.linear2(out)
        yPred = torch.sigmoid(out)
        return yPred
    
model = Model(input_size=28*28, hidden_size=5,numClasses = 3)
loss = nn.CrossEntropyLoss()  