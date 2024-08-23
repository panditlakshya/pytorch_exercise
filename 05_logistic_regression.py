import torch
import torch.nn as nn
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


bc = datasets.load_breast_cancer()
x,y = bc.data, bc.target

nSamples,nFeatures = x.shape

xTrain,xTest, yTrain, yTest = train_test_split(x,y,test_size=0.2,random_state=1111)
sc = StandardScaler()
xTrain = sc.fit_transform(xTrain)
xTest = sc.transform(xTest)

xTrain = torch.from_numpy(xTrain.astype(np.float32))
xTest = torch.from_numpy(xTest.astype(np.float32))
yTrain = torch.from_numpy(yTrain.astype(np.float32))
yTest = torch.from_numpy(yTest.astype(np.float32))

yTrain = yTrain.view(yTrain.shape[0],1)
yTest = yTest.view(yTest.shape[0],1)

class LogisticRegression(nn.Module):
    def __init__(self,nInputFeatures):
        super().__init__()
        self.linear=nn.Linear(nInputFeatures,1)
        
    def forward(self,x):
        return torch.sigmoid(self.linear(x))    
    
model = LogisticRegression(nFeatures)    

nEpochs = 100
learningRate = 0.01
loss = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learningRate)

for epochs in range(nEpochs):
    yPred = model(xTrain)
    l = loss(yPred,yTrain)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if(epochs+1)%10 == 0:
        [w, b] = model.parameters()
        print(w,w.grad)
        print(f'epoch {epochs+1}: loss = {l.item():.8f}')


with torch.no_grad():
    yPredicted = model(xTest)
    yPredictedCls = yPredicted.round()
    acc = yPredictedCls.eq(yTest).sum() / float(yTest.shape[0])
    print(f'accuracy: {acc.item():.4f}')