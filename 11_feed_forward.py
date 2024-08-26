import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn

inputSize = 28*28
hiddenSize = 500
numClasses = 10
numEpochs = 2
batchSize = 100
learningRate = 0.001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trainData = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),  
                                           download=True)
testData = torchvision.datasets.MNIST(root='./data', 
                                           train=False, 
                                           transform=torchvision.transforms.ToTensor())
trainLoader = DataLoader(dataset=trainData, batch_size=100, shuffle=True)
testLoader = DataLoader(dataset=testData, batch_size=100, shuffle=True)

data = iter(trainLoader)
features,labels = next(data)

print(trainLoader.dataset.data.shape)

for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(features[i][0], cmap='gray')
# plt.show()

class Model(nn.Module):
    def __init__(self,inputSize,hiddenSize,numClasses):
        super(Model,self).__init__()
        self.linear1 = nn.Linear(inputSize,hiddenSize)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hiddenSize,numClasses)
        
    def forward(self,x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        return out
    
model = Model(inputSize,hiddenSize,numClasses)

CEloss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learningRate)

nTotalSteps = len(trainLoader)

for epoch in range(numEpochs):
    for i, (images, labels) in enumerate(trainLoader): 
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = CEloss(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print (f'Epoch [{epoch+1}/{numEpochs}], Step [{i+1}/{nTotalSteps}], Loss: {loss.item():.4f}')
            
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    nCorrect = 0
    nSamples = 0
    for images, labels in testLoader:
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        outputs = model(images)
        print(outputs)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        nSamples += labels.size(0)
        nCorrect += (predicted == labels).sum().item()

    acc = 100.0 * nCorrect / nSamples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
