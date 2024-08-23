import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import numpy as np
import math

class WineData(Dataset):
    def __init__(self):
        data = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = data.shape[0]
        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(data[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(data[:, [0]]) # size [n_samples, 1]
        
        # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    
# create dataset
dataset = WineData()

# get first sample and unpack
# first_data = dataset[0]
# features, labels = first_data
# print(features, labels)  

dataLoader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

# convert to an iterator and look at one random sample
dataIter = iter(dataLoader)
data = next(dataIter)
# print(data)

features, labels = data
print(features, labels)

# Dummy Training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataLoader):
        
        # here: 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
        # Run your training process
        if (i+1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')
            

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),  
                                           download=True)

train_loader = DataLoader(dataset=train_dataset, 
                                           batch_size=3, 
                                           shuffle=True)

# look at one random sample
dataiter = iter(train_loader)
data = next(dataiter)
inputs, targets = data
print(inputs.shape, targets.shape)

  
        