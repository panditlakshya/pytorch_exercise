import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np

class WineData(Dataset):
    def __init__(self,transform=None):
        data = np.loadtxt('./data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = data.shape[0]
        self.x_data = data[:, 1:] 
        self.y_data = data[:, [0]] 
        self.transform = transform
        
    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        sample = self.x_data[index], self.y_data[index]
        
        if self.transform:
            sample = self.transform(sample)
        return sample

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
class ToTensor:
    def __call__(self,sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
class MulTransform:
    # multiply inputs with a given factor
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        inputs, targets = sample
        inputs *= self.factor
        return inputs, targets

    
print('Without Transform') 
dataset = WineData()   
print(dataset[0]) 
firstData = dataset[0]
features, labels = firstData
print(type(features), type(labels))
print(features, labels)

print('With Transform') 
dataset = WineData(transform=ToTensor())   
print(dataset[0]) 
firstData = dataset[0]
features, labels = firstData
print(type(features), type(labels))
print(features, labels)

print('With MulTransform') 
dataset = WineData(transform=MulTransform(2))  
print(dataset[0]) 
firstData = dataset[0]
features, labels = firstData
print(type(features), type(labels))
print(features, labels)

print('With Tensor and Multiplication Transform')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset = WineData(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)
