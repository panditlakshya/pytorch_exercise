import torch
import torch.utils
import torch.utils.data
from torchvision.utils import make_grid
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset has PILImage images of range [0, 1]. 
# We transform them to Tensors of normalized range [-1, 1]
# mean(0.5,0.5,0.5) and std(0.5,0.5,0.5),
# corresponding to the mean and standard deviation for each of the three channels (R, G, B) in the image
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

trainDataset = torchvision.datasets.CIFAR10(root='./data', train=True,transform=transform, download=True)
testDataset = torchvision.datasets.CIFAR10(root='./data', train=False,transform=transform, download=True)

#hyper parameters
batchSize = 4
numEpochs = 5
learningRate = 0.001

trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=batchSize, shuffle=True)
trainLoader = torch.utils.data.DataLoader(testDataset, batch_size=batchSize, shuffle=True)

print(trainDataset.__getitem__(0))

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
# get some random training images
dataiter = iter(trainLoader)
images, labels = next(dataiter)

# show images
imshow(make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(batchSize)))

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 =  nn.Conv2d(in_channels=3, out_channels=6,kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        # -> n, 3, 32, 32
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 6, 14, 14
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 16, 5, 5
        x = x.view(-1, 16 * 5 * 5)            # -> n, 400
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = self.fc3(x)                       # -> n, 10
        return x
    
model = ConvNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
nTotalSteps = len(trainLoader)

