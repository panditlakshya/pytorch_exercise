import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

xNumpy, yNumpy = datasets.make_regression(n_samples=200, n_features=1, noise=50, random_state=1)

x=torch.from_numpy(xNumpy.astype(np.float32))
y=torch.from_numpy(yNumpy.astype(np.float32))

y=y.view(y.shape[0],1)

nSamples,nFeatures = x.shape

inputSize = nFeatures
outputSize = 1

model = nn.Linear(inputSize, outputSize)

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

numEpochs = 100
for epoch in range(numEpochs):
    # Forward pass and loss
    yPred = model(x)
    l = loss(yPred, y)

    # Backward pass
    l.backward()

    # Update weights
    optimizer.step()

    # zero the gradients after updating
    optimizer.zero_grad()

    if (epoch+1) % 10 == 0:
        print(f'epoch {epoch+1}: w = {model.weight.item():.3f}, loss = {l.item():.8f}')
        
#plot
#model(x).detach() makes requires_grad to false
predicted = model(x).detach().numpy()
plt.plot(xNumpy, yNumpy, 'ro')
plt.plot(xNumpy, predicted, 'b')
plt.show()        