import torch
import torch.nn as nn

x = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

# w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

nSamples,nFeatures = x.shape
print("nSamples,nFeatures",nSamples,nFeatures)

# test data prediction (5)
xTest = torch.tensor([5],dtype=torch.float32)

inputSize = nFeatures
outputSize = nFeatures

# model = nn.Linear(inputSize, outputSize)

class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        # define diferent layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(inputSize, outputSize)


print(f'Prediction before training: f(5) = {model(xTest).item():.3f}')

# 2) Define loss and optimizer
learningRate = 0.01
nIters = 100

loss= nn.MSELoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=learningRate)

# Training
for epoch in range(nIters):
    #forward pass
    yPred = model(x)
    
    #loss
    l = loss(y,yPred)
    
    #backward propagation (Compute parameter updates)
    l.backward()
    
    #make the updates for each parameter
    optimizer.step()
    
    #a clean up step for PyTorch
    optimizer.zero_grad()
    
    if epoch % 10 == 0:
        [w, b] = model.parameters() # unpack parameters
        print('epoch ', epoch+1, ': w = ', w[0][0].item(), ' loss = ', l)
        
print(f'Prediction after training: f(5) = {model(xTest).item():.3f}')        
    
    







