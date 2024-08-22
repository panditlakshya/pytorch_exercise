import torch

x = torch.tensor([[1],[2],[3],[4]],dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]],dtype=torch.float32)

w = torch.tensor(0.0,dtype=torch.float32,requires_grad=True)

