import torch.nn  as nn
import torch
from torch.nn.modules.activation import ReLU


l = nn.Linear(2, 5)
v = torch.FloatTensor([1,2])
print(v)
print(l(v))
print(type(nn.Linear))

s = nn.Sequential(
    nn.Linear(2, 5),
    nn.ReLU(),
    nn.Linear(5, 20),
    nn.ReLU(),
    nn.Linear(20,10),
    nn.Dropout(p=0.3),
    nn.Softmax(dim=1)
)
print(s)