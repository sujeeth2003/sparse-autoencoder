'''
import torch
import matplotlib.pyplot as plt

dtype = torch.float
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
torch.set_default_dtype(dtype)
torch.set_default_device(device)

print(f"using {device} device and {dtype} dtype")

a = torch.randn((), requires_grad=True)
b = torch.randn((), requires_grad=True)
c = torch.randn((), requires_grad=True)
d = torch.randn((), requires_grad=True)

learning_rate = 0.001

x = torch.linspace(-1,1,2000)
y = torch.exp(x)

yp = torch.zeros(2000)

for i in range(200000):
    yp = a* x**3 + b* x**2 + c* x + d

    loss = torch.mean((yp - y)**2)
    loss.backward()

    with torch.no_grad():
        a-= learning_rate * a.grad
        b-= learning_rate * b.grad
        c-= learning_rate * c.grad
        d-= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None
    if i % 1000 == 0:
        print(f"step: {i} loss: {loss.item()} a: {a.item()} b: {b.item()} c: {c.item()} d: {d.item()}")

plt.plot(x.data,yp.data)
plt.plot(x.data,y.data)
plt.show()

'''


import torch
import torch.nn as nn

class NNlayer(nn.Module):
    def __init__(self,inputlayer , outputlayer):
        super().__init__()
        self.output = nn.Linear(inputlayer,outputlayer)
    def forward(self,x):
        x=self.output(x)
        return x

model = NNlayer(inputlayer=1,outputlayer=1)

loss = nn.MSELoss()

optimize = torch.optim.SGD(model.parameters(),lr=0.01)

xtrain = torch.tensor([1.0,2.0,3.0,4.0,5.0]).view(-1,1)
ytain = torch.tensor([2.0,4.0,6.0,8.0,10.0]).view(-1,1)

for i in range(10000):
    prediction = model(xtrain)
    l = loss(prediction,ytain)

    optimize.zero_grad()
    l.backward()
    optimize.step()

result = model(torch.tensor([9.0]))
print(result)