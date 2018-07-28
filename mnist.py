import time
import torch.utils.data
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch import optim
from torchvision import datasets
from torchvision import transforms


class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(28 * 28, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.BatchNorm1d(128),
                                 nn.Linear(128, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 27),
                                 nn.ReLU())

    def forward(self, x):
        return self.net(x)

batch_size = 32
epochs = 1
trans = transforms.ToTensor()

mnist_trainset = datasets.MNIST(root='./mnist', train=True, download=True, transform=trans)
mnist_testset = datasets.MNIST(root='./mnist', train=False, download=True, transform=trans)
mnist_train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size)
mnist_test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=1)

model = LinearModel()
params = model.parameters()
optimizer = optim.SGD(params, lr=1e-3, momentum=0.9)
loss_func = nn.CrossEntropyLoss()

for _ in range(epochs):
    for i, (x, y) in enumerate(mnist_train_loader):
        x_flat = x.view(x.size(0), -1)
        pred = model(x_flat)
        loss = loss_func(pred, y)
        loss.backward()
        optimizer.step()
        model.zero_grad()
        if i % 100 == 0:
            print(loss.item())

model.eval()

correct = 0

for i, (x, y) in enumerate(mnist_test_loader):
    x_flat = x.view(1, -1)
    pred = model(x_flat)
    pred = pred.argmax()
    correct += 1 if pred.item() == y.item() else 0

    pixels = np.array((x * 256).numpy(), dtype=np.uint8)[0][0]
    plt.title('Predicted is {}, Label is {}'.format(pred.item(), y.item()))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    time.sleep(3)
    plt.close('all')

print('Accuracy: {}'.format(correct / len(mnist_test_loader)))
