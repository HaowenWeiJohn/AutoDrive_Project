import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50, num_classes)


    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = NN(784,10)
x = torch.rand(64,784)
print(model(x).shape)

device = torch.device('cpu')

input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epoch = 5

train_dataset = datasets.MNIST(root='dataset/',train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/',train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

model = NN(input_size=input_size, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epoch):
    print(epoch)
    for batch_idex, (data, target) in enumerate(train_loader):
        data = data.to(device=device)
        target = target.to(device=device)

        #get correct shape
        data = data.reshape(data.shape[0], -1)
        #
        scores = model(data)
        loss = criterion(scores, target) # input and target tensor
        #backward
        optimizer.zero_grad() # does not store  the grad from previous
        loss.backward()
        optimizer.step()


# check the accuracy

def check_accuracy(loader, model):
    if loader.dataset.train:
        print('Training data')
    else:
        print('Test data')
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            _, prediction = scores.max(1) # prediction is  n*10 index of max second dim
            num_correct+= (prediction==y).sum()
            num_samples+=prediction.size(0)

        print(num_correct/num_samples)

    model.train()

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
