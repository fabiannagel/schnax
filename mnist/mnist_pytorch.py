import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 10)

        # self.layers = nn.Sequential(
        #     nn.Linear(28*28, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 10),
        #     # Compute softmax along first dimension of a (10,)-shaped tensor
        #     nn.Softmax(dim=0)
        # )

    def forward(self, x):
        # return self.layers(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=0)
        return x

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)),])
    batch_size = 5

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader

# hyperparameters
step_size = 0.001
# momentum_mass = 0.9
num_epochs = 10
batch_size = 128

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mlp = Net()
mlp.to(device)

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(mlp.parameters(), lr=step_size, momentum=momentum_mass)
optimizer = optim.SGD(mlp.parameters(), lr=step_size)
trainloader, testloader = load_data()

for epoch in range(num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # inputs = inputs.view(inputs.shape[0], -1)
        inputs = torch.flatten(inputs, start_dim=1)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = mlp(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
print('Final loss: {}'.format(loss.item()))

PATH = 'pytorch_weights_mnist.torch'
torch.save(mlp.state_dict(), PATH)

