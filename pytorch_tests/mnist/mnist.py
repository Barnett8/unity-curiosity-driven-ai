 # Adam Barnett 5/16/21
 # based on https://nextjournal.com/gkoehler/pytorch-mnist

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def _init_(self):
        # create the first layer, a convolutional layer with:
        # 1 channel from the input image because it's greyscale (values between 0 and 1)
        # 10 channels that the convolution layer can produce
        # and a kernel size of 5, the size of the convolving matrix that is applied
        
        # the rest of the values are the defaults set by pytorch for Conv2d: 
        # stride can change the rate at which the convolution matrix is passed over the image, default of 1 pixel
        # padding gives the ability to add zeros to the edges of the input
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 10, kernel_size = 5, stride = 1, padding = 0, padding_mode = 'zeros', dilation = 1, groups = 1, bias = True)
        # create the second layer, a ReLU layer 
        self.fc1 = nn.Linear(320, 50)
        # create the second layer, a convolution layer that turns the 10 channels of the first layer into 20
        self.conv2 = nn.Conv2d(10, 20, 5)
        # modify the second convolution layer so that it has dropout
        self.conv2_drop = nn.Dropout2d()
        # a final relu layer that takes the output of the second convolution layer and turns it into 10 outputs, the 10 digits
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        # take the input, x and run it through the first convolutional layer 
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), '/results/model.pth')
            torch.save(optimizer.state_dict(), '/results/optimizer.pth')

def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Settings

# number of epochs to run training
n_epochs = 3 
# size of the training batch for each epoch
batch_size_train = 64
# number of tests in the training batch
batch_size_test = 1000
#
learning_rate = 0.01
#
momentum = 0.5
#
log_interval = 10
#
random_seed = 1 
# Disable the GPU training for simplicity
torch.backends.cudnn.enabled = False
# set the random seed to a known seed for repeatability
torch.manual_seed(random_seed)

# gather the data for the train and test sets from the MNIST dataset
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=True, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./', train=False, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ])),
        batch_size=batch_size_test, shuffle=True)

network = Net()
#optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# test the untrained network
test()
# train the network for n_epochs, testing after each epoch
for epoch in range(1, n_epochs + 1):
  train(epoch)
  test()
