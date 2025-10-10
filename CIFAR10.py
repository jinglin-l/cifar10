# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import sys
from datetime import datetime


# load and normalize CIFAR10 dataset

transform = transforms.Compose( # Compose chains several transforms together
    [transforms.ToTensor(), # convert a PIL to tensor, pixel values are scaled to [0,1] by
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))] # normalize tensors values from [0,1] to [-1,1]
)

# how does one choose an appropriate batch size?
batch_size = 4

# training set (how big?)
training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download = True, transform=transform)

# test set (how big?)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download = True, transform=transform)


# lookup table, 0 is plane, 1 is car, ...
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def show_img(img):
    global npimg
    img = img / 2 + 0.5     # unnormalize from [-1,1] to [0,1]
    npimg = img.numpy()   # convert from tensor to numpy array
    print(f'npimg.shape before transpose: {npimg.shape}') # (C,H,W)
    npimg_transposed = np.transpose(npimg, (1,2,0)) # convert from (C,H,W) to (H,W,C)
    print(f'npimg.shape after transpose: {npimg_transposed.shape}') # (C,H,W)
    plt.show()

# define a Convolutional Neural Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32*32*3, 3000)
        self.fc2 = nn.Linear(3000, 1000)
        self.fc3 = nn.Linear(1000, 10)

    def forward(self, x):
        # (B, 32, 32)
        x = torch.flatten(x, 1) # (B, 32*32*3) flatten dims of image C * H * W
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




if __name__ == "__main__":
    # parameters for this training run
    learning_rate = 0.001
    batch_size = 4
    num_epochs = 2

    # file redirect model training metrics to a file

    # Create experiment identifier
    exp_name = f"lr{lr}_bs{batch_size}_ep{num_epochs}_{datetime.now().strftime('%H%M%S')}"

    with open(f"results_{exp_name}.txt", "w") as f:
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Final Loss: {final_loss}\n")




    train_loader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    # get some random training images
    dataiter = iter(train_loader) # create an iterator from the dataloader

    images, labels = next(dataiter) # get the next batch of images and labels

    # print dims of images and labels
    print(f"batch shape:" , images.shape) # (batch size, number of channels (RGB), height, width) (4, 3, 32, 32)
    print(f"single image shape:" , images[0].shape) # (channels, height, width) (3, 32, 32)
    print(f"flattened size:" , images[0].numel()) # C*H*W

    # show images
    show_img(torchvision.utils.make_grid(images)) # make a grid from batch
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    net = Net()

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train the network

    for epoch in range(2): # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    # test the network on the whole dataset
    correct = 0
    total = 0
    with torch.no_grad(): # no need to track gradients since we are not training
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1) # get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')









