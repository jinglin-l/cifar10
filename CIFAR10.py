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
    [transforms.ToTensor(), # convert a PIL to tensor, pixel values are scaled to [0,1] by dividing by 255
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # normalize tensors values from [0,1] to [-1,1]



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

# define a Neural Network
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # neel told me this should be a cone
#         self.fc1 = nn.Linear(32 * 32 * 3, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 10)  # last out_feature layer should correspond to how many classes we have
#
#
#     def forward(self, x):
#         # (B, 32, 32)
#         x = torch.flatten(x, 1) # (B, 32*32*3) flatten dims of image C * H * W
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# define a CNN

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # parameters for this training run
    learning_rate = 0.0001 # usually between 1e-3 (7b) and 1e-5 (128b), use smaller values for larger datasets. larger values learn faster but may be more inaccurate
    batch_size = 8 # should be a power of 2
    num_epochs = 2 # how many times to loop through the dataset

    # Create experiment identifier
    exp_name = f"lr{learning_rate}_bs{batch_size}_ep{num_epochs}_{datetime.now().strftime('%H%M%S')}"

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

    device = torch.device("mps")
    net.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate) # optimizer

    # train the network

    final_loss = 0.0
    val_batch = next(iter(test_loader)) # get a single batch from the test set for validation
    loss_record = []
    val_loss_record = []

    # Initialize gradient tracking dictionaries
    grad_records = {}
    for name, param in net.named_parameters():
        grad_records[name] = []

    for epoch in range(num_epochs): # loop over the dataset multiple times
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            optimizer.zero_grad() # zero out the gradient matrix, otherwise they will accumulate between batches

            # zero the parameter gradients
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs) # forward pass
            loss = criterion(outputs, labels) # compute loss
            loss.backward() # backward pass, compute gradients

            # Track gradient magnitudes every 100 batches
            if i % 100 == 0:
                for name, param in net.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        grad_records[name].append(grad_norm)

            optimizer.step() # apply gradients & update weights


            # implementing batches by ourself (not recommended)
            # # forward + backward + optimize
            # for i in range(inputs.size(0)):
            #     outputs = net(inputs[i])
            #
            #     loss = criterion(outputs, labels[i])
            #     loss.backward() # calculates the gradients, adds onto past gradients, i.e. accumulates gradients
            # optimizer.step() # applies the gradients that were accumulated with .backward()
            # optimizer.zero_grad() # zeroes the gradients, otherwise they will accumulate between batches



            # print statistics
            final_loss = loss.item()  # capture the last loss value
            if i % 20 == 0:    # print every 20th batch
                inputs, labels = val_batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                val_loss = criterion(outputs, labels)

                val_loss = val_loss.item()
                val_loss_record.append(val_loss)
                loss_record.append(final_loss)

                print(f'[{epoch + 1}, {i + 1:5d}] loss: {final_loss :.3f}')
                print(f'[{epoch + 1}, {i + 1:5d}] val_loss: {val_loss:.3f}')

    print('Finished Training')

    # test the network on the whole dataset
    correct = 0
    total = 0
    with torch.no_grad(): # no need to track gradients since we are not training
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)  # move test data to device
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1) # get the index of the max log-probability
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {accuracy} %')

    # Write results to file
    with open(f"results_{exp_name}.txt", "w") as f:
        f.write(f"Learning rate: {learning_rate}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Number of epochs: {num_epochs}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Final Loss: {final_loss:.6f}\n")

    # Plot loss curves
    plt.figure()
    plt.plot(loss_record, label='Training Loss')
    plt.plot(val_loss_record, label='Validation Loss')
    plt.xlabel('Iteration (x20)')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.savefig(f"loss_curve_{exp_name}.png")

    # Plot gradient magnitudes
    plt.figure(figsize=(12, 8))
    for name, grad_values in grad_records.items():
        if len(grad_values) > 0:
            plt.plot(grad_values, label=name, marker='o', markersize=3)
    plt.xlabel('Iteration (x100)')
    plt.ylabel('Gradient Magnitude (L2 Norm)')
    plt.title('Gradient Magnitudes Over Training')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"gradient_magnitudes_{exp_name}.png")

    # Plot gradient magnitudes separately for weights and biases
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot weights
    for name, grad_values in grad_records.items():
        if 'weight' in name and len(grad_values) > 0:
            ax1.plot(grad_values, label=name, marker='o', markersize=3)
    ax1.set_xlabel('Iteration (x100)')
    ax1.set_ylabel('Gradient Magnitude (L2 Norm)')
    ax1.set_title('Weight Gradient Magnitudes')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot biases
    for name, grad_values in grad_records.items():
        if 'bias' in name and len(grad_values) > 0:
            ax2.plot(grad_values, label=name, marker='o', markersize=3)
    ax2.set_xlabel('Iteration (x100)')
    ax2.set_ylabel('Gradient Magnitude (L2 Norm)')
    ax2.set_title('Bias Gradient Magnitudes')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"gradient_magnitudes_split_{exp_name}.png")









