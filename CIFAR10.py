# https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import sys
from datetime import datetime
import matplotlib.pyplot as plt


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
        self.pool = nn.MaxPool2d(2, 2) # kernel size, stride

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) # convolution layyer will decrease the image size (called spatial dim) by a certain fixed amount at each layer. that fixed amount can be calculated using some formula that depends on kernel size, stride, padding
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # out channels should increase as we go deeper, not sure why or how much at each step and overall
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  #
        self.fc = nn.Linear(128, 10)  # 128 features → 10 classes

    def forward(self, x):
        # Input: (B, 3, 32, 32) - batch of RGB images
        x = F.relu(self.conv1(x))      # (B, 32, 32, 32) - 32 feature maps, spatial dims preserved
        x = F.relu(self.conv2(x))      # (B, 64, 32, 32) - 64 feature maps, spatial dims preserved
        x = self.pool(x)               # (B, 64, 16, 16) - spatial dims halved by pooling
        x = F.relu(self.conv3(x))      # (B, 64, 16, 16) - same channels, spatial dims preserved
        x = F.relu(self.conv4(x))      # (B, 128, 16, 16) - 128 feature maps, spatial dims preserved
        x = self.pool(x)               # (B, 128, 8, 8) - spatial dims halved again
        x = F.relu(self.conv5(x))      # (B, 128, 8, 8) - same channels, spatial dims preserved
        x = self.global_pool(x)        # (B, 128, 1, 1) - global pooling reduces to 1×1
        x = torch.flatten(x, 1)        # (B, 128) - flatten spatial dims, keep batch and channels
        x = self.fc(x)                 # (B, 10) - fully connected layer outputs class logits
        return x


if __name__ == "__main__":
    # parameters for this training run
    learning_rate = 0.001 # usually between 1e-3 (7b) and 1e-5 (128b), use smaller values for larger datasets. larger values learn faster but may be more inaccurate
    batch_size = 32 # should be a power of 2
    num_epochs = 15 # how many times to loop through the dataset
    nn_type = "cnn"
    date = datetime.now().strftime('%Y-%m-%d%H:%M:%S')
    experiment_name = "adding scheduler"

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
    #mean and std of the first image
    print(f"mean: {images[0].mean():.3f}, std: {images[0].std():.3f}")



    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    net = Net()

    device = torch.device("mps")
    net.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate) # optimizer

    # define a learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # train the network

    final_loss = 0.0
    val_batch = next(iter(test_loader)) # get a single batch from the test set for validation
    loss_record = []
    val_loss_record = []
    train_acc_record = []
    val_acc_record = []
    lr_record = []

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
            if i % 20 == 0:
                # train accuracy
                _, predicted = torch.max(outputs.data, 1) # get the index of the max
                total = labels.size(0)
                correct = (predicted == labels).long().sum().item()
                train_accuracy = 100 * correct / total


                # print every 20th batch
                inputs, labels = val_batch
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                val_loss = criterion(outputs, labels)

                val_loss = val_loss.item()
                val_loss_record.append(val_loss)
                loss_record.append(final_loss)
                # val accuracy
                _, predicted = torch.max(outputs.data, 1) # get the index of the max
                total = labels.size(0)
                correct = (predicted == labels).sum().item()
                val_accuracy = 100 * correct / total

                # track metrics for plotting
                train_acc_record.append(train_accuracy)
                val_acc_record.append(val_accuracy)
                lr_record.append(optimizer.param_groups[0]["lr"])

                print(f'[{epoch + 1}, {i + 1:5d}] loss: {final_loss :.3f} val_loss: {val_loss:.3f} train_acc: {train_accuracy:.2f}% val_acc: {val_accuracy:.2f}% lr: {optimizer.param_groups[0]["lr"]:.6f}')

        scheduler.step(val_loss) # step the scheduler based on validation loss

    print('Finished Training')

    # plot loss curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Loss curves
    axes[0, 0].plot(loss_record, label='Training Loss', alpha=0.7)
    axes[0, 0].plot(val_loss_record, label='Validation Loss', alpha=0.7)
    axes[0, 0].set_xlabel('Checkpoint (every 20 batches)')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Accuracy curves
    axes[0, 1].plot(train_acc_record, label='Training Accuracy', alpha=0.7)
    axes[0, 1].plot(val_acc_record, label='Validation Accuracy', alpha=0.7)
    axes[0, 1].set_xlabel('Checkpoint (every 20 batches)')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Accuracy Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Learning rate over time
    axes[1, 0].plot(lr_record, color='green', alpha=0.7)
    axes[1, 0].set_xlabel('Checkpoint (every 20 batches)')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')  # log scale helps visualize LR changes

    # Plot 4: Train-Val gap (overfitting indicator)
    loss_gap = [val - train for train, val in zip(loss_record, val_loss_record)]
    acc_gap = [train - val for train, val in zip(train_acc_record, val_acc_record)]

    ax4 = axes[1, 1]
    ax4.plot(loss_gap, label='Loss Gap (Val - Train)', alpha=0.7, color='red')
    ax4.set_xlabel('Checkpoint (every 20 batches)')
    ax4.set_ylabel('Loss Gap', color='red')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    ax4_twin = ax4.twinx()
    ax4_twin.plot(acc_gap, label='Acc Gap (Train - Val)', alpha=0.7, color='blue')
    ax4_twin.set_ylabel('Accuracy Gap (%)', color='blue')
    ax4_twin.tick_params(axis='y', labelcolor='blue')
    ax4.set_title('Train-Val Gap (Overfitting Indicator)')

    plt.suptitle(f'Training Metrics - {exp_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'training_metrics_{exp_name}.png', dpi=150, bbox_inches='tight')
    print(f'Training metrics saved to training_metrics_{exp_name}.png')
    plt.show()  # Display plot interactively
    plt.close()

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
        f.write(f"NN type: {nn_type}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Final Loss: {final_loss:.6f}\n")
        f.write(f"Date: {date}\n")






