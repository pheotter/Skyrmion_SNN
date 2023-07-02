import snntorch as snn
# from snntorch import spikeplot as splt
# from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import numpy as np
import itertools

# dataloader arguments
batch_size = 128

# Network Architecture
num_inputs = 28*28
num_hidden = 1000
num_outputs = 10

# Temporal Dynamics
num_steps = 25
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x)
            # print(f"cur1 size = {cur1.size()}") -> cur1 size = torch.Size([128, 1000])
            spk1, mem1 = self.lif1(cur1, mem1)
            # print(f"spk1 size = {spk1.size()}") -> spk1 size = torch.Size([128, 1000])
            cur2 = self.fc2(spk1)
            # print(f"cur2 size = {cur2.size()}") -> cur2 size = torch.Size([128, 10])
            spk2, mem2 = self.lif2(cur2, mem2)
            # print(f"spk2 size = {spk2.size()}") -> spk2 size = torch.Size([128, 10])
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# pass data into the network, sum the spikes over time
# and compare the neuron with the highest number of spikes
# with the target

def print_batch_accuracy(data, targets, train=False):
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(
    data, targets, epoch,
    counter, iter_counter,
        loss_hist, test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")

if __name__ == '__main__':

    dtype = torch.float
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Define a transform
    transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize((0,), (1,))])

    mnist_train = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)

    num_epochs = 1
    loss_hist = []
    test_loss_hist = []
    counter = 0
    net = Net().to(device)
    # bef_weight = net.fc1.weight

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))


    # Outer training loop
    for epoch in range(num_epochs):
        iter_counter = 0
        train_batch = iter(train_loader)

        # Minibatch training loop
        for data, targets in train_batch:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass
            net.train()
            spk_rec, mem_rec = net(data.view(batch_size, -1))
            # print(f"mem_rec size = {mem_rec.size()}") -> mem_rec size = torch.Size([25, 128, 10])

            # initialize the loss & sum over time
            loss_val = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_val += loss(mem_rec[step], targets)
                # print(f"step = {step}, mem_rec = {mem_rec[step]}, loss_val = {loss_val}")
                # mem_rec size [25, 128. 784]
                # loss_val size [128], sum of the 128 values

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Test set
            with torch.no_grad():
                net.eval()
                test_data, test_targets = next(iter(test_loader))
                test_data = test_data.to(device)
                test_targets = test_targets.to(device)

                # Test set forward pass
                test_spk, test_mem = net(test_data.view(batch_size, -1))

                # Test set loss
                test_loss = torch.zeros((1), dtype=dtype, device=device)
                for step in range(num_steps):
                    test_loss += loss(test_mem[step], test_targets)
                test_loss_hist.append(test_loss.item())

                # Print train/test loss/accuracy
                if counter % 50 == 0:
                    train_printer(data, targets, epoch, counter, iter_counter,
                        loss_hist, test_loss_hist, test_data, test_targets)
                counter += 1
                iter_counter +=1


    # TEST SET ACCURACY
    total = 0
    correct = 0

    # drop_last switched to False to keep all samples
    test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

    with torch.no_grad():
      net.eval()
      i = 0
      for data, targets in test_loader:
        print(f"i = {i}")
        i += 1
        data = data.to(device)
        targets = targets.to(device)
        # print(f"data shape: {data.size()}") -> data shape: torch.Size([128, 1, 28, 28])
        # print(f"targets shape: {targets.size()}") -> targets shape: torch.Size([128])

        # forward pass
        test_spk, _ = net(data.view(data.size(0), -1))
        # print(f"data.view(data.size(0), -1) size: {data.view(data.size(0), -1).size()}") -> torch.Size([128, 784]
        # print(f"test_spk size: {test_spk.size()}") -> test_spk size: torch.Size([25, 128, 10])

        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    print(f"Total correctly classified test set images: {correct}/{total}")
    print(f"Test Set Accuracy: {100 * correct / total:.2f}%")

    print(f"net: {net}")
    print(f"weight size: {net.fc1.weight.size()}")
    print(f"bias size: {net.fc1.bias.size()}")

    fc1_weight = net.fc1.weight
    fc2_weight = net.fc2.weight
    fc1_bias = net.fc1.bias
    fc2_bias = net.fc2.bias

    # save model state_dict
    FILE = 'model_state_dict.pth'
    print("Model's state_dict:")
    for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())

    torch.save(net.state_dict(), FILE)
