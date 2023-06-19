import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import itertools
import leaky_cpp # our module
from snntest import Net

FILE = 'model_state_dict.pt'
net = Net()
net.eval()
net.load_state_dict(torch.load(FILE))
print(f"import weight: {net.fc1.weight}")


# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_test = datasets.MNIST(root="data", train=False, download=True, transform=transform)


# Network Architecture
num_inputs = 36# 28*28
num_hidden = 5# 1000
num_outputs = 3# 10

# Temporal Dynamics
num_steps = 25

class SNNLeaky(nn.Module, leaky_cpp.Leaky):
    def __init__(self, input_size, output_size, weight, bias):
        nn.Module.__init__(self)
        leaky_cpp.Leaky.__init__(self, input_size, output_size)
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        leaky_cpp.Leaky.initialize_weights(self, weight, bias)
        print(f"input size: {self.getInputSize()}")
        print(f"output size: {self.getOutputSize()}")

    def forward(self, input_):
        return self.leaky_forward(input_)

class SNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.x = torch.nn.Parameter(torch.tensor(2.4, dtype=torch.float32))
        w1 = torch.rand(num_hidden, num_inputs)
        b1 = torch.rand(num_hidden)
        w2 = torch.rand(num_outputs, num_hidden)
        b2 = torch.rand(num_outputs)
        self.lif1 = SNNLeaky(num_inputs, num_hidden, w1, b1)
        print(f"snn1")
        self.lif2 = SNNLeaky(num_hidden, num_outputs, w2, b2)
        print(f"snn2")

    def forward(self, data):
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        row = data.size(0) # print(f"row: {row}")->128
        col = data.size(1) # print(f"col: {col}")->784

        for step in range(num_steps):
            print(f"step {step}")
            spk2_tmp = torch.Tensor()
            mem2_tmp = torch.Tensor()
            for index in range(row):
                print(f"index {index}")
                spk1, mem1 = self.lif1(data[index])
                spk2, mem2 = self.lif2(spk1)
                spk2_tmp = torch.cat((spk2_tmp, spk2))
                mem2_tmp = torch.cat((mem2_tmp, spk2))
            # print(f"spk2 size = {spk2_tmp.size()}")
            spk2_tmp = spk2_tmp.view(-1, num_outputs)
            mem2_tmp = mem2_tmp.view(-1, num_outputs)
            # print(f"spk2_tmp size: {spk2_tmp.size()}")
            # print(f"mem2_tmp size: {mem2_tmp.size()}")
            spk2_rec.append(spk2_tmp)
            mem2_rec.append(mem2_tmp)
        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# TEST SET ACCURACY
batch_size = 128
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"device: {device}")
snn = SNN()

total = 0
correct = 0

# drop_last switched to False to keep all samples
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

with torch.no_grad():
    print(f"before eval")
    snn.eval()
    print(f"label: {len(test_loader)}")
    for data, targets in test_loader:
        print(f"data")
        data = data.to(device)
        print(f"targets")
        targets = targets.to(device)

        # print(f"data shape: {data.size()}") -> data shape: torch.Size([128, 1, 28, 28])
        # print(f"targets shape: {targets.size()}") -> targets shape: torch.Size([128])

        # forward pass
        print(f"before snn")
        test_spk, _ = snn(data.view(data.size(0), -1))
        print(f"after snn")
        # print(f"data.view(data.size(0), -1) size: {data.view(data.size(0), -1).size()}") -> torch.Size([128, 784]
        # print(f"test_spk size: {test_spk.size()}") -> test_spk size: torch.Size([25, 128, 10])

        # calculate total accuracy
        _, predicted = test_spk.sum(dim=0).max(1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Total correctly classified test set images: {correct}/{total}")
print(f"Test Set Accuracy: {100 * correct / total:.2f}%")
