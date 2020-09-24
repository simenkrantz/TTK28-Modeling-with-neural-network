# Standard libraries
import pandas as pd
import torch
import torchvision as tv
import numpy as np
from torch.utils.data import DataLoader as dl
from math import sqrt

# Personal libraries
from datahandler import DataHandler


class FullNeuralNetwork(torch.nn.Module):
    """The full neural network model, used for comparison.
    
    Two 2D convolutional layers are followed by two linear, fully-connected layers.
    No regularization is used, this is applied in a different model.
    Activation function - rectified linear units (ReLUs).
    Credited: https://nextjournal.com/gkoehler/pytorch-mnist, accessed 2020-09-24

    Inherits from torch.nn.Module, https://www.educba.com/torch-dot-nn-module/
    """
    def __init__(self):
        super(FullNeuralNetwork, self).__init__()
        self.convolution_layer_1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.convolution_layer_2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.linear_layer_1 = torch.nn.Linear(320, 50)
        self.linear_layer_2 = torch.nn.Linear(50, 10)

    def ReLU(self, input):
        return torch.nn.functional.relu(input)

    def maxpool2D(self, unit, input):
        return torch.nn.functional.max_pool2d(unit, input)

    def forward(self, input):
        tensor = self.ReLU(self.maxpool2D(self.convolution_layer_1(input), 2))
        tensor = self.ReLU(self.maxpool2D(self.convolution_layer_2(tensor), 2))

        # Reshape tensor to 320 columns and an appropriate number of rows
        tensor = tensor.view(-1, 320)

        tensor = self.ReLU(self.linear_layer_1(tensor))
        return torch.nn.functional.log_softmax(self.linear_layer_2(tensor))

    def train_model(self, learning_rate, momentum, epoch, batch_size, log_interval):
        """train_model -> [train_losses, train_counter, len_train_set]
        Training of the full NN model on MNIST dataset.
        Normalize parameters:
            0.1307 (Global mean of MNIST dataset)
            0.3081 (Global standard deviation of MNIST dataset)

        Optimizer: Stochastic gradient descent
        """
        DH = DataHandler()
        train_loader = torch.utils.data.DataLoader(
            tv.datasets.MNIST("./", train=True, download=False,
                transform=tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=batch_size, shuffle=True)
        len_dataset = len(train_loader.dataset)

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        train_losses = []
        train_counter = []

        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = torch.nn.functional.nll_loss(self(data), target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print("Epoch {0}: {1:.1f}%\tLoss: {2:.5f}".format(
                    epoch, 100.*batch_idx/len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append(batch_idx*64 + (epoch-1)*len_dataset)
                DH.save_to_file(self.state_dict(), 'full_model.pth')
                DH.save_to_file(optimizer.state_dict(), 'full_optimizer.pth')

        return (train_losses, train_counter, len_dataset)


    def test_model(self, len_train_set, num_epochs, batch_size):
        """test_model -> [test_losses, test_counter]
        """
        test_loader = torch.utils.data.DataLoader(
            tv.datasets.MNIST("./", train=False, download=False,
                transform=tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=batch_size, shuffle=True)
        len_dataset = len(test_loader.dataset)
        test_losses = []
        test_counter = [i*len_train_set for i in range(num_epochs+1)]

        self.eval()
        loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                out = self(data)
                loss += torch.nn.functional.nll_loss(out,target, size_average=False).item()
                pred = out.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        loss /= len_dataset
        test_losses.append(loss)

        print("\nTest set average loss {:.5f}\tAccuracy {}/{} ({:.1f}%)\n".format(
            loss, correct, len_dataset, 100.*correct/len_dataset))

        return (test_losses, test_counter)
        

def main():
    torch.backends.cudnn.enabled = False

    # Manual seed for repeatability
    ran_seed = 96554
    torch.manual_seed(ran_seed)

    num_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000

    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    DH = DataHandler()
    NN = FullNeuralNetwork()

    for epoch in range(1, num_epochs + 1):
        (train_loss, train_count, len_train_data) = NN.train_model(
            learning_rate,momentum,epoch,batch_size_train,log_interval)
        (test_loss, test_count) = NN.test_model(
            len_train_data, num_epochs, batch_size_test)

    DH.save_to_file(train_loss, 'train_loss_full_model.pt')
    DH.save_to_file(train_count, 'train_count_full_model.pt')
    DH.save_to_file(test_loss, 'test_loss_full_model.pt')
    DH.save_to_file(test_count, 'test_count_full_model.pt')

    #DH.compare_train_test_losses(
    #    num_epochs, train_count, train_loss, test_count, test_loss,'test_train_loss_full_model.png')


if __name__ == "__main__":
    main()