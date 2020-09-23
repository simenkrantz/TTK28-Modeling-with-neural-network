# Standard libraries
import pandas as pd
import torch
import torchvision as tv
import numpy as np
from mlxtend.data import loadlocal_mnist
from torch.utils.data import DataLoader as dl
from math import sqrt

# Personal libraries
from datahandler import DataHandler

train_images_path = 'ubyte_files/train-images-idx3-ubyte',
train_labels_path = 'ubyte_files/train-labels-idx1-ubyte',
test_images_path = 'ubyte_files/t10k-images-idx3-ubyte',
test_labels_path = 'ubyte_files/t10k-labels-idx3-ubyte'


def train_model(num_epochs = 4, batch_size = 64):
    """Training of the full NN model on MNIST dataset
    
    Parameters:
        num_epochs (int): Number of epochs, loopings over the complete training set
        batch_size (int): Size of training batch
    """
    

def test_model(batch_size):
    pass



def test():
    batch_size_train = 64
    batch_size_test = 1000
    #train_loader = torch.utils.data.DataLoader(
    #    tv.datasets.MNIST("./", train=True, download=False,
    #        transform=tv.transforms.Compose([
    #        tv.transforms.ToTensor(),
    #        tv.transforms.Normalize((0.1307,), (0.3081,))])),
    #    batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        tv.datasets.MNIST("./", train=False, download=False,
            transform=tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.1307,), (0.3081,))])),
        batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    
    # Return values: batch_idx, (example_data, example_targets)
    _, (example_data, example_targets) = next(examples)

    print("\nSize test data batch:\n"+str(example_data.shape)+"\n")

    DH = DataHandler()
    DH.plotter(6, example_data, example_targets, True, "test.pdf")




def main():
    # Manual seed for repeatability
    ran_seed = 96554
    torch.manual_seed(ran_seed)

    train_images_path = 'MNIST/raw/train-images-idx3-ubyte'
    train_labels_path = 'MNIST/raw/train-labels-idx1-ubyte'

    X, _ = loadlocal_mnist(train_images_path, train_labels_path)

    print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
    print(train_model.__doc__)

    test()


if __name__ == "__main__":
    main()