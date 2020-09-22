import pandas as pd
import torch
import numpy as np
from mlxtend.data import loadlocal_mnist
from torch.utils.data import DataLoader as dl
from math import sqrt


class FullNeuralNetwork:
    """The neural network trained on MNIST dataset, with standard parameters only."""
    def __init__(
        self,
        num_epochs,
        train_images_path = 'train-images-idx3-ubyte',
        train_labels_path = 'train-labels-idx1-ubyte',
        test_images_path = 't10k-images-idx3-ubyte',
        test_labels_path = 't10k-labels-idx3-ubyte'
        ):
        self.num_epochs = num_epochs
        self.train_images_path = train_images_path
        self.train_labels_path = train_labels_path
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path


    def train_model(self):
        pass

    def test_model(self):
        pass




def main():
    # Manual seed for repeatability
    ran_seed = 96554
    torch.manual_seed(ran_seed)

    train_images_path = 'train-images-idx3-ubyte'
    train_labels_path = 'train-labels-idx1-ubyte'
    test_images_path = 't10k-images-idx3-ubyte'
    test_labels_path = 't10k-labels-idx3-ubyte'

    X, y = loadlocal_mnist(train_images_path, train_labels_path)

    print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
    print(FullNeuralNetwork.__doc__)


if __name__ == "__main__":
    main()