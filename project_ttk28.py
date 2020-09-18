import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np
from mlxtend.data import loadlocal_mnist
from torch.utils.data import DataLoader as dl
from math import sqrt

# Manual seed for repeatability
ran_seed = 96554
torch.manual_seed(ran_seed)
print("Finished setting up")

images_file = 'train-images-idx3-ubyte'
labels_file = 'train-labels-idx1-ubyte'

X, y = loadlocal_mnist(images_file, labels_file)

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))