import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader as dl
from math import sqrt

# Manual seed for repeatability
ranSeed = 96554
torch.manual_seed(ranSeed)
print("Finished setting up")