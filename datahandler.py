import torch
import matplotlib.pyplot as plt

class DataHandler:
    """A class for plotting data from PyTorch, saving Torch tensors to files and
    load tensors from files.

    Dependencies: torch, matplotlib.pyplot
    """
    def __init__(self):
        pass

    def save_to_file(self, torch_tensor, filename):
        """File type: .pt"""
        torch.save(torch_tensor, filename)

    def load_file(self, filename):
        """File type: .pt"""
        torch.load(filename)

    def plotter(self):
        plt.figure(figsize=(16,9))
        

def main():
    print(DataHandler.__doc__)

if __name__ == "__main__":
    main()