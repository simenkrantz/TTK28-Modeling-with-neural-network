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

    def save_figure(self, filename, filepath):
        plt.savefig(filepath+filename, bbox_inches='tight')

    # num_subplot must be even int
    def plotter(self, num_subplots, torch_data, torch_targets, bool_save_fig, filename, filepath="SavedFiles/"):
        """Plot PyTorch data"""
        plt.figure(figsize=(16,9))
        for i in range(num_subplots):
            plt.subplot(2, num_subplots//2, i+1)
            plt.tight_layout()
            plt.imshow(torch_data[i][0], cmap='gray', interpolation='none')
            plt.title("Truth: {}".format(torch_targets[i]))
            plt.xticks([])
            plt.yticks([])
        if(bool_save_fig):
            self.save_figure(filename, filepath)
        else:
            plt.show()

def main():
    print(DataHandler.__doc__)

if __name__ == "__main__":
    main()