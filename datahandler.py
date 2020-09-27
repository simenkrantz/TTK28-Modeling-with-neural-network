import torch
import matplotlib.pyplot as plt

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class DataHandler:
    """A class for plotting data from PyTorch, saving Torch tensors to files and
    load tensors from files.

    Dependencies: torch, matplotlib.pyplot
    """
    def __init__(self):
        pass

    def save_to_file(self, torch_tensor, filename):
        """Save to SavedFiles/__filename__
        File type: .pt"""
        torch.save(torch_tensor, 'SavedFiles/'+filename)

    def load_file(self, filename):
        """File type: .pt"""
        return torch.load('SavedFiles/'+filename)

    def save_figure(self, filename):
        plt.savefig('SavedFiles/'+filename, bbox_inches='tight')

    # Save figure to SavedFiles/__filename__
    def compare_train_test_losses(self, num_epochs, train_counter, train_losses, test_counter, test_losses, filename):
        # TODO:
        #   Unable to plot the training data
        #   Unable to scatter test data
        plt.figure()
        plt.title("Training and test losses after {} epochs".format(num_epochs))
        print("\nPlotting...\n")
        plt.plot(train_counter, train_losses, color='blue')
        print("\nScattering...\n")
        plt.scatter(test_counter, test_losses, color='red')
        plt.legend(['Train loss', 'Test loss'], loc='upper right')
        plt.xlabel("Training examples the model has seen")
        plt.ylabel("Neg. log likelihood loss")
        print("\nSaving figure...\n")
        self.save_figure(filename)
        print("Figure saved to path: SavedFiles/{}".format(filename))




def main():
    print(DataHandler.__doc__)

if __name__ == "__main__":
    main()