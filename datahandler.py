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
        """File type: .pt or .pth"""
        return torch.load('SavedFiles/'+filename)

    def save_figure(self, filename):
        plt.savefig('SavedFiles/'+filename, bbox_inches='tight')

    # Save figure to SavedFiles/__filename__
    def compare_train_test_losses(self, num_epochs, train_counter, train_losses, test_counter, test_losses, filename):
        plt.figure()
        plt.title("Training and test losses after {} epochs".format(num_epochs))
        print("\nScattering...\n")
        plt.scatter(test_counter, test_losses, color='red')
        print("\nPlotting...\n")
        plt.plot(train_counter, train_losses, color='blue')
        plt.legend(['Train loss', 'Test loss'], loc='upper right')
        plt.xlabel("Training examples the model has seen")
        plt.ylabel("Neg. log likelihood loss")
        print("\nSaving figure...\n")
        self.save_figure(filename)
        print(f"{bcolors.OKBLUE}Figure saved to path: SavedFiles/{filename}{bcolors.ENDC}")

    def compare_predict(self, NN, test_loader, model_name, filename):
        NN.load_state_dict(self.load_file(model_name))
        
        examples = enumerate(test_loader)

        _, (ex_data, _) = next(examples)

        with torch.no_grad():
            out = NN(ex_data)

        plt.figure()
        for i in range(300, 306):
            plt.subplot(2,3,i-299)
            plt.tight_layout()
            plt.imshow(ex_data[i][0], cmap='gray', interpolation='none')
            plt.title("Prediction: {}".format(
                out.data.max(1, keepdim=True)[1][i].item()))
            plt.xticks([])
            plt.yticks([])
        self.save_figure(filename)


def compare_full_epochs():
    #TODO
    pass



def main():
    print(DataHandler.__doc__)

if __name__ == "__main__":
    main()