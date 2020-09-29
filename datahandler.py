import torch
import matplotlib.pyplot as plt
import numpy as np

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


def compare_test_losses_full_epochs():
    DH = DataHandler()
    test_count_full = DH.load_file("Full_Model/test_count_full_model_full_epochs.pt")
    test_loss_full = DH.load_file("Full_Model/test_loss_full_model_full_epochs.pt")

    test_count_partly = DH.load_file("Partly_Regularization/test_count_full_regularized_epochs.pt")
    test_loss_partly = DH.load_file("Partly_Regularization/test_loss_full_regularized_epochs.pt")

    test_count_reg = DH.load_file("Regularized_5_layers_2_dropouts/test_count_full_regularized_epochs.pt")
    test_loss_reg = DH.load_file("Regularized_5_layers_2_dropouts/test_loss_full_regularized_epochs.pt")


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt.title("Comparing the test losses for the three models")
    ax1.scatter(test_count_full, test_loss_full, color='blue')
    ax1.scatter(test_count_partly, test_loss_partly, color='red')
    ax1.scatter(test_count_reg, test_loss_reg, color='green')
    plt.legend(['Full model', 'One dropout layer', 'Two dropout layers'], loc='upper right')
    plt.xlabel("Training examples the model has seen")
    plt.ylabel("Neg. log likelihood loss")
    DH.save_figure('compare_test_losses_full_epochs.png')
    

    print(f"{bcolors.OKGREEN}Finished comparing full epochs{bcolors.ENDC}\n{bcolors.OKBLUE}File saved\n{bcolors.ENDC}")

def compare_training_losses_full_epochs():
    DH = DataHandler()
    train_count_full = DH.load_file("Full_Model/train_count_full_model_full_epochs.pt")
    train_loss_full = DH.load_file("Full_Model/train_loss_full_model_full_epochs.pt")

    train_count_partly = DH.load_file("Partly_Regularization/train_count_regularized_full_epochs.pt")
    train_loss_partly = DH.load_file("Partly_Regularization/train_loss_regularized_full_epochs.pt")

    train_count_reg = DH.load_file("Regularized_5_layers_2_dropouts/train_count_regularized_full_epochs.pt")
    train_loss_reg = DH.load_file("Regularized_5_layers_2_dropouts/train_loss_regularized_full_epochs.pt")

    plt.figure()
    plt.title("Comparing the train losses for the three models")
    plt.plot(train_count_full, train_loss_full, color='royalblue', linewidth=0.75)
    plt.plot(train_count_partly, train_loss_partly, color='salmon', linewidth=0.75)
    plt.plot(train_count_reg, train_loss_reg, color='limegreen', linewidth=0.75)
    plt.legend(['Full model', 'One dropout layer', 'Two dropout layers'], loc='upper right')
    plt.xlabel("Training examples the model has seen")
    plt.ylabel("Neg. log likelihood loss")
    DH.save_figure('compare_train_losses.pdf')


def compare_training_difference():
    DH = DataHandler()
    train_count_full = DH.load_file("Full_Model/train_count_full_model_full_epochs.pt")
    train_loss_full = DH.load_file("Full_Model/train_loss_full_model_full_epochs.pt")

    train_loss_partly = DH.load_file("Partly_Regularization/train_loss_regularized_full_epochs.pt")
    train_loss_reg = DH.load_file("Regularized_5_layers_2_dropouts/train_loss_regularized_full_epochs.pt")

    diff_partly = np.subtract(train_loss_partly, train_loss_full)
    diff_reg = np.subtract(train_loss_reg, train_loss_full)

    plt.figure()
    plt.title("Comparing training losses compared to 'Full model'")
    plt.plot(train_count_full, diff_partly, color='blue', linewidth=0.8)
    plt.plot(train_count_full, diff_reg, color='red', linewidth=0.8)
    plt.axhline(y=0, color='green', ls=':')
    plt.axhline(y=(sum(diff_partly)/len(diff_partly)), color='royalblue', ls='--')
    plt.axhline(y=(sum(diff_reg)/len(diff_reg)), color='salmon', ls='--')
    plt.legend(['Difference with one dropout layer', 'Difference with two dropout layers',
    'Zero difference', 'Average of one dropout layer', 'Average of two dropout layers'], loc='lower right')
    plt.xlabel("Training examples the model has seen")
    plt.ylabel("Neg. log likelihood loss")
    DH.save_figure('differences_train_losses.pdf')



def main():
    print(DataHandler.__doc__)

    compare_training_difference()

if __name__ == "__main__":
    main()