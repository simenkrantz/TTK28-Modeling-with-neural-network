import torch
import torchvision as tv
import torch.optim as optim

from datahandler import DataHandler, bcolors
from full_mnist_model import FullNeuralNetwork

class Regularization(FullNeuralNetwork):
    """Class for regularized neural network. Dropout is conducted in
    convolution layers 1 and 2, with three linear layers at the end.

    Activation function: ReLU

    Optimizer: Adam
    """
    def __init__(self):
        super(Regularization, self).__init__()
        self.conv_layer_1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv_1_dropout = torch.nn.Dropout2d()
        self.conv_layer_2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv_2_dropout = torch.nn.Dropout2d()
        self.lin_layer_1 = torch.nn.Linear(320, 140)
        self.lin_layer_2 = torch.nn.Linear(140, 50)
        self.lin_layer_3 = torch.nn.Linear(50, 10)

    def forward(self, input):
        tensor = self.ReLU(self.maxpool2D(self.conv_1_dropout(self.conv_layer_1(input)),2))
        tensor = self.ReLU(self.maxpool2D(self.conv_2_dropout(self.conv_layer_2(tensor)),2))
        tensor = self.ReLU(self.lin_layer_1(tensor.view(-1, 320)))
        tensor = self.ReLU(self.lin_layer_2(tensor))
        return torch.nn.functional.log_softmax(self.lin_layer_3(tensor))


class PartlyRegularization(FullNeuralNetwork):
    """Class for regularized neural network. Dropout is conducted in
    convolution layer 2, with two linear layers at the end. This is the same
    as the Full Model

    Activation function: ReLU

    Optimizer: Adam
    """
    def __init__(self):
        super(PartlyRegularization, self).__init__()
        self.conv_layer_1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv_layer_2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.conv_2_dropout = torch.nn.Dropout2d()
        self.lin_layer_1 = torch.nn.Linear(320, 50)
        self.lin_layer_2 = torch.nn.Linear(50, 10)

    def forward(self, input):
        tensor = self.ReLU(self.maxpool2D(self.conv_layer_1(input),2))
        tensor = self.ReLU(self.maxpool2D(self.conv_2_dropout(self.conv_layer_2(tensor)),2))
        tensor = self.ReLU(self.lin_layer_1(tensor.view(-1, 320)))
        return torch.nn.functional.log_softmax(self.lin_layer_2(tensor))


def main():
    print(f"{bcolors.HEADER} Running regularized model {bcolors.ENDC}\n")
    torch.backends.cudnn.enabled = False
    torch.manual_seed(556)
    num_epochs = 3
    train_batch_size = 64
    test_batch_size = 1000
    lr = 0.01
    log_interval = 10

    test_loader = torch.utils.data.DataLoader(
                tv.datasets.MNIST("./", train=False, download=False,
                    transform=tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=1000, shuffle=True)

    partly_regularization = True

    DH = DataHandler()

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*60000 for i in range(num_epochs+1)]

    # ------------------------------------------------------------------------- #
    if partly_regularization:
        print(f"{bcolors.HEADER}Partly regularization model \n{bcolors.ENDC}")

        NN_partly = PartlyRegularization()
        optimizer = optim.Adam(NN_partly.parameters(), lr=lr)
        
        (test_losses, test_counter) = NN_partly.test_model(
            test_losses, test_counter, 60000, test_batch_size)

        for epoch in range(1, num_epochs+1):
            (train_losses, train_counter, len_train_data) = NN_partly.train_model(
                train_losses, train_counter, optimizer, epoch, train_batch_size, log_interval,
                'Partly_Regularization/regularized_model.pth', 'Partly_Regularization/regularized_optimizer.pth'
            )

            (test_losses, test_counter) = NN_partly.test_model(
                test_losses, test_counter, len_train_data, test_batch_size
            )
        
        print(f"\n{bcolors.OKBLUE} Saving files...\n{bcolors.ENDC}")

        DH.save_to_file(train_losses, "Partly_Regularization/train_losses_regularized.pt")
        DH.save_to_file(train_counter, "Partly_Regularization/train_counter_regularized.pt")
        DH.save_to_file(test_losses, "Partly_Regularization/test_losses_regularized.pt")
        DH.save_to_file(test_counter, "Partly_Regularization/test_counter_regularized.pt")
        
        print(f"\n{bcolors.OKBLUE} Loading files...\n{bcolors.ENDC}")

        train_losses = DH.load_file("Partly_Regularization/train_losses_regularized.pt")
        train_counter = DH.load_file("Partly_Regularization/train_counter_regularized.pt")
        test_losses = DH.load_file("Partly_Regularization/test_losses_regularized.pt")
        test_counter = DH.load_file("Partly_Regularization/test_counter_regularized.pt")
        

        # Continue training for 4 more epochs
        NN_part_reg_cont = PartlyRegularization()
        NN_part_reg_cont.load_state_dict(DH.load_file("Partly_Regularization/regularized_model.pth"))

        DH.compare_predict(NN_part_reg_cont, test_loader,'Partly_Regularization/regularized_model.pth', 'Partly_Regularization/compare_predict_partly_reg_3_epochs.png')

        optimizer_cont = optim.Adam(NN_part_reg_cont.parameters(), lr=lr)
        optimizer_cont.load_state_dict(DH.load_file("Partly_Regularization/regularized_optimizer.pth"))

        for epoch in range(4, 8):
            test_counter.append(epoch*60000)
            (train_losses, train_counter, len_train_data) = NN_part_reg_cont.train_model(
                train_losses, train_counter, optimizer_cont, epoch, train_batch_size, log_interval,
                'Partly_Regularization/regularized_model.pth', 'Partly_Regularization/regularized_optimizer.pth'
            )

            (test_losses, test_counter) = NN_part_reg_cont.test_model(
                test_losses, test_counter, len_train_data, test_batch_size
            )

        DH.save_to_file(train_losses,"Partly_Regularization/train_loss_regularized_full_epochs.pt")
        DH.save_to_file(train_counter,"Partly_Regularization/train_count_regularized_full_epochs.pt")
        DH.save_to_file(test_losses,"Partly_Regularization/test_loss_full_regularized_epochs.pt")
        DH.save_to_file(test_counter,"Partly_Regularization/test_count_full_regularized_epochs.pt")
        

        train_losses = DH.load_file("Partly_Regularization/train_loss_regularized_full_epochs.pt")
        train_counter = DH.load_file("Partly_Regularization/train_count_regularized_full_epochs.pt")
        test_losses = DH.load_file("Partly_Regularization/test_loss_full_regularized_epochs.pt")
        test_counter = DH.load_file("Partly_Regularization/test_count_full_regularized_epochs.pt")

        print("Train losses: {}\nTrain count: {}\nTest losses: {}\nTest count: {}".format(
            len(train_losses), len(train_counter), len(test_losses), len(test_counter)
        ))

        DH.compare_train_test_losses(
            7, train_counter, train_losses, test_counter, test_losses,
            'Partly_Regularization/test_train_loss_regularized_full_epoch.png')

    # ------------------------------------------------------------------------- #
    else:
        print(f"{bcolors.HEADER}Full regularization model \n{bcolors.ENDC}")

        NN_reg = Regularization()
        optimizer = optim.Adam(NN_reg.parameters(), lr=lr)
        
        (test_losses, test_counter) = NN_reg.test_model(
            test_losses, test_counter, 60000, test_batch_size)

        for epoch in range(1, num_epochs+1):
            (train_losses, train_counter, len_train_data) = NN_reg.train_model(
                train_losses, train_counter, optimizer, epoch, train_batch_size, log_interval,
                'Regularized_5_layers_2_dropouts/regularized_model.pth', 'Regularized_5_layers_2_dropouts/regularized_optimizer.pth'
            )

            (test_losses, test_counter) = NN_reg.test_model(
                test_losses, test_counter, len_train_data, test_batch_size
            )
        
        print(f"\n{bcolors.OKBLUE} Saving files...\n{bcolors.ENDC}")

        DH.save_to_file(train_losses, "Regularized_5_layers_2_dropouts/train_losses_regularized.pt")
        DH.save_to_file(train_counter, "Regularized_5_layers_2_dropouts/train_counter_regularized.pt")
        DH.save_to_file(test_losses, "Regularized_5_layers_2_dropouts/test_losses_regularized.pt")
        DH.save_to_file(test_counter, "Regularized_5_layers_2_dropouts/test_counter_regularized.pt")
        
        print(f"\n{bcolors.OKBLUE} Loading files...\n{bcolors.ENDC}")

        train_losses = DH.load_file("Regularized_5_layers_2_dropouts/train_losses_regularized.pt")
        train_counter = DH.load_file("Regularized_5_layers_2_dropouts/train_counter_regularized.pt")
        test_losses = DH.load_file("Regularized_5_layers_2_dropouts/test_losses_regularized.pt")
        test_counter = DH.load_file("Regularized_5_layers_2_dropouts/test_counter_regularized.pt")


        # Continue training for 4 more epochs
        NN_reg_cont = Regularization()
        NN_reg_cont.load_state_dict(DH.load_file("Regularized_5_layers_2_dropouts/regularized_model.pth"))

        optimizer_cont = optim.Adam(NN_reg_cont.parameters(), lr=lr)
        optimizer_cont.load_state_dict(DH.load_file("Regularized_5_layers_2_dropouts/regularized_optimizer.pth"))

        for epoch in range(4, 8):
            test_counter.append(epoch*60000)
            (train_losses, train_counter, len_train_data) = NN_reg_cont.train_model(
                train_losses, train_counter, optimizer_cont, epoch, train_batch_size, log_interval,
                'Regularized_5_layers_2_dropouts/regularized_model.pth', 'Regularized_5_layers_2_dropouts/regularized_optimizer.pth'
            )

            (test_losses, test_counter) = NN_reg_cont.test_model(
                test_losses, test_counter, len_train_data, test_batch_size
            )

        DH.save_to_file(train_losses,"Regularized_5_layers_2_dropouts/train_loss_regularized_full_epochs.pt")
        DH.save_to_file(train_counter,"Regularized_5_layers_2_dropouts/train_count_regularized_full_epochs.pt")
        DH.save_to_file(test_losses,"Regularized_5_layers_2_dropouts/test_loss_full_regularized_epochs.pt")
        DH.save_to_file(test_counter,"Regularized_5_layers_2_dropouts/test_count_full_regularized_epochs.pt")


        DH.compare_train_test_losses(
            7, train_counter, train_losses, test_counter, test_losses,
            'Regularized_5_layers_2_dropouts/test_train_loss_regularized_full_epoch.png')
    # ------------------------------------------------------------------------- #

    print(f"\n{bcolors.OKGREEN} Regularized model finished\n{bcolors.ENDC}")


if __name__ == "__main__":
    main()