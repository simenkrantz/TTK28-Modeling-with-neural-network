# Standard libraries
import torch
import collections
import torchvision as tv
import torch.optim as optim


# Personal libraries
from datahandler import DataHandler, bcolors


class FullNeuralNetwork(torch.nn.Module):
    """The full neural network model, used for comparison.
    
    Two 2D convolutional layers are folellowed by two linear, fully-connected layers.
    No regularization is used, this is applied in a different model.
    Activation function - rectified linear units (ReLUs).
    Credited: https://nextjournal.com/gkoehler/pytorch-mnist, accessed 2020-09-24

    Inherits from torch.nn.Module, https://www.educba.com/torch-dot-nn-module/
    """
    def __init__(self):
        super(FullNeuralNetwork, self).__init__()
        self.convolution_layer_1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.convolution_layer_2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.linear_layer_1 = torch.nn.Linear(320, 50)
        self.linear_layer_2 = torch.nn.Linear(50, 10)

    def ReLU(self, input):
        return torch.nn.functional.relu(input)

    def maxpool2D(self, unit, input):
        return torch.nn.functional.max_pool2d(unit, input)

    def forward(self, input):
        tensor = self.ReLU(self.maxpool2D(self.convolution_layer_1(input), 2))
        tensor = self.ReLU(self.maxpool2D(self.convolution_layer_2(tensor), 2))

        # Reshape tensor to 320 columns and an appropriate number of rows
        tensor = tensor.view(-1, 320)

        tensor = self.ReLU(self.linear_layer_1(tensor))
        return torch.nn.functional.log_softmax(self.linear_layer_2(tensor))

    def train_model(
        self,
        train_losses,
        train_counter,
        optimizer,
        learning_rate,
        momentum,
        epoch,
        batch_size,
        log_interval):
        """train_model -> [train_losses, train_counter, len_train_set]
        Training of the full NN model on MNIST dataset.
        Normalize parameters:
            0.1307 (Global mean of MNIST dataset)
            0.3081 (Global standard deviation of MNIST dataset)

        Optimizer: Stochastic gradient descent
        """
        DH = DataHandler()
        train_loader = torch.utils.data.DataLoader(
            tv.datasets.MNIST("./", train=True, download=False,
                transform=tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=batch_size, shuffle=True)
        len_dataset = len(train_loader.dataset)

        self.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = torch.nn.functional.nll_loss(self(data), target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print("Epoch {0}: {1:.1f}%\tLoss: {2:.5f}".format(
                    epoch, 100.*batch_idx/len(train_loader), loss.item()))
                train_losses.append(loss.item())
                train_counter.append((batch_idx*64) + ((epoch-1)*len_dataset))
                DH.save_to_file(self.state_dict(), 'full_model.pth')
                DH.save_to_file(optimizer.state_dict(), 'full_optimizer.pth')

        return (train_losses, train_counter, len_dataset)


    def test_model(
        self,
        test_losses,
        test_counter,
        len_train_set,
        num_epochs,
        batch_size):
        """test_model -> [test_losses, test_counter]
        """
        test_loader = torch.utils.data.DataLoader(
            tv.datasets.MNIST("./", train=False, download=False,
                transform=tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.1307,), (0.3081,))])),
            batch_size=batch_size, shuffle=True)
        len_dataset = len(test_loader.dataset)
        loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in test_loader:
                out = self(data)
                loss += torch.nn.functional.nll_loss(out,target, size_average=False).item()
                pred = out.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        loss /= len_dataset
        test_losses.append(loss)

        print("\nTest set average loss {:.5f}\tAccuracy {}/{} ({:.1f}%)\n".format(
            loss, correct, len_dataset, 100.*correct/len_dataset))

        return (test_losses, test_counter)


def run_few_epochs(NN, DH, num_epochs):
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    train_losses = []
    train_counter = []
    test_losses = []
    # 60000 being len(train_loader.dataset)
    test_counter = [i*60000 for i in range(num_epochs+1)]
    
    optimizer = optim.SGD(NN.parameters(),lr=learning_rate,momentum=momentum)

    (test_losses, test_counter) = NN.test_model(
            test_losses, test_counter, 60000, num_epochs, batch_size_test)

    for epoch in range(1, num_epochs+1):
        (train_losses, train_counter, len_train_data) = NN.train_model(
            train_losses, train_counter, optimizer, learning_rate, momentum, epoch, batch_size_train, log_interval)
        
        (test_losses, test_counter) = NN.test_model(
            test_losses, test_counter, len_train_data, num_epochs, batch_size_test)

    DH.save_to_file(train_losses, 'train_loss_full_model.pt')
    DH.save_to_file(train_counter, 'train_count_full_model.pt')
    DH.save_to_file(test_losses, 'test_loss_full_model.pt')
    DH.save_to_file(test_counter, 'test_count_full_model.pt')
    
    DH.compare_train_test_losses(
        num_epochs, train_counter, train_losses, test_counter, test_losses,'test_train_loss_full_model.png')

    print("\nFinished running few epochs\n\n")


def main():
    torch.backends.cudnn.enabled = False

    # Manual seed for repeatability
    ran_seed = 96554
    torch.manual_seed(ran_seed)

    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    #test_loader = torch.utils.data.DataLoader(
                tv.datasets.MNIST("./", train=False, download=False,
                    transform=tv.transforms.Compose([
                    tv.transforms.ToTensor(),
                    tv.transforms.Normalize((0.1307,), (0.3081,))])),
                batch_size=1000, shuffle=True)

    DH = DataHandler()
    #NN = FullNeuralNetwork()

    #run_few_epochs(NN, DH, num_epochs = 3)

    #DH.compare_predict(NN, test_loader, "full_model.pth","compare_predict_3_epochs.png")

    train_losses = DH.load_file("train_loss_full_model.pt")
    train_counter = DH.load_file("train_count_full_model.pt")
    test_losses = DH.load_file("test_loss_full_model.pt")
    test_counter = DH.load_file("test_count_full_model.pt")

    # Continue training for 4 more epochs
    NN_cont = FullNeuralNetwork()
    NN_cont.load_state_dict(DH.load_file("full_model.pth"))
    optimizer_cont = optim.SGD(NN_cont.parameters(), lr=learning_rate, momentum=momentum)
    optimizer_cont.load_state_dict(DH.load_file("full_optimizer.pth"))

    for epoch in range(4, 8):
        test_counter.append(epoch*60000)
        (train_losses, train_counter, len_train_data) = NN_cont.train_model(
            train_losses,train_counter,optimizer_cont,learning_rate,momentum,epoch,batch_size_train,log_interval)
        (test_losses, test_counter) = NN_cont.test_model(
            test_losses,test_counter,len_train_data,7,batch_size_test)

    DH.save_to_file(train_losses,"train_loss_full_model_full_epochs.pt")
    DH.save_to_file(train_counter,"train_count_full_model_full_epochs.pt")
    DH.save_to_file(test_losses,"test_loss_full_model_full_epochs.pt")
    DH.save_to_file(test_counter,"test_count_full_model_full_epochs.pt")

    DH.compare_train_test_losses(
        7,train_counter,train_losses,test_counter,test_losses,"test_train_loss_full_model_full_epoch.png")

    print(f"{bcolors.OKGREEN}\nMain finished !\n{bcolors.ENDC}")

if __name__ == "__main__":
    main()