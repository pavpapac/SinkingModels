import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ETL import ETL
import matplotlib.pyplot as plt


class DeepModels:

    def data_prep(self, X_train, y_train, X_test, y_test):
        # First we want to convert oru dataset to Torch tensors
        train_inputs = torch.tensor(X_train).float()
        train_targets = torch.tensor(y_train).float()
        test_inputs = torch.tensor(X_test).float()
        test_targets = torch.tensor(y_test).float()
        Din = X_train.shape[1]
        Dout = 1

        return train_inputs, train_targets, test_inputs, test_targets, Din, Dout

    def SimpleNet(self, Din, Dout, num_hidden_layers=3, num_nodes=256):
        # Define a simple neural network. Main parameters to determine are:
        # number of hidden layers (1,2,3), number of nodes per hidden layer (512 or 256 or 128)

        if num_hidden_layers == 1:

            model = nn.Sequential(
                nn.Linear(Din, num_nodes),
                nn.ReLU(),
                nn.Linear(num_nodes, Dout)
            )
        elif num_hidden_layers == 2:

            model = nn.Sequential(
                nn.Linear(Din, num_nodes),
                nn.ReLU(),
                nn.Linear(num_nodes, num_nodes),
                nn.Linear(num_nodes, Dout)
            )
        elif num_hidden_layers == 3:

            model = nn.Sequential(
                nn.Linear(Din, num_nodes),
                nn.ReLU(),
                nn.Linear(num_nodes, num_nodes),
                nn.Linear(num_nodes, num_nodes),
                nn.Linear(num_nodes, Dout)
            )

        return model

    def optimization_params(self, model, optimizer_name='SGD', lr=0.01, momentum=0.95):
        # Define the criterion to calculate the loss. In this case we will use binary cross entropy
        criterion = nn.BCEWithLogitsLoss()
        if optimizer_name == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=0.01)

        return optimizer, criterion

    def train_net(self, model, optimizer, criterion, train_inputs, train_targets, test_inputs, test_targets, n_epochs=100):
        train_losses = []
        test_losses = []
        test_accuracies = []
        for it in range(n_epochs):
            # zero the gradients
            model.zero_grad()

            # Pass the inputs in the model
            train_outputs = model(train_inputs)

            # Calculate the loss value relative to the targets
            train_loss = criterion(train_outputs.flatten(), train_targets.flatten())
            train_losses.append(train_loss.item())

            # Calculate also the accuracy on test set per epoch
            test_acc = self._accuracy(test_inputs, test_targets.flatten())
            test_accuracies.append(test_acc)

            # Now calculate the gradient and apply the step
            train_loss.backward()
            optimizer.step()

        return train_losses, test_accuracies

    def _accuracy(self, test_inputs, test_targets):
        # Function to calculate the accuracy between predictions and test dataset
        test_outputs = model(test_inputs)
        sig = nn.Sigmoid()
        test_predictions = sig(test_outputs)
        test_predictions_np = test_predictions.flatten().detach().numpy()
        test_targets_np = test_targets.detach().numpy()
        acc = np.mean((test_predictions_np > 0.5) == test_targets_np)

        return acc


if __name__ == "__main__":
    etl = ETL(path='./Data/titanic/train.csv')
    etl.import_data_df()
    list_cat = ['Sex', 'Embarked']
    list_num = ['Age', 'SibSp', 'Pclass', 'Parch', 'Fare']
    target_label = ['Survived']
    etl.preprocess_pipeline(list_cat, list_num, target_label)
    dm = DeepModels()
    train_inputs, train_targets, test_inputs, test_targets, Din, Dout = dm.data_prep(etl.X_train, etl.y_train, etl.X_test, etl.y_test)
    model = dm.SimpleNet(Din, Dout, num_hidden_layers=3, num_nodes=512)
    optimizer, criterion = dm.optimization_params(model, optimizer_name='Adam', lr=0.001, momentum=0.99)
    train_losses, test_accuracies = dm.train_net(model, optimizer, criterion, train_inputs, train_targets, test_inputs,
                                              test_targets, n_epochs=1000)


    plt.plot(test_accuracies)
    plt.show()
