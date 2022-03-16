# import the modules
import torch
from torch import nn, optim


# Creating a training class
class Training():
    def __init__(self, num_epochs = None, model=None,criterion = None,optimizer = None, trainloader=None,testloader=None):

        self.num_epochs = num_epochs
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

        self.train_losses = []
        self.test_losses = []

        for epoch in range(self.num_epochs):
            total_train_loss = 0
            for images , labels in trainloader:
                self.optimizer.zero_grad()
                logits = self.model(images)
                loss = self.criterion(logits, labels)

                total_train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            else:
                total_test_loss = 0
                num_correct_predictions = 0

                with torch.no_grad():
                    for images, labels in testloader:
                        logits = self.model(images)
                        loss = self.criterion(logits, labels)
                        total_test_loss += loss.item()

                        probabilities = torch.exp(logits)
                        top_p, top_class = probabilities.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        num_correct_predictions += equals.sum().item()
                
                # Calculating the mean loss to enable comparison between training and test sets.
                train_loss = total_train_loss/len(trainloader.dataset)
                test_loss = total_test_loss/len(testloader.dataset)
                # Creating list of train and test losses
                self.train_losses.append(train_loss)
                self.test_losses.append(test_loss)


                # Printing the results
                print("Epoch: {}/{}... ".format(epoch+1,self.num_epochs),
                "Training Loss: {:.3f}... ".format(train_loss),
                "Test Loss: {: .3f}... ".format(test_loss),
                "Test Accuracy: {: .3f}".format(num_correct_predictions/len(testloader.dataset)))