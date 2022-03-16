# import the modules
from torch import nn, optim
import torch
from torchvision import transforms
from dataloader import ImageDataLoader
from torch.utils.data import DataLoader
from torchvision.models import GoogLeNet, mobilenet, squeezenet
from Lenet import LeNet
from train import Training
import neptune.new as neptune


# Initialing the neptune logger.
# run = neptune.init(
#     project="sathwik-panchangam/pytorch-deep-learning",
#     api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3M2NjMWM4NC02OWMyLTQzMmQtYmYxMC01MmM1NjAyMGRhMjIifQ==",
# )


# Creating a transform for preprocessing
transform = transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])

# Path for the dataset
root_dir = '/home/sathwik/PyTorch-Deep-Learning/Datasets/Animals_small/dataset/dataset'

# Loading the training dataset.
trainset = ImageDataLoader(root=root_dir, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=20, shuffle=True)

# Loading the testing dataset.
testset = ImageDataLoader(root=root_dir, test=True, transform=transform)
testloader = DataLoader(testset, batch_size=20, shuffle=True)


# Creating a model for the network
model = LeNet()
print(model)
# Creating a loss function
criterion = nn.CrossEntropyLoss()
# Creating an optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Start training the model.
training = Training(num_epochs=10,model=model,criterion=criterion,optimizer=optimizer,trainloader=trainloader,testloader=testloader)
print("Training Ended !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")





