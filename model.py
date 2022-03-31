# import the modules
from torch import nn, optim
import torch
from torchvision import transforms
from dataloader import ImageDataLoader
from torch.utils.data import DataLoader
from Lenet5 import LeNet5
from train import Training
import neptune.new as neptune
from helper_plots import Metrics
import matplotlib.pyplot as plt


# Initialing the neptune logger.
run = neptune.init(project='sathwik-panchangam/pytorch-deep-learning',api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3M2NjMWM4NC02OWMyLTQzMmQtYmYxMC01MmM1NjAyMGRhMjIifQ==",
)
# Creating a dictionary for the parameters.
parameters = { 'lr': 1e-2,
               'batch_size': 64,
               'num_classes': 6,
               'num_epochs': 30,
               'model_name':'LeNet5',
               'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")}

# Logging hyperparameters.
run['config/hyperparameters'] = parameters

# Creating a transform for preprocessing
transform = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize((32,32)),
                                transforms.RandomHorizontalFlip(p=0.5), 
                                transforms.ToTensor(),
                                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])

# Path for the dataset
root_dir = '/home/sathwik/Pytorch_Deep_Learning/Datasets/Chess/Chessman_image_dataset/Chess'
# root_dir = '/home/sathwik/Pytorch_Deep_Learning/Datasets/Animals_small/dataset/dataset'
 
# Loading the training dataset.
trainset = ImageDataLoader(root=root_dir, train=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=20, shuffle=True)

# Loading the testing dataset.
testset = ImageDataLoader(root=root_dir, test=True, transform=transform)
testloader = DataLoader(testset, batch_size=20, shuffle=True)


dataset_size = {'train':len(trainset), 'test':len(testset)}

# Logging the data and transformations.
run['config/dataset/path'] = root_dir
run['config/dataset/transforms'] = transform
run['config/dataset/size'] = dataset_size


# Creating a model for the network
model = LeNet5(num_classes=6)
print(model)
# Creating a loss function
criterion = nn.CrossEntropyLoss()
# Creating an optimizer
optimizer = torch.optim.SGD(model.parameters(), parameters['lr'],momentum=0.9)


# Logging the model parameters and model architecture.
run['config/model'] = type(model).__name__
run['config/criterion'] = type(criterion).__name__
run['config/optimizer'] = type(optimizer).__name__


fname = parameters['model_name']
with open(f"./{fname}_arch.txt", "w") as f:
    f.write(str(model))

torch.save(model.state_dict(), f"./{fname}_model.pth")

run[f"io_files/{parameters['model_name']}_arch"].upload(f"./{parameters['model_name']}_arch.txt")
run[f"io_files/{parameters['model_name']}_model"].upload(f"./{parameters['model_name']}_model.pth")

# Start training the model.
train = Training(num_epochs=parameters['num_epochs'],model=model,criterion=criterion,optimizer=optimizer,trainloader=trainloader,testloader=testloader,logger=run)
training = train.train()
labels = training[2]
targets = training[3]
train_loss = training[0]
test_loss = training[1]
classes = trainset.get_classes()

print("Training Ended !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


# METRICS
metrics = Metrics(targets,labels,classes)

# Accuracy
accuracy = metrics.get_accuracy()
print("Accuracy : ",accuracy)

# Precision
precision = metrics.get_precision_score()
print("Precision : ",precision)

# Confusion Matrix
conf_mat1 = metrics.plot_confusion_matrix1()
conf_mat2 = metrics.plot_confusion_matrix2()

# Classification Report.
class_report = metrics.get_classification_report()
print(class_report)

run.stop()