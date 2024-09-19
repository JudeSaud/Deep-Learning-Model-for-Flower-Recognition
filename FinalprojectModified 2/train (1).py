#I'm having trouble with this workspace
import argparse
import time
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torchvision import transforms
import time
from PIL import Image
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json

parser = argparse.ArgumentParser(description='Train a neural network using a specified dataset')
parser.add_argument('data_directory', help='Path to the dataset used for training the neural network')
parser.add_argument('--save_dir', help='Path to the directory where the checkpoint should be saved')
parser.add_argument('--arch', default='vgg16', help='Network architecture to use (default: \'vgg16\')')
parser.add_argument('--learning_rate', type=float, help='Learning rate for the training process')
parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the network')
parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
parser.add_argument('--gpu', action='store_true', help='Use GPU for training')

args = parser.parse_args()




print("----loading the data----")

data_dir  = args_in.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir  = data_dir + '/test'

# TODO: Define your transforms for the training, validation, and testing sets

train_transforms = transforms.Compose([transforms.RandomRotation(50),
                                              transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.485, 0.456, 0.406],
                                                                   [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406],
                                                                  [0.229, 0.224, 0.225])])


        
train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)    

# TODO: Using the image datasets and the trainforms, define the dataloaders
train_loader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle = True)
valid_loader = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)
test_loader  = torch.utils.data.DataLoader(test_datasets,  batch_size = 64)



with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
print("----done with data loading ----") 

print("---- building the model ----")

layers        = args_in.hidden_units
learning_rate = args_in.learning_rate

if args_in.arch =='vgg16':
    model = models.vgg16(pretrained = True)
    # Freezing the parameters 
    for param in model.parameters():
        param.requires_grad = False
        
 classifier = nn.Sequential(
          nn.Linear(25088, 512),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(256, 102),
          nn.LogSoftmax(dim = 1)
        )
else:
    raise ValueError('Model arch error.')
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.0025)

model.to(device)

print("----Finished building the network----")
    
print("------ training the model ----------------")

epochs = args_in.epochs
steps = 0
print_every = 10
training_loss = 0

start = time.time()
print('Start of training')

for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1

        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        training_loss += loss.item()

        if steps % print_every == 0 or steps == len(train_loader):
            valid_loss = 0
            valid_accuracy = 0

            model.eval()

            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    # Calculate validation accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}, "
                  f"Train loss: {training_loss/(print_every if steps % print_every == 0 else steps % print_every):.3f}, "
                  f"Valid loss: {valid_loss/len(valid_loader):.3f}, "
                  f"Valid accuracy: {valid_accuracy/len(valid_loader):.3f}")

            training_loss = 0

            model.train()

time_elapsed = time.time() - start
print("\nTime spent training: {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60)

print("----- Finished model training -----")
      
print("----- Testing the model ----")
      
start = time.time()
print('Start of testing')

#Validation on the test set
test_loss = 0
test_accuracy = 0
model.eval()  


with torch.no_grad():
    for batch_idx, (inputs, labels) in enumerate(test_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()

        # Calculate the accuracy of test set
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

        # Print progress
        print(f"Validation Batch [{batch_idx}/{len(test_loader)}], "
              f"Loss: {batch_loss.item():.3f}, "
              f"Accuracy: {torch.mean(equals.type(torch.FloatTensor)).item():.3f}")

print(f"Test loss: {test_loss / len(test_loader):.3f}, "
      f"Test accuracy: {test_accuracy / len(test_loader):.3f}")
running_loss = 0

time_elapsed = time.time() - start
print("\nTime spent on testing: {:.0f}m {:.0f}s".format(time_elapsed//60, time_elapsed % 60)

print("----- Finished model testing -----")
      
print("----- Saving the checkpoint -----") 
checkpoint = {
    'epochs': epochs,
    'learning_rate': 0.0025,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion_state_dict': criterion.state_dict(),
    'class_to_idx': train_datasets.class_to_idx
}

torch.save(checkpoint, 'checkpoint.pth')      