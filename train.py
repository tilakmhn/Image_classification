
import torch as torch
import numpy as np
import pandas as pds
import torchvision as tv
from torchvision import transforms
from torchvision import datasets
from torchvision import models
from collections import OrderedDict
from torch import nn
from torch import optim
import torch.nn.functional as F
import json
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import utility

def classify_images(model, trainloader, validLoader,  epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0
    print("Start classify_images")

    # change to cuda
    model.to(device)
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloader:
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            #Calculate loss
            loss = criterion(outputs, labels)
            #Do backward pass
            loss.backward()
            #Optimize waits
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every))

                running_loss = 0

               # Print result on validation data set
                for inputs, labels in validLoader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    model.eval() 
                    # Forward and backward passes
                    outputs = model.forward(inputs)
                    #Calculate loss
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                print("Validation Epoch: {}/{}... ".format(e+1, epochs),
                              "Loss: {:.4f}".format(running_loss/print_every))
                running_loss = 0

    print("Completed classify_images")

def validate_accuracy_on_test_dataset(model, testloader, device='cpu'):
    correct = 0
    total = 0
    print("start")
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network : %d %%' % (100 * correct / total))


def create_data_loaders(train_dir, valid_dir, test_dir):
    data_transforms = {"train": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),

        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),

       "validation": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])}

    # TODO: Load the datasets with ImageFolder
    image_datasets = {"train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
                      "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
                      "validation": datasets.ImageFolder(valid_dir, transform=data_transforms["validation"])}

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=64, shuffle=True),
                   "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=64),
                   "validation": torch.utils.data.DataLoader(image_datasets["validation"], batch_size=64)
                   }
    return dataloaders, image_datasets

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Neural Network')
    parser.add_argument('data_directory', metavar='save_dir', type=str, help='data directory')
    parser.add_argument('--save_dir', metavar='data_dir',  type=str, help='Save Directory')
    parser.add_argument('--arch', metavar='arch',  type=str, help='Model Name')
    parser.add_argument('--learning_rate', metavar='learning rate',  type=float, help='Learning Rate')
    parser.add_argument('--hidden_units', metavar='hidden units', nargs='+', type=int, help='Hidden Units')
    parser.add_argument('--epochs', metavar='epochs',  type=int, help='Epoch')
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()

    data_directory = args.data_directory
    save_dir = args.save_dir
    model_name = args.arch
    learning_rate = args.learning_rate
    hidden_units = args.hidden_units
    epochs = args.epochs
    gpu = args.gpu

    # Set default values

    if learning_rate is None:
        learning_rate = 0.001
    if epochs is None:
        epochs = 3
    if save_dir is None:
        save_dir = "checkpoint/checkpoint.pth"
    if model_name is None:
        model_name = 'vgg19'
    if hidden_units is None:
        hidden_units = [4096]
    return data_directory, save_dir, model_name, learning_rate, hidden_units, epochs, gpu

def save_checkpoint(model, dataset, epochs, input_size, hidden_layer, output_size, learning_rate, dropout, checkpoint_path, model_name):
    checkpoint = {"epochs": epochs,
                 "state_dict": model.state_dict(),
                 "class_to_idx":dataset['train'].class_to_idx,
                 "input_size": input_size,
                 "hidden_layer": hidden_layer,
                 "output_size": output_size,
                 "learning_rate": learning_rate,
                  "model_name": model_name,
                 "dropout": dropout}
    torch.save(checkpoint,checkpoint_path)

if __name__ == '__main__':

    data_directory, save_dir, model_name, learning_rate, hidden_units, epochs, gpu = parse_arguments()

    train_dir = data_directory + '/train'
    valid_dir = data_directory + '/valid'
    test_dir = data_directory + '/test'

    dataloaders, dataset = create_data_loaders(train_dir, valid_dir, test_dir)
    model = None
    output_size = 102
    dropout = 0.2
    if model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        input_size = 25088
    else:
        model = models.vgg13(pretrained=True)
        input_size = 25088

    for param in model.parameters():
        param.requires_grad = False

    classifier = utility.buildClassifier(input_size, output_size, hidden_units, dropout)
    model.classifier = classifier
    device = 'cuda' if gpu else 'cpu'

    classify_images(model, dataloaders["train"], dataloaders["validation"], epochs, 40, nn.NLLLoss(), optim.Adam(model.classifier.parameters(), lr=learning_rate), device)
    validate_accuracy_on_test_dataset(model, dataloaders["test"], device)
    save_checkpoint(model, dataset, epochs, input_size, hidden_units, output_size, learning_rate, dropout, save_dir, model_name)
