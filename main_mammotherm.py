from PIL import Image
import torch
from torchvision import datasets, transforms
from torchvision import transforms as transform
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import csv


import torchvision.models as models
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

import numpy as np
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

#from efficientnet_pytorch import EfficientNet

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.files = [os.path.join(root_dir, class_name, file) for class_name in self.classes for file in os.listdir(os.path.join(root_dir, class_name))]

    def __len__(self):
        return len(self.files)
 
    def __getitem__(self, idx):
        image_path = self.files[idx]
        image = Image.open(image_path).convert('L')  # Convertendo para tons de cinza
        label = self.classes.index(os.path.basename(os.path.dirname(image_path)))

        if self.transform:
            image = self.transform(image)
 
        return image, label

transform = transforms.Compose([
    transform.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(1),
    transforms.RandomRotation(10),
    transforms.Normalize((0.5,), (0.5,))
])


data_root = "./databases/mammotherm_database"  
dataset = CustomDataset(data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
num_classes = len(dataset.classes)

def compute_metrics(true, pred, num_classes):

    true_tensor = torch.tensor(true)
    pred_tensor = torch.tensor(pred)

    true_onehot = nn.functional.one_hot(true_tensor, num_classes)
    pred_onehot = nn.functional.one_hot(pred_tensor, num_classes)

    accuracy = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro')
    precision = precision_score(true, pred, average='macro')
    recall = recall_score(true, pred, average='macro')

    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    
    auc = roc_auc_score(true_onehot, pred_onehot, average='macro', multi_class='ovo')
    
    return accuracy, f1, precision, recall, sensitivity, specificity, auc

def save_metrics_to_csv(metrics, filename):
    with open(filename, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Loss', 'Accuracy', 'F1', 'Precision', 'Recall', 'Sensitivity', 'Specificity', 'AUC'])
        writer.writerows(metrics)



def plot_metrics(metrics, title):
    epochs = range(1, len(metrics) + 1)
    
    plt.figure(figsize=(12,8))
    plt.plot(epochs, [m[1] for m in metrics], label='Loss')
    plt.plot(epochs, [m[2] for m in metrics], label='Accuracy')
    plt.plot(epochs, [m[3] for m in metrics], label='F1')
    plt.plot(epochs, [m[4] for m in metrics], label='Precision')
    plt.plot(epochs, [m[5] for m in metrics], label='Recall')
    plt.plot(epochs, [m[6] for m in metrics], label='Sensitivity')
    plt.plot(epochs, [m[7] for m in metrics], label='Specificity')
    plt.plot(epochs, [m[8] for m in metrics], label='AUC')
    
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.savefig(title + '.png')
    plt.show()


# MobileNet
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=(224, 224), stride=(2, 2), padding=(1, 1), bias=False) 
mobilenet.classifier[1] = nn.Linear(1280, num_classes)
 
# SqueezeNet
squeezenet = models.squeezenet1_0(pretrained=True)
squeezenet.features[0] = nn.Conv2d(1, 32, kernel_size=(224, 224), stride=(2, 2))  
squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

# VGG16
vgg16 = models.vgg16(pretrained=True)
vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)) 
vgg16.classifier[6] = nn.Linear(4096, num_classes)
 
# ResNet
resnet = models.resnet18(pretrained=True)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(224, 224), stride=(2, 2), padding=(3, 3), bias=False)  
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
 
efficientnet = EfficientNet.from_pretrained('efficientnet-b0')
efficientnet._conv_stem = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
efficientnet._fc = nn.Linear(efficientnet._fc.in_features, num_classes)

# Criação do modelo DenseNet
densenet = models.densenet121(pretrained=True)
densenet.features.conv0 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)


inception = models.inception_v3(pretrained=True)
print(models.inception_v3.__doc__)
inception.Conv2d_1a_3x3 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
inception.fc = nn.Linear(inception.fc.in_features, num_classes)

alexnet = models.alexnet(pretrained=True)
alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
alexnet.classifier[6] = nn.Linear(4096, num_classes)

models_dict = {
   #"SqueezeNet": squeezenet,
   # "densenet": densenet,
   #"efficientnet": efficientnet,
   #"Inception": inception,
   "Alexnet": alexnet,
   "VGG16": vgg16 ,
    # "MobileNet": mobilenet,
   "ResNet": resnet
    
}
 
all_metrics = {}
num_epochs = 2 
criterion = nn.CrossEntropyLoss()

for model_name, model in models_dict.items():
    print(f"Treinando {model_name}...")
 
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_acc = 0.0
    model_metrics = []
 
    for epoch in tqdm(range(num_epochs)):  
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
 
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
 
        epoch_acc = 100 * correct / total
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), f'./models/{model_name}_best.pth')
 

        epoch_loss = running_loss / len(dataloader)
        true_labels, pred_labels = [], []
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
 
        epoch_metrics = compute_metrics(true_labels, pred_labels, num_classes)
        print(epoch_metrics)
        model_metrics.append((epoch, epoch_loss) + epoch_metrics)
 
    torch.save(model.state_dict(), f'./models/{model_name}_last.pth')
    all_metrics[model_name] = model_metrics
 
    save_metrics_to_csv(model_metrics, f'{model_name}_metrics.csv')
 
    plot_metrics(model_metrics, f'{model_name} Metrics')
