from PIL import Image
import torch
from torchvision import datasets, transforms
from torchvision import transforms as transform
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os

import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import csv
import random


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
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
       
        self.files = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                try:
                    # Tentar abrir a imagem
                    with Image.open(file_path) as img:
                        # Verificar se é um arquivo .jpg e se é um arquivo regular
                        if file.lower().endswith('.jpg') and os.path.isfile(file_path):
                            self.files.append(file_path)
                except Exception as e:
                    print(f"Erro ao abrir {file_path}: {str(e)}")
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


data_root = "./databases/mammotherm_databse/"  

#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
full_dataset = CustomDataset(data_root, transform=transform)

# dataset = CustomDataset(data_root, transform=transform)
# imageloader = ImageFolder(root=data_root, transform=transform)
# dataloader = torch.utils.data.DataLoader(imageloader, batch_size=32, shuffle=True)
# num_classes = len(dataset.classes)

# Divisão do conjunto de dados
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Dataloaders para treino e teste
train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

num_classes = len(full_dataset.classes)


def compute_metrics(true, pred, num_classes):

    true_tensor = torch.tensor(true)
    pred_tensor = torch.tensor(pred)

    true_onehot = nn.functional.one_hot(true_tensor, num_classes)
    pred_onehot = nn.functional.one_hot(pred_tensor, num_classes)

    accuracy = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro')
    precision = precision_score(true, pred, average='macro')
    recall = recall_score(true, pred, average='macro')

    confusion_mat = confusion_matrix(true, pred, labels=list(range(num_classes)))

    tn, fp, fn, tp = confusion_mat[0, 0], confusion_mat[0, 1], confusion_mat[1, 0], confusion_mat[1, 1]
   # tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
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
   "Efficientnet": efficientnet,
   "ResNet": resnet,
   "Alexnet": alexnet,
   "VGG16": vgg16
}
 
all_metrics = {}
num_epochs = 50 
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
        
        for inputs, labels in train_dataloader:
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
 

        epoch_loss = running_loss / len(train_dataloader)
        true_labels, pred_labels = [], []
        for inputs, labels in train_dataloader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
 
        epoch_metrics = compute_metrics(true_labels, pred_labels, num_classes)
        print(epoch_metrics)
        model_metrics.append((epoch, epoch_loss) + epoch_metrics)
 
    torch.save(model.state_dict(), f'./models/mammotherm/{model_name}_last.pth')
    all_metrics[model_name] = model_metrics
 
    save_metrics_to_csv(model_metrics, f'./methrics/mammotherm/{model_name}_metrics.csv')
 
    plot_metrics(model_metrics, f'./methrics/mammotherm/{model_name} Metrics')
    
      # Avaliação no conjunto de teste
    model.eval()
    true_labels, pred_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_dataloader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(predicted.cpu().numpy())

        # Calcular métricas no conjunto de teste
    epoch_metrics_test = compute_metrics(true_labels, pred_labels, num_classes)
    print(epoch_metrics_test)
    model_metrics_test = [(epoch, epoch_loss) + epoch_metrics_test]

    torch.save(model.state_dict(), f'./models/mammotherm/tests/{model_name}_last_test.pth')
    all_metrics[model_name + "_test"] = model_metrics_test

    save_metrics_to_csv(model_metrics_test, f'./methrics/mammotherm/tests/{model_name}_metrics_test.csv')

    plot_metrics(model_metrics_test, f'./methrics/mammotherm/tests/{model_name} Metrics Test')