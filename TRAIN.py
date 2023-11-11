import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
 
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
 
# Transformações: redimensionamento, conversão para tensor e normalização
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
 
# Cria uma instância do DataLoader
data_root = "path_to_data_folder"  # substituir isso pelo caminho da sua pasta de dados
dataset = CustomDataset(data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
num_classes = len(dataset.classes)

import torchvision.models as models
import torch.nn as nn

# MobileNet
mobilenet = models.mobilenet_v2(pretrained=True)
mobilenet.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)  # Adaptar para 1 canal
mobilenet.classifier[1] = nn.Linear(1280, num_classes)
 
# SqueezeNet
squeezenet = models.squeezenet1_0(pretrained=True)
squeezenet.features[0] = nn.Conv2d(1, 96, kernel_size=(7, 7), stride=(2, 2))  # Adaptar para 1 canal
squeezenet.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))

# VGG16
vgg16 = models.vgg16(pretrained=True)
vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # Adaptar para 1 canal
vgg16.classifier[6] = nn.Linear(4096, num_classes)
 
# ResNet
resnet = models.resnet18(pretrained=True)
resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # Adaptar para 1 canal
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)
 
import torch.optim as optim
from tqdm import tqdm
 
# Modelos a serem treinados
models_dict = {
    "VGG16": vgg16,
    "ResNet": resnet,
    "MobileNet": mobilenet,
    "SqueezeNet": squeezenet
}
 
# Lista para salvar as métricas de todos os modelos
all_metrics = {}
 
for model_name, model in models_dict.items():
    print(f"Treinando {model_name}...")
 
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    best_acc = 0.0
    model_metrics = []
 
    for epoch in tqdm(range(num_epochs)):  # Barra de progresso
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
 
        # Métricas para o CSV e gráficos
        epoch_loss = running_loss / len(dataloader)
        true_labels, pred_labels = [], []
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(predicted.cpu().numpy())
 
        epoch_metrics = compute_metrics(true_labels, pred_labels, num_classes)
        model_metrics.append((epoch, epoch_loss) + epoch_metrics)
 
    # Salvar o último modelo
    torch.save(model.state_dict(), f'./models/{model_name}_last.pth')
    all_metrics[model_name] = model_metrics
 
    # Salvar métricas no CSV
    save_metrics_to_csv(model_metrics, f'{model_name}_metrics.csv')
 
    # Gerar gráficos
    plot_metrics(model_metrics, f'{model_name} Metrics')