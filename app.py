from flask import Flask, render_template, request, jsonify
import os
import torch
from torchvision import transforms
import torchvision.models as models
from PIL import Image

app = Flask(__name__)

num_classes = 2  # Substitua pelo número correto de classes

# Carregar o modelo treinado
model = models.resnet18(pretrained=False)
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(224, 224), stride=(2, 2), padding=(3, 3), bias=False)  # Ajustar para 1 canal
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load('./models/ResNet_best.pth', map_location=torch.device('cpu')))
model.eval()

# Função de pré-processamento da imagem
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor

# Função de inferência
def predict_cancer(image_path):
    input_tensor = preprocess_image(image_path)
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        predicted_class = torch.argmax(probabilities).item()
    return predicted_class, probabilities.numpy().tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    image_path = os.path.join('./uploads', file.filename)
    file.save(image_path)

    predicted_class, probabilities = predict_cancer(image_path)
    os.remove(image_path)  # Remover a imagem após a previsão

    return jsonify({'class': predicted_class, 'probabilities': probabilities})

from flask import Flask, render_template, request, jsonify, send_from_directory

# Adicione esta rota para exibir a página de resultados
@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/<model>_metrics.csv')
def download_metrics_file(model):
    return send_from_directory('./', f'{model}_metrics.csv', as_attachment=True)



if __name__ == '__main__':
    app.run(debug=True)
