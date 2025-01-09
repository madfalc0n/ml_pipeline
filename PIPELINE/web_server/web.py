from flask import Flask, render_template_string
from torchvision import datasets, transforms
import torch
import numpy as np
import base64
import io
from PIL import Image
from collections import Counter

# Flask 앱 생성
app = Flask(__name__)

# MNIST 데이터 로드
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

# 학습된 모델 로드
def model_load():
    model = torch.jit.load('D:/02.작업물/02.LAB/03.2025/web/model_scripted.pt')
    model.eval()
    return model

model = model_load()

# HTML 템플릿
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>MNIST Viewer</title>
    <script>
        setInterval(function() {
            window.location.reload();
        }, 2000);
    </script>
    <style>
        table {
            width: 100%;
            border-collapse: collapse;
        }
        td {
            text-align: center;
            padding: 10px;
        }
        img {
            width: 100px;
            height: 100px;
        }
    </style>
</head>
<body>
    <h1>MNIST Data Viewer with Prediction, Total ACC: {{ total_acc }}%</h1>

    <table border="1">
        <tr>
        {% for item in data %}
            <td>
                <img src="data:image/png;base64,{{ item.image_data }}" alt="MNIST Image">
                <p>True: {{ item.label }}</p>
                <p>Pred: {{ item.prediction }}</p>
            </td>
            {% if loop.index % 5 == 0 %}
        </tr><tr>
            {% endif %}
        {% endfor %}
        </tr>
    </table>
    <table border="1">
        <tr>
        {% for item2 in aggregate %}
            <td>
                <p>Label: {{ item2.label }}</p>
                <p>Total Detect: {{ item2.total }}</p>
                <p>Total Correct: {{ item2.correct }}</p>
                <p>Accuracy: {{ item2.Accuracy }}%</p>
            </td>
            {% if loop.index % 5 == 0 %}
        </tr><tr>
            {% endif %}
        {% endfor %}
        </tr>
    </table>
</body>
</html>
"""

# 이미지 데이터를 Base64로 변환하는 함수
def image_to_base64(image, scale=10):
    buffered = io.BytesIO()
    pil_image = Image.fromarray(image)
    pil_image = pil_image.resize((image.shape[0] * scale, image.shape[1] * scale), Image.NEAREST)
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

result = {}
for i in range(10):
    result[i] = [0,0] #total , correct

@app.route('/')
def index():
    # 랜덤으로 10개 이미지 선택
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images = images.to('cuda')

    # 모델 예측
    with torch.no_grad():
        predictions, _ = model(images)
        predicted_labels = predictions.argmax(dim=1).cpu().tolist()

    # 이미지와 결과를 저장
    data = []
    for i in range(len(images)):
        image = images[i].cpu().numpy().squeeze()
        image = (image * 255).astype(np.uint8)
        image_data = image_to_base64(image)
        data.append({
            'image_data': image_data,
            'label': labels[i].item(),
            'prediction': predicted_labels[i]
        })
        result[i][0] += 1
        if labels[i].item() == predicted_labels[i]:
            result[i][1] += 1
    aggregate = []
    for i in range(10):
        aggregate.append({
            'label': i,
            'total': result[i][0],
            'correct': result[i][1],
            'Accuracy':  round(result[i][1] / result[i][0] * 100, 1)
        })

    detect = 0
    correct = 0
    total_acc = 0
    for i in range(10):
        detect += result[i][0]
        correct += result[i][1]
        total_acc = round(correct / detect * 100, 0)
    print("total_acc:", total_acc)
    return render_template_string(html_template, data=data, aggregate=aggregate, total_acc = total_acc)

if __name__ == '__main__':
    app.run(debug=True)
