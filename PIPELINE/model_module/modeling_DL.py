## MNIST ANOMALY DETECTION
## CHAT GPT

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import numpy as np

# 데이터셋 정의 (비정상 데이터: 8)
class loadDataset(Dataset):
    def __init__(self, path_dict):
        self.data = np.load(path_dict[0]).reshape(-1,1,28,28) # MNIST
        self.data = torch.from_numpy(self.data).float()
        self.data /= 255.0 #normalize
        self.label = np.load(path_dict[1])
        print(self.data.size())
        print(self.label.shape)
        self.normal_label = 8  # 비정상 클래스

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.label[idx]
        is_anomaly = 1 if label == self.normal_label else 0
        return image, is_anomaly

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x


def modeling_main(path_dict:dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_param = {
        "device":device,
        "learning_rate":0.001,
        "batch_size":64,
        "epochs":10,
        "cost_function":"BCE"
    }
    # 데이터 로더
    train_dataset = loadDataset(path_dict['train'])
    valid_dataset = loadDataset(path_dict['valid'])
    test_dataset = loadDataset(path_dict['test'])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 모델, 손실 함수, 옵티마이저 정의
    model = CNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=train_param["learning_rate"])



    # MLflow 설정
    experiment_name = "DL Experiment"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # 하이퍼파라미터 로깅
        for param_, value in train_param.items():
            mlflow.log_param(param_, value)
        
        # 학습 루프
        for epoch in range(10):  # 에포크 수
            model.train()
            running_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.float().to(device).unsqueeze(1)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        # 테스트 루프
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        mlflow.log_metric("test_accuracy", accuracy)
        print(f"Accuracy: {accuracy:.2f}%")

        # 모델 저장
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    data_dict = {
        'train': ['/home/madfalcon/data/MNIST_trainable/train_x.npy', 
        '/home/madfalcon/data/MNIST_trainable/train_y.npy'], 
        'valid': ['/home/madfalcon/data/MNIST_trainable/valid_x.npy', 
        '/home/madfalcon/data/MNIST_trainable/valid_y.npy'], 
        'test': ['/home/madfalcon/data/MNIST_trainable/test_x.npy', 
        '/home/madfalcon/data/MNIST_trainable/test_y.npy']
    }
    modeling_main(data_dict)
