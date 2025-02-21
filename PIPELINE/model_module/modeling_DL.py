## MNIST ANOMALY DETECTION
## CHAT GPT
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

#mlflow
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# 데이터셋 정의 (비정상 데이터: 8)
class loadDataset(Dataset):
    def __init__(self, path_dict, ignore_label=[]):
        self.ignore_label = ignore_label
        self.data = np.load(path_dict[0]).reshape(-1,1,28,28) # MNIST
        print(self.data.shape)
        self.label = np.load(path_dict[1]).reshape(-1)
        print(self.label.shape)
        self.ignore_()
        
        self.data = torch.from_numpy(self.data).float()
        self.label = torch.from_numpy(self.label).type(torch.LongTensor)

    def ignore_(self):
        print("ignore label list", self.ignore_label)
        for label_ in self.ignore_label:
            indice = np.where(self.label != label_)
            self.data = self.data[indice]
            self.label = self.label[indice]
            print(np.unique(self.label), f"label {label_} is ignored")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
            cls mode
        """
        image, label = self.data[idx], self.label[idx]
        return image / 255.0, label

# CNN 모델 정의
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # input: [64, 1, 28, 28]
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # ic, oc, kernel, stride
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        fv = torch.flatten(x, 1) # flatten(vector)
        x = F.relu(self.fc1(fv))
        x = self.fc2(x)
        return x

def modeling_main(path_dict:dict, ignore_label:list=[], save_path:str="/data"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_param = {
        "device":device,
        "learning_rate":0.01,
        "batch_size":64,
        "epochs":10,
        "cost_function":"CE"
    }
    # 데이터 로더
    train_dataset = loadDataset(path_dict['train'], ignore_label=ignore_label)
    valid_dataset = loadDataset(path_dict['valid'])
    test_dataset = loadDataset(path_dict['test'])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 모델, 손실 함수, 옵티마이저 정의
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_param["learning_rate"])


    # MLflow 설정
    mlflow.set_tracking_uri(uri="http://172.17.0.1:5000")
    experiment_name = "MLOPS Classification"
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
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            avg_loss = running_loss / len(train_loader)
            mlflow.log_metric("Train_loss", avg_loss, step=epoch+1)
            print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        # 테스트 루프
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                labels=labels.view(-1)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.data).sum().item()

        accuracy = 100 * correct / total
        mlflow.log_metric("Accuracy", accuracy)
        print(f"Accuracy: {accuracy:.2f}%")
        model_scripted = torch.jit.script(model) # Export to TorchScript
        model_scripted.save(save_path+'/model_scripted.pt') # Save
        mlflow.pytorch.log_model(model, "model")
        print(f"save complete")

if __name__ == "__main__":
    data_dict = {
        'train': ['/data/MNIST_trainable/train_x.npy', 
        '/data/MNIST_trainable/train_y.npy'], 
        'valid': ['/data/MNIST_trainable/valid_x.npy', 
        '/data/MNIST_trainable/valid_y.npy'], 
        'test': ['/data/MNIST_trainable/test_x.npy', 
        '/data/MNIST_trainable/test_y.npy']
    }
    ignore_label = []
    modeling_main(data_dict, ignore_label)