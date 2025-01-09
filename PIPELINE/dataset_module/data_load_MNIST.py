from sklearn.model_selection import train_test_split
import numpy as np
from torchvision import datasets

class c_dataset:
    """
        데이터를 학습 또는 예측가능한 데이터로 변환해주는 클래스
        압축형태로 된 MNIST 데이터를 학습이 가능하도록 numpy 형태로 변환
        MNIST(npy) source: https://www.kaggle.com/datasets/sivasankaru/mnist-npy-file-dataset?resource=download
    """
    def __init__(self, data_path="/home/madfalcon/data/MNIST", 
            save_path="/home/madfalcon/data/MNIST_trainable",
            ignore_label = []):
        self.ignore_label = ignore_label
        self.data_path = data_path
        self.save_path = save_path
        self.origin_data = {}
        self.data = {}

    def load_dataset(self):
        # x, y
        self.origin_data['train_x'] = np.load(self.data_path+'/train_images.npy')
        self.origin_data['train_y'] = np.load(self.data_path+'/train_labels.npy')
        self.origin_data['test_x'] = np.load(self.data_path+'/test_images.npy')
        self.origin_data['test_y'] = np.load(self.data_path+'/test_labels.npy')
        print("Data Loading Complete.")
    
    def save_data(self, path=None):
        result={}
        if path is None:
            path = self.save_path
        print("save path:", path)
        for key, value in self.data.items():
            np.save(path+f'/{key}_x.npy', value[0])
            np.save(path+f'/{key}_y.npy', value[1])
            print(f"{key} save.")
            result[key] = [path+f'/{key}_x.npy', path+f'/{key}_y.npy']
        print(result)
        print("save complete")
        return result 

    def split_data(self, ratio=0.8):
        x, y = self.origin_data['train_x'], self.origin_data['train_y']
        # Split the data into training and test sets
        train_x, valid_x, train_y, valid_y = train_test_split(
            x, y, test_size=1-ratio, random_state=42
        )
        self.data["train"] = [train_x, train_y]
        self.data["valid"] = [valid_x, valid_y]
        self.data["test"] = [self.origin_data['test_x'], self.origin_data['test_y']]
        print(f"Data Split(Ratio:{ratio}) Complete")

        print("ignore label list", self.ignore_label)
        for label_ in self.ignore_label:
            indice = np.where(self.data["train"][1] != label_)
            self.data["train"][0] = self.data["train"][0][indice]
            self.data["train"][1] = self.data["train"][1][indice]
            print(np.unique(self.data["train"][1]), f"label {label_} is ignored")

    def get_data(self):
        return self.data

if __name__ == "__main__":
    dset = c_dataset()
    dset.load_dataset()
    dset.split_data()
    dset.save_data()