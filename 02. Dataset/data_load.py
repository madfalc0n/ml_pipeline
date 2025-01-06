from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

class dataset:
    def __init__(self):
        self.path = "/home/madfalcon/data"
        self.data_init = None
        self.data = {}

    def load_dataset(self):
        # x, y
        self.data_init = datasets.load_iris(return_X_y=True)
        print("Data Loading Complete")

    def split_data(self, ratio=0.8):
        x,y = self.data_init
        # Split the data into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=1-ratio, random_state=42
        )
        self.data['x_train'] = x_train
        self.data['y_train'] = y_train
        self.data['x_test'] = x_test
        self.data['y_test'] = y_test
        print(f"Data Split(Ratio:{ratio}) Complete")


    def get_data(self):
        return self.data


    def save_data(self, path=None):
        if path is None:
            path = self.path
        np.save(path+'/x_train.npy', self.data['x_train'])
        np.save(path+'/y_train.npy', self.data['y_train'])
        np.save(path+'/x_test.npy', self.data['x_test'])
        np.save(path+'/y_test.npy', self.data['y_test'])
        result = {
            'x_train':path+'/x_train.npy',
            'y_train':path+'/y_train.npy',
            'x_test':path+'/x_test.npy',
            'y_test':path+'/y_test.npy',
        }
        return result