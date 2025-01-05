from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class model_():
    def __init__(self):
        # Define the model hyperparameters
        self.save_path = "/home/madfalcon/MLmodel"
        self.model = None
        self.params = {
            "solver": "lbfgs",
            "max_iter": 1000,
            "multi_class": "auto",
            "random_state": 8888,
        }
        self.model_load()

    def model_load(self):
        self.model = LogisticRegression(**self.params)

    def fit(self, data):
        x,y = data
        self.model.fit(x,y)
        print("model fit complete")

    def predict(self, data):
        x,y = data
        pred = self.model.predict(x)
        # Calculate metrics
        accuracy = accuracy_score(y, pred)
        print("accuracy: ",accuracy)

    def save_(self):
        import joblib
        joblib.dump(self.model, self.save_path +"/model.pkl") 
        print("model save complete")
        
    def load_(self, model_path):
        import joblib
        self.model = joblib.load(model_path) 