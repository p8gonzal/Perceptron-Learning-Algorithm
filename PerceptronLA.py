#Author: Peter Gonzalez
import numpy as np

class Perceptron():
    def __init__(self, passes = 1):
        self.passes = passes
        self.predictionRule = self.predictionRule
        self.weights = None


    # Constrain to two classes
    def predictionRule(self, val):
        return np.where(val >= 0, 1, -1)
        
    def initializeDataSets(self, X, y, label):
        self.X_train = X
        self.y_train = np.array([1 if i == label else -1 for i in y])

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        # Update perceptron weight on misclassifications
        for _ in (self.passes):
            index = 0
            for x_train in self.X_train:
                if (self.y_train[index] * np.dot(self.weights,x_train) ) <= 0:
                    self.weights = self.weights + x_train*self.y_train[index]
                index = index + 1

    # Predict using wTx
    def predict(self, X):
        n_samples, n_features = X.shape
        output = np.zeros(n_samples)
        index = 0
        for x_test in X:
            output[index] = np.dot(self.weights.transpose(), x_test)
            index = index + 1
        y_predicted = self.predictionRule(output)
        return np.array([1 if i == 1 else -1 for i in y_predicted])
