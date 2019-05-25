# Abstract Class for model

class Model():
    def __init__(self):
        raise NotImplementedError

    def train(self, X, Y):
        # store model in self.model
        # each row of X is one feature vector
        # Y is a vector of labels
        raise NotImplementedError

    def predict(self, X):
        # each row of X is one feature vector
        # output is a vector of returns
        raise NotImplementedError

    def getModelInfo():
        raise NotImplementedError
