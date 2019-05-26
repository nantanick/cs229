import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


class SVMModel():
    def __init__(self, n):
        if (n == 1):
            K = 'linear'
            self.kernel = 1
            self.model = SVC(kernel = K)
        elif (n == 2):
            K = 'poly'
            self.kernel = 2
            self.model = SVC(kernel = K, degree = 3) #Degree
        elif (n == 3):
            K = 'rbf'
            self.kernel = 3
            self.model = SVC(kernel = K)
        elif (n == 4):
            K = 'sigmoid'
            self.kernel = 4
            self.model = SVC(kernel = K)
        else:
            self.model = SVC(kernel = 'linear')

    def train(self, X, Y):
        assert len(X) == len(Y)
        if (self.kernel == 2):
            poly = PolynomialFeatures(3)
            XtrainPoly = poly.fit_transform(X)
            self.model.fit(XtrainPoly, Y)
        else:
            self.model.fit(X, Y)

    def predict(self, X):
        return np.array(self.model.predict(X))
