from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

class RegressionModel():
    def __init__(self, n):
        self.regularization= n
        if(n == 1):
        	self.model = LogisticRegression()
        elif(n == 2):
        	self.model = Ridge()
        elif (n == 3):
        	self.model = Lasso()
        else:
        	self.model = LogisticRegression()

    def getModelInfo(self):
        if(self.regularization == 1):
            return "Logistic Regression"
        if(self.regularization == 2):
            return "Ridge Regression"
        if(self.regularization == 3):
            return "Lasso"
        if(self.regularization == 4):
            return "Interaction Only"
        if(self.regularization == 5):
            return "Interaction and Polynomial"

    def train(self, X, Y):
        assert len(X)==len(Y)
        if(self.regularization == 4):
        	poly = PolynomialFeatures(interaction_only=True,include_bias = False)
        if(self.regularization == 5):
        	poly = PolynomialFeatures(3)
        if(self.regularization == 4 or self.regularization == 5):
        	XtrainPoly = poly.fit_transform(X)
        	self.model.fit(XtrainPoly, Y)
        else:
        	self.model.fit(X,Y)

    def predict(self, X):
        if self.regularization == 1:
            return np.array(self.model.predict_proba(X))
        else:
            return np.array(self.model.predict(X))
