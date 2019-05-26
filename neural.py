import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from util import get_clean_data

class NN_wrapper():
	def __init__(self, input_dim, lr = 1e-4):
		self.model = NeuralNet(input_dim)
		self.criterion = nn.BCELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)


	def to_tensor(self,df):
		#convert data frame into x,y tensors
		return torch.tensor(df).type(torch.FloatTensor)

	def train(self, X, Y, epochs =3000):
		x_train = self.to_tensor(X)
		y_train = self.to_tensor(Y)
		for epoch in range(epochs):
			y_hat = self.model.forward(x_train)
			self.optimizer.zero_grad()
			loss = self.criterion(y_hat,y_train.unsqueeze(1))
			loss.backward()
			self.optimizer.step()
			if (epoch + 1) % 100 == 0:
				print('Epoch: [%d/%d], Loss: ' %(epoch+1, epochs), loss.data)

	def predict(self, X):
		x_pred = self.to_tensor(X)
		return self.model.forward(x_pred).squeeze(1).data.numpy()



class NeuralNet(nn.Module):
	def __init__(self, input_dim):
		super(NeuralNet, self).__init__()
		self.layer_1 = nn.Linear(input_dim,30)
		#nn.init.xavier_uniform_(self.layer_1)
		self.layer_2 = nn.Linear(30,10)
		#nn.init.xavier_uniform_(self.layer_2)
		self.layer_3 = nn.Linear(10,1)
		#nn.init.xavier_uniform_(self.layer_3)

	def forward(self, x):
		x = torch.sigmoid(self.layer_1(x))
		x = torch.sigmoid(self.layer_2(x))
		y_hat = torch.sigmoid(self.layer_3(x))
		return y_hat


if __name__ == '__main__':

	train, validation, test = get_clean_data()
	X = train.drop(['Date','Class','Minute'], axis=1)
	Y = train['Class']
	NN = NN_wrapper(X.shape[1])
	NN.train(X,Y)

	X = validation.drop(['Date','Class','Minute'], axis=1)
	print(NN.predict(X))
