import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from util import get_clean_data
import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class CNN_wrapper():
	def __init__(self, input_dim, lr = 1e-4, look_back = 5):
		self.model = CNN(input_dim, 1, 5)
		self.criterion = nn.BCELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
		self.look_back = look_back


	def to_tensor(self,df):
		#convert data frame into x,y tensors
		return torch.tensor(df).type(torch.FloatTensor)

	def train(self, X, Y, epochs = 100000):
		x_train, y_train = self.get_series(X, Y)
		x_train = self.to_tensor(x_train)
		y_train = self.to_tensor(y_train)
		epoch = 0
		while True:
			for x, y in zip(x_train, y_train):
				y_hat = self.model.forward(x.unsqueeze(0))
				self.optimizer.zero_grad()
				loss = self.criterion(y_hat,y)
				loss.backward()
				self.optimizer.step()
				if (epoch) % 100 == 0:
					print('Epoch: [%d/%d], Loss: ' %(epoch+1, epochs), loss.data)
				if epoch >= epochs:
					#self.model.save('./outputs/cnn_model')
					#torch.save(optimizer.state_dict(), './outputs/cnn_optim')
					return
				epoch +=1

	def predict(self, X):
		x_test, _ = self.get_series(X, Y)
		x_test = self.to_tensor(x_test)
		y_hat = []
		for x in x_test:
			y_hat.append(self.model.forward(x.unsqueeze(0)).data.numpy())
		return y_hat

	def get_series(self, X, Y):
		scale = MinMaxScaler(feature_range=(0, 1))
		X = scale.fit_transform(X)
		new_X = []
		new_Y = []
		for i in range(len(X)-self.look_back):
			new_X.append(X[i:i+self.look_back])
			new_Y.append(Y[i+self.look_back])
		return new_X, new_Y
	
	# def batch_iter(self, X, Y, batch_size, shuffle=False):
	# 	batch_num = math.ceil(len(X) / batch_size)
	# 	index_array = list(range(len(X)))

	# 	if shuffle:
	# 		np.random.shuffle(index_array)

	# 	for i in range(batch_num):
	# 		indices = index_array[i * batch_size: (i + 1) * batch_size]
	# 		print(indices)
	# 		x = torch.tensor([X[i] for i in indices])
	# 		y = torch.tensor([Y[i] for i in indices])
	# 		yield x,y




class CNN(nn.Module):
	def __init__(self, embed_size, filters, input_size, kernel_size=5):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv1d(embed_size, 20, kernel_size)
		self.dropout = nn.Dropout(0.3)
		self.lin = nn.Linear(20,1)


	def forward(self, x):
		x_reshaped = x.permute(0,2,1)
		x_conv1 = torch.sigmoid(self.conv1(x_reshaped))
		x_conv1_reshape =  x_conv1.squeeze(dim=0).squeeze(dim=1)
		x_lin = self.lin(x_conv1_reshape)
		out = torch.sigmoid(x_lin)

		return out

if __name__ == '__main__':
	train, validation, test = get_clean_data()
	X = train.drop(['Date','Class','Minute'], axis=1).values
	Y = train['Class'].values
	CNN = CNN_wrapper(X.shape[1])
	CNN.train(X,Y)

	X = validation.drop(['Date','Class','Minute'], axis=1).values
	train_out = pd.DataFrame(CNN.predict(X))
	train_out.to_csv('cnn_train.csv')
	test_out = pd.DataFrame(CNN.predict(X))
	test_out.to_csv('cnn_test.csv')

