import numpy as np
import tqdm
import random as rd

class HopfieldNet:
	def __init__(self, data_len, learning_rate):
		self.data_len = data_len
		self.learning_rate = learning_rate
		self.weights = np.zeros((data_len, data_len),dtype=float)

	def train(self,data_arr):
		new = self.weights.copy()
		for data in data_arr:
			for i in range(self.data_len):
				for j in range(self.data_len) :
					if i != j :
						new[i,j] += (data[i] * data[j]) * self.learning_rate
		self.weights = np.add(self.weights, new / self.data_len)

	def compute_step(self, data, res):
		i = rd.randrange(len(data))
		for j in range(0, len(data)):
			if i != j :
				res[i] += self.weights[i,j] * data[j]
		return res

	def run(self,data, steps=2):
		res = data.copy()
		frames = []
		for k in tqdm.tqdm(range(steps)):
			frames.append(self.compute_step(data,res))
		return [int(np.sign(i)) for i in res], frames