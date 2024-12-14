import numpy as np
import tqdm
import random as rd

class HopfieldNet:

	def __init__(self, data_len, learning_rate):
		self.data_len = data_len
		self.learning_rate = learning_rate
		self.N = 0
		self.weights = np.zeros((data_len, data_len),dtype=float)

	def train(self,data_arr):
		self.N += 1
		self.weights += np.outer(data_arr, data_arr)

	def compute_step(self, v_1):
		v_1 += np.einsum('ij,j->i', self.weights, v_1)
		return v_1

	def run(self, data, steps=2):
		frames = []
		for k in tqdm.tqdm(range(steps)):
			frames.append(self.compute_step(data))
		return np.sign(data), frames