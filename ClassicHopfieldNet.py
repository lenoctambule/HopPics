import numpy as np
import tqdm
import random as rd

class HopfieldNet:
	def __init__(self, data_len, learning_rate):
		self.data_len = data_len
		self.learning_rate = learning_rate
		self.N = 0
		self.W = np.zeros((data_len, data_len),dtype=float)

	def train(self,v):
		self.N += 1
		self.W += np.outer(v, v) * self.learning_rate

	def compute_step(self, v_n):
		v_n += np.einsum('ij,j->i', self.W, v_n)
		return v_n

	def run(self, v_0, steps=2):
		frames = [v_0.copy()]
		for k in tqdm.tqdm(range(steps)):
			frames.append(self.compute_step(v_0))
		return np.sign(v_0), frames