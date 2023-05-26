from PIL import Image
import matplotlib.pyplot as plt
import random as rd
from HopfieldNet import *

class HopPics:
	def __init__(self, path) -> None:
		self.path = path
		self.img = Image.open(self.path).convert('1')
		self.pixels = np.asarray(self.img.getdata()).flatten()
		self.pixels[self.pixels == 0] = -1
		self.pixels[self.pixels == 255] = 1
		self.datalen = self.img.size[0] * self.img.size[1]
		self.hp = HopfieldNet(data_len=self.datalen)
		print("Starting training ...")
		self.hp.train([self.pixels])
		print("Training complete.")

	def plot(self, test, res):
		plt.subplot(1,4,4)
		plt.imshow(self.img)
		plt.title('Original')
		plt.subplot(1,4,3)
		plt.imshow(self.hp.weights)
		plt.title('Weights')
		plt.subplot(1,4,1)
		plt.imshow(np.reshape(test, (self.img.size[1], self.img.size[0])))
		plt.title('Before')
		plt.subplot(1,4,2)
		plt.imshow(np.reshape(res, (self.img.size[1],self.img.size[0])))
		plt.title('After')
		plt.show()

	def reconstruct_from_noise(self, noise_amount=50, steps=4):
		test = np.asarray(self.pixels.copy(), dtype=float)
		for i in range(noise_amount) :
			test[rd.randint(0,self.datalen - 1)] = -1 if rd.randint(0,2) == 0 else 1
		print("Starting reconstruction.")
		res = self.hp.steps(data=test, steps=steps)
		print("Reconstruction complete")
		self.plot(test, res)

