from PIL import Image
import matplotlib.pyplot as plt
import random as rd
from ClassicHopfieldNet import *
import numpy as np

class HopPics:
	gif_images = []

	def __init__(self, path, learning_rate) -> None:
		self.path = path
		self.img = Image.open(self.path).convert('1')
		self.pixels = np.asarray(self.img.getdata()).flatten()
		self.pixels[self.pixels == 0] = -1
		self.pixels[self.pixels == 255] = 1
		self.datalen = self.img.size[0] * self.img.size[1]
		self.hp = HopfieldNet(data_len=self.datalen, learning_rate=learning_rate)
		print("Starting training ...")
		self.hp.train(self.pixels)
		print("Training complete.")

	def plot(self, test, res, steps):
		img = Image.new('1', self.img.size, 0)
		plt.subplot(1,4,4)
		plt.imshow(self.img)
		plt.title('Original')
		plt.subplot(1,4,3)
		plt.imshow(self.hp.W)
		plt.title('Weights')
		plt.subplot(1,4,1)
		plt.imshow(np.reshape(test, (self.img.size[1], self.img.size[0])))
		plt.title('Before')
		plt.subplot(1,4,2)
		plt.imshow(np.reshape(res, (self.img.size[1], self.img.size[0])))
		plt.title('After')
		plt.show()

	def	gen_gif(self, steps):
		res = []
		for i in range(0, len(steps), 100):
			step = [0 if int(np.sign(i)) < 0 else 255 for i in steps[i]]
			step = np.reshape(step, (self.img.size[1], self.img.size[0]))
			step = step.astype(np.uint8)
			res.append(Image.fromarray(step).convert('RGB'))
		frame_0 = res[0]
		frame_0.save("res.gif", format="GIF", append_images=res, save_all=True, duration=1, loop=0)

	def reconstruct_from_noise(self, noise_amount=50, n_steps=4):
		test = np.asarray(self.pixels.copy(), dtype=float)
		for i in range(noise_amount) :
			test[rd.randint(0,self.datalen - 1)] = -1 if rd.randint(0,2) == 0 else 1
		print("Starting reconstruction.")
		res, steps = self.hp.run(v_0=test.copy(), steps=n_steps)
		print("Reconstruction complete")
		# self.gen_gif(steps)
		self.plot(test, res, steps)

