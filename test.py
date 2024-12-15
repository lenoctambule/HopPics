from HopPics import *
import sys

if __name__ == "__main__" :
	if len(sys.argv) != 4 :
		print("Usage : py test.py <image_path> <nsteps> <noise_amount>")
		exit()
	hp = HopPics(sys.argv[1], 0.1)
	hp.reconstruct_from_noise(noise_amount=int(sys.argv[3]), n_steps=int(sys.argv[2]))