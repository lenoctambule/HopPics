from HopPics import *
import sys

if __name__ == "__main__" :
	if len(sys.argv) != 3 :
		print("Usage : py test.py <image_path> <nsteps>")
		exit()
	hp = HopPics(sys.argv[1], 50)
	hp.reconstruct_from_noise(noise_amount=1_000, n_steps=int(sys.argv[2]))