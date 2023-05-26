from HopPics import *
import sys

if __name__ == "__main__" :
	if len(sys.argv) != 2 :
		print("Usage : py test.py <image_path>")
		exit()
	hp = HopPics(sys.argv[1])
	hp.reconstruct_from_noise(noise_amount=5_000)