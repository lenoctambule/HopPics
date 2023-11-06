from HopPics import *
import sys

if __name__ == "__main__" :
	if len(sys.argv) != 3 :
		print("Usage : py test.py <image_path> <nsteps>")
		exit()
	try :
		hp = HopPics(sys.argv[1], 50)
	except :
		print("Image could not be read.")
		exit()
	hp.reconstruct_from_noise(noise_amount=10000, n_steps=int(sys.argv[2]))