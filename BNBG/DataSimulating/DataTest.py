import numpy as np
import scipy
import MathFunctions

import matplotlib.pyplot as plt

from BNBG.Pipeline.BetterBackgroundSubtractStep import modelBackgroundFromImage


class TestSlit:
	def __init__(self, xsize, ysize, surface, **kwargs):
		"""
		Creates a Slit Object containing an image and a bivariate function. The function should be such that
		surface = surface(x,y)
		Parameters
		----------
		xsize the width
		ysize the height
		surface bivariate function
		"""
		self.xsize = xsize
		self.ysize = ysize
		self.function = surface
		XX,YY = np.meshgrid(np.linspace(0,xsize,xsize),np.linspace(0,ysize,ysize))

		self.data = surface(XX,YY,**kwargs)



class TestSlitlet:
	def __init__(self, Xsize, Ysize, sigmaNoise=1,
				 continuumX=None, continuumZ=None, signalX=None, signalZ=None, peaks=None,
				 peaksAmp=1, Lwidth=1, sigmaEnvelope=10, rotationAngle=30):
		"""
		Creates a Slitlet, composed of 3 slits
		Parameters
		----------
		Ysize : Scale along spatial / vertical direction. Should be a multiple of 3
		Xsize : Scale along wavelength / horizontal direction.
		sigmaNoise : Standard deviation of noise
		continuumX : Wavelength position of known continuum values
		continuumZ : Values of known continuum
		peaks : Wavelength positions of peaks
		peaksAmp : Height of peaks
		Lwidth : Width of peaks
		sigmaEnvelope : Standard deviation of enveloppe along spatial direction
		rotationAngle : Slits rotation angle in degrees
		"""

		self.rotationAngle = rotationAngle

		dy = int(Ysize/3)
		self.background = TestSlit(Xsize, dy, MathFunctions.background,
								   continuumX=continuumX, continuumZ=continuumZ, sigma=sigmaEnvelope)
		self.signal = TestSlit(Xsize, dy, MathFunctions.signal, continuumX=signalX, continuumZ=signalZ,
							   sigma=sigmaEnvelope, A=peaksAmp, peaks=peaks, Lwidth=Lwidth)
		self.signal.data += self.background.data

		template = np.zeros((Ysize, Xsize))


		# The 1 is the position of the signal slit, left to right being up to down
		self.slitlet = {"100":template.copy(), "010":template.copy(), "001":template.copy()}
		self.slitlet["100"][:dy,:] = self.signal.data
		self.slitlet["100"][dy:2*dy,:] = self.background.data
		self.slitlet["100"][2*dy:3*dy,:] = self.background.data

		self.slitlet["010"][:dy,:] = self.background.data
		self.slitlet["010"][dy:2*dy,:] = self.signal.data
		self.slitlet["010"][2*dy:3*dy,:] = self.background.data

		self.slitlet["001"][:dy,:] = self.background.data
		self.slitlet["001"][dy:2*dy,:] = self.background.data
		self.slitlet["001"][2*dy:3*dy,:] = self.signal.data

		# Wavelength map
		#self.wavelength = np.linspace(0.6,5,self.slitlet["100"].shape[1])
		self.wavelength = np.linspace(0, self.slitlet["100"].shape[1], self.slitlet["100"].shape[1])
		self.wavelength = np.tile(self.wavelength, (self.slitlet["100"].shape[0], 1))
		self.wavelengthRotated = scipy.ndimage.rotate(self.wavelength, self.rotationAngle, cval=np.nan, order=0)

		# Rotating
		self.rotated = {"100": scipy.ndimage.rotate(self.slitlet["100"], self.rotationAngle, cval=np.nan, order=0),
						"010": scipy.ndimage.rotate(self.slitlet["010"], self.rotationAngle, cval=np.nan, order=0),
						"001": scipy.ndimage.rotate(self.slitlet["001"], self.rotationAngle, cval=np.nan, order=0)}

		# Adding noise

		self.noise = np.random.normal(0, sigmaNoise, self.rotated["100"].shape)
		self.noise[np.isnan(self.rotated["100"])] = np.nan

		self.data = {"100": self.rotated["100"] + self.noise,
					 "010": self.rotated["010"] + self.noise,
					 "001": self.rotated["001"] + self.noise}

		# Anti rotation

		self.antiRotated = {"100":None,"010":None,"001":None}

		for ID in ["100","010","001"]:
			self.antiRotated[ID] = scipy.ndimage.rotate(self.data[ID], -self.rotationAngle,cval=np.nan,order=0)
			start_x = (self.antiRotated[ID].shape[0] - self.slitlet[ID].shape[0]) // 2
			start_y = (self.antiRotated[ID].shape[1] - self.slitlet[ID].shape[1]) // 2
			self.antiRotated[ID] = self.antiRotated[ID][start_x:start_x + self.slitlet[ID].shape[0],
								   start_y:start_y + self.slitlet[ID].shape[1]]

			self.noiseCorrected = scipy.ndimage.rotate(self.noise, -self.rotationAngle, cval=np.nan, order=0)
			start_x = (self.noiseCorrected.shape[0] - self.slitlet[ID].shape[0]) // 2
			start_y = (self.noiseCorrected.shape[1] - self.slitlet[ID].shape[1]) // 2
			self.noiseCorrected = self.noiseCorrected[start_x:start_x + self.slitlet[ID].shape[0],
								   start_y:start_y + self.slitlet[ID].shape[1]]

		# Anti Rotated Noise Map

		self.antiNoise = {"100": self.antiRotated["100"] - self.slitlet["100"],
						  "010": self.antiRotated["010"] - self.slitlet["010"],
						  "001": self.antiRotated["001"] - self.slitlet["001"]}


	@staticmethod
	def analyseNoise(image):
		plt.hist(image.ravel(),bins=64,alpha=0.3)
		print(f"Mean : {np.mean(image[~np.isnan(image)])}, Sigma : {np.std(image[~np.isnan(image)])}")

	@staticmethod
	def calculateBackground(slitlet):
		plt.figure(0)
		plt.imshow(slitlet.antiRotated["100"], origin="lower", vmin=0)
		plt.title("CORRECTED")

		plt.figure(1)
		plt.imshow(slitlet.wavelength, origin="lower")
		plt.title("WAVELENGTH")

		plt.figure(2)
		plt.imshow(slitlet.data["100"], origin="lower")
		plt.title("RAW")

		plt.figure(3)
		plt.imshow(slitlet.noise, origin="lower")
		plt.title("ERROR")

		plt.figure(4)
		plt.imshow(slitlet.noiseCorrected, origin="lower")
		plt.title("ERROR CORRECTED")

		plt.figure(5)
		plt.imshow(slitlet.wavelengthRotated, origin="lower")
		plt.title("WAVELENGTH RAW")

		modelBackground = []
		for i, ID in enumerate(["100","010","001"]):
			bkg = modelBackgroundFromImage(slitlet.data[ID],
									 slitlet.wavelengthRotated,
									 slitlet.antiRotated[ID],
									 np.ones_like(slitlet.wavelength)*2,
									 slitlet.wavelength,
									 i)
			modelBackground.append(bkg)
			break
		#plt.figure()
		#plt.imshow(modelBackground[0], origin="lower")
		#plt.show()

####################
# MAIN
####################

if __name__ == "__main__":
	slitlet = TestSlitlet(2000, 300, 10,
						  continuumX=[0,2000,400,1300], continuumZ=[200,10,140,100],
						  signalX=[0,2000,1400,600], signalZ=[150,300,240,160],
						  peaks=[500,550,650,1400,1430,1900], peaksAmp=[1000, 1200, 750, 1700, 1750, 600], Lwidth=5,
						  sigmaEnvelope=20, rotationAngle=12)

	# Show the different stages
	plt.close("all")
	#slitlet.show()

	# Show the noise after rotation and counter rotation
	# Except for a small std, the noise distribution is practically the same as the original gaussian
	#plt.figure()
	#TestSlitlet.analyseNoise(slitlet.antiNoise["100"])
	#TestSlitlet.analyseNoise(slitlet.antiNoise["010"])
	#TestSlitlet.analyseNoise(slitlet.antiNoise["001"])

	TestSlitlet.calculateBackground(slitlet)
	plt.show()
