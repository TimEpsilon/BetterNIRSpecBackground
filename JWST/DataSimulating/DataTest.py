import numpy as np
import scipy
import MathFunctions

import matplotlib.pyplot as plt


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
				 continuumX=None, continuumZ=None, peaks=None,
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
		self.signal = TestSlit(Xsize, dy, MathFunctions.signal, continuumX=continuumX, continuumZ=continuumZ,
							   sigma=sigmaEnvelope, A=peaksAmp, peaks=peaks, Lwidth=Lwidth)

		template = np.zeros((Ysize, Xsize))


		# The 1 is the position of the signal slit, left to right being up to down
		self.slitlet_100 = template.copy()
		self.slitlet_100[:dy,:] = self.signal.data
		self.slitlet_100[dy:2*dy,:] = self.background.data
		self.slitlet_100[2*dy:3*dy,:] = self.background.data

		self.slitlet_010 = template.copy()
		self.slitlet_010[:dy,:] = self.background.data
		self.slitlet_010[dy:2*dy,:] = self.signal.data
		self.slitlet_010[2*dy:3*dy,:] = self.background.data

		self.slitlet_001 = template.copy()
		self.slitlet_001[:dy,:] = self.background.data
		self.slitlet_001[dy:2*dy,:] = self.background.data
		self.slitlet_001[2*dy:3*dy,:] = self.signal.data

		# Rotating

		self.rotated_100 = scipy.ndimage.rotate(self.slitlet_100, self.rotationAngle,cval=np.nan)
		self.rotated_010 = scipy.ndimage.rotate(self.slitlet_010, self.rotationAngle,cval=np.nan)
		self.rotated_001 = scipy.ndimage.rotate(self.slitlet_001, self.rotationAngle,cval=np.nan)

		# Adding noise

		self.noise = np.random.normal(0, sigmaNoise, self.rotated_100.shape)
		self.noise[np.isnan(self.rotated_100)] = 0

		self.data_100 = self.rotated_100 + self.noise
		self.data_010 = self.rotated_010 + self.noise
		self.data_001 = self.rotated_001 + self.noise

	def show(self, ID):
		if ID == "100":
			plt.figure(figsize=(12,6))
			plt.imshow(self.data_100, origin="lower",)

		if ID == "010":
			plt.figure(figsize=(12, 6))
			plt.imshow(self.data_010, origin="lower")

		if ID == "001":
			plt.figure(figsize=(12, 6))
			plt.imshow(self.data_001, origin="lower")

		return

slitlet = TestSlitlet(2000, 300, 30,
					  continuumX=[0,2000,400,1300], continuumZ=[10,200,100,160],
					  peaks=[500,550,650,1400,1430,1900], peaksAmp=[1000, 1200, 750, 1700, 1750, 600], Lwidth=5,
					  sigmaEnvelope=10, rotationAngle=12)

plt.close("all")
slitlet.show("100")
slitlet.show("010")
slitlet.show("001")
plt.show()
