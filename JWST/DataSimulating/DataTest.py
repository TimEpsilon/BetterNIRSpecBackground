import numpy as np
import scipy
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
		YY,XX = np.meshgrid(np.linspace(0,xsize,xsize),np.linspace(0,ysize,ysize))

		self.data = surface(XX,YY,**kwargs)