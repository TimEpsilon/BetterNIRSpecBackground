import numpy as np
from scipy.interpolate import interp1d

def gauss(x, x0, sigma):
	return np.exp(-(x-x0)**2/(2*sigma**2))

def lorentzian(x, x0, L):
	return L/(2*np.pi) / (L**2/4 + (x-x0)**2)

def continuum(x,x0, y0):
	# x0 and y0 are a list of numbers
	interp = interp1d(x0,y0,kind='cubic')
	return interp(x)

def signal(x,y,peaks=[],sigma=10,Lwidth=1):
	# TODO
	result = 0

