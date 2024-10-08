import numpy as np
from scipy.interpolate import interp1d


def add2Dto1D(array2D, array1D):
	"""
	Creates a 3D array, where each element of the 1D array is added to a copy of the 2D array
	Parameters
	----------
	array1D
	array2D

	Returns :
	array3D
	-------
	"""
	if isinstance(array1D,int):
		array1D = np.array([array1D])
	elif isinstance(array1D,list):
		array1D = np.array(array1D)

	if isinstance(array2D,float) or isinstance(array2D,int):
		array2D = np.array([[array2D]])
	return array2D[:, :, np.newaxis] + array1D[np.newaxis, np.newaxis, :]

def gauss(x, x0, sigma):
	return np.exp(-(x-x0)**2/(2*sigma**2))

def lorentzian(x, x0, L):
	# x0 can be a single point or a list
	# should be of size Dx,Dy,len(x0)
	X = add2Dto1D(x,-x0)
	return (L/(2*np.pi) / (L**2/4 + (X)**2)).sum(axis=2)

def continuum(x,x0, y0):
	# x0 and y0 are a list of numbers
	interp = interp1d(x0,y0,kind='cubic')
	return interp(x)

def signal(x, y, continuumX=None, continuumZ=None, peaks=None, Lwidth=1, sigma=10):
	"""
	Generates the image of a signal with lorentzian peaks, a smooth continuum and a gaussian spatial envelope
	Parameters
	----------
	x : abscissa, wavelength dependant
	y : ordinate, spatial dependant
	continuumX :
	continuumZ
	sigma
	Lwidth

	Returns
	-------

	"""
	# Default : 3 points
	if continuumZ is None:
		continuumZ = [0, 1, 0]
	if continuumX is None:
		continuumX = [x[0], (x[-1] + x[0]) / 2, x[-1]]
	if peaks is None:
		peaks = [(x[-1] + x[0]) / 2]

	results = (continuum(x, continuumX, continuumZ) + lorentzian(peaks,0.2)) * gauss(y,(y.max() + y.min())/2, 2)
	return results

