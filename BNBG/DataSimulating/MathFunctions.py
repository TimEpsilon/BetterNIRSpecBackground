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


def cast1Dto3D(array1D,shape):
	"""
	Creates a 3D array A[ijk], where for a given k, A[ijk] = array1D[k]
	Parameters
	----------
	array1D : a 1D array
	shape : a tuple representing the shape of the array, the last coordinate should be of length len(array1D)

	Returns : the 3D array
	-------
	"""
	if len(array1D) != shape[2]:
		raise ValueError(f"Incorrect shape, should be {len(array1D)} but got {shape[2]}")
	return np.zeros(shape) + array1D[np.newaxis, np.newaxis, :]

def gauss(x, x0, sigma):
	if sigma == 0:
		return np.ones_like(x)
	else :
		return np.exp(-(x-x0)**2/(2*sigma**2))

def lorentzian(x, x0, A, L=1):
	# x0 can be a single point or a list
	# should be of size Dx,Dy,len(x0)
	# A can be a number or a list of numbers
	X = add2Dto1D(x,[-_ for _ in x0])
	if isinstance(A, list):
		A = np.array(A)
	if isinstance(A, np.ndarray):
		A = cast1Dto3D(A, X.shape)
	return (A / (L ** 2 / 4 + X ** 2)).sum(axis=2)

def continuum(x,x0, y0):
	# x0 and y0 are a list of numbers
	interp = interp1d(x0,y0,kind='cubic')
	return interp(x)

def signal(x, y, continuumX=None, continuumZ=None, peaks=None, A=1 ,Lwidth=1, sigma=0):
	"""
	Generates the image of a signal with lorentzian peaks, a smooth continuum and a gaussian spatial envelope
	Parameters
	----------
	x : abscissa, wavelength dependant
	y : ordinate, spatial dependant
	continuumX : the points at which the continuum is defined
	continuumZ : the values of said points
	peaks : the positions of the peaks, can either be a number or a list of numbers
	sigma : the standard deviation of the gaussian spatial envelope, default is 0 and means no envelope
	A : the lorentzian maximum. If a number, every peak will have this height. If a list, needs to be of len(peaks)
	Lwidth : the lorentzian width of the peaks.
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

	results = (continuum(x, continuumX, continuumZ) + lorentzian(x,peaks,A,L=Lwidth)) * gauss(y,(y.max() + y.min())/2, sigma)
	return results


def background(x, y, continuumX=None, continuumZ=None, sigma=0):
	"""
	Generates only the background continuum and a gaussian spatial envelope
	Parameters
	----------
	x : abscissa, wavelength dependant
	y : ordinate, spatial dependant
	continuumX : the points at which the continuum is defined
	continuumZ : the values of said points
	sigma : standard deviation of the envelope, if left at 0, no envelope will be applied

	Returns
	-------

	"""
	# Default : 3 points
	if continuumZ is None:
		continuumZ = [0, 1, 0]
	if continuumX is None:
		continuumX = [x[0], (x[-1] + x[0]) / 2, x[-1]]

	results = (continuum(x, continuumX, continuumZ)) * gauss(y,(y.max() + y.min())/2, sigma)
	return results



