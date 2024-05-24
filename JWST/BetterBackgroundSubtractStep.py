import numpy as np
from scipy.signal import find_peaks_cwt
from utils import *
from scipy.optimize import curve_fit as cfit
import matplotlib.pyplot as plt
import stdatamodels.jwst.datamodels as dm
from scipy.ndimage import rotate
from scipy import interpolate


def BetterBackgroundStep(name):
	"""
	 Creates a _bkg file from a _srctype file

	 Params
	 ---------
	 name : str
		Path to the file to open. Must be a _srctype
	"""
	if not "_srctype" in name:
		logConsole(f"{name.split('/')[-1]} not a _srctype file. Skipping...",source="WARNING")
		pass

	# 1st draft Algorithm :
	logConsole(f"Starting Custom Background Substraction on {name.split('/')[-1]}",source="BetterBackground")
	multi_hdu = dm.open(name)

	# For a given _srctype, for every slit
	for slit in multi_hdu.slits:
		logConsole(f"Opened slitlet {slit.slitlet_id}")
		data = np.ma.masked_invalid(slit.data)

		#TODO : Eventually, work on error propagation

		shutter_id = WhichShutterOpen(slit.shutter_state)
		if shutter_id is None:
			logConsole("Not a 3 shutter slit!")
			continue

		slice_indices, rotated = SelectSlice(slit)

		sliceFail = np.any(slice_indices is None)
		if sliceFail:
			logConsole("Can't find 3 spectra. Defaulting to equal slices",source="WARNING")
			first_column = data[:,int(data.shape[1]/2)]
			start_indice = np.where(first_column > 0)[0][0]
			end_indice = np.where(first_column > 0)[0][-1]
			n = end_indice - start_indice
			xmin = np.array([0,int(n/3),int(2*n/3)]) + start_indice
			xmax = np.array([int(n/3),int(2*n/3),n]) + start_indice
			slice_indices = np.array([xmin,xmax]).T

		bkg_slice = []
		bkg_interp = []
		coeff = []

		for j in range(2):
			# Get 2 background strips
			bkg_slice.append(data[slice_indices[shutter_id-j-1][0]:slice_indices[shutter_id-j-1][1],:])
			_ = bkg_slice[j] < 0
			bkg_slice[j][_] = np.nan
			bkg_slice[j][_].mask = True

			new_bkg_slice, c = AdjustModelToBackground(bkg_slice[j])
			bkg_interp.append(new_bkg_slice)
			coeff.append(c)

		if coeff[0] is None or coeff[1] is None:
			continue

		new_bkg = np.copy(data)
		new_bkg[:,:] = np.nan

		new_bkg[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1],:] = bkg_interp[0]
		new_bkg[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1],:] = bkg_interp[1]

		new_bkg = polynomialExtrapolation(new_bkg,*coeff,slice_indices,shutter_id)

		slit.data = np.ma.getdata(data - new_bkg)

	logConsole(f"Saving File {name.split('/')[-1]}",source="BetterBackground")
	multi_hdu.writeto(name.replace("_srctype","_bkg"),overwrite=True)
	multi_hdu.close()
	pass


def AdjustModelToBackground(bkg):
	"""

	Params
	-----------
	bkg : 2D array
		The background strip

	Returns
	-----------
	img : 2D array
		The fitted / interpolated slice
	c : array
		List of coefficients
	"""
	Y,X = np.indices(bkg.shape)

	# Starting parameter
	p0 = [
		4, # sigma
		bkg.shape[0]/2, # y0
		bkg.mean(), # constant shift, order 0 polynomial
		1, # order 1
		1, # order 2
		1, # order 3
		1 # order 4
	]
	try :
		coeff, err = cfit(betterPolynomial,[X,Y], bkg.ravel(), p0=p0)
		fit = betterPolynomial([X, Y], *coeff).reshape(bkg.shape)
		return fit, coeff
	except :
		logConsole("Optimal parameters not Found. Skipping")
		return None, None


def rotateSlit(slit):
	"""
	Rotates the slit in order to have horizontal strips

	Parameters
	----------
	slit

	Returns
	-------
	(data,
	wavelength,
	ra,
	dec,
	err)
	"""
	data = slit.data
	# Maps the WCS info
	Y, X = np.indices(slit.data.shape)
	ra, dec, wavelength = slit.meta.wcs.transform('detector', 'world', X, Y)
	ra[np.isnan(ra)] = 0
	dec[np.isnan(dec)] = 0
	data[np.isnan(wavelength)] = 0
	wavelength[np.isnan(wavelength)] = 0

	# 0.2'' < 0.46'' the cross dispersion size of a single shutter
	# This is in order to select a thin 1-2 pixels wide line, centered on the object
	eps = 0.2 / 3600
	distance = np.sqrt((slit.source_dec - dec) ** 2 + (slit.source_ra - ra) ** 2)
	distance[np.isnan(distance)] = 1000

	mask = np.logical_and(
		np.logical_not(np.isnan(slit.wavelength)),
		distance < eps
	)
	y, x = np.where(mask)
	# We fit a line and take the angle of the slope
	coeff, _ = cfit(lambda x, a, b: a * x + b, x, y)
	alpha = np.arctan(coeff[0]) * 180 / np.pi

	return (rotate(data, alpha, mode='constant',order=1),
			rotate(wavelength, alpha, mode='constant',order=1),
			rotate(ra, alpha, mode='constant',order=1),
			rotate(dec, alpha, mode='constant',order=1),
			rotate(slit.err, alpha, mode='constant',order=1))



def SelectSlice(slit):
	"""
	Selects 3 slices (2 background, 1 signal) by analysing a cross section, finding a pixel position for each peak,
	searching for a subpixel position, and slicing at the midpoints

	Params
	---------
	slit

	Returns
	---------
	slice_indices : list of arrays
		A list containing the lower and upper bound of each slit
	rotated : tuple of arrays

	"""
	rotated = rotateSlit(slit)
	data = rotated[0]
	data = np.ma.masked_invalid(data)

	# Get vertical cross section by summing horizontally
	horiz_sum = np.ma.mean(data,axis=1)
	horiz_err = np.ma.std(data, axis=1)

	# Determine 3 maxima for 3 slits
	peaks = []
	j = 2

	while not len(peaks) == 3:
		if j > 6:
			break
		peaks = find_peaks_cwt(horiz_sum,j)
		j += 1
	if not len(peaks) == 3:
		return None, rotated
	# Subpixel peaks
	peaks = np.sort(getPeaksPrecise(range(len(horiz_sum)),horiz_sum,horiz_err,peaks))

	if np.any(peaks > len(horiz_sum)) or np.any(peaks < 0):
		return None, rotated

	# Cut horizontally at midpoint between maxima -> 3 strips
	slice_indices = getPeakSlice(peaks,0,len(horiz_sum),horiz_sum)

	return slice_indices, rotated


def getPeaksPrecise(x,y,err,peaks):
	"""
	Returns the sub-pixel peaks
	"""
	try :
		coeff, err, info, msg, ier = cfit(slitletModel, x, y,sigma=err, p0=[*peaks,*y[peaks],0.5,0],full_output=True)
	except :
		logConsole("Can't find appropriate fit. Defaulting to input","ERROR")
		return peaks
	return np.array(coeff[:3])


def getPeakSlice(peaks,imin,imax,signal):
	"""
	Returns slices for a set of peaks
	"""
	d1 = (peaks[1] - peaks[0])/2
	d2 = (peaks[2] - peaks[1])/2

	xmin = np.array([
		smartRound(max(imin,peaks[0]-d1),signal),
		smartRound(peaks[1]-d1,signal),
		smartRound(peaks[2]-d2,signal)])

	xmax = np.array([
		smartRound(peaks[0]+d1,signal),
		smartRound(peaks[1]+d2,signal),
		smartRound(min(imax,peaks[2]+d2),signal)])

	if xmax[-1] > imax:
		xmax[-1] = imax

	return np.array([xmin,xmax]).T


def polynomialExtrapolation(img,cA,cB,slices,shutter_id):
	"""
	Extrapolates a full image using slices already fitted by polynomials.
	The data is supposed to be an image with the rows such that : [.,A,x,B,.]
	Where A and B have been fitted by a polynomial.
	x will then be calculated as a weighted sum of both A and B coefficients.

	Parameters
	----------
	img : 2D array
		The image to extrapolate
	cA : array
		The coefficients of the polynomial A
	cB : array
		The coefficients of the polynomial B
	slices : array
		A 2D array of coordinates of vertical slices. 1st index is the slice number, 2nd index is the starting/end point
	shutter_id : int
		An int between 0, 1 and 2, indicating which slice is x and which are A and B

	Returns
	-------
	img : 2D array
		The extrapolated image

	"""

	# Memo : the array img is such that for img[x,y], x (axis=0) is the vertical position, y (axis=1) is the horizontal position
	# we thus suppose img is, along x, [.,A,.,B,.]

	signal_indices = (slices[shutter_id][0],slices[shutter_id][1])

	# Middle case
	Y,X = np.indices(img[signal_indices[0]:signal_indices[1],:].shape)

	fitA = betterPolynomial([X,Y],*cA).reshape(X.shape)
	fitB = betterPolynomial([X,Y],*cB).reshape(X.shape)

	fit = (fitB + fitA)/2
	img[signal_indices[0]:signal_indices[1],:] = fit

	img[np.isnan(img)] = 0

	return img

def betterPolynomial(X,s,y0,*coeffs):
	"""
	We admit the convention that x is the horizontal direction, y the vertical direction
	Returns a 2D array, where the profile along Y is a gaussian, and the profile along X is a polynomial
	Those 2 profiles are then multiplied together

	Parameters
	----------
	X : [x,y
	s : gaussian sigma
	y0 : gaussian position
	coeffs : m=n+1 coeffs of the polynomial. We assume that coeffs[i] is such that we have coeffs[i] * x**i

	Returns
	-------
	2D array
	"""
	coeffs = np.array(coeffs)
	x,y = X

	# gaussian
	img = np.exp(-(y0-y)**2/(2*s**2))

	# repeat m times the X 2D array along 1st axis
	m = len(coeffs)
	x = x[np.newaxis, :, :]
	x = np.repeat(x, m, axis=0)

	# Make power series to the right shape
	power = np.arange(m)
	power = power[:, np.newaxis, np.newaxis]

	# Take the power along the 1st axis, multiply by coeffs along the same axis, sum along the axis
	polyn = np.power(x,power)
	polyn = polyn * coeffs[:, None, None]
	polyn = polyn.sum(axis=0)

	img = img * polyn
	return img.ravel()


