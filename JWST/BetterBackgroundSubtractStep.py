import numpy as np
from scipy.signal import find_peaks_cwt
from utils import *
from scipy.optimize import curve_fit as cfit
import matplotlib.pyplot as plt
import stdatamodels.jwst.datamodels as dm
from scipy.ndimage import rotate
from astropy.stats import sigma_clip
from jwst.flatfield import FlatFieldStep
from jwst.pathloss import PathLossStep
from jwst.barshadow import BarShadowStep
from jwst.photom import PhotomStep
from jwst.resample import ResampleSpecStep
from scipy import interpolate
from jwst.master_background import nirspec_utils


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
	logConsole("Applying Pre-Calibration...")
	precal = FlatFieldStep.call(multi_hdu)
	precal = PathLossStep.call(precal,source_type="EXTENDED")
	precal = BarShadowStep.call(precal,source_type="EXTENDED")
	precal = PhotomStep.call(precal,source_type="EXTENDED")

	# This is only useful for the extraction of the 1D spectrum
	resampled = ResampleSpecStep.call(precal)

	# For a given _srctype, for every slit
	for i,slit in enumerate(resampled.slits):
		logConsole(f"Opened slitlet {slit.slitlet_id}")

		#TODO : Eventually, work on error propagation

		shutter_id = WhichShutterOpen(slit.shutter_state)
		if shutter_id is None:
			logConsole("Not a 3 shutter slit!")
			continue

		# The slices are NOT the size of the individual shutter dispersion
		# They are chosen by analysing the vertical maxima and choosing a radius of lines above and beyond those maxima
		slice_indices = SelectSlice(slit)

		logConsole("Starting on modeling of background")
		Y, X = np.indices(slit.data.shape)
		_, _, lamb = slit.meta.wcs.transform("detector", "world", X, Y)

		x = np.append(lamb[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1]].mean(axis=0),
					  lamb[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1]].mean(axis=0))

		y = np.append(slit.data[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1]].mean(axis=0),
					  slit.data[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1]].mean(axis=0))

		dy = np.append(slit.data[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1]].std(axis=0),
					   slit.data[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1]].std(axis=0))
		dy = dy**2
		dy += np.append((slit.err[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1]]**2).mean(axis=0),
						(slit.err[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1]]**2).mean(axis=0))
		dy = np.sqrt(dy)

		indices = np.argsort(x)

		# Sorts in rising wavelength order, ignores aberrant y values
		x = x[indices]
		y = y[indices]
		dy = dy[indices]
		x = x[y > 0]
		dy = dy[y > 0]
		y = y[y > 0]

		# Weights, as a fraction of total sum, else it breaks the fitting
		w = 1 / dy
		w /= w.mean()

		# The s value should usually not cause the fitting to fail
		# In the case it does, a larger, less harsh s value is used
		# This is done by verifying if the returned function is nan on one of the 10 points in the wavelength range
		interp = interpolate.UnivariateSpline(x, y, w=w, s=0.01, k=5, check_finite=True)
		_ = interp(np.linspace(x.min(), x.max(), 10))
		if not np.all(np.isfinite(_)):
			logConsole("Ideal spline not found. Defaulting to a spline of s=1")
			interp = interpolate.UnivariateSpline(x, y, w=w, s=1, k=5, check_finite=True)

		# The 2D background model obtained from the 1D spectrum
		Y, X = np.indices(precal[i].data.shape)
		_, _, lamb = precal[i].meta.wcs.transform("detector", "world", X, Y)
		fitted = interp(lamb)
		precal[i].data = fitted
		logConsole("Background Calculated!")

	# Reverse the calibration
	logConsole("Reversing the Pre-Calibration...")
	background = PhotomStep.call(precal, inverse=True)
	background = BarShadowStep.call(background, inverse=True)
	background = PathLossStep.call(background, inverse=True)
	background = FlatFieldStep.call(background, inverse=True)

	result = nirspec_utils.apply_master_background(multi_hdu, background)

	del background
	del resampled
	del precal

	result.write(name.replace("srctype","_bkg"))
	logConsole(f"Saving File {name.split('/')[-1]}")

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

	return (np.ma.masked_invalid(rotate(data, alpha, mode='constant',cval=np.nan,order=1)),
			np.ma.masked_invalid(rotate(wavelength, alpha, mode='constant',cval=np.nan,order=1)),
			np.ma.masked_invalid(rotate(ra, alpha, mode='constant',cval=np.nan,order=1)),
			np.ma.masked_invalid(rotate(dec, alpha, mode='constant',cval=np.nan,order=1)),
			np.ma.masked_invalid(rotate(slit.err, alpha, mode='constant',cval=np.nan,order=1)))



def SelectSlice(slit):
	"""
	Selects 3 slices (2 background, 1 signal) by analysing a cross section, finding a pixel position for each peak,
	searching for a subpixel position, and slicing at the midpoints

	Params
	---------
	slit, is supposed to be aligned with the pixel grid, so a resampledStep is needed before that

	Returns
	---------
	slice_indices : list of arrays
		A list containing the lower and upper bound of each slit
	rotated : tuple of arrays

	"""
	# The radius of each slice
	radius = 1

	data = sigma_clip(slit.data, sigma=5, masked=True)

	# Get vertical cross section by summing horizontally
	horiz_sum = data.mean(axis=1)

	# Determine 3 maxima for 3 slits
	peaks = []
	j = 2

	while not len(peaks) == 3:
		if j > 6:
			break
		peaks = find_peaks_cwt(horiz_sum,j)
		j += 1
	if not len(peaks) == 3 or np.any(peaks > len(horiz_sum)) or np.any(peaks < 0):
		logConsole("Can't find 3 spectra. Defaulting to equal slices", source="WARNING")
		start_indice = np.where(horiz_sum > 0)[0][0]
		end_indice = np.where(horiz_sum > 0)[0][-1]
		n = end_indice - start_indice
		xmin = np.array([round(n / 6)-radius, round(n/2)-radius, round(5*n/6)-radius]) + start_indice
		xmax = np.array([round(n / 6)+radius, round(n/2)+radius, round(5*n/6)+radius]) + start_indice + 1
		slice_indices = np.array([xmin, xmax]).T
		return slice_indices

	# Subpixel peaks
	peaks = np.sort(getPeaksPrecise(range(len(horiz_sum)),horiz_sum,peaks))


	slice_indices = getPeakSlice(peaks,radius)

	return slice_indices


def getPeaksPrecise(x,y,peaks):
	"""
	Returns the sub-pixel peaks
	"""
	try :
		coeff, err, info, msg, ier = cfit(slitletModel, x, y, p0=[*peaks,*y[peaks],0.5,0],full_output=True)
	except :
		logConsole("Can't find appropriate fit. Defaulting to input","ERROR")
		return peaks
	return np.array(coeff[:3])


def getPeakSlice(peaks,n):
	"""
	Returns slices for a set of peaks
	"""

	# Slice radius
	xmin = np.array([
		round(peaks[0]-n),
		round(peaks[1]-n),
		round(peaks[2]-n)])

	xmax = np.array([
		round(peaks[0]+n+1),
		round(peaks[1]+n+1),
		round(peaks[2]+n+1)])

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


