import numpy as np
from scipy.signal import find_peaks_cwt
from ..utils import *
from scipy.optimize import curve_fit as cfit, OptimizeWarning
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import stdatamodels.jwst.datamodels as dm
from astropy.stats import sigma_clip
from jwst.flatfield import FlatFieldStep
from jwst.pathloss import PathLossStep
from jwst.barshadow import BarShadowStep
from jwst.photom import PhotomStep
from jwst.resample import ResampleSpecStep
from scipy import interpolate
from jwst.master_background import nirspec_utils
from astropy.visualization import ZScaleInterval


def BetterBackgroundStep(name,saveBackgroundImage=False):
	"""
	 Creates a _bkg file from a _srctype file

	 Params
	 ---------
	 name : str
		Path to the file to open. Must be a _srctype
	saveBackgroundImage : bool
		If true, will save the background image as a fits

	"""
	if not "_srctype" in name:
		logConsole(f"{name.split('/')[-1]} not a _srctype file. Skipping...",source="WARNING")
		pass

	# 1st draft Algorithm :
	logConsole(f"Starting Custom Background Subtraction on {name.split('/')[-1]}",source="BetterBackground")
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

		shutter_id = WhichShutterOpen(slit.shutter_state)
		if shutter_id is None:
			logConsole("Not a 3 shutter slit!")
			precal[i].data = np.zeros_like(precal.slits[i].data)
			continue

		logConsole("Starting on modeling of background")
		fitted = modelBackgroundFromSlit(slit,shutter_id,precal.slits[i])

		if saveBackgroundImage:
			z = ZScaleInterval()
			z1, z2 = z.get_limits(slit.data)
			plt.figure(figsize=(16,4))
			plt.subplot(2,1,1)
			plt.imshow(precal.slits[i].data,origin="lower",interpolation="none",vmin=z1,vmax=z2)
			plt.subplot(2, 1, 2)
			plt.imshow(fitted, origin="lower", interpolation="none", vmin=z1, vmax=z2)
			plt.savefig(f"{precal.slits[i].slitlet_id}.png")
			plt.close()

		precal.slits[i].data = fitted
		logConsole("Background Calculated!")

	# Reverse the calibration
	logConsole("Reversing the Pre-Calibration...")
	background = PhotomStep.call(precal, inverse=True, source_type="EXTENDED")
	background = BarShadowStep.call(background, inverse=True, source_type="EXTENDED")
	background = PathLossStep.call(background, inverse=True, source_type="EXTENDED")
	background = FlatFieldStep.call(background, inverse=True)

	# The result is somehow inverted?
	background.write(name.replace("srctype","background"))
	result = nirspec_utils.apply_master_background(multi_hdu, background)

	multi_hdu.close()
	del multi_hdu
	del background
	del resampled
	del precal

	result.write(name.replace("srctype","bkg"))
	logConsole(f"Saving File {name.replace('srctype','bkg').split('/')[-1]}")

	pass

def modelBackgroundFromSlit(slit, shutter_id, precalSlit):

	Y, X = np.indices(slit.data.shape)
	_, _, dataLambda = slit.meta.wcs.transform("detector", "world", X, Y)

	Y, X = np.indices(precalSlit.data.shape)
	_, _, precalLambda = precalSlit.meta.wcs.transform("detector", "world", X, Y)

	# The slices are NOT the size of the individual shutter dispersion
	# They are chosen by analysing the vertical maxima and choosing a radius of lines above and beyond those maxima

	return modelBackgroundFromImage(precalSlit.data,
									precalLambda,
									slit.data,
									slit.err,
									dataLambda,
									shutter_id)

def modelBackgroundFromImage(preCalibrationData : np.ndarray,
							 preCalibrationWavelength : np.ndarray,
							 data : np.ndarray,
							 error : np.ndarray,
							 wavelength : np.ndarray,
							 shutter_id : int,
							 modelImage = None):
	"""
	Creates a 2D image model based on the pre-calibration wavelengths positions of the background

	Parameters
	----------
	preCalibrationData : np.ndarray, 2D array of the untreated image
	preCalibrationWavelength : np.ndarray, 2D array representing the wavelength at each pixel
	data : np.ndarray, 2D array of the treated image
	error : np.ndarray, 2D array of the error of the treated image
	wavelength : np.ndarray, 2D array representing the wavelength of the treated image
	shutter_id : int, ID of the shutter which contains the signal (0,1,2)
	modelImage : np.ndarray, 2D image of same size as data, representing the noiseless and envelope-less signal

	Returns
	-------
	np.ndarray, 2D array of a smooth model of background, or zeros if a fit cannot be made
	"""
	isModelValid = verifySimilarImages(data, modelImage)
	yModel = None

	slice_indices = SelectSlice(data)

	if slice_indices is None:
		logConsole("Data is empty")
		return np.zeros_like(preCalibrationData)

	#### TEST ####
	plt.figure()
	plt.imshow(data, origin="lower", vmin=0)
	plt.hlines(slice_indices,0,data.shape[1],linestyle="--",color="red")
	plt.fill_between([0, data.shape[1]], [slice_indices[shutter_id - 1][0], slice_indices[shutter_id - 1][0]],
	[slice_indices[shutter_id - 1][1], slice_indices[shutter_id - 1][1]], color='r', alpha=0.3)
	plt.fill_between([0, data.shape[1]], [slice_indices[shutter_id - 2][0], slice_indices[shutter_id - 2][0]],
					 [slice_indices[shutter_id - 2][1], slice_indices[shutter_id - 2][1]], color='r', alpha=0.3)
	##############

	x = extract1DBackgroundFromImage(wavelength, slice_indices, shutter_id)
	y = extract1DBackgroundFromImage(data,slice_indices,shutter_id)
	dy = extract1DBackgroundFromImage(error**2,slice_indices,shutter_id)

	mask = np.logical_or(
		np.logical_or(
			np.isnan(x),
			np.isnan(y)
		),
		np.isnan(dy)
	)

	if isModelValid:
		yModel = extract1DBackgroundFromImage(modelImage, slice_indices, shutter_id)
		mask = np.logical_or(
			mask,
			np.isnan(yModel)
		)
	mask = ~mask

	x = x[mask]
	y = y[mask]
	dy = dy[mask]

	_, indices = np.unique(x, return_index=True)
	indices = np.sort(indices)
	x = x[indices]
	y = y[indices]
	dy = dy[indices]

	if isModelValid:
		yModel = yModel[mask]
		yModel = yModel[indices]

	dy = np.sqrt(dy)

	# Sorts in rising wavelength order, ignores aberrant y values
	indices = np.argsort(x)
	x = x[indices]
	y = y[indices]
	dy = dy[indices]

	if isModelValid:
		yModel = yModel[indices]
		yModel = yModel[y>0]

	x = x[y > 0]
	dy = dy[y > 0]
	y = y[y > 0]

	# Weights, as a fraction of total sum, else it breaks the fitting
	w = 1 / dy
	w /= w.mean()

	if len(x) <= 5:
		logConsole("Not Enough Points to interpolate", source="WARNING")
		return np.zeros_like(preCalibrationData)

	if isModelValid:
		interp = makeInterpolation(x,y,w,showPlots=True,realData=yModel)
	else:
		interp = makeInterpolation(x,y,w)

	# The 2D background model obtained from the 1D spectrum
	return interp(preCalibrationWavelength)

def extract1DBackgroundFromImage(data : np.ndarray, slice_indices : iter, shutter_id : int) -> np.ndarray:
	"""
	Sums vertically the image on 2 horizontal slices and appends the 2
	Parameters
	----------
	data : 2D array, image
	slice_indices : 3x2 iterable, the 1st index is the n° of the slice, the 2nd is the [start,end] index
	shutter_id : int, 0 is for the top slice, 1 for the middle, 2 for the bottom

	Returns
	-------
	1D array, wavelength dependant

	"""
	return np.append(
					np.median(data[slice_indices[shutter_id - 1][0]:slice_indices[shutter_id - 1][1]], axis=0),
				  	np.median(data[slice_indices[shutter_id - 2][0]:slice_indices[shutter_id - 2][1]], axis=0)
	)


def makeInterpolation(x: np.ndarray, y: np.ndarray, w: np.ndarray, S = 45, showPlots = True, realData = None):
	"""
    Creates a spline interpolation / approximation of order 3.

    Parameters
    ----------
    x : wavelength 1D array
    y : corresponding data 1D array
    w : weights of the data points, inverse of their error
    S : float, defined as S = s / len(x), needs to be tweaked for a smoother fit
    showPlots : bool, if True plots the spline interpolation, the error in function of S and the number of knots in function of S
    realData : 1D array, the real data values corresponding to the points x

    Returns
    -------
    interp : a function which approximates the data
    """
	interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=S * len(w))

	if showPlots:
		mean = []
		knots = []
		interpolation_list = []
		n = 50
		Slist = 10 ** np.linspace(1, 3, n)

		# Precompute the interpolations
		for current_S in Slist:
			logConsole(f"Calculating spline for S = {current_S}")
			_ = interpolate.UnivariateSpline(x, y, w=w, k=3, s=current_S * len(w))
			interpolation_list.append(_)
			Y = _(x)
			mean.append(np.mean(abs(Y - realData)))
			knots.append(len(_.get_knots()))

		# Set up figure and subplots
		fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 8))

		# Init spline and data points
		line_spline, = ax1.plot(x, np.zeros_like(x), color='b', label="Spline Fit")
		ax1.errorbar(x, y, yerr=1 / w, color='k', marker='+', linestyle='None', label="Data Points")
		if isinstance(realData, np.ndarray) and len(realData) == len(x):
			ax1.plot(x, realData, color="red", label="True Signal")

		# List for knots
		knot_scat, = ax1.plot([], [], 'go', label="Knots")
		ax1.legend()
		ax1.set_xlabel("Wavelength")

		# Fourier Transform
		freqs = np.fft.fftshift(np.fft.fftfreq(len(x), x[1]-x[0]))
		ax2.plot(freqs, abs(np.fft.fft(y)), color="black", label="Data")
		if isinstance(realData, np.ndarray) and len(realData) == len(x):
			ax2.plot(freqs, abs(np.fft.fft(realData)), color="red", label="True Signal")
		splineFFT, = ax2.plot(freqs, np.zeros_like(freqs), color="b", label="Spline Fit")

		ax2.legend()
		ax2.set_xscale("log")
		ax2.set_yscale("log")
		ax2.set_xlabel("Frequency")


		# Plot mean error on second subplot
		ax3.plot(Slist, mean, label="Error", marker='+')
		vline_mean, = ax3.plot([], [], color='red', linestyle='--', label="Current S")
		ax3.set_xscale('log')
		ax3.set_yscale('log')
		ax3.set_xlabel("S")
		ax3.set_ylabel("Error")
		ax3.legend()

		# Plot number of knots on third subplot
		ax4.plot(Slist, knots, label="# Knots", color='purple')
		vline_knots, = ax4.plot([], [], color='red', linestyle='--', label="Current S")
		ax4.set_xscale('log')
		ax4.set_xlabel("S")
		ax4.set_ylabel("# Knots")
		ax4.legend()

		# Animation update function
		def update(frame):
			current_S = Slist[frame]
			interpolation = interpolation_list[frame]
			Y = interpolation(x)
			line_spline.set_ydata(Y)

			# Update knot locations
			knot_positions = interpolation.get_knots()
			knot_scat.set_data(knot_positions, interpolation(knot_positions))

			# Update vertical lines
			vline_mean.set_data([current_S, current_S], [min(mean), max(mean)])
			vline_knots.set_data([current_S, current_S], [min(knots), max(knots)])

			# FFT
			splineFFT.set_data(freqs, abs(np.fft.fft(Y)))

			return line_spline, vline_mean, vline_knots, knot_scat, splineFFT

		# Create the animation
		anim = FuncAnimation(fig, update, frames=n, blit=True)
		plt.show()

	return interp


def SelectSlice(slitData : np.ndarray) :
	"""
	Selects 3 slices (2 background, 1 signal) by analysing a cross-section, finding a pixel position for each peak,
	searching for a subpixel position, and slicing at the midpoints

	Params
	---------
	slitData, 2D array, is supposed to be aligned with the pixel grid, so a resampledStep is needed before that

	Returns
	---------
	slice_indices : list of arrays
		A list containing the lower and upper bound of each slit

	"""
	# The radius of each slice
	radius = 1
	data = sigma_clip(slitData, sigma=5, masked=True)

	# Get vertical cross-section by summing horizontally
	horiz_sum = data.mean(axis=1)

	# Naive guess of the peaks position as equal slices
	start_index = np.where(horiz_sum > 0)[0][0]
	end_index = np.where(horiz_sum > 0)[0][-1]
	n = end_index - start_index
	peaks = np.array([round(n / 6), round(n / 2), round(5 * n / 6)])

	# Subpixel peaks
	peaks = np.sort(getPeaksPrecise(np.arange(0,slitData.shape[0]),horiz_sum,peaks))
	print(peaks)
	return np.clip(getSliceFromPeaks(peaks, radius), 0, slitData.shape[0])


def getPeaksPrecise(x : np.ndarray, y : np.ndarray, peaks) -> np.ndarray:
	"""
	Gets a subpixel position for each peak
	Parameters
	----------
	x : spatial dependant, 1D array
	y : value, 1D array
	peaks : pixel position of the peaks, list

	Returns
	-------
	array of peak positions, subpixel if a gaussian fit was found, integer if not
	"""
	try :
		coeff, err, info, msg, ier = cfit(slitletModel, x, y, p0=[*peaks,*y[peaks],0.5,0], full_output=True, method="dogbox",
										  bounds=([0,0,0,0,0,0,0,0],[x.max(),x.max(),x.max(),np.inf,np.inf,np.inf,np.inf,np.inf]))
		#### TEST ####
		plt.figure()
		plt.scatter(x, y, marker='+', color='k')
		X = np.linspace(x.min(), x.max(), 100)
		plt.plot(X, slitletModel(X, *coeff))
		plt.vlines(peaks, 0, y.max(), color='g', linestyle='--')
		plt.vlines(coeff[:3], 0, y.max(), color='r', linestyle='--')
		##############
		return coeff[:3]

	except OptimizeWarning:
		logConsole("Can't find appropriate fit. Defaulting to input","ERROR")
		return peaks


def getSliceFromPeaks(peaks, n : int) -> np.ndarray:
	"""
	Gets the indices for 3 slices around the peaks
	Parameters
	----------
	peaks : list of peak positions
	n : radius

	Returns
	-------
	a 2x3 array of indices of 3 slices

	"""
	# Slice radius
	xmin = np.array(np.round([peaks[0]-n, peaks[1]-n, peaks[2]-n]), dtype=int)
	xmax = np.array(np.round([np.round(peaks[0]+n+1), np.round(peaks[1]+n+1), np.round(peaks[2]+n+1)]), dtype=int)

	return np.array([xmin,xmax]).T