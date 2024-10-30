from scipy.interpolate import interp1d
from scipy.signal import find_peaks_cwt
from ..utils import *
from scipy.optimize import curve_fit as cfit
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
							 shutter_id : int):
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

	Returns
	-------
	np.ndarray, 2D array of a smooth model of background, or zeros if a fit cannot be made
	"""

	slice_indices = SelectSlice(data)


	##### TEST #####
	plt.figure(0)
	plt.hlines(np.array(slice_indices).ravel(), 0, data.shape[1], color='r')
	plt.xlim(data.shape[0], data.shape[1])
	################

	if slice_indices is None:
		logConsole("Data is empty")
		return np.zeros_like(preCalibrationData)

	x = extract1DBackgroundFromImage(wavelength, slice_indices, shutter_id)
	y = extract1DBackgroundFromImage(data,slice_indices,shutter_id)
	dy = extract1DBackgroundFromImage(error**2,slice_indices,shutter_id)

	mask = ~np.logical_or(
		np.logical_or(
			np.isnan(x),
			np.isnan(y)
		),
		np.isnan(dy)
	)
	x = x[mask]
	y = y[mask]
	dy = dy[mask]

	_, indices = np.unique(x, return_index=True)
	indices = np.sort(indices)
	x = x[indices]
	y = y[indices]
	dy = dy[indices]

	dy = np.sqrt(dy)

	# Sorts in rising wavelength order, ignores aberrant y values
	indices = np.argsort(x)
	x = x[indices]
	y = y[indices]
	dy = dy[indices]
	x = x[y > 0]
	dy = dy[y > 0]
	y = y[y > 0]

	# Weights, as a fraction of total sum, else it breaks the fitting
	w = 1 / dy
	w /= w.mean()

	if len(x) <= 5:
		logConsole("Not Enough Points to interpolate", source="WARNING")
		return np.zeros_like(preCalibrationData)

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


def makeInterpolation(x : np.ndarray, y : np.ndarray, w : np.ndarray):
	"""
	Creates a spline interpolation / approximation of order 5

	Parameters
	----------
	x : wavelength 1D array
	y : corresponding data 1D array
	w : weights of the data points, inverse of their error

	Returns
	-------
	interp : a function which approximates the data
	"""
	mean = []
	std = []
	knot_counts = []
	n = 50
	S_values = 10 ** np.linspace(0, 3, n)

	# Precompute mean, std, and knot counts for all S values
	for S in S_values:
		interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=S * len(w))
		Y = interp(x)
		mean.append(np.mean((Y - y)**2))
		std.append(np.std(Y - y))
		knot_counts.append(len(interp.get_knots()))  # Number of knots

	# Create a cubic interpolation for the true line (`real`) as needed
	real = interpolate.interp1d([0,2000,400,1300,450,800,1600], [200,10,140,100,40,150,60], kind='cubic', fill_value="extrapolate")

	# Set up the figure and subplots
	fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

	# Initial plot elements for the spline and data points
	line_spline, = ax1.plot(x, np.zeros_like(x), color='b', label="Spline Fit")
	scat_data = ax1.scatter(x, y, color='k', marker='+', label="Data Points")
	line_real, = ax1.plot(x, real(x), color='r', label="True Function")

	# Placeholder for knots; initially empty
	knot_scat, = ax1.plot([], [], 'go', label="Knots")  # Green dots for knots

	ax1.set_title("Spline Fitting for Varying S")
	ax1.legend()

	# Plot the mean error on the second subplot
	line_mean, = ax2.plot(S_values, mean, label="Mean Error", marker='+')
	vline_mean, = ax2.plot([], [], color='red', linestyle='--', label="Current S")

	ax2.set_xscale('log')
	ax2.set_yscale('log')
	ax2.set_xlabel("S")
	ax2.set_ylabel("Mean Error")
	ax2.legend()
	ax2.vlines(1, 0, 10, color='red')
	ax2.vlines([1 - np.sqrt(2 / len(x)), 1 + np.sqrt(2 / len(x))], 0, 10, color='red', linestyle='--')

	# Plot the number of knots on the third subplot
	line_knots, = ax3.plot(S_values, knot_counts, label="Number of Knots", marker='o', color='purple')
	vline_knots, = ax3.plot([], [], color='red', linestyle='--', label="Current S")

	ax3.set_xscale('log')
	ax3.set_xlabel("S")
	ax3.set_ylabel("Number of Knots")
	ax3.legend()
	ax3.set_title("Number of Knots vs S")
	ax3.hlines(len(x),S_values.min(), S_values.max(), color='red', linestyle='--')

	# Animation update function
	def update(frame):
		S = S_values[frame]

		# Update spline for the current S
		interp = interpolate.UnivariateSpline(x, y, w=w, k=3, s=S * len(w))
		Y = interp(x)
		line_spline.set_ydata(Y)

		# Update knot locations
		knots = interp.get_knots()
		knot_scat.set_data(knots, interp(knots))  # Plot knots in green

		# Update vertical line in mean error plot
		vline_mean.set_data([S, S], [min(mean), max(mean)])

		# Update vertical line in knot count plot
		vline_knots.set_data([S, S], [min(knot_counts), max(knot_counts)])

		# Update title with current S
		ax1.set_title(f"Spline Fitting for S = {S:.2e}")
		return line_spline, vline_mean, vline_knots, knot_scat

	# Create the animation
	ani = FuncAnimation(fig, update, frames=n, blit=True)

	# Display or save the animation
	plt.show()  # To display in a Jupyter Notebook
	# ani.save('spline_animation.mp4', writer='ffmpeg', fps=5)  # Uncomment to save as mp4
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

	# Determine 3 maxima for 3 slits
	peaks = []
	# Looks for peaks of different width
	j = 2
	while not len(peaks) == 3:
		if j > 6:
			break
		peaks = find_peaks_cwt(horiz_sum,j)
		j += 1

	# Gets equal slices if no peaks are found
	if not len(peaks) == 3 or np.any(peaks > len(horiz_sum)) or np.any(peaks < 0):
		logConsole("Can't find 3 spectra. Defaulting to equal slices", source="WARNING")
		# Somehow this edge case exists
		if np.all(horiz_sum == 0):
			return None
		start_index = np.where(horiz_sum > 0)[0][0]
		end_index = np.where(horiz_sum > 0)[0][-1]
		n = end_index - start_index
		xmin = np.array([round(n / 6)-radius, round(n/2)-radius, round(5*n/6)-radius]) + start_index
		xmax = np.array([round(n / 6)+radius, round(n/2)+radius, round(5*n/6)+radius]) + start_index + 1
		return np.array([xmin, xmax]).T

	# Subpixel peaks
	peaks = np.sort(getPeaksPrecise(np.array(range(len(horiz_sum))),horiz_sum,peaks))

	return np.clip(getSliceFromPeaks(peaks, radius), 0, slitData.shape[0])


def getPeaksPrecise(x : np.ndarray, y : np.ndarray, peaks) -> np.ndarray:
	"""
	Gets a subpixel position for each peak
	Parameters
	----------
	x : wavelength dependant, 1D array
	y : value, 1D array
	peaks : wavelength position of the peaks, list

	Returns
	-------
	array of peak positions, subpixel if a gaussian fit was found, integer if not
	"""
	try :
		coeff, err, info, msg, ier = cfit(slitletModel, x, y, p0=[*peaks,*y[peaks],0.5,0],full_output=True)
	except :
		logConsole("Can't find appropriate fit. Defaulting to input","ERROR")
		return np.array(peaks)
	return np.array(coeff[:3])


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
	xmin = np.array([
		round(peaks[0]-n),
		round(peaks[1]-n),
		round(peaks[2]-n)])

	xmax = np.array([
		round(peaks[0]+n+1),
		round(peaks[1]+n+1),
		round(peaks[2]+n+1)])

	return np.array([xmin,xmax]).T