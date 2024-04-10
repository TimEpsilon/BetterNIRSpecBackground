import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt

from utils import *



def BetterBackgroundStep(name,threshold=0.3):
	"""
	 Creates a _bkg file from a _srctype file

	 Params
	 ---------
	 name : str
	 	Path to the file to open. Must be a _srctype
	 threshold : float
	 	A number between 0 and 1. Represents the proportion of high intensity pixels that will be cutoff, 
	 	1 being no pixels removed, 0 being every pixel removed
	"""
	if not "_srctype" in name:
		logConsole(f"{name.split("/")[-1]} not a _srctype file. Skipping...")
		pass

	# 1st draft Algorithm :	
	logConsole(f"Starting Custom Bakcground Substraction on {name.split("/")[-1]}")
	multi_hdu = fits.open(name)

	# For a given _srctype, for every SCI inside
	for i,hdu in enumerate(multi_hdu):
		if not hdu.name == 'SCI':
			continue
		hdr = hdu.header
		data = np.ma.array(hdu.data, mask=np.isnan(hdu.data))

		#TODO : Eventually, work on error propagation

		shutter_id = WhichShutterOpen(hdr)
		if shutter_id == None:
			continue

		logConsole(f"Extension {i} is SCI. Open shutter is {shutter_id+1}")

		slice_indices = SelectSlice(data)

		# Get 2 background strips
		bkg1 = data[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1],:]
		bkg2 = data[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1],:]


		# Determine non background sources : sudden spikes, high correlation with source strip, etc -> flag pixels
		# TODO : Better background detection
		mask1 = bkg1 > bkg1.min() + (bkg1.max() - bkg1.min())*threshold
		mask2 = bkg2 > bkg2.min() + (bkg2.max() - bkg2.min())*threshold

		mask1 = np.logical_or(mask1, bkg1 == np.nan)
		mask2 = np.logical_or(mask2, bkg2 == np.nan)

		bkg1_keep = np.ma.array(bkg1,mask=mask1,fill_value=np.nan)
		bkg2_keep = np.ma.array(bkg2,mask=mask2,fill_value=np.nan)

		master_background = [bkg1_keep,bkg2_keep]


		# Remove pixels + interpolate on a given strip (ignore source strip)
		bkg_interp = []

		for bkg in master_background:
			mask_bkg = np.ma.getmask(bkg)
			non_nan = np.where(np.logical_not(mask_bkg))
			x = non_nan[0]
			y = non_nan[1]
			z = bkg[non_nan]

			interp = IDWExtrapolation(np.c_[x, y], z, power=10)
			
			X = np.arange(bkg.shape[0])
			Y = np.arange(bkg.shape[1])
			YY,XX = np.meshgrid(Y,X)
			bkg_interp.append(interp(XX,YY))

		new_data = np.copy(data)
		new_data[:,:] = np.nan

		new_data[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1],:] = bkg_interp[0]
		new_data[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1],:] = bkg_interp[1]

		non_nan = np.where(np.logical_not(np.isnan(new_data)))

		x = non_nan[0]
		y = non_nan[1]
		z = new_data[non_nan]


		interp = IDWExtrapolation(np.c_[x, y], z, power=5)
		
		X = np.arange(new_data.shape[0])
		Y = np.arange(new_data.shape[1])
		YY,XX = np.meshgrid(Y,X)
		new_data = interp(XX,YY)

		hdu.data = data - new_data



		"""
		plt.figure()

		plt.subplot(4,1,1)
		plt.title("Raw")
		plt.imshow(data,interpolation='none',vmin=data.min(),vmax=data.max())
		plt.subplot(4,1,2)
		plt.title("Bkg1")
		plt.imshow(bkg_interp[0],interpolation='none',vmin=data.min(),vmax=data.max())
		plt.subplot(4,1,3)
		plt.title("Bkg2")
		plt.imshow(bkg_interp[1],interpolation='none',vmin=data.min(),vmax=data.max())
		plt.subplot(4,1,4)
		plt.title("Substracted")
		plt.imshow(new_data,interpolation='none',vmin=data.min(),vmax=data.max())
		plt.hlines(slice_indices.ravel(),0,data.shape[1],color='red',linestyle='dashed',linewidth=1)

		plt.savefig(str(i)+".png")
		plt.close()
		"""


	multi_hdu.writeto(name.replace("_srctype","_bkg"),overwrite=True)
	multi_hdu.close()
	pass


def SelectSlice(data):
	"""
	Selects 3 slices (2 background, 1 signal) by analysing a cross section, finding a pixel position for each peak, 
	searching for a subpixel position, and slicing at the midpoints

	Params
	---------
	data : 2D array
		An individual image obtained from an individual "SCI" extension in a fits

	Returns
	---------
	slice_indices : list of arrays
		A list containing the lower and upper bound of each slit

	"""
	# Get vertical cross section by summing horizontally
	horiz_sum = np.mean(data,axis=1)

	# Determine 3 maxima for 3 slits
	peaks = []
	j = 2
	while not len(peaks) == 3:
		if j > 6:
			break
		peaks = find_peaks_cwt(horiz_sum,j)
		j += 1
	if not len(peaks) == 3:
		continue
	# Subpixel peaks
	peaks = np.sort(getPeaksPrecise(range(len(horiz_sum)),horiz_sum,peaks))

	# Cut horizontally at midpoint between maxima -> 3 strips
	slice_indices = getPeakSlice(peaks,0,len(horiz_sum))

	return slice_indices


"""
Returns the sub-pixel peaks
"""
def getPeaksPrecise(x,y,peaks):
    coeff, err, info, msg, ier = cfit(slitletModel, x, y, p0=[*peaks,*y[peaks],0.5],full_output=True)
    return np.array(coeff[:3])

"""
Returns slices for a set of peaks
"""
def getPeakSlice(peaks,imin,imax):
	d1 = (peaks[1] - peaks[0])/2
	d2 = (peaks[2] - peaks[1])/2

	xmin = np.array([round(max(imin,peaks[0]-d1)),round(peaks[1]-d1),round(peaks[2]-d2)])
	xmax = np.array([round(peaks[0]+d1), round(peaks[1]+d2), round(min(imax,peaks[2]+d2))])

	return np.array([xmin,xmax]).T


def IDWExtrapolation(xy, ui, power=1):
	"""
	Rough implementation of the Inverse Distance Weighting algorithm

	Parameters
	----------
	xy : ndarray, shape (npoints, ndim)
		Coordinates of data points
	ui : ndarray, shape (npoints)
		Values at data points

	Returns
	-------
	func : callable
	"""
	x = xy[:, 0]
	y = xy[:, 1]

	def new_f(xx,yy):

		xy_ravel = np.column_stack((xx.ravel(),yy.ravel()))
		x_ravel = xy_ravel[:, 0]
		y_ravel = xy_ravel[:, 1]

		X1, X2 = np.meshgrid(x,x_ravel)
		Y1, Y2 = np.meshgrid(y,y_ravel)

		d = ((X1-X2)**2 + (Y1-Y2)**2).T

		w = d**(-power/2)

		w_ui_sum = ui[:, None]*w
		w_ui_sum = w_ui_sum.sum(axis=0)

		wsum = w.sum(axis=0)

		result = w_ui_sum / wsum
		result = result.reshape(np.shape(xx))
		result[x,y] = ui

		return result

	return new_f


def NNExtrapolation(xy, z):
	"""
	Code From https://docs.scipy.org/doc/scipy/tutorial/interpolate/extrapolation_examples.html

	CT interpolator + nearest-neighbor extrapolation.

	Parameters
	----------
	xy : ndarray, shape (npoints, ndim)
		Coordinates of data points
	z : ndarray, shape (npoints)
		Values at data points

	Returns
	-------
	func : callable
		A callable object which mirrors the CT behavior,
		with an additional neareast-neighbor extrapolation
		outside of the data range.
	"""
	x = xy[:, 0]
	y = xy[:, 1]
	f = CT(xy, z)

	# this inner function will be returned to a user
	def new_f(xx, yy):
		# evaluate the CT interpolator. Out-of-bounds values are nan.
		zz = f(xx, yy)
		nans = np.isnan(zz)

		if nans.any():
			# for each nan point, find its nearest neighbor
			inds = np.argmin(
				(x[:, None] - xx[nans])**2 +
				(y[:, None] - yy[nans])**2
				, axis=0)
			# ... and use its value
			zz[nans] = z[inds]
		return zz

	return new_f