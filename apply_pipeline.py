import os

os.environ['CRDS_PATH'] = '/home/tim/crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from jwst.pipeline import Detector1Pipeline
from jwst.pipeline import Spec2Pipeline
from jwst.pipeline import Spec3Pipeline
from glob import glob
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt

from jwst.master_background import MasterBackgroundMosStep as BkgMosStep
import stdatamodels.jwst.datamodels as dm

from utils import *


"""
 Creates a _bkg file from a _srctype file
"""
def BetterBackground(name):
	# 1st draft Algorithm :	
	multi_hdu = fits.open(name)

	# For a given _srctype, for every SCI inside
	for i,hdu in enumerate(multi_hdu):
		if not hdu.name == 'SCI':
			continue
		hdr = hdu.header
		data = np.ma.array(hdu.data, mask=np.isnan(hdu.data))

		shutter_id = WhichShutterOpen(hdr)
		if shutter_id == None:
			continue

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
		peaks = np.sort(getPeaksPrecise(range(len(horiz_sum)),horiz_sum,peaks))

		# Cut horizontally at midpoint between maxima -> 3 strips
		slice_indices = getPeakSlice(peaks,0,len(horiz_sum))

		# Get 2 background strips
		#TODO : Flag main slit and outside bkg slits, interpolate those values, substract this new image
		src = data[slice_indices[shutter_id][0]:slice_indices[shutter_id][1],:]
		bkg1 = data[slice_indices[shutter_id-1][0]:slice_indices[shutter_id-1][1],:]
		bkg2 = data[slice_indices[shutter_id-2][0]:slice_indices[shutter_id-2][1],:]


		# Determine non background sources : sudden spikes, high correlation with source strip -> flag pixels
		# TODO : Better background detection
		threshold = 0.3
		mask1 = bkg1 > bkg1.min() + (bkg1.max() - bkg1.min())*threshold
		mask2 = bkg2 > bkg2.min() + (bkg2.max() - bkg2.min())*threshold

		bkg1_keep = np.ma.array(bkg1,mask=mask1,fill_value=np.nan)
		bkg2_keep = np.ma.array(bkg2,mask=mask2,fill_value=np.nan)

		height = min(bkg1_keep.shape[0],bkg2_keep.shape[0])
		print(height,data.shape[0],slice_indices[0][1]-slice_indices[1][1])

		bkg_master = np.ma.dstack((bkg1_keep[:height,:],bkg2_keep[:height,:])).mean(axis=2)
		mask_master = np.ma.getmask(bkg_master)


		# Remove pixels + interpolate on a given strip (ignore source strip)
		non_nan = np.where(np.logical_not(mask_master))
		x = non_nan[0]
		y = non_nan[1]
		z = bkg_master[non_nan]

		interp = NNExtrapolation(np.c_[x, y], z)
		
		X = np.arange(bkg_master.shape[0])
		Y = np.arange(bkg_master.shape[1])
		YY,XX = np.meshgrid(Y,X)
		bkg_interp = interp(XX,YY)

		

		plt.figure()

		plt.subplot(4,1,1)
		plt.title("Background 1")
		plt.imshow(bkg1,interpolation='none',vmin=bkg1.min(),vmax=bkg1.max())
		plt.subplot(4,1,2)
		plt.title("Background 2")
		plt.imshow(bkg2,interpolation='none',vmin=bkg2.min(),vmax=bkg2.max())
		plt.subplot(4,1,3)
		plt.title("Master Background")
		plt.imshow(bkg_master,interpolation='none',vmin=bkg1.min(),vmax=bkg1.max())
		plt.subplot(4,1,4)
		plt.title("Inverse Distance Weighting")
		interp = IDWExtrapolation(np.c_[x,y], z, power = 8)
		bkg_interp = interp(XX,YY)
		plt.imshow(bkg_interp,interpolation='none',vmin=bkg1.min(),vmax=bkg1.max())

		plt.savefig(str(i)+".png")
		plt.close()
		
	multi_hdu.close()

	return multi_hdu


BetterBackground("jw01345063001_03101_00001_nrs1_srctype.fits")

exit()



working_dir = "./mastDownload/JWST/"

#########
# Monkey Patch Master Background method
#########

#BkgMosStep.process = BetterMasterBackgroundMosStep
#logConsole("Overriding Master Background method")


logConsole(f"Found {len(os.listdir(working_dir))} folders")
for folder in os.listdir(working_dir):
	path = working_dir + folder + "/"
	logConsole(f"Starting on {folder}")

	##########
	# Stage 1
	##########


	uncal_list = glob(path+"*_uncal.fits")
	logConsole(f"Found {len(uncal_list)} uncalibrated files")

	for n,uncal in enumerate(uncal_list):
		logConsole(f"Starting Stage 1 ({n+1}/{len(uncal_list)})")
		if os.path.exists(uncal.replace("uncal","rate")):
			continue
		# Steps Params By Pablo Arrabal Haro
		steps = {
                'jump': {'expand_large_events': True,
                         # 1st flag groups after jump above DN threshold.
                         'after_jump_flag_dn1': 0,
                         # 1st flag groups after jump groups within
                         # specified time.
                         'after_jump_flag_time1': 0,
                         # 2nd flag groups after jump above DN threshold.
                         'after_jump_flag_dn2': 0,
                         # 2nd flag groups after jump groups within
                         # specified time.
                         'after_jump_flag_time2': 0,
                         # Minimum required area for the central saturation
                         # of snowballs.
                         'min_sat_area': 15.0,
                         # Minimum area to trigger large events processing.
                         'min_jump_area': 15.0,
                         # The expansion factor for the enclosing circles
                         # or ellipses.
                         'expand_factor': 2.0},
                }
		det1 = Detector1Pipeline(steps=steps)
		det1.jump.maximum_cores='half'
		det1.ramp_fit.maximum_cores='half'
		det1.save_results = True
		det1.output_dir = path
		det1.run(uncal)

		det1 = None

	##########
	# Stage 2
	##########

	logConsole(f"Stage 1 Finished. Preparing Stage 2")
	rate_list = glob(path+"*_rate.fits")
	logConsole(f"Found {len(rate_list)} countrate files")

	for n,rate in enumerate(rate_list):
		logConsole(f"Starting Stage 2 ({n+1}/{len(rate_list)})")
		if os.path.exists(rate.replace("rate","srctype")):
			continue
		steps={'srctype': {'save_results':True},
				'photom': {'skip':True},
				'flat_field': {'skip':True},
				'master_background_mos': {'skip':True},
				'wavecorr': {'skip':True},
				'pathloss': {'skip':True},
				'barshadow': {'skip':True},
				'pixel_replace': {'skip':True},
				'extract_1d': {'skip':True},
				'cube_build': {'skip':True},
				'resample_spec': {'skip':True}}

		spec2 = Spec2Pipeline(steps=steps)
		spec2.save_results = True
		spec2.output_dir = path
		spec2.bkg_subtract.skip = True
		
		spec2.run(rate)

		spec2 = None


	##########
	# TODO : Master Background
	# 0 - Apply Spec2 up until srctype (included) and save them
	# 1 - Modify the _srctype files, all spectra
	# 2 - Create a _bkg file, exact copy
	# 3 - Modify the spec2_asn.json files in order to apply to those _bkg
	# 4 - Apply the rest of Spec2 to spec2_asn.json
	# 5 - Apply Spec3 to the spec3_asn.json
	##########



	##########
	# Stage 3
	##########

	logConsole(f"Stage 2 Finished. Preparing Stage 3")
	asn_list = glob(path+"*_spec2_*_asn.json")
	logConsole(f"Found {len(asn_list)} association files")

	for n,asn in enumerate(asn_list):
		logConsole(f"Starting Stage 3 ({n+1}/{len(asn_list)})")
		Spec3Pipeline.call(asn,save_results=True,output_dir=path)

	break

