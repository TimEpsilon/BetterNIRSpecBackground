###############################################################################
# Simple script to download and arrange specific data products from a
# particular observing program in MAST.
#
# Author: Pablo Arrabal Haro.
# Last update: 19 March 2024.
###############################################################################

from astroquery.mast import Observations
import numpy as np
import os
import time
from astropy.table import vstack
from astropy.io import fits
import pandas as pd


def sec_to_dhms(secs):
    """
    Function to show any time in seconds in a days:hours:mins:secs format.

    Arguments:
    secs -- float. Time in seconds.

    Returns:
    fmt_time -- str. Time reformatted to days:hours:mins:secs.

    """

    fmt_time = '%.2fs' % secs
    if secs >= 60:
        mins = secs // 60.
        secs = secs % 60.
        fmt_time = '%dm:%.2fs' % (mins, secs)
        if mins >= 60:
            hours = mins // 60.
            mins = mins % 60.
            fmt_time = '%dh:%dm:%.2fs' % (hours, mins, secs)
            if hours >= 60:
                days = hours // 24.
                hours = hours % 24.
                fmt_time = '%dd:%dh:%dm:%.2fs' % (days, hours, mins, secs)

    return fmt_time


# Directory to save the retrieved files.
savedir = './test'
if not os.path.exists(savedir):
    os.makedirs(savedir)

# =========================================================================
# Downloading data from MAST. CEERS NIRSpec case.
# =========================================================================

# NOTE: It takes ~2h:45min to download all the CEERS NIRSpec uncal data.
#       ~9 min to download a single observation (e.g., P8 prism).
#       The bottleneck is the compilation of files, not the download itself.

t1 = time.time()

print('Sending data query...')
obs_table = Observations.query_criteria(
    obs_collection='JWST',
    instrument_name='NIRSPEC/MSA',
    # filters=['CLEAR'],
    # target_name='CEERS-NIRSPEC-P8-PRISM-MSATA',
    proposal_id=['1345'])[0:10]

print('Extracting file names and information '
      '(this can take a variable time, from some minutes to a few hours, '
      'depending on the size of the query)...')
product_list = [Observations.get_product_list(obs) for obs in obs_table]
products = vstack(product_list)

# Pick your desired subproducts.
# products_to_download = ['UNCAL', 'RATE', 'MSA', ...]
products_to_download = ['UNCAL', 'MSA']

for p_type in products_to_download:
    idx = np.where((products['productSubGroupDescription'] == p_type))[0]
    data_names = np.unique(products['dataURI'][idx])
    print('\n')
    print('Downloading %s files...' % p_type, data_names)
    print('\n')
    for name in data_names:
        filename = name.split('/')[-1]
        print('wget https://mast.stsci.edu/api/v0.1/Download/file?uri=%s'
                  ' -O %s/%s' % (name, savedir, filename))

t2 = time.time()
print('Download completed. It took %s.' % sec_to_dhms(t2 - t1))


# =========================================================================
# Arranging data. CEERS NIRSpec case.
# =========================================================================

# # Extracting FITS files from individual folders into a common folder.
# # Not needed for script download, but useful in case of manual download of
# # a subset of files from the MAST portal.
# for root, dirs, files in os.walk(savedir):
#     for file_ in files:
#         file_n = os.path.join(root, file_)
#         if '.fits' in file_n:
#             os.system('mv %s %s/' % (file_n, savedir))
#
# exts = ['uncal']
# for p in [4, 5, 7, 8, 9, 10, 11, 12]:
#     all_files = ['%s/%s' % (savedir, x) for x in os.listdir(savedir)
#                  if '.fits' in x and 'msa.fits' not in x]
#
#     if len(all_files) == 0:
#         break
#
#     # Compiling files for the pointing.
#     p_files = []
#     for file in all_files:
#         head = fits.open(file)[0].header
#         if 'P%d' % p in head['OBSLABEL']:
#             p_files.append(file)
#
#     if len(p_files) == 0:
#         continue
#
#     pdir = '%s/CEERS_p%d' % (savedir, p)
#     if not os.path.exists(pdir):
#         os.makedirs(pdir)
#
#     for ext in exts:
#         extdir = '%s/%s' % (pdir, ext)
#         if not os.path.exists(extdir):
#             os.makedirs(extdir)
#
#         for p_file in p_files:
#             if '_%s.fits' % ext in p_file:
#                 os.system('mv %s %s/' % (p_file, extdir))


# =========================================================================
# Moving TA and MSATA images apart. CEERS NIRSpec case.
# =========================================================================

# exts = ['uncal']
# keys = ['FILENAME', 'INSTRUME', 'DETECTOR', 'FILTER', 'GRATING',
#         'EXP_TYPE', 'MSAMETFL', 'MSAMETID', 'MSACONID']
#
# for p in [4, 5, 7, 8, 9, 10, 11, 12]:
#
#     pdir = '%s/CEERS_p%d' % (savedir, p)
#     if not os.path.exists(pdir):
#         continue
#
#     for ext in exts:
#         extdir = '%s/%s' % (pdir, ext)
#
#         files = np.sort(['%s/%s' % (extdir, x) for x in os.listdir(extdir)
#                          if '_%s.fits' % ext in x])
#
#         head_info = []
#         for file in files:
#             fit = fits.open(file)
#
#             head = fit[0].header
#
#             aux_head = []
#             for key in keys:
#                 try:
#                     aux_head.append(head[key])
#                 except KeyError:
#                     aux_head.append(np.nan)
#
#             head_info.append(aux_head)
#
#             if head['EXP_TYPE'] != 'NRS_MSASPEC':
#                 extrasdir = '%s/extras' % extdir
#                 if not os.path.exists(extrasdir):
#                     os.makedirs(extrasdir)
#
#                 os.system('mv %s %s/%s' % (file, extrasdir,
#                                            file.split('/')[-1]))
#
#         head_pd = pd.DataFrame(head_info, columns=keys)
#         head_pd.to_csv('%s/headers_summary_%s.csv' % (extdir, ext),
#                        index=False)


# =========================================================================
# Splitting by disperser. CEERS NIRSpec case.
# =========================================================================

# exts = ['uncal']
# for p in [4, 5, 7, 8, 9, 10, 11, 12]:
#
#     pdir = '%s/CEERS_p%d' % (savedir, p)
#     if not os.path.exists(pdir):
#         continue
#
#     for ext in exts:
#         extdir = '%s/%s' % (pdir, ext)
#
#         for disp in ['PRISM', 'G140M', 'G235M', 'G395M']:
#             files = np.sort(['%s/%s' % (extdir, x)
#                              for x in os.listdir(extdir)
#                              if '_%s.fits' % ext in x])
#
#             if len(files) == 0:
#                 break
#
#             dispdir = '%s/%s' % (extdir, disp)
#             if not os.path.exists(dispdir):
#                 os.makedirs(dispdir)
#
#             for file in files:
#                 fit = fits.open(file)
#
#                 head = fit[0].header
#                 if disp in head['GRATING']:
#                     os.system('mv %s %s/%s' % (file, dispdir,
#                                                file.split('/')[-1]))

# =========================================================================
# Ring when finished.
os.system('aplay /home/parrabalh/Documents/Thermomix_alarm.wav')
