An experimental new way to subtract the background of **NIRSpec** images using interpolation of background slits in a 3 slits slitlet.


To install the conda environment :

```conda env create -f environment.yml```

The environment variable ```os.environ['CRDS_PATH']``` defined in ```apply_pipeline.py```and ```MainPipeline.py``` should be modified to the path where you wish to install the calibration files for your data.
