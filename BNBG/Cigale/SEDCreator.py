from matplotlib.lines import Line2D
from pcigale.warehouse import SedWarehouse
import matplotlib.pyplot as plt
import numpy as np
import logging
from matplotlib.widgets import Slider
from copy import deepcopy

# Logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

WAREHOUSE = SedWarehouse()

SED_PARAMETERS = {
	'sfhdelayed': {
		'tau_main': np.linspace(10,5000, 1000),
		'age_main': np.linspace(100, 13_000, 1000),
		'tau_burst': np.linspace(100, 10_000, 100),
		'age_burst': np.linspace(1, 100, 1000),
		'f_burst': np.linspace(0, 1, 100),
	},
	'bc03': {
		'imf': [0,1],
		'metallicity': [0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05],
		'separation_age': np.linspace(1,100,100),
	},
	'nebular': {
		'logU': [-4.0, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6,
				 -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1.0],
		'zgas' : [0.0001, 0.0004, 0.001, 0.002, 0.0025, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.011,
				  0.012, 0.014, 0.016, 0.019, 0.020, 0.022, 0.025, 0.03, 0.033, 0.037, 0.041, 0.046, 0.051],
		'f_esc': np.linspace(0,1,100),
		'f_dust': np.linspace(0,1,100),
		'lines_width': np.linspace(100,400,10),
		'ne': [10, 100, 1000],
		'emission': True,
		'line_list': " ",
	},
	'dustatt_modified_starburst': {
		'E_BV_lines': np.geomspace(1e-5, 5, 100),
		'E_BV_factor': np.linspace(0,5, 100),
		'uv_bump_wavelength': 217.5,  # fixed
		'uv_bump_width': 35.0,  # fixed
		'uv_bump_amplitude': np.linspace(0, 3, 100),
		'powerlaw_slope': np.linspace(-2, 2, 100),
		'Ext_law_emission_lines': [1, 2, 3],  # fixed
		'Rv': np.linspace(2, 4, 50),
	},
	'dl2014' : {
		'qpah': [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58, 5.26, 5.95, 6.63, 7.32],
		'umin': [0.100, 0.120, 0.150, 0.170, 0.200, 0.250, 0.300, 0.350, 0.400, 0.500, 0.600, 0.700, 0.800,
				 1.000, 1.200, 1.500, 1.700, 2.000, 2.500, 3.000, 3.500, 4.000, 5.000, 6.000, 7.000, 8.000,
				 10.00, 12.00, 15.00, 17.00, 20.00, 25.00, 30.00, 35.00, 40.00, 50.00],
		'alpha': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
				  2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0],
		'gamma': np.linspace(0, 1, 100),
	},
	'redshifting': {
		'redshift': np.linspace(0, 25, 2501)
	}
}
def generateSED(parameters : dict):
	sed = WAREHOUSE.get_sed(
		module_list=list(parameters.keys()),
		parameter_list=[v for v in parameters.values()])
	return sed

def createPlot():
	fig = plt.figure(figsize=(16, 10))

	ax_spectra = fig.add_axes((0.05, 0.05, 0.7, 0.9), xlabel=r"$\lambda$ (µm)", ylabel=r"$L_\lambda (W/nm)$", xscale="log", yscale="log")
	ax_spectra.grid(True, "both", alpha=0.5, linestyle="dashed")
	ax_spectra.set_xlim(10**-2, 10**6)
	ax_spectra.set_ylim(1.e6, 1.e30)

	ax_sfh = fig.add_axes((0.77, 0.7, 0.18, 0.25), xlabel=r"$t$ (Gyr)", ylabel=r"$SFR$ ($M_\odot / yr$)")
	ax_sfh.grid(True, "both")
	ax_sfh.yaxis.set_label_position('right')
	ax_sfh.yaxis.set_ticks_position('right')

	def _createSlider(ypos, name, N, ninit=0):
		ax = fig.add_axes((0.8, ypos, 0.16, 0.015))
		slider = Slider(ax, label=name, valmin=0, valmax=N-1, valinit=ninit, valstep=1)
		slider.valtext.set_visible(False)
		text = ax.text(1.02, ypos, "", transform=ax.transAxes)
		return slider, text

	sed_sliders = {
		'sfhdelayed': {
			'tau_main': _createSlider(0.60, r"$\tau_{main}$", 1000, 100),
			'age_main': _createSlider(0.58, r"$age_{main}$", 1000, 400),
			'tau_burst': _createSlider(0.56, r"$\tau_{burst}$", 100, 99),
			'age_burst': _createSlider(0.54, r"$age_{burst}$", 1000, 500),
			'f_burst': _createSlider(0.52, r"$f_{burst}$", 100, 10),
		},
		'bc03': {
			'imf': _createSlider(0.48, r"$IMF$", 2, 1),
			'metallicity': _createSlider(0.46, r"$Z_*$", 6, 3),
			'separation_age': _createSlider(0.44, r"$\Delta_{age}$", 100, 10),
		},
		'nebular': {
			'logU': _createSlider(0.40, r"$\log U$", 31, 15),
			'zgas': _createSlider(0.38, r"$Z_{gas}$", 26, 10),
			'f_esc': _createSlider(0.36, r"$f_{esc}$", 100),
			'f_dust': _createSlider(0.34, r"$f_{dust}$", 100),
			'lines_width': _createSlider(0.32, r"$\Delta v$", 10, 2),
			'ne': _createSlider(0.30, r"$n_e$", 3, 2),
		},
		'dustatt_modified_starburst': {
			'E_BV_lines': _createSlider(0.26, r"$E_{BV-lines}$", 100, 40),
			'E_BV_factor': _createSlider(0.24, r"$E_{BV-factor}$", 100, 20),
			'uv_bump_amplitude': _createSlider(0.22, r"$A_{UV_{bump}}$", 100, 10),
			'powerlaw_slope': _createSlider(0.20, r"$A_{slope}$", 100, 30),
			'Ext_law_emission_lines': _createSlider(0.18, r"$Ext_{emission}$", 3, 2),
			'Rv': _createSlider(0.16, r"$R_V$", 50, 40),
		},
		'dl2014': {
			'qpah': _createSlider(0.12, r"$q_{PAH}$", 11, 4),
			'umin': _createSlider(0.10, r"$U_{min}$", 36, 10),
			'alpha': _createSlider(0.08, r"$\alpha$", 21, 5),
			'gamma': _createSlider(0.06, r"$\gamma$", 100, 30),
		},
		'redshifting': {
			'redshift': _createSlider(0.02, r"$z$", 2501),
		}
	}

	sfh : list[Line2D] = ax_sfh.plot([],[], color='r')
	fnu : list[Line2D] = ax_spectra.plot([],[], color='k', label=r"$F_\nu$")
	# Stellar lines
	stellar = [
		ax_spectra.plot([], [], color="xkcd:royal blue", linestyle='--', label="Stellar Old")[0],
		ax_spectra.plot([], [], color="xkcd:bright blue", linestyle='--', label="Stellar Young")[0]
	]

	# Nebular lines
	nebular = [
		ax_spectra.plot([], [], color="xkcd:grass green", linestyle='--', label="Nebular Old")[0],
		ax_spectra.plot([], [], color="xkcd:lime green", linestyle='--', label="Nebular Young")[0]
	]

	# Dust lines
	dust = [
		ax_spectra.plot([], [], color="xkcd:purple", linestyle='--', label=r"Dust $U_{min}$")[0],
		ax_spectra.plot([], [], color="xkcd:lavender", linestyle='--', label=r"Dust $U_{max}$")[0]
	]

	igm : list[Line2D] = ax_spectra.plot([], [], color="orange", linestyle="--", label="IGM")

	leg = ax_spectra.legend()

	def updatePlot(val):
		params = deepcopy(SED_PARAMETERS)

		# Updating values
		for module in SED_PARAMETERS.keys():
			for p in SED_PARAMETERS[module].keys():
				if p in sed_sliders[module].keys():
					index = int(sed_sliders[module][p][0].val)
					value = SED_PARAMETERS[module][p][index]
					params[module][p] = value
					sed_sliders[module][p][1].set_text(value)

		# Create SED
		sed = generateSED(params)

		# SFH
		sfh[0].set_xdata(np.linspace(-len(sed.sfh), 0, len(sed.sfh)))
		sfh[0].set_ydata(sed.sfh)

		Dx = -len(sed.sfh)/2
		Y = (np.nanmax(sed.sfh) + np.nanmin(sed.sfh))/2
		Dy = (np.nanmax(sed.sfh) - np.nanmin(sed.sfh))/2

		ax_sfh.set_xlim(Dx*2.05, Dx*(-0.05))
		ax_sfh.set_ylim(Y-Dy*1.05, Y+Dy*1.05)


		# Fnu
		fnu[0].set_xdata(sed.wavelength_grid/1000)
		fnu[0].set_ydata(sed.luminosity)

		# Luminosities

		# Stellar
		stellar[0].set_data(sed.wavelength_grid / 1000, sed.luminosities["stellar.old"] + sed.luminosities["attenuation.stellar.old"])
		stellar[1].set_data(sed.wavelength_grid / 1000, sed.luminosities["stellar.young"] + sed.luminosities["attenuation.stellar.young"])

		# Nebular
		nebular[0].set_data(sed.wavelength_grid / 1000, sed.luminosities["nebular.absorption_old"] + sed.luminosities["nebular.emission_old"] + sed.luminosities["attenuation.nebular.emission_old"])
		nebular[1].set_data(sed.wavelength_grid / 1000, sed.luminosities["nebular.absorption_young"] + sed.luminosities["nebular.emission_young"] + sed.luminosities["attenuation.nebular.emission_young"])

		# Dust
		dust[0].set_data(sed.wavelength_grid / 1000, sed.luminosities["dust.Umin_Umin"])
		dust[1].set_data(sed.wavelength_grid / 1000, sed.luminosities["dust.Umin_Umax"])

		# IGM
		igm[0].set_data(sed.wavelength_grid / 1000, sed.luminosities["igm"])


	# Assigning method to sliders
	for module in sed_sliders.keys():
		for p in sed_sliders[module].keys():
			sed_sliders[module][p][0].on_changed(updatePlot)

	updatePlot(0)

	# Pickable legend
	legendMap = {}

	for legLine, axLine in zip(leg.get_lines(),fnu + stellar + nebular + dust + igm):
		legLine.set_picker(5)
		legendMap[legLine] = axLine

	def onPick(event):
		legLine = event.artist
		if legLine not in legendMap:
			return
		axLine = legendMap[legLine]
		isVisible = axLine.get_visible()
		axLine.set_visible(not isVisible)
		legLine.set_alpha(1.0 if not isVisible else 0.2)
		fig.canvas.draw()

	fig.canvas.mpl_connect('pick_event', onPick)

	plt.show()





createPlot()