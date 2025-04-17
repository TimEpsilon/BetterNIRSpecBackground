#  Copyright (c) 2025. Tim Dewachter, LAM

import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt

def BSpline(Ti, k=3, lowerSpline=None):
	if len(Ti) < k+1:
		print(f"Not enough knots, got {len(Ti)} but needs at least {k+1}")
		pass

	if k == 0:
		splines = []
		for i in range(len(Ti) - 1):
			def B_i0(x,i=i):
				return np.where((x >= Ti[i]) & (x < Ti[i+1]), 1, 0)
			splines.append(B_i0)
		return splines

	else :
		splines = []
		if lowerSpline is None:
			lowerSpline = BSpline(Ti, k-1)
		for i in range(len(Ti) - k - 1):
			def B_ip(x,i=i):
				return (x - Ti[i]) / (Ti[i+k] - Ti[i]) * lowerSpline[i](x) + (Ti[i+k+1] - x) / (Ti[i+k+1] - Ti[i+1]) * lowerSpline[i+1](x)
			splines.append(B_ip)
		return splines

###### MAIN ######

if __name__ == '__main__':
	Ti = np.linspace(0,10,5)

	spline_0 = BSpline(Ti, k=0)
	spline_1 = BSpline(Ti, k=1, lowerSpline=spline_0)
	spline_2 = BSpline(Ti, k=2, lowerSpline=spline_1)
	X = np.linspace(-2,12,1000)

	plt.figure()
	for _ in spline_0:
		plt.plot(X,_(X),color='b')
	for _ in spline_1:
		plt.plot(X,_(X),color='r')
	for _ in spline_2:
		plt.plot(X,_(X),color='g')

	plt.show()
