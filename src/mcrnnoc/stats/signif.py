# source https://github.com/milzj/SAA4PDE/blob/semilinear_complexity/base/signif.py


import numpy as np

def signif(x, precision=3):
	"""Rounds the input to significant figures.

	Parameters:
	----------
		x : float
			a floating point number

		precision : int (optional)
			number of significant figures

	"""
	y = np.format_float_positional(x, precision=precision, unique=True, trim="k", fractional=False)
	return np.float64(y)

def _signif(beta):
	"Returns three significant figures of input float."
	_beta = float('{0:.3g}'.format(beta))
	return _beta
