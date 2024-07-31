# source https://github.com/milzj/SAA4PDE/blob/semilinear_complexity/base/lsqs_label.py

from .signif import signif

def lsqs_label(constant=0.0, rate=0.0, base=10.0, precision=3):
	constant = signif(constant, precision=precision)
	rate = signif(rate, precision=precision)
	return r"${}\cdot {}^{}$".format(constant, base, "{"+ str(rate)+"}")
