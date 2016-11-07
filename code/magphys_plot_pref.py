import matplotlib as mpl
import math
import numpy as np

# plotting preferences
mpl.rcParams.update({'font.size': 16})
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
try:
	mpl.rcParams['xtick.major.width'] = 1
	mpl.rcParams['ytick.major.width'] = 1
	mpl.rcParams['xtick.minor.width'] = 0.5
	mpl.rcParams['ytick.minor.width'] = 0.5
except KeyError:
	pass
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['lines.markersize'] = 8
mpl.pyplot.ioff() # don't pop up a window for each plot

class jLogFormatter(mpl.ticker.LogFormatter):
	'''
	this changes the format from exponential to floating point.
	'''

	def __call__(self, x, pos=None):
		"""Return the format for tick val *x* at position *pos*"""
		vmin, vmax = self.axis.get_view_interval()
		d = abs(vmax - vmin)
		b = self._base
		if x == 0.0:
			return '0'
		sign = np.sign(x)
		# only label the decades
		fx = math.log(abs(x)) / math.log(b)
		isDecade = mpl.ticker.is_close_to_int(fx)
		if not isDecade and self.labelOnlyBase:
			s = ''
		elif x > 10000:
			s = '{0:.3g}'.format(x)
		elif x < 1:
			s = '{0:.3g}'.format(x)
		else:
			s = self.pprint_val(x, d)
		if sign == -1:
			s = '-%s' % s
		return self.fix_minus(s)
