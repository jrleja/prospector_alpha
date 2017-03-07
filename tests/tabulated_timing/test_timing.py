import fsps
import numpy as np
import time

sps = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=2)
w = sps.wavelengths

sps.params['sfh'] = 0 #composite
#sps.params['add_neb_emission'] = True
ncalc = 101

t_out = []
for i in range(ncalc):
	t1 = time.time()
	sps.params['logzsol'] = np.random.uniform(low=-1.0,high=0)
	wave, spec = sps.get_spectrum(tage=0)
	d1 = time.time()-t1
	print('calculation took {0}s'.format(d1))
	if i > 0:
		t_out.append(d1)

t_out = np.array(t_out,dtype=float)
print('average parametric time over {0} calculations: {1}s'.format(ncalc-1,np.mean(t_out)))

sps.params['sfh'] = 3 #tabular

### calculate times at edges of bins
agelims = [0.0,8.0,8.5,9.0,9.5,9.8,10.14]
agebins = np.array([agelims[:-1], agelims[1:]])
in_years = 10**agebins/1e9
t = np.concatenate((np.ravel(in_years)*0.9999, np.ravel(in_years)*1.001))
t.sort()
t = np.unique(t)[1:-1] # remove older than oldest bin, younger than youngest bin
t_regular = in_years.max()-t

sps.set_tabular_sfh(t_regular,np.ones_like(t_regular))

t_out = []
for i in range(ncalc):
	t1 = time.time()
	sps.params['logzsol'] = np.random.uniform(low=-1.0,high=0)
	wave, spec = sps.get_spectrum(tage=13.7)
	d1 = time.time()-t1
	print('calculation took {0}s'.format(d1))
	if i > 0:
		t_out.append(d1)

t_out = np.array(t_out,dtype=float)
print('average tabular time over {0} calculations: {1}s'.format(ncalc-1,np.mean(t_out)))


print 1/0
