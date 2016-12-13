import numpy as np

datname = 'galaxy_flux.dat'
with open(datname, 'r') as f:
    hdr = f.readline()

hdr = hdr[1:-2].split(',')
dtype = np.dtype([(hdr[0],'S40')] + [(n, np.float) for n in hdr[1:]])
dat = np.loadtxt(datname, comments = '#', delimiter=',', dtype = dtype)

#### pick out galaxies to fit
# 2MASS + SDSS + WISE + GALEX
to_fit = (dat['twomass_Ks'] > 0) & \
		 (dat['twomass_H'] > 0) & \
		 (dat['twomass_J'] > 0) & \
		 (dat['sdss_z0'] > 0) & \
		 (dat['sdss_i0'] > 0) & \
		 (dat['sdss_r0'] > 0) & \
		 (dat['sdss_g0'] > 0) & \
		 (dat['sdss_u0'] > 0) & \
		 (dat['wise_w1'] > 0) & \
		 (dat['wise_w2'] > 0) & \
		 (dat['wise_w3'] > 0) & \
		 (dat['wise_w4'] > 0) & \
		 (dat['galex_NUV'] > 0) & \
		 (dat['galex_FUV'] > 0)

### print out galaxy names
with open('villar_list.dat', 'w') as f:
	for name in dat['Gal_Name'][to_fit]:
		f.write(name+'\n')
	for name in dat['Gal_Name'][~to_fit]:
		f.write(name+'\n')