import read_sextractor, read_data
import numpy as np
from astropy.table import Table, vstack
from astropy.io import ascii

# calculate a UVJ flag
# 0 = quiescent, 1 = starforming
def calc_uvj_flag(rf):

	import numpy as np

	umag = 25 - 2.5*np.log10(rf['L153'])
	vmag = 25 - 2.5*np.log10(rf['L155'])
	jmag = 25 - 2.5*np.log10(rf['L161'])
	
	u_v = umag-vmag
	v_j = vmag-jmag  
	
	# initialize flag to 3, for quiescent
	uvj_flag = np.zeros(len(umag))+3
	
	# star-forming
	sfing = (u_v < 1.3) | (v_j >= 1.5)
	uvj_flag[sfing] = 1
	sfing = (v_j >= 0.92) & (v_j <= 1.6) & (u_v <= 0.8*v_j+0.7)
	uvj_flag[sfing] = 1
	
	# dusty star-formers
	dusty_sf = (uvj_flag == 1) & (u_v <= -1.2*v_j+3.2)
	uvj_flag[dusty_sf] = 2
	
	# outside box
	# from van der wel 2014
	outside_box = (u_v < 0) | (u_v > 2.5) | (v_j > 2.3) | (v_j < -0.3)
	uvj_flag[outside_box] = 0
	
	return uvj_flag

def build_sample():

	'''
	selects a sample of galaxies "randomly" (actually in order of ID, which is position on chip, but whatever)
	to add: output for the EAZY parameters, so I can include p(z) [or whatever I need for that]
	'''

	# variables
	n_per_bin = 3
	sn_bins = ((3,10),(10,40),(40,1e5))
	z_bins  = ((0.2,0.5),(0.5,1.0),(1.0,1.5),(1.5,2.0))
	uvj_bins = [1,2,3]  # 0 = outside box, 1 = sfing, 2 = dusty sf, 3 = quiescent

	# output
	field = 'COSMOS'
	fast_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_testsamp.fout'
	phot_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_testsamp.cat'
	id_str_out   = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_testsamp.ids'

	# use photometric redshifts for now, good enough
	phot = read_sextractor.load_phot_v41(field)
	fast = read_sextractor.load_fast_v41(field)
	rf = read_sextractor.load_rf_v41(field)
	
	# remove junk
	# 153, 155, 161 are U, V, J
	good = (phot['use_phot'] == 1) & (rf['L153'] > 0) & (rf['L155'] > 0) & (rf['L161'] > 0)
	phot = phot[good]
	fast = fast[good]
	rf = rf[good]
	
	# define UVJ flag, S/N
	uvj_flag = calc_uvj_flag(rf)
	sn_F160W = phot['f_F160W']/phot['e_F160W']
	
	# split into bins
	fast_out = Table(names=fast.columns)
	phot_out = Table(names=phot.columns)
	for sn in sn_bins:
		for z in z_bins:
			for uvj in uvj_bins:
				selection = (sn_F160W >= sn[0]) & (sn_F160W < sn[1]) & (uvj_flag == uvj) & (fast['z'] >= z[0]) & (fast['z'] < z[1])
				
				if np.sum(selection) < n_per_bin:
					print 'ERROR: Not enough galaxies in bin!'
					break
				fast_out = vstack([fast_out,fast[selection][:n_per_bin]])
				phot_out = vstack([phot_out,phot[selection][:n_per_bin]])
	
	# output photometric catalog
	ascii.write(phot_out, output=phot_str_out, delimiter=' ', format='commented_header')
	ascii.write(fast_out, output=fast_str_out, delimiter=' ', format='commented_header')
	ascii.write([np.array(phot_out['id'],dtype='int')], output=id_str_out, Writer=ascii.NoHeader)
	