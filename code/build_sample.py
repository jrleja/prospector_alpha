import read_sextractor, read_data, random
import numpy as np
from astropy.table import Table, vstack
from astropy.io import ascii
random.seed(25001)

def load_linelist():
	
	filename='/Users/joel/data/3d_hst/master.zbest.dat'
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()

	dat = np.loadtxt(filename, comments = '#',
	                 dtype = {'names':([n for n in hdr[1:]]),
	                          'formats':('S40', 'f10','f10','S40','f10','f10','f16','f16','f16','f16')})
	
	fields = [f for f in dat.dtype.names]
	spec_ind = fields.index('spec_id')

	specnames = [x[spec_ind] for x in dat]
	in_field = [x[0] == 'COSMOS' for x in dat]
	spec_flag = [x[5] for x in dat if x[0] == 'COSMOS']
	id = [x[2] for x in dat if x[0] == 'COSMOS']

	halpha_eqw = np.empty(0)
	halpha_eqw_err = np.empty(0)
	zgris = np.empty(0)
	
	for jj in xrange(len(specnames)):
		if dat[jj][0] != 'COSMOS':
			pass
		else:
			if dat[jj][spec_ind] == '00000':
				zgris = np.append(zgris,-99)
				halpha_eqw = np.append(halpha_eqw,-99)
				halpha_eqw_err = np.append(halpha_eqw_err,-99)
			else:
				fieldno = dat[jj][spec_ind].split('-')[1]
				filename='/Users/joel/data/3d_hst/spectra/cosmos-wfc3-spectra_v4.1.4/cosmos-'+ fieldno+'/LINE/DAT/'+dat[jj][spec_ind]+'.linefit.dat'
				
				# if there's no line detected, there's no line file
				try:
					with open(filename, 'r') as f:
						hdr = f.readline().split()
						zgris_new = float(f.readline().split('=')[1].strip())
					linedat = np.loadtxt(filename, comments='#',
			                     dtype = {'names':([n for n in hdr[1:]]),
	                                      'formats':('S40', 'f16','f16','f16','f16','f16')})
					#linefields = [f for f in linedat.dtype.names]
				
					# pull out data
					heqw = [x[4] for x in linedat if x[0] == 'Ha']
					heqw_err = [x[5] for x in linedat if x[0] == 'Ha']
					zgris = np.append(zgris,zgris_new)
				except:
					heqw = []
					heqw_err =[]
					zgris = np.append(zgris,-99)
				
				# write data
				if len(heqw) != 0:
					halpha_eqw = np.append(halpha_eqw,heqw)
					halpha_eqw_err = np.append(halpha_eqw_err,heqw_err)
				else:
					halpha_eqw = np.append(halpha_eqw,-99)
					halpha_eqw_err = np.append(halpha_eqw_err,-99)
				if len(zgris) != len(halpha_eqw):
					print 1/0
	
	
	# convert from observed EQW to intrinsic EQW
	halpha_eqw = halpha_eqw/(1+zgris)
	halpha_eqw_err = halpha_eqw_err/(1+zgris)
	lineinfo = Table([np.array(id),
	                  np.array(halpha_eqw),
	                  np.array(halpha_eqw_err),
	                  np.array(zgris),
	                  np.array(spec_flag)],
	                  names=['id','ha_eqw','ha_eqw_err','zgris','spec_flag'])
	return lineinfo
				
	
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
	dusty_sf = (uvj_flag == 1) & (u_v >= -1.2*v_j+2.8)
	uvj_flag[dusty_sf] = 2
	
	# outside box
	# from van der wel 2014
	outside_box = (u_v < 0) | (u_v > 2.5) | (v_j > 2.3) | (v_j < -0.3)
	uvj_flag[outside_box] = 0
	
	return uvj_flag

def print_info(id,ids,uvj_flag,sn_F160W,z):

	index=list(ids).index(id)
	
	print 'UVJ flag: ' +"{:10.2f}".format(uvj_flag[index])
	print 'S/N F160W: ' +"{:10.2f}".format(sn_F160W[index])
	print 'redshift: ' +"{:10.2f}".format(z[index])
	
	return None

def build_sample():

	'''
	selects a sample of galaxies "randomly"
	to add: output for the EAZY parameters, so I can include p(z) [or whatever I need for that]
	'''

	# output
	field = 'COSMOS'
	fast_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_testsamp.fout'
	ancil_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_testsamp.dat'
	phot_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_testsamp.cat'
	id_str_out   = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_testsamp.ids'

	# load data
	# use grism redshifts
	phot = read_sextractor.load_phot_v41(field)
	fast = read_sextractor.load_fast_v41(field)
	rf = read_sextractor.load_rf_v41(field)
	lineinfo = load_linelist()
	
	# remove junk
	# 153, 155, 161 are U, V, J
	good = (phot['use_phot'] == 1) & (rf['L153'] > 0) & (rf['L155'] > 0) & (rf['L161'] > 0) & (phot['f_IRAC4'] < 1e7) & (phot['f_IRAC3'] < 1e7) & (phot['f_IRAC2'] < 1e7) & (phot['f_IRAC1'] < 1e7) & (lineinfo['spec_flag'] == 1) & (lineinfo['ha_eqw']/lineinfo['ha_eqw_err'] > 2)

	phot = phot[good]
	fast = fast[good]
	rf = rf[good]
	lineinfo = lineinfo[good]
	
	# define UVJ flag, S/N, HA EQW
	uvj_flag = calc_uvj_flag(rf)
	sn_F160W = phot['f_F160W']/phot['e_F160W']
	ha_eqw = lineinfo['ha_eqw']
	fast['z'] = lineinfo['zgris']
	fast['ha_eqw']   = ha_eqw
	fast['uvj_flag'] = uvj_flag
	fast['sn_F160W'] = sn_F160W
	
	# split into bins
	lowlim = np.percentile(ha_eqw,65)
	highlim = np.percentile(ha_eqw,95)
	
	selection = (ha_eqw > lowlim) & (ha_eqw < highlim)
	random_index = random.sample(xrange(np.sum(selection)), 108)
	fast_out = fast[selection][random_index]
	phot_out = phot[selection][random_index]
	
	# variables
	#n_per_bin = 3
	#sn_bins = ((12,20),(20,100),(100,1e5))
	#z_bins  = ((0.2,0.6),(0.6,1.0),(1.0,1.5),(1.5,2.0))
	#uvj_bins = [1,2,3]  # 0 = outside box, 1 = sfing, 2 = dusty sf, 3 = quiescent
	
	#for sn in sn_bins:
	#	for z in z_bins:
	#		for uvj in uvj_bins:
	#			selection = (sn_F160W >= sn[0]) & (sn_F160W < sn[1]) & (uvj_flag == uvj) & (fast['z'] >= z[0]) & (fast['z'] < z[1])

	#			if np.sum(selection) < n_per_bin:
	#				print 'ERROR: Not enough galaxies in bin!'
	#				print sn, z, uvj
				
				# choose random set of indices
	#			random_index = random.sample(xrange(np.sum(selection)), n_per_bin)
	
	#			fast_out = vstack([fast_out,fast[selection][random_index]])
	#			phot_out = vstack([phot_out,phot[selection][random_index]])
	
				#fast_out = vstack([fast_out,fast[selection][:n_per_bin]])
				#phot_out = vstack([phot_out,phot[selection][:n_per_bin]])
	
	# output photometric catalog, fast catalog, id list
	# artificially add "z" to ancillary outputs
	# later, will draw z from zbest
	# later still, will do PDF...
	ascii.write(phot_out, output=phot_str_out, 
	            delimiter=' ', format='commented_header')
	ascii.write(fast_out, output=fast_str_out, 
	            delimiter=' ', format='commented_header',
	            include_names=fast.keys()[:11])
	ascii.write(fast_out, output=ancil_str_out, 
	            delimiter=' ', format='commented_header',
	            include_names=[fast.keys()[0]]+['z']+fast.keys()[11:])
	ascii.write([np.array(phot_out['id'],dtype='int')], output=id_str_out, Writer=ascii.NoHeader)
	