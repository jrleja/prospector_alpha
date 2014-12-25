import numpy as np
import os

def return_mwave_custom(filters):

	"""
	returns effective wavelength based on filter names
	"""

	loc = os.getenv('APPS')+'/threedhst_bsfh/filters/'
	key_str = 'filter_keys_threedhst.txt'
	lameff_str = 'lameff_threedhst.txt'
	
	lameff = np.loadtxt(loc+lameff_str)
	keys = np.loadtxt(loc+key_str, dtype='S20',usecols=[1])
	keys = keys.tolist()
	keys = np.array([keys.lower() for keys in keys], dtype='S20')
	
	lameff_return = [[lameff[keys == filters[i]]][0] for i in range(len(filters))]
	lameff_return = [item for sublist in lameff_return for item in sublist]
	assert len(filters) == len(lameff_return), "Filter name is incorrect"

	return lameff_return

def load_ancil_data(filename,objnum):

	'''
	loads ancillary plotting information
	'''
	
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	objdat = dat[dat['id'] == float(objnum)]

	return objdat

def load_mips_data(filename,objnum):
	
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	objdat = dat[dat['id'] == float(objnum)]
	
	return objdat

def load_obs_3dhst(filename, objnum, mips=None, min_error = None):
	"""
	Load 3D-HST photometry file, return photometry for a particular object.
	min_error: set the minimum photometric uncertainty to be some fraction
	of the flux. if not set, use default errors.
	"""
	obs ={}
	fieldname=filename.split('/')[-1].split('_')[0].upper()
	with open(filename, 'r') as f:
		hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',
					 dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	obj_ind = np.where(dat['id'] == int(objnum))[0][0]
	
	# extract fluxes+uncertainties for all objects and all filters
	flux_fields = [f for f in dat.dtype.names if f[0:2] == 'f_']
	unc_fields = [f for f in dat.dtype.names if f[0:2] == 'e_']
	filters = [f[2:] for f in flux_fields]

	# extract fluxes for particular object, converting from record array to numpy array
	flux = dat[flux_fields].view(float).reshape(len(dat),-1)[obj_ind]
	unc  = dat[unc_fields].view(float).reshape(len(dat),-1)[obj_ind]

	# add mips
	if mips:
		mips_dat = load_mips_data(mips,objnum)
		flux=np.append(flux,mips_dat['f24tot'])
		unc=np.append(unc,mips_dat['ef24tot'])
		filters.append('MIPS_24um')

	# define all outputs
	filters = [filter.lower()+'_'+fieldname.lower() for filter in filters]
	wave_effective = np.array(return_mwave_custom(filters))
	phot_mask = np.logical_or(np.logical_or((flux != unc),(flux > 0)),flux != -99.0)
	maggies = flux/(10**10)
	maggies_unc = unc/(10**10)

	# set minimum photometric error
	if min_error is not None:
		under = maggies_unc < min_error*maggies
		maggies_unc[under] = min_error*maggies[under]
	
	# sort outputs based on effective wavelength
	points = zip(wave_effective,filters,phot_mask,maggies,maggies_unc)
	sorted_points = sorted(points)

	# build output dictionary
	obs['wave_effective'] = np.array([point[0] for point in sorted_points])
	obs['filters'] = np.array([point[1] for point in sorted_points])
	obs['phot_mask'] =  np.array([point[2] for point in sorted_points])
	obs['maggies'] = np.array([point[3] for point in sorted_points])
	obs['maggies_unc'] =  np.array([point[4] for point in sorted_points])
	obs['wavelength'] = None
	obs['spectrum'] = None

	return obs

def load_fast_3dhst(filename, objnum):
	"""
	Load FAST output for a particular object
	Returns a dictionary of inputs for BSFH
	"""

	# filter through header junk, load data
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))

	# extract field names, search for ID, pull out object info
	fields = [f for f in dat.dtype.names]
	id_ind = fields.index('id')
	obj_ind = [int(x[id_ind]) for x in dat].index(int(objnum))
	values = dat[fields].view(float).reshape(len(dat),-1)[obj_ind]

	return values, fields

def integrate_sfh(t1,t2,tage,tau,sf_start,tburst,fburst):
	
	if t2 < sf_start:
		return 0.0
	elif t2 > tage:
		return 1.0
	else:
		intsfr = (np.exp(-t1/tau)*(1+t1/tau) - 
		          np.exp(-t2/tau)*(1+t2/tau))
		norm=(1.0-fburst)/(np.exp(-sf_start/tau)*(sf_start/tau+1)-
				  np.exp(-tage    /tau)*(tage    /tau+1))
		intsfr=intsfr*norm
		
	if t2 > tburst:
		intsfr=intsfr+fburst

	return intsfr