from astropy.io import ascii, fits
import numpy as np
from astropy.table import Table, vstack
import os

def load_phot_v41(field):

	loc = '/Users/joel/data/3d_hst/v4.1_cats/'+field.lower()+'_3dhst.v4.1.cats/Catalog/'+field.lower()+'_3dhst.v4.1.cat'
	x = ascii.read(loc)

	return x

def load_zbest(field):

	loc = '/Users/joel/data/3d_hst/v4.1_spectral/'+field.lower()+'_3dhst_v4.1.5_catalogs/'+field.lower()+'_3dhst.v4.1.5.zbest.dat'
	x = ascii.read(loc)

	return x

def load_rf_v41(field):

	loc = '/Users/joel/data/3d_hst/v4.1_cats/'+field.lower()+'_3dhst.v4.1.cats/RF_colors/'+field.lower()+'_3dhst.v4.1.master.RF'
	x = ascii.read(loc)

	return x

def load_fast_v41(field):

	loc = '/Users/joel/data/3d_hst/v4.1_cats/'+field.lower()+'_3dhst.v4.1.cats/Fast/'+field.lower()+'_3dhst.v4.1.fout'
	x = ascii.read(loc, names=['id','z','ltau','metal','lage','Av','lmass','lsfr','lssfr','la2t','chi2'])

	return x

def load_zp_offsets(field):

	### add in dash
	if field == 'GOODSN' or field == 'GOODSS':
		field = field[:5]+'-'+field[5]

	filename = os.getenv('APPS')+'/prospector_alpha/data/3dhst/zp_offsets_tbl11_skel14.txt'
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dtype = [np.dtype((str, 35)),np.dtype((str, 35)),np.float,np.float]
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(hdr[n+1], dtype[n]) for n in xrange(len(hdr)-1)]))

	if field is not None:
		good = dat['Field'] == field
		if np.sum(good) == 0:
			print 'Not an acceptable field name! Returning None'
			return None
		else:
			dat = dat[good]

	return dat

def load_grism_dat(field,process=False):
	"""if process, turn into a manageable size and interpret some things
	note that we return REST-FRAME equivalent width!
	else return the whole shebang
	"""
	loc = '/Users/joel/data/3d_hst/v4.1_spectral/'+field.lower()+'_3dhst_v4.1.5_catalogs/'+field.lower()+'_3dhst.v4.1.5.linefit.linematched.fits'
	hdu = fits.open(loc)
	dat = hdu[1].data

	if not process:
		return dat
	
	# grab specific lines
	hdr, hdr_type = [], []
	lines_to_save = ['Ha']
	line_list = ['_FLUX', '_FLUX_ERR', '_EQW', '_EQW_ERR']
	for line in lines_to_save:
		for ltype in line_list:
			hdr.append(line+ltype)
			hdr_type = hdr_type + [float]

	# grab specific fields
	dat_to_save = ['number', 'grism_id', 'z_max_grism']
	for idx in dat_to_save:
		hdr.append(idx)
		hdr_type.append(dat.dtype[idx])
	
	# buil output array
	out = np.recarray(dat.shape, dtype=[(x,y) for x,y in zip(hdr,hdr_type)]) 
	for idx in hdr: out[idx][:] = dat[idx][:]
	return out

def load_linelist(field='COSMOS'):
	
	'''
	uses pieter's zbest catalog to determine objects with spectra
	returns all line information, zgris, and use flag
	note that equivalent widths are in observed frame, convert by dividing by (1+z)
	THIS IS DEPRECATED
	'''
	filename='/Users/joel/data/3d_hst/master.zbest.dat'
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()

	dat = np.loadtxt(filename, comments = '#',
	                 dtype = {'names':([n for n in hdr[1:]]),
	                          'formats':('S40','f16','f16','S40','f16','f16','f16','f16','f16','f16')})
	
	# cut down master catalog to only in-field objects
	in_field = dat['field'] == field
	dat      = dat[in_field]
	
	# build output
	t = Table([dat['id'],
		      [-99.0]*len(dat),
		      dat['z_best'],
		      dat['use_grism1'],
		      [-99.0]*len(dat),
		      [-99.0]*len(dat)],
	          names=('id','zgris','zbest','use_grism1','s0','s1'))

	# include only these lines
	good_lines = ['Ha', 'OIII', 'OIIIx', 'OII']
	for jj in xrange(len(dat)):
		
		# no spectrum
		if dat['spec_id'][jj] == '00000':
			pass
		else: # let's get to work
			fieldno = dat['spec_id'][jj].split('-')[1]
			filename='/Users/joel/data/3d_hst/spectra/'+field.lower()+'-wfc3-spectra_v4.1.4/'+field.lower()+'-'+ fieldno+'/LINE/DAT/'+dat['spec_id'][jj]+'.linefit.dat'
			
			# if the line file doesn't exist, abandon all hope (WHY DO SOME NOT EXIST?!?!?!)
			# some don't exist, others are empty
			# empty ones just don't enter the value assigning loop
			# may be difference in version between pieter's master catalog and current line catalog
			try:
				with open(filename, 'r') as f:
					hdr = f.readline().split()
					t['zgris'][jj] = float(f.readline().split('=')[1].strip())
			except:
				print filename+" doesn't exist"
				continue
			
			linedat = np.loadtxt(filename, comments='#',
			                     dtype = {'names':([n for n in hdr[1:]]),
	                                      'formats':('S40', 'f16','f16','f16','f16','f16')})
			
			# tilt correction
			with open(filename) as search:
				for line in search:
					if 'tilt correction' in line:
						splitline = line.split(' ')
						t[jj]['s0'] = float(splitline[splitline.index('s0')+2])
						t[jj]['s1'] = float(splitline[splitline.index('s1')+2])

			# if there's only one line, loadtxt returns a different thing
			if linedat.size == 1:
				linedat=np.array(linedat.reshape(1),dtype=linedat.dtype)
	        
			for kk in xrange(linedat.size): #for each line
				if linedat['line'][kk] in good_lines: # only include lines we're interested in
					if linedat['line'][kk]+'_flux' not in t.keys(): #check for existing column names
							for line in good_lines: # only add lines we're intersetd in!
								for name in linedat.dtype.names[1:]: t[line+'_'+name] = [-99.0]*len(t)
					for name in linedat.dtype.names[1:]: t[jj][linedat['line'][kk]+'_'+name] = linedat[name][kk]

	# convert to rest-frame eqw
	for key in t.keys():
		if 'EQW' in key:
			detection = t[key] != -99
			t[key][detection] = t[key][detection]/(1+t['zgris'][detection])

	return t
		

def load_mips_data(field,objnum=None):
	
	filename = '/Users/joel/data/3d_hst/v4.1_cats/'+field.lower()+'_3dhst.v4.1.cats/'+field.lower()+'_3dhst.v4.1.sfr'
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	
	if objnum is not None:
		objdat = dat[dat['id'] == float(objnum)]
		return objdat

	return dat

def return_fast_sed(fastname,objname, sps=None, obs=None, dustem = False):

	'''
	give the fast parameters straight from the FAST out file
	return observables with best-fit FAST parameters, main difference hopefully being stellar population models
	'''

	# load fast parameters
	fast, fields = load_fast_3dhst(fastname, objname)
	fields = np.array(fields)

	# load fast model
	param_file = os.getenv('APPS')+'/prospector_alpha/parameter_files/fast_mimic/fast_mimic.py'
	model = model_setup.load_model(param_file)
	parnames = np.array(model.theta_labels())

	# feed parameters into model
	model.params['zred']                   = np.array(fast[fields == 'z'])
	model.initial_theta[parnames=='tage']  = np.clip(np.array((10**fast[fields == 'lage'])/1e9),0.101,10000)
	model.initial_theta[parnames=='tau']   = np.array((10**fast[fields == 'ltau'])/1e9)
	model.initial_theta[parnames=='dust2'] = np.array(av_to_dust2(fast[fields == 'Av']))
	model.initial_theta[parnames=='mass']  = np.array(10**fast[fields == 'lmass'])

	print 'z,tage,tau,dust2,mass'
	print model.params['zred'],model.initial_theta[parnames=='tage'],model.initial_theta[parnames=='tau'],model.initial_theta[parnames=='dust2'],model.initial_theta[parnames=='mass']

	# get dust emission, if desired
	if dustem:
		model.params['add_dust_emission']  = np.array(True)
		model.params['add_agb_dust_model'] = np.array(True)


	spec, mags,sm = model.mean_model(model.initial_theta, obs, sps=sps, norm_spec=True)
	w = sps.wavelengths
	return spec,mags,w,fast,fields


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
	
	
	if objnum is None:
		values = dat[fields].view(float).reshape(len(dat),-1)
	else:
		values = dat[fields].view(float).reshape(len(dat),-1)
		id_ind = fields.index('id')
		obj_ind = [int(x[id_ind]) for x in dat].index(int(objnum))
		values = values[obj_ind]

	return values, fields

def load_ancil_data(filename,objnum):

	'''
	loads ancillary plotting information
	'''
	
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))

	if objnum:
		objdat = dat[dat['id'] == float(objnum)]
		return objdat

	return dat
