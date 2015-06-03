import read_sextractor, read_data, random, os, threed_dutils
import numpy as np
from astropy.table import Table, vstack
from astropy.io import ascii
from astropy.coordinates import ICRS
from astropy import units as u
random.seed(25001)

def load_linelist(field='COSMOS'):
	
	'''
	uses pieter's zbest catalog to determine objects with spectra
	returns all line information, zgris, and use flag
	note that equivalent widths are in observed frame, convert by dividing by (1+z)
	'''
	filename='/Users/joel/data/3d_hst/master.zbest.dat'
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()

	dat = np.loadtxt(filename, comments = '#',
	                 dtype = {'names':([n for n in hdr[1:]]),
	                          'formats':('S40','f10','f10','S40','f10','f10','f16','f16','f16','f16')})
	
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

def return_test_sfhs(test_sfhs):

	'''
	return special constraints on sf_start and tau
	custom-built for SFH recovery tests
	'''

	# always implement mass priors
	if test_sfhs != 0:
		mass_bounds      = np.array([[1e10,1e11],[1e10,1e11]])

	if test_sfhs == 1:
		tau_bounds      = np.array([[50,100],[50,100]])
		sf_start_bounds = np.array([[0,4],[0,4]])
		dust_bounds     = np.array([[0.0,4.0],[0.0,4.0]])
		descriptor      = 'constant sfh'

	elif test_sfhs == 2:
		tau_bounds      = np.array([[0.1,3],[0.1,3]])
		sf_start_bounds = np.array([[0,4],[0,4]])
		dust_bounds     = np.array([[0.0,4.0],[0.0,4.0]])
		descriptor      = 'quiescent'		

	elif test_sfhs == 3:
		tau_bounds      = np.array([[0.1,3],[0.1,3]])
		sf_start_bounds = np.array([[12,14],[12,14]])
		dust_bounds     = np.array([[0.0,4.0],[0.0,4.0]])
		descriptor      = 'burst'

	elif test_sfhs == 4:
		tau_bounds      = np.array([[0.1,7],[0.1,3]])
		sf_start_bounds = np.array([[0,3],[12,14]])
		dust_bounds     = np.array([[0.0,4.0],[0.0,4.0]])
		descriptor      = 'old+burst'

	elif test_sfhs == 5:
		tau_bounds      = np.array([[4.0,20.0],[0.1,3]])
		sf_start_bounds = np.array([[0,12],[12.5,14]])
		dust_bounds     = np.array([[0.0,4.0],[3.0,4.0]])
		descriptor      = 'dusty_burst'

	parnames = np.array(['mass','tau','sf_start','dust2'])
	bounds   = np.array([mass_bounds,tau_bounds,sf_start_bounds,dust_bounds])

	return bounds,parnames,descriptor

def return_bounds(parname,model,i,test_sfhs=False):

	'''
	returns parameter boundaries
	if test_sfhs is on, puts special constraints on certain variables
	these special constraints are defined in return_test_sfhs
	'''

	bounds = model.theta_bounds()[i]
	if test_sfhs != False:
		bounds,parnames,descriptor=return_test_sfhs(test_sfhs)
		if parname[:-2] in parnames:
			bounds = bounds[parnames == parname[:-2]][0][int(parname[-1])-1]

	return bounds[0],bounds[1]

def parname_strip(parname):

	try:
		int(parname[-1])
		return parname[-2:]
	except:
		return parname

def build_sample_test(basename,outname=None,add_zp_err=False):

	'''
	Generate model SEDs and add noise
	IMPORTANT: linked+outlier noise will NOT be added if those variables are not free 
	parameters in the passed parameter file!
	'''

	from bsfh import model_setup

	#### output names ####
	if outname == None:
		outname = '/Users/joel/code/python/threedhst_bsfh/data/'+basename
	parmfile='/Users/joel/code/python/threedhst_bsfh/parameter_files/'+basename+'/'+basename+'_params.py'

	#### load test model, build sps  ####
	model = model_setup.load_model(parmfile)
	obs   = model_setup.load_obs(parmfile)
	sps = threed_dutils.setup_sps()

	#### basic parameters ####
	ngals_per_model     = 60
	noise               = 0.00            # perturb fluxes
	reported_noise      = 0.01            # reported noise
	test_sfhs           = [1,2,3,4,5]     # which test sfhs to use?
	test_sfhs           = [0]
	ntest               = len(test_sfhs)
	ngals               = ntest*ngals_per_model
	
	#### band-specific noise ####
	if 'gp_filter_amps' in model.free_params:
		band_specific_noise = [0.0,0.15,0.25] # add band-specific noise?

	#### outlier noise ####
	if 'gp_outlier_locs' in model.free_params:
		outliers_noise      = 0.5             # add outlier noise
		outliers_bands      = [5,22,29]
	else:
		outliers_bands=[]

	#### generate random model parameters ####
	nparams = len(model.initial_theta)
	testparms = np.zeros(shape=(ngals,nparams))
	parnames = np.array(model.theta_labels())

	for jj in xrange(ntest):
		for ii in xrange(nparams):
			
			# random in logspace for mass + tau
			# also enforce priors
			if parname_strip(parnames[ii]) == 'mass' or parname_strip(parnames[ii]) == 'tau':
				
				min,max = np.log10(return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj]))

				# ensure that later priors CAN be enforced
				if parnames[ii] == 'mass_1': max = np.log10(10**max/20)
				if parnames[ii] == 'tau_1': min = np.log10(10**min*2)

				# enforce priors on mass and tau
				if parnames[ii] == 'mass_2':
					max = np.log10(testparms[:,parnames == 'mass_1']*20)
					for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = 10**(random.random()*(max[kk]-min)+min)
				elif parnames[ii] == 'tau_2':
					max = np.log10(testparms[:,parnames == 'tau_1']/2.)
					for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = 10**(random.random()*(max[kk]-min)+min)
				else:
					for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = 10**(random.random()*(max-min)+min)
			
			#### generate specific SFHs if necessary ####
			elif parname_strip(parnames[ii]) == 'sf_start':
				min,max = return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj])
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = random.random()*(max-min)+min

			#### tone down the dust a bit-- flat in prior means lots of Av = 2.0 galaxies ####
			elif parname_strip(parnames[ii]) == 'dust2':
				if test_sfhs[jj] != 5:
					min = model.theta_bounds()[ii][0]
					max = model.theta_bounds()[ii][1]
					for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = np.clip(random.gauss(0.5, 0.5),min,max)
				else:
					min,max = return_bounds(parnames[ii],model,ii,test_sfhs=test_sfhs[jj])
					for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = random.random()*(max-min)+min

			#### apply dust_index prior! ####
			elif parname_strip(parnames[ii]):
				min = model.theta_bounds()[ii][0]
				max = model.theta_bounds()[ii][1]
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = np.clip(random.gauss(-0.7, 0.5),min,max)

			#### general photometric jitter ####
			elif parname_strip(parnames[ii]):
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = 0.0
		
			#### linked filter noise ####
			elif parname_strip(parnames[ii]) == 'gp_filter_amps':
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = band_specific_noise[int(parnames[ii][-1])-1]

			#### outliers ####
			elif parname_strip(parnames[ii]) == 'gp_outlier_amps':
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = outliers_noise
			elif parname_strip(parnames[ii]) == 'gp_outlier_locs':
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = outliers_bands[int(parnames[ii][-1])-1]

			else:
				min = model.theta_bounds()[ii][0]
				max = model.theta_bounds()[ii][1]
				for kk in xrange(jj*ngals_per_model,(jj+1)*ngals_per_model): testparms[kk,ii] = random.random()*(max-min)+min
			print parname_strip(parnames[ii])
			print min,max

	#### make sure priors are satisfied
	for ii in xrange(ngals):
		assert np.isfinite(model.prior_product(testparms[ii,:]))

	#### write out thetas ####
	with open(outname+'.dat', 'w') as f:
		
		### header ###
		f.write('# ')
		for theta in model.theta_labels():
			f.write(theta+' ')
		f.write('\n')

		### data ###
		for ii in xrange(ngals):
			for kk in xrange(nparams):
				f.write(str(testparms[ii,kk])+' ')
			f.write('\n')

	#### set up photometry output ####
	nfilters = len(obs['filters'])
	maggies     = np.zeros(shape=(ngals,nfilters))
	maggies_unc = np.zeros(shape=(ngals,nfilters))

	#### generate photometry, add noise ####
	for ii in xrange(ngals):
		model.initial_theta = testparms[ii,:]
		_,maggiestemp,_ = model.mean_model(model.initial_theta, obs, sps=sps,norm_spec=False)
		maggies[ii,:] = maggiestemp

		#### record noise ####
		maggies_unc[ii,:] = maggies[ii,:]*reported_noise

		#### add noise ####
		for kk in xrange(nfilters): 
			
			###### general noise
			tnoise = noise
			
			##### linked filter noise
			filtlist = model.params.get('gp_filter_locs',[])			
			for mm in xrange(len(filtlist)):
				if obs['filters'][kk].lower() in filtlist[mm]:
					tnoise = (tnoise**2+band_specific_noise[mm]**2)**0.5

			##### outlier noise
			if kk in outliers_bands:
				tnoise = (tnoise**2+outliers_noise**2)**0.5
			add_noise = random.gauss(0, tnoise)
			print obs['filters'][kk].lower()+': ' + "{:.2f}".format(add_noise)
			maggies[ii,kk] += add_noise*maggies[ii,kk]

	#### add zeropoint offsets ####
	if add_zp_err:
		zp_offsets = threed_dutils.load_zp_offsets('COSMOS')
		for kk in xrange(len(zp_offsets)):
			filter = zp_offsets[kk]['Band'].lower()+'_cosmos'
			index  = obs['filters'] == filter
			maggies[:,index] = maggies[:,index]*zp_offsets[kk]['Flux-Correction']
			if np.sum(index) == 0:
				print 1/0

	#### output ####
	#### ids first ####
	ids =  np.arange(ngals)+1
	with open(outname+'.ids', 'w') as f:
	    for id in ids:
	        f.write(str(id)+'\n')

	#### photometry ####
	with open(outname+'.cat', 'w') as f:
		
		### header ###
		f.write('# id ')
		for filter in obs['filters']:
			f.write('f_'+filter+' e_' +filter+' ')
		f.write('\n')

		### data ###
		for ii in xrange(ngals):
			f.write(str(ids[ii])+' ')
			for kk in xrange(nfilters):
				f.write(str(maggies[ii,kk])+' '+str(maggies_unc[ii,kk]) + ' ')
			f.write('\n')

def build_sample_onekrun(rm_zp_offsets=True):

	'''
	selects a sample of galaxies for 1k run
	evenly spaced in five redshift bins out to z=3
	'''

	zbins = [(0.4,0.8),(0.8,1.2),(1.2,1.6),(1.6,2.0),(2.0,2.5)]

	# output
	field = 'COSMOS'
	basename = 'onek'
	fast_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.fout'
	ancil_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.dat'
	phot_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.cat'
	id_str_out   = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.ids'

	# load data
	# use grism redshift
	phot = read_sextractor.load_phot_v41(field)
	fast = read_sextractor.load_fast_v41(field)
	rf = read_sextractor.load_rf_v41(field)
	lineinfo = load_linelist()
	mips = threed_dutils.load_mips_data(os.getenv('APPS')+'/threedhst_bsfh/data/MIPS/cosmos_3dhst.v4.1.4.sfr')
	
	# remove junk
	# 153, 155, 161 are U, V, J
	good = (phot['use_phot'] == 1) & \
	       (phot['f_IRAC4'] < 1e7) & (phot['f_IRAC3'] < 1e7) & (phot['f_IRAC2'] < 1e7) & (phot['f_IRAC1'] < 1e7) & \
	       (fast['lmass'] > 10.3)

	phot = phot[good]
	fast = fast[good]
	rf = rf[good]
	lineinfo = lineinfo[good]
	mips = mips[good]
	
	# define UVJ flag, S/N, HA EQW
	uvj_flag = calc_uvj_flag(rf)
	sn_F160W = phot['f_F160W']/phot['e_F160W']
	Ha_EQW_obs = lineinfo['Ha_EQW_obs']
	lineinfo.rename_column('zgris' , 'z')
	lineinfo['uvj_flag'] = uvj_flag
	lineinfo['sn_F160W'] = sn_F160W

	# mips
	phot['f_MIPS_24um'] = mips['f24tot']
	phot['e_MIPS_24um'] = mips['ef24tot']

	for i in xrange(len(mips.dtype.names)):
		tempname=mips.dtype.names[i]
		if tempname != 'z_best' and tempname != 'id':
			lineinfo[tempname] = mips[tempname]
		elif tempname == 'z_best':
			lineinfo['z_sfr'] = mips[tempname]
	
	# split into bins
	ngal = 1000
	n_per_bin = ngal / len(zbins)
	
	for z in zbins:
		selection = np.where((lineinfo['zbest'] >= z[0]) & (lineinfo['zbest'] < z[1]))[0]

		if np.sum(selection) < n_per_bin:
			print 'ERROR: Not enough galaxies in bin!'
			print np.sum(selection),n_per_bin
				
		# choose random set of indices
		random_index = random.sample(xrange(len(selection)), n_per_bin)
		if z[0] != zbins[0][0]:
			fast_out = vstack([fast_out,fast[selection[random_index]]])
			phot_out = vstack([phot_out,phot[selection[random_index]]])
			lineinfo_out = vstack([lineinfo_out,lineinfo[selection[random_index]]])
		else:
			fast_out = fast[selection[random_index]]
			phot_out = phot[selection[random_index]]
			lineinfo_out = lineinfo[selection[random_index]]
	
	# rename bands in photometric catalogs
	for column in phot_out.colnames:
		if column[:2] == 'f_' or column[:2] == 'e_':
			phot_out.rename_column(column, column.lower()+'_'+field.lower())	

	if rm_zp_offsets:
		bands_exempt = ['irac1_cosmos','irac2_cosmos','irac3_cosmos','irac4_cosmos',\
                        'f606w_cosmos','f814w_cosmos','f125w_cosmos','f140w_cosmos',\
                        'f160w_cosmos','mips_24um_cosmos']
		phot_out = remove_zp_offsets(field,phot_out,bands_exempt=bands_exempt)

	ascii.write(phot_out, output=phot_str_out, 
	            delimiter=' ', format='commented_header')
	ascii.write(fast_out, output=fast_str_out, 
	            delimiter=' ', format='commented_header',
	            include_names=fast.keys()[:11])
	ascii.write(lineinfo_out, output=ancil_str_out, 
	            delimiter=' ', format='commented_header')
	ascii.write([np.array(phot_out['id'],dtype='int')], output=id_str_out, Writer=ascii.NoHeader)
	print 1/0

def build_sample_general():

	'''
	selects a sample of galaxies "randomly"
	to add: output for the EAZY parameters, so I can include p(z) [or whatever I need for that]
	'''

	# output
	field = 'COSMOS'
	basename = 'gensamp'
	fast_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.fout'
	ancil_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.dat'
	phot_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.cat'
	id_str_out   = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.ids'

	# load data
	# use grism redshift
	phot = read_sextractor.load_phot_v41(field)
	fast = read_sextractor.load_fast_v41(field)
	rf = read_sextractor.load_rf_v41(field)
	lineinfo = load_linelist()
	mips = threed_dutils.load_mips_data(os.getenv('APPS')+'/threedhst_bsfh/data/MIPS/cosmos_3dhst.v4.1.4.sfr')
	
	# remove junk
	# 153, 155, 161 are U, V, J
	good = (phot['use_phot'] == 1) & \
	       (phot['f_IRAC4'] < 1e7) & (phot['f_IRAC3'] < 1e7) & (phot['f_IRAC2'] < 1e7) & (phot['f_IRAC1'] < 1e7) & \
	       (phot['f_F160W']/phot['e_F160W'] > 100) & (fast['lmass'] > 10)

	phot = phot[good]
	fast = fast[good]
	rf = rf[good]
	lineinfo = lineinfo[good]
	mips = mips[good]
	
	# define UVJ flag, S/N, HA EQW
	uvj_flag = calc_uvj_flag(rf)
	sn_F160W = phot['f_F160W']/phot['e_F160W']
	Ha_EQW_obs = lineinfo['Ha_EQW_obs']
	lineinfo.rename_column('zgris' , 'z')
	lineinfo['uvj_flag'] = uvj_flag
	lineinfo['sn_F160W'] = sn_F160W

	# mips
	phot['f_MIPS_24um'] = mips['f24tot']
	phot['e_MIPS_24um'] = mips['ef24tot']

	for i in xrange(len(mips.dtype.names)):
		tempname=mips.dtype.names[i]
		if tempname != 'z_best' and tempname != 'id':
			lineinfo[tempname] = mips[tempname]
		elif tempname == 'z_best':
			lineinfo['z_sfr'] = mips[tempname]
	
	# split into bins
	selection = np.arange(0,np.sum(good))
	random_index = random.sample(xrange(len(selection)), 108)
	fast_out = fast[selection][random_index]
	phot_out = phot[selection][random_index]
	lineinfo = lineinfo[selection][random_index]
	
	# rename bands in photometric catalogs
	for column in phot_out.colnames:
		if column[:2] == 'f_' or column[:2] == 'e_':
			phot_out.rename_column(column, column.lower()+'_'+field.lower())	

	ascii.write(phot_out, output=phot_str_out, 
	            delimiter=' ', format='commented_header')
	ascii.write(fast_out, output=fast_str_out, 
	            delimiter=' ', format='commented_header',
	            include_names=fast.keys()[:11])
	ascii.write(lineinfo, output=ancil_str_out, 
	            delimiter=' ', format='commented_header')
	ascii.write([np.array(phot_out['id'],dtype='int')], output=id_str_out, Writer=ascii.NoHeader)
	print 1/0

def remove_zp_offsets(field,phot,bands_exempt=None):

	zp_offsets = threed_dutils.load_zp_offsets(field)
	nbands     = len(zp_offsets)

	for kk in xrange(nbands):
		filter = zp_offsets[kk]['Band'].lower()+'_'+field.lower()
		if filter not in bands_exempt:
			print filter
			phot['f_'+filter] = phot['f_'+filter]/zp_offsets[kk]['Flux-Correction']
			phot['e_'+filter] = phot['e_'+filter]/zp_offsets[kk]['Flux-Correction']

	return phot

def build_sample_halpha(rm_zp_offsets=True):

	'''
	selects a sample of galaxies "randomly"
	to add: output for the EAZY parameters, so I can include p(z) [or whatever I need for that]
	'''

	# output
	field = 'COSMOS'
	basename = 'testsamp'
	fast_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.fout'
	ancil_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.dat'
	phot_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.cat'
	id_str_out   = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'.ids'

	if rm_zp_offsets:
		fast_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'_zp.fout'
		ancil_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'_zp.dat'
		phot_str_out = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'_zp.cat'
		id_str_out   = '/Users/joel/code/python/threedhst_bsfh/data/'+field+'_'+basename+'_zp.ids'

	# load data
	# use grism redshift
	phot = read_sextractor.load_phot_v41(field)
	fast = read_sextractor.load_fast_v41(field)
	rf = read_sextractor.load_rf_v41(field)
	lineinfo = load_linelist()
	mips = threed_dutils.load_mips_data(os.getenv('APPS')+'/threedhst_bsfh/data/MIPS/cosmos_3dhst.v4.1.4.sfr')
	
	# remove junk
	# 153, 155, 161 are U, V, J
	good = (phot['use_phot'] == 1) & \
	       (rf['L153'] > 0) & (rf['L155'] > 0) & (rf['L161'] > 0) & \
	       (phot['f_IRAC4'] < 1e7) & (phot['f_IRAC3'] < 1e7) & (phot['f_IRAC2'] < 1e7) & (phot['f_IRAC1'] < 1e7) & \
	       (lineinfo['use_grism1'] == 1) & (lineinfo['Ha_EQW_obs']/lineinfo['Ha_EQW_obs_err'] > 2)

	phot = phot[good]
	fast = fast[good]
	rf = rf[good]
	lineinfo = lineinfo[good]
	mips = mips[good]
	
	# define UVJ flag, S/N, HA EQW
	uvj_flag = calc_uvj_flag(rf)
	sn_F160W = phot['f_F160W']/phot['e_F160W']
	Ha_EQW_obs = lineinfo['Ha_EQW_obs']
	lineinfo.rename_column('zgris' , 'z')
	lineinfo['uvj_flag'] = uvj_flag
	lineinfo['sn_F160W'] = sn_F160W

	# mips
	phot['f_MIPS_24um'] = mips['f24tot']
	phot['e_MIPS_24um'] = mips['ef24tot']

	for i in xrange(len(mips.dtype.names)):
		tempname=mips.dtype.names[i]
		if tempname != 'z_best' and tempname != 'id':
			lineinfo[tempname] = mips[tempname]
		elif tempname == 'z_best':
			lineinfo['z_sfr'] = mips[tempname]
	
	# split into bins
	lowlim = np.percentile(Ha_EQW_obs,65)
	highlim = np.percentile(Ha_EQW_obs,95)
	
	selection = (Ha_EQW_obs > lowlim) & (Ha_EQW_obs < highlim)
	random_index = random.sample(xrange(np.sum(selection)), 108)
	fast_out = fast[selection][random_index]
	phot_out = phot[selection][random_index]
	lineinfo = lineinfo[selection][random_index]
	
	# rename bands in photometric catalogs
	for column in phot_out.colnames:
		if column[:2] == 'f_' or column[:2] == 'e_':
			phot_out.rename_column(column, column.lower()+'_'+field.lower())	

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

	if rm_zp_offsets:
		phot_out = remove_zp_offsets(field,phot_out)

	ascii.write(phot_out, output=phot_str_out, 
	            delimiter=' ', format='commented_header')
	ascii.write(fast_out, output=fast_str_out, 
	            delimiter=' ', format='commented_header',
	            include_names=fast.keys()[:11])
	ascii.write(lineinfo, output=ancil_str_out, 
	            delimiter=' ', format='commented_header')
	ascii.write([np.array(phot_out['id'],dtype='int')], output=id_str_out, Writer=ascii.NoHeader)

def load_rachel_sample():

	loc = os.getenv('APPS')+'/threedhst_bsfh/data/bezanson_2015_disps.txt'
	data = ascii.read(loc,format='cds') 
	return data

from itertools import compress, count, imap, islice
from functools import partial
from operator import eq

def nth_item(n, item, iterable):

	'''
	indexing tool, used to match catalogs
	'''

	indicies = compress(count(), imap(partial(eq, item), iterable))
	return next(islice(indicies, n, None), -1)

def build_sample_dynamics():

	'''
	finds Rachel's galaxies in the threedhst catalogs
	basically just matches on (ra, dec, mass), where distance < 2.5" and mass within 0.4 dex
	this will FAIL if we have two identical IDs in UDS/COSMOS... unlikely but if used
	in the future as a template for multi-field runs, beware!!!
	'''

	# output
	field = ['COSMOS','UDS']
	bez = load_rachel_sample()

	fast_str_out = '/Users/joel/code/python/threedhst_bsfh/data/twofield_dynsamp.fout'
	ancil_str_out = '/Users/joel/code/python/threedhst_bsfh/data/twofield_dynsamp.dat'
	phot_str_out = '/Users/joel/code/python/threedhst_bsfh/data/twofield_dynsamp.cat'
	id_str_out   = '/Users/joel/code/python/threedhst_bsfh/data/twofield_dynsamp.ids'

	for bb in xrange(len(field)):

		# load data
		# use grism redshift
		phot = read_sextractor.load_phot_v41(field[bb])
		fast = read_sextractor.load_fast_v41(field[bb])
		rf = read_sextractor.load_rf_v41(field[bb])
		morph = read_sextractor.read_morphology(field[bb],'F160W')

		# do this properly... not just COSMOS
		lineinfo = Table(load_linelist(field=field[bb]))
		mips = Table(threed_dutils.load_mips_data(os.getenv('APPS')+'/threedhst_bsfh/data/MIPS/'+field[bb].lower()+'_3dhst.v4.1.4.sfr'))
		
		# remove phot_flag=0
		good = phot['use_phot'] == 1
		phot = phot[good]
		fast = fast[good]
		rf   = rf[good]
		lineinfo = lineinfo[good]
		mips = mips[good]
		morph = morph[good]

		# matching parameters
		matching_mass   = 0.4 #dex
		matching_radius = 3.0 # arcseconds
		matching_size   = 100.5 # fractional difference
		from astropy.cosmology import WMAP9
		# note: converting to physical sizes, not comoving. I assume this is right.
		arcsec_size = bez['Re']*WMAP9.arcsec_per_kpc_proper(bez['z']).value
		

		matches = np.zeros(len(bez),dtype=np.int)-1
		distance = np.zeros(len(bez))-1
		for nn in xrange(len(bez)):
			catalog = ICRS(bez['RAdeg'][nn], bez['DEdeg'][nn], unit=(u.degree, u.degree))
			close_mass = np.logical_and(
				         (np.abs(fast['lmass']-bez[nn]['logM']) < matching_mass),
			             (fast['z'] < 2.0))
			threed_cat = ICRS(phot['ra'][close_mass], phot['dec'][close_mass], unit=(u.degree, u.degree))
			idx, d2d, d3d = catalog.match_to_catalog_sky(threed_cat)
			if d2d.value*3600 < matching_radius:
				matches[nn] = nth_item(idx,True,close_mass)
				distance[nn] = d2d.value*3600
		

		# print out relevant parameters for visual inspection
		for kk in xrange(len(matches)):
			if matches[kk] == -1:
				continue
			print fast[matches[kk]]['lmass'],bez[kk]['logM'],\
			      fast[matches[kk]]['z'],bez[kk]['z'],\
			      morph[matches[kk]]['re'],arcsec_size[kk],\
			      distance[kk]

			if fast[matches[kk]]['z'] > 2:
				print 1/0

		# define useful things
		lineinfo['uvj_flag'] = calc_uvj_flag(rf)
		lineinfo['sn_F160W'] = phot['f_F160W']/phot['e_F160W']
		lineinfo['sfr'] = mips['sfr']
		lineinfo['z_sfr'] = mips['z_best']

		# mips
		phot['f_MIPS_24um'] = mips['f24tot']
		phot['e_MIPS_24um'] = mips['ef24tot']
		
		# save rachel info
		lineinfo_temp = lineinfo[matches[matches >= 0]]
		bezind = matches > 0
		for i in xrange(len(bez.dtype.names)):
			tempname=bez.dtype.names[i]
			if tempname != 'z' and tempname != 'ID' and tempname != 'Filter':
				lineinfo_temp[tempname] = bez[bezind][tempname]
			elif tempname == 'z':
				lineinfo_temp['z_bez'] = bez[bezind][tempname]

		matches = matches[matches >= 0]

		# rename bands in photometric catalogs
		for column in phot.colnames:
			if column[:2] == 'f_' or column[:2] == 'e_':
				phot.rename_column(column, column.lower()+'_'+field[bb].lower())		

		if bb > 0:
			phot_out = vstack([phot_out, phot[matches]], join_type=u'outer')
			fast_out = vstack([fast_out, fast[matches]],join_type=u'outer')
			rf_out = vstack([rf_out, rf[matches]],join_type=u'outer')
			lineinfo_out = vstack([lineinfo_out, lineinfo_temp],join_type=u'outer')
			mips_out = vstack([mips_out,mips[matches]],join_type=u'outer')
		else:
			phot_out = phot[matches]
			fast_out = fast[matches]
			rf_out = rf[matches]
			lineinfo_out = lineinfo_temp
			mips_out = mips[matches]
	
	print 'n_galaxies = ' + str(len(phot_out))
	
	# note that we're filling masked values in the phot out with -99
	# this means that COSMOS bands have -99 for UDS objects, and vice versa
	ascii.write(phot_out.filled(-99), output=phot_str_out, 
	            delimiter=' ', format='commented_header')
	ascii.write(fast_out, output=fast_str_out, 
	            delimiter=' ', format='commented_header',
	            include_names=fast.keys()[:11])
	ascii.write(lineinfo_out, output=ancil_str_out, 
	            delimiter=' ', format='commented_header')
	ascii.write([np.array(phot_out['id'],dtype='int')], output=id_str_out, Writer=ascii.NoHeader)

