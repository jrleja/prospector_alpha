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

def build_sample_halpha():

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
			             (np.abs((morph['re']-arcsec_size[nn])/arcsec_size[nn]) < matching_size))
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

