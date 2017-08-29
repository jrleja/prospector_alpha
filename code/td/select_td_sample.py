import td_io, read_data, os, prosp_dutils
import numpy as np
from astropy.table import Table, vstack
from astropy.io import ascii
from astropy.coordinates import ICRS
from astropy import units as u

apps = os.getenv('APPS')

def remove_zp_offsets(field,phot,bands_exempt=None):

	zp_offsets = td_io.load_zp_offsets(field)
	nbands     = len(zp_offsets)

	for kk in xrange(nbands):
		filter = zp_offsets[kk]['Band'].lower()+'_'+field.lower()
		if filter not in bands_exempt:
			phot['f_'+filter] = phot['f_'+filter]/zp_offsets[kk]['Flux-Correction']
			phot['e_'+filter] = phot['e_'+filter]/zp_offsets[kk]['Flux-Correction']

	return phot

def select_massive(phot=None,fast=None,zbest=None,**extras):
	# consider removing zbest['use_zgrism'] cut in the future!! no need for good grism data really
	return (phot['use_phot'] == 1) & (fast['lmass'] > 11) & (zbest['use_zgrism'] == 1)
def select_ha(phot=None,fast=None,zbest=None,gris=None,**extras):
	np.random.seed(2)
	idx = np.where((phot['use_phot'] == 1) & (zbest['use_zgrism'] == 1)  & (gris['Ha_FLUX']/gris['Ha_FLUX_ERR'] > 10) & (gris['Ha_EQW'] > 10))[0]
	return np.random.choice(idx,40)


massive_sample = {
	       		  'selection_function': select_massive,
	       		  'runname': 'td_massive',
	       		  'rm_zp_offsets': True
		  		  }

ha_sample = {
	         'selection_function': select_ha,
	         'runname': 'td_ha',
	         'rm_zp_offsets': True
		  	}

def build_sample(sample=None):
	"""general function to select a sample of galaxies from the 3D-HST catalogs
	each sample is defined by a series of keywords in a dictionary
	"""
	### output
	fields = ['AEGIS','COSMOS','GOODSN','GOODSS','UDS']
	id_str_out = '/Users/joel/code/python/prospector_alpha/data/3dhst/'+sample['runname']+'.ids'
	ids = []

	for field in fields:
		outbase = '/Users/joel/code/python/prospector_alpha/data/3dhst/'+field+'_'+sample['runname']

		# load data
		phot = td_io.load_phot_v41(field)
		fast = td_io.load_fast_v41(field)
		zbest = td_io.load_zbest(field)
		mips = td_io.load_mips_data(field)
		gris = td_io.load_grism_dat(field,process=True)
	
		# make catalog cuts
		good = sample['selection_function'](fast=fast,phot=phot,zbest=zbest,gris=gris)

		phot = phot[good]
		fast = fast[good]
		zbest = zbest[good]
		mips = mips[good]
		gris = gris[good]

		# add in mips
		phot['f_MIPS_24um'] = mips['f24tot']
		phot['e_MIPS_24um'] = mips['ef24tot']

		# save UV+IR SFRs + emission line info in .dat file
		for name in mips.dtype.names:
			if name != 'z' and name != 'id':
				zbest[name] = mips[name]
			elif name == 'z':
				zbest['z_sfr'] = mips[name]
		for name in gris.dtype.names: zbest[name] = gris[name]

		# rename bands in photometric catalogs
		for column in phot.colnames:
			if column[:2] == 'f_' or column[:2] == 'e_':
				phot.rename_column(column, column.lower()+'_'+field.lower())	

		# are we removing the zeropoint offsets? (don't do space photometry!)
		if sample['rm_zp_offsets']:
			bands_exempt = ['irac1_cosmos','irac2_cosmos','irac3_cosmos','irac4_cosmos',\
	                        'f606w_cosmos','f814w_cosmos','f125w_cosmos','f140w_cosmos',\
	                        'f160w_cosmos','mips_24um_cosmos']
			phot = remove_zp_offsets(field,phot,bands_exempt=bands_exempt)

		# write out
		ascii.write(phot, output=outbase+'.cat', 
		            delimiter=' ', format='commented_header',overwrite=True)
		ascii.write(fast, output=outbase+'.fout', 
		            delimiter=' ', format='commented_header',
		            include_names=fast.keys()[:11],overwrite=True)
		ascii.write(zbest, output=outbase+'.dat', 
		            delimiter=' ', format='commented_header',overwrite=True)
		### save IDs
		ids = ids + [field+'_'+str(id) for id in phot['id']]

	ascii.write([ids], output=id_str_out, Writer=ascii.NoHeader,overwrite=True)



























def calc_uvj_flag(rf):
	"""calculate a UVJ flag
	0 = quiescent, 1 = starforming"""
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


def build_sample_onekrun(rm_zp_offsets=True):

	'''
	selects a sample of galaxies for 1k run
	evenly spaced in five redshift bins out to z=3
	'''

	zbins = [(0.4,0.8),(0.8,1.2),(1.2,1.6),(1.6,2.0),(2.0,2.5)]

	# output
	field = 'COSMOS'
	basename = 'onek'
	fast_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.fout'
	ancil_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.dat'
	phot_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.cat'
	id_str_out   = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.ids'

	# load data
	# use grism redshift
	phot = read_sextractor.load_phot_v41(field)
	fast = read_sextractor.load_fast_v41(field)
	rf = read_sextractor.load_rf_v41(field)
	lineinfo = load_linelist()
	mips = prosp_dutils.load_mips_data(field)
	
	# remove junk
	# 153, 155, 161 are U, V, J
	good = (phot['use_phot'] == 1) & \
	       (phot['f_IRAC4'] < 1e7) & (phot['f_IRAC3'] < 1e7) & (phot['f_IRAC2'] < 1e7) & (phot['f_IRAC1'] < 1e7) & \
	       (fast['lmass'] > 10.3)

	print 1/0
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

def build_sample_general():

	'''
	selects a sample of galaxies "randomly"
	to add: output for the EAZY parameters, so I can include p(z) [or whatever I need for that]
	'''

	# output
	field = 'COSMOS'
	basename = 'gensamp'
	fast_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.fout'
	ancil_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.dat'
	phot_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.cat'
	id_str_out   = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.ids'

	# load data
	# use grism redshift
	phot = read_sextractor.load_phot_v41(field)
	fast = read_sextractor.load_fast_v41(field)
	rf = read_sextractor.load_rf_v41(field)
	lineinfo = load_linelist()
	mips = prosp_dutils.load_mips_data(os.getenv('APPS')+'/prospector_alpha/data/MIPS/cosmos_3dhst.v4.1.4.sfr')
	
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
	ascii.write([np.array(phot['id'],dtype='int')], output=id_str_out, Writer=ascii.NoHeader)
	print 1/0

def build_sample_halpha(rm_zp_offsets=True):

	'''
	selects a sample of galaxies "randomly"
	to add: output for the EAZY parameters, so I can include p(z) [or whatever I need for that]
	'''

	# output
	field = 'COSMOS'
	basename = 'testsamp'
	fast_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.fout'
	ancil_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.dat'
	phot_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.cat'
	id_str_out   = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'.ids'

	if rm_zp_offsets:
		fast_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'_zp.fout'
		ancil_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'_zp.dat'
		phot_str_out = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'_zp.cat'
		id_str_out   = '/Users/joel/code/python/prospector_alpha/data/'+field+'_'+basename+'_zp.ids'

	# load data
	# use grism redshift
	phot = read_sextractor.load_phot_v41(field)
	fast = read_sextractor.load_fast_v41(field)
	rf = read_sextractor.load_rf_v41(field)
	lineinfo = load_linelist()
	mips = prosp_dutils.load_mips_data(os.getenv('APPS')+'/prospector_alpha/data/MIPS/cosmos_3dhst.v4.1.4.sfr')
	
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

	loc = os.getenv('APPS')+'/prospector_alpha/data/bezanson_2015_disps.txt'
	data = ascii.read(loc,format='cds') 
	return data

def nth_item(n, item, iterable):

	'''
	indexing tool, used to match catalogs
	'''

	from itertools import compress, count, imap, islice
	from functools import partial
	from operator import eq

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

	fast_str_out = '/Users/joel/code/python/prospector_alpha/data/twofield_dynsamp.fout'
	ancil_str_out = '/Users/joel/code/python/prospector_alpha/data/twofield_dynsamp.dat'
	phot_str_out = '/Users/joel/code/python/prospector_alpha/data/twofield_dynsamp.cat'
	id_str_out   = '/Users/joel/code/python/prospector_alpha/data/twofield_dynsamp.ids'

	for bb in xrange(len(field)):

		# load data
		# use grism redshift
		phot = read_sextractor.load_phot_v41(field[bb])
		fast = read_sextractor.load_fast_v41(field[bb])
		rf = read_sextractor.load_rf_v41(field[bb])
		morph = read_sextractor.read_morphology(field[bb],'F160W')

		# do this properly... not just COSMOS
		lineinfo = Table(load_linelist(field=field[bb]))
		mips = Table(prosp_dutils.load_mips_data(os.getenv('APPS')+'/prospector_alpha/data/MIPS/'+field[bb].lower()+'_3dhst.v4.1.4.sfr'))
		
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

