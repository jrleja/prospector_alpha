import pickle
import numpy as np
import os

#### where do alldata pickle files go?
outpickle = '/Users/joel/code/magphys/data/pickles'

def save_spec_cal(spec_cal,runname='brownseds'):
	output = outpickle+'/spec_calibration.pickle'
	pickle.dump(spec_cal,open(output, "wb"))

def load_spec_cal(runname='brownseds'):
	with open(outpickle+'/spec_calibration.pickle', "rb") as f:
		spec_cal=pickle.load(f)
	return spec_cal

def load_alldata(runname='brownseds'):

	output = outpickle+'/'+runname+'_alldata.pickle'
	with open(output, "rb") as f:
		alldata=pickle.load(f)
	return alldata

def load_moustakas_data(objnames = None):

	'''
	specifically written to load optical emission line fluxes, of the "radial strip" variety
	this corresponds to the aperture used in the Brown sample

	if we pass a list of object names, return a sorted, matched list
	otherwise return everything

	returns in units of 10^-15^erg/s/cm^2
	'''

	#### load data
	# arcane vizier formatting means I'm using astropy tables here
	from astropy.io import ascii
	foldername = os.getenv('APPS')+'/threedhst_bsfh/data/Moustakas+10/'
	filename = 'table3.dat'
	readme = 'ReadMe'
	table = ascii.read(foldername+filename, readme=foldername+readme)

	#### filter to only radial strips
	accept = table['Spectrum'] == 'Radial Strip'
	table = table[accept.data]

	#####
	if objnames is not None:
		outtable = []
		for name in objnames:
			match = table['Name'] == name
			if np.sum(match.data) == 0:
				outtable.append(None)
				continue
			else:
				outtable.append(table[match.data])
	else:
		outtable = table

	return outtable

def load_moustakas_newdat(objnames = None):

	'''
	access (new) Moustakas line fluxes, from email in Jan 2016

	if we pass a list of object names, return a sorted, matched list
	otherwise return everything

	returns in units of erg/s/cm^2
	'''

	#### load data
	from astropy.io import fits
	filename = os.getenv('APPS')+'/threedhst_bsfh/data/Moustakas_new/atlas_specdata_solar_drift_v1.1.fits'
	hdulist = fits.open(filename)

	##### match
	if objnames is not None:
		outtable = []
		objnames = np.core.defchararray.replace(objnames, ' ', '')  # strip spaces
		for name in objnames:
			match = hdulist[1].data['GALAXY'] == name
			if np.sum(match.data) == 0:
				outtable.append(None)
				continue
			else:
				outtable.append(hdulist[1].data[match])
	else:
		outtable = hdulist.data

	return outtable

def save_alldata(alldata,runname='brownseds'):

	output = outpickle+'/'+runname+'_alldata.pickle'
	pickle.dump(alldata,open(output, "wb"))

def write_results(alldata,outfolder):
	'''
	create table for Prospector-Alpha paper, write out in AASTeX format
	'''

	data, errup, errdown, names, fmts, objnames = [], [], [], [], [], []
	objnames = [dat['objname'] for dat in alldata]

	#### gather regular parameters
	par_to_write = ['logmass','dust2','logzsol']
	theta_names = alldata[0]['pquantiles']['parnames']
	names.extend([r'log(M/M$_{\odot}$)',r'$\tau_{\mathrm{diffuse}}$',r'log(Z/Z$_{\odot}$)'])
	fmts.extend(["{:.2f}","{:.2f}","{:.2f}"])
	for p in par_to_write: 
		idx = theta_names == p
		data.append([dat['pquantiles']['q50'][idx][0] for dat in alldata])
		errup.append([dat['pquantiles']['q84'][idx][0]-dat['pquantiles']['q50'][idx][0] for dat in alldata])
		errdown.append([dat['pquantiles']['q50'][idx][0]-dat['pquantiles']['q16'][idx][0] for dat in alldata])

	#### gather error parameters
	epar_to_write = ['sfr_100','ssfr_100','half_time']
	theta_names = alldata[0]['pextras']['parnames']
	names.extend([r'log(SFR)',r'log(sSFR)',r'log(t$_{\mathrm{half}}$)'])
	fmts.extend(["{:.2f}","{:.2f}","{:.2f}"])
	for p in epar_to_write: 
		idx = theta_names == p
		data.append([np.log10(dat['pextras']['q50'][idx][0]) for dat in alldata])
		errup.append([np.log10(dat['pextras']['q84'][idx][0]) - np.log10(dat['pextras']['q50'][idx][0]) for dat in alldata])
		errdown.append([np.log10(dat['pextras']['q50'][idx][0])-np.log10(dat['pextras']['q16'][idx][0]) for dat in alldata])

	#### write formatted data (for putting into the above)
	nobj = len(objnames)
	ncols = len(data)
	with open(outfolder+'results.dat', 'w') as f:
		for i in xrange(nobj):
			f.write(objnames[i])
			for j in xrange(ncols):
				string = ' & $'+fmts[j].format(data[j][i])+'^{+'+fmts[j].format(errup[j][i])+'}_{-'+fmts[j].format(errdown[j][i])+'}$'
				f.write(string)
			f.write(' \\\ \n')

def load_spectra(objname, nufnu=True):
	
	# flux is read in as ergs / s / cm^2 / Angstrom
	# the source key is:
	# 0 = model
	# 1 = optical spectrum
	# 2 = Akari
	# 3 = Spitzer IRS

	foldername = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/spectra/'
	rest_lam, flux, obs_lam, source = np.loadtxt(foldername+objname.replace(' ','_')+'_spec.dat',comments='#',unpack=True)

	lsun = 3.846e33  # ergs/s
	flux_lsun = flux / lsun #

	# convert to flam * lam
	flux = flux * obs_lam

	# convert to janskys, then maggies * Hz
	flux = flux * 1e23 / 3631

	out = {}
	out['rest_lam'] = rest_lam
	out['flux'] = flux
	out['flux_lsun'] = flux_lsun
	out['obs_lam'] = obs_lam
	out['source'] = source

	return out

def load_coordinates(dec_in_string=False):

	from astropy.io import fits

	location = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/photometry/table1.fits'
	hdulist = fits.open(location)

	### convert from hours to degrees
	ra, dec = [], []
	for i, x in enumerate(hdulist[1].data['RAm']):
		r = hdulist[1].data['RAh'][i] * (360./24) + hdulist[1].data['RAm'][i] * (360./(24*60)) + hdulist[1].data['RAs'][i] * (360./(24*60*60))
		
		if dec_in_string:
			d = str(hdulist[1].data['DE-'][i])+str(hdulist[1].data['DEd'][i])+' '+str(hdulist[1].data['DEm'][i])+' '+"{:.1f}".format(float(hdulist[1].data['DEs'][i]))
		else:
			d = hdulist[1].data['DEd'][i] + hdulist[1].data['DEm'][i] / 60. + hdulist[1].data['DEs'][i] / 3600.
			if str(hdulist[1].data['DE-'][i]) == '-':
				d = -d
		ra.append(r)
		dec.append(d)

	return np.array(ra),np.array(dec),hdulist[1].data['Name'] 

def write_coordinates():

	outloc = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/photometry/coords.dat'
	ra, dec, name = load_coordinates(dec_in_string=True)
	with open(outloc, 'w') as f:
		for r, d in zip(ra,dec):
			f.write(str(r)+', '+d+'; ')

def load_xray_mastercat(xmatch = True,maxradius=30):

	'''
	returns flux in (erg/cm^2/s) and object name from brown catalog
	flux is the brightest x-ray source within 1'
	by taking the brightest over multiple tables, we are biasing high (also blended sources?
	    prefer observatory with highest resolution (Chandra ?) to avoid blending
	    think about a cut on location (e.g., within 10'', or 30'')
	    would like to do ERROR_RADIUS cut but most entries don't have it. similar with EXPOSURE.
	    if we could translate COUNT_RATE into FLUX for ARBITRARY TELESCOPE AND DATA TABLE then we could include many more sources
	'''

	location = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/photometry/xray/xray_mastercat.dat'

	#### extract headers
	with open(location, 'r') as f:
		for line in f:
			if line[:5] != '|name':
				continue
			else:
				hdr = line
				hdr = hdr.replace(' ', '').split('|')[:-1]
				break

	#### load
	# names = ('', 'name', 'ra', 'dec', 'count_rate', 'count_rate_error', 'flux', 'database_table', 'observatory','error_radius', 'exposure', 'class', '_Search_Offset')
	dat = np.loadtxt(location, comments = '#', delimiter='|',skiprows=5,
                     dtype = {'names':([str(n) for n in hdr]),\
                              'formats':(['S40','S40','S40','S40','f16','f16','f16','S40','S40','S40','S40','S40','S40','S40'])})

	### remove whitespace from strings
	for i in xrange(dat.shape[0]):
		dat['database_table'][i] = str(np.core.defchararray.strip(dat['database_table'][i]))
		dat['observatory'][i] = str(np.core.defchararray.strip(dat['observatory'][i]))

	#### match based on query string
	if xmatch == True:
		
		#### load Brown positional data
		#### mock up as query parameters
		ra, dec, objname = load_coordinates(dec_in_string=True)
		bcoords = []
		for r, d in zip(ra,dec):
			bcoords.append(str(r)+', '+d)
		match, offset = [], []
		for query in dat['_Search_Offset']:
			match_str = (query.split('('))[1].split(')')[0]
			match.append(objname[bcoords.index(match_str)])	

			n = 0
			offset_float = None
			while offset_float == None:
				try: 
					offset_float = float(query.split(' ')[n])
				except ValueError:
					n+=1
			offset.append(offset_float)

		offset = np.array(offset)

		### take brightest X-ray detection per object
		flux, flux_err, observatory, database = [], [], [], []
		for i, name in enumerate(objname):

			### find matches in the query with nonzero flux entries within MAXIMUM radius in arcseconds (max is 1')
			# forbidden datatables either use bandpasses above 10 keV or flux definition is unclear
			idx = (np.array(match) == name) & (dat['flux'] != 0.0) & (offset < maxradius/60.) & \
				  (dat['database_table'] != 'INTAGNCAT') & (dat['database_table'] != 'INTIBISAGN') & \
				  (dat['database_table'] != 'BMWHRICAT') & (dat['database_table'] != 'IBISCAT4') & \
				  (dat['database_table'] != 'INTIBISASS') & (dat['database_table'] != 'ULXRBCAT')
			### if no detections, give it a dummy number
			if idx.sum() == 0:
				flux.append(-99)
				flux_err.append(0.0)
				observatory.append('no match')
				database.append('no match')
				continue 

			### choose the detection to keep
			# prefer chandra
			ch_idx = np.core.defchararray.strip(dat['observatory'][idx]) == 'CHANDRA'
			if ch_idx.sum() > 0:
				dat['flux'][idx][~ch_idx] = -1
			idx_keep = dat['flux'][idx].argmax()

			### fill out data
			cfactor = correct_for_window(dat['database_table'][idx][idx_keep])
			flux.append(dat['flux'][idx][idx_keep]*cfactor)
			fractional_count_err = dat['count_rate_error'][idx][idx_keep]/dat['count_rate'][idx][idx_keep]
			if np.isnan(fractional_count_err):
				fractional_count_err = 0.0
			flux_err.append(flux[-1] * fractional_count_err)
			observatory.append(dat['observatory'][idx][idx_keep])
			database.append(dat['database_table'][idx][idx_keep])

		out = {'objname':objname,
		       'flux':np.array(flux),
		       'flux_err':np.array(flux_err),
		       'observatory':np.array(observatory),
		       'database':np.array(database)}
		return out
	else:
		return dat

def table_window(table):

	if table == 'ASCAGIS':
		low, high = 0.7, 7.0
	elif table == 'CHNGPSCLIU':
		low, high = 0.3, 8.0
	elif table == 'CSC':
		low, high = 0.5, 7.0
	elif table == 'CXOXASSIST':
		low, high = 0.3, 8.0
	elif table == 'EINGALCAT':
		low, high = 0.2, 4.0
	elif table == 'ETGALXRAY': #CAREFUL
		low, high = 0.088, 17.25
	elif table == 'RASSBSCPGC':
		low, high = 0.1, 2.4
	elif table == 'RASSDSSAGN':
		low, high = 0.1, 2.4
	elif table == 'RBSCNVSS':
		low, high = 0.1, 2.4
	elif table == 'ROSATRQQ':
		low, high = 0.1, 2.4
	elif table == 'ROXA':
		low, high = 0.1, 2.4
	elif table == 'SACSTPSCAT': 
		low, high = 0.5, 8.0
	elif table == 'TARTARUS':
		low, high = (2,7.5), (5,10)
	elif table == 'ULXNGCAT':
		low, high = 0.3, 10
	elif table == 'WGACAT':
		low, high = 0.05, 2.5
	elif table == 'XMMSLEWCLN':
		low, high = 0.2, 12
	elif table == 'XMMSSC':
		low, high = 0.2, 12
	elif table == 'XMMSSCLWBS':
		low, high = 0.2, 12
	else:
		print 1/0

	# return list if it's not a tuple
	try:
		len(low)
	except TypeError: 
		low = [low]
		high = [high]

	return low, high


def correct_for_window(table, targlow = 0.5, targhigh = 8):

	# Want 0.5-8 keV
	# n(E)dE proportional to E^-gamma dE
	# n(E)dE = c E^-gamma dE where c is a constant
	# integrate
	# Etot = int_Elow^Ehigh (c E^-gamma dE)
	# Etot = (1./(-gamma+1)) c E^(-gamma+1) |_Elow^Ehigh
	# Etot = (1./(-gamma+1)) c [Ehigh^(-gamma+1) - Elow^(-gamma+1)]

	factor = 0.0
	gamma = -1.8

	low, high = table_window(table)

	for l, h in zip(low, high):
		factor += h**(gamma+1) - l**(gamma+1)

	fscale = (targhigh**(1+gamma) - targlow**(1+gamma)) / factor

	return fscale

def plot_brown_coordinates():

	'''
	plot the coordinates above
	''' 

	import matplotlib.pyplot as plt

	ra, dec = load_coordinates()

	plt.plot(ra, dec, 'o', linestyle=' ', mew=2, alpha = 0.8, ms = 10)
	plt.xlabel('Right Ascension [degrees]')
	plt.ylabel('Declination [degrees]')

	plt.show()

def write_euphrasio_data(alldata):

	# All I need is UV (intrinsic and observed), and mid-IR fluxes, sSFRs, SFRs, Mstars, and tau_Vs (tau_FUVs if it's also available)

