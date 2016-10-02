import pickle
import numpy as np

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
	create table, write out in AASTeX format
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

def load_xray_mastercat(xmatch = True):

	'''
	returns flux in (erg/cm^2/s) and object name from brown catalog
	flux is the brightest x-ray source within 1'
	by taking the brightest over multiple tables, we are biasing high (also blended sources?
	    refine in future: we have the count rate error so can get S/Name
	    prefer observatory with highest resolution (Chandra ?) to avoid blending
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
	# names = ('', 'name', 'ra', 'dec', 'count_rate', 'count_rate_error', 'flux', 'database_table', 'observatory', 'class', '_Search_Offset')
	dat = np.loadtxt(location, comments = '#', delimiter='|',skiprows=5,
                     dtype = {'names':([str(n) for n in hdr]),\
                              'formats':(['S40','S40','S40','S40','f16','f16','f16','S40','S40','S40','S40','S40'])})

	#### match based on query string
	if xmatch == True:
		
		#### load Brown positional data
		#### mock up as query parameters
		ra, dec, objname = load_coordinates(dec_in_string=True)
		bcoords = []
		for r, d in zip(ra,dec):
			bcoords.append(str(r)+', '+d)
		match = []
		for query in dat['_Search_Offset']:
			match_str = (query.split('('))[1].split(')')[0]

			match.append(objname[bcoords.index(match_str)])	

		### take brightest X-ray detection per object
		flux = []
		for i, name in enumerate(objname):
			idx = np.array(match) == name

			### if no detections, give it a dummy number
			if idx.sum() == 0:
				flux.append(-99)
				continue 

			### fill it up the easiest way
			flux.append( dat['flux'][idx].max())

		out = {'objname':objname,'flux':np.array(flux)}
		return out
	else:
		return dat

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

