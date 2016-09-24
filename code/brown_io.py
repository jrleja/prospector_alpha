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











