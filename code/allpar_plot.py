import numpy as np
import matplotlib.pyplot as plt
import threed_dutils
import magphys_plot_pref
from matplotlib.ticker import MaxNLocator
import copy
import matplotlib as mpl

mpl.rcParams.update({'font.size': 18})
mpl.rcParams.update({'font.weight': 500})
mpl.rcParams.update({'axes.labelweight': 500})

def arrange_data(alldata):

	### normal parameter labels
	parnames = alldata[0]['pquantiles']['parnames']
	parlabels = [r'log(M/M$_{\odot}$)', 'SFH 0-100 Myr', 'SFH 100-300 Myr', 'SFH 300 Myr-1 Gyr', 
	         'SFH 1-3 Gyr', 'SFH 3-6 Gyr', 'diffuse dust', r'log(Z/Z$_{\odot}$)', 'diffuse dust index',
	         'birth-cloud dust', r'dust emission Q$_{\mathrm{PAH}}$',r'dust emission $\gamma$',r'dust emission U$_{\mathrm{min}}$']
	if len(parnames) == 15:
		parlabels.extend([r'log(f$_{\mathrm{AGN}}$)',r'$\tau_{\mathrm{AGN}}$'])

	### extra parameters
	eparnames_all = alldata[0]['pextras']['parnames']
	eparnames = ['sfr_100', 'ssfr_100', 'half_time']
	eparlabels = ['log(SFR) [100 Myr]','log(sSFR) [100 Myr]', r"log(t$_{\mathrm{half-mass}})$ [Gyr]"]

	### setup dictionary
	outvals, outq, outerrs, outlabels = {},{},{},{}
	for ii,par in enumerate(parnames): 
		outvals[par] = []
		outq[par] = {}
		outq[par]['q50'],outq[par]['q84'],outq[par]['q16'] = [],[],[]
		outlabels[par] = parlabels[ii]
	for ii,par in enumerate(eparnames):
		outvals[par] = []
		outq[par] = {}
		outq[par]['q50'],outq[par]['q84'],outq[par]['q16'] = [],[],[]
		outlabels[par] = eparlabels[ii]

	### fill with data
	for dat in alldata:
		for ii,par in enumerate(parnames):
			if par == 'fagn':
				for q in outq[par].keys(): outq[par][q].append(np.log10(dat['pquantiles'][q][ii]))
				outvals[par].append(np.log10(dat['pquantiles']['q50'][ii]))
			else:
				for q in outq[par].keys(): outq[par][q].append(dat['pquantiles'][q][ii])
				outvals[par].append(dat['pquantiles']['q50'][ii])
		for par in eparnames:
			match = eparnames_all == par
			for q in outq[par].keys(): outq[par][q].append(np.log10(dat['pextras'][q][match][0]))
			outvals[par].append(np.log10(dat['pextras']['q50'][match][0]))

	### do the errors
	for par in outlabels.keys():
		outerrs[par] = threed_dutils.asym_errors(np.array(outq[par]['q50']), 
				                                 np.array(outq[par]['q84']),
				                                 np.array(outq[par]['q16']),log=False)
		outvals[par] = np.array(outvals[par])

	### fill output
	out = {}
	out['median'] = outvals
	out['errs'] = outerrs
	out['labels'] = outlabels
	out['ordered_labels'] = np.concatenate((parnames,eparnames))

	return out
	
def allpar_plot(alldata,hflag,outfolder,lowmet=True):

	### load data
	dat = arrange_data(alldata)
	npars = len(dat['labels'].keys())

	### plot preferences
	fig, ax = plt.subplots(ncols=npars, nrows=npars, figsize=(npars*3,npars*3))
	fig.subplots_adjust(wspace=0.0,hspace=0.0,top=0.95,bottom=0.05,left=0.05,right=0.95)
	opts = {
	        'color': '#1C86EE',
	        'mew': 1.5,
	        'alpha': 0.6,
	        'fmt': 'o'
	       }
	hopts = copy.deepcopy(opts)
	hopts['color'] = '#FF420E'
	outname = outfolder+'all_parameter.png'

	### color low-metallicity, high-mass galaxies
	# hijack Herschel flag
	if lowmet:
		parnames = alldata[0]['pquantiles']['parnames']
		met_idx = parnames == 'logzsol'
		mass_idx = parnames == 'logmass'
		met_q50 = np.array([data['pquantiles']['q50'][met_idx][0] for data in alldata])
		mass_q50 = np.array([data['pquantiles']['q50'][mass_idx][0] for data in alldata])

		hflag = (met_q50 < -1.0) & (mass_q50 > 9.5)

		outname = outfolder+'all_parameter_lowmet.png'

	for yy, ypar in enumerate(dat['ordered_labels']):
		for xx, xpar in enumerate(dat['ordered_labels']):

			# turn off the dumb ones
			if xx >= yy:
				ax[yy,xx].axis('off')
				continue

			ax[yy,xx].errorbar(dat['median'][xpar][~hflag],dat['median'][ypar][~hflag],
			                   xerr=[dat['errs'][xpar][0][~hflag],dat['errs'][xpar][1][~hflag]], 
			                   yerr=[dat['errs'][ypar][0][~hflag],dat['errs'][ypar][1][~hflag]], 
			                   **opts)

			ax[yy,xx].errorbar(dat['median'][xpar][hflag],dat['median'][ypar][hflag],
			                   xerr=[dat['errs'][xpar][0][hflag],dat['errs'][xpar][1][hflag]], 
			                   yerr=[dat['errs'][ypar][0][hflag],dat['errs'][ypar][1][hflag]], 
			                   **hopts)

			#### RANGE
			minx,maxx = dat['median'][xpar].min(),dat['median'][xpar].max()
			dynx = (maxx-minx)*0.1
			ax[yy,xx].set_xlim(minx-dynx, maxx+dynx)

			miny,maxy = dat['median'][ypar].min(),dat['median'][ypar].max()
			dyny = (maxy-miny)*0.1
			ax[yy,xx].set_ylim(miny-dyny, maxy+dyny)


			#### LABELS
			if xx % npars == 0:
				ax[yy,xx].set_ylabel(dat['labels'][ypar])
			else:
				for tl in ax[yy,xx].get_yticklabels():tl.set_visible(False)

			if yy == npars-1:
				ax[yy,xx].set_xlabel(dat['labels'][xpar])
			else:
				for tl in ax[yy,xx].get_xticklabels():tl.set_visible(False)

			ax[yy,xx].xaxis.set_major_locator(MaxNLocator(5))
			ax[yy,xx].yaxis.set_major_locator(MaxNLocator(5))

	plt.savefig(outname,dpi=100)
	plt.close()

	mpl.rcParams.update({'font.weight': 400})
	mpl.rcParams.update({'axes.labelweight': 400})





















