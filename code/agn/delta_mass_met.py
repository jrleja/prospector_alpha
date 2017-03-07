import numpy as np
from threedhst_diag import add_sfh_plot
import os
import matplotlib.pyplot as plt
import magphys_plot_pref
import matplotlib as mpl
from magphys_plots import median_by_band
import threed_dutils

blue = '#1C86EE' 

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

def collate_data(alldata, alldata_noagn):

	### package up information
	the_data = [alldata, alldata_noagn]
	data_label = ['agn','no_agn']
	output = {}
	for ii in xrange(2):

		# model parameters
		objname = []
		model_pars = {}
		pnames = ['fagn', 'logzsol', 'stellar_mass']
		for p in pnames: 
			model_pars[p] = {'q50':[],'q84':[],'q16':[]}

		#### load model information
		for dat in the_data[ii]:

			parnames = dat['pquantiles']['parnames']
			eparnames = dat['pextras']['parnames']

			for key in model_pars.keys():
				if key in parnames:
					model_pars[key]['q50'].append(dat['pquantiles']['q50'][parnames==key][0])
					model_pars[key]['q84'].append(dat['pquantiles']['q84'][parnames==key][0])
					model_pars[key]['q16'].append(dat['pquantiles']['q16'][parnames==key][0])
				elif key in eparnames:
					model_pars[key]['q50'].append(dat['pextras']['q50'][eparnames==key][0])
					model_pars[key]['q84'].append(dat['pextras']['q84'][eparnames==key][0])
					model_pars[key]['q16'].append(dat['pextras']['q16'][eparnames==key][0])

		for key in model_pars.keys(): 
			for key2 in model_pars[key].keys():
				model_pars[key][key2] = np.array(model_pars[key][key2])

		output[data_label[ii]] = model_pars

	return output

def plot_comparison(runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,alldata_noagn=None,outfolder=None):

	#### load alldata
	if alldata is None:
		import brown_io

		alldata = brown_io.load_alldata(runname=runname)
		alldata_noagn = brown_io.load_alldata(runname=runname_noagn)

	#### make output folder if necessary
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)
	
	### collate data
	### choose galaxies with largest 10 F_AGN
	pdata = collate_data(alldata,alldata_noagn)

	#### MASS-METALLICITY
	fig,ax = plot_massmet(pdata)
	fig.savefig(outfolder+'delta_massmet.png',dpi=150)
	plt.close()

def plot_massmet(pdata):

	fig = plt.figure(figsize=(13, 6))
	ax = [fig.add_axes([0.07, 0.1, 0.35, 0.8]), fig.add_axes([0.5, 0.1, 0.45, 0.8])]

	### pull out fagn
	fagn = np.log10(pdata['agn']['fagn']['q50'])
	ngal = 129
	ylim = (-2.1,0.3)

	### pick out errors
	err_mass_agn = threed_dutils.asym_errors(pdata['agn']['stellar_mass']['q50'],pdata['agn']['stellar_mass']['q84'], pdata['agn']['stellar_mass']['q16'],log=True)
	err_mass_noagn = threed_dutils.asym_errors(pdata['no_agn']['stellar_mass']['q50'],pdata['no_agn']['stellar_mass']['q84'], pdata['no_agn']['stellar_mass']['q16'],log=True)
	err_met_agn = threed_dutils.asym_errors(pdata['agn']['logzsol']['q50'],pdata['agn']['logzsol']['q84'], pdata['agn']['logzsol']['q16'])
	err_met_noagn = threed_dutils.asym_errors(pdata['no_agn']['logzsol']['q50'],pdata['no_agn']['logzsol']['q84'], pdata['no_agn']['logzsol']['q16'])
	mass_agn = np.log10(pdata['agn']['stellar_mass']['q50'])
	mass_noagn = np.log10(pdata['no_agn']['stellar_mass']['q50'])

	### plots
	ax[0].errorbar(mass_noagn,pdata['no_agn']['logzsol']['q50'],xerr=err_mass_noagn,yerr=err_met_noagn,fmt='o', ecolor=blue, capthick=1,elinewidth=1,ms=0.0,alpha=0.5,zorder=-5)
	pts = ax[0].scatter(mass_noagn,pdata['no_agn']['logzsol']['q50'], marker='o', c=fagn, cmap=plt.cm.plasma,s=50,zorder=10)

	ax[0].set_title('No AGN template')
	ax[0].set_xlabel(r'log(stellar mass)')
	ax[0].set_ylabel(r'log(metallicity)')
	ax[0].set_ylim(ylim)

	ax[1].errorbar(mass_agn,pdata['agn']['logzsol']['q50'],xerr=err_mass_agn,yerr=err_met_agn,fmt='o', ecolor=blue, capthick=1,elinewidth=1,ms=0.0,alpha=0.5,zorder=-5)
	pts = ax[1].scatter(mass_agn,pdata['agn']['logzsol']['q50'], marker='o', c=fagn, cmap=plt.cm.plasma,s=50,zorder=10)

	ax[1].set_title('AGN on')
	ax[1].set_xlabel(r'log(stellar mass)')
	ax[1].set_ylabel(r'log(metallicity)')
	ax[1].set_ylim(ylim)

	### color bar
	cb = fig.colorbar(pts, ax=ax[1], aspect=10)
	cb.set_label(r'f$_{\mathrm{MIR}}$')
	cb.solids.set_rasterized(True)
	cb.solids.set_edgecolor("face")

	return fig,ax
