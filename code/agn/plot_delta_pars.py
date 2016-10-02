import numpy as np
import brown_io
import matplotlib.pyplot as plt
import agn_plot_pref
from corner import quantile
import os
from threed_dutils import asym_errors

dpi = 150

def collate_data(alldata,alldata_noagn):

	### number of random draws
	size = 10000

	### normal parameter labels
	parnames = alldata_noagn[0]['pquantiles']['parnames']
	parlabels = [r'log(M/M$_{\odot}$)', 'SFH 0-100 Myr', 'SFH 100-300 Myr', 'SFH 300 Myr-1 Gyr', 
	         'SFH 1-3 Gyr', 'SFH 3-6 Gyr', 'diffuse dust', r'log(Z/Z$_{\odot}$)', 'diffuse dust index',
	         'birth-cloud dust', r'dust emission Q$_{\mathrm{PAH}}$',r'dust emission $\gamma$',r'dust emission U$_{\mathrm{min}}$']

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
	for dat,datnoagn in zip(alldata,alldata_noagn):
		for ii,par in enumerate(parnames):
			p1 = np.random.choice(dat['pquantiles']['random_chain'][:,ii].squeeze(),size=size)
			p2 = np.random.choice(datnoagn['pquantiles']['random_chain'][:,ii].squeeze(),size=size)
			ratio = p1 - p2
			for q in outq[par].keys(): 
				quant = float(q[1:])/100
				outq[par][q].append(quantile(ratio, [quant])[0])
			outvals[par].append(outq[par]['q50'][-1])
		for par in eparnames:
			match = eparnames_all == par
			p1 = np.random.choice(np.log10(dat['pextras']['flatchain'][:,match]).squeeze(),size=size)
			p2 = np.random.choice(np.log10(datnoagn['pextras']['flatchain'][:,match]).squeeze(),size=size)
			ratio = p1 - p2
			for q in outq[par].keys(): 
				quant = float(q[1:])/100
				outq[par][q].append(quantile(ratio, [quant])[0])
			outvals[par].append(outq[par]['q50'][-1])

	### do the errors
	for par in outlabels.keys():
		outerrs[par] = asym_errors(np.array(outq[par]['q50']), 
				                   np.array(outq[par]['q84']),
				                   np.array(outq[par]['q16']),log=False)
		outvals[par] = np.array(outvals[par])

	### AGN parameters
	agn_pars = {}
	pnames = ['fagn', 'agn_tau']
	for p in pnames: agn_pars[p] = []
	agn_parnames = alldata[0]['pquantiles']['parnames']
	for dat in alldata:
		for key in agn_pars.keys():
			agn_pars[key].append(dat['pquantiles']['q50'][agn_parnames==key][0])

	### fill output
	out = {}
	out['median'] = outvals
	out['errs'] = outerrs
	out['labels'] = outlabels
	out['ordered_labels'] = np.concatenate((parnames,eparnames))
	out['agn_pars'] = agn_pars
	return out

def plot(runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,alldata_noagn=None,outfolder=None):

	#### load alldata
	if alldata is None:
		alldata = brown_io.load_alldata(runname=runname)
	if alldata_noagn is None:
		alldata_noagn = brown_io.load_alldata(runname=runname_noagn)

	#### make output folder if necessary
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)

	#### collate data
	pdata = collate_data(alldata,alldata_noagn)

	### BPT PLOT
	fig,ax = plot_dpars(pdata,
		                xpar='fagn',xparlabel=r'log(f$_{\mathrm{AGN}}$)',
		                log_xpar=True)
	plt.tight_layout()
	plt.savefig(outfolder+'delta_fitpars.png',dpi=dpi)
	plt.close()

	os.system('open '+outfolder+'delta_fitpars.png')
	print 1/0

def plot_dpars(pdata,xpar=None,xparlabel=None,log_xpar=False):
	'''
	plots a color-color BPT scatterplot
	'''

	#### generate color mapping
	xpar_plot = np.array(pdata['agn_pars'][xpar])
	if log_xpar:
		xpar_plot = np.log10(xpar_plot)

	#### plot photometry
	fig, ax = plt.subplots(4,4, figsize=(16,16))
	ax = np.ravel(ax)

	opts = {
	        'color': '#1C86EE',
	        'mew': 1.5,
	        'alpha': 0.6,
	        'fmt': 'o'
	       }

	for ii, par in enumerate(pdata['ordered_labels']):

			ax[ii].errorbar(xpar_plot,pdata['median'][par], yerr=pdata['errs'][par],zorder=5,  **opts)
			ax[ii].set_ylabel('AGN--no AGN')
			ax[ii].set_xlabel(xparlabel)
			ax[ii].set_title(pdata['labels'][par])

			ax[ii].plot([ax[ii].get_xlim()[0],ax[ii].get_xlim()[1]],[0.0,0.0], linestyle='--',color='0.5',lw=1.5,zorder=-5)


	return fig, ax













