import numpy as np
import prospector_io
import matplotlib.pyplot as plt
import agn_plot_pref
from corner import quantile
import os
from prosp_dutils import asym_errors, running_median, transform_zfraction_to_sfrfraction
from matplotlib.ticker import MaxNLocator
from astropy.cosmology import WMAP9

dpi = 150
red = '#FF3D0D'
blue = '#1C86EE' 

def collate_data(alldata,alldata_noagn):

	### number of random draws
	size = 10000

	### normal parameter labels
	parnames = alldata_noagn[0]['pquantiles']['parnames'].tolist()
	parnames2 = alldata[0]['pquantiles']['parnames'].tolist()
	parlabels = [r'log(M$_{\mathrm{form}}$/M$_{\odot}$)', 'SFH 0-100 Myr', 'SFH 100-300 Myr', 'SFH 300 Myr-1 Gyr', 
	         'SFH 1-3 Gyr', 'SFH 3-6 Gyr', r'$\tau_{\mathrm{V,diffuse}}$', r'log(Z/Z$_{\odot}$)', 'diffuse dust index',
	         'birth-cloud dust', r'dust emission Q$_{\mathrm{PAH}}$',r'dust emission $\gamma$',r'dust emission U$_{\mathrm{min}}$']

	### extra parameters
	eparnames_all = alldata[0]['pextras']['parnames']
	eparnames = ['stellar_mass','sfr_100', 'ssfr_100', 'half_time']
	eparlabels = [r'log(M$_*$) [M$_{\odot}$]',r'log(SFR) [M$_{\odot}$ yr$^{-1}$]',r'log(sSFR) [yr$^{-1}$]', r"log(t$_{\mathrm{half-mass}}$) [Gyr]"]

	### let's do something special here
	fparnames = ['halpha','m23_frac']
	fparlabels = [r'log(H$_{\alpha}$ flux)',r'M$_{\mathrm{0.1-1 Gyr}}$/M$_{\mathrm{total}}$']
	objname = []

	### setup dictionary
	outvals, outq, outerrs, outlabels = {},{},{},{}
	alllabels = parlabels + eparlabels + fparlabels
	for ii,par in enumerate(parnames+eparnames+fparnames): 
		outvals[par] = []
		outq[par] = {}
		outq[par]['q50'],outq[par]['q84'],outq[par]['q16'] = [],[],[]
		outlabels[par] = alllabels[ii]

	### fill with data
	for dat,datnoagn in zip(alldata,alldata_noagn):
		objname.append(dat['objname'])
		for ii,par in enumerate(parnames):
			p1 = np.random.choice(dat['pquantiles']['sample_chain'][:,ii].squeeze(),size=size)
			p2 = np.random.choice(datnoagn['pquantiles']['sample_chain'][:,ii].squeeze(),size=size)
			ratio = p1 - p2
			for q in outq[par].keys(): 
				quant = float(q[1:])/100
				outq[par][q].append(quantile(ratio, [quant])[0])
			outvals[par].append(outq[par]['q50'][-1])
		for par in eparnames:
			match = eparnames_all == par
			match2 = datnoagn['pextras']['parnames'] == par
			p1 = np.random.choice(np.log10(dat['pextras']['flatchain'][:,match]).squeeze(),size=size)
			p2 = np.random.choice(np.log10(datnoagn['pextras']['flatchain'][:,match2]).squeeze(),size=size)
			ratio = p1 - p2
			for q in outq[par].keys(): 
				quant = float(q[1:])/100
				outq[par][q].append(quantile(ratio, [quant])[0])
			outvals[par].append(outq[par]['q50'][-1])

		par = 'halpha'
		ha_idx = dat['model_emline']['emnames'] == 'Halpha'
		p1 = np.random.choice(np.log10(dat['model_emline']['flux']['chain'][:,ha_idx]).squeeze(),size=size)
		p2 = np.random.choice(np.log10(datnoagn['model_emline']['flux']['chain'][:,ha_idx]).squeeze(),size=size)
		ratio = p1 - p2
		for q in outq[par].keys(): 
			quant = float(q[1:])/100
			outq[par][q].append(quantile(ratio, [quant])[0])
		outvals[par].append(outq[par]['q50'][-1])

		### this is super ugly but it works
		# calculate tuniv, create agelim array
		par = 'm23_frac'
		zfrac_idx = np.array(['z_fraction' in p for p in parnames],dtype=bool)
		zfrac_idx2 = np.array(['z_fraction' in p for p in parnames2],dtype=bool)

		tuniv = WMAP9.age(dat['residuals']['phot']['z']).value
		agelims = [0.0,8.0,8.5,9.0,9.5,9.8,10.0]
		agelims[-1] = np.log10(tuniv*1e9)
		time_per_bin = []
		for i in xrange(len(agelims)-1): time_per_bin.append(10**agelims[i+1]-10**agelims[i])

		# now calculate fractions for each of them
		sfrfrac = transform_zfraction_to_sfrfraction(dat['pquantiles']['sample_chain'][:,zfrac_idx2])
		full = np.concatenate((sfrfrac,(1-sfrfrac.sum(axis=1))[:,None]),axis=1)
		mass_fraction = full*np.array(time_per_bin)
		mass_fraction /= mass_fraction.sum(axis=1)[:,None]
		m23_agn = mass_fraction[:,1:3].sum(axis=1)

		sfrfrac = transform_zfraction_to_sfrfraction(datnoagn['pquantiles']['sample_chain'][:,zfrac_idx])
		full = np.concatenate((sfrfrac,(1-sfrfrac.sum(axis=1))[:,None]),axis=1)
		mass_fraction = full*np.array(time_per_bin)
		mass_fraction /= mass_fraction.sum(axis=1)[:,None]
		m23_noagn = mass_fraction[:,1:3].sum(axis=1)

		ratio = np.random.choice(m23_agn,size=size) - np.random.choice(m23_noagn,size=size)
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
	out['ordered_labels'] = np.concatenate((eparnames,np.array(parnames),np.array(fparnames)))
	out['agn_pars'] = agn_pars
	out['objname'] = objname
	return out

def plot(runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,alldata_noagn=None,outfolder=None,idx=None,**popts):

	#### load alldata
	if alldata is None:
		alldata = prospector_io.load_alldata(runname=runname)
	if alldata_noagn is None:
		alldata_noagn = prospector_io.load_alldata(runname=runname_noagn)

	#### make output folder if necessary
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/agn_plots/'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)

	#### collate data
	pdata = collate_data(alldata,alldata_noagn)

	### delta parameters plot
	fig,ax = plot_dpars(pdata,
		                xpar='fagn',xparlabel=r'log(f$_{\mathrm{AGN,MIR}}$)',
		                log_xpar=True,
		                agn_idx=idx,
		                **popts)
	plt.tight_layout()
	plt.savefig(outfolder+'delta_fitpars.png',dpi=dpi)
	plt.close()

def plot_dpars(pdata,xpar=None,xparlabel=None,log_xpar=False, agn_idx=None, **popts):
	'''
	plots a scatterplot
	'''

	#### generate color mapping
	xpar_plot = np.array(pdata['agn_pars'][xpar])
	if log_xpar:
		# add minimum
		min_xpar = (xpar_plot[xpar_plot > 0]).min()
		xpar_plot = np.log10(np.clip(xpar_plot,min_xpar,np.inf))

	#### plot photometry
	#fig, ax = plt.subplots(4,5, figsize=(21,16))
	fig, ax = plt.subplots(2,3, figsize=(15,9.5))
	ax = np.ravel(ax)

	opts = {
	        #'color': blue,
	        'mew': 1.5
	        #'ms': 10
	       }

	toplot = np.array(['half_time','dust2', 'm23_frac','stellar_mass','logzsol','ssfr_100'])

	idx = 0
	for par in toplot:

		# find the complement
		cidx = np.ones_like(xpar_plot,dtype=bool)
		cidx[agn_idx] = False

		errs = pdata['errs'][par]
		ax[idx].errorbar(xpar_plot[cidx],pdata['median'][par][cidx], yerr=[errs[0][cidx],errs[1][cidx]], zorder=-3, 
			             fmt=popts['nofmir_shape'],alpha=popts['nofmir_alpha'],ms=5,color=popts['nofmir_color'],**opts)
		ax[idx].errorbar(xpar_plot[agn_idx],pdata['median'][par][agn_idx], yerr=[errs[0][agn_idx],errs[1][agn_idx]], zorder=-3, 
			             fmt=popts['fmir_shape'],alpha=popts['fmir_alpha'],ms=10,color=popts['fmir_color'],**opts)

		ax[idx].set_ylabel(r'$\Delta(\mathrm{on-off}) $ '+pdata['labels'][par])
		ax[idx].set_xlabel(xparlabel)

		ax[idx].plot([ax[idx].get_xlim()[0],ax[idx].get_xlim()[1]],[0.0,0.0], linestyle='--',color='0.5',lw=1.5,zorder=-5)
		ax[idx].xaxis.set_major_locator(MaxNLocator(5))

		x, y = running_median(xpar_plot,pdata['median'][par],nbins=8,weights=1./(np.array(errs)[0]+np.array(errs)[1]),avg=True)
		ax[idx].plot(x,y,color=red,lw=4,alpha=0.9)

		ylim = np.abs(ax[idx].get_ylim()).max()
		ax[idx].set_ylim(-ylim,ylim)
		#ax[idx].set_xlim(-4,0.2)

		idx +=1

	return fig, ax














