import numpy as np
import brown_io
from scipy.interpolate import interp1d
import magphys_plot_pref
import matplotlib.pyplot as plt
import os
from scipy import stats
from prosp_dutils import smooth_spectrum

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

text_fs = 16

def prep_spectra(alldata,nbins=None,log_qpah=False,equal_n_bins=True):

	### QPAH location
	pnames = alldata[0]['pquantiles']['parnames']
	qpah_idx = pnames == 'duste_qpah'

	# remove galaxies with no IRS spectrum and below log(M) 10
	qpah_stack = np.array([dat['pquantiles']['q50'][qpah_idx][0] for dat in alldata if (dat['residuals'].get('Spitzer IRS',None) is not None) & (dat['pquantiles']['q50'][0] > 10)])

	if log_qpah:
		qpah_stack = np.log10([dat['pquantiles']['q50'][qpah_idx][0] for dat in alldata])

	### bin information
	bin_edges = np.linspace(qpah_stack.min()-0.0001,qpah_stack.max(),num=nbins+1)
	if equal_n_bins:
		bin_edges = stats.mstats.mquantiles(qpah_stack, np.linspace(0,1,nbins+1))
		bin_edges[0] -= 0.0001

	### all QPAH, for iterative purposes
	qpah = np.array([dat['pquantiles']['q50'][qpah_idx][0] for dat in alldata])

	### output containers
	# lambda is just the rest-frame wavelength for the first spectrum, in microns
	upper_lam = 30
	lam = alldata[0]['residuals']['Spitzer IRS']['obs_restlam']
	lam = lam[lam < upper_lam]
	spec = np.zeros(shape=(lam.shape[0],nbins))
	ncount = np.zeros(nbins)

	### load each spectrum
	for ii,dat in enumerate(alldata):

		### load spectrum 
		obs_spec = brown_io.load_spectra(dat['objname'])
		irs_idx = obs_spec['source'] == 3
		if (np.sum(irs_idx) == 0) or (dat['pquantiles']['q50'][0] < 10): # don't stack if we don't have spectra, or low-mass
			continue

		### which bin?
		bin_idx = np.where((qpah[ii] > bin_edges[:-1]) & (qpah[ii] <= bin_edges[1:]))[0][0]

		### normalize spectrum
		# flux_lsun is Lsun/AA (important to do it in observed frame wavelength + flambda, so we get F(total))
		obs_lam = obs_spec['obs_lam'][irs_idx]
		obs_idx = obs_lam < upper_lam*1e4
		norm = np.trapz(obs_spec['flux_lsun'][irs_idx][obs_idx], obs_lam[obs_idx])

		### interpolate
		flux_interp = interp1d(obs_spec['rest_lam'][irs_idx][obs_idx]/1e4, obs_spec['flux'][irs_idx][obs_idx], bounds_error = False, fill_value = 0)
		spec[:,bin_idx] += flux_interp(lam) / norm
		ncount[bin_idx] += 1
		if np.isfinite(spec[:,bin_idx]).sum() != lam.shape[0]:
			print 1/0



	### normalize all to one
	for ii in xrange(nbins): spec[:,ii] /= np.trapz(spec[:,ii], lam)

	out = {}
	out['spec'] = spec
	out['lam'] = lam
	out['bin_edges'] = bin_edges
	out['ncount'] = ncount

	return out

def label_pah(ax):

	lam = [6.2,7.7,8.6,11.3,12.6,17] #8.3
	#for ii in xrange(len(lam)): ax.plot([lam[ii],lam[ii]],[0,1e5],linestyle='--',lw=2,color='k')

	neb_color = '0.7'
	alpha = 0.7
	delta = 0.025
	for wave in lam:
		ax.fill_between([wave*(1-delta),wave*(1+delta)], [0.,0.], [1e5,1e5], 
	                    color=neb_color,
	                    alpha=alpha)

def cloudy_spectrum(ax):

	from prospect.models import model_setup

	param_file = '/Users/joel/code/python/prospector_alpha/parameter_files/brownseds_np/brownseds_np_params.py'
	
	run_params = model_setup.get_run_params(param_file=param_file)
	sps = model_setup.load_sps(**run_params)
	model = model_setup.load_model(**run_params)
	model.params['dust2'] = np.array(0.0)
	obs = model_setup.load_obs(**run_params)
	spec,_,_ = model.mean_model(model.initial_theta, obs, sps=sps)
	model.params['add_neb_emission'] = np.array(False)
	model.params['add_neb_continuum'] = np.array(False)
	spec_neboff,_,_ = model.mean_model(model.initial_theta, obs, sps=sps)

	spec_neb = (spec-spec_neboff)*3e18/sps.wavelengths**2
	in_plot = (sps.wavelengths/1e4 > 6) & (sps.wavelengths/1e4 < 30)
	spec_neb_smooth = smooth_spectrum(sps.wavelengths[in_plot], spec_neb[in_plot], 3500)
	spec_neb_smooth *= 1e5 / spec_neb_smooth.max()

	'''
	neb_color = '0.7'
	alpha = 0.7
	ax.fill_between(sps.wavelengths[in_plot]/1e4, np.zeros_like(spec_neb_smooth), spec_neb_smooth, 
                    color=neb_color,
                    alpha=alpha)

	### label H2 + [ArII] (second is in cloudy but very small)
	ax.fill_between([9.55,9.85],[0,0],[1,1],color=neb_color,alpha=alpha)
	ax.fill_between([6.8,7.1],[0,0],[1,1],color=neb_color,alpha=alpha)
	'''
	lines = ['[ArII]',r'H$_2$','[SIV]','[NeIII]','[SIII]']
	lam = [6.95,9.7,10.45,15.5,18.7]
	# removed because they're too weak, just distracting
	#lines = ['[ArIII]','[NeII]']
	#lam = [9.0,12.8]

	for ii in xrange(len(lines)):
		ax.text(lam[ii]*1.008,0.14,lines[ii],
			    ha='left',fontsize=9.5)
	for ii in xrange(len(lam)): ax.plot([lam[ii],lam[ii]],[0,1e5],linestyle='--',lw=1.5,color='k')


def plot_stacks(outfolder=None,alldata=None,runname='brownseds_np',log_qpah=False,
	            smooth=True,add_cloudy_spectrum=True):

	#### load data if necessary
	if alldata is None:
		alldata = brown_io.load_alldata(runname=runname)
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/pcomp/'

	if log_qpah:
		qpah_label = r'<log(Q$_{\mathrm{PAH}}$)<'
	else:
		qpah_label = r'<Q$_{\mathrm{PAH}}$<'

	#### prepare spectra
	nbins = 4
	spec = prep_spectra(alldata,nbins=nbins,log_qpah=log_qpah)

	### plotting information
	colors = ['#9400D3','#31A9B8','#FF9100','#FF420E']
	fig, ax = plt.subplots(1,1, figsize=(12, 6.5))
	for ii in xrange(nbins):
		pflux = spec['spec'][:,ii]
		if smooth:
			pflux = smooth_spectrum(spec['lam'], spec['spec'][:,ii], 1500)
		ax.plot(spec['lam'],pflux,color=colors[ii],alpha=0.8,lw=2)
		ax.text(0.96,0.23-0.05*ii,"{:.2f}".format(spec['bin_edges'][ii])+qpah_label+"{:.2f}".format(spec['bin_edges'][ii+1]),
			    transform = ax.transAxes,color=colors[ii],ha='right',fontsize=text_fs)
		print spec['ncount'][ii]
	
	if add_cloudy_spectrum:
		cloudy_spectrum(ax)
	label_pah(ax)

	### labels and axes
	ax.set_xlabel(r'wavelength [$\mu$m]')
	ax.set_ylabel(r'$\nu$f$_{\nu}$ [normalized]')

	ax.set_xscale('log',nonposx='clip',subsx=(1,1.2,1.5,2,2.5,4,6,7,8,9))
	#ax.set_xscale('log',nonposx='clip',subsx=(1,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.92,2.5,4,6,7,8))
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)

	ax.set_yscale('log',nonposy='clip',subsy=(1,2,4))
	ax.yaxis.set_minor_formatter(minorFormatter)
	ax.yaxis.set_major_formatter(majorFormatter)

	ax.set_ylim(0.01,0.17)
	ax.set_xlim(5.8,30.5)

	plt.savefig(outfolder+'irs_stack.png')
	os.system('open '+outfolder+'irs_stack.png')

	plt.close()



