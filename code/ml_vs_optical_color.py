import threed_dutils
import os
import numpy as np
import matplotlib.pyplot as plt
import magphys_plot_pref
from astropy.cosmology import WMAP9 as cosmo

def collate_data(alldata):

	### data definitions from 
	# http://www.ucolick.org/~cnaw/sun.html
	bmag_sun = 5.47
	kmag_sun = 3.28
	gmag_sun = 5.21

	ab_to_vega_b = 0.09
	ab_to_vega_R = -0.21
	ab_to_vega_k = -1.9
	ab_to_vega_g = 0.1
	ab_to_vega_r = -0.15
	ab_to_vega_i = -0.38

	### save all model parameters
	model_pars = {}
	parnames = alldata[0]['pquantiles']['parnames']
	for p in parnames: model_pars[p] = {'q50':[],'q84':[],'q16':[]}
	eparnames = ['sfr_100', 'ssfr_100', 'half_time']
	for p in eparnames: model_pars[p] = {'q50':[],'q84':[],'q16':[]}

	### loop over galaxies
	ngal = len(alldata)
	ml_b, ml_k, br_color = [np.zeros(shape=(ngal,3)) for i in xrange(3)]
	for i,dat in enumerate(alldata):

		### find indices
		if i == 0:
			 fnames = dat['mphot']['mags']['name']
			 bidx = fnames == 'bessell_B'
			 ridx = fnames == 'bessell_R'
			 kidx = fnames == 'twomass_Ks'

			 pnames = dat['pquantiles']['parnames']
			 mass_idx = pnames == 'logmass'

		### load up model parameters
		for key in model_pars.keys():
			try:
				model_pars[key]['q50'].append(dat['pquantiles']['q50'][parnames==key][0])
				model_pars[key]['q84'].append(dat['pquantiles']['q84'][parnames==key][0])
				model_pars[key]['q16'].append(dat['pquantiles']['q16'][parnames==key][0])
			except IndexError:
				idx = dat['pextras']['parnames'] == key
				model_pars[key]['q50'].append(dat['pextras']['q50'][idx][0])
				model_pars[key]['q84'].append(dat['pextras']['q84'][idx][0])
				model_pars[key]['q16'].append(dat['pextras']['q16'][idx][0])

		### M/L calculations
		mass = 10**dat['pquantiles']['random_chain'][:,mass_idx].squeeze()
		bmag = -2.5*np.log10(dat['mphot']['mags']['mags'][bidx,:]).squeeze()
		rmag = -2.5*np.log10(dat['mphot']['mags']['mags'][ridx,:]).squeeze()
		kmag = -2.5*np.log10(dat['mphot']['mags']['mags'][kidx,:]).squeeze()

		ml_b_chain = mass / (10**((bmag_sun-(bmag+ab_to_vega_b))/2.5))
		ml_k_chain = mass / (10**((kmag_sun-(kmag+ab_to_vega_k))/2.5))
		br_color_chain = (bmag+ab_to_vega_b) - (rmag+ab_to_vega_R)

		ml_b[i,:] = np.percentile(ml_b_chain,[50.0,84.0,16.0])
		ml_k[i,:] = np.percentile(ml_k_chain,[50.0,84.0,16.0])
		br_color[i,:] = np.percentile(br_color_chain,[50.0,84.0,16.0])

	#### do it again, for g-r, g-i, and g-band M/L + K-band M/L
	ml_g, ml_kobs, gr_color, gi_color = [np.zeros(shape=(ngal,3)) for i in xrange(4)]
	for i,dat in enumerate(alldata):

		fnames = dat['filters']
		g_idx = fnames == 'sdss_g0'
		r_idx = fnames == 'sdss_r0'
		i_idx = fnames == 'sdss_i0'
		k_idx = fnames == 'twomass_Ks'

		if g_idx.sum() + r_idx.sum() + i_idx.sum() != 3:
			print 1/0

		### from observed distance to 10pc
		# divide by 1+z since we're in fnu
		z = dat['residuals']['phot']['z']
		lumdist = cosmo.luminosity_distance(z).value
		dfactor = (lumdist*1e5)**2 / (1+z) 

		### sample random flux
		gmag = dat['obs_maggies'][g_idx]
		rmag = dat['obs_maggies'][r_idx]
		imag = dat['obs_maggies'][i_idx]
		if k_idx.sum() == 1:
			kmag = dat['obs_maggies'][k_idx]
		else:
			kmag = np.nan

		nsamp = 2000
		gflux = np.random.normal(loc=gmag, scale=gmag*0.05, size=nsamp)
		rflux = np.random.normal(loc=rmag, scale=rmag*0.05, size=nsamp)
		iflux = np.random.normal(loc=imag, scale=imag*0.05, size=nsamp)
		kflux = np.random.normal(loc=kmag, scale=kmag*0.05, size=nsamp)

		### convert to absolute magnitude
		gmag = -2.5*np.log10(gflux*dfactor)
		rmag = -2.5*np.log10(rflux*dfactor)
		imag = -2.5*np.log10(iflux*dfactor)
		kmag = -2.5*np.log10(kflux*dfactor)

		### convert to M/L
		mass = 10**dat['pquantiles']['random_chain'][:,mass_idx].squeeze()
		ml_g_chain = mass / (10**((gmag_sun-(gmag+ab_to_vega_g))/2.5))
		ml_kobs_chain = mass / (10**((kmag_sun-(kmag+ab_to_vega_k))/2.5))
		gr_color_chain = (gmag+ab_to_vega_g) - (rmag+ab_to_vega_r)
		gi_color_chain = (gmag+ab_to_vega_g) - (imag+ab_to_vega_i)

		ml_g[i,:] = np.percentile(ml_g_chain,[50.0,84.0,16.0])
		ml_kobs[i,:] = np.percentile(ml_kobs_chain,[50.0,84.0,16.0])
		gr_color[i,:] = np.percentile(gr_color_chain,[50.0,84.0,16.0])
		gi_color[i,:] = np.percentile(gi_color_chain,[50.0,84.0,16.0])

	out = {}
	out['ml_b'] = np.log10(ml_b[:,0])
	out['ml_b_err'] = threed_dutils.asym_errors(ml_b[:,0], ml_b[:,1], ml_b[:,2], log=True)

	out['ml_k'] = np.log10(ml_k[:,0])
	out['ml_k_err'] = threed_dutils.asym_errors(ml_k[:,0], ml_k[:,1], ml_k[:,2], log=True)

	out['br_color'] = br_color[:,0]
	out['br_color_err'] = threed_dutils.asym_errors(br_color[:,0], br_color[:,1], br_color[:,2], log=False)

	out['ml_g'] = np.log10(ml_g[:,0])
	out['ml_g_err'] = threed_dutils.asym_errors(ml_g[:,0], ml_g[:,1], ml_g[:,2], log=True)

	out['ml_kobs'] = np.log10(ml_kobs[:,0])
	out['ml_kobs_err'] = threed_dutils.asym_errors(ml_kobs[:,0], ml_kobs[:,1], ml_kobs[:,2], log=True)

	out['gr_color'] = gr_color[:,0]
	out['gr_color_err'] = threed_dutils.asym_errors(gr_color[:,0], gr_color[:,1], gr_color[:,2], log=False)

	out['gi_color'] = gi_color[:,0]
	out['gi_color_err'] = threed_dutils.asym_errors(gi_color[:,0], gi_color[:,1], gi_color[:,2], log=False)

	out['model_parameters'] = model_pars
	return out

def do_all(runname='brownseds_np',alldata=None,outfolder=None):

	#### load alldata
	if alldata is None:
		alldata = brown_io.load_alldata(runname=runname)

	#### make output folder if necessary
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/pcomp/'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)

	pdata = collate_data(alldata)

	#### plot in observed ML versus color
	outname = outfolder+'ml_vs_gr_obs.png'
	plot_obs(pdata,outname=outname,cpar='ssfr_100',cpar_label=r'log(sSFR/yr$^{-1}$)',log_cpar=True,
		     xlim = [0.0,1.4])

	outname = outfolder+'ml_vs_gi_obs.png'
	plot_obs(pdata,outname=outname,cpar='ssfr_100',cpar_label=r'log(sSFR/yr$^{-1}$)',log_cpar=True,
		     color='gi',color_label='observed g-i (Vega magnitudes)',xlim=[-0.5,2.5])

	#### array of plots in model ML versus color
	outname = outfolder+'ml_vs_color_logzsol.png'
	plot(pdata,outname=outname,cpar='logzsol',cpar_label=r'log(Z/Z$_{\odot}$)')

	outname = outfolder+'ml_vs_color_ssfr.png'
	plot(pdata,outname=outname,cpar='ssfr_100',cpar_label=r'log(sSFR/yr$^{-1}$)',log_cpar=True)

	outname = outfolder+'ml_vs_color_thalf.png'
	plot(pdata,outname=outname,cpar='half_time',cpar_label=r'log(t$_{\mathrm{half}}$/Gyr)',log_cpar=True)

	outname = outfolder+'ml_vs_color_dust2.png'
	plot(pdata,outname=outname,cpar='dust2',cpar_label=r'$\tau_{\mathrm{diffuse}}$')

	outname = outfolder+'ml_vs_color_didx.png'
	plot(pdata,outname=outname,cpar='dust_index',cpar_label=r'attenuation curve shape')

def plot_obs(pdata,outname=None,cpar=None,cpar_label=None,cpar_range=None,log_cpar=False,
	         color='gr', color_label='observed g-r (Vega magnitudes)',xlim=None):

	### plot data
	xplot = pdata[color+'_color']
	xerr = pdata[color+'_color_err']
	yplot = pdata['ml_g']
	yerr = pdata['ml_g_err']
	yplot2 = pdata['ml_kobs']
	yerr2 = pdata['ml_kobs_err']

	### information
	fig, ax = plt.subplots(1,2,figsize=(13,5.5))
	alpha = 0.5
	mew=2.2
	ms=10
	color = '0.2'

	#### generate color mapping
	cpar_plot = np.array(pdata['model_parameters'][cpar]['q50'])
	if log_cpar:
		cpar_plot = np.log10(cpar_plot)
	if cpar_range is not None:
		cpar_plot = np.clip(cpar_plot,cpar_range[0],cpar_range[1])

	ax[0].errorbar(xplot,yplot,xerr=xerr,yerr=yerr,fmt='o',alpha=alpha,color=color,ms=0.0,mew=0.0,zorder=-5)
	pts = ax[0].scatter(xplot, yplot, marker='o', c=cpar_plot, cmap=plt.cm.jet,s=70,alpha=0.6)
	ax[0].set_ylim(-1.6,1.4)
	ax[0].set_xlim(xlim)
	ax[0].set_ylabel(r'log(M/L$_\mathrm{g-obs}$)')
	ax[0].set_xlabel(color_label)

	ax[1].errorbar(xplot,yplot2,xerr=xerr,yerr=yerr2,fmt='o',alpha=alpha,color=color,mew=0.0,ms=0.0,zorder=-5)
	pts = ax[1].scatter(xplot, yplot2, marker='o', c=cpar_plot, cmap=plt.cm.jet,s=70,alpha=0.6)
	ax[1].set_ylim(-1.6,0.5)
	ax[1].set_xlim(xlim)
	ax[1].set_xlabel(color_label)
	ax[1].set_ylabel(r'log(M/L$_\mathrm{K-obs}$)')

	#### add colorbar
	fig.subplots_adjust(right=0.84,left=0.1,wspace=0.26,bottom=0.12,top=0.95)
	cbar_ax = fig.add_axes([0.86, 0.12, 0.05, 0.76])
	cb = fig.colorbar(pts, cax=cbar_ax)
	cb.set_label(cpar_label)
	cb.solids.set_rasterized(True)
	cb.solids.set_edgecolor("face")

	#plt.tight_layout()
	plt.savefig(outname,dpi=150)
	os.system('open '+outname)
	plt.close()

def plot(pdata,outname=None,cpar=None,cpar_label=None,cpar_range=None,log_cpar=False):

	### plot data
	xplot = pdata['br_color']
	xerr = pdata['br_color_err']
	yplot = pdata['ml_b']
	yerr = pdata['ml_b_err']
	yplot2 = pdata['ml_k']
	yerr2 = pdata['ml_k_err']

	### information
	fig, ax = plt.subplots(1,2,figsize=(13,5.5))
	xlim = [0.2,1.7]
	alpha = 0.5
	mew=2.2
	ms=10
	color = '0.2'

	#### generate color mapping
	cpar_plot = np.array(pdata['model_parameters'][cpar]['q50'])
	if log_cpar:
		cpar_plot = np.log10(cpar_plot)
	if cpar_range is not None:
		cpar_plot = np.clip(cpar_plot,cpar_range[0],cpar_range[1])

	ax[0].errorbar(xplot,yplot,xerr=xerr,yerr=yerr,fmt='o',alpha=alpha,color=color,ms=0.0,mew=0.0,zorder=-5)
	pts = ax[0].scatter(xplot, yplot, marker='o', c=cpar_plot, cmap=plt.cm.jet,s=70,alpha=0.6)
	ax[0].set_ylim(-1.6,1.4)
	ax[0].set_xlim(xlim)
	ax[0].set_ylabel(r'log(M/L$_\mathrm{B}$)')
	ax[0].set_xlabel('model B-R (Vega magnitudes)')

	ax[1].errorbar(xplot,yplot2,xerr=xerr,yerr=yerr2,fmt='o',alpha=alpha,color=color,mew=0.0,ms=0.0,zorder=-5)
	pts = ax[1].scatter(xplot, yplot2, marker='o', c=cpar_plot, cmap=plt.cm.jet,s=70,alpha=0.6)
	ax[1].set_ylim(-1.6,0.5)
	ax[1].set_xlim(xlim)
	ax[1].set_xlabel('model B-R (Vega magnitudes)')
	ax[1].set_ylabel(r'log(M/L$_\mathrm{K}$)')

	#### add colorbar
	fig.subplots_adjust(right=0.84,left=0.1,wspace=0.26,bottom=0.12,top=0.95)
	cbar_ax = fig.add_axes([0.86, 0.12, 0.05, 0.76])
	cb = fig.colorbar(pts, cax=cbar_ax)
	cb.set_label(cpar_label)
	cb.solids.set_rasterized(True)
	cb.solids.set_edgecolor("face")

	#plt.tight_layout()
	plt.savefig(outname,dpi=150)
	os.system('open '+outname)
	plt.close()

'''
def plot(runname='brownseds_np'):

	filebase,params,ancilname=threed_dutils.generate_basenames(runname)
	ngals = len(filebase)

	nfail = 0

	# name output
	outname = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/pcomp/ml_vs_color.png'
	outname_2 = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/pcomp/twoml_vs_color.png'

	# filler arrays
	br_color = np.zeros(ngals)
	ml_b     = np.zeros(ngals)
	ml_k     = np.zeros(ngals)

	# calulate optical and K-band mass-to-light ratios for each galaxy
	sps = None
	for jj in xrange(ngals):

		try:
			sample_results, powell_results, model = threed_dutils.load_prospector_data(filebase[jj])
		except:
			print 1/0

		if sps is None:
			# generate SPS model
			from prospect.models import model_setup
			sps = model_setup.load_sps(**sample_results['run_params'])
			wave = sps.wavelengths

		# grab maximum probability spectrum at z = 0.0
		thetas = sample_results['bfit']['maxprob_params']
		model.params['zred'] = np.array([0.0])
		specmax,magsmax,_ = model.mean_model(thetas, sample_results['obs'], sps=sps)

		# calculate magnitudes
		# spectra are in AB maggies
		# need to be in Lsun / Hz
		specmax *= 3631 * 1e-23 / 3.846e33 * (4*np.pi*(10*3.08568e18)**2)
		bmag,blum=threed_dutils.integrate_mag(wave,specmax,'b_cosmos')
		rmag,rlum=threed_dutils.integrate_mag(wave,specmax,'r_cosmos')
		kmag,klum=threed_dutils.integrate_mag(wave,specmax,'k_cosmos')

		bmag_sun = 5.47
		kmag_sun = 3.33

		ab_to_vega_b = 0.09
		ab_to_vega_r = -0.21
		ab_to_vega_k = -1.9

		mass = 10**thetas[0]
		ml_b[jj] = mass / (10**((bmag_sun-(bmag+ab_to_vega_b))/2.5))
		ml_k[jj] = mass / (10**((kmag_sun-(kmag+ab_to_vega_k))/2.5))
		br_color[jj] = (bmag+ab_to_vega_b) - (rmag+ab_to_vega_r)

		print ml_b[jj], ml_k[jj]
	
	xlim = [0.2,1.6]
	alpha = 0.5
	mew=2.2
	ms=10
	color = '0.2'
	fig, ax = plt.subplots(2,1,figsize=(6,10))
	plt.subplots_adjust(hspace=0.0,top=0.95,bottom=0.05)
	ax[0].plot(br_color,np.log10(ml_b),'o',alpha=alpha,color=color,ms=ms,mew=mew)
	ax[0].set_ylim(-1.6,1.4)
	ax[0].set_xlim(xlim)
	ax[0].set_ylabel(r'log(M/L$_\mathrm{B}$)')
	ax[0].set_xticklabels([])

	ax[1].plot(br_color,np.log10(ml_k),'o',alpha=alpha,color=color,mew=mew,ms=ms)
	ax[1].set_ylim(-1.6,0.5)
	ax[1].set_xlim(xlim)
	ax[1].set_xlabel('B-R (mag)')
	ax[1].set_ylabel(r'log(M/L$_\mathrm{K}$)')

	plt.tight_layout()
	plt.savefig(outname,dpi=150)
	os.system('open '+outname)
	plt.close()
	print 1/0
'''