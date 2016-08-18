import threed_dutils
import numpy as np
import matplotlib.pyplot as plt
import os
import magphys_plot_pref
import pickle
from matplotlib.ticker import MaxNLocator
from corner import quantile
from prospect.models import model_setup
import sys
from extra_output import post_processing

minsfr = 1e-4
minssfr = 1e-15

obscolor = '#FF420E'
modcolor = '#375E97'

dpi = 120

def bdec_to_ext(bdec):
	return 2.5*np.log10(bdec/2.86)

def norm_resid(fit,truth):
	
	# define output
	out = np.zeros_like(truth)
	
	# define errors
	edown_fit = fit[:,0] - fit[:,2]
	eup_fit = fit[:,1] - fit[:,0]

	# find out which side of error bar to use
	undershot = truth > fit[:,0]
	
	# create output
	out[undershot] = (truth[undershot] - fit[undershot,0]) / eup_fit[undershot]
	out[~undershot] = (truth[~undershot] - fit[~undershot,0]) / edown_fit[~undershot]

	return out

def gaussian(x, mu, sig):
	'''
	can't believe there's not a package for this
	x is x-values, mu is mean, sig is sigma
	'''
	return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def make_plots(runname='nonparametric_mocks', recollate_data = False):

	outpickle = os.getenv('APPS')+'/threedhst_bsfh/data/'+runname+'_extradat.pickle'
	if not os.path.isfile(outpickle) or recollate_data == True:
		alldata = collate_data(runname=runname,outpickle=outpickle)
	else:
		with open(outpickle, "rb") as f:
			alldata=pickle.load(f)

	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/paper_plots/'
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)

	plot_fit_parameters(alldata,outfolder=outfolder,cdf=False)
	plot_derived_parameters(alldata,outfolder=outfolder,cdf=False)
	plot_spectral_parameters(alldata,outfolder=outfolder)
	plot_likelihood(alldata,outfolder=outfolder)
	#### PLOTS TO ADD
	# SFR_10 deviations versus Halpha deviations
	# max fit likelihood divided by truth likelihood
	# hdelta, halpha, hbeta absorption: what about wide / narrow indexes?

def pdf_distance_unitized(chain,truth,parnames,bins, truthnames=None):

	model_dict = {}
	obs_dict = {}
	for ii,p in enumerate(parnames):

		print p
		tempchain = chain[:,ii]

		# we're in extra_truths, do some sorting
		if truthnames != None:
			match = truthnames == parnames[ii]

			if match.sum() == 0:
				model_dict[p], obs_dict[p] = None, None
				continue

			temptruth = truth[match]
			# if truths are in log...
			if 'sfr' in parnames[ii]:
				tempchain = np.log10(chain[:,ii])
			if 'half_time' in parnames[ii]:
				temptruth = np.log10(truth[match])
				tempchain = np.log10(chain[:,ii])
		else:
			temptruth = truth[ii]


		#### chain properties
		model_chain_center = np.median(tempchain)
		bmin, bmax = bins[ii].min(),bins[ii].max()

		#### model PDF
		clipped_centered_chain = np.clip(tempchain-model_chain_center,bmin,bmax)
		model_dict[p],_ = np.histogram(clipped_centered_chain,bins=bins[ii],density=True)

		#### observed PDF
		clipped_truth = np.clip(temptruth-model_chain_center,bmin,bmax)
		obs_dict[p],_ = np.histogram(clipped_truth,bins=bins[ii],density=True)

	out = {}
	out['model_pdf'] = model_dict
	out['obs_pdf'] = obs_dict

	return out

def pdf_stats(bins,truth,model_pdf,bottom_lim,top_lim):

	### CALCULATE RANGE IN PREDICTIONS
	total = np.sum(model_pdf)
	cumsum = np.cumsum(model_pdf)
	lower_bin = np.interp(total*bottom_lim, cumsum, bins) # x_you_want, x_you_have, y_you_have
	upper_bin = np.interp(total*top_lim, cumsum, bins)

	### CALCULATE WHAT FRACTION OF TRUTHS FALL IN THIS RANGE
	total_truth = np.sum(truth)
	cumsum_truth = np.cumsum(truth)
	lower_tbin = np.interp(lower_bin, bins, cumsum_truth) # x_you_want, x_you_have, y_you_have
	upper_tbin = np.interp(upper_bin, bins, cumsum_truth)

	return (upper_tbin-lower_tbin)/total_truth

def pdf_distance(chain, truths, chainnames=None, truthnames=None):

	#### everything is properly ordered
	if (chainnames == None) & (truthnames == None):
		npars = chain.shape[1]
		nsamp = float(chain.shape[0])
		pdf_dist = np.zeros(npars)

		for i in xrange(npars): pdf_dist[i] = (chain[:,i] > truths[i]).sum()/nsamp
	#### we do the sorting ourselves
	else: 
		npars = len(truthnames) # assumes that everything in truths is also in the chain
		nsamp = float(chain.shape[0])
		pdf_dist = np.zeros(npars)

		for i in xrange(npars):
			match = chainnames == truthnames[i]
			# truths are in log...
			temptruths = truths[i]
			if 'sfr' in truthnames[i]:
				temptruths = 10**truths[i]
			pdf_dist[i] = (chain[:,match] > temptruths).sum()/nsamp

			print truthnames[i],pdf_dist[i]

	return pdf_dist

def collate_data(runname='ha_80myr',outpickle=None):

	filebase, parm_basename, ancilname=threed_dutils.generate_basenames(runname)

	out = []
	sps = None

	#['logmass', 'sfr_fraction_1', 'sfr_fraction_2', 'sfr_fraction_3',
    #   'sfr_fraction_4', 'sfr_fraction_5', 'dust2', 'logzsol',
    #   'dust_index', 'dust1', 'duste_qpah', 'duste_gamma', 'duste_umin']
	nbins = 31
	bins = [np.linspace(-0.3,0.3,nbins),
			np.linspace(-0.4,0.4,nbins),
			np.linspace(-0.4,0.4,nbins),
			np.linspace(-0.4,0.4,nbins),
			np.linspace(-0.4,0.4,nbins),
			np.linspace(-0.4,0.4,nbins),
			np.linspace(-0.3,0.3,nbins),
			np.linspace(-0.9,0.9,nbins),
			np.linspace(-1.0,1.0,nbins),
			np.linspace(-0.5,0.5,nbins),
			np.linspace(-1.5,1.5,nbins),
			np.linspace(-0.5,0.5,nbins),
			np.linspace(-7.0,7.0,nbins)]

	# ['half_time', 'sfr_10', 'sfr_100', 'sfr_1000', 'ssfr_10', 'ssfr_100',
    #   'totmass', 'emp_ha', 'bdec_cloudy', 'bdec_calc', 'total_ext5500']
	ebins = [np.linspace(-0.8,0.8,nbins), # half_time
			 np.linspace(-0.5,0.5,nbins),
			 np.linspace(-0.6,0.6,nbins), # sfr_100
			 np.linspace(-0.5,0.5,nbins),
			 np.linspace(-0.5,0.5,nbins),
			 np.linspace(-0.8,0.8,nbins), # ssfr_100
			 np.linspace(-0.5,0.5,nbins), 
			 np.linspace(-0.6,0.6,nbins), 
			 np.linspace(-1,1,nbins), 
			 np.linspace(-1,1,nbins),
			 np.linspace(-0.6,0.6,nbins)]

	for jj in xrange(len(filebase)):

		#### load sampler
		outdat = {}
		try:
			sample_results, powell_results, model = threed_dutils.load_prospector_data(filebase[jj])
		except:
			print 'failed to load number ' + str(int(jj))
			continue

		try:
			sample_results['quantiles']
		except:
			param_name = os.getenv('APPS')+'/threed'+sample_results['run_params']['param_file'].split('/threed')[1]
			post_processing(param_name, add_extra=True)
			sample_results, powell_results, model = threed_dutils.load_prospector_data(filebase[jj])

		if sps == None:
			sps = model_setup.load_sps(**sample_results['run_params'])

		### load truths
		truename = os.getenv('APPS')+'/threed'+sample_results['run_params']['truename'].split('/threed')[1]
		outdat['truths'] = threed_dutils.load_truths(os.getenv('APPS')+'/threed'+sample_results['run_params']['param_file'].split('/threed')[1],
			                                         sps=sps, calc_prob = True)

		### save all fit + derived parameters
		outdat['parnames'] = np.array(sample_results['quantiles']['parnames'])
		outdat['q50'] = sample_results['quantiles']['q50']
		outdat['q84'] = sample_results['quantiles']['q84']
		outdat['q16'] = sample_results['quantiles']['q16']
		outdat['pdf_dist'] = pdf_distance(sample_results['flatchain'],outdat['truths']['truths'])
		outdat['parameter_unit_pdfs'] = pdf_distance_unitized(sample_results['flatchain'], outdat['truths']['truths'],outdat['parnames'], bins)
		outdat['bins'] = bins

		outdat['eparnames'] = sample_results['extras']['parnames']
		outdat['eq50'] = sample_results['extras']['q50']
		outdat['eq84'] = sample_results['extras']['q84']
		outdat['eq16'] = sample_results['extras']['q16']
		outdat['epdf_dist'] = pdf_distance(sample_results['extras']['flatchain'],outdat['truths']['extra_truths'],
			                               chainnames=sample_results['extras']['parnames'], truthnames=outdat['truths']['extra_parnames'])
		outdat['eparameter_unit_pdfs'] = pdf_distance_unitized(sample_results['extras']['flatchain'], outdat['truths']['extra_truths'], outdat['eparnames'],ebins,
			    												truthnames=outdat['truths']['extra_parnames'])
		outdat['ebins'] = ebins

		### save spectral parameters
		outdat['eline_flux_q50'] = sample_results['model_emline']['flux']['q50']
		outdat['eline_flux_q84'] = sample_results['model_emline']['flux']['q84']
		outdat['eline_flux_q16'] = sample_results['model_emline']['flux']['q16']

		outdat['absline_eqw_q50'] = sample_results['spec_info']['eqw']['q50']
		outdat['absline_eqw_q84'] = sample_results['spec_info']['eqw']['q84']
		outdat['absline_eqw_q16'] = sample_results['spec_info']['eqw']['q16']

		outdat['dn4000_q50'] = sample_results['spec_info']['dn4000']['q50']
		outdat['dn4000_q84'] = sample_results['spec_info']['dn4000']['q84']
		outdat['dn4000_q16'] = sample_results['spec_info']['dn4000']['q16']

		### halpha extinction, dust1 / dust2
		flatchain = sample_results['flatchain']
		ransamp = flatchain[np.random.choice(flatchain.shape[0], 4000, replace=False),:]
		d1 = ransamp[:,outdat['parnames']=='dust1']
		d2 = ransamp[:,outdat['parnames']=='dust2']
		didx = ransamp[:,outdat['parnames']=='dust_index']
		ha_ext = threed_dutils.charlot_and_fall_extinction(6563.0, d1, d2, -1.0, didx, kriek=True)
		outdat['ha_ext_q16'], outdat['ha_ext_q50'], outdat['ha_ext_q84'] = quantile(ha_ext, [0.16, 0.5, 0.84])
		outdat['d1_d2_q16'], outdat['d1_d2_q50'], outdat['d1_d2_q84'] = quantile(d1/d2, [0.16, 0.5, 0.84])

		### max probability
		outdat['maxprob'] = sample_results['bfit']['maxprob']

		### add in d1_d2, ha_ext
		d1_t = outdat['truths']['truths'][outdat['parnames']=='dust1']
		d2_t = outdat['truths']['truths'][outdat['parnames']=='dust2']
		didx_t = outdat['truths']['truths'][outdat['parnames']=='dust_index']
		outdat['truths']['ha_ext'] = threed_dutils.charlot_and_fall_extinction(6563.0, d1_t, d2_t, -1.0, didx_t, kriek=True)[0]
		outdat['truths']['d1_d2'] = d1_t/d2_t

		out.append(outdat)
	
	pickle.dump(out,open(outpickle, "wb"))

def plot_fit_parameters(alldata,outfolder=None, cdf=False):

	#### check parameter space
	pars = alldata[0]['parnames']
	if len(pars) > 12:
		xfig, yfig = 5,3
		size = (14,20)
		parlabels = [r'log(M/M$_{\odot}$)', 'SFH 0-100 Myr', 'SFH 100-300 Myr', 'SFH 300 Myr-1 Gyr', 
		         'SFH 1-3 Gyr', 'SFH 3-6 Gyr', 'diffuse dust', r'log(Z/Z$_{\odot}$)', 'diffuse dust index',
		         'birth-cloud dust', r'dust emission Q$_{\mathrm{PAH}}$',r'dust emission $\gamma$',r'dust emission U$_{\mathrm{min}}$']
	else:
		xfig, yfig = 3,4
		size = (20,14)
		parlabels = [r'log(M/M$_{\odot}$)', 'SFH 0-100 Myr', 'SFH 100-300 Myr', 'SFH 300 Myr-1 Gyr', 
		         'SFH 1-3 Gyr', 'diffuse dust', r'log(Z/Z$_{\odot}$)', 'diffuse dust index',
		         'birth-cloud dust', r'dust emission Q$_{\mathrm{PAH}}$',r'dust emission $\gamma$',r'dust emission U$_{\mathrm{min}}$']

	#### REGULAR PARAMETERS
	fig, axes = plt.subplots(xfig, yfig, figsize = size)
	plt.subplots_adjust(wspace=0.3,hspace=0.3)
	ax = np.ravel(axes)

	### DISTRIBUTION OF ERRORS
	fig_err, axes_err = plt.subplots(xfig, yfig, figsize = (11,19.5))
	plt.subplots_adjust(wspace=0.2,hspace=0.4)
	ax_err = np.ravel(axes_err)

	for ii,par in enumerate(pars):

		#### fit parameter
		y = np.array([dat['q50'][ii] for dat in alldata])
		yup = np.array([dat['q84'][ii] for dat in alldata])
		ydown = np.array([dat['q16'][ii] for dat in alldata])
		yerr = threed_dutils.asym_errors(y,yup,ydown,log=False)

		#### truth
		x = np.array([dat['truths']['truths'][ii] for dat in alldata])

		#### plot
		ax[ii].errorbar(x,y,yerr,fmt='o',alpha=0.8,color='#1C86EE')
		ax[ii].set_xlabel('true '+parlabels[ii])
		ax[ii].set_ylabel('fit '+parlabels[ii])

		ax[ii] = threed_dutils.equalize_axes(ax[ii], x,y)
		mean_offset,scat = threed_dutils.offset_and_scatter(x,y)
		ax[ii].text(0.96,0.12, 'scatter='+"{:.2f}".format(scat),transform = ax[ii].transAxes,ha='right')
		ax[ii].text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset), transform = ax[ii].transAxes,ha='right')

		ax[ii].xaxis.set_major_locator(MaxNLocator(5))
		ax[ii].yaxis.set_major_locator(MaxNLocator(5))

		##### gather the PDF
		pdf_dist = np.array([dat['pdf_dist'][ii] for dat in alldata])

		if cdf:
			##### plot histogram
			nbins_hist = 25

			n, bins, patches = ax_err[ii].hist(pdf_dist, range=(0.0,1.0),
		                 			           bins=nbins_hist, histtype='step',
		                 			           alpha=0.7,lw=2,color=obscolor,
		                 			           cumulative=True,normed=True)

			ax_err[ii].plot([0,1],[0,1],color=truecolor,lw=2,alpha=0.5)

			ax_err[ii].text(0.05,0.9,'ideal distribution',transform=ax_err[ii].transAxes,color=modcolor,ha='left',fontsize=12,weight='bold')
			ax_err[ii].text(0.05,0.82,'mock distribution',transform=ax_err[ii].transAxes,color=obscolor,ha='left',fontsize=12,weight='bold')

			ax_err[ii].set_xlabel(r'location of truth within PDF')
			ax_err[ii].set_ylabel('cumulative density')
			ax_err[ii].set_title(parlabels[ii])
			ax_err[ii].set_ylim(0,1)
		else:
			##### gather the PDF
			bins = alldata[0]['bins'][ii]
			nbins = len(bins)
			model_dist, obs_dist = np.zeros(nbins-1),np.zeros(nbins-1)
			for kk, dat in enumerate(alldata):
				model_dist += dat['parameter_unit_pdfs']['model_pdf'][par]
				obs_dist += dat['parameter_unit_pdfs']['obs_pdf'][par]

			#### create step function
			plotx = np.empty((nbins*2,), dtype=float)
			plotx[0::2] = bins
			plotx[1::2] = bins

			ploty_mod, ploty_obs = [np.empty((model_dist.size*2,), dtype=model_dist.dtype) for X in xrange(2)]
			ploty_mod[0::2] = model_dist
			ploty_mod[1::2] = model_dist
			ploty_mod = np.concatenate((np.atleast_1d(0.0),ploty_mod,np.atleast_1d(0.0)))
			ploty_obs[0::2] = obs_dist
			ploty_obs[1::2] = obs_dist
			ploty_obs = np.concatenate((np.atleast_1d(0.0),ploty_obs,np.atleast_1d(0.0)))

			ax_err[ii].plot(plotx,ploty_obs,alpha=0.8,lw=2,color=obscolor)
			ax_err[ii].plot(plotx,ploty_mod,alpha=0.8,lw=2,color=modcolor)

			ax_err[ii].fill_between(plotx, np.zeros_like(ploty_obs), ploty_obs, 
							   color=obscolor,
							   alpha=0.3)
			ax_err[ii].fill_between(plotx, np.zeros_like(ploty_mod), ploty_mod, 
							   color=modcolor,
							   alpha=0.3)

			xunit = ''
			if 'log' in parlabels[ii]:
				xunit = ' [dex]'

			ax_err[ii].set_xlabel(r'$\Delta$(prediction)'+xunit)
			for tl in ax_err[ii].get_yticklabels():tl.set_visible(False) # no tick labels
			if ii % 3 == 0: # only label every third y-axis
				ax_err[ii].set_ylabel('density')
			ax_err[ii].xaxis.set_major_locator(MaxNLocator(4)) # only label up to x tickmarks
			for tick in ax_err[ii].xaxis.get_major_ticks(): tick.label.set_fontsize(12) 
			ax_err[ii].set_title(parlabels[ii])
			ax_err[ii].set_ylim(0.0,ax_err[ii].get_ylim()[1]*1.125)
			ax_err[ii].set_xlim(bins.min(),bins.max())

			fs = 12
			ax_err[ii].text(0.06,0.88,r'$\Sigma$ (model posteriors)',transform=ax_err[ii].transAxes,color=modcolor,fontsize=fs)
			ax_err[ii].text(0.06,0.8,'truth',transform=ax_err[ii].transAxes,color=obscolor,fontsize=fs)

			### from the summed histogram
			#onesig_perc = pdf_stats((bins[1:] + bins[:-1])/2.,obs_dist,model_dist,0.16,0.84)/0.68
			#twosig_perc = pdf_stats((bins[1:] + bins[:-1])/2.,obs_dist,model_dist,0.025,0.975)/0.95

			### from individual measurements
			onesig_perc = ((pdf_dist >= 0.16) & (pdf_dist <= 0.84)).sum()/float(pdf_dist.shape[0]) / 0.68
			twosig_perc = ((pdf_dist >= 0.025) & (pdf_dist <= 0.975)).sum()/float(pdf_dist.shape[0]) / 0.95

			ax_err[ii].text(0.06,0.7,r'$\frac{1\sigma_{\mathrm{mock}}}{1\sigma_{\mathrm{true}}}$:'+"{:.2f}".format(onesig_perc),transform=ax_err[ii].transAxes,fontsize=fs)
			ax_err[ii].text(0.06,0.58,r'$\frac{2\sigma_{\mathrm{mock}}}{2\sigma_{\mathrm{true}}}$:'+"{:.2f}".format(twosig_perc),transform=ax_err[ii].transAxes,fontsize=fs)
			
	# turn the remaining axes off
	for i in xrange(ii+1,ax.shape[0]):
		ax[i].axis('off')
		ax_err[i].axis('off')

	fig.tight_layout()
	fig.savefig(outfolder+'fit_parameter_recovery.png',dpi=dpi)
	fig_err.savefig(outfolder+'fit_parameter_PDF.png',dpi=dpi)
	plt.close()

def plot_derived_parameters(alldata,outfolder=None, cdf=False):

	##### EXTRA PARAMETERS
	epars = alldata[0]['eparnames']
	epars_truth = alldata[0]['truths']['extra_parnames']
	pars_to_plot = ['sfr_100','ssfr_100', 'half_time']#,'ssfr_10','ssfr_100','half_time']
	parlabels = ['log(SFR) [100 Myr]','log(sSFR) [100 Myr]', r"log(t$_{\mathrm{half-mass}})$ [Gyr]"]

	fig, axes = plt.subplots(1, 3, figsize = (15,5))
	plt.subplots_adjust(wspace=0.33,bottom=0.15,top=0.85,left=0.1,right=0.93)

	### DISTRIBUTION OF ERRORS
	fig_err, axes_err = plt.subplots(1, 3, figsize = (12,4))
	plt.subplots_adjust(wspace=0.2,bottom=0.15,top=0.85,left=0.05,right=0.93)

	ax = np.ravel(axes)
	ax_err = np.ravel(axes_err)
	for ii,par in enumerate(pars_to_plot):

		#### derived parameter
		idx = epars == par
		y = np.squeeze([dat['eq50'][idx] for dat in alldata])
		yup = np.squeeze([dat['eq84'][idx] for dat in alldata])
		ydown = np.squeeze([dat['eq16'][idx] for dat in alldata])

		if 'sfr' in par and 'ssfr' not in par:
			y = np.clip(y,minsfr,np.inf)
			yup = np.clip(yup,minsfr,np.inf)
			ydown = np.clip(ydown,minsfr,np.inf)
		if 'ssfr' in par:
			y = np.clip(y,minssfr,np.inf)
			yup = np.clip(yup,minssfr,np.inf)
			ydown = np.clip(ydown,minssfr,np.inf)

		#### truth
		idx_true = epars_truth == par
		x = np.squeeze([dat['truths']['extra_truths'][idx_true] for dat in alldata])

		### log all parameters
		yerr = threed_dutils.asym_errors(y,yup,ydown,log=True)
		yup = np.log10(yup)
		ydown = np.log10(ydown)
		y = np.log10(y)

		if par == 'half_time':
			#yup = 10**yup
			#ydown = 10**ydown
			#y = 10**y
			#yerr = threed_dutils.asym_errors(y,yup,ydown,log=False)
			x = np.log10(x)

		### plot that shit
		ax[ii].errorbar(x,y,yerr,fmt='o',alpha=0.8,color='#1C86EE')
		ax[ii].set_xlabel('true '+parlabels[ii])
		ax[ii].set_ylabel('fit '+parlabels[ii])

		ax[ii] = threed_dutils.equalize_axes(ax[ii], x, y)
		mean_offset,scat = threed_dutils.offset_and_scatter(x,y)
		ax[ii].text(0.96,0.12, 'scatter='+"{:.2f}".format(scat)+' dex',transform = ax[ii].transAxes,ha='right')
		ax[ii].text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset)+' dex', transform = ax[ii].transAxes,ha='right')

		ax[ii].xaxis.set_major_locator(MaxNLocator(5))
		ax[ii].yaxis.set_major_locator(MaxNLocator(5))

		##### gather the PDF
		pdf_dist = np.array([dat['epdf_dist'][idx] for dat in alldata])

		if cdf:

			##### plot histogram
			nbins_hist = 25
			histcolor = '#0000CD'
			truecolor = '#FF420E'

			n, bins, patches = ax_err[ii].hist(pdf_dist, range=(0.0,1.0),
		                 			           bins=nbins_hist, histtype='step',
		                 			           alpha=0.7,lw=2,color=histcolor,
		                 			           cumulative=True,normed=True)

			ax_err[ii].plot([0,1],[0,1],color=truecolor,lw=2,alpha=0.5)

			ax_err[ii].text(0.05,0.9,'ideal distribution',transform=ax_err[ii].transAxes,color=truecolor,ha='left',fontsize=12,weight='bold')
			ax_err[ii].text(0.05,0.82,'mock distribution',transform=ax_err[ii].transAxes,color=histcolor,ha='left',fontsize=12,weight='bold')

			ax_err[ii].set_xlabel(r'location of truth within PDF')
			ax_err[ii].set_ylabel('cumulative density')
			ax_err[ii].set_title(parlabels[ii])
			ax_err[ii].set_ylim(0,1)

		else:
			bins = np.array(alldata[0]['ebins'])[idx][0]
			nbins = bins.shape[0]
			model_dist, obs_dist = np.zeros(nbins-1),np.zeros(nbins-1)
			for kk, dat in enumerate(alldata):
				model_dist += dat['eparameter_unit_pdfs']['model_pdf'][par]
				obs_dist += dat['eparameter_unit_pdfs']['obs_pdf'][par]

			#### create step function
			plotx = np.empty((nbins*2,), dtype=float)
			plotx[0::2] = bins
			plotx[1::2] = bins

			ploty_mod, ploty_obs = [np.empty((model_dist.size*2,), dtype=model_dist.dtype) for X in xrange(2)]
			ploty_mod[0::2] = model_dist
			ploty_mod[1::2] = model_dist
			ploty_mod = np.concatenate((np.atleast_1d(0.0),ploty_mod,np.atleast_1d(0.0)))
			ploty_obs[0::2] = obs_dist
			ploty_obs[1::2] = obs_dist
			ploty_obs = np.concatenate((np.atleast_1d(0.0),ploty_obs,np.atleast_1d(0.0)))

			ax_err[ii].plot(plotx,ploty_obs,alpha=0.8,lw=2,color=obscolor)
			ax_err[ii].plot(plotx,ploty_mod,alpha=0.8,lw=2,color=modcolor)

			ax_err[ii].fill_between(plotx, np.zeros_like(ploty_obs), ploty_obs, 
							   color=obscolor,
							   alpha=0.3)
			ax_err[ii].fill_between(plotx, np.zeros_like(ploty_mod), ploty_mod, 
							   color=modcolor,
							   alpha=0.3)

			xunit = ''
			if 'log' in parlabels[ii]:
				xunit = ' [dex]'


			ax_err[ii].set_xlabel(r'$\Delta$(prediction)'+xunit)
			for tl in ax_err[ii].get_yticklabels():tl.set_visible(False) # no tick labels
			ax_err[ii].set_ylabel('density')
			ax_err[ii].set_title(parlabels[ii])
			ax_err[ii].xaxis.set_major_locator(MaxNLocator(5)) # only label up to x tickmarks
			ax_err[ii].set_ylim(0.0,ax_err[ii].get_ylim()[1]*1.125)
			ax_err[ii].set_xlim(bins.min(),bins.max())

			fs = 16
			for tick in ax_err[ii].xaxis.get_major_ticks(): tick.label.set_fontsize(fs) 
			
			ax_err[ii].text(0.06,0.88,r'$\Sigma$ (model posteriors)',transform=ax_err[ii].transAxes,color=modcolor,fontsize=fs)
			ax_err[ii].text(0.06,0.8,'truth',transform=ax_err[ii].transAxes,color=obscolor,fontsize=fs)

			### from the summed histogram
			# onesig_perc = pdf_stats((bins[1:] + bins[:-1])/2.,obs_dist,model_dist,0.16,0.84)/0.68
			# twosig_perc = pdf_stats((bins[1:] + bins[:-1])/2.,obs_dist,model_dist,0.025,0.975)/0.95

			onesig_perc = ((pdf_dist >= 0.16) & (pdf_dist <= 0.84)).sum()/float(pdf_dist.shape[0]) / 0.68
			twosig_perc = ((pdf_dist >= 0.025) & (pdf_dist <= 0.975)).sum()/float(pdf_dist.shape[0]) / 0.95


			ax_err[ii].text(0.06,0.7,r'$\frac{1\sigma_{\mathrm{mock}}}{1\sigma_{\mathrm{true}}}$:'+"{:.2f}".format(onesig_perc),transform=ax_err[ii].transAxes,fontsize=fs)
			ax_err[ii].text(0.06,0.58,r'$\frac{2\sigma_{\mathrm{mock}}}{2\sigma_{\mathrm{true}}}$:'+"{:.2f}".format(twosig_perc),transform=ax_err[ii].transAxes,fontsize=fs)
			
	fig.tight_layout()
	fig.savefig(outfolder+'derived_parameter_recovery.png',dpi=dpi)
	fig_err.savefig(outfolder+'derived_parameter_PDF.png',dpi=dpi)

	plt.close()

def plot_spectral_parameters(alldata,outfolder=None):

	##### OBSERVABLES
	# halpha + hbeta + dn4000 + balmer decrement
	# add hdelta absorption? which index? both?
	fig, axes = plt.subplots(2, 2, figsize = (12,12))
	ax = np.ravel(axes)
	to_plot = ['Halpha','Hbeta']
	plot_names = [r'log(H$_{\alpha}$ flux)',r'log(H$_{\beta}$ flux)']
	emnames = alldata[0]['truths']['emnames']
	for ii,par in enumerate(to_plot):

		##### predicted
		idx = emnames == par
		y = np.array([dat['eline_flux_q50'][idx] for dat in alldata])
		yup = np.array([dat['eline_flux_q84'][idx] for dat in alldata])
		ydown = np.array([dat['eline_flux_q16'][idx] for dat in alldata])

		##### true
		x = np.array([dat['truths']['emflux'][idx] for dat in alldata])

		##### clip
		minflux = np.min(ydown[ydown != 0])
		y = np.clip(y,minflux,np.inf)
		ydown = np.clip(ydown,minflux,np.inf)
		yup = np.clip(yup,minflux,np.inf)
		x = np.clip(x,minflux,np.inf)

		#### errors + logify
		yerr = threed_dutils.asym_errors(y,yup,ydown,log=True)
		y = np.log10(y)
		x = np.log10(x)

		#### plot
		ax[ii].errorbar(x,y,yerr,fmt='o',alpha=0.8,color='#1C86EE')
		ax[ii].set_xlabel('true '+plot_names[ii])
		ax[ii].set_ylabel('fit '+plot_names[ii])

		ax[ii] = threed_dutils.equalize_axes(ax[ii], x, y)
		mean_offset,scat = threed_dutils.offset_and_scatter(x,y)
		ax[ii].text(0.96,0.12, 'scatter='+"{:.2f}".format(scat),transform = ax[ii].transAxes,ha='right')
		ax[ii].text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset), transform = ax[ii].transAxes,ha='right')

		ax[ii].xaxis.set_major_locator(MaxNLocator(5))
		ax[ii].yaxis.set_major_locator(MaxNLocator(5))

	#### dn4000
	y = np.array([dat['dn4000_q50'] for dat in alldata])
	yup = np.array([dat['dn4000_q84'] for dat in alldata])
	ydown = np.array([dat['dn4000_q16'] for dat in alldata])
	yerr = threed_dutils.asym_errors(y,yup,ydown,log=False)
	x = np.array([dat['truths']['dn4000'] for dat in alldata])

	ax[ii+1].errorbar(x,y,yerr,fmt='o',alpha=0.8,color='#1C86EE')
	ax[ii+1].set_xlabel(r'true D$_n$4000')
	ax[ii+1].set_ylabel(r'fit D$_n$4000')

	ax[ii+1] = threed_dutils.equalize_axes(ax[ii+1], x, y)
	mean_offset,scat = threed_dutils.offset_and_scatter(x,y)
	ax[ii+1].text(0.96,0.12, 'scatter='+"{:.2f}".format(scat),transform = ax[ii+1].transAxes,ha='right')
	ax[ii+1].text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset), transform = ax[ii+1].transAxes,ha='right')

	ax[ii+1].xaxis.set_major_locator(MaxNLocator(5))
	ax[ii+1].yaxis.set_major_locator(MaxNLocator(5))

	#### balmer decrement
	idx = alldata[0]['eparnames'] == 'bdec_cloudy'
	y = np.array([dat['eq50'][idx] for dat in alldata])
	ha = np.array([dat['truths']['emflux'][emnames=='Halpha'] for dat in alldata])
	hb = np.array([dat['truths']['emflux'][emnames=='Hbeta'] for dat in alldata])
	x = ha/hb

	# we remove galaxies where Halpha flux = 0
	# could replace with the calculated bdec, but whatever
	# that has a small offset relative to cloudy bdec anyway
	# ALSO remove true galaxies where balmer decrement is less than 2.8
	# not sure what's going on here? limit of cloudy?
	good = (~np.isnan(y)) & (x > 2.8)
	x = bdec_to_ext(x[good])
	y = bdec_to_ext(y[good])
	yup = bdec_to_ext(np.array([dat['eq84'][idx] for dat in alldata])[good])
	ydown = bdec_to_ext(np.array([dat['eq16'][idx] for dat in alldata])[good])
	yerr = threed_dutils.asym_errors(y,yup,ydown,log=False)

	ax[ii+2].errorbar(x,y,yerr,fmt='o',alpha=0.8,color='#1C86EE')
	ax[ii+2].set_xlabel(r'true A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$ [magnitudes]')
	ax[ii+2].set_ylabel(r'fit A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$ [magnitudes]')

	ax[ii+2] = threed_dutils.equalize_axes(ax[ii+2], x, y)
	mean_offset,scat = threed_dutils.offset_and_scatter(x,y)
	ax[ii+2].text(0.96,0.12, 'scatter='+"{:.2f}".format(scat)+' mags',transform = ax[ii+2].transAxes,ha='right')
	ax[ii+2].text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset)+' mags', transform = ax[ii+2].transAxes,ha='right')

	ax[ii+2].xaxis.set_major_locator(MaxNLocator(5))
	ax[ii+2].yaxis.set_major_locator(MaxNLocator(5))

	plt.savefig(outfolder+'spectral_parameter_recovery.png',dpi=dpi)
	plt.close()

def plot_mass_resid(alldata,outfolder=None):

	#### true slope + true mass
	pars = np.array(alldata[0]['parnames'])
	idx_mass = pars == 'logmass'
	true_mass =  np.array([dat['truths']['truths'][idx_mass] for dat in alldata])

	#### ratio with fit stellar mass
	ratio = true_mass-[dat['q50'][idx_mass] for dat in alldata]
	yup = true_mass - [dat['q84'][idx_mass] for dat in alldata]
	ydown = true_mass - [dat['q16'][idx_mass] for dat in alldata]
	yerr = threed_dutils.asym_errors(ratio,yup,ydown,log=False)

	#### true half-mass
	epars_truth = alldata[0]['truths']['extra_parnames']
	idx_true = epars_truth == 'half_time'
	true_halfmass = np.log10(np.squeeze([dat['truths']['extra_truths'][idx_true] for dat in alldata]))

	#### fit half-mass
	epars = alldata[0]['eparnames']
	idx = epars == 'half_time'

	halfmass_ratio = true_halfmass-np.log10(np.squeeze([dat['eq50'][idx] for dat in alldata]))
	yup = true_halfmass-np.log10(np.squeeze([dat['eq84'][idx] for dat in alldata]))
	ydown = true_halfmass-np.log10(np.squeeze([dat['eq16'][idx] for dat in alldata]))
	halfmass_ratio_err = threed_dutils.asym_errors(halfmass_ratio,yup,ydown,log=False)

	#### make plot
	fig, ax = plt.subplots(1, 1, figsize = (7,7))
	ax.errorbar(halfmass_ratio,ratio,yerr=yerr,xerr=halfmass_ratio_err,fmt='o',alpha=0.8,color='#1C86EE')
	ax.set_xlabel(r'log(t$_{\mathrm{half,true}}$/t$_{\mathrm{half,fit}}$)')
	ax.set_ylabel('log(M$_{\mathrm{true}}$/M$_{\mathrm{fit}}$)')

	ax.axhline(0.0, ls="dashed", color='k',linewidth=1.5)
	ax.axvline(0.0, ls="dashed", color='k',linewidth=1.5)

	ax.set_xlim(-1.2,1.2)
	ax.set_ylim(-0.5,0.5)

	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(5))

	plt.savefig(outfolder+'mass_residual.png',dpi=dpi)
	plt.close()

def plot_likelihood(alldata,outfolder=None):


	likelihood_fit = np.array([dat['maxprob'] for dat in alldata])
	likelihood_true = np.array([dat['truths']['truthprob'] for dat in alldata])

	fig, ax = plt.subplots(1, 1, figsize = (7,7))
	nbins = 20
	range = (-20,20)
	hplot = np.clip(likelihood_true-likelihood_fit,range[0],range[1])
	nn, b, p = ax.hist(hplot, nbins, histtype='bar', color='#1C86EE', alpha=0.6,lw=2,edgecolor='grey',range=range)
	ax.set_xlabel('true ln(prob) - best-fit ln(prob)')
	ax.set_ylabel('N')

	plt.savefig(outfolder+'likelihood_ratio.png',dpi=dpi)
	plt.close()

if __name__ == "__main__":
   sys.exit(make_plots())





