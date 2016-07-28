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

minsfr = 1e-4
minssfr = 1e-15

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

	plot_fit_parameters(alldata,outfolder=outfolder)
	plot_derived_parameters(alldata,outfolder=outfolder)
	plot_spectral_parameters(alldata,outfolder=outfolder)
	#plot_sfr_resid(alldata,outfolder=outfolder)
	plot_mass_resid(alldata,outfolder=outfolder)
	plot_likelihood(alldata,outfolder=outfolder)
	#### PLOTS TO ADD
	# SFR_10 deviations versus Halpha deviations
	# max fit likelihood divided by truth likelihood
	# hdelta, halpha, hbeta absorption: what about wide / narrow indexes?
	# SFR_10 (truth) versus SFR_100 (truth) [there better be differences, goddamnit]


def calc_onesig(fit_pars,true_pars):

	'''
	(1) calculate normalized residual distribution
	(2) calculate 16th, 50th, and 84th percentiles of this distribution
	(3) return average(84th-50th, 50th-16th) as 1 sigma, and 50th as mean
	'''

	residual_distribution = norm_resid(fit_pars,true_pars)
	residual_percentiles = quantile(residual_distribution,[0.16,0.5,0.84])

	mean = residual_percentiles[1]
	onesig = (residual_percentiles[2] - residual_percentiles[0])/2.

	return residual_distribution, onesig, mean


def collate_data(runname='ha_80myr',outpickle=None):

	filebase, parm_basename, ancilname=threed_dutils.generate_basenames(runname)

	out = []
	sps = None
	for jj in xrange(len(filebase)):

		outdat = {}
		try:
			sample_results, powell_results, model = threed_dutils.load_prospector_data(filebase[jj])
		except:
			print 'failed to load number ' + str(int(jj))
			continue

		if sps == None:
			sps = model_setup.load_sps(**sample_results['run_params'])

		### save all fit + derived parameters
		outdat['parnames'] = np.array(sample_results['quantiles']['parnames'])
		outdat['q50'] = sample_results['quantiles']['q50']
		outdat['q84'] = sample_results['quantiles']['q84']
		outdat['q16'] = sample_results['quantiles']['q16']

		outdat['eparnames'] = sample_results['extras']['parnames']
		outdat['eq50'] = sample_results['extras']['q50']
		outdat['eq84'] = sample_results['extras']['q84']
		outdat['eq16'] = sample_results['extras']['q16']

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

		### load truths, calculate details
		truename = os.getenv('APPS')+'/threed'+sample_results['run_params']['truename'].split('/threed')[1]
		outdat['truths'] = threed_dutils.load_truths(os.getenv('APPS')+'/threed'+sample_results['run_params']['param_file'].split('/threed')[1],
			                                         sps=sps, calc_prob = True)

		### add in d1_d2, ha_ext
		d1_t = outdat['truths']['truths'][outdat['parnames']=='dust1']
		d2_t = outdat['truths']['truths'][outdat['parnames']=='dust2']
		didx_t = outdat['truths']['truths'][outdat['parnames']=='dust_index']
		outdat['truths']['ha_ext'] = threed_dutils.charlot_and_fall_extinction(6563.0, d1_t, d2_t, -1.0, didx_t, kriek=True)[0]
		outdat['truths']['d1_d2'] = d1_t/d2_t

		out.append(outdat)
		
	pickle.dump(out,open(outpickle, "wb"))

def plot_fit_parameters(alldata,outfolder=None):

	#### check parameter space
	pars = alldata[0]['parnames']
	if len(pars) > 12:
		xfig, yfig = 4,4
		size = (20,20)
	else:
		xfig, yfig = 3,4
		size = (20,14)

	#### REGULAR PARAMETERS
	fig, axes = plt.subplots(xfig, yfig, figsize = size)
	plt.subplots_adjust(wspace=0.3,hspace=0.3)
	ax = np.ravel(axes)

	### DISTRIBUTION OF ERRORS
	fig_err, axes_err = plt.subplots(xfig, yfig, figsize = size)
	plt.subplots_adjust(wspace=0.3,hspace=0.3)
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
		ax[ii].set_xlabel('true '+par)
		ax[ii].set_ylabel('fit '+par)

		ax[ii] = threed_dutils.equalize_axes(ax[ii], x,y)
		mean_offset,scat = threed_dutils.offset_and_scatter(x,y)
		ax[ii].text(0.96,0.12, 'scatter='+"{:.2f}".format(scat),transform = ax[ii].transAxes,ha='right')
		ax[ii].text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset), transform = ax[ii].transAxes,ha='right')

		ax[ii].xaxis.set_major_locator(MaxNLocator(5))
		ax[ii].yaxis.set_major_locator(MaxNLocator(5))

		##### distribution of errors
		fit_pars = np.transpose(np.vstack((y,yup,ydown)))
		residual_distribution, onesig, median = calc_onesig(fit_pars,x)

		# plot histogram, overplot gaussian 1sig, write onesig and mean
		nbins_hist = 25
		histcolor = '#0000CD'
		gausscolor = '#FF0000'

		if np.max(np.abs(residual_distribution)) > 20:
			range = (-20,20)
		else:
			range = None

		n, bins, patches = ax_err[ii].hist(residual_distribution,
	                 			           nbins_hist, histtype='bar',
	                 			           alpha=0.9,lw=2,color=histcolor,
	                 			           range=range)

		### Need to multiply with Gaussian amplitude A such that AREA = NPOINTS
		# AREA = AMPLITUDE * SIGMA * (2*pi)**0.5
		gnorm = residual_distribution.shape[0]/(onesig*np.sqrt(2*np.pi))*(bins[1]-bins[0])
		xplot = np.linspace(np.min(bins),np.max(bins),1e4)
		plot_gauss = gaussian(xplot,median,onesig)*gnorm

		ax_err[ii].plot(xplot,plot_gauss,color=gausscolor,lw=3)

		ax_err[ii].text(0.95,0.9,'median='+"{:.2f}".format(median),transform=ax_err[ii].transAxes,color=gausscolor,ha='right')
		ax_err[ii].text(0.95,0.8,r'1$\sigma$='+"{:.2f}".format(onesig),transform=ax_err[ii].transAxes,color=gausscolor,ha='right')

		ax_err[ii].set_xlabel(r'(true-fit)/1$\sigma$ error')
		ax_err[ii].set_ylabel('N')
		ax_err[ii].set_title(par)


	fig.savefig(outfolder+'fit_parameter_recovery.png',dpi=150)
	fig_err.savefig(outfolder+'residual_fitpars.png',dpi=150)
	plt.close()

def plot_derived_parameters(alldata,outfolder=None):

	##### EXTRA PARAMETERS
	epars = alldata[0]['eparnames']
	epars_truth = alldata[0]['truths']['extra_parnames']
	pars_to_plot = ['sfr_100','ssfr_100', 'half_time']#,'ssfr_10','ssfr_100','half_time']
	parlabels = ['log(SFR) [100 Myr]','log(sSFR) [100 Myr]', r"t$_{\mathrm{half-mass}}$ [Gyr]"]

	fig, axes = plt.subplots(1, 3, figsize = (15,5))
	plt.subplots_adjust(wspace=0.33,bottom=0.15,top=0.85,left=0.1,right=0.93)

	### DISTRIBUTION OF ERRORS
	fig_err, axes_err = plt.subplots(1, 3, figsize = (15,5))
	plt.subplots_adjust(wspace=0.33,bottom=0.15,top=0.85,left=0.05,right=0.93)

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
			yup = 10**yup
			ydown = 10**ydown
			y = 10**y
			yerr = threed_dutils.asym_errors(y,yup,ydown,log=False)
		'''
			x = np.log10(x)
		'''

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

		##### distribution of errors
		fit_pars = np.transpose(np.vstack((y,yup,ydown)))
		residual_distribution, onesig, median = calc_onesig(fit_pars,x)

		# plot histogram, overplot gaussian 1sig, write onesig and mean
		nbins_hist = 25
		histcolor = '#0000CD'
		gausscolor = '#FF0000'

		if np.max(np.abs(residual_distribution)) > 20:
			range = (-20,20)
		else:
			range = None

		n, bins, patches = ax_err[ii].hist(residual_distribution,
	                 			           nbins_hist, histtype='bar',
	                 			           alpha=0.9,lw=2,color=histcolor,
	                 			           range=range)

		### Need to multiply with Gaussian amplitude A such that AREA = NPOINTS
		# AREA = AMPLITUDE * SIGMA * (2*pi)**0.5
		gnorm = residual_distribution.shape[0]/(onesig*np.sqrt(2*np.pi))*(bins[1]-bins[0])
		xplot = np.linspace(np.min(bins),np.max(bins),1e4)
		plot_gauss = gaussian(xplot,median,onesig)*gnorm

		ax_err[ii].plot(xplot,plot_gauss,color=gausscolor,lw=3)

		ax_err[ii].text(0.95,0.9,'median='+"{:.2f}".format(median),transform=ax_err[ii].transAxes,color=gausscolor,ha='right')
		ax_err[ii].text(0.95,0.8,r'1$\sigma$='+"{:.2f}".format(onesig),transform=ax_err[ii].transAxes,color=gausscolor,ha='right')

		ax_err[ii].set_xlabel(r'(true-fit)/1$\sigma$ error')
		ax_err[ii].set_ylabel('N')
		ax_err[ii].set_title(parlabels[ii])

	fig.savefig(outfolder+'derived_parameter_recovery.png',dpi=150)
	fig_err.savefig(outfolder+'residual_derivedpars.png',dpi=150)

	plt.close()

def plot_spectral_parameters(alldata,outfolder=None):

	##### OBSERVABLES
	# halpha + hbeta + dn4000 + balmer decrement
	# add hdelta absorption? which index? both?
	fig, axes = plt.subplots(2, 2, figsize = (12,12))
	ax = np.ravel(axes)
	to_plot = ['Halpha','Hbeta']
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
		par = 'log('+par+' flux)'

		#### plot
		ax[ii].errorbar(x,y,yerr,fmt='o',alpha=0.8,color='#1C86EE')
		ax[ii].set_xlabel('true '+par)
		ax[ii].set_ylabel('fit '+par)

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
	ax[ii+1].set_xlabel('true dn4000')
	ax[ii+1].set_ylabel('fit dn4000')

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
	x = x[good]
	y = y[good]
	yup = np.array([dat['eq84'][idx] for dat in alldata])[good]
	ydown = np.array([dat['eq16'][idx] for dat in alldata])[good]
	yerr = threed_dutils.asym_errors(y,yup,ydown,log=False)

	ax[ii+2].errorbar(x,y,yerr,fmt='o',alpha=0.8,color='#1C86EE')
	ax[ii+2].set_xlabel('true Balmer decrement')
	ax[ii+2].set_ylabel('fit Balmer decrement')

	ax[ii+2] = threed_dutils.equalize_axes(ax[ii+2], x, y)
	mean_offset,scat = threed_dutils.offset_and_scatter(x,y)
	ax[ii+2].text(0.96,0.12, 'scatter='+"{:.2f}".format(scat),transform = ax[ii+2].transAxes,ha='right')
	ax[ii+2].text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset), transform = ax[ii+2].transAxes,ha='right')
	#ax[ii+2].axis((2.86,6,2.86,6))

	ax[ii+2].xaxis.set_major_locator(MaxNLocator(5))
	ax[ii+2].yaxis.set_major_locator(MaxNLocator(5))

	plt.savefig(outfolder+'spectral_parameter_recovery.png',dpi=150)
	plt.close()

def plot_sfr_resid(alldata,outfolder=None):

	#### what ratio to clip to?
	rclip = (10**-0.5,10**0.5)
	rbounds = (-0.55,0.55)

	#### grab names to index
	pars = alldata[0]['parnames']
	epars = alldata[0]['eparnames']
	epars_truth = alldata[0]['truths']['extra_parnames']

	#### grab Halpha predicted + true
	emnames = alldata[0]['truths']['emnames']
	idx = emnames == 'Halpha'
	true_ha = np.array([dat['truths']['emflux'][idx] for dat in alldata])
	ha_ratio = true_ha/np.array([dat['eline_flux_q50'][idx] for dat in alldata])
	ydown = true_ha/np.array([dat['eline_flux_q84'][idx] for dat in alldata])
	yup =  true_ha/np.array([dat['eline_flux_q16'][idx] for dat in alldata])

	##### clip
	ha_ratio = np.clip(ha_ratio,rclip[0],rclip[1])
	ydown = np.clip(ydown,rclip[0],rclip[1])
	yup = np.clip(yup,rclip[0],rclip[1])

	#### errors + logify
	ha_ratio_err= threed_dutils.asym_errors(ha_ratio,yup,ydown,log=True)
	ha_ratio = np.log10(ha_ratio)


	#### SFR10
	idx = epars == 'sfr_10'
	idx_true = epars_truth == 'sfr_10'
	sfr10_true = 10**np.squeeze([dat['truths']['extra_truths'][idx_true] for dat in alldata])
	sfr10_ratio = sfr10_true/np.squeeze([dat['eq50'][idx] for dat in alldata])
	ydown = sfr10_true/np.squeeze([dat['eq84'][idx] for dat in alldata])
	yup = sfr10_true/np.squeeze([dat['eq16'][idx] for dat in alldata])

	##### clip
	sfr10_ratio = np.clip(sfr10_ratio,rclip[0],rclip[1])
	ydown = np.clip(ydown,rclip[0],rclip[1])
	yup = np.clip(yup,rclip[0],rclip[1])
	sfr10_ratio_err = threed_dutils.asym_errors(sfr10_ratio,yup,ydown,log=True)
	sfr10_ratio = np.log10(sfr10_ratio)



	#### SFR100
	idx = epars == 'sfr_100'
	idx_true = epars_truth == 'sfr_100'
	sfr100_true = 10**np.squeeze([dat['truths']['extra_truths'][idx_true] for dat in alldata])
	sfr100_ratio = sfr100_true/np.squeeze([dat['eq50'][idx] for dat in alldata])
	ydown = sfr100_true/np.squeeze([dat['eq84'][idx] for dat in alldata])
	yup = sfr100_true/np.squeeze([dat['eq16'][idx] for dat in alldata])

	#### clip
	sfr100_ratio = np.clip(sfr100_ratio,rclip[0],rclip[1])
	ydown = np.clip(ydown,rclip[0],rclip[1])
	yup = np.clip(yup,rclip[0],rclip[1])
	sfr100_ratio_err = threed_dutils.asym_errors(sfr100_ratio,yup,ydown,log=True)
	sfr100_ratio = np.log10(sfr100_ratio)



	#### LOGZSOL
	idx = pars == 'logzsol'
	logzsol_true = np.squeeze([dat['truths']['truths'][idx] for dat in alldata])
	logzsol_ratio = logzsol_true-np.squeeze([dat['q50'][idx] for dat in alldata])
	ydown = logzsol_true-np.squeeze([dat['q84'][idx] for dat in alldata])
	yup = logzsol_true-np.squeeze([dat['q16'][idx] for dat in alldata])
	logzsol_ratio_err = threed_dutils.asym_errors(logzsol_ratio,yup,ydown,log=False)



	### ha extinction ratio
	true_ha_ext = np.squeeze([dat['truths']['ha_ext'] for dat in alldata])
	ha_ext = true_ha_ext/np.squeeze([dat['ha_ext_q50'] for dat in alldata])
	ha_ext_up = true_ha_ext/np.squeeze([dat['ha_ext_q16'] for dat in alldata])
	ha_ext_down = true_ha_ext/np.squeeze([dat['ha_ext_q84'] for dat in alldata])
	ha_ext_err = threed_dutils.asym_errors(ha_ext,ha_ext_up,ha_ext_down,log=True)
	ha_ext = np.log10(ha_ext)

	### d1_d2 ratio
	true_d1_d2 = np.squeeze([dat['truths']['d1_d2'] for dat in alldata])
	d1_d2 = true_d1_d2/np.squeeze([dat['d1_d2_q50'] for dat in alldata])
	d1_d2_up = true_d1_d2/np.squeeze([dat['d1_d2_q16'] for dat in alldata])
	d1_d2_down = true_d1_d2/np.squeeze([dat['d1_d2_q84'] for dat in alldata])
	d1_d2_err = threed_dutils.asym_errors(d1_d2,d1_d2_up,d1_d2_down,log=False)

	#### true slope
	idx_slope = pars == 'sf_tanslope'
	true_slope = np.array([dat['truths']['truths'][idx_slope] for dat in alldata])

	##### plots
	fig, axes = plt.subplots(2, 4, figsize = (24,12))
	plt.subplots_adjust(wspace=0.3,hspace=0.2,bottom=0.1,top=0.9,left=0.1,right=0.9)
	ax = np.ravel(axes)

	ax[0].errorbar(np.log10(sfr100_true),np.log10(sfr10_true),fmt='o',alpha=0.8,color='#1C86EE')
	ax[0].set_xlabel('true log(SFR) [100 Myr]')
	ax[0].set_ylabel('true log(SFR) [10 Myr]')

	ax[0] = threed_dutils.equalize_axes(ax[0], np.log10(sfr100_true),np.log10(sfr10_true))
	mean_offset,scat = threed_dutils.offset_and_scatter(np.log10(sfr100_true),np.log10(sfr10_true))
	ax[0].text(0.96,0.12, 'scatter='+"{:.2f}".format(scat),transform = ax[0].transAxes,ha='right')
	ax[0].text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset), transform = ax[0].transAxes,ha='right')

	ax[0].xaxis.set_major_locator(MaxNLocator(5))
	ax[0].yaxis.set_major_locator(MaxNLocator(6))

	ax[1].errorbar(sfr10_ratio,ha_ratio,xerr=sfr10_ratio_err,yerr=ha_ratio_err,fmt='o',alpha=0.8,color='#1C86EE')
	ax[1].set_xlabel(r'log(SFR$_{\mathrm{true}}$/SFR$_{\mathrm{fit}}$) [10 Myr]')
	ax[1].set_ylabel(r'log(H$\alpha_{\mathrm{true}}$/H$\alpha_{\mathrm{fit}}$)')
	ax[1].axis((rbounds[0],rbounds[1],rbounds[0],rbounds[1]))

	ax[1].axvline(0.0, ls="dashed", color='k',linewidth=1.5)
	ax[1].axhline(0.0, ls="dashed", color='k',linewidth=1.5)

	ax[1].xaxis.set_major_locator(MaxNLocator(5))
	ax[1].yaxis.set_major_locator(MaxNLocator(6))

	ax[2].errorbar(sfr100_ratio,ha_ratio,xerr=sfr100_ratio_err,yerr=ha_ratio_err,fmt='o',alpha=0.8,color='#1C86EE')
	ax[2].set_xlabel(r'log(SFR$_{\mathrm{true}}$/SFR$_{\mathrm{fit}}$) [100 Myr]')
	ax[2].set_ylabel(r'log(H$\alpha_{\mathrm{true}}$/H$\alpha_{\mathrm{fit}}$)')
	ax[2].axis((rbounds[0],rbounds[1],rbounds[0],rbounds[1]))

	ax[2].axvline(0.0, ls="dashed", color='k',linewidth=1.5)
	ax[2].axhline(0.0, ls="dashed", color='k',linewidth=1.5)

	ax[2].xaxis.set_major_locator(MaxNLocator(5))
	ax[2].yaxis.set_major_locator(MaxNLocator(6))

	ax[3].errorbar(true_slope,ha_ratio,yerr=ha_ratio_err,fmt='o',alpha=0.8,color='#1C86EE')
	ax[3].set_xlabel(r'sf_tanslope [true]')
	ax[3].set_ylabel(r'log(H$\alpha_{\mathrm{true}}$/H$\alpha_{\mathrm{fit}}$) [flux]')
	ax[3].set_ylim(rbounds)

	ax[3].axhline(0.0, ls="dashed", color='k',linewidth=1.5)

	ax[3].xaxis.set_major_locator(MaxNLocator(5))
	ax[3].yaxis.set_major_locator(MaxNLocator(6))

	ax[4].errorbar(ha_ext,ha_ratio,xerr=ha_ext_err,yerr=ha_ratio_err,fmt='o',alpha=0.8,color='#1C86EE')
	ax[4].set_xlabel(r'log(true attenuation/fit attenuation) [6563 $\AA$]')
	ax[4].set_ylabel(r'log(H$\alpha_{\mathrm{extinction}}$/H$\alpha_{\mathrm{fit}}$) [flux]')
	ax[4].set_ylim(rbounds)
	ax[4].set_xlim(-0.3,0.3)


	ax[4].axhline(0.0, ls="dashed", color='k',linewidth=1.5)
	ax[4].axvline(0.0, ls="dashed", color='k',linewidth=1.5)

	ax[4].xaxis.set_major_locator(MaxNLocator(5))
	ax[4].yaxis.set_major_locator(MaxNLocator(6))

	ax[6].errorbar(sfr10_ratio,sfr100_ratio,xerr=sfr10_ratio_err,yerr=sfr100_ratio_err,fmt='o',alpha=0.8,color='#1C86EE')
	ax[6].set_xlabel(r'log(SFR$_{\mathrm{true}}$/SFR$_{\mathrm{fit}}$) [10 Myr]')
	ax[6].set_ylabel(r'log(SFR$_{\mathrm{true}}$/SFR$_{\mathrm{fit}}$) [100 Myr]')
	ax[6].set_ylim(rbounds)
	ax[6].set_xlim(rbounds)

	ax[6].axvline(0.0, ls="dashed", color='k',linewidth=1.5)
	ax[6].axhline(0.0, ls="dashed", color='k',linewidth=1.5)

	ax[6].xaxis.set_major_locator(MaxNLocator(5))
	ax[6].yaxis.set_major_locator(MaxNLocator(6))

	ax[7].errorbar(true_slope,sfr100_ratio,yerr=sfr100_ratio_err,fmt='o',alpha=0.8,color='#1C86EE')
	ax[7].set_xlabel(r'sf_tanslope [true]')
	ax[7].set_ylabel(r'log(SFR$_{\mathrm{true}}$/SFR$_{\mathrm{fit}}$) [100 Myr]')
	ax[7].set_ylim(rbounds)

	ax[7].axhline(0.0, ls="dashed", color='k',linewidth=1.5)

	ax[7].xaxis.set_major_locator(MaxNLocator(5))
	ax[7].yaxis.set_major_locator(MaxNLocator(6))

	plt.savefig(outfolder+'sfh_variability.png',dpi=150)
	plt.close()

	fig, ax = plt.subplots(1, 1, figsize = (8,8))

	ax.errorbar(true_slope,ha_ratio,yerr=ha_ratio_err,fmt='o',alpha=0.8,color='#1C86EE')
	ax.set_xlabel(r'sf_tanslope [true]')
	ax.set_ylabel(r'log(H$\alpha_{\mathrm{true}}$/H$\alpha_{\mathrm{fit}}$) [flux]')
	ax.set_ylim(rbounds)

	ax.axhline(0.0, ls="dashed", color='k',linewidth=1.5)

	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(6))

	plt.savefig(outfolder+'ha_slope.png',dpi=150)
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

	plt.savefig(outfolder+'mass_residual.png',dpi=150)
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

	plt.savefig(outfolder+'likelihood_ratio.png',dpi=150)
	plt.close()

if __name__ == "__main__":
   sys.exit(make_plots())





