import numpy as np
import matplotlib.pyplot as plt
import os, threed_dutils
import matplotlib as mpl
from astropy import constants
import magphys_plot_pref
import copy
from scipy.optimize import minimize
import pickle
import corner
from matplotlib.ticker import MaxNLocator
import brown_quality_cuts
from np_mocks_analysis import gaussian

#### set up colors and plot style
prosp_color = '#e60000'
obs_color = '#95918C'
magphys_color = '#1974D2'
dpi = 150

#### herschel / non-herschel
nhargs = {'fmt':'o','alpha':0.7,'color':'0.2'}
hargs = {'fmt':'o','alpha':0.7,'color':'0.2'}
herschdict = [copy.copy(hargs),copy.copy(nhargs)]

#### AGN
colors = ['blue', 'purple', 'red']
labels = ['SF', 'SF/AGN', 'AGN']

sfargs = {'color':'blue', 'label':'SF'}
compargs = {'color': 'purple', 'label': 'SF/AGN'}
agnargs = {'color': 'red', 'label': 'AGN','fillstyle':'none'}
bptdict = [copy.copy(sfargs),copy.copy(compargs),copy.copy(agnargs)]

#### minimum SFR
minsfr = 1e-3

#### Halpha plot limit
ha_flux_lim = (3.5,10.0)
ha_eqw_lim = (-1.3,3.5)

### lambda for extinction curve delta 
lam1_extdiff = 5450.
lam2_extdiff = 5550.

def minlog(x,axis=None):
	'''
	Given a numpy array, take the base-10 logarithm
	IF THERE ARE ZEROES, first set them to the minimum of the array
	'''
	if axis is None:
		zeros = x == 0.0
		x[zeros] = np.amin(x[~zeros])
		x[zeros] = np.mean(x[~zeros])
	else:
		for kk in xrange(x.shape[1]):
			zeros = x[:,kk] == 0.0
			x[zeros,kk] = np.amin(x[~zeros,kk])
			#x[zeros,kk] = np.mean(x[~zeros,kk])

	return np.log10(x)

def pdf_quantiles(bins,truth,model_pdf,bottom_lim,top_lim):

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

def pdf_stats(bins,pdf):

	total = np.sum(pdf)
	cumsum = np.cumsum(pdf)
	median = np.interp(total/2., cumsum, bins)
	onesig_range = np.interp(total*0.84, cumsum, bins) - np.interp(total*0.16, cumsum, bins)

	return median,onesig_range

def pdf_distance(chain,truth,truth_chain,bins,delta_functions=False,center_obs=False):

	#### chain properties
	model_chain_center = np.median(chain)
	bmin, bmax = bins.min(),bins.max()

	#### model PDF
	clipped_centered_chain = np.clip(chain-model_chain_center,bmin,bmax)
	model_pdf,_ = np.histogram(clipped_centered_chain,bins=bins,density=True)

	#### observed PDF
	if (truth_chain == None) or (delta_functions): # we have no errors (dn4000), use delta functions
		clipped_truth = np.clip(truth-model_chain_center,bmin,bmax)
		if center_obs:
			clipped_truth = np.clip(0.0,bmin,bmax)
		obs_pdf,_ = np.histogram(clipped_truth,bins=bins,density=True)

	elif len(truth_chain) > 1: # we have a chain
		clipped_centered_chain = np.clip(truth_chain-model_chain_center,bmin,bmax)
		if center_obs:
			clipped_centered_chain = np.clip(truth_chain-np.median(truth_chain),bmin,bmax)
		clipped_centered_chain = clipped_centered_chain[np.isfinite(clipped_centered_chain)] # remove nans
		obs_pdf,_ = np.histogram(clipped_centered_chain,bins=bins,density=True)

	elif len(truth_chain) == 1: # we have a sigma, sample from a Gaussian
		ransamps = np.random.normal(loc=truth-model_chain_center, scale=truth_chain, size=1000)
		if center_obs:
			ransamps = np.random.normal(loc=0.0, scale=truth_chain, size=1000)
		ransamps = np.clip(ransamps,bmin,bmax)
		obs_pdf,_ = np.histogram(ransamps,bins=bins,density=True)


	return model_pdf, obs_pdf

def specpar_pdf_distance(pinfo,alldata, delta_functions=True, center_obs=True):

	#### names and labels
	test_labels = [r'H$\alpha$ flux',r'H$\beta$ flux','Balmer decrement',
	               r'log(Z/Z$_{\odot}$)',r'D$_n$4000',r'H$\delta$ EQW']
	obs_names = ['f_ha','f_hb','bdec',
	             None,'dn4000','hdel_eqw'] # will have to jig metallicity

	# hdelta flags
	index_flags = np.loadtxt('/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/hdelta_index.txt', dtype = {'names':('name','flag'),'formats':('S40','i4')})
	anames = alldata[0]['spec_info']['absnames']

	### setup figure
	out = {}
	for ii, par in enumerate(test_labels):

		if ii == 0: # halpha
			keep_idx = brown_quality_cuts.halpha_cuts(pinfo)
			bins = np.linspace(-1,1,51) # dex
			if delta_functions:
				bins = np.linspace(-1,1,26) # dex
			xunit = 'dex'
		if ii == 1: # hbeta
			keep_idx = brown_quality_cuts.halpha_cuts(pinfo)
			bins = np.linspace(-1,1,51) # dex
			if delta_functions:
				bins = np.linspace(-1,1,26) # dex
			xunit = 'dex'
		if ii == 2: # bdec
			keep_idx = brown_quality_cuts.halpha_cuts(pinfo)
			bins = np.linspace(-1,1,51) # BALMER DECREMENT UNITS (?)
			if delta_functions:
				bins = np.linspace(-0.5,0.5,26) # dex
			xunit = 'magnitudes'
		if ii == 3: # met
			_, _, truemet, truemet_errs, a3d_alpha, keep_idx = brown_quality_cuts.load_atlas3d(pinfo)
			bins = np.linspace(-0.6,0.6,51) # dex
			if delta_functions:
				bins = np.linspace(-0.21,0.21,16) # dex
			xunit = 'dex'
		if ii == 4: # dn4000
			keep_idx = brown_quality_cuts.dn4000_cuts(pinfo)
			bins = np.linspace(-0.35,0.35,61) # Dn4000
			if delta_functions:
				bins = np.linspace(-0.35,0.35,31) # Dn4000
			xunit = None
		if ii == 5: # hdelta absorption
			keep_idx = brown_quality_cuts.hdelta_cuts(pinfo)
			bins = np.linspace(-0.6,0.6,51) # dex
			if delta_functions:
				bins = np.linspace(-0.4,0.4,26) # dex
			xunit = 'dex'

		modpdf = np.zeros(bins.shape[0]-1)
		obspdf = np.zeros(bins.shape[0]-1)

		for kk, dat in enumerate(alldata):

			if keep_idx[kk] == False:
				continue

			if ii == 0: # halpha
				chain = np.log10(np.squeeze(dat['model_emline']['flux']['chain'][:,dat['model_emline']['emnames']=='Halpha']))
				truth = np.log10(pinfo['obs'][obs_names[ii]][kk,0])
				truth_chain = np.log10(pinfo['obs']['pdf_ha'][kk])

			if ii == 1: # hbeta
				chain = np.log10(np.squeeze(dat['model_emline']['flux']['chain'][:,dat['model_emline']['emnames']=='Hbeta']))
				truth = np.log10(pinfo['obs'][obs_names[ii]][kk,0])
				truth_chain = np.log10(pinfo['obs']['pdf_hb'][kk])

			if ii == 2: # bdec
				chain = bdec_to_ext(dat['pextras']['flatchain'][:,dat['pextras']['parnames']=='bdec_cloudy'])
				truth = bdec_to_ext(pinfo['obs'][obs_names[ii]][kk])
				truth_chain = bdec_to_ext(np.random.choice(pinfo['obs']['pdf_ha'][kk],size=1000) / np.random.choice(pinfo['obs']['pdf_hb'][kk],size=1000))

			if ii == 3: # met (MAKE SURE THIS WORKS)
				chain = np.squeeze(dat['pquantiles']['random_chain'][:,dat['pquantiles']['parnames']=='logzsol'])
				truth = truemet[np.sum(keep_idx[:kk])][0]
				truth_chain = truemet_errs[np.sum(keep_idx[:kk])]

			if ii == 4: # dn4000
				chain = dat['spec_info']['dn4000']['chain']
				truth = pinfo['obs'][obs_names[ii]][kk]
				truth_chain = np.atleast_1d(0.05)

			if ii == 5: # hdelta absorption
				match = index_flags['name'] == dat['objname'].replace(' ','')
				flag = index_flags['flag'][match]
				if flag == 1: hdelta_ind = 'hdelta_narrow'
				if flag == 0: hdelta_ind = 'hdelta_wide'
				chain = np.log10(np.squeeze(dat['spec_info']['eqw']['chain'][:,dat['spec_info']['absnames'] == hdelta_ind]))

				truth = np.log10(pinfo['obs'][obs_names[ii]][kk,0])
				truth_chain = np.log10(np.clip(pinfo['obs']['hdel_eqw_chain'][kk],0.01,np.inf))


			tmodpdf, tobspdf = pdf_distance(chain,truth,truth_chain,bins,delta_functions=delta_functions,center_obs=center_obs)
			modpdf += tmodpdf
			obspdf += tobspdf

		outtemp = {}
		outtemp['model_pdf'] = modpdf
		outtemp['obs_pdf'] = obspdf
		outtemp['bins'] = bins
		outtemp['plot_bins'] = (bins[:-1] + bins[1:])/2.
		outtemp['N'] = np.sum(keep_idx)
		outtemp['xunit'] = xunit
		out[par] = outtemp

	return out

def specpar_pdf_plot(pdf,outname=None):

	fig, axarr = plt.subplots(1,2, figsize = (12,6.5))
	ax = np.ravel(axarr)

	obscolor = '#FF420E'
	modcolor = '#375E97'

	keys = [r'H$\alpha$ flux',r'H$\beta$ flux']

	for i, key in enumerate(keys):

		#### create step function
		plotx = np.empty((pdf[key]['bins'].size*2,), dtype=pdf[key]['bins'].dtype)
		plotx[0::2] = pdf[key]['bins']
		plotx[1::2] = pdf[key]['bins']

		ploty_mod, ploty_obs = [np.empty((pdf[key]['model_pdf'].size*2,), dtype=pdf[key]['model_pdf'].dtype) for X in xrange(2)]
		ploty_mod[0::2] = pdf[key]['model_pdf']
		ploty_mod[1::2] = pdf[key]['model_pdf']
		ploty_mod = np.concatenate((np.atleast_1d(0.0),ploty_mod,np.atleast_1d(0.0)))
		ploty_obs[0::2] = pdf[key]['obs_pdf']
		ploty_obs[1::2] = pdf[key]['obs_pdf']
		ploty_obs = np.concatenate((np.atleast_1d(0.0),ploty_obs,np.atleast_1d(0.0)))

		ax[i].plot(plotx,ploty_obs,alpha=0.8,lw=2,color=obscolor)
		ax[i].plot(plotx,ploty_mod,alpha=0.8,lw=2,color=modcolor)

		ax[i].fill_between(plotx, np.zeros_like(ploty_obs), ploty_obs, 
						   color=obscolor,
						   alpha=0.3)
		ax[i].fill_between(plotx, np.zeros_like(ploty_mod), ploty_mod, 
						   color=modcolor,
						   alpha=0.3)

		xunit = ''
		if pdf[key]['xunit'] != None:
			xunit = pdf[key]['xunit']

		ax[i].set_xlabel(r'position within model posterior [dex]')
		ax[i].set_ylabel('density')
		ax[i].set_title(key)
		ax[i].set_ylim(0.0,ax[i].get_ylim()[1]*1.125)
		ax[i].set_xlim(pdf[key]['bins'].min(),pdf[key]['bins'].max())
		for tl in ax[i].get_yticklabels():tl.set_visible(False) # no tick labels

		ax[i].text(0.04,0.93,r'$\Sigma$ (model posteriors)',transform=ax[i].transAxes,color=modcolor)
		ax[i].text(0.04,0.88,'observations',transform=ax[i].transAxes,color=obscolor)
		ax[i].text(0.04,0.83,'N='+str(pdf[key]['N']),transform=ax[i].transAxes)

		onesig_perc = pdf_quantiles(pdf[key]['plot_bins'],pdf[key]['obs_pdf'],pdf[key]['model_pdf'],0.16,0.84)/0.68
		twosig_perc = pdf_quantiles(pdf[key]['plot_bins'],pdf[key]['obs_pdf'],pdf[key]['model_pdf'],0.025,0.975)/0.95

		ax[i].text(0.05,0.76,r'$\frac{1\sigma_{\mathrm{obs}}}{1\sigma_{\mathrm{true}}}$:'+"{:.2f}".format(onesig_perc),transform=ax[i].transAxes)
		ax[i].text(0.05,0.69,r'$\frac{2\sigma_{\mathrm{obs}}}{2\sigma_{\mathrm{true}}}$:'+"{:.2f}".format(twosig_perc),transform=ax[i].transAxes)
		
	plt.tight_layout()
	plt.savefig(outname,dpi=150)
	plt.close()

def merge_dicts(*dict_args):
    '''
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    '''
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def translate_line_names(linenames):
	'''
	translate from my names to Moustakas names
	'''
	translate = {r'H$\alpha$': 'Ha',
				 '[OIII] 5007': 'OIII',
	             r'H$\beta$': 'Hb',
	             '[NII] 6583': 'NII'}

	return np.array([translate[line] for line in linenames])

def new_translate_line_names(linenames):
	'''
	translate from my names to Moustakas names
	'''
	translate = {r'H$\alpha$': 'H_ALPHA',
				 '[OIII] 5007': 'OIII_5007',
	             r'H$\beta$': 'H_BETA',
	             '[NII] 6583': 'NII_6584'}

	return np.array([translate[line] for line in linenames])

def remove_doublets(x, names):

	if any('[OIII]' in s for s in list(names)):
		keep = np.array(names) != '[OIII] 4959'
		x = x[keep]
		names = names[keep]
		#if not isinstance(x[0],basestring):
		#	x[np.array(names) == '[OIII] 4959'] *= 3.98

	if any('[NII]' in s for s in list(names)):
		keep = np.array(names) != '[NII] 6549'
		x = x[keep]
		names = names[keep]

	return x

def ret_inf(alldata,field, model='obs',name=None):

	'''
	returns information from alldata
	'''

	# emission line names and indexes
	emline_names = alldata[0]['residuals']['emlines']['em_name']
	idx = emline_names == name

	#### if we want absorption
	if 'balmer' in field:
		absline_names = alldata[0]['residuals']['emlines'][model]['balmer_names']
		idx = absline_names == name

	### define filler
	# if EQW, we don't have that vector defined; use flux
	if field is 'eqw_rest':
		fillvalue = 0.0
	else:
		fillvalue = np.zeros_like(alldata[0]['residuals']['emlines'][model][field])

	# if we have a 'None' name, we want all of them!
	if name is None:
		return np.squeeze(np.array([f['residuals']['emlines'][model][field] if f['residuals']['emlines'] is not None else fillvalue for f in alldata]))
	
	### for backward compatibility, if we ask for 'eqw_rest', return a vector of shape (ngals,3)
	# with middle, up, down values
	if field is 'eqw_rest':
		cont = ret_cont(alldata,model=model,name=name)
		out = np.zeros(shape=(129,3))
		out[:,0] = np.array([f['residuals']['emlines'][model]['flux'][idx][0]/cont[i] if f['residuals']['emlines'] is not None else fillvalue for i,f in enumerate(alldata)])
		out[:,1] = np.array([f['residuals']['emlines'][model]['flux_errup'][idx][0]/cont[i] if f['residuals']['emlines'] is not None else fillvalue for i,f in enumerate(alldata)])
		out[:,2] = np.array([f['residuals']['emlines'][model]['flux_errdown'][idx][0]/cont[i] if f['residuals']['emlines'] is not None else fillvalue for i,f in enumerate(alldata)])		
	else:
		out = np.squeeze(np.array([f['residuals']['emlines'][model][field] if f['residuals']['emlines'] is not None else fillvalue for f in alldata])[:,idx])

	return out

def ret_cont(alldata,model='obs',name='Halpha'):
	'''
	return continuum
	'''

	### why did I name these different?
	if model == 'obs': idx = 'continuum_obs'
	if model != 'obs': idx = 'continuum_mod'

	### find names
	abslines = alldata[0]['residuals']['emlines']['obs']['balmer_names']

	#### define continuum for different zones
	if (name == '[OIII] 4959') or (name == '[OIII] 5007') or name == ('H$\\beta$'):
		cont_idx = abslines =='hbeta'
	if (name == '[NII] 6549') or (name == '[NII] 6583') or (name == 'H$\\alpha$'):
		cont_idx = abslines == 'halpha_wide'
	if name == 'hdelta':
		cont_idx = abslines == 'hdelta_wide'

	continuum = np.squeeze([f['residuals']['emlines'][model][idx][cont_idx][0] if f['residuals']['emlines'] is not None else 0.0 for f in alldata])

	return continuum

def compare_moustakas_newfluxes(alldata,dat,eline_to_plot,objnames,outname='test.png',outdec='bdec.png',model='obs'):

	##########
	#### extract info for objects with measurements in both catalogs
	##########

	idx_moust = []
	moust_objnames = []
	prosp_names = alldata[0]['residuals']['emlines']['em_name']
	moust_names = new_translate_line_names(eline_to_plot)
	yplot = None

	# pull out moustakas fluxes and errors
	for ii in xrange(len(dat)):
		idx_moust.append(False)
		if dat[ii] is not None:
			idx_moust[ii] = True

			yflux = np.array([dat[ii][name][0][0,None] for name in moust_names])
			yflux_err = np.array([dat[ii][name][0][1,None] for name in moust_names])

			if yplot is None:
				yplot = yflux
				yplot_err = yflux_err
			else:
				yplot = np.concatenate((yplot,yflux),axis=1)
				yplot_err = np.concatenate((yplot_err,yflux_err),axis=1)

			moust_objnames.append(objnames[ii])

	##### grab Prospector information
	ind = np.array(idx_moust,dtype=bool)

	xplot = np.transpose(ret_inf(alldata,'flux',model=model))[:,ind]
	xplot_errup = np.transpose(ret_inf(alldata,'flux_errup',model=model))[:,ind]
	xplot_errdown = np.transpose(ret_inf(alldata,'flux_errdown',model=model))[:,ind]

	#### grab continuum
	# must use luminosity
	lum = np.transpose(ret_inf(alldata,'lum',model=model))[:,ind] / constants.L_sun.cgs.value
	hb_cont = ret_cont(alldata,model=model,name='H$\\beta$')[ind]
	ha_cont = ret_cont(alldata,model=model,name='H$\\alpha$')[ind]

	#### plot information
	nplot = len(moust_names)
	ncols = int(np.round((nplot)/2.))
	fig, axes = plt.subplots(ncols, 2, figsize = (12,6*ncols))
	axes = np.ravel(axes)

	#### loop over Moustakas emission lines
	for ii in xrange(nplot):
		
		#### find match in Prospector
		pro_idx = eline_to_plot[ii] == prosp_names

		#### demand that the both Moustakas and my fluxes are nonzero
		ok_idx = (yplot[ii,:] > 0.0) & (np.squeeze(xplot[pro_idx,:]) > 0.0)
		yp = yplot[ii,ok_idx]
		yp_err = yplot_err[ii,ok_idx]
		xp = xplot[pro_idx,ok_idx]

		#### calculate errors
		xp_errup = xplot_errup[pro_idx,ok_idx]
		xp_errdown = xplot_errdown[pro_idx,ok_idx]
		xp_err = threed_dutils.asym_errors(xp, xp_errdown, xp_errup)

		#### measure difference
		typ = np.log10(yp)-np.log10(xp)

		# calculate EQW
		if moust_names[ii] == 'OIII_5007' or moust_names[ii] == 'H_BETA':
			eqw = lum[pro_idx,ok_idx] / hb_cont[ok_idx]
		elif moust_names[ii] == 'NII_6584' or moust_names[ii] == 'H_ALPHA':
			eqw = lum[pro_idx,ok_idx] / ha_cont[ok_idx]
		else:
			print 1/0

		#### NO NaNs!
		if np.sum(np.isfinite(typ)) != typ.shape[0]:
			print 1/0

		axes[ii].errorbar(np.log10(eqw),typ,
						  xerr=xp_err, 
			              linestyle=' ',
			              **nhargs)
		maxyval = np.max(np.abs(typ))
		axes[ii].set_ylim(-maxyval,maxyval)

		axes[ii].set_ylabel('log(Moustakas+10/measured) '+eline_to_plot[ii])
		axes[ii].set_xlabel('log(EQW '+eline_to_plot[ii]+')')
		off,scat = threed_dutils.offset_and_scatter(np.log10(xp),np.log10(yp),biweight=True)
		axes[ii].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) +' dex',
				  transform = axes[ii].transAxes,horizontalalignment='right')
		axes[ii].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = axes[ii].transAxes,horizontalalignment='right')
		axes[ii].axhline(0, linestyle='--', color='0.1')

		if moust_names[ii] == 'H_BETA':
			axes[ii].set_xlim(-0.3,np.log10(80))
		if moust_names[ii] == 'H_ALPHA':
			axes[ii].set_xlim(0.5,np.log10(400))

	plt.tight_layout()
	plt.savefig(outname,dpi=dpi)
	plt.close()

def compare_moustakas_fluxes(alldata,dat,emline_names,objnames,outname='test.png',outdec='bdec.png',model='obs'):

	##########
	#### extract info for objects with measurements in both catalogs
	##########
	idx_moust = []
	moust_objnames = []
	emline_names_doubrem = remove_doublets(emline_names,emline_names)
	moust_names = translate_line_names(emline_names_doubrem)
	yplot = None

	for ii in xrange(len(dat)):
		idx_moust.append(False)
		if dat[ii] is not None:
			idx_moust[ii] = True

			yflux = np.array([dat[ii]['F'+name][0] for name in moust_names])*1e-15
			yfluxerr = np.array([dat[ii]['e_F'+name][0] for name in moust_names])*1e-15
			if yplot is None:
				yplot = yflux[:,None]
				yerr = yfluxerr[:,None]
			else:
				yplot = np.concatenate((yplot,yflux[:,None]),axis=1)
				yerr = np.concatenate((yerr,yfluxerr[:,None]),axis=1)

			moust_objnames.append(objnames[ii])

	##### grab Prospector information
	ind = np.array(idx_moust,dtype=bool)

	xplot = remove_doublets(np.transpose(ret_inf(alldata,'flux',model=model)),emline_names)[:,ind]
	xplot_errup = remove_doublets(np.transpose(ret_inf(alldata,'flux_errup',model=model)),emline_names)[:,ind]
	xplot_errdown = remove_doublets(np.transpose(ret_inf(alldata,'flux_errdown',model=model)),emline_names)[:,ind]

	#### grab continuum
	# must use luminosity
	lum = remove_doublets(np.transpose(ret_inf(alldata,'lum',model=model)),emline_names)[:,ind] / constants.L_sun.cgs.value
	hb_cont = ret_cont(alldata,model=model,name='H$\\beta$')[ind]
	ha_cont = ret_cont(alldata,model=model,name='H$\\alpha$')[ind]

	#### plot information
	# remove NaNs from Moustakas here, which are presumably emission lines
	# where the flux was measured to be negative
	nplot = len(moust_names)
	ncols = int(np.round((nplot)/2.))
	fig, axes = plt.subplots(ncols, 2, figsize = (12,6*ncols))
	axes = np.ravel(axes)
	for ii in xrange(nplot):
		
		ok_idx = np.isfinite(yplot[ii,:])
		yp = yplot[ii,ok_idx]
		yp_err = threed_dutils.asym_errors(yp,
			                 yplot[ii,ok_idx]+yerr[ii,ok_idx],
			                 yplot[ii,ok_idx]-yerr[ii,ok_idx])

		# if I measure < 0 where Moustakas measures > 0,
		# clip to Moustakas minimum measurement, and
		# set errors to zero
		bad = xplot[ii,ok_idx] < 0
		xp = xplot[ii,ok_idx]
		xp_errup = xplot_errup[ii,ok_idx]
		xp_errdown = xplot_errdown[ii,ok_idx]
		if np.sum(bad) > 0:
			xp[bad] = np.min(np.concatenate((yplot[ii,ok_idx],xplot[ii,ok_idx][~bad])))*0.6
			xp_errup[bad] = 0.0
			xp_errdown[bad] = 1e-99
		xp_err = threed_dutils.asym_errors(xp, xp_errdown, xp_errup)

		typ = np.log10(yp)-np.log10(xp)

		# calculate EQW
		if moust_names[ii] == 'OIII' or moust_names[ii] == 'Hb':
			eqw = lum[ii,ok_idx] / hb_cont[ok_idx]
		elif moust_names[ii] == 'NII' or moust_names[ii] == 'Ha':
			eqw = lum[ii,ok_idx] / ha_cont[ok_idx]
		else:
			print 1/0

		axes[ii].errorbar(eqw,typ,yerr=yp_err,
						  xerr=xp_err, 
			              linestyle=' ',
			              **nhargs)
		maxyval = np.max(np.abs(typ))
		axes[ii].set_ylim(-maxyval,maxyval)

		axes[ii].set_ylabel('log(Moustakas+10/measured) '+emline_names_doubrem[ii])
		axes[ii].set_xlabel('EQW '+emline_names_doubrem[ii])
		off,scat = threed_dutils.offset_and_scatter(np.log10(xp),np.log10(yp),biweight=True)
		axes[ii].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) +' dex',
				  transform = axes[ii].transAxes,horizontalalignment='right')
		axes[ii].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = axes[ii].transAxes,horizontalalignment='right')
		axes[ii].axhline(0, linestyle='--', color='0.1')

	plt.tight_layout()
	plt.savefig(outname,dpi=dpi)
	plt.close()

	##### PLOT OBS VS OBS BALMER DECREMENT
	hb_idx_me = emline_names_doubrem == 'H$\\beta$'
	ha_idx_me = emline_names_doubrem == 'H$\\alpha$'
	hb_idx_mo = moust_names == 'Hb'
	ha_idx_mo = moust_names == 'Ha'

	# must have a positive flux in all measurements of all emission lines
	idx = np.isfinite(yplot[hb_idx_mo,:]) & \
          np.isfinite(yplot[ha_idx_mo,:]) & \
          (xplot[hb_idx_me,:] > 0) & \
          (xplot[ha_idx_me,:] > 0)
	idx = np.squeeze(idx)
	mydec = xplot[ha_idx_me,idx] / xplot[hb_idx_me,idx]
	modec = yplot[ha_idx_mo,idx] / yplot[hb_idx_mo,idx]
  
	fig, ax = plt.subplots(1,1, figsize = (10,10))
	ax.errorbar(mydec, modec, fmt='o',alpha=0.6,linestyle=' ')
	ax.set_xlabel('measured Balmer decrement')
	ax.set_ylabel('Moustakas+10 Balmer decrement')
	ax = threed_dutils.equalize_axes(ax, mydec,modec)
	off,scat = threed_dutils.offset_and_scatter(mydec,modec,biweight=True)
	ax.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat),
			  transform = ax.transAxes,horizontalalignment='right')
	ax.text(0.96,0.1, 'mean offset='+"{:.2f}".format(off),
			      transform = ax.transAxes,horizontalalignment='right')
	ax.plot([2.86,2.86],[0.0,15.0],linestyle='-',color='black')
	ax.plot([0.0,15.0],[2.86,2.86],linestyle='-',color='black')
	ax.set_xlim(1,10)
	ax.set_ylim(1,10)
	plt.savefig(outdec,dpi=dpi)
	plt.close()



def compare_model_flux(alldata, emline_names, outname = 'test.png'):

	#################
	#### plot Prospector versus MAGPHYS flux
	#################
	ncol = int(np.ceil(len(emline_names)/2.))
	fig, axes = plt.subplots(ncol,2, figsize = (11,ncol*5))
	axes = np.ravel(axes)
	for ii,emname in enumerate(emline_names):
		magdat = np.log10(ret_inf(alldata,'lum',model='MAGPHYS',name=emname)) 
		prodat = np.log10(ret_inf(alldata,'lum',model='obs',name=emname)) 
		yplot = prodat-magdat

		xplot = np.log10(ret_inf(alldata,'eqw_rest',model='obs',name=emname))
		idx = np.isfinite(yplot)

		axes[ii].errorbar(xplot[idx,0], yplot[idx],linestyle=' ',**nhargs)
		maxyval = np.max(np.abs(yplot[idx]))
		axes[ii].set_ylim(-maxyval,maxyval)
		
		xlabel = r"log({0} EQW) [Prospector]"
		ylabel = r"log(Prosp/MAGPHYS) [{0} flux]"
		axes[ii].set_xlabel(xlabel.format(emname))
		axes[ii].set_ylabel(ylabel.format(emname))

		# horizontal line
		axes[ii].axhline(0, linestyle=':', color='grey')

		# equalize axes, show offset and scatter
		off,scat = threed_dutils.offset_and_scatter(magdat[idx],
			                                        prodat[idx],
			                                        biweight=True)
		axes[ii].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
				  transform = axes[ii].transAxes,horizontalalignment='right')
		axes[ii].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = axes[ii].transAxes,horizontalalignment='right')
	
	# save
	plt.tight_layout()
	plt.savefig(outname,dpi=dpi)
	plt.close()	

def fmt_emline_info(alldata,add_abs_err = True):

	ngals = len(alldata)

	##### Observed quantities
	## emission line fluxes and EQWs, from CGS to Lsun
	obslines = {}
	mag      = {}
	prosp    = {}

	##### continuum first
	continuum =  ret_inf(alldata,'continuum_obs',model='obs')
	lam_continuum = ret_inf(alldata,'continuum_lam',model='obs')
	absline_names = alldata[0]['residuals']['emlines']['obs']['balmer_names']
	emline_names = alldata[0]['residuals']['emlines']['em_name']
	fillvalue = None

	obslines['ha_obs_cont'] = continuum[:,absline_names == 'halpha_wide'][:,0]
	obslines['hb_obs_cont'] = continuum[:,absline_names == 'hbeta'][:,0]
	obslines['hd_obs_cont'] = continuum[:,absline_names == 'hdelta_wide'][:,0]

	##### emission line EQWs and fluxes
	obslines['f_ha'] = np.transpose([ret_inf(alldata,'lum',model='obs',name='H$\\alpha$'),
		                             ret_inf(alldata,'lum_errup',model='obs',name='H$\\alpha$'),
		                             ret_inf(alldata,'lum_errdown',model='obs',name='H$\\alpha$')]) / constants.L_sun.cgs.value
	obslines['err_ha'] = (obslines['f_ha'][:,1] - obslines['f_ha'][:,2])/2.
	idx = emline_names == 'H$\\alpha$'
	obslines['pdf_ha'] = np.array([np.squeeze(f['residuals']['emlines']['obs']['lum_chain'][:,idx] / constants.L_sun.cgs.value) if f['residuals']['emlines'] is not None else fillvalue for f in alldata])

	obslines['f_hb'] = np.transpose([ret_inf(alldata,'lum',model='obs',name='H$\\beta$'),
		                             ret_inf(alldata,'lum_errup',model='obs',name='H$\\beta$'),
		                             ret_inf(alldata,'lum_errdown',model='obs',name='H$\\beta$')]) / constants.L_sun.cgs.value
	obslines['err_hb'] = (obslines['f_hb'][:,1] - obslines['f_hb'][:,2])/2.
	idx = emline_names == 'H$\\beta$'
	obslines['pdf_hb'] = np.array([np.squeeze(f['residuals']['emlines']['obs']['lum_chain'][:,idx] / constants.L_sun.cgs.value) if f['residuals']['emlines'] is not None else fillvalue for f in alldata])

	obslines['f_hd'] = np.transpose([ret_inf(alldata,'lum',model='obs',name='H$\\delta$'),
		                             ret_inf(alldata,'lum_errup',model='obs',name='H$\\delta$'),
		                             ret_inf(alldata,'lum_errdown',model='obs',name='H$\\delta$')]) / constants.L_sun.cgs.value
	obslines['err_hd'] = (obslines['f_hd'][:,1] - obslines['f_hd'][:,2])/2.

	obslines['f_nii'] = np.transpose([ret_inf(alldata,'lum',model='obs',name='[NII] 6583'),
		                              ret_inf(alldata,'lum_errup',model='obs',name='[NII] 6583'),
		                              ret_inf(alldata,'lum_errdown',model='obs',name='[NII] 6583')]) / constants.L_sun.cgs.value

	obslines['err_nii'] = (obslines['f_nii'][:,1] - obslines['f_nii'][:,2])/2.
	obslines['eqw_nii'] = obslines['f_nii'] / obslines['ha_obs_cont'][:,None]
	obslines['eqw_err_nii'] = obslines['err_nii'] / obslines['ha_obs_cont'][:,None]

	# sum [OIII] lines
	obslines['f_oiii'] = np.transpose([ret_inf(alldata,'lum',model='obs',name='[OIII] 5007'),
		                             ret_inf(alldata,'lum_errup',model='obs',name='[OIII] 5007'),
		                             ret_inf(alldata,'lum_errdown',model='obs',name='[OIII] 5007')])  / constants.L_sun.cgs.value
	obslines['err_oiii'] = (obslines['f_oiii'][:,1] - obslines['f_oiii'][:,2])/2.
	obslines['eqw_oiii'] = obslines['f_oiii'] / obslines['hb_obs_cont'][:,None]
	obslines['eqw_err_oiii'] = obslines['err_oiii'] / obslines['hb_obs_cont'][:,None]

	##### SIGNAL TO NOISE AND EQW CUTS
	# cuts
	# obslines sncut is 1.0 for now, because otherwise NGC 4473 is in the sample, 
	# which has S/N ~ 0.95 for Halpha (clearly a crap detection)  12/10/15
	obslines['sn_cut'] = 5.0
	obslines['eqw_cut'] = 0.0
	obslines['hdelta_sn_cut'] = 5.0
	obslines['hdelta_eqw_cut'] = 1.0

	####### Dn4000, obs + MAGPHYS
	obslines['dn4000'] = ret_inf(alldata,'dn4000',model='obs')
	#mag['dn4000'] = ret_inf(alldata,'Dn4000',model='MAGPHYS')

	####### model absorption properties (Prospector: marginalized + best-fit
	# which index should we use; wide or narrow?
	index_flags = np.loadtxt('/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/hdelta_index.txt', dtype = {'names':('name','flag'),'formats':('S40','i4')})
	anames = alldata[0]['spec_info']['absnames']
	
	ha_lum_pro_marg,ha_eqw_pro_marg,\
	hb_lum_pro_marg,hb_eqw_pro_marg = [np.zeros(shape=(ngals,3)) for i in xrange(4)]

	ha_lum_pro, ha_eqw_pro, \
	hb_lum_pro, hb_eqw_pro = [np.zeros(ngals) for i in xrange(4)]

	for kk, dat in enumerate(alldata):

		#### hbeta, halpha marginalized
		hb_lum_pro_marg[kk,:] = np.log10([dat['spec_info']['flux']['q50'][anames == 'hbeta'],
						            	  dat['spec_info']['flux']['q84'][anames == 'hbeta'],
						            	  dat['spec_info']['flux']['q16'][anames == 'hbeta']])[:,0]
		hb_eqw_pro_marg[kk,:] = np.log10([dat['spec_info']['eqw']['q50'][anames == 'hbeta'],
						            	  dat['spec_info']['eqw']['q84'][anames == 'hbeta'],
						           		  dat['spec_info']['eqw']['q16'][anames == 'hbeta']])[:,0]

		match = index_flags['name'] == dat['objname'].replace(' ','')
		flag = index_flags['flag'][match]
		
		if flag == 1: halph_ind = 'halpha_narrow'
		if flag == 0: halph_ind = 'halpha_wide'

		ha_lum_pro_marg[kk,:] = np.log10([dat['spec_info']['flux']['q50'][anames == halph_ind],
						           	      dat['spec_info']['flux']['q84'][anames == halph_ind],
						            	  dat['spec_info']['flux']['q16'][anames == halph_ind]])[:,0]
		ha_eqw_pro_marg[kk,:] = np.log10([dat['spec_info']['eqw']['q50'][anames == halph_ind],
						            	  dat['spec_info']['eqw']['q84'][anames == halph_ind],
						            	  dat['spec_info']['eqw']['q16'][anames == halph_ind]])[:,0]

		#### are there useful spectra?
		if dat['residuals']['emlines'] is None:
			continue

		#### best-fit
		hb_ind = dat['residuals']['emlines']['mod']['balmer_names'] == 'hbeta'
		hb_lum_pro[kk] = dat['residuals']['emlines']['mod']['balmer_lum'][hb_ind]
		hb_eqw_pro[kk] = dat['residuals']['emlines']['mod']['balmer_eqw_rest'][hb_ind]

		ha_ind = dat['residuals']['emlines']['mod']['balmer_names'] == halph_ind
		ha_lum_pro[kk] = dat['residuals']['emlines']['mod']['balmer_lum'][ha_ind]
		ha_eqw_pro[kk] = dat['residuals']['emlines']['mod']['balmer_eqw_rest'][ha_ind]

	# save marginalized halpha, hbeta
	prosp['hbeta_abs_marg'] = hb_lum_pro_marg
	prosp['hbeta_eqw_marg'] = hb_eqw_pro_marg
	prosp['halpha_abs_marg'] = ha_lum_pro_marg
	prosp['halpha_eqw_marg'] = ha_eqw_pro_marg

	# save best-fit halpha, hbeta
	prosp['hbeta_abs'] = hb_lum_pro
	prosp['hbeta_eqw'] = hb_eqw_pro
	prosp['halpha_abs'] = ha_lum_pro
	prosp['halpha_eqw'] = ha_eqw_pro

	##### add Halpha, Hbeta absorption to errors
	if add_abs_err:
		# add 25% of Balmer absorption value to 1 sigma errors
		hdel_scatter = 0.25
		halpha_corr = hdel_scatter*(10**prosp['halpha_abs_marg'][:,0])
		hbeta_corr = hdel_scatter*(10**prosp['hbeta_abs_marg'][:,0])

		# keep this around, for balmer decrement error plot
		obslines['err_ha_orig'] = copy.copy(obslines['err_ha'])
		obslines['err_hb_orig'] = copy.copy(obslines['err_hb'])
		obslines['f_ha_orig'] = copy.copy(obslines['f_ha'])
		obslines['f_hb_orig'] = copy.copy(obslines['f_hb'])

		# these are 'cosmetic' errors for S/N cuts, also used for Balmer decrements
		obslines['err_ha'] = np.sqrt(obslines['err_ha']**2 + halpha_corr**2)
		obslines['err_hb'] = np.sqrt(obslines['err_hb']**2 + hbeta_corr**2)

		# these are 'true' errors
		# we add in quadrature in up/down errors, which is probably wrong in detail
		obslines['f_ha'][:,1] = obslines['f_ha'][:,0] + np.sqrt((obslines['f_ha'][:,1] - obslines['f_ha'][:,0])**2+halpha_corr**2)
		obslines['f_ha'][:,2] = obslines['f_ha'][:,0] - np.sqrt((obslines['f_ha'][:,2] - obslines['f_ha'][:,0])**2+hbeta_corr**2)

		obslines['f_hb'][:,1] = obslines['f_hb'][:,0] + np.sqrt((obslines['f_hb'][:,1] - obslines['f_hb'][:,0])**2+halpha_corr**2)
		obslines['f_hb'][:,2] = obslines['f_hb'][:,0] - np.sqrt((obslines['f_hb'][:,2] - obslines['f_hb'][:,0])**2+hbeta_corr**2)
	else:
		obslines['err_ha_orig'] = copy.copy(obslines['err_ha'])
		obslines['err_hb_orig'] = copy.copy(obslines['err_hb'])
		obslines['f_ha_orig'] = copy.copy(obslines['f_ha'])
		obslines['f_hb_orig'] = copy.copy(obslines['f_hb'])


	##### Balmer series emission line EQWs
	# here so that the error adjustment above is propagated into EQWs
	obslines['eqw_ha'] = obslines['f_ha'] / obslines['ha_obs_cont'][:,None]
	obslines['eqw_err_ha'] = obslines['err_ha'] / obslines['ha_obs_cont'][:,None]

	obslines['eqw_hb'] = obslines['f_hb'] / obslines['hb_obs_cont'][:,None]
	obslines['eqw_err_hb'] = obslines['err_hb'] / obslines['hb_obs_cont'][:,None]

	####### hdelta absorption properties, for all
	# Prospector is marginalized PLUS best-fit, data is bootstrapped.
	# also record emline fill-in
	# which index should we use; wide or narrow?
	hd_lum_prosp_marg,hd_eqw_prosp_marg, \
	hd_lum_obs, hd_eqw_obs = [np.zeros(shape=(ngals,3)) for i in xrange(4)]
	hd_eqw_obs_chain = np.zeros(shape=(ngals,100))

	hd_lum_prosp, hd_eqw_prosp, \
	hd_lum_eline_prosp, hd_eqw_eline_prosp = [np.zeros(ngals) for i in xrange(4)]
	for kk, dat in enumerate(alldata):
		
		# figure out index
		match = index_flags['name'] == dat['objname'].replace(' ','')
		flag = index_flags['flag'][match]
		if flag == 1: hdelta_ind = 'hdelta_narrow'
		if flag == 0: hdelta_ind = 'hdelta_wide'

		#### Prospector marginalized
		hd_lum_prosp_marg[kk,:] = [dat['spec_info']['flux']['q50'][anames == hdelta_ind],
									dat['spec_info']['flux']['q84'][anames == hdelta_ind],
									dat['spec_info']['flux']['q16'][anames == hdelta_ind]]
		hd_eqw_prosp_marg[kk,:] = [dat['spec_info']['eqw']['q50'][anames == hdelta_ind],
								   dat['spec_info']['eqw']['q84'][anames == hdelta_ind],
								   dat['spec_info']['eqw']['q16'][anames == hdelta_ind]]

		#### are there useful spectra?
		if dat['residuals']['emlines'] is None:
			continue

		#### best-fit, Prospector. also best-fit + emission lines
		hd_ind = dat['residuals']['emlines']['mod']['balmer_names'] == hdelta_ind
		hd_lum_prosp[kk] = dat['residuals']['emlines']['mod']['balmer_lum'][hd_ind]
		hd_eqw_prosp[kk] = dat['residuals']['emlines']['mod']['balmer_eqw_rest'][hd_ind]
		hd_lum_eline_prosp[kk] = dat['residuals']['emlines']['mod']['balmer_lum_addem'][hd_ind]
		hd_eqw_eline_prosp[kk] = dat['residuals']['emlines']['mod']['balmer_eqw_rest_addem'][hd_ind]

		#### observed
		hd_lum_obs[kk,:] = dat['residuals']['emlines']['obs']['balmer_lum'][hd_ind,:]
		hd_eqw_obs[kk,:] = dat['residuals']['emlines']['obs']['balmer_eqw_rest'][hd_ind,:]
		hd_eqw_obs_chain[kk,:] = dat['residuals']['emlines']['obs']['balmer_eqw_rest_chain'][hd_ind,:]

	##### hdelta absorption
	prosp['hdel_abs'] = hd_lum_prosp
	prosp['hdel_eqw'] = hd_eqw_prosp
	prosp['hdel_abs_marg'] = hd_lum_prosp_marg
	prosp['hdel_eqw_marg'] = hd_eqw_prosp_marg
	prosp['hdel_abs_addem'] = hd_lum_eline_prosp
	prosp['hdel_eqw_addem'] = hd_eqw_eline_prosp

	obslines['hdel'] = hd_lum_obs
	obslines['hdel_err'] = (obslines['hdel'][:,1] - obslines['hdel'][:,2]) / 2.
	obslines['hdel_eqw'] = hd_eqw_obs
	obslines['hdel_eqw_err'] = (obslines['hdel_eqw'][:,1] - obslines['hdel_eqw'][:,2])/2.
	obslines['hdel_eqw_chain'] = hd_eqw_obs_chain

	##### names
	objnames = np.array([f['objname'] for f in alldata])

	####### calculate observed emission line ratios, propagate errors
	# Balmer decrement, OIII / Hb, NII / Ha
	# assuming independent variables (mostly true)
	# really should calculate in bootstrapping procedure
	obslines['bdec'] = obslines['f_ha'][:,0] / obslines['f_hb'][:,0]
	obslines['bdec_err'] = obslines['bdec'] * np.sqrt((obslines['err_ha']/obslines['f_ha'][:,0])**2+(obslines['err_hb']/obslines['f_hb'][:,0])**2)
	obslines['oiii_hb'] = obslines['f_oiii'][:,0] / obslines['f_hb'][:,0]
	obslines['oiii_hb_err'] = obslines['oiii_hb'] * np.sqrt((obslines['err_oiii']/obslines['f_oiii'][:,0])**2+(obslines['err_hb']/obslines['f_hb'][:,0])**2)
	obslines['nii_ha'] = obslines['f_nii'][:,0] / obslines['f_ha'][:,0]
	obslines['nii_ha_err'] = obslines['nii_ha'] * np.sqrt((obslines['err_nii']/obslines['f_nii'][:,0])**2+(obslines['err_ha']/obslines['f_ha'][:,0])**2)

	##### NAME VARIABLES
	# Prospector model variables
	parnames = alldata[0]['pquantiles']['parnames']
	logmass_idx = parnames == 'logmass'
	dinx_idx = parnames == 'dust_index'
	dust1_idx = parnames == 'dust1'
	dust2_idx = parnames == 'dust2'
	met_idx = parnames == 'logzsol'

	slope_idx = parnames == 'sf_tanslope'
	trunc_idx = parnames == 'delt_trunc'
	tage_idx = parnames == 'tage'

	# Prospector extra variables
	parnames = alldata[0]['pextras']['parnames']
	bcalc_idx = parnames == 'bdec_calc'
	bcloud_idx = parnames == 'bdec_cloudy'
	emp_ha_idx = parnames == 'emp_ha'
	sfr_10_idx_p = parnames == 'sfr_10'
	sfr_100_idx_p = parnames == 'sfr_100'

	# Prospect emission line variables
	linenames = alldata[0]['model_emline']['emnames']
	ha_em = linenames == 'Halpha'
	hb_em = linenames == 'Hbeta'
	hd_em = linenames == 'Hdelta'
	oiii_em = linenames == '[OIII]2'
	nii_em = linenames == '[NII]'

	# MAGPHYS variables
	mparnames = alldata[0]['model']['parnames']
	mu_idx = mparnames == 'mu'
	tauv_idx = mparnames == 'tauv'

	# magphys full
	mparnames = alldata[0]['model']['full_parnames']
	sfr_10_idx = mparnames == 'SFR_10'
	mmet_idx = mparnames == 'Z/Zo'
	mmass_idx = mparnames == 'M*/Msun'
	msfr_100_idx = mparnames == 'SFR(1e8)'

	#### calculate expected Balmer decrement for Prospector, MAGPHYS
	# best-fits + marginalized
	bdec_cloudy_bfit,bdec_calc_bfit,bdec_magphys, ha_magphys, sfr_10_mag, sfr_100_mag, \
	ha_ext_mag, met_mag = [np.zeros(ngals) for i in xrange(8)]
	
	bdec_cloudy_marg, bdec_calc_marg, cloudy_ha, cloudy_hb, cloudy_hd, \
	cloudy_nii, cloudy_oiii, ha_emp, pmet, ha_ratio, oiii_hb, \
	nii_ha, dn4000, d1, d2, didx,sfr_10,sfr_100,ha_ext,sfr_100_mag_marginalized,\
	cloudy_ha_eqw, cloudy_hb_eqw, cloudy_oiii_eqw, cloudy_nii_eqw, \
	cloudy_hd_eqw, d1_d2,mass, dtau_dlam_prosp = [np.zeros(shape=(ngals,3)) for i in xrange(28)]
	for ii,dat in enumerate(np.array(alldata)):

		####### BALMER DECREMENTS
		### best-fit calculated balmer decrement
		bdec_calc_bfit[ii] = dat['bfit']['bdec_calc']

		### best-fit CLOUDY balmer decrement
		bdec_cloudy_bfit[ii] = dat['bfit']['bdec_cloudy']

		#### marginalized CLOUDY balmer decrement
		bdec_cloudy_marg[ii,0] = dat['pextras']['q50'][bcloud_idx]
		bdec_cloudy_marg[ii,1] = dat['pextras']['q84'][bcloud_idx]
		bdec_cloudy_marg[ii,2] = dat['pextras']['q16'][bcloud_idx]

		# marginalized calculated balmer decrement
		bdec_calc_marg[ii,0] = dat['pextras']['q50'][bcalc_idx]
		bdec_calc_marg[ii,1] = dat['pextras']['q84'][bcalc_idx]
		bdec_calc_marg[ii,2] = dat['pextras']['q16'][bcalc_idx]

		### if we can't generate emission lines due to age, use the marginalized version
		if np.sum(np.isfinite(bdec_cloudy_marg[ii,:])) != 3: bdec_cloudy_marg[ii,:] = bdec_calc_marg[ii,:]

		#MAGPHYS balmer decrement
		tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
		tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
		bdec = threed_dutils.calc_balmer_dec(tau1, tau2, -1.3, -0.7)
		bdec_magphys[ii] = np.squeeze(bdec)

		#### CLOUDY emission lines
		cloudy_ha[ii,0] = dat['model_emline']['flux']['q50'][ha_em]
		cloudy_ha[ii,1] = dat['model_emline']['flux']['q84'][ha_em]
		cloudy_ha[ii,2] = dat['model_emline']['flux']['q16'][ha_em]

		cloudy_hb[ii,0] = dat['model_emline']['flux']['q50'][hb_em]
		cloudy_hb[ii,1] = dat['model_emline']['flux']['q84'][hb_em]
		cloudy_hb[ii,2] = dat['model_emline']['flux']['q16'][hb_em]

		cloudy_hd[ii,0] = dat['model_emline']['flux']['q50'][hd_em]
		cloudy_hd[ii,1] = dat['model_emline']['flux']['q84'][hd_em]
		cloudy_hd[ii,2] = dat['model_emline']['flux']['q16'][hd_em]

		cloudy_nii[ii,0] = dat['model_emline']['flux']['q50'][nii_em]
		cloudy_nii[ii,1] = dat['model_emline']['flux']['q84'][nii_em]
		cloudy_nii[ii,2] = dat['model_emline']['flux']['q16'][nii_em]

		cloudy_oiii[ii,0] = dat['model_emline']['flux']['q50'][oiii_em]
		cloudy_oiii[ii,1] = dat['model_emline']['flux']['q84'][oiii_em]
		cloudy_oiii[ii,2] = dat['model_emline']['flux']['q16'][oiii_em]

		cloudy_ha_eqw[ii,0] = dat['model_emline']['eqw']['q50'][ha_em]
		cloudy_ha_eqw[ii,1] = dat['model_emline']['eqw']['q84'][ha_em]
		cloudy_ha_eqw[ii,2] = dat['model_emline']['eqw']['q16'][ha_em]

		cloudy_hb_eqw[ii,0] = dat['model_emline']['eqw']['q50'][hb_em]
		cloudy_hb_eqw[ii,1] = dat['model_emline']['eqw']['q84'][hb_em]
		cloudy_hb_eqw[ii,2] = dat['model_emline']['eqw']['q16'][hb_em]

		cloudy_hd_eqw[ii,0] = dat['model_emline']['eqw']['q50'][hd_em]
		cloudy_hd_eqw[ii,1] = dat['model_emline']['eqw']['q84'][hd_em]
		cloudy_hd_eqw[ii,2] = dat['model_emline']['eqw']['q16'][hd_em]

		cloudy_nii_eqw[ii,0] = dat['model_emline']['eqw']['q50'][nii_em]
		cloudy_nii_eqw[ii,1] = dat['model_emline']['eqw']['q84'][nii_em]
		cloudy_nii_eqw[ii,2] = dat['model_emline']['eqw']['q16'][nii_em]

		cloudy_oiii_eqw[ii,0] = dat['model_emline']['eqw']['q50'][oiii_em]
		cloudy_oiii_eqw[ii,1] = dat['model_emline']['eqw']['q84'][oiii_em]
		cloudy_oiii_eqw[ii,2] = dat['model_emline']['eqw']['q16'][oiii_em]

		#### Empirical emission lines
		ha_emp[ii,0] = dat['pextras']['q50'][emp_ha_idx] / constants.L_sun.cgs.value
		ha_emp[ii,1] = dat['pextras']['q84'][emp_ha_idx] / constants.L_sun.cgs.value
		ha_emp[ii,2] = dat['pextras']['q16'][emp_ha_idx] / constants.L_sun.cgs.value

		###### best-fit MAGPHYS Halpha
		sfr_10_mag[ii] = dat['model']['full_parameters'][sfr_10_idx]
		sfr_100_mag[ii] = dat['model']['full_parameters'][msfr_100_idx] * dat['model']['full_parameters'][mmass_idx]
		sfr_100_mag_marginalized[ii] = dat['magphys']['percentiles']['SFR'][1:4]
		tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
		tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
		ha_magphys[ii] = threed_dutils.synthetic_halpha(sfr_10_mag[ii], tau1, tau2, -1.3, -0.7) / constants.L_sun.cgs.value
		ha_ext_mag[ii] = threed_dutils.charlot_and_fall_extinction(6563.0,tau1,tau2,-1.3,-0.7,kriek=False)
		met_mag[ii] = dat['model']['full_parameters'][mmet_idx]

		##### CLOUDY Halpha / empirical halpha, chain calculation
		ratio = np.log10(dat['model_emline']['flux']['chain'][:,ha_em]*constants.L_sun.cgs.value / dat['pextras']['flatchain'][:,emp_ha_idx])
		ha_ratio[ii,:] = corner.quantile(ratio, [0.5, 0.84, 0.16])

		##### BPT information
		ratio = np.log10(dat['model_emline']['flux']['chain'][:,oiii_em] / dat['model_emline']['flux']['chain'][:,hb_em])
		oiii_hb[ii,:] = corner.quantile(ratio, [0.5, 0.84, 0.16])
		try:
			ratio = np.log10(dat['model_emline']['flux']['chain'][:,nii_em] / dat['model_emline']['flux']['chain'][:,ha_em])
			nii_ha[ii,:] = corner.quantile(ratio, [0.5, 0.84, 0.16])
		except ValueError as e:
			pass

		##### marginalized metallicity
		pmet[ii,0] = dat['pquantiles']['q50'][met_idx]
		pmet[ii,1] = dat['pquantiles']['q84'][met_idx]
		pmet[ii,2] = dat['pquantiles']['q16'][met_idx]

		##### marginalized dn4000
		dn4000[ii,0] = dat['spec_info']['dn4000']['q50']
		dn4000[ii,1] = dat['spec_info']['dn4000']['q84']
		dn4000[ii,2] = dat['spec_info']['dn4000']['q16']

		##### marginalized dust properties + SFR + mass
		d1[ii,0] = dat['pquantiles']['q50'][dust1_idx]
		d1[ii,1] = dat['pquantiles']['q84'][dust1_idx]
		d1[ii,2] = dat['pquantiles']['q16'][dust1_idx]

		mass[ii,0] = 10**dat['pquantiles']['q50'][logmass_idx]
		mass[ii,1] = 10**dat['pquantiles']['q84'][logmass_idx]
		mass[ii,2] = 10**dat['pquantiles']['q16'][logmass_idx]

		d2[ii,0] = dat['pquantiles']['q50'][dust2_idx]
		d2[ii,1] = dat['pquantiles']['q84'][dust2_idx]
		d2[ii,2] = dat['pquantiles']['q16'][dust2_idx]

		didx[ii,0] = dat['pquantiles']['q50'][dinx_idx]
		didx[ii,1] = dat['pquantiles']['q84'][dinx_idx]
		didx[ii,2] = dat['pquantiles']['q16'][dinx_idx]

		ratio = dat['pquantiles']['random_chain'][:,dust1_idx] / dat['pquantiles']['random_chain'][:,dust2_idx]
		ratio[~np.isfinite(ratio)] = 2.0
		d1_d2[ii,:] = corner.quantile(ratio, [0.5, 0.84, 0.16])

		sfr_10[ii,0] = dat['pextras']['q50'][sfr_10_idx_p]
		sfr_10[ii,1] = dat['pextras']['q84'][sfr_10_idx_p]
		sfr_10[ii,2] = dat['pextras']['q16'][sfr_10_idx_p]

		sfr_100[ii,0] = dat['pextras']['q50'][sfr_100_idx_p]
		sfr_100[ii,1] = dat['pextras']['q84'][sfr_100_idx_p]
		sfr_100[ii,2] = dat['pextras']['q16'][sfr_100_idx_p]

		##### marginalized extinction at Halpha wavelengths
		d1_chain = dat['pquantiles']['random_chain'][:,dust1_idx]
		d2_chain = dat['pquantiles']['random_chain'][:,dust2_idx]
		didx_chain = dat['pquantiles']['random_chain'][:,dinx_idx]
		ha_ext_chain = threed_dutils.charlot_and_fall_extinction(6563.0,d1_chain,d2_chain,-1.0,didx_chain,kriek=False)
		ha_ext[ii,:] = corner.quantile(ha_ext_chain, [0.5, 0.84, 0.16])

		#### calculate dtau / dlambda (tau_0) (lambda = 5500 angstroms)
		tau1 = -np.log(threed_dutils.charlot_and_fall_extinction(lam1_extdiff,np.zeros_like(d2_chain),d2_chain,np.zeros_like(d2_chain),didx_chain, kriek=True,nobc=True))
		tau2 = -np.log(threed_dutils.charlot_and_fall_extinction(lam2_extdiff,np.zeros_like(d2_chain),d2_chain,np.zeros_like(d2_chain),didx_chain, kriek=True,nobc=True))
		dtau_dlam_chain = ((tau2-tau1) / (lam2_extdiff-lam1_extdiff)) / d2_chain
		dtau_dlam_prosp[ii,:] = corner.quantile(dtau_dlam_chain, [0.5, 0.84, 0.16])


	prosp['bdec_cloudy_bfit'] = bdec_cloudy_bfit
	prosp['bdec_calc_bfit'] = bdec_calc_bfit
	prosp['bdec_cloudy_marg'] = bdec_cloudy_marg
	prosp['bdec_calc_marg'] = bdec_calc_marg
	prosp['cloudy_ha'] = cloudy_ha
	prosp['cloudy_ha_eqw'] = cloudy_ha_eqw
	prosp['cloudy_hb'] = cloudy_hb
	prosp['cloudy_hb_eqw'] = cloudy_hb_eqw
	prosp['cloudy_hd'] = cloudy_hd
	prosp['cloudy_hd_eqw'] = cloudy_hd_eqw
	prosp['cloudy_nii'] = cloudy_nii
	prosp['cloudy_nii_eqw'] = cloudy_nii_eqw
	prosp['cloudy_oiii'] = cloudy_oiii
	prosp['cloudy_oiii_eqw'] = cloudy_oiii_eqw
	prosp['oiii_hb'] = oiii_hb
	prosp['nii_ha'] = nii_ha
	prosp['ha_emp'] = ha_emp
	prosp['ha_emp_eqw'] = ha_emp / (cloudy_ha[:,0]/cloudy_ha_eqw[:,0])[:,None]
	prosp['ha_ratio'] = ha_ratio
	prosp['met'] = pmet
	prosp['dn4000'] = dn4000
	prosp['d1'] = d1
	prosp['d2'] = d2
	prosp['d1_d2'] = d1_d2
	prosp['didx'] = didx
	prosp['sfr_10'] = sfr_10
	prosp['sfr_100'] = sfr_100
	prosp['ha_ext'] = ha_ext
	prosp['mass'] = mass
	prosp['dtau_dlam'] = dtau_dlam_prosp

	mag['bdec'] = bdec_magphys
	mag['ha'] = ha_magphys
	mag['ha_eqw'] = ha_magphys
	mag['sfr_10'] = sfr_10_mag
	mag['sfr_100'] = sfr_100_mag
	mag['sfr_100_marginalized'] = sfr_100_mag_marginalized
	mag['ha_ext'] = ha_ext_mag
	mag['met'] = np.log10(met_mag)

	##### INCLINATION
	# match to local galaxies
	# full catalog in a .fits format is also available in same directory
	inc_fldr = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/inclination/'
	inc = np.loadtxt(inc_fldr+'moustakas06_inclination.dat', {'names':('name','inclination'), 'formats':('S20','f16')},comments='#')
	inclination = np.zeros(ngals)
	for kk, dat in enumerate(alldata):

		# figure out index 
		match = np.core.defchararray.upper(inc['name']) == dat['objname'].replace(' ','').upper()
		if np.sum(match) == 1:
			inclination[kk] = inc['inclination'][match]
		else:
			inclination[kk] = np.nan

	obslines['inclination'] = inclination

	##### ASSEMBLE OUTPUT
	eline_info = {'obs': obslines, 'mag': mag, 'prosp': prosp, 'objnames':objnames}

	return eline_info

def atlas_3d_met(e_pinfo,hflag,outfolder=''):

	prosp_met, prosp_met_err, a3d_met, a3d_met_err, a3d_alpha, obj_idx = brown_quality_cuts.load_atlas3d(e_pinfo)

	fig, ax = plt.subplots(1,1,figsize=(7,7))
	fig.subplots_adjust(left=0.15,bottom=0.1,top=0.95,right=0.95)
	ax.errorbar(a3d_met,prosp_met,xerr=a3d_met_err,yerr=prosp_met_err, color='#1C86EE',alpha=0.9,fmt='o')
	ax.set_xlabel('log(Z$_{\mathrm{ATLAS-3D}}$/Z$_{\odot}$)')
	ax.set_ylabel('log(Z$_{\mathrm{Prosp}}$/Z$_{\odot}$)')

	ax = threed_dutils.equalize_axes(ax,a3d_met+0.1,prosp_met-0.1, dynrange=0.1, line_of_equality=True, log=False)

	off,scat = threed_dutils.offset_and_scatter(a3d_met,prosp_met, biweight=True)
	ax.text(0.03,0.95, r'N = '+str(prosp_met.shape[0]), transform = ax.transAxes,horizontalalignment='left')
	ax.text(0.03,0.9, 'biweight scatter='+"{:.2f}".format(scat) +' dex', transform = ax.transAxes,horizontalalignment='left')
	ax.text(0.03,0.85, 'mean offset='+"{:.2f}".format(off)+ ' dex', transform = ax.transAxes,horizontalalignment='left')

	plt.savefig(outfolder+'atlas3d_starmet.png',dpi=150)
	plt.close()


def gas_phase_metallicity(e_pinfo, hflag, outfolder='',ssfr_cut=False):

	#### cuts first
	sn_ha = e_pinfo['obs']['f_ha'][:,0] / np.abs(e_pinfo['obs']['err_ha'])
	sn_hb = e_pinfo['obs']['f_hb'][:,0] / np.abs(e_pinfo['obs']['err_hb'])
	sn_oiii = e_pinfo['obs']['f_oiii'][:,0] / np.abs(e_pinfo['obs']['err_oiii'])
	sn_nii = e_pinfo['obs']['f_nii'][:,0] / np.abs(e_pinfo['obs']['err_nii'])
	ssfr = e_pinfo['prosp']['sfr_100'][:,0] / e_pinfo['prosp']['mass'][:,0]

	eqw_ha = e_pinfo['obs']['eqw_ha'][:,0]

	sn_cut = 3
	if ssfr_cut:
		keep_idx = np.squeeze((sn_ha > sn_cut) & (sn_hb > sn_cut) & (sn_oiii > sn_cut) & (sn_nii > sn_cut) & (ssfr > 3e-10))
	else:
		keep_idx = np.squeeze((sn_ha > sn_cut) & (sn_hb > sn_cut) & (sn_oiii > sn_cut) & (sn_nii > sn_cut))

	#### get BPT status
	sfing, composite, agn = return_agn_str(keep_idx)
	keys = [sfing, composite, agn]

	#### get gas-phase metallicity
	nii_ha = e_pinfo['obs']['f_nii'][keep_idx,0] / e_pinfo['obs']['f_ha'][keep_idx,0]
	oiii_hb = e_pinfo['obs']['f_oiii'][keep_idx,0] / e_pinfo['obs']['f_hb'][keep_idx,0]
	logzgas = np.zeros_like(nii_ha)
	for ii in xrange(len(logzgas)): logzgas[ii] = threed_dutils.gas_met(nii_ha[ii],oiii_hb[ii])

	#### get stellar metallicity
	logzsol = e_pinfo['prosp']['met'][keep_idx,0]

	fig, ax = plt.subplots(1,1,figsize=(7,7))
	fig.subplots_adjust(left=0.15,bottom=0.1,top=0.95,right=0.95)
	for ii,key in enumerate(keys): ax.plot(logzgas[key],logzsol[key],'o',**bptdict[ii])
	ax.set_xlabel('log(Z$_{\mathrm{gas}}$/Z$_{\odot}$)')
	ax.set_ylabel('log(Z$_{*}$/Z$_{\odot}$)')

	ax = threed_dutils.equalize_axes(ax, logzgas,logzsol, dynrange=0.1, line_of_equality=True, log=False)

	ax.text(0.03,0.93, r'S/N (H$\alpha$,H$\beta$,[OIII],[NII]) > '+str(int(sn_cut)), transform = ax.transAxes,horizontalalignment='left')
	ax.text(0.03,0.87, r'N = '+str(np.sum(keep_idx)), transform = ax.transAxes,horizontalalignment='left')

	if ssfr_cut:
		ax.text(0.03,0.81, 'sSFR > 3e-10', transform = ax.transAxes,horizontalalignment='left')
		plt.savefig(outfolder+'gas_to_stellar_metallicity_highssfr.png',dpi=150)
	else:
		plt.savefig(outfolder+'gas_to_stellar_metallicity.png',dpi=150)

	plt.close()

def bpt_diagram(e_pinfo,hflag,outname=None):

	########################################
	## plot obs BPT, predicted BPT, resid ##
	########################################
	axlim = (-2.2,0.5,-1.0,1.0)

	# cuts first
	sn_ha = e_pinfo['obs']['f_ha'][:,0] / np.abs(e_pinfo['obs']['err_ha'])
	sn_hb = e_pinfo['obs']['f_hb'][:,0] / np.abs(e_pinfo['obs']['err_hb'])
	sn_oiii = e_pinfo['obs']['f_oiii'][:,0] / np.abs(e_pinfo['obs']['err_oiii'])
	sn_nii = e_pinfo['obs']['f_nii'][:,0] / np.abs(e_pinfo['obs']['err_nii'])

	sn_cut=3.0
	keep_idx = np.squeeze((sn_ha > 3.0) & \
		                  (sn_hb > 3.0) & \
		                  (sn_oiii > 3.0) & \
		                  (sn_nii > 3.0))

	##### CREATE PLOT QUANTITIES
	mod_oiii_hb = e_pinfo['prosp']['oiii_hb'][keep_idx,:]
	mod_nii_ha = e_pinfo['prosp']['nii_ha'][keep_idx,:]

	obs_oiii_hb = e_pinfo['obs']['oiii_hb'][keep_idx]
	obs_oiii_hb_err = e_pinfo['obs']['oiii_hb_err'][keep_idx]
	obs_nii_ha = e_pinfo['obs']['nii_ha'][keep_idx]
	obs_nii_ha_err = e_pinfo['obs']['nii_ha_err'][keep_idx]

	##### AGN identifiers
	sfing, composite, agn = return_agn_str(keep_idx)
	keys = [sfing, composite, agn]

	##### herschel identifier
	hflag = [hflag[keep_idx],~hflag[keep_idx]]

	#### TWO PLOTS
	# first plot: observed BPT
	# second plot: model BPT
	# third plot: residuals versus residuals (obs - mod)
	fig1, ax1 = plt.subplots(1,3, figsize = (18.75,6))

	# loop and put points on plot
	for ii in xrange(len(labels)):
		for kk in xrange(len(hflag)):

			### update colors, define sample
			pdict = merge_dicts(herschdict[kk],bptdict[ii])
			plt_idx = keys[ii] & hflag[kk]

			### errors for this sample
			err_obs_x = threed_dutils.asym_errors(obs_nii_ha[plt_idx], 
		                                           obs_nii_ha[plt_idx]+obs_nii_ha_err[plt_idx],
		                                           obs_nii_ha[plt_idx]-obs_nii_ha_err[plt_idx], log=True)
			err_obs_y = threed_dutils.asym_errors(obs_oiii_hb[plt_idx], 
		                                           obs_oiii_hb[plt_idx]+obs_oiii_hb_err[plt_idx],
		                                           obs_oiii_hb[plt_idx]-obs_oiii_hb_err[plt_idx], log=True)
			err_mod_x = threed_dutils.asym_errors(mod_nii_ha[plt_idx,0], 
		                                           mod_nii_ha[plt_idx,1],
		                                           mod_nii_ha[plt_idx,2], log=False)
			err_mod_y = threed_dutils.asym_errors(mod_oiii_hb[plt_idx,0], 
		                                           mod_oiii_hb[plt_idx,1],
		                                           mod_oiii_hb[plt_idx,2], log=False)

			ax1[0].errorbar(np.log10(obs_nii_ha[plt_idx]), np.log10(obs_oiii_hb[plt_idx]), xerr=err_obs_x, yerr=err_obs_y,
				           linestyle=' ',**pdict)
			ax1[1].errorbar(mod_nii_ha[plt_idx,0], mod_oiii_hb[plt_idx,0], xerr=err_mod_x, yerr=err_mod_y,
				           linestyle=' ',**pdict)
			ax1[2].errorbar(np.log10(obs_nii_ha[plt_idx])-mod_nii_ha[plt_idx,0], 
			                np.log10(obs_oiii_hb[plt_idx])-mod_oiii_hb[plt_idx,0],
				            linestyle=' ',**pdict)

	#### plot bpt line
	# Kewley+06
	# log(OIII/Hbeta) < 0.61 /[log(NII/Ha) - 0.05] + 1.3 (star-forming to the left and below)
	# log(OIII/Hbeta) < 0.61 /[log(NII/Ha) - 0.47] + 1.19 (between AGN and star-forming)
	# x = 0.61 / (y-0.47) + 1.19
	x1 = np.linspace(-2.2,0.0,num=50)
	x2 = np.linspace(-2.2,0.35,num=50)
	for ax in ax1[:2]:
		ax.plot(x1,0.61 / (x1 - 0.05) + 1.3 , linestyle='--',color='0.5')
		ax.plot(x2,0.61 / (x2-0.47) + 1.19, linestyle='--',color='0.5')


	#### determine BPT status based off my measurements
	bpt_flag = np.empty(len(e_pinfo['obs']['f_ha'][:,0]),dtype='|S6')
	bpt_flag_idx = np.empty(np.sum(keep_idx),dtype='|S6')
	bpt_flag_idx[:] = 'SF'
	sf_line1 = 0.61 / (np.log10(obs_nii_ha) - 0.05) + 1.3
	sf_line2 = 0.61 / (np.log10(obs_nii_ha) - 0.47) + 1.19
	composite = (np.log10(obs_oiii_hb) > sf_line1) & (np.log10(obs_oiii_hb) < sf_line2)
	agn = np.log10(obs_oiii_hb) > sf_line2
	bpt_flag_idx[composite] = 'SF/AGN'
	bpt_flag_idx[agn] = 'AGN'
	bpt_flag[:] = 'SF'
	bpt_flag[keep_idx] = bpt_flag_idx
	pickle.dump(bpt_flag,open(os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/joel_bpt.pickle', "wb"))

	ax1[0].text(0.04,0.1, r'S/N (H$\alpha$,H$\beta$,[OIII],[NII]) > '+str(int(e_pinfo['obs']['sn_cut'])), transform = ax1[0].transAxes,horizontalalignment='left')
	ax1[0].set_xlabel(r'log([NII 6583]/H$_{\alpha}$) [observed]')
	ax1[0].set_ylabel(r'log([OIII 5007]/H$_{\beta}$) [observed]')
	ax1[0].axis(axlim)

	ax1[1].set_xlabel(r'log([NII 6583]/H$_{\alpha}$) [Prospector]')
	ax1[1].set_ylabel(r'log([OIII 5007]/H$_{\beta}$) [Prospector]')
	ax1[1].axis(axlim)

	ax1[2].set_xlabel(r'log([NII 6583]/H$_{\alpha}$) [obs - model]')
	ax1[2].set_ylabel(r'log([OIII 5007]/H$_{\beta}$) [obs - model]')

	plt.tight_layout()
	plt.savefig(outname,dpi=dpi)
	plt.close()

def obs_vs_kennicutt_ha(e_pinfo,hflag,outname_prosp='test.png',outname_mag='testmag.png',
					    outname_cloudy='test_cloudy.png',
						outname_ha_inpt='test_ha_inpt.png',
						outname_sfr_margcomp='test_sfr_margcomp.png',
	                    standardized_ha_axlim = True, eqw=False):
	
	#################
	#### plot observed Halpha versus model Halpha from Kennicutt relationship
	#################

	keep_idx = brown_quality_cuts.halpha_cuts(e_pinfo)

	##### create plot quantities
	if eqw:
		pl_ha_mag = minlog(e_pinfo['mag']['ha_eqw'][keep_idx])
		pl_ha_obs = minlog(e_pinfo['obs']['eqw_ha'][keep_idx,:])
		pl_ha_emp = minlog(e_pinfo['prosp']['ha_emp_eqw'][keep_idx,:]) 
		pl_ha_cloudy = minlog(e_pinfo['prosp']['cloudy_ha_eqw'][keep_idx,:])
		xlab_ha = [r'log(H$_{\alpha}$ EQW) [observed]',
		           r'log(H$_{\alpha}$ EQW) [observed]',
		           r'log(CLOUDY H$_{\alpha}$ EQW) [Prospector]']
		ylab_ha = [r'log(Kennicutt H$_{\alpha}$ EQW) [Prospector]',
		           r'log(Kennicutt H$_{\alpha}$ EQW) [MAGPHYS]',
		           r'log(Kennicutt H$_{\alpha}$ EQW) [Prospector]']
		ha_lim = ha_eqw_lim
	else:
		pl_ha_mag = minlog(e_pinfo['mag']['ha'][keep_idx])
		pl_ha_obs = minlog(e_pinfo['obs']['f_ha'][keep_idx,:])
		pl_ha_emp = minlog(e_pinfo['prosp']['ha_emp'][keep_idx,:]) 
		pl_ha_cloudy = minlog(e_pinfo['prosp']['cloudy_ha'][keep_idx,:]) 
		xlab_ha = [r'log(H$_{\alpha}$) [observed]',
		           r'log(H$_{\alpha}$) [observed]',
		           r'log(CLOUDY H$_{\alpha}$) [Prospector]']
		ylab_ha = [r'log(Kennicutt H$_{\alpha}$) [Prospector]',
		           r'log(Kennicutt H$_{\alpha}$) [MAGPHYS]',
		           r'log(Kennicutt H$_{\alpha}$) [Prospector]']
		ha_lim = ha_flux_lim

	pmet = e_pinfo['prosp']['met'][keep_idx,:]
	pl_ha_ratio = e_pinfo['prosp']['ha_ratio'][keep_idx,:]

	#### plot3+plot4 quantities
	mmet = e_pinfo['mag']['met'][keep_idx]
	msfr10 = np.log10(np.clip(e_pinfo['mag']['sfr_10'][keep_idx],minsfr,np.inf))
	msfr100 = np.log10(np.clip(e_pinfo['mag']['sfr_100'][keep_idx],minsfr,np.inf))
	msfr100_marginalized = np.log10(np.clip(10**e_pinfo['mag']['sfr_100_marginalized'][keep_idx,:],minsfr,np.inf))
	mha_ext = np.log10(1./e_pinfo['mag']['ha_ext'][keep_idx])
	ha_ext = np.log10(1./e_pinfo['prosp']['ha_ext'][keep_idx,:])
	sfr10 = np.log10(np.clip(e_pinfo['prosp']['sfr_10'][keep_idx,:],minsfr,np.inf))
	sfr100 = np.log10(np.clip(e_pinfo['prosp']['sfr_100'][keep_idx,:],minsfr,np.inf))

	#### fit the ha_ratio -- metallicity relationship
	fit_and_save(pmet,pl_ha_ratio)

	##### AGN identifiers
	sfing, composite, agn = return_agn_str(keep_idx)
	keys = [sfing, composite, agn]

	##### herschel identifier
	hflag = [hflag[keep_idx],~hflag[keep_idx]]

	#### THREE PLOTS
	# first plot: (obs v prosp) Kennicutt, (obs v mag) Kennicutt
	# second plot: (kennicutt v CLOUDY), (kennicutt/cloudy v met)
	# third plot: (mag SFR10 v Prosp SFR10), (mag ext(ha) v Prosp ext(ha)), (mag met v Prosp met)
	# fourth plot: (mag SFR100 v Prosp SFR100), (mag SFR10 v Prosp SFR10)
	fig1, ax1 = plt.subplots(1,1, figsize = (6,6))
	figmag, axmag = plt.subplots(1,1, figsize = (6,6))
	fig2, ax2 = plt.subplots(1,1, figsize = (6,6))
	fig3, ax3 = plt.subplots(1,2, figsize = (12.5,6))
	fig4, ax4 = plt.subplots(1,2, figsize = (12.5,6))

	for ii in xrange(len(labels)):
		for kk in xrange(len(hflag)):

			### update colors, define region
			pdict = merge_dicts(herschdict[kk],bptdict[ii])
			plt_idx = keys[ii] & hflag[kk]

			### setup errors
			prosp_emp_err = threed_dutils.asym_errors(pl_ha_emp[plt_idx,0],pl_ha_emp[plt_idx,1], pl_ha_emp[plt_idx,2],log=False)
			prosp_cloud_err = threed_dutils.asym_errors(pl_ha_cloudy[plt_idx,0],pl_ha_cloudy[plt_idx,1], pl_ha_cloudy[plt_idx,2],log=False)
			obs_err = threed_dutils.asym_errors(pl_ha_obs[plt_idx,0],pl_ha_obs[plt_idx,1], pl_ha_obs[plt_idx,2],log=False)
			ratio_err = threed_dutils.asym_errors(pl_ha_ratio[plt_idx,0],pl_ha_ratio[plt_idx,1], pl_ha_ratio[plt_idx,2],log=False)
			pmet_err = threed_dutils.asym_errors(pmet[plt_idx,0],pmet[plt_idx,1], pmet[plt_idx,2],log=False)
			ha_ext_err = threed_dutils.asym_errors(ha_ext[plt_idx,0], ha_ext[plt_idx,1], ha_ext[plt_idx,2],log=False)
			sfr10_err = threed_dutils.asym_errors(sfr10[plt_idx,0], sfr10[plt_idx,1], sfr10[plt_idx,2],log=False)
			sfr100_err = threed_dutils.asym_errors(sfr100[plt_idx,0], sfr100[plt_idx,1], sfr100[plt_idx,2],log=False)
			msfr100_err = threed_dutils.asym_errors(msfr100_marginalized[plt_idx,1], msfr100_marginalized[plt_idx,0], msfr100_marginalized[plt_idx,2],log=False)


			ax1.errorbar(pl_ha_obs[plt_idx,0], pl_ha_emp[plt_idx,0], xerr=obs_err, yerr=prosp_emp_err,
				           linestyle=' ',**pdict)
			axmag.errorbar(pl_ha_obs[plt_idx,0], pl_ha_mag[plt_idx], xerr=obs_err,
				           linestyle=' ',**pdict)
			ax2.errorbar(pl_ha_ratio[plt_idx,0],pmet[plt_idx,0],
				           linestyle=' ',**pdict)

			ax3[0].errorbar(sfr10[plt_idx,0], msfr10[plt_idx], xerr=sfr10_err, 
				           linestyle=' ',**pdict)
			ax3[1].errorbar(ha_ext[plt_idx,0], mha_ext[plt_idx], xerr=ha_ext_err, 
				           linestyle=' ',**pdict)

			ax4[0].errorbar(msfr100_marginalized[plt_idx,1], msfr100[plt_idx], xerr=msfr100_err, 
	           linestyle=' ',**pdict)
			ax4[1].errorbar(sfr100[plt_idx,0], msfr100_marginalized[plt_idx,1], xerr=sfr100_err, yerr=msfr100_err,
	           linestyle=' ',**pdict)

	ax1.text(0.04,0.92, r'S/N (H$\alpha$,H$\beta$) > {0}'.format(int(e_pinfo['obs']['sn_cut'])), transform = ax1.transAxes,horizontalalignment='left')
	#ax1.text(0.04,0.92, r'EQW H$\alpha$ > {0} $\AA$'.format(int(e_pinfo['obs']['eqw_cut'])), transform = ax1.transAxes,horizontalalignment='left')
	ax1.text(0.04,0.87, r'N = '+str(int(np.sum(keep_idx))), transform = ax1.transAxes,horizontalalignment='left')
	ax1.set_xlabel(xlab_ha[0])
	ax1.set_ylabel(ylab_ha[0])
	if standardized_ha_axlim:
		ax1.axis((ha_lim[0],ha_lim[1],ha_lim[0],ha_lim[1]))
		ax1.plot(ha_lim,ha_lim,linestyle='--',color='0.1',alpha=0.8)
	else:
		ax1 = threed_dutils.equalize_axes(ax1, pl_ha_obs[:,0], pl_ha_emp[:,0])
	off,scat = threed_dutils.offset_and_scatter(pl_ha_obs[:,0], pl_ha_emp[:,0], biweight=True)
	ax1.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) +' dex', transform = ax1.transAxes,horizontalalignment='right')
	ax1.text(0.96,0.1, 'mean offset='+"{:.2f}".format(off)+ ' dex', transform = ax1.transAxes,horizontalalignment='right')

	axmag.text(0.04,0.92, r'S/N (H$\alpha$,H$\beta$) > {0}'.format(int(e_pinfo['obs']['sn_cut'])), transform = axmag.transAxes,horizontalalignment='left')
	#axmag.text(0.04,0.92, r'EQW H$\alpha$ > {0} $\AA$'.format(int(e_pinfo['obs']['eqw_cut'])), transform = axmag.transAxes,horizontalalignment='left')
	axmag.text(0.04,0.87, r'N = '+str(int(np.sum(keep_idx))), transform = axmag.transAxes,horizontalalignment='left')
	axmag.set_xlabel(xlab_ha[1])
	axmag.set_ylabel(ylab_ha[1])
	if standardized_ha_axlim:
		axmag.axis((ha_lim[0],ha_lim[1],ha_lim[0],ha_lim[1]))
		axmag.plot(ha_lim,ha_lim,linestyle='--',color='0.1',alpha=0.8)
	else:
		axmag = threed_dutils.equalize_axes(axmag, pl_ha_obs[:,0], pl_ha_mag)
		axmag.axis((3,10,3,10))
	off,scat = threed_dutils.offset_and_scatter(pl_ha_obs[:,0], pl_ha_mag, biweight=True)
	axmag.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) +' dex', transform = axmag.transAxes,horizontalalignment='right')
	axmag.text(0.96,0.1, 'mean offset='+"{:.2f}".format(off)+ ' dex', transform = axmag.transAxes,horizontalalignment='right')

	ax2.text(0.97,0.92, r'S/N (H$\alpha$,H$\beta$) > {0}'.format(int(e_pinfo['obs']['sn_cut'])), transform = ax2.transAxes,horizontalalignment='right')
	#ax2.text(0.97,0.92, r'EQW H$\alpha$ > {0} $\AA$'.format(int(e_pinfo['obs']['eqw_cut'])), transform = ax2.transAxes,horizontalalignment='right')
	ax2.text(0.97,0.87, r'N = '+str(int(np.sum(keep_idx))), transform = ax2.transAxes,horizontalalignment='right')
	ax2.set_ylabel(r'log(Z/Z$_{\odot}$) [Prospector]')
	ax2.set_xlabel(r'log(H$_{\alpha}$ CLOUDY/Kennicutt) [Prospector]')
	ax2.xaxis.set_major_locator(MaxNLocator(4))


	ax3[0].text(0.04,0.92, r'S/N (H$\alpha$,H$\beta$) > {0}'.format(int(e_pinfo['obs']['sn_cut'])), transform = ax3[0].transAxes,horizontalalignment='left')
	#ax3[0].text(0.04,0.92, r'EQW H$\alpha$ > {0} $\AA$'.format(int(e_pinfo['obs']['eqw_cut'])), transform = ax3[0].transAxes,horizontalalignment='left')
	ax3[0].text(0.04,0.87, r'N = '+str(int(np.sum(keep_idx))), transform = ax3[0].transAxes,horizontalalignment='left')
	ax3[0].set_xlabel(r'log(SFR [10 Myr]) [marginalized, Prospector]')
	ax3[0].set_ylabel(r'log(SFR [10 Myr]) [best-fit, MAGPHYS]')
	ax3[0] = threed_dutils.equalize_axes(ax3[0], sfr10[:,0], msfr10)
	off,scat = threed_dutils.offset_and_scatter(sfr10[:,0], msfr10, biweight=True)
	ax3[0].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) +' dex', transform = ax3[0].transAxes,horizontalalignment='right')
	ax3[0].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off)+ ' dex', transform = ax3[0].transAxes,horizontalalignment='right')

	ax3[1].set_xlabel(r'log(F$_{\mathrm{emit}}$/F$_{\mathrm{obs}}$) (6563 $\AA$) [marginalized, Prospector]')
	ax3[1].set_ylabel(r'log(F$_{\mathrm{emit}}$/F$_{\mathrm{obs}}$) (6563 $\AA$) [best-fit, MAGPHYS]')
	ax3[1] = threed_dutils.equalize_axes(ax3[1], ha_ext[:,0], mha_ext)
	off,scat = threed_dutils.offset_and_scatter(ha_ext[:,0], mha_ext, biweight=True)
	ax3[1].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex', transform = ax3[1].transAxes,horizontalalignment='right')
	ax3[1].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex', transform = ax3[1].transAxes,horizontalalignment='right')

	ax4[0].text(0.04,0.92, r'S/N (H$\alpha$,H$\beta$) > {0}'.format(int(e_pinfo['obs']['sn_cut'])), transform = ax4[0].transAxes,horizontalalignment='left')
	#ax4[0].text(0.04,0.92, r'EQW H$\alpha$ > {0} $\AA$'.format(int(e_pinfo['obs']['eqw_cut'])), transform = ax4[0].transAxes,horizontalalignment='left')
	ax4[0].text(0.04,0.87, r'N = '+str(int(np.sum(keep_idx))), transform = ax4[0].transAxes,horizontalalignment='left')
	ax4[0].set_xlabel(r'log(SFR [100 Myr]) [marginalized, MAGPHYS]')
	ax4[0].set_ylabel(r'log(SFR [100 Myr]) [best-fit, MAGPHYS]')
	ax4[0] = threed_dutils.equalize_axes(ax4[0], msfr100_marginalized[:,1], msfr100)
	off,scat = threed_dutils.offset_and_scatter(msfr100_marginalized[:,1], msfr100, biweight=True)
	ax4[0].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) +' dex', transform = ax4[0].transAxes,horizontalalignment='right')
	ax4[0].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off)+ ' dex', transform = ax4[0].transAxes,horizontalalignment='right')

	ax4[1].set_xlabel(r'log(SFR [100 Myr]) [marginalized, Prospector]')
	ax4[1].set_ylabel(r'log(SFR [100 Myr]) [marginalized, MAGPHYS]')
	ax4[1] = threed_dutils.equalize_axes(ax4[1], sfr100[:,0], msfr100_marginalized[:,1])
	off,scat = threed_dutils.offset_and_scatter(sfr100[:,0], msfr100_marginalized[:,1], biweight=True)
	ax4[1].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) +' dex', transform = ax4[1].transAxes,horizontalalignment='right')
	ax4[1].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off)+ ' dex', transform = ax4[1].transAxes,horizontalalignment='right')

	fig1.tight_layout()
	fig1.savefig(outname_prosp,dpi=dpi)
	figmag.tight_layout()
	figmag.savefig(outname_mag,dpi=dpi)
	fig2.tight_layout()
	fig2.savefig(outname_cloudy,dpi=dpi)
	fig3.tight_layout()
	fig3.savefig(outname_ha_inpt,dpi=dpi)
	fig4.tight_layout()
	fig4.savefig(outname_sfr_margcomp,dpi=dpi)
	plt.close()

def fit_and_save(met,ha_ratio):

	### clean nans
	good = np.isfinite(ha_ratio[:,0])
	fit_ha_ratio = ha_ratio[good,0]
	fit_met = met[good,0]

	z = np.polyfit(fit_met, fit_ha_ratio, 7)

	outloc = '/Users/joel/code/python/threedhst_bsfh/data/pickles/ha_ratio.pickle'
	pickle.dump(z,open(outloc, "wb"))

def bdec_corr_eqn(x, hdel_eqw_obs, hdel_eqw_model,
	              halpha_obs, hbeta_obs,
	              halpha_abs_eqw, hbeta_abs_eqw,
	              halpha_continuum, hbeta_continuum,
	              additive):

	##### how do we translate Hdelta offset into Halpha / Hbeta offset?
	##### then, pull out measured halpha, hbeta EQW to adjust for discrepancy in balmer absorption
	if additive:
		ratio = hdel_eqw_obs - hdel_eqw_model
		use_ratio = ratio*x

		halpha_new = halpha_obs + use_ratio * halpha_continuum
		hbeta_new = hbeta_obs + use_ratio * hbeta_continuum

	else:
		ratio = hdel_eqw_obs / hdel_eqw_model
		use_ratio = (ratio-1.0)*x + 1.0

		halpha_new = halpha_obs + halpha_abs_eqw*(use_ratio-1) * halpha_continuum
		hbeta_new = hbeta_obs + hbeta_abs_eqw*(use_ratio-1) * hbeta_continuum

	##### bdec corrected
	bdec_corrected = bdec_to_ext(halpha_new/hbeta_new)

	return bdec_corrected

def minimize_bdec_corr_eqn(x, hdel_eqw_obs, hdel_eqw_model, halpha_obs, hbeta_obs,
	                          halpha_abs_eqw, hbeta_abs_eqw,
	                          halpha_continuum, hbeta_continuum,
	                          additive,
	                          bdec_model):
	'''
	minimize the scatter in bdec_to_ext(obs_bdec), bdec_to_ext(model_bdec)
	by some function bdec_corr_eqw() described above
	'''

	bdec_corrected = bdec_corr_eqn(x, hdel_eqw_obs, hdel_eqw_model,
	                                  halpha_obs, hbeta_obs,
						              halpha_abs_eqw, hbeta_abs_eqw,
						              halpha_continuum, hbeta_continuum,
						              additive)

	off, scat = threed_dutils.offset_and_scatter(bdec_corrected, bdec_model,biweight=True)

	return scat

def paper_summary_plot(e_pinfo, hflag, outname='test.png'):

	#### paper font preferences
	pweight = 'semibold'
	pfontsize = 18
	px = 0.03
	py = 0.85

	keep_idx = brown_quality_cuts.halpha_cuts(e_pinfo)

	##### AGN identifiers
	sfing, composite, agn = return_agn_str(keep_idx)
	keys = [sfing, composite, agn]

	##### halpha log tricks
	f_ha = e_pinfo['obs']['f_ha'][keep_idx,:]
	model_ha = e_pinfo['prosp']['cloudy_ha'][keep_idx,:]
	ylabel = r'log(H$_{\alpha}$ luminosity) [Prospector]'
	xlabel = r'log(H$_{\alpha}$ luminosity) [observed]'

	xplot_ha = minlog(f_ha,axis=0)
	yplot_ha = minlog(model_ha,axis=0)

	##### figure
	fig, ax = plt.subplots(1,2, figsize = (12,6))

	for ii in xrange(len(labels)):
		pdict = merge_dicts(herschdict[0],bptdict[ii])

		xerr_ha = threed_dutils.asym_errors(xplot_ha[keys[ii],0],xplot_ha[keys[ii],1], xplot_ha[keys[ii],2],log=False)
		yerr_ha = threed_dutils.asym_errors(yplot_ha[keys[ii],0],yplot_ha[keys[ii],1], yplot_ha[keys[ii],2],log=False)

		ax[0].errorbar(xplot_ha[keys[ii],0], yplot_ha[keys[ii],0], yerr=yerr_ha, xerr=xerr_ha, 
			           linestyle=' ',**pdict)

	ax[0].set_xlabel(r'log(L[H$\alpha$]) from observed spectrum')
	ax[0].set_ylabel(r'log(L[H$\alpha$]) from fit to photometry')

	off,scat = threed_dutils.offset_and_scatter(xplot_ha[:,0], yplot_ha[:,0],biweight=True)
	ax[0].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat)+' dex',
			  transform = ax[0].transAxes,horizontalalignment='right')
	ax[0].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off)+' dex',
			      transform = ax[0].transAxes,horizontalalignment='right')
	ax[0] = threed_dutils.equalize_axes(ax[0], xplot_ha, yplot_ha)
	ax[0].text(px,py, 'test of model \n star formation rate',
			   transform = ax[0].transAxes,ha='left',fontsize=pfontsize,weight=pweight,multialignment='center')


	#### dn4000
	dn_idx = brown_quality_cuts.dn4000_cuts(e_pinfo)
	dn4000_obs = e_pinfo['obs']['dn4000'][dn_idx]
	dn4000_prosp = e_pinfo['prosp']['dn4000'][dn_idx,0]

	##### AGN identifiers
	sfing, composite, agn = return_agn_str(dn_idx)
	keys = [sfing, composite, agn]

	#### plot dn4000
	for ii in xrange(len(labels)):
		pdict = merge_dicts(herschdict[0],bptdict[ii])

		pl_dn4000_obs = dn4000_obs[keys[ii]]
		pl_dn4000_prosp = dn4000_prosp[keys[ii]]

		errs_pro = threed_dutils.asym_errors(e_pinfo['prosp']['dn4000'][dn_idx,0][keys[ii]],
			                                 e_pinfo['prosp']['dn4000'][dn_idx,1][keys[ii]],
			                                 e_pinfo['prosp']['dn4000'][dn_idx,2][keys[ii]])

		ax[1].errorbar(pl_dn4000_obs, pl_dn4000_prosp, yerr=errs_pro, linestyle=' ', **pdict)

	ax[1].set_xlabel(r'D$_n$(4000) from observed spectrum')
	ax[1].set_ylabel(r'D$_n$(4000) from fit to photometry')

	off,scat = threed_dutils.offset_and_scatter(dn4000_obs,dn4000_prosp,biweight=True)
	ax[1].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat),
			  transform = ax[1].transAxes,horizontalalignment='right')
	ax[1].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off),
			      transform = ax[1].transAxes,horizontalalignment='right')
	ax[1] = threed_dutils.equalize_axes(ax[1], dn4000_obs, dn4000_prosp)

	ax[1].text(px,py, 'test of model \n star formation history',
			   transform = ax[1].transAxes,ha='left',fontsize=pfontsize,weight=pweight,multialignment='center')

	#### close and save
	fig.tight_layout()
	fig.savefig(outname,dpi=dpi)
	plt.close()



def obs_vs_prosp_balmlines(e_pinfo,hflag,outname='test.png',outname_resid='test.png',model='obs',
	                       standardized_ha_axlim = False):

	#################
	#### plot observed Halpha versus expected (PROSPECTOR ONLY)
	#################
	keep_idx = brown_quality_cuts.halpha_cuts(e_pinfo)

	##### AGN identifiers
	sfing, composite, agn = return_agn_str(keep_idx)
	keys = [sfing, composite, agn]

	##### herschel identifier
	hflag = [hflag[keep_idx],~hflag[keep_idx]]

	norm_errs = []
	norm_flag = []

	##### plot!
	fig1, ax1 = plt.subplots(2,2, figsize = (12,12))
	fig2, ax2 = plt.subplots(1,1, figsize = (6,6))

	for jj in xrange(2):

		### jj = 0: flux
		### jj = 1: EQW
		if jj == 1:
			f_ha = e_pinfo['obs']['eqw_ha'][keep_idx,:]
			f_hb = e_pinfo['obs']['eqw_hb'][keep_idx,:]
			model_ha = e_pinfo['prosp']['cloudy_ha_eqw'][keep_idx,:]
			model_hb = e_pinfo['prosp']['cloudy_hb_eqw'][keep_idx,:]
			ylabel = [r'log(H$_{\alpha}$ EQW) [Prospector]', r'log(H$_{\beta}$ EQW) [Prospector]']
			xlabel = [r'log(H$_{\alpha}$ EQW) [observed]', r'log(H$_{\beta}$ EQW) [observed]']
			ha_lim = ha_eqw_lim
		else:
			f_ha = e_pinfo['obs']['f_ha'][keep_idx,:]
			f_hb = e_pinfo['obs']['f_hb'][keep_idx,:]
			model_ha = e_pinfo['prosp']['cloudy_ha'][keep_idx,:]
			model_hb = e_pinfo['prosp']['cloudy_hb'][keep_idx,:]
			ylabel = [r'log(H$_{\alpha}$ luminosity) [Prospector]', r'log(H$_{\beta}$ luminosity) [Prospector]']
			xlabel = [r'log(H$_{\alpha}$ luminosity) [observed]', r'log(H$_{\beta}$ luminosity) [observed]']
			ha_lim = ha_flux_lim

		ax = ax1[jj,:]

		xplot_ha = minlog(f_ha,axis=0)
		yplot_ha = minlog(model_ha,axis=0)

		xplot_hb = minlog(f_hb,axis=0)
		yplot_hb = minlog(model_hb,axis=0)

		for ii in xrange(len(labels)):
			for kk in xrange(len(hflag)):

				### update colors, define region
				pdict = merge_dicts(herschdict[kk],bptdict[ii])
				plt_idx = keys[ii] & hflag[kk]

				### setup errors
				xerr_ha = threed_dutils.asym_errors(xplot_ha[plt_idx,0],xplot_ha[plt_idx,1], xplot_ha[plt_idx,2],log=False)
				xerr_hb = threed_dutils.asym_errors(xplot_hb[plt_idx,0],xplot_hb[plt_idx,1], xplot_hb[plt_idx,2],log=False)

				yerr_ha = threed_dutils.asym_errors(yplot_ha[plt_idx,0],yplot_ha[plt_idx,1], yplot_ha[plt_idx,2],log=False)
				yerr_hb = threed_dutils.asym_errors(yplot_hb[plt_idx,0],yplot_hb[plt_idx,1], yplot_hb[plt_idx,2],log=False)

				norm_errs.append(normalize_error(yplot_ha[plt_idx,0],yerr_ha,xplot_ha[plt_idx,0],xerr_ha))
				norm_flag.append([labels[ii]]*np.sum(plt_idx))

				ax[0].errorbar(xplot_ha[plt_idx,0], yplot_ha[plt_idx,0], yerr=yerr_ha, xerr=xerr_ha, 
					           linestyle=' ',**pdict)
				ax[1].errorbar(xplot_hb[plt_idx,0], yplot_hb[plt_idx,0], yerr=yerr_hb, xerr=xerr_hb,
		                       linestyle=' ',**pdict)
				if jj == 0:
					ax2.errorbar(yplot_ha[plt_idx,0] - xplot_ha[plt_idx,0],yplot_hb[plt_idx,0] - xplot_hb[plt_idx,0],
							     linestyle=' ',**pdict)

		ax[0].text(0.04,0.92, r'S/N (H$\alpha$,H$\beta$) > {0}'.format(int(e_pinfo['obs']['sn_cut'])), transform = ax[0].transAxes,horizontalalignment='left')
		#ax[0].text(0.04,0.92, r'EQW (H$\alpha$,H$\beta$) > {0} $\AA$'.format(int(e_pinfo['obs']['eqw_cut'])), transform = ax[0].transAxes,horizontalalignment='left')
		ax[0].text(0.04,0.87, r'N = '+str(int(np.sum(keep_idx))), transform = ax[0].transAxes,horizontalalignment='left')
		ax[0].set_ylabel(ylabel[0])
		ax[0].set_xlabel(xlabel[0])
		if standardized_ha_axlim:
			ax[0].axis((ha_lim[0],ha_lim[1],ha_lim[0],ha_lim[1]))
			ax[0].plot(ha_lim,ha_lim,linestyle='--',color='0.1',alpha=0.8)
		else:
			ax[0] = threed_dutils.equalize_axes(ax[0],xplot_ha[:,0],yplot_ha[:,0])
		off,scat = threed_dutils.offset_and_scatter(xplot_ha[:,0],yplot_ha[:,0],biweight=True)
		ax[0].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
				  transform = ax[0].transAxes,horizontalalignment='right')
		ax[0].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
				      transform = ax[0].transAxes,horizontalalignment='right')
		#ax[0].legend(loc=2)

		ax[1].set_ylabel(ylabel[1])
		ax[1].set_xlabel(xlabel[1])
		ax[1] = threed_dutils.equalize_axes(ax[1],xplot_hb[:,0],yplot_hb[:,0])
		off,scat = threed_dutils.offset_and_scatter(xplot_hb[:,0],yplot_hb[:,0],biweight=True)
		ax[1].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
				  transform = ax[1].transAxes,horizontalalignment='right')
		ax[1].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
				      transform = ax[1].transAxes,horizontalalignment='right')

		if jj == 0:
			ax2.set_ylabel(r'log(model/obs) H$_{\alpha}$)')
			ax2.set_xlabel(r'log(model/obs) H$_{\beta}$)')
			max = np.max([np.abs(ax2.get_ylim()).max(),np.abs(ax2.get_xlim()).max()])
			max = 0.8
			ax2.plot([0.0,0.0],[-max,max],linestyle='--',alpha=1.0,color='0.4')
			ax2.plot([-max,max],[0.0,0.0],linestyle='--',alpha=1.0,color='0.4')
			ax2.axis((-max,max,-max,max))

			### annotate directions
			ax2.arrow(0.722, 0.15, 0.04, 0.04, head_width=0.02, head_length=0.03, fc='k', ec='k',width=0.002,transform = ax2.transAxes)
			ax2.arrow(0.722, 0.15,-0.04, 0.04, head_width=0.02, head_length=0.03, fc='k', ec='k',width=0.002,transform = ax2.transAxes)

			ax2.text(0.757, 0.12, 'normalization \n error',transform = ax2.transAxes,horizontalalignment='left',fontsize=10,multialignment='center',weight='semibold')
			ax2.text(0.687, 0.12, 'reddening \n error',transform = ax2.transAxes,horizontalalignment='right',fontsize=10,multialignment='center',weight='semibold')

			threed_dutils.return_n_outliers(np.log10(f_ha[:,0]),np.log10(model_ha[:,0]),e_pinfo['objnames'][keep_idx],10,cp_files='halpha')

	fig1.tight_layout()
	fig1.savefig(outname,dpi=dpi)

	fig2.tight_layout()
	fig2.savefig(outname_resid,dpi=dpi)
	plt.close()

	return norm_errs, norm_flag

def obs_vs_model_hdelta(e_pinfo,hflag,outname=None,outname_dnplt=None,eqw=False):

	'''
	THESE ARE MY OPTIONS FOR HDELTA
	##### hdelta absorption
	prosp['hdel_abs'] = hd_lum_prosp
	prosp['hdel_eqw'] = hd_eqw_prosp
	prosp['hdel_abs_marg'] = hd_lum_prosp_marg
	prosp['hdel_eqw_marg'] = hd_eqw_prosp_marg
	prosp['hdel_abs_addem'] = hd_lum_eline_prosp
	prosp['hdel_eqw_addem'] = hd_eqw_eline_prosp

	'''

	good_idx = brown_quality_cuts.hdelta_cuts(e_pinfo)

	##### for dn4000 plots, if necessary
	dn4000_obs = e_pinfo['obs']['dn4000'][good_idx]
	dn4000_prosp = e_pinfo['prosp']['dn4000'][good_idx,0]

	if eqw:
		min = 0.2
		max = 1.0
		plotlim = (min,max,min,max)

		hdel_obs = e_pinfo['obs']['hdel_eqw'][good_idx]

		hdel_prosp_marg = e_pinfo['prosp']['hdel_eqw_marg'][good_idx]

		hdel_prosp_em = np.clip(e_pinfo['prosp']['hdel_eqw_addem'][good_idx],0.2,np.inf)

		hdel_prosp = np.log10(e_pinfo['prosp']['hdel_eqw'][good_idx])
		xtit = [r'observed log(H$_{\delta}$ EQW)', 
		        r'observed log(H$_{\delta}$ EQW)',
		        r'observed log(H$_{\delta}$ EQW)']
		ytit = [r'model log(H$_{\delta}$ EQW) [absorption + emission]',
		        r'model log(H$_{\delta}$ EQW) [best-fit]',
		        r'model log(H$_{\delta}$ EQW) [marginalized]']

		# only make this plot in EQW
		fig2, ax2 = plt.subplots(1,2, figsize=(12.5,6))

	else:
		min = 5.0
		max = 8.0
		plotlim = (min,max,min,max)

		hdel_obs = e_pinfo['obs']['hdel'][good_idx]
		#hdel_plot_errs = threed_dutils.asym_errors(e_pinfo['obs']['hdel'][good_idx,0], e_pinfo['obs']['hdel'][good_idx,1], e_pinfo['obs']['hdel'][good_idx,2], log=True)
		
		hdel_prosp_marg = e_pinfo['prosp']['hdel_abs_marg'][good_idx]
		#hdel_prosp_marg_errs = threed_dutils.asym_errors(e_pinfo['prosp']['hdel_abs_marg'][good_idx,0], e_pinfo['prosp']['hdel_abs_marg'][good_idx,1], e_pinfo['prosp']['hdel_abs_marg'][good_idx,2], log=True)
		
		hdel_prosp_em = e_pinfo['prosp']['hdel_abs_addem'][good_idx]

		hdel_prosp = np.log10(e_pinfo['prosp']['hdel_abs'][good_idx])
		xtit = [r'observed log(-H$_{\delta}$)', 
		        r'observed log(-H$_{\delta}$)',
		        r'observed log(-H$_{\delta}$)']
		ytit = [r'model log(-H$_{\delta}$) [absorption + emission]',
		        r'model log(-H$_{\delta}$) [best-fit]',
		        r'model log(-H$_{\delta}$) [marginalized]']

	##### AGN identifiers
	sfing, composite, agn = return_agn_str(good_idx)
	keys = [sfing, composite, agn]

	##### herschel identifier
	hflag = [hflag[good_idx],~hflag[good_idx]]

	### Hdelta plot
	# fig, ax = plt.subplots(2,2, figsize = (12,12))
	#fig, ax = plt.subplots(1,2, figsize = (12,6))
	fig, ax = plt.subplots(1,1, figsize = (6,6))
	norm_errs = []
	norm_flag = []

	ax = np.ravel(ax)
	# loop and put points on plot
	for ii in xrange(len(labels)):
		for kk in xrange(len(hflag)):

			### update colors, define sample
			pdict = merge_dicts(herschdict[kk],bptdict[ii])
			plt_idx = keys[ii] & hflag[kk]

			### define quantities
			pl_hdel_obs = hdel_obs[plt_idx,0]
			pl_hdel_prosp = hdel_prosp_marg[plt_idx,0]
			pl_hdel_prosp_bestfit = hdel_prosp[plt_idx]
			pl_hdel_prosp_em = hdel_prosp_em[plt_idx]

			### errors
			errs_pro = threed_dutils.asym_errors(hdel_prosp_marg[plt_idx,0],
				                                 hdel_prosp_marg[plt_idx,1],
				                                 hdel_prosp_marg[plt_idx,2],log=True)

			errs_obs = threed_dutils.asym_errors(hdel_obs[plt_idx,0],
				                                 hdel_obs[plt_idx,1],
				                                 hdel_obs[plt_idx,2],log=True)

			norm_errs.append(normalize_error(np.log10(pl_hdel_obs),errs_obs,
				                             np.log10(pl_hdel_prosp),errs_pro))
			norm_flag.append([labels[ii]]*np.sum(plt_idx))

			# ax[0].errorbar(np.log10(pl_hdel_obs), np.log10(pl_hdel_prosp_em), xerr=errs_obs, linestyle=' ',  **pdict)
			# ax[1].errorbar(np.log10(pl_hdel_obs), pl_hdel_prosp_bestfit, xerr=errs_obs, linestyle=' ', **pdict)
			ax[0].errorbar(np.log10(pl_hdel_obs), np.log10(pl_hdel_prosp), xerr=errs_obs, yerr=errs_pro, linestyle=' ', **pdict)
			'''
			ax[1].errorbar(np.log10(pl_hdel_obs), np.log10(pl_hdel_prosp_em), xerr=errs_obs, linestyle=' ', **pdict)
			'''
			if eqw:
				ax2[0].errorbar(np.log10(pl_hdel_obs), dn4000_obs[plt_idx], linestyle=' ',**pdict)
				ax2[1].errorbar(np.log10(pl_hdel_prosp), dn4000_prosp[plt_idx], linestyle=' ',**pdict)

	ax[0].text(0.04,0.92, r'S/N H$\delta$ > {0}'.format(int(e_pinfo['obs']['hdelta_sn_cut'])), transform = ax[0].transAxes,horizontalalignment='left')
	#ax[0].text(0.04,0.92, r'EQW H$\delta$ < -{0} $\AA$'.format(int(e_pinfo['obs']['hdelta_eqw_cut'])), transform = ax[0].transAxes,horizontalalignment='left')
	ax[0].text(0.04,0.87, r'N = '+str(int(np.sum(good_idx))), transform = ax[0].transAxes,horizontalalignment='left')
	ax[0].set_xlabel(xtit[2])
	ax[0].set_ylabel(ytit[2])
	off,scat = threed_dutils.offset_and_scatter(np.log10(hdel_obs[:,0]), np.log10(hdel_prosp_marg[:,0]),biweight=True)
	ax[0].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex', transform = ax[0].transAxes,horizontalalignment='right')
	ax[0].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex', transform = ax[0].transAxes,horizontalalignment='right')
	ax[0].axis(plotlim)
	ax[0].plot([min,max],[min,max],linestyle='--',color='0.1',alpha=0.8)

	'''
	ax[1].set_xlabel(xtit[0])
	ax[1].set_ylabel(ytit[0])
	off,scat = threed_dutils.offset_and_scatter(np.log10(hdel_obs[:,0]), np.log10(hdel_prosp_em),biweight=True)
	ax[1].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex', transform = ax[1].transAxes,horizontalalignment='right')
	ax[1].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex', transform = ax[1].transAxes,horizontalalignment='right')
	ax[1].axis(plotlim)
	ax[1].plot([min,max],[min,max],linestyle='--',color='0.1',alpha=0.8)
	'''

	### dn4000 plot options
	if eqw:
		ax2[0].set_xlabel('observed log(H$_{\delta}$ EQW)')
		ax2[1].set_xlabel('Prospector log(H$_{\delta}$ EQW)')

		ax2[0].set_ylabel('observed Dn(4000)')
		ax2[1].set_ylabel('Prospector Dn(4000)')

		axlims = (0.3,1.0,0.9,1.5)
		ax2[0].axis(axlims)
		ax2[1].axis(axlims)

		fig2.tight_layout()
		fig2.savefig(outname_dnplt, dpi=dpi)

		threed_dutils.return_n_outliers(np.log10(hdel_obs[:,0]),np.log10(hdel_prosp_marg[:,0]),e_pinfo['objnames'][good_idx],5,cp_files='hdelta_abs')

	fig.tight_layout()
	fig.savefig(outname, dpi=dpi)
	plt.close()

	return norm_errs, norm_flag

def obs_vs_model_dn(e_pinfo,hflag,outname=None):

	### define limits
	dn_idx = e_pinfo['obs']['dn4000'] > 0.5

	##### AGN identifiers
	sfing, composite, agn = return_agn_str(dn_idx)
	keys = [sfing, composite, agn]

	##### plot quantities
	dn4000_obs = e_pinfo['obs']['dn4000'][dn_idx]
	dn4000_prosp = e_pinfo['prosp']['dn4000'][dn_idx,0]

	##### herschel identifier
	hflag = [hflag[dn_idx],~hflag[dn_idx]]

	### plot comparison
	### Dn4000 first
	fig, ax = plt.subplots(1,1, figsize = (6,6))
	norm_errs = []
	norm_flag = []

	# loop and put points on plot
	for ii in xrange(len(labels)):
		for kk in xrange(len(hflag)):

			### update colors, define sample
			pdict = merge_dicts(herschdict[kk],bptdict[ii])
			plt_idx = keys[ii] & hflag[kk]

			### define quantities
			pl_dn4000_obs = dn4000_obs[plt_idx]
			pl_dn4000_prosp = dn4000_prosp[plt_idx]

			### Prospector errors
			errs_pro = threed_dutils.asym_errors(e_pinfo['prosp']['dn4000'][dn_idx,0][plt_idx],
				                                 e_pinfo['prosp']['dn4000'][dn_idx,1][plt_idx],
				                                 e_pinfo['prosp']['dn4000'][dn_idx,2][plt_idx])

			norm_errs.append(normalize_error(pl_dn4000_obs,[np.zeros_like(pl_dn4000_obs),np.zeros_like(pl_dn4000_obs)],
				                             pl_dn4000_prosp,errs_pro))
			norm_flag.append([labels[ii]]*np.sum(plt_idx))

			ax.errorbar(pl_dn4000_obs, pl_dn4000_prosp, yerr=errs_pro, linestyle=' ', **pdict)

	ax.set_xlabel(r'observed D$_n$(4000)')
	ax.set_ylabel(r'Prospector D$_n$(4000)')
	ax = threed_dutils.equalize_axes(ax, dn4000_obs, dn4000_prosp)
	off,scat = threed_dutils.offset_and_scatter(dn4000_obs, dn4000_prosp,biweight=True)
	ax.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat), transform = ax.transAxes,horizontalalignment='right')
	ax.text(0.96,0.1, 'mean offset='+"{:.2f}".format(off), transform = ax.transAxes,horizontalalignment='right')
	ax.text(0.04,0.9, r'N = '+str(int(np.sum(dn_idx))), transform = ax.transAxes,horizontalalignment='left')

	threed_dutils.return_n_outliers(dn4000_obs,dn4000_prosp,e_pinfo['objnames'][dn_idx],3,cp_files='dn4000')
	plt.tight_layout()
	plt.savefig(outname, dpi=dpi)
	plt.close()

	return norm_errs, norm_flag

def bdec_to_ext(bdec):
	return 2.5*np.log10(bdec/2.86)

def normalize_error(obs,obserr,mod,moderr):
	
	# define output
	out = np.zeros_like(obs)
	
	# find out which side of error bar to use
	undershot = obs > mod
	
	# create output
	out[~undershot] = (obs[~undershot] - mod[~undershot]) / np.sqrt(moderr[0][~undershot]**2+obserr[0][~undershot]**2)
	out[undershot] = (obs[undershot] - mod[undershot]) / np.sqrt(moderr[1][undershot]**2+obserr[1][undershot]**2)

	return out

def gauss_fit(x,y):

	from astropy.modeling import fitting, functional_models
	init = functional_models.Gaussian1D(mean=0.0,stddev=1,amplitude=1)
	fitter = fitting.LevMarLSQFitter()
	fit = fitter(init,x,y)

	return fit


def onesig_error_plot(bdec_errs,bdec_flag,dn4000_errs,dn4000_flag,hdelta_errs,hdelta_flag,ha_errs,ha_flag,outbase):


	# NOTE: THROWING OUT HUGE OUTLIERS (80sigma)
	# next step: pass in an AGN / composite mask, make plot with and without AGN

	kwargs = {'color':'0.5','alpha':0.8,'histtype':'bar','lw':2,'normed':1,'range':(-10,10)}
	nbins = 20

	x = [np.array([item for sublist in bdec_errs for item in sublist]),
	     np.array([item for sublist in dn4000_errs for item in sublist]),
	     np.array([item for sublist in hdelta_errs for item in sublist]),
	     np.array([item for sublist in ha_errs for item in sublist])]
	tits = [r'(obs - model) / 1$\sigma$ error [A$_{H\beta}$-A$_{H\alpha}$]',
	        r'(obs - model) / 1$\sigma$ error [D$_n$(4000)]',
	        r'(obs - model) / 1$\sigma$ error [H$\delta$ EQW]',
	        r'(obs - model) / 1$\sigma$ error [H$_{\alpha}$ luminosity]']
	pltname = ['bdec','dn4000','hdelta_eqw','halpha_lum']
	flags = [np.array([item for sublist in bdec_flag for item in sublist]),
	    	 np.array([item for sublist in dn4000_flag for item in sublist]),
	     	 np.array([item for sublist in hdelta_flag for item in sublist]),
	         np.array([item for sublist in ha_flag for item in sublist])]

	#### ALL GALAXIES
	for ii in xrange(len(x)):
		fig, ax = plt.subplots(1, 1, figsize = (8,8))
		num, b, p = ax.hist(x[ii],nbins,**kwargs)
		save_xlim = ax.get_xlim()
		b = (b[:-1] + b[1:])/2.
		ax.set_ylabel('N')
		ax.set_xlabel(tits[ii])
		fit = gauss_fit(b,num)
		ax.plot(b,fit(b),lw=5, color='red',linestyle='--')
		ax.text(0.98,0.9,r'$\sigma$='+"{:.2f}".format(fit.stddev.value),transform = ax.transAxes,ha='right')
		ax.set_xlim(-np.max(np.abs(save_xlim)),np.max(np.abs(save_xlim)))

		#### for balmer decrements, want equalized axes
		if ii == 0:
			ax.set_xlim(-30,30)

		plt.savefig(outbase+pltname[ii]+'_all_errs.png',dpi=dpi)
		plt.close()

	#### NO AGN
	for ii in xrange(len(x)):
		fig, ax = plt.subplots(1, 1, figsize = (8,8))
		keepers = flags[ii] != 'AGN'
		num, b, p = ax.hist(x[ii][keepers],nbins,**kwargs)
		save_xlim = ax.get_xlim()
		b = (b[:-1] + b[1:])/2.
		ax.set_ylabel('N')
		ax.set_xlabel(tits[ii])
		fit = gauss_fit(b,num)
		ax.plot(b,fit(b),lw=5, color='red',linestyle='--')
		ax.text(0.98,0.9,r'$\sigma$='+"{:.2f}".format(fit.stddev.value),transform = ax.transAxes,ha='right')
		ax.set_xlim(-np.max(np.abs(save_xlim)),np.max(np.abs(save_xlim)))
		plt.savefig(outbase+pltname[ii]+'_no_agn_errs.png',dpi=dpi)
		plt.close()

	#### ONLY STAR-FORMING
	for ii in xrange(len(x)):
		fig, ax = plt.subplots(1, 1, figsize = (8,8))
		keepers = flags[ii] == 'SF'
		num, b, p = ax.hist(x[ii][keepers],nbins,**kwargs)
		save_xlim = ax.get_xlim()
		b = (b[:-1] + b[1:])/2.
		ax.set_ylabel('N')
		ax.set_xlabel(tits[ii])
		fit = gauss_fit(b,num)
		ax.plot(b,fit(b),lw=5, color='red',linestyle='--')
		ax.text(0.98,0.9,r'$\sigma$='+"{:.2f}".format(fit.stddev.value),transform = ax.transAxes,ha='right')
		ax.set_xlim(-np.max(np.abs(save_xlim)),np.max(np.abs(save_xlim)))
		plt.savefig(outbase+pltname[ii]+'_sf_only_errs.png',dpi=dpi)
		plt.close()

def eline_errs(e_pinfo,hflag,outname='test.png'):

	'''
	plot sigma(bdec residuals) versus error
	first calculate error width by using 1sigma definition
	try also fitting a gaussian to see what that does
	consider a three-panel, for halpha, hbeta, bdec

	obslines['err_ha_orig']
	obslines['err_hb_orig']
	obslines['f_ha_orig']
	obslines['f_hb_orig']
	'''

	error_fraction = np.linspace(0.0,1.0,50)
	hdel_scat = 0.35

	keep_idx = brown_quality_cuts.halpha_cuts(e_pinfo)

	#### pull out original halpha / hbeta calculations
	f_ha = e_pinfo['obs']['f_ha_orig']
	f_hb = e_pinfo['obs']['f_hb_orig']

	#### open up variables
	ha_sig,hb_sig,bdec_sig = [],[],[]

	### loop over errors, calculate sigma
	for ii, efrac in enumerate(error_fraction):

		##### inflate errors
		# we add in quadrature in up/down errors, which is probably wrong in detail
		halpha_corr = efrac*(10**e_pinfo['prosp']['halpha_abs_marg'][:,0])
		hbeta_corr = efrac*(10**e_pinfo['prosp']['hbeta_abs_marg'][:,0])

		f_ha[:,1] = e_pinfo['obs']['f_ha_orig'][:,0] + np.sqrt((e_pinfo['obs']['f_ha_orig'][:,1] - e_pinfo['obs']['f_ha_orig'][:,0])**2+halpha_corr**2)
		f_ha[:,2] = e_pinfo['obs']['f_ha_orig'][:,0] - np.sqrt((e_pinfo['obs']['f_ha_orig'][:,0] - e_pinfo['obs']['f_ha_orig'][:,2])**2+hbeta_corr**2)

		f_hb[:,1] = e_pinfo['obs']['f_hb_orig'][:,0] + np.sqrt((e_pinfo['obs']['f_hb_orig'][:,1] - e_pinfo['obs']['f_hb_orig'][:,0])**2+halpha_corr**2)
		f_hb[:,2] = e_pinfo['obs']['f_hb_orig'][:,0] - np.sqrt((e_pinfo['obs']['f_hb_orig'][:,0] - e_pinfo['obs']['f_hb_orig'][:,2])**2+hbeta_corr**2)

		# calculate balmer decrement with inflated emission line errors
		# consider changing this to asymmetric error?
		err_ha = (f_ha[:,1] - f_ha[:,2])/2.
		err_hb = (f_hb[:,1] - f_hb[:,2])/2.

		bdec = f_ha[:,0] / f_hb[:,0]
		bdec_err = bdec * np.sqrt((err_ha/f_ha[:,0])**2+(err_hb/f_hb[:,0])**2)

		##### calculate normalized residuals
		# mock up asymmetric errors to use threed_dutils.normalize_error_asym
		bdec_obs_fake = np.transpose(np.vstack((bdec,bdec+bdec_err,bdec-bdec_err)))

		ha_sig_distr = threed_dutils.normalize_error_asym(f_ha[keep_idx,:],e_pinfo['prosp']['cloudy_ha'][keep_idx,:])
		hb_sig_distr = threed_dutils.normalize_error_asym(f_hb[keep_idx,:],e_pinfo['prosp']['cloudy_hb'][keep_idx,:])
		bdec_sig_distr = threed_dutils.normalize_error_asym(bdec_obs_fake[keep_idx,:],e_pinfo['prosp']['bdec_cloudy_marg'][keep_idx,:])
		
		##### calculate average (over +/- 1 sigma) distance from observations at 1 sigma
		# could also fit a Gaussian
		bdec_q = corner.quantile(bdec_sig_distr,[0.16,0.84]) - float(corner.quantile(bdec_sig_distr,[0.5]))
		bdec_sig.append(np.mean(np.abs(bdec_q)))
		ha_q = corner.quantile(ha_sig_distr,[0.16,0.84]) - float(corner.quantile(ha_sig_distr,[0.5]))
		ha_sig.append(np.mean(np.abs(ha_q)))
		hb_q = corner.quantile(hb_sig_distr,[0.16,0.84]) - float(corner.quantile(hb_sig_distr,[0.5]))
		hb_sig.append(np.mean(np.abs(hb_q)))

		
	#### create plots
	fig, ax = plt.subplots(2,1, figsize = (5,8))
	expect_color = '0.35'
	hdel_color = 'blue'

	'''
	ax[0].plot(error_fraction, ha_sig,lw=2,color='k')
	ax[0].set_xticklabels([])
	ax[0].yaxis.get_major_ticks()[0].label1.set_visible(False)
	ax[0].set_ylabel(r'1$\sigma$ (residuals/err) [H$\alpha$]')
	ax[0].axhline(1.0, linestyle=':', color=expect_color,lw=1.5)
	ax[0].axvline(hdel_scat, linestyle=':', color=hdel_color,lw=1.5)
	ax[0].text(0.59,2.0,'observed scatter' ,ha='center',fontsize=12,color=hdel_color)
	ax[0].text(0.47,1.85,'in H$\delta$',ha='center',fontsize=12,color=hdel_color)
	ax[0].text(0.66,1.85,'absorption',ha='center',fontsize=12,color=hdel_color)

	ax[0].text(0.2,1.08,'expected error \n distribution',ha='center',multialignment='center',fontsize=12,color=expect_color)
	'''

	ax[0].plot(error_fraction, hb_sig,lw=2,color='k')
	ax[0].set_xticklabels([])
	ax[0].yaxis.get_major_ticks()[0].label1.set_visible(False)
	ax[0].set_ylabel(r'1$\sigma$ (residuals/err) [H$\beta$]')
	ax[0].axhline(1.0, linestyle=':', color=expect_color,lw=1.5)
	ax[0].axvline(hdel_scat, linestyle=':', color=hdel_color,lw=1.5)

	ax[0].text(0.57,2.5,'observed scatter' ,ha='center',fontsize=12,color=hdel_color)
	ax[0].text(0.45,2.35,'in H$\delta$',ha='center',fontsize=12,color=hdel_color)
	ax[0].text(0.64,2.35,'absorption',ha='center',fontsize=12,color=hdel_color)

	ax[0].text(0.7,1.08,'expected error \n distribution',ha='center',multialignment='center',fontsize=12,color=expect_color)

	ax[1].plot(error_fraction, bdec_sig,lw=2,color='k')
	ax[1].set_xlabel('fraction of absorption added to error')
	ax[1].set_ylabel(r'1$\sigma$ (residuals/err) [H$\alpha$/H$\beta$]')
	ax[1].axhline(1.0, linestyle=':', color=expect_color,lw=1.5)
	ax[1].axvline(hdel_scat, linestyle=':', color=hdel_color,lw=1.5)

	fig.tight_layout()
	fig.subplots_adjust(hspace=0.0)
	plt.savefig(outname,dpi=150)

def obs_vs_model_bdec(e_pinfo,hflag,outname1='test.png',outname2='test.png'):
	
	#################
	#### plot observed Balmer decrement versus expected
	#################
	# first is Prospector CLOUDY marg + MAGPHYS versus observations
	# second is Prospector CLOUDY bfit, Prospector calc bfit, Prospector calc marg versus observations

	keep_idx = brown_quality_cuts.halpha_cuts(e_pinfo)

	##### write down plot variables
	pl_bdec_cloudy_marg = bdec_to_ext(e_pinfo['prosp']['bdec_cloudy_marg'][keep_idx,:])
	pl_bdec_calc_marg = bdec_to_ext(e_pinfo['prosp']['bdec_calc_marg'][keep_idx,:])
	pl_bdec_cloudy_bfit = bdec_to_ext(e_pinfo['prosp']['bdec_cloudy_bfit'][keep_idx])
	pl_bdec_calc_bfit = bdec_to_ext(e_pinfo['prosp']['bdec_calc_bfit'][keep_idx])
	pl_bdec_magphys = bdec_to_ext(e_pinfo['mag']['bdec'][keep_idx])
	pl_bdec_measured = bdec_to_ext(e_pinfo['obs']['bdec'][keep_idx])

	##### BPT classifications, herschel flag
	sfing, composite, agn = return_agn_str(keep_idx)
	keys = [sfing, composite, agn]
	hflag = [hflag[keep_idx],~hflag[keep_idx]]

	#### create plots
	fig1, ax1 = plt.subplots(1,1, figsize = (6,6))
	fig2, ax2 = plt.subplots(1,3, figsize = (18.75,6))
	axlims = (-0.1,1.0)
	norm_errs = []
	norm_flag = []

	# loop and put points on plot
	for ii in xrange(len(labels)):
		for kk in xrange(len(hflag)):

			### update colors, define sample
			pdict = merge_dicts(herschdict[kk],bptdict[ii])
			plt_idx = keys[ii] & hflag[kk]

			### errors for this sample
			errs_obs = threed_dutils.asym_errors(pl_bdec_measured[plt_idx], 
		                                         bdec_to_ext(e_pinfo['obs']['bdec'][keep_idx][plt_idx]+e_pinfo['obs']['bdec_err'][keep_idx][plt_idx]),
		                                         bdec_to_ext(e_pinfo['obs']['bdec'][keep_idx][plt_idx]-e_pinfo['obs']['bdec_err'][keep_idx][plt_idx]), log=False)
			errs_cloudy_marg = threed_dutils.asym_errors(pl_bdec_cloudy_marg[plt_idx,0],
				                                   pl_bdec_cloudy_marg[plt_idx,1], 
				                                   pl_bdec_cloudy_marg[plt_idx,2],log=False)
			errs_calc_marg = threed_dutils.asym_errors(pl_bdec_cloudy_marg[plt_idx,0],
				                                   pl_bdec_cloudy_marg[plt_idx,1], 
				                                   pl_bdec_cloudy_marg[plt_idx,2],log=False)

			norm_errs.append(normalize_error(pl_bdec_measured[plt_idx],errs_obs,pl_bdec_cloudy_marg[plt_idx,0],errs_cloudy_marg))
			norm_flag.append([labels[ii]]*np.sum(plt_idx))

			ax1.errorbar(pl_bdec_measured[plt_idx], pl_bdec_cloudy_marg[plt_idx,0], xerr=errs_obs, yerr=errs_cloudy_marg,
				         linestyle=' ',**pdict)

			ax2[0].errorbar(pl_bdec_measured[plt_idx], pl_bdec_calc_marg[plt_idx,0], xerr=errs_obs, yerr=errs_calc_marg,
				           linestyle=' ',**pdict)
			ax2[1].errorbar(pl_bdec_measured[plt_idx], pl_bdec_calc_bfit[plt_idx], xerr=errs_obs,
				           linestyle=' ',**pdict)
			ax2[2].errorbar(pl_bdec_measured[plt_idx], pl_bdec_cloudy_bfit[plt_idx], xerr=errs_obs,
				           linestyle=' ',**pdict)

	#### MAIN FIGURE ERRATA
	ax1.text(0.04,0.92, r'S/N (H$\alpha$,H$\beta$) > {0}'.format(int(e_pinfo['obs']['sn_cut'])), transform = ax1.transAxes,horizontalalignment='left')
	#ax1.text(0.04,0.92, r'EQW (H$\alpha$,H$\beta$) > {0} $\AA$'.format(int(e_pinfo['obs']['eqw_cut'])), transform = ax1.transAxes,horizontalalignment='left')
	ax1.text(0.04,0.87, r'N = '+str(int(np.sum(keep_idx))), transform = ax1.transAxes,horizontalalignment='left')
	ax1.set_xlabel(r'observed A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$')
	ax1.set_ylabel(r'Prospector A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$')
	ax1 = threed_dutils.equalize_axes(ax1, pl_bdec_measured,pl_bdec_cloudy_marg[:,0],axlims=axlims)
	off,scat = threed_dutils.offset_and_scatter(pl_bdec_measured,pl_bdec_cloudy_marg[:,0],biweight=True)
	ax1.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat), transform = ax1.transAxes,horizontalalignment='right')
	ax1.text(0.96,0.1, 'mean offset='+"{:.2f}".format(off), transform = ax1.transAxes,horizontalalignment='right')

	#### SECONDARY FIGURE ERRATA
	ax2[0].set_xlabel(r'observed A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$')
	ax2[0].set_ylabel(r'Prospector calc marginalized A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$')
	ax2[0] = threed_dutils.equalize_axes(ax2[0], pl_bdec_measured,pl_bdec_calc_marg[:,0],axlims=axlims)
	off,scat = threed_dutils.offset_and_scatter(pl_bdec_measured,pl_bdec_calc_marg[:,0],biweight=True)
	ax2[0].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat), transform = ax2[0].transAxes,horizontalalignment='right')
	ax2[0].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off), transform = ax2[0].transAxes,horizontalalignment='right')

	ax2[1].set_xlabel(r'observed A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$')
	ax2[1].set_ylabel(r'Prospector calc best-fit A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$')
	ax2[1] = threed_dutils.equalize_axes(ax2[1], pl_bdec_measured,pl_bdec_calc_bfit,axlims=axlims)
	off,scat = threed_dutils.offset_and_scatter(pl_bdec_measured,pl_bdec_calc_bfit,biweight=True)
	ax2[1].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat), transform = ax2[1].transAxes,horizontalalignment='right')
	ax2[1].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off), transform = ax2[1].transAxes,horizontalalignment='right')

	ax2[2].set_xlabel(r'observed A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$')
	ax2[2].set_ylabel(r'Prospector CLOUDY best-fit A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$')
	ax2[2] = threed_dutils.equalize_axes(ax2[2], pl_bdec_measured,pl_bdec_cloudy_bfit,axlims=axlims)
	off,scat = threed_dutils.offset_and_scatter(pl_bdec_measured,pl_bdec_cloudy_bfit,biweight=True)
	ax2[2].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat), transform = ax2[2].transAxes,horizontalalignment='right')
	ax2[2].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off), transform = ax2[2].transAxes,horizontalalignment='right')


	fig1.tight_layout()
	fig1.savefig(outname1,dpi=dpi)

	fig2.tight_layout()
	fig2.savefig(outname2,dpi=dpi)
	plt.close()

	return norm_errs, norm_flag

def obs_vs_prosp_sfr(e_pinfo,hflag,outname='test.png'):

	#### pull out observed Halpha, observed Balmer decrement ####
	# Make same cuts as Balmer decrement calculation
	keep_idx = brown_quality_cuts.halpha_cuts(e_pinfo)
	
	# halpha
	f_ha = e_pinfo['obs']['f_ha'][keep_idx,0]
	mod_ha = e_pinfo['prosp']['cloudy_ha'][keep_idx,0]

	# Balmer decrements
	bdec_obs = e_pinfo['obs']['bdec'][keep_idx]
	bdec_mod = e_pinfo['prosp']['bdec_cloudy_marg'][keep_idx]

	#### AGN+Herschel identifiers ######
	sfing, composite, agn = return_agn_str(keep_idx)
	keys = [sfing, composite, agn]
	hflag = [hflag[keep_idx],~hflag[keep_idx]]


	#### turn Balmer decrement into tau(lambda = 6563) #####
	# we'll do this by adjusting dust1 ONLY
	# try adjusting by dust2 ONLY later, see if results change
	# start by defining wavelengths and pulling out galaxy parameters
	ha_lam = 6562.801
	hb_lam = 4861.363
	d1 = e_pinfo['prosp']['d1'][keep_idx,0]
	d2 = e_pinfo['prosp']['d2'][keep_idx,0]
	didx = e_pinfo['prosp']['didx'][keep_idx,0]

	ha_ext_obs = np.zeros(len(d1))
	ha_ext_mod = np.zeros(len(d1))
	for ii in xrange(len(d1)):
		# create dust1 test array
		dust_test = np.linspace(0.0,4.0,2000)
		balm_dec_test = 2.86*threed_dutils.charlot_and_fall_extinction(ha_lam,dust_test,d2[ii],-1.0,didx[ii],kriek=True) / \
		                     threed_dutils.charlot_and_fall_extinction(hb_lam,dust_test,d2[ii],-1.0,didx[ii],kriek=True)

		# pull out dust1 that best matches observed Balmer decrement
		# note that this will violate my model priors on dust1/dust2 if it wants to
		d1_new = dust_test[(np.abs(bdec_obs[ii] - balm_dec_test)).argmin()]

		# calculate extinction at Halpha
		ha_ext_obs[ii] = threed_dutils.charlot_and_fall_extinction(ha_lam,d1_new,d2[ii],-1.0,didx[ii],kriek=True)
		ha_ext_mod[ii] = threed_dutils.charlot_and_fall_extinction(ha_lam,d1[ii],d2[ii],-1.0,didx[ii],kriek=True)


	#### correct observed Halpha into intrinsic Halpha, and into CGS
	f_ha_corr = f_ha / ha_ext_obs * constants.L_sun.cgs.value

	#### compute observed SFR, model SFR
	obs_sfr = f_ha_corr / (1.7*1.26e41)
	mod_sfr = e_pinfo['prosp']['sfr_10'][keep_idx,0]

	#### compute balmer decrement residuals
	# obs - model
	bdec_resid = e_pinfo['obs']['bdec'][keep_idx] - e_pinfo['prosp']['bdec_cloudy_marg'][keep_idx,0]

	#### plot
	fig, ax = plt.subplots(1,3, figsize = (18.75,6))

	# loop and put points on plot
	for ii in xrange(len(labels)):
		for kk in xrange(len(hflag)):

			### update colors, define sample
			pdict = merge_dicts(herschdict[kk],bptdict[ii])
			plt_idx = keys[ii] & hflag[kk]

			ax[0].errorbar(bdec_resid[plt_idx], obs_sfr[plt_idx]/mod_sfr[plt_idx],
				           linestyle=' ',**pdict)
			ax[1].errorbar(ha_ext_mod[plt_idx]/ha_ext_obs[plt_idx], obs_sfr[plt_idx]/mod_sfr[plt_idx],
				           linestyle=' ',**pdict)
			ax[2].errorbar(ha_ext_mod[plt_idx]/ha_ext_obs[plt_idx], f_ha[plt_idx]/mod_ha[plt_idx],
				           linestyle=' ',**pdict)

	ax[0].text(0.04,0.92, r'S/N (H$\alpha$,H$\beta$) > '+str(int(e_pinfo['obs']['sn_cut'])), transform = ax[0].transAxes,horizontalalignment='left')
	#ax[0].text(0.04,0.92, r'EQW (H$\alpha$,H$\beta$) > '+str(int(e_pinfo['obs']['eqw_cut'])), transform = ax[0].transAxes,horizontalalignment='left')
	ax[0].set_xlabel(r'(obs-model) [Balmer decrement]')
	ax[0].set_ylabel(r'SFR$_{\mathrm{obs}}$(H$\alpha$)/SFR$_{\mathrm{model}}$(10 Myr)')
	ax[0].set_ylim(0.0,4.0)

	ax[1].errorbar([0.0,4.0],[0.0,4.0],linestyle='--',alpha=0.5,color='0.5')
	ax[1].set_xlabel(r'$e^{-\tau_{mod}(\mathrm{H}\alpha)} / e^{-\tau_{obs}(\mathrm{H}\alpha)}$')
	ax[1].set_ylabel(r'SFR$_{\mathrm{obs}}$(H$\alpha$)/SFR$_{\mathrm{model}}$(10 Myr)')
	ax[1].set_ylim(0.0,4.0)
	ax[1].set_xlim(0.0,4.0)

	ax[2].errorbar([0.0,3.3],[0.0,3.3],linestyle='--',alpha=0.5,color='0.5')
	ax[2].set_xlabel(r'$e^{-\tau_{mod}(\mathrm{H}\alpha)} / e^{-\tau_{obs}(\mathrm{H}\alpha)}$')
	ax[2].set_ylabel(r'H$_{\alpha}$(obs)/H$_{\alpha}$(mod)')
	ax[2].set_ylim(0.0,3.3)
	ax[2].set_xlim(0.0,3.3)

	plt.savefig(outname, dpi=dpi)
	plt.close()

def return_agn_str(idx, string=False):

	'''
	# OLD VERSION
	from astropy.io import fits
	hdulist = fits.open(os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/table1.fits')
	agn_str = hdulist[1].data['Class']
	hdulist.close()
	'''
	# NEW VERSION, WITH MY FLUXES
	with open(os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/joel_bpt.pickle', "rb") as f:
		agn_str=pickle.load(f)

	agn_str = agn_str[idx]
	sfing = (agn_str == 'SF') | (agn_str == '---')
	composite = (agn_str == 'SF/AGN')
	agn = agn_str == 'AGN'

	if string:
		return agn_str
	else:
		return sfing, composite, agn

def residual_plots(e_pinfo,hflag,outfolder):

	fldr = outfolder+'residuals/'
	
	keep_idx = brown_quality_cuts.halpha_cuts(e_pinfo)

	mass = np.log10(e_pinfo['prosp']['mass'][keep_idx,0])
	sfr_100 = np.log10(e_pinfo['prosp']['sfr_100'][keep_idx,0])
	inclination = e_pinfo['obs']['inclination'][keep_idx]

	#### AGN+Herschel identifiers ######
	sfing, composite, agn = return_agn_str(keep_idx)
	keys = [sfing, composite, agn]
	hflag = [hflag[keep_idx],~hflag[keep_idx]]

	#### bdec resid versus ha resid
	bdec_resid = bdec_to_ext(e_pinfo['prosp']['bdec_cloudy_marg'][keep_idx,0])-bdec_to_ext(e_pinfo['obs']['bdec'][keep_idx])
	bdec_resid_errup_prosp = bdec_to_ext(e_pinfo['prosp']['bdec_cloudy_marg'][keep_idx,1])-bdec_to_ext(e_pinfo['prosp']['bdec_cloudy_marg'][keep_idx,0])
	bdec_resid_errup_obs = bdec_to_ext(e_pinfo['obs']['bdec'][keep_idx]+e_pinfo['obs']['bdec_err'][keep_idx])-bdec_to_ext(e_pinfo['obs']['bdec'][keep_idx])
	bdec_resid_errup = np.sqrt(bdec_resid_errup_prosp**2+bdec_resid_errup_obs**2)
	bdec_resid_errdo_prosp = bdec_to_ext(e_pinfo['prosp']['bdec_cloudy_marg'][keep_idx,0])-bdec_to_ext(e_pinfo['prosp']['bdec_cloudy_marg'][keep_idx,2])
	bdec_resid_errdo_obs = bdec_to_ext(e_pinfo['obs']['bdec'][keep_idx])-bdec_to_ext(e_pinfo['obs']['bdec'][keep_idx]-+e_pinfo['obs']['bdec_err'][keep_idx])
	bdec_resid_errdo = np.sqrt(bdec_resid_errdo_prosp**2+bdec_resid_errdo_obs**2)

	bdec_resid_err = [bdec_resid_errdo,bdec_resid_errup]

	ha_resid = minlog(e_pinfo['obs']['f_ha'][keep_idx,0]) - minlog(e_pinfo['prosp']['cloudy_ha'][keep_idx,0])

	fig, ax = plt.subplots(1,1, figsize = (8,8))

	xplot = ha_resid
	yplot = bdec_resid
	for ii in xrange(len(labels)): ax.errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax.set_xlabel(r'log(observed/model) [H$_{\alpha}$]')
	ax.set_ylabel(r'A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$ [model - observed]')
	ax.axis((-0.8,0.8,-1.0,1.0))
	ax.axhline(0, linestyle=':', color='grey')
	ax.axvline(0, linestyle=':', color='grey')

	plt.savefig(fldr+'bdec_resid_versus_ha_resid.png', dpi=dpi)
	plt.close()

	fig, ax = plt.subplots(1,2, figsize = (18,8))
	
	xplot = e_pinfo['prosp']['d1_d2'][keep_idx,0]
	yplot = bdec_resid
	for ii in xrange(len(labels)):
		ax[0].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[0].axhline(0, linestyle=':', color='grey')
	ax[0].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))
	ax[0].set_xlabel(r'dust1/dust2')
	ax[0].set_ylabel(r'A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$ [model - observed]')

	yplot = ha_resid
	for ii in xrange(len(labels)):
		ax[1].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[1].axhline(0, linestyle=':', color='grey')
	ax[1].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))	
	ax[1].set_xlabel(r'dust1/dust2')
	ax[1].set_ylabel(r'log(observed/model) [H$_{\alpha}$]')
	ax[1].legend()
	
	plt.savefig(fldr+'dust1_dust2_residuals.png', dpi=dpi)
	plt.close()

	#### dust2_index
	fig, ax = plt.subplots(1,2, figsize = (18,8))

	xplot = e_pinfo['prosp']['didx'][keep_idx,0]
	yplot = bdec_resid
	for ii in xrange(len(labels)):
		ax[0].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[0].set_xlabel(r'diffuse dust index')
	ax[0].set_ylabel(r'A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$ [model - observed]')
	ax[0].axhline(0, linestyle=':', color='grey')
	ax[0].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))

	yplot = ha_resid
	for ii in xrange(len(labels)):
		ax[1].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[1].set_xlabel('diffuse dust index')
	ax[1].set_ylabel(r'log(observed/model) [H$_{\alpha}$]')
	ax[1].legend(loc=3)
	ax[1].axhline(0, linestyle=':', color='grey')
	ax[1].set_ylim(-np.max(np.abs(yplot)*1.05),np.max(np.abs(yplot)*1.05))
	
	plt.savefig(fldr+'dust_index_residuals.png', dpi=dpi)
	plt.close()

	#### mass residuals
	fig, ax = plt.subplots(1,2,figsize=(18,8))
	for ii in xrange(len(labels)):
		ax[0].errorbar(mass[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[0].set_xlabel(r'log(M)')
	ax[0].set_ylabel(r'log(observed/model) [H$_{\alpha}$]')
	ax[0].axhline(0, linestyle=':', color='grey')
	ax[0].set_ylim(-np.max(np.abs(yplot)*1.05),np.max(np.abs(yplot)*1.05))

	for ii in xrange(len(labels)):
		ax[1].errorbar(mass[keys[ii]], bdec_resid[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[1].set_xlabel(r'log(M)')
	ax[1].set_ylabel(r'A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$ [model - measured]')
	ax[1].axhline(0, linestyle=':', color='grey')
	ax[1].set_ylim(-np.max(np.abs(bdec_resid)*1.05),np.max(np.abs(bdec_resid)*1.05))
	
	ax[0].set_ylim(-0.4,0.4)
	ax[1].set_ylim(-0.5,0.5)

	plt.savefig(fldr+'mass_residuals.png', dpi=dpi)
	plt.close()

	#### total attenuation at 5500 angstroms
	fig, ax = plt.subplots(1,2, figsize = (18,8))

	xplot = e_pinfo['prosp']['d1'][keep_idx,0] + e_pinfo['prosp']['d2'][keep_idx,0]
	yplot = bdec_resid
	for ii in xrange(len(labels)):
		ax[0].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[0].set_xlabel(r'total attenuation [5500 $\AA$]')
	ax[0].set_ylabel(r'A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$ [model - measured]')
	ax[0].legend(loc=4)
	ax[0].axhline(0, linestyle=':', color='grey')
	ax[0].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))

	yplot = ha_resid
	for ii in xrange(len(labels)):
		ax[1].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii],)
	ax[1].set_xlabel(r'total attenuation [5500 $\AA$]')
	ax[1].set_ylabel(r'log(Prospector/obs) [H$_{\alpha}$]')
	ax[1].axhline(0, linestyle=':', color='grey')
	ax[1].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))
	
	plt.savefig(fldr+'total_attenuation_residuals.png', dpi=dpi)
	plt.close()

	#### sfr_100 residuals
	fig, ax = plt.subplots(1,2, figsize = (18,8))

	xplot = sfr_100
	yplot = bdec_resid
	for ii in xrange(len(labels)):
		ax[0].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[0].set_xlabel(r'SFR$_{100 \mathrm{ Myr}}$ [M$_{\odot}$/yr]')
	ax[0].set_ylabel(r'A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$ [model - measured]')
	ax[0].legend(loc=3)
	ax[0].axhline(0, linestyle=':', color='grey')
	ax[0].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))

	yplot = ha_resid
	for ii in xrange(len(labels)):
		ax[1].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[1].set_xlabel(r'SFR$_{100 \mathrm{ Myr}}$ [M$_{\odot}$/yr]')
	ax[1].set_ylabel(r'log(Prospector/obs) [H$_{\alpha}$]')
	ax[1].axhline(0, linestyle=':', color='grey')
	ax[1].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))
	
	plt.savefig(fldr+'sfr_100_residuals.png', dpi=dpi)
	plt.close()

	#### dust2_index vs dust2
	fig, ax = plt.subplots(1,1, figsize = (6,6))

	dtau_dlam_prosp = e_pinfo['prosp']['dtau_dlam']
	dust2_prosp = e_pinfo['prosp']['d2']

	#### now calculate it for the chevallard+13 extinction curve
	modtau = np.linspace(0.0,2.0,100)
	tau1 = threed_dutils.chev_extinction(modtau, lam1_extdiff,ebars=True)
	tau2 = threed_dutils.chev_extinction(modtau, lam2_extdiff,ebars=True)
	dtau_dlam_chev = ((tau2-tau1) / (lam2_extdiff-lam1_extdiff)) / modtau

	sfingn, compositen, agnn = return_agn_str(np.ones_like(keep_idx))
	nkeys = [sfingn, compositen, agnn]
	for ii in xrange(len(labels)): 
		dtau_dlam_prosp_errors = threed_dutils.asym_errors(dtau_dlam_prosp[nkeys[ii],0],
			                                               dtau_dlam_prosp[nkeys[ii],1],
			                                               dtau_dlam_prosp[nkeys[ii],2],
			                                               log=False)
		dust2_errors = threed_dutils.asym_errors(dust2_prosp[nkeys[ii],0],
			                                     dust2_prosp[nkeys[ii],1],
			                                     dust2_prosp[nkeys[ii],2],
			                                     log=False)
		ax.errorbar(dust2_prosp[nkeys[ii],0], dtau_dlam_prosp[nkeys[ii],0], xerr=dust2_errors,yerr=dtau_dlam_prosp_errors, 
			        linestyle=' ', **merge_dicts(herschdict[0],bptdict[ii]))

	lw=2
 	ax.plot(modtau,dtau_dlam_chev[0,:],color='k',lw=lw)
 	ax.plot(modtau,dtau_dlam_chev[1,:],color='k',linestyle='--',lw=lw)
 	ax.plot(modtau,dtau_dlam_chev[2,:],color='k',linestyle='--',lw=lw)

	ax.set_xlabel('diffuse dust optical depth')
	ax.set_ylabel(r'dlog($\tau_{\mathrm{diffuse}}$)/d$\lambda$')

	ax.set_xlim(-0.1,2.0)
	ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

	ax.text(0.98,0.04, 'radiative transfer models \n from Chevallard+13', ha='right',va='bottom',multialignment='right',\
		    transform=ax.transAxes,fontsize=15,weight='roman')

	ax.xaxis.set_major_locator(MaxNLocator(5))
	plt.tight_layout()
	plt.savefig(fldr+'dust_properties.png', dpi=dpi)
	plt.close()

	#### inclination
	fig, ax = plt.subplots(1,1, figsize = (8,8))

	xplot = inclination
	yplot = bdec_resid
	for ii in xrange(len(labels)): ax.errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax.axhline(0, linestyle=':', color='grey')
	ax.set_xlabel(r'inclination [degrees]')
	ax.set_ylabel(r'A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$ [model - measured]')
	ax.set_ylim(-0.6,0.6)
	ax.set_xlim(-5,95)

	off,scat = threed_dutils.offset_and_scatter(bdec_to_ext(e_pinfo['prosp']['bdec_cloudy_marg'][keep_idx,0]),bdec_to_ext(e_pinfo['obs']['bdec'][keep_idx]))
	ax.text(0.98,0.94, 'biweight scatter='+"{:.2f}".format(scat)+' magnitudes', ha='right',transform=ax.transAxes,fontsize=15,weight='roman')

	ax.arrow(0.06, 0.52, 0.0, 0.10, head_width=0.02, head_length=0.03, fc='k', ec='k',width=0.003,transform = ax.transAxes)
	ax.arrow(0.06, 0.48, 0.0,-0.10, head_width=0.02, head_length=0.03, fc='k', ec='k',width=0.003,transform = ax.transAxes)

	ax.text(0.08, 0.60, 'too much \n model dust',transform = ax.transAxes,horizontalalignment='left',fontsize=14,multialignment='center',weight='roman')
	ax.text(0.08, 0.355, 'too little \n model dust',transform = ax.transAxes,horizontalalignment='left',fontsize=14,multialignment='center',weight='roman')

	ax.arrow(0.52, 0.04, 0.15, 0.00, head_width=0.02, head_length=0.03, fc='k', ec='k',width=0.003,transform = ax.transAxes)
	ax.arrow(0.48, 0.04, -0.15,-0.00, head_width=0.02, head_length=0.03, fc='k', ec='k',width=0.003,transform = ax.transAxes)

	ax.text(0.52, 0.06, 'more edge-on',transform = ax.transAxes,horizontalalignment='left',fontsize=14,multialignment='center',weight='roman')
	ax.text(0.48, 0.06, 'more face-on',transform = ax.transAxes,horizontalalignment='right',fontsize=14,multialignment='center',weight='roman')

	plt.savefig(fldr+'inclination_bdecresid.png', dpi=dpi)
	plt.close()

	fig, ax = plt.subplots(1,1, figsize = (7,7))

	xplot = e_pinfo['obs']['inclination']
	yplot = e_pinfo['prosp']['d1_d2'][:,0]
	for ii in xrange(len(labels)): 
		errors = threed_dutils.asym_errors(e_pinfo['prosp']['d1_d2'][nkeys[ii],0],e_pinfo['prosp']['d1_d2'][nkeys[ii],1],e_pinfo['prosp']['d1_d2'][nkeys[ii],2],log=False)
		ax.errorbar(xplot[nkeys[ii]], yplot[nkeys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax.axhline(0, linestyle=':', color='grey')
	ax.set_xlabel(r'inclination [degrees]')
	ax.set_ylabel(r'$\tau_{\mathrm{birth-cloud}}/\tau_{\mathrm{diffuse}}$')
	ax.set_ylim(0.0,1.0)
	ax.set_xlim(-5,95)

	ax.arrow(0.52, 0.04, 0.25, 0.00, head_width=0.02, head_length=0.03, fc='k', ec='k',width=0.003,transform = ax.transAxes)
	ax.arrow(0.48, 0.04, -0.25,-0.00, head_width=0.02, head_length=0.03, fc='k', ec='k',width=0.003,transform = ax.transAxes)

	ax.text(0.52, 0.06, 'more edge-on',transform = ax.transAxes,horizontalalignment='left',fontsize=14,multialignment='center',weight='roman')
	ax.text(0.48, 0.06, 'more face-on',transform = ax.transAxes,horizontalalignment='right',fontsize=14,multialignment='center',weight='roman')


	plt.savefig(fldr+'inclination_d1d2.png', dpi=dpi)
	plt.close()


	#### inclination part 2
	fig, ax = plt.subplots(2,3, figsize = (18.75,12))
	fig.subplots_adjust(wspace=0.25,left=0.075,right=0.95,bottom=0.075,top=0.95)
	ax = np.ravel(ax)

	xplot = e_pinfo['obs']['inclination']

	yplot = e_pinfo['prosp']['d1_d2'][:,0]
	for ii in xrange(len(labels)): ax[0].errorbar(xplot[nkeys[ii]], yplot[nkeys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[0].set_xlabel(r'inclination [degrees]')
	ax[0].set_ylabel(r'dust1/dust2')

	yplot = e_pinfo['prosp']['didx'][:,0]
	for ii in xrange(len(labels)): ax[1].errorbar(xplot[nkeys[ii]], yplot[nkeys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[1].set_xlabel(r'inclination[degrees]')
	ax[1].set_ylabel(r'dust2_index')

	yplot = e_pinfo['prosp']['d1'][:,0]+e_pinfo['prosp']['d2'][:,0]
	for ii in xrange(len(labels)): ax[2].errorbar(xplot[nkeys[ii]], yplot[nkeys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[2].set_xlabel(r'inclination[degrees]')
	ax[2].set_ylabel(r'dust1+dust2')

	yplot = e_pinfo['prosp']['d2'][:,0]
	for ii in xrange(len(labels)): ax[3].errorbar(xplot[nkeys[ii]], yplot[nkeys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[3].set_xlabel(r'inclination[degrees]')
	ax[3].set_ylabel(r'dust2')

	yplot = ha_resid
	xplot = inclination
	for ii in xrange(len(labels)): ax[4].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[4].axhline(0, linestyle=':', color='grey')
	ax[4].set_xlabel(r'inclination[degrees]')
	ax[4].set_ylabel(r'log(Prospector/obs) [H$_{\alpha}$]')

	yplot = bdec_resid
	xplot = inclination
	for ii in xrange(len(labels)): ax[5].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[5].axhline(0, linestyle=':', color='grey')
	ax[5].set_xlabel(r'inclination[degrees]')
	ax[5].set_ylabel(r'A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$ [model - measured]')
	ax[5].set_ylim(-0.5,0.5)

	#plt.tight_layout()
	plt.savefig(fldr+'inclination_full.png', dpi=dpi)
	plt.close()

def plot_emline_comp(alldata,outfolder,hflag):
	'''
	emission line luminosity comparisons:
		(1) Observed luminosity, Prospector vs MAGPHYS continuum subtraction
		(2) Moustakas+10 comparisons
		(3) model Balmer decrement (from dust) versus observed Balmer decrement
		(4) model Halpha (from Kennicutt + dust) versus observed Halpha
	'''

	##### Pull relevant information out of alldata
	emline_names = alldata[0]['residuals']['emlines']['em_name']

	##### load moustakas+10 line flux information
	objnames = np.array([f['objname'] for f in alldata])
	dat = threed_dutils.load_moustakas_data(objnames = list(objnames))

	##### load new moustakas line flux information (from email, january 2016)
	newdat = threed_dutils.load_moustakas_newdat(objnames = list(objnames))

	'''
	##### plots, one by one
	# observed line fluxes, with MAGPHYS / Prospector continuum
	compare_model_flux(alldata,emline_names,outname = outfolder+'continuum_model_flux_comparison.png')

	# observed line fluxes versus Moustakas+10
	# TURN THIS BACK ON AND FIX IT
	compare_moustakas_fluxes(alldata,dat,emline_names,objnames,
							 outname=outfolder+'moustakas_flux_comp.png',
							 outdec=outfolder+'moustakas_bdec_comp.png')
	compare_moustakas_newfluxes(alldata,newdat,np.array(['[OIII] 5007','H$\\beta$','[NII] 6583','H$\\alpha$']),objnames,
							    outname=outfolder+'moustakas_flux_newcomp.png',
							    outdec=outfolder+'moustakas_bdec_comp.png')
	'''
	##### format emission line data for plotting
	e_pinfo = fmt_emline_info(alldata)

	##### add in 'location in truth' PDF
	delta_functions = True # for just the median of the observational posteriors
	center_obs = False # to center the observations at zero too
	if delta_functions:
		outname = outfolder+'posterior_PDF_delta_functions.png'
	elif center_obs: 
		outname = outfolder+'posterior_PDF_obscenter.png'
	else:
		outname = outfolder+'posterior_PDF.png'

	pdf = specpar_pdf_distance(e_pinfo,alldata,delta_functions=delta_functions,center_obs=center_obs)
	specpar_pdf_plot(pdf,outname=outname)

	# errors
	eline_errs(e_pinfo,hflag,outname=outfolder+'error_sig.png')

	# dn4000 and halpha paper plots
	paper_summary_plot(e_pinfo, hflag, outname=outfolder+'paper_summary_plot.png')

	# gas-phase versus stellar metallicity
	atlas_3d_met(e_pinfo, hflag,outfolder=outfolder)
	gas_phase_metallicity(e_pinfo, hflag, outfolder=outfolder)

	# model versus observations, Balmer decrement
	bdec_errs,bdec_flag = obs_vs_model_bdec(e_pinfo, hflag, outname1=outfolder+'bdec_comparison.png',outname2=outfolder+'prospector_bdec_comparison.png')

	# model versus observations, Hdelta
	dn4000_errs,dn4000_flag = obs_vs_model_dn(e_pinfo, hflag, 
		                                      outname=outfolder+'dn4000_comp.png')
	hdelta_errs,hdelta_flag = obs_vs_model_hdelta(e_pinfo, hflag, 
		                      outname=outfolder+'hdelta_comp_eqw.png',
		                      outname_dnplt=outfolder+'hdelta_dn_comp.png',
		                      eqw=True)
	_,_ = obs_vs_model_hdelta(e_pinfo, hflag, 
		                      outname=outfolder+'hdelta_comp_flux.png',
		                      eqw=False)

	# model versus observations, Halpha + Hbeta
	ha_errs,ha_flag = obs_vs_prosp_balmlines(e_pinfo,hflag,
								 outname=outfolder+'balmer_line_comp.png',
								 outname_resid=outfolder+'balmer_line_resid.png')

	# model SFR versus observed SFR(Ha) corrected for dust attenuation
	obs_vs_prosp_sfr(e_pinfo,hflag,outname=outfolder+'obs_sfr_comp.png')

	# error plots
	# onesig_error_plot(bdec_errs,bdec_flag,dn4000_errs,dn4000_flag,hdelta_errs,hdelta_flag,ha_errs,ha_flag,outfolder)

	# model versus observations for Kennicutt Halphas
	obs_vs_kennicutt_ha(e_pinfo,hflag, eqw=False,
		                outname_prosp=outfolder+'empirical_halpha_prosp.png',
		                outname_mag=outfolder+'empirical_halpha_mag.png',
		                outname_cloudy=outfolder+'empirical_halpha_versus_cloudy.png',
		                outname_ha_inpt=outfolder+'kennicutt_ha_input.png',
		                outname_sfr_margcomp=outfolder+'sfr_margcomp.png')

	obs_vs_kennicutt_ha(e_pinfo,hflag, eqw=True,
		                outname_prosp=outfolder+'empirical_halpha_prosp_eqw.png',
		                outname_mag=outfolder+'empirical_halpha_mag_eqw.png',
		                outname_cloudy=outfolder+'empirical_halpha_versus_cloudy_eqw.png',
		                outname_ha_inpt=outfolder+'kennicutt_ha_input.png',
		                outname_sfr_margcomp=outfolder+'sfr_margcomp.png')

	# model versus observations for BPT diagram
	bpt_diagram(e_pinfo,hflag,outname=outfolder+'bpt.png')

	residual_plots(e_pinfo,hflag,outfolder)

def plot_relationships(alldata,outfolder):

	'''
	mass-metallicity
	mass-SFR
	etc
	'''

	##### set up plots
	fig = plt.figure(figsize=(13,6.0))
	gs1 = mpl.gridspec.GridSpec(1, 2)
	msfr = plt.Subplot(fig, gs1[0])
	mz = plt.Subplot(fig, gs1[1])

	fig.add_subplot(msfr)
	fig.add_subplot(mz)

	alpha = 0.6
	ms = 6.0

	##### find prospector indexes
	parnames = alldata[0]['pquantiles']['parnames']
	idx_logmass = parnames == 'logmass'
	idx_met = parnames == 'logzsol'

	eparnames = alldata[0]['pextras']['parnames']
	idx_sfr = eparnames == 'sfr_100'

	##### find magphys indexes
	idx_mmet = alldata[0]['model']['full_parnames'] == 'Z/Zo'

	##### extract mass, SFR, metallicity
	magmass, promass, magsfr, prosfr, promet = [np.empty(shape=(0,3)) for x in xrange(5)]
	magmet = np.empty(0)
	for data in alldata:
		if data:
			
			# mass
			tmp = np.array([data['pquantiles']['q16'][idx_logmass][0],
				            data['pquantiles']['q50'][idx_logmass][0],
				            data['pquantiles']['q84'][idx_logmass][0]])
			promass = np.concatenate((promass,np.atleast_2d(tmp)),axis=0)
			magmass = np.concatenate((magmass,np.atleast_2d(data['magphys']['percentiles']['M*'][1:4])))

			# SFR
			tmp = np.array([data['pextras']['q16'][idx_sfr][0],
				            data['pextras']['q50'][idx_sfr][0],
				            data['pextras']['q84'][idx_sfr][0]])
			tmp = np.log10(np.clip(tmp,minsfr,np.inf))
			prosfr = np.concatenate((prosfr,np.atleast_2d(tmp)))
			magsfr = np.concatenate((magsfr,np.atleast_2d(data['magphys']['percentiles']['SFR'][1:4])))

			# metallicity
			tmp = np.array([data['pquantiles']['q16'][idx_met][0],
				            data['pquantiles']['q50'][idx_met][0],
				            data['pquantiles']['q84'][idx_met][0]])
			promet = np.concatenate((promet,np.atleast_2d(tmp)))
			magmet = np.concatenate((magmet,np.log10(np.atleast_1d(data['model']['full_parameters'][idx_mmet][0]))))

	##### Errors on Prospector+MAGPHYS quantities
	# mass errors
	proerrs_mass = [promass[:,1]-promass[:,0],
	                promass[:,2]-promass[:,1]]
	magerrs_mass = [magmass[:,1]-magmass[:,0],
	                magmass[:,2]-magmass[:,1]]

	# SFR errors
	proerrs_sfr = [prosfr[:,1]-prosfr[:,0],
	               prosfr[:,2]-prosfr[:,1]]
	magerrs_sfr = [magsfr[:,1]-magsfr[:,0],
	               magsfr[:,2]-magsfr[:,1]]

	# metallicity errors
	proerrs_met = [promet[:,1]-promet[:,0],
	               promet[:,2]-promet[:,1]]


	##### STAR-FORMING SEQUENCE #####
	msfr.errorbar(promass[:,1],prosfr[:,1],
		          fmt='o', alpha=alpha,
		          color=prosp_color,
		          label='Prospector',
			      xerr=proerrs_mass, yerr=proerrs_sfr,
			      ms=ms)
	msfr.errorbar(magmass[:,1],magsfr[:,1],
		          fmt='o', alpha=alpha,
		          color=magphys_color,
		          label='MAGPHYS',
			      xerr=magerrs_mass, yerr=magerrs_sfr,
			      ms=ms)

	# Chang et al. 2015
	# + 0.39 dex, -0.64 dex
	chang_color = 'orange'
	chang_mass = np.linspace(7,12,40)
	chang_sfr = 0.8 * np.log10(10**chang_mass/1e10) - 0.23
	chang_scatlow = 0.64
	chang_scathigh = 0.39

	msfr.plot(chang_mass, chang_sfr,
		          color=chang_color,
		          lw=2.5,
		          label='Chang+15',
		          zorder=-1)

	msfr.fill_between(chang_mass, chang_sfr-chang_scatlow, chang_sfr+chang_scathigh, 
		                  color=chang_color,
		                  alpha=0.3)


	#### Salim+07
	ssfr_salim = -0.35*(chang_mass-10)-9.83
	salim_sfr = np.log10(10**ssfr_salim*10**chang_mass)

	msfr.plot(chang_mass, salim_sfr,
		          color='green',
		          lw=2.5,
		          label='Salim+07',
		          zorder=-1)

	# legend
	msfr.legend(loc=2, prop={'size':12},
			    frameon=False)

	msfr.set_xlabel(r'log(M/M$_{\odot}$)')
	msfr.set_ylabel(r'log(SFR/M$_{\odot}$/yr)')

	##### MASS-METALLICITY #####
	mz.errorbar(promass[:,1],promet[:,1],
		          fmt='o', alpha=alpha,
		          color=prosp_color,
			      xerr=proerrs_mass, yerr=proerrs_met,
			      ms=ms)
	mz.errorbar(magmass[:,1],magmet,
		          fmt='o', alpha=alpha,
		          color=magphys_color,
			      xerr=magerrs_mass,
			      ms=ms)	

	# Gallazzi+05
	# shape: mass q50 q16 q84
	# IMF is probably Kroupa, though not stated in paper
	# must add correction...
	massmet = np.loadtxt(os.getenv('APPS')+'/threedhst_bsfh/data/gallazzi_05_massmet.txt')

	mz.plot(massmet[:,0], massmet[:,1],
		          color='green',
		          lw=2.5,
		          label='Gallazzi+05',
		          zorder=-1)

	mz.fill_between(massmet[:,0], massmet[:,2], massmet[:,3], 
		                  color='green',
		                  alpha=0.3)


	# legend
	mz.legend(loc=4, prop={'size':12},
			    frameon=False)

	mz.set_ylim(-2.0,0.3)
	mz.set_xlim(9,11.8)
	mz.set_xlabel(r'log(M/M$_{\odot}$)')
	mz.set_ylabel(r'log(Z/Z$_{\odot}$/yr)')

	plt.savefig(outfolder+'relationships.png',dpi=dpi)
	plt.close

def prospector_comparison(alldata,outfolder,hflag):

	'''
	For Prospector:
	dust_index versus total attenuation
	dust_index versus SFR
	dust1 versus dust2, everything below -0.45 dust index highlighted
	PAH fraction versus dust_index
	'''
	
	# if it doesn't exist, make it
	if not os.path.isdir(outfolder):
		os.makedirs(outfolder)

	#### find prospector indexes
	parnames = alldata[0]['pquantiles']['parnames']
	idx_mass = parnames == 'logmass'
	didx_idx = parnames == 'dust_index'
	d1_idx = parnames == 'dust1'
	d2_idx = parnames == 'dust2'
	qpah_idx = parnames == 'duste_qpah'
	met_idx = parnames == 'logzsol'

	#### agn flags
	sfing, composite, agn = return_agn_str(np.ones_like(hflag,dtype=bool))
	agn_flags = [sfing,composite,agn]

	#### best-fits
	d1 = np.array([x['bfit']['maxprob_params'][d1_idx][0] for x in alldata])
	d2 = np.array([x['bfit']['maxprob_params'][d2_idx][0] for x in alldata])
	didx = np.array([x['bfit']['maxprob_params'][didx_idx][0] for x in alldata])
	sfr_100 = np.log10(np.clip([x['bfit']['sfr_100'] for x in alldata],1e-4,np.inf))
	sfr_100_marginalized = np.log10(np.clip([x['pextras']['q50'][x['pextras']['parnames'] == 'sfr_100'] for x in alldata],1e-4,np.inf))
	sfr_10_marginalized = np.log10(np.clip([x['pextras']['q50'][x['pextras']['parnames'] == 'sfr_10'] for x in alldata],1e-4,np.inf))
	sfr_10 = np.log10(np.clip([x['bfit']['sfr_10'] for x in alldata],1e-4,np.inf))

	#### total attenuation
	dusttot = np.zeros_like(d1)
	for ii in xrange(len(dusttot)): dusttot[ii] = -np.log10(threed_dutils.charlot_and_fall_extinction(5500.,d1[ii],d2[ii],-1.0,didx[ii], kriek=True))

	#### plotting series of comparisons
	fig, ax = plt.subplots(2,3, figsize = (22,12))

	ax[0,0].errorbar(didx,dusttot, fmt='o',alpha=0.6,linestyle=' ',color='0.4')
	ax[0,0].set_ylabel(r'total attenuation [5500 $\AA$]')
	ax[0,0].set_xlabel(r'dust_index')
	ax[0,0].axis((didx.min()-0.1,didx.max()+0.1,dusttot.min()-0.1,dusttot.max()+0.1))

	for flag,pdict in zip(agn_flags,bptdict): ax[0,1].errorbar(didx[flag],sfr_100[flag], fmt='o',alpha=0.6,linestyle=' ',**pdict)
	ax[0,1].set_xlabel(r'dust_index')
	ax[0,1].set_ylabel(r'SFR$_{100 \mathrm{ Myr}}$ [M$_{\odot}$/yr]')
	ax[0,1].axis((didx.min()-0.1,didx.max()+0.1,sfr_100.min()-0.1,sfr_100.max()+0.1))
	for ii in xrange(len(bptdict)): ax[0,1].text(0.04,0.92-0.08*ii,bptdict[ii]['label'],color=bptdict[ii].get('color','red'),transform = ax[0,1].transAxes,weight='bold')

	l_idx = didx > -0.45
	h_idx = didx < -0.45
	ax[0,2].errorbar(d1[h_idx],d2[h_idx], fmt='o',alpha=0.6,linestyle=' ',color='0.4')
	ax[0,2].errorbar(d1[l_idx],d2[l_idx], fmt='o',alpha=0.6,linestyle=' ',color='blue')
	ax[0,2].set_xlabel(r'dust1')
	ax[0,2].set_ylabel(r'dust2')
	ax[0,2].axis((d1.min()-0.1,d1.max()+0.1,d2.min()-0.1,d2.max()+0.1))
	ax[1,0].text(0.96,0.05,'dust_index > -0.45',color='blue',transform = ax[0,2].transAxes,weight='bold',ha='right')

	l_idx = didx > -0.45
	h_idx = didx < -0.45
	ax[1,0].errorbar(didx[~hflag],d1[~hflag]/d2[~hflag], fmt='o',alpha=0.6,linestyle=' ', label='no herschel',color='0.4')
	ax[1,0].errorbar(didx[hflag],d1[hflag]/d2[hflag], fmt='o',alpha=0.6,linestyle=' ', label='has herschel',color='red')
	ax[1,0].set_xlabel(r'dust_index')
	ax[1,0].set_ylabel(r'dust1/dust2')
	ax[1,0].axis((didx.min()-0.1,didx.max()+0.1,(d1/d2).min()-0.1,(d1/d2).max()+0.1))
	ax[1,0].text(0.05,0.92,'Herschel-detected',color='red',transform = ax[1,0].transAxes,weight='bold')

	ax[1,1].errorbar(sfr_100_marginalized, sfr_10_marginalized, fmt='o',alpha=0.6,linestyle=' ',color=obs_color)
	ax[1,1].set_xlabel(r'log(SFR [100 Myr]) [marginalized]')
	ax[1,1].set_ylabel(r'log(SFR [10 Myr]) [marginalized]')
	ax[1,1] = threed_dutils.equalize_axes(ax[1,1], sfr_100_marginalized,sfr_10_marginalized)
	off,scat = threed_dutils.offset_and_scatter(sfr_100_marginalized,sfr_10_marginalized,biweight=True)
	ax[1,1].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat),
			  transform = ax[1,1].transAxes,horizontalalignment='right')
	ax[1,1].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off),
			      transform = ax[1,1].transAxes,horizontalalignment='right')

	ax[1,2].errorbar(sfr_10, sfr_10_marginalized, fmt='o',alpha=0.6,linestyle=' ',color=obs_color)
	ax[1,2].set_xlabel(r'log(SFR [10 Myr]) [best-fit]')
	ax[1,2].set_ylabel(r'log(SFR [10 Myr]) [marginalized]')
	ax[1,2] = threed_dutils.equalize_axes(ax[1,2], sfr_10,sfr_10_marginalized)
	off,scat = threed_dutils.offset_and_scatter(sfr_10,sfr_10_marginalized,biweight=True)
	ax[1,2].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat),
			  transform = ax[1,2].transAxes,horizontalalignment='right')
	ax[1,2].text(0.96,0.1, 'mean offset='+"{:.2f}".format(off),
			      transform = ax[1,2].transAxes,horizontalalignment='right')



	plt.savefig(outfolder+'bestfit_param_comparison.png', dpi=dpi)
	plt.close()

	#### qpah plots
	fig, ax = plt.subplots(1,2, figsize = (12.5,6))
	qpah_low = 0.001 # lower plot limit for qpah, must clip errors for presentation purposes

	mass = np.array([x['pquantiles']['q50'][idx_mass][0] for x in alldata])
	m_errup = np.array([x['pquantiles']['q84'][idx_mass][0] for x in alldata])
	m_errdo = np.array([x['pquantiles']['q16'][idx_mass][0] for x in alldata])
	mass_err = threed_dutils.asym_errors(mass,m_errup,m_errdo,log=False)

	logzsol = np.array([x['pquantiles']['q50'][met_idx][0] for x in alldata])
	logzsol_errup = np.array([x['pquantiles']['q84'][met_idx][0] for x in alldata])
	logzsol_errdo = np.array([x['pquantiles']['q16'][met_idx][0] for x in alldata])
	logzsol_err = threed_dutils.asym_errors(logzsol,logzsol_errup,logzsol_errdo,log=False)

	didx = np.array([x['pquantiles']['q50'][didx_idx][0] for x in alldata])
	didx_errup = np.array([x['pquantiles']['q84'][didx_idx][0] for x in alldata])
	didx_errdo = np.array([x['pquantiles']['q16'][didx_idx][0] for x in alldata])
	didx_err = threed_dutils.asym_errors(didx,didx_errup,didx_errdo,log=False)

	qpah = np.array([x['pquantiles']['q50'][qpah_idx][0] for x in alldata])
	qpah_errup = np.array([x['pquantiles']['q84'][qpah_idx][0] for x in alldata])
	qpah_errdo = np.array([x['pquantiles']['q16'][qpah_idx][0] for x in alldata])
	qpah_err = threed_dutils.asym_errors(qpah,qpah_errup,np.clip(qpah_errdo,qpah_low,np.inf),log=True)

	lir, lir_up, lir_do = [],[],[]
	for dat in alldata:
		lir_quant = corner.quantile(dat['lir'], [0.5, 0.84, 0.16])
		lir.append(lir_quant[0])
		lir_up.append(lir_quant[1])
		lir_do.append(lir_quant[2])

	lir_err = threed_dutils.asym_errors(lir,lir_up,lir_do,log=True)
	lir = np.log10(lir)

	ax[0].errorbar(mass,np.log10(qpah),xerr=mass_err,yerr=qpah_err, alpha=0.6, fmt='o', color='#1C86EE')
	ax[0].set_xlabel(r'log(M/M$_{\odot}$)')
	ax[0].set_ylabel(r'log(Q$_{\mathrm{PAH}}$)')
	ax[0].set_ylim(np.log10(qpah_low),np.log10(11.0))

	ax[1].errorbar(lir,np.log10(qpah),xerr=lir_err,yerr=qpah_err, alpha=0.6, fmt='o', color='#1C86EE')
	ax[1].set_xlabel(r'log(L$_{\mathrm{IR}}$)')
	ax[1].set_ylabel(r'log(Q$_{\mathrm{PAH}}$)')
	ax[1].set_ylim(np.log10(0.001),np.log10(11.0))
	ax[1].set_xlim(6.5,12.5)

	plt.tight_layout()
	plt.savefig(outfolder+'qpah_comp.png', dpi=dpi)
	plt.close()

	#### qpah versus dust_index, metallicity
	fig, ax = plt.subplots(1,2, figsize = (12.5,6))

	ax[0].errorbar(logzsol, np.log10(qpah), xerr=logzsol_err, yerr=qpah_err, alpha=0.6, fmt='o', color='#1C86EE')
	ax[0].set_xlabel(r'log(Z/Z$_{\odot}$)')
	ax[0].set_ylabel(r'log(Q$_{\mathrm{PAH}}$)')
	ax[0].set_ylim(np.log10(qpah_low),1)

	ax[1].errorbar(didx, np.log10(qpah), xerr=didx_err, yerr=qpah_err, alpha=0.6, fmt='o', color='#1C86EE')
	ax[1].set_xlabel(r'dust index')
	ax[1].set_ylabel(r'log(Q$_{\mathrm{PAH}}$)')
	ax[1].set_ylim(np.log10(qpah_low),1)

	plt.tight_layout()
	plt.savefig(outfolder+'qpah_comp_additional.png', dpi=dpi)
	plt.close()

def plot_comparison(alldata,outfolder):

	'''
	mass vs mass
	sfr vs sfr
	etc
	'''

	##### set up plots
	fig = plt.figure(figsize=(18,12))
	gs1 = mpl.gridspec.GridSpec(2, 3)
	gs1.update(hspace=0.2,wspace=0.25,left=0.075,right=0.95,bottom=0.075,top=0.95)
	mass = plt.Subplot(fig, gs1[0])
	sfr = plt.Subplot(fig, gs1[1])
	met = plt.Subplot(fig, gs1[2])
	ext_diff = plt.Subplot(fig,gs1[3])
	balm = plt.Subplot(fig,gs1[4])
	ext_tot = plt.Subplot(fig,gs1[5])

	fig.add_subplot(mass)
	fig.add_subplot(sfr)
	fig.add_subplot(met)
	fig.add_subplot(balm)
	fig.add_subplot(ext_tot)
	fig.add_subplot(ext_diff)

	color = '0.6'
	mew = 1.5
	alpha = 0.6
	fmt = 'o'

	##### find prospector indexes
	parnames = alldata[0]['pquantiles']['parnames']
	idx_mass = parnames == 'logmass'
	idx_met = parnames == 'logzsol'
	dinx_idx = parnames == 'dust_index'
	dust1_idx = parnames == 'dust1'
	dust2_idx = parnames == 'dust2'

	eparnames = alldata[0]['pextras']['parnames']
	idx_sfr = eparnames == 'sfr_100'

	##### find magphys indexes
	mparnames = alldata[0]['model']['parnames']
	mfparnames = alldata[0]['model']['full_parnames']
	idx_mmet = mfparnames == 'Z/Zo'
	mu_idx = mparnames == 'mu'
	tauv_idx = mparnames == 'tauv'


	##### mass
	magmass, promass = np.empty(shape=(0,3)), np.empty(shape=(0,3))
	for data in alldata:
		if data:
			tmp = np.array([data['pquantiles']['q16'][idx_mass][0],
				            data['pquantiles']['q50'][idx_mass][0],
				            data['pquantiles']['q84'][idx_mass][0]])
			promass = np.concatenate((promass,np.atleast_2d(tmp)),axis=0)
			magmass = np.concatenate((magmass,np.atleast_2d(data['magphys']['percentiles']['M*'][1:4])))

	proerrs = [promass[:,1]-promass[:,0],
	           promass[:,2]-promass[:,1]]
	magerrs = [magmass[:,1]-magmass[:,0],
	           magmass[:,2]-magmass[:,1]]
	mass.errorbar(promass[:,1],magmass[:,1],
		          fmt=fmt, alpha=alpha,
			      xerr=proerrs, yerr=magerrs, color=color,
			      mew=mew)

	# labels
	mass.set_xlabel(r'log(M$_*$) [Prospector, 100 Myr]',labelpad=13)
	mass.set_ylabel(r'log(M$_*$) [MAGPHYS, 100 Myr]')
	mass = threed_dutils.equalize_axes(mass,promass[:,1],magmass[:,1])

	# text
	off,scat = threed_dutils.offset_and_scatter(promass[:,1],magmass[:,1],biweight=True)
	mass.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
			  transform = mass.transAxes,horizontalalignment='right')
	mass.text(0.96,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
		      transform = mass.transAxes,horizontalalignment='right')

	##### SFR
	magsfr, prosfr = np.empty(shape=(0,3)), np.empty(shape=(0,3))
	for data in alldata:
		if data:
			tmp = np.array([data['pextras']['q16'][idx_sfr][0],
				            data['pextras']['q50'][idx_sfr][0],
				            data['pextras']['q84'][idx_sfr][0]])
			tmp = np.log10(np.clip(tmp,minsfr,np.inf))
			prosfr = np.concatenate((prosfr,np.atleast_2d(tmp)))
			magsfr = np.concatenate((magsfr,np.atleast_2d(data['magphys']['percentiles']['SFR'][1:4])))

	proerrs = [prosfr[:,1]-prosfr[:,0],
	           prosfr[:,2]-prosfr[:,1]]
	magerrs = [magsfr[:,1]-magsfr[:,0],
	           magsfr[:,2]-magsfr[:,1]]
	sfr.errorbar(prosfr[:,1],magsfr[:,1],
		          fmt=fmt, alpha=alpha,
			      xerr=proerrs, yerr=magerrs, color=color,
			      mew=mew)

	# labels
	sfr.set_xlabel(r'log(SFR) [Prospector]')
	sfr.set_ylabel(r'log(SFR) [MAGPHYS]')
	sfr = threed_dutils.equalize_axes(sfr,prosfr[:,1],magsfr[:,1])

	# text
	off,scat = threed_dutils.offset_and_scatter(prosfr[:,1],magsfr[:,1],biweight=True)
	sfr.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
			  transform = sfr.transAxes,horizontalalignment='right')
	sfr.text(0.96,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
		      transform = sfr.transAxes,horizontalalignment='right')

	##### metallicity
	# check that we're using the same solar abundance
	magmet, promet = np.empty(0),np.empty(shape=(0,3))
	for data in alldata:
		if data:
			tmp = np.array([data['pquantiles']['q16'][idx_met][0],
				            data['pquantiles']['q50'][idx_met][0],
				            data['pquantiles']['q84'][idx_met][0]])
			promet = np.concatenate((promet,np.atleast_2d(tmp)))
			magmet = np.concatenate((magmet,np.log10(np.atleast_1d(data['model']['full_parameters'][idx_mmet][0]))))

	proerrs = [promet[:,1]-promet[:,0],
	           promet[:,2]-promet[:,1]]
	met.errorbar(promet[:,1],magmet,
		          fmt=fmt, alpha=alpha,
			      xerr=proerrs, color=color,
			      mew=mew)

	# labels
	met.set_xlabel(r'log(Z/Z$_{\odot}$) [Prospector]',labelpad=13)
	met.set_ylabel(r'log(Z/Z$_{\odot}$) [best-fit MAGPHYS]')
	met = threed_dutils.equalize_axes(met,promet[:,1],magmet)

	# text
	off,scat = threed_dutils.offset_and_scatter(promet[:,1],magmet,biweight=True)
	met.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
			  transform = met.transAxes,horizontalalignment='right')
	met.text(0.96,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
		      transform = met.transAxes,horizontalalignment='right')
	

	#### Balmer decrement
	bdec_magphys, bdec_prospector = [],[]
	for ii,dat in enumerate(alldata):
		tau1 = dat['bfit']['maxprob_params'][dust1_idx][0]
		tau2 = dat['bfit']['maxprob_params'][dust2_idx][0]
		dindex = dat['bfit']['maxprob_params'][dinx_idx][0]
		bdec = threed_dutils.calc_balmer_dec(tau1, tau2, -1.0, dindex,kriek=True)
		bdec_prospector.append(bdec)
		
		tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
		tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
		bdec = threed_dutils.calc_balmer_dec(tau1, tau2, -1.3, -0.7)
		bdec_magphys.append(np.squeeze(bdec))
	
	bdec_magphys = np.array(bdec_magphys)
	bdec_prospector = np.array(bdec_prospector)

	balm.errorbar(bdec_prospector,bdec_magphys,
		          fmt=fmt, alpha=alpha, color=color,mew=mew)

	# labels
	balm.set_xlabel(r'Prospector H$_{\alpha}$/H$_{\beta}$',labelpad=13)
	balm.set_ylabel(r'MAGPHYS H$_{\alpha}$/H$_{\beta}$')
	balm = threed_dutils.equalize_axes(balm,bdec_prospector,bdec_magphys)

	# text
	off,scat = threed_dutils.offset_and_scatter(bdec_prospector,bdec_magphys,biweight=True)
	balm.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat),
			  transform = balm.transAxes,horizontalalignment='right')
	balm.text(0.96,0.1, 'mean offset='+"{:.2f}".format(off),
		      transform = balm.transAxes,horizontalalignment='right')

	#### Extinction
	tautot_magphys,tautot_prospector,taudiff_magphys,taudiff_prospector = [],[], [], []
	for ii,dat in enumerate(alldata):
		tau1 = dat['bfit']['maxprob_params'][dust1_idx][0]
		tau2 = dat['bfit']['maxprob_params'][dust2_idx][0]
		dindex = dat['bfit']['maxprob_params'][dinx_idx][0]

		dust2 = threed_dutils.charlot_and_fall_extinction(5500.,tau1,tau2,-1.0,dindex, kriek=True, nobc=True)
		dusttot = threed_dutils.charlot_and_fall_extinction(5500.,tau1,tau2,-1.0,dindex, kriek=True)
		taudiff_prospector.append(-np.log(dust2))
		tautot_prospector.append(-np.log10(dusttot))
		
		tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
		tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
		taudiff_magphys.append(tau2)
		tautot_magphys.append(tau1+tau2)
	
	taudiff_prospector = np.array(taudiff_prospector)
	taudiff_magphys = np.array(taudiff_magphys)
	tautot_magphys = np.array(tautot_magphys)
	tautot_prospector = np.array(tautot_prospector)

	ext_tot.errorbar(tautot_prospector,tautot_magphys,
		          fmt=fmt, alpha=alpha, color=color,mew=mew)

	# labels
	ext_tot.set_xlabel(r'Prospector diffuse+birth-cloud $\tau_{5500}$',labelpad=13)
	ext_tot.set_ylabel(r'MAGPHYS diffuse+birth-cloud $\tau_{5500}$')
	ext_tot = threed_dutils.equalize_axes(ext_tot,tautot_prospector,tautot_magphys)

	# text
	off,scat = threed_dutils.offset_and_scatter(tautot_prospector,tautot_magphys,biweight=True)
	ext_tot.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat),
			  transform = ext_tot.transAxes,horizontalalignment='right')
	ext_tot.text(0.96,0.1, 'mean offset='+"{:.2f}".format(off),
		      transform = ext_tot.transAxes,horizontalalignment='right',)

	ext_diff.errorbar(taudiff_prospector,taudiff_magphys,
		          fmt=fmt, alpha=alpha, color='0.4',
		          mew=mew)

	# labels
	ext_diff.set_xlabel(r'Prospector diffuse $\tau_{5500}$',labelpad=13)
	ext_diff.set_ylabel(r'MAGPHYS diffuse $\tau_{5500}$')
	ext_diff = threed_dutils.equalize_axes(ext_diff,taudiff_prospector,taudiff_magphys)

	# text
	off,scat = threed_dutils.offset_and_scatter(taudiff_prospector,taudiff_magphys,biweight=True)
	ext_diff.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat),
			  transform = ext_diff.transAxes,horizontalalignment='right')
	ext_diff.text(0.96,0.1, 'mean offset='+"{:.2f}".format(off),
		      transform = ext_diff.transAxes,horizontalalignment='right')

	plt.savefig(outfolder+'basic_comparison.png',dpi=dpi)
	plt.close()


