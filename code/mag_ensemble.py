import numpy as np
import matplotlib.pyplot as plt
import os, threed_dutils
import matplotlib as mpl
from astropy import constants
from astropy.cosmology import WMAP9
import magphys_plot_pref


#### set up colors and plot style
prosp_color = '#e60000'
obs_color = '#95918C'
magphys_color = '#1974D2'

minsfr = 1e-3

def translate_line_names(linenames):
	'''
	translate from my names to Moustakas names
	'''
	translate = {r'H$\alpha$': 'Ha',
				 '[OIII] 4959': 'OIII',
	             r'H$\beta$': 'Hb',
	             '[NII] 6583': 'NII'}

	return np.array([translate[line] for line in linenames])

def remove_doublets(x, names):

	if any('[OIII]' in s for s in list(names)):
		keep = np.array(names) != '[OIII] 5007'
		x = x[keep]
		names = names[keep]
		if not isinstance(x[0],basestring):
			x[np.array(names) == '[OIII] 4959'] *= 3.98

	if any('[NII]' in s for s in list(names)):
		keep = np.array(names) != '[NII] 6549'
		x = x[keep]
		names = names[keep]
		#if not isinstance(x[0],basestring):
		#	x[np.array(names) == '[NII] 6583'] *= 3.93

	return x

def ret_inf(alldata,field, model='Prospectr',name=None):

	'''
	returns information from alldata
	'''

	# fields
	# flux, flux_errup, flux_errdown, lum, lum_errup, lum_errdown, eqw
	emline_names = alldata[0]['residuals']['emlines']['em_name']
	nlines = len(emline_names)

	# sigh... this is a hack
	if field == 'eqw_rest':
		fillvalue = np.zeros(shape=(3,nlines))
	else:
		fillvalue = np.zeros_like(alldata[0]['residuals']['emlines'][model][field])

	if name is None:
		return np.squeeze(np.array([f['residuals']['emlines'][model][field] if f['residuals']['emlines'] is not None else fillvalue for f in alldata]))
	else:
		emline_names = alldata[0]['residuals']['emlines']['em_name']
		idx = emline_names == name
		if field == 'eqw_rest':
			return np.squeeze(np.array([f['residuals']['emlines'][model][field] if f['residuals']['emlines'] is not None else fillvalue for f in alldata])[:,:,idx])
		else:
			return np.squeeze(np.array([f['residuals']['emlines'][model][field] if f['residuals']['emlines'] is not None else fillvalue for f in alldata])[:,idx])

def compare_moustakas_fluxes(alldata,dat,emline_names,objnames,outname='test.png',outdec='bdec.png',model='Prospectr'):

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

	##### grab Prospectr information
	ind = np.array(idx_moust,dtype=bool)

	xplot = remove_doublets(np.transpose(ret_inf(alldata,'flux',model=model)),emline_names)[:,ind]
	xplot_errup = remove_doublets(np.transpose(ret_inf(alldata,'flux_errup',model=model)),emline_names)[:,ind]
	xplot_errdown = remove_doublets(np.transpose(ret_inf(alldata,'flux_errdown',model=model)),emline_names)[:,ind]
	eqw = remove_doublets(np.transpose(ret_inf(alldata,'eqw_rest',model=model))[:,0,:],emline_names)[:,ind]

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

		axes[ii].errorbar(eqw[ii,ok_idx],np.log10(yp)-np.log10(xp),yerr=yp_err,
						  xerr=xp_err,
			              fmt='o', alpha=0.6,
			              linestyle=' ')

		axes[ii].set_ylabel('log(Moustakas+10/measured) '+emline_names_doubrem[ii])
		axes[ii].set_xlabel('EQW '+emline_names_doubrem[ii])
		off,scat = threed_dutils.offset_and_scatter(np.log10(xp),np.log10(yp),biweight=True)
		axes[ii].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat) +' dex',
				  transform = axes[ii].transAxes,horizontalalignment='right')
		axes[ii].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off) + ' dex',
			      transform = axes[ii].transAxes,horizontalalignment='right')
		axes[ii].axhline(0, linestyle=':', color='grey')


		# print outliers
		diff = np.log10(xp) - np.log10(yp)
		outliers = np.abs(diff) > 3*scat
		print emline_names_doubrem[ii] + ' outliers:'
		for jj in xrange(len(outliers)):
			if outliers[jj] == True:
				print np.array(moust_objnames)[ok_idx][jj]+' ' + "{:.3f}".format(diff[jj]/scat)

	plt.tight_layout()
	plt.savefig(outname,dpi=300)
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
	ax.text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat),
			  transform = ax.transAxes,horizontalalignment='right')
	ax.text(0.99,0.1, 'mean offset='+"{:.3f}".format(off),
			      transform = ax.transAxes,horizontalalignment='right')
	ax.plot([2.86,2.86],[0.0,15.0],linestyle='-',color='black')
	ax.plot([0.0,15.0],[2.86,2.86],linestyle='-',color='black')
	ax.set_xlim(1,10)
	ax.set_ylim(1,10)
	plt.savefig(outdec,dpi=300)
	plt.close()


def compare_model_flux(alldata, emline_names, outname = 'test.png'):

	#################
	#### plot Prospectr versus MAGPHYS flux
	#################
	ncol = int(np.ceil(len(emline_names)/2.))
	fig, axes = plt.subplots(ncol,2, figsize = (10,ncol*5))
	axes = np.ravel(axes)
	for ii,emname in enumerate(emline_names):
		magdat = np.log10(ret_inf(alldata,'lum',model='MAGPHYS',name=emname)) 
		prodat = np.log10(ret_inf(alldata,'lum',model='Prospectr',name=emname)) 
		yplot = prodat-magdat
		xplot = np.log10(ret_inf(alldata,'eqw_rest',model='Prospectr',name=emname))
		idx = np.isfinite(yplot)

		axes[ii].plot(xplot[idx,0], yplot[idx], 
			    	  'o',
			    	  alpha=0.6)
		
		xlabel = r"log({0} EQW) [Prospectr]"
		ylabel = r"log(Prosp/MAGPHYS) [{0} flux]"
		axes[ii].set_xlabel(xlabel.format(emname))
		axes[ii].set_ylabel(ylabel.format(emname))

		# equalize axes, show offset and scatter
		off,scat = threed_dutils.offset_and_scatter(magdat[idx],
			                                        prodat[idx],
			                                        biweight=True)
		axes[ii].text(0.99,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
				  transform = axes[ii].transAxes,horizontalalignment='right')
		axes[ii].text(0.99,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = axes[ii].transAxes,horizontalalignment='right')
	
	# save
	plt.tight_layout()
	plt.savefig(outname,dpi=300)
	plt.close()	

def obs_vs_kennicutt_ha(alldata,emline_names,obs_info,outname='test.png',outname_cloudy='test_cloudy.png'):
	
	#################
	#### plot observed Halpha versus model Halpha from Kennicutt relationship
	#################

	f_ha = ret_inf(alldata,'flux',model='Prospectr',name='H$\\alpha$')
	f_ha_errup = ret_inf(alldata,'flux_errup',model='Prospectr',name='H$\\alpha$')
	f_ha_errdown = ret_inf(alldata,'flux_errdown',model='Prospectr',name='H$\\alpha$')

	keep_idx = obs_info['keep_idx']

	##### indexes
	mparnames = alldata[0]['model']['parnames']
	mu_idx = mparnames == 'mu'
	tauv_idx = mparnames == 'tauv'

	parnames = alldata[0]['pquantiles']['parnames']
	met_idx = parnames == 'logzsol'
	slope_idx = parnames == 'sf_tanslope'
	trunc_idx = parnames == 'delt_trunc'
	tage_idx = parnames == 'tage'

	##### below code is to calculate SFR_1 Myr
	runname = 'brownseds'
	#### load up prospectr results
	filebase, parm_basename, ancilname=threed_dutils.generate_basenames(runname)
	sps = threed_dutils.setup_sps(custom_filter_key=None)


	##### Pull out empirical Halphas from MAGPHYS + Prospectr
	##### observed Halphas from pipeline (convert to luminosity)
	ha_mag, ha_pro, pmet, cloudy_ha, pslope, ptrunc, ptage, psfr, pssfr, pattn, pd1, pd2, pdind,sfr_1 = [], [], [], [], [], [], [], [], [], [], [], [], [], []
	for ii,dat in enumerate(alldata):

		'''
		#### calculate sfr_1myr
		sample_results, powell_results, model = threed_dutils.load_prospectr_data(filebase[ii])
		maxprob = sample_results['bfit']['maxprob_params']
		sfh_params = threed_dutils.find_sfh_params(sample_results['model'],maxprob,
	                                       sample_results['obs'],sps)
		sfr_1.append(threed_dutils.calculate_sfr(sfh_params, 0.001, minsfr=-np.inf, maxsfr=np.inf))
		'''

		# best-fit empirical Halpha, for Prospectr
		ha_pro.append(threed_dutils.synthetic_halpha(dat['bfit']['sfr_100'], dat['bfit']['maxprob_params'][parnames == 'dust1'][0], dat['bfit']['maxprob_params'][parnames == 'dust2'][0], -1.0, dat['bfit']['maxprob_params'][parnames == 'dust_index'][0]))
		#ha_pro.append(dat['bfit']['emp_ha'])


		tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
		tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
		
		ha_mag.append(threed_dutils.synthetic_halpha(dat['model']['parameters'][dat['model']['parnames'] == 'SFR'][0], tau1, tau2, -1.3, -0.7))
		#ha_mag.append(threed_dutils.synthetic_halpha(dat['model']['parameters'][dat['model']['parnames'] == 'SFR'][0], tau1, tau2, -1.3, -0.7))

		pc2cm = 3.08567758e18
		distance = WMAP9.luminosity_distance(dat['residuals']['phot']['z']).value*1e6*pc2cm
		dfactor = (4*np.pi*distance**2)
		f_ha[ii] = f_ha[ii] * dfactor

		cloudy_ha.append(obs_info['model_ha'][ii]*dfactor)

		# prospectr info
		pmet.append(dat['bfit']['maxprob_params'][met_idx][0])
		pslope.append(dat['bfit']['maxprob_params'][slope_idx][0])
		ptrunc.append(dat['bfit']['maxprob_params'][trunc_idx][0])
		ptage.append(dat['bfit']['maxprob_params'][tage_idx][0])
		psfr.append(dat['bfit']['sfr_10'])
		pssfr.append(psfr[-1]/dat['bfit']['maxprob_params'][0])
		tau1 = dat['bfit']['maxprob_params'][parnames == 'dust1'][0]
		tau2 = dat['bfit']['maxprob_params'][parnames == 'dust2'][0]
		dind = dat['bfit']['maxprob_params'][parnames == 'dust_index'][0]
		pattn.append(threed_dutils.charlot_and_fall_extinction(6563, tau1, tau2, -1.0, dind))
		pd1.append(tau1)
		pd2.append(tau2)
		pdind.append(dind)


	ha_mag = np.log10(np.array(ha_mag)[keep_idx])
	ha_pro = np.log10(np.array(ha_pro)[keep_idx])
	ha_cloudy = np.log10(np.array(cloudy_ha)[keep_idx])
	pmet = np.array(pmet)[keep_idx]
	pslope = np.array(pslope)[keep_idx]
	ptrunc = np.array(ptrunc)[keep_idx]
	ptage = np.array(ptage)[keep_idx]
	psfr = np.array(psfr)[keep_idx]
	pssfr = np.array(pssfr)[keep_idx]
	pattn = np.array(pattn)[keep_idx]
	pd1 = np.array(pd1)[keep_idx]
	pd2 = np.array(pd2)[keep_idx]
	pdind = np.array(pdind)[keep_idx]



	fig, ax = plt.subplots(1,3, figsize = (22,6))

	ax[0].errorbar(ha_mag, ha_pro, fmt='o',alpha=0.6,linestyle=' ')
	ax[0].set_xlabel(r'log(Kennicutt H$_{\alpha}$) [MAGPHYS]')
	ax[0].set_ylabel(r'log(Kennicutt H$_{\alpha}$) [Prospectr]')
	ax[0] = threed_dutils.equalize_axes(ax[0], ha_mag, ha_pro)
	off,scat = threed_dutils.offset_and_scatter(ha_mag, ha_pro,biweight=True)
	ax[0].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat) +' dex',
			  transform = ax[0].transAxes,horizontalalignment='right')
	ax[0].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off)+ ' dex',
			      transform = ax[0].transAxes,horizontalalignment='right')

	ax[1].errorbar(ha_mag,np.log10(f_ha[keep_idx]), fmt='o',alpha=0.6,linestyle=' ')
	ax[1].set_xlabel(r'log(Kennicutt H$_{\alpha}$) [MAGPHYS]')
	ax[1].set_ylabel(r'log(H$_{\alpha}$) [observed]')
	ax[1] = threed_dutils.equalize_axes(ax[1], ha_mag, np.log10(f_ha[keep_idx]))
	off,scat = threed_dutils.offset_and_scatter(ha_mag, np.log10(f_ha[keep_idx]),biweight=True)
	ax[1].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat) +' dex', transform = ax[1].transAxes,horizontalalignment='right')
	ax[1].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off)+ ' dex', transform = ax[1].transAxes,horizontalalignment='right')

	ax[2].errorbar(ha_pro, np.log10(f_ha[keep_idx]), fmt='o',alpha=0.6,linestyle=' ')
	ax[2].set_xlabel(r'log(Kennicutt H$_{\alpha}$) [Prospectr]')
	ax[2].set_ylabel(r'log(H$_{\alpha}$) [observed]')
	ax[2] = threed_dutils.equalize_axes(ax[2], ha_pro, np.log10(f_ha[keep_idx]))
	off,scat = threed_dutils.offset_and_scatter(ha_pro, np.log10(f_ha[keep_idx]),biweight=True)
	ax[2].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat) +' dex', transform = ax[2].transAxes,horizontalalignment='right')
	ax[2].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off)+ ' dex', transform = ax[2].transAxes,horizontalalignment='right')


	plt.savefig(outname,dpi=300)
	plt.close()

	fig, ax = plt.subplots(1,2, figsize = (16,8))

	ax[0].errorbar(ha_pro, ha_cloudy[:,0], fmt='o',alpha=0.6,linestyle=' ',color='0.4')
	ax[0].set_xlabel(r'log(Kennicutt H$_{\alpha}$) [Prospectr]')
	ax[0].set_ylabel(r'log(CLOUDY H$_{\alpha}$) [Prospectr]')
	ax[0] = threed_dutils.equalize_axes(ax[0], ha_pro, ha_cloudy[:,0])
	off,scat = threed_dutils.offset_and_scatter(ha_pro, ha_cloudy[:,0],biweight=True)
	ax[0].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat) +' dex', transform = ax[0].transAxes,horizontalalignment='right')
	ax[0].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off)+ ' dex', transform = ax[0].transAxes,horizontalalignment='right')

	#cm = plt.cm.get_cmap('RdYlBu')
	#sc=ax[1].scatter(pmet, ha_pro-ha_cloudy[:,0], c=pslope, marker='o',alpha=0.6,linewidth=1.0,cmap=cm)
	#plt.colorbar(sc)
	ax[1].errorbar(pmet, ha_pro-ha_cloudy[:,0],fmt='o', alpha=0.6, linestyle=' ', color='0.4')
	ax[1].set_ylabel(r'log(Kennicutt/CLOUDY H$_{\alpha}$) [Prospectr]')
	ax[1].set_xlabel(r'log(metallicity) [Prospectr]')

	plt.tight_layout()
	plt.savefig(outname_cloudy,dpi=300)
	plt.close()

	# trunc time versus Halpha flux
	fig, ax = plt.subplots(1,2, figsize = (16,8))

	trunc_time = np.log10(ptage*(1-ptrunc))
	ax[0].errorbar(ha_cloudy[:,0], trunc_time, fmt='o',alpha=0.6,linestyle=' ',color='0.4')
	ax[0].set_xlabel(r'log(CLOUDY H$_{\alpha}$) [Prospectr]')
	ax[0].set_ylabel(r'log(time since truncation) [Gyr]')

	recent = trunc_time < 0.0
	ax[1].errorbar(ha_cloudy[recent,0], pslope[recent], fmt='o',alpha=0.6,linestyle=' ',color='red')
	ax[1].errorbar(ha_cloudy[~recent,0], pslope[~recent], fmt='o',alpha=0.6,linestyle=' ',color='0.4')
	ax[1].set_xlabel(r'log(CLOUDY H$_{\alpha}$) [Prospectr]')
	ax[1].set_ylabel(r'tan(sf_slope)')
	ax[1].text(0.99,0.05, 'truncated in last Gyr', transform = ax[1].transAxes,horizontalalignment='right',color='red')


	plt.tight_layout()
	plt.savefig('/Users/joel/code/python/threedhst_bsfh/plots/brownseds/pcomp/cloudyha_versus_truncation.png',dpi=300)
	plt.close()

	# trunc_time < 1 Gyr, slope versus Halpha flux
	recent = trunc_time < 0.0
	fig, ax = plt.subplots(1,1, figsize = (8,8))

	ax.errorbar(ha_cloudy[recent,0], pslope[recent], fmt='o',alpha=0.6,linestyle=' ',color='0.4')
	ax.set_xlabel(r'log(CLOUDY H$_{\alpha}$) [Prospectr]')
	ax.set_ylabel(r'log(time since truncation) [Gyr]')

	plt.tight_layout()
	plt.savefig('/Users/joel/code/python/threedhst_bsfh/plots/brownseds/pcomp/cloudyha_versus_slope.png',dpi=300)
	plt.close()

def obs_vs_prosp_ha(alldata,emline_names,keep_idx,outname='test.png',model='Prospectr'):

	#################
	#### plot observed Halpha versus expected (PROSPECTR ONLY)
	#################
	# first pull out observed Halphas
	# add an S/N cut... ? remove later maybe

	f_ha = ret_inf(alldata,'flux',model=model,name='H$\\alpha$')
	f_ha_errup = ret_inf(alldata,'flux_errup',model=model,name='H$\\alpha$')
	f_ha_errdown = ret_inf(alldata,'flux_errdown',model=model,name='H$\\alpha$')

	ha_p_idx = alldata[0]['model_emline']['name'] == 'Halpha'
	model_ha = np.zeros(shape=(len(alldata),3))

	for ii, dat in enumerate(alldata):

		# comes out in Lsun
		# convert to CGS flux
		pc2cm = 3.08567758e18
		distance = WMAP9.luminosity_distance(dat['residuals']['phot']['z']).value*1e6*pc2cm
		dfactor = (4*np.pi*distance**2)/constants.L_sun.cgs.value

		model_ha[ii,0] = dat['model_emline']['q50'][ha_p_idx] / dfactor
		model_ha[ii,1] = dat['model_emline']['q84'][ha_p_idx] / dfactor
		model_ha[ii,2] = dat['model_emline']['q16'][ha_p_idx] / dfactor

	sfing, composite, agn = return_agn_str(keep_idx)
	keys = [sfing, composite, agn]
	colors = ['blue', 'purple', 'red']
	labels = ['SF', 'SF/AGN', 'AGN']

	fig, ax = plt.subplots(1,1, figsize = (10,10))
	xplot = np.log10(model_ha[keep_idx][:,0])
	yplot = np.log10(f_ha[keep_idx])
	yerr = threed_dutils.asym_errors(f_ha[keep_idx],f_ha_errup[keep_idx], f_ha_errdown[keep_idx],log=True)

	for ii in xrange(len(labels)):
		ax.errorbar(xplot[keys[ii]], yplot[keys[ii]], yerr=[yerr[0][keys[ii]],yerr[1][keys[ii]]],fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	
	ax.set_ylabel(r'log(observed H$_{\alpha}$)')
	ax.set_xlabel(r'log(best-fit Prospectr H$_{\alpha}$)')
	ax = threed_dutils.equalize_axes(ax,xplot,yplot)
	off,scat = threed_dutils.offset_and_scatter(xplot,yplot,biweight=True)
	ax.text(0.99,0.05, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
			  transform = ax.transAxes,horizontalalignment='right')
	ax.text(0.99,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = ax.transAxes,horizontalalignment='right')
	ax.legend(loc=2)
	plt.savefig(outname,dpi=300)
	plt.close()

	pinfo = {}
	pinfo['model_ha'] = model_ha
	pinfo['f_ha'] = f_ha
	pinfo['f_ha_errup'] = f_ha_errup
	pinfo['f_ha_errdown'] = f_ha_errdown
	pinfo['keep_idx'] = keep_idx

	return pinfo

def obs_vs_model_hdelta_dn(alldata,outname=None):

	hdelta_sn_cut = 2

	### from observations
	### pull out hdelta, dn4000
	hdel_obs = -ret_inf(alldata,'hdelta_flux',model='obs')
	hdel_errup_obs = -ret_inf(alldata,'hdelta_flux_errup',model='obs')
	hdel_errdown_obs = -ret_inf(alldata,'hdelta_flux_errdown',model='obs')
	hdel_symerr_obs = (hdel_errup_obs - hdel_errdown_obs) / 2.

	dn4000_obs = ret_inf(alldata,'dn4000',model='obs')


	### pull out model info
	hdel_prosp = np.log10(-ret_inf(alldata,'hdelta_flux',model='Prospectr'))
	hdel_mag = np.log10(-ret_inf(alldata,'hdelta_flux',model='MAGPHYS'))

	dn4000_prosp = ret_inf(alldata,'Dn4000',model='Prospectr')
	dn4000_mag = ret_inf(alldata,'Dn4000',model='MAGPHYS')

	### define limits
	good_idx = (np.abs((hdel_obs / hdel_symerr_obs)) > hdelta_sn_cut) & (hdel_obs > 0) & (hdel_mag[:,0] > -18)
	hdel_plot_errs = threed_dutils.asym_errors(hdel_obs[good_idx], hdel_errup_obs[good_idx], hdel_errdown_obs[good_idx], log=True)
	hdel_prosp = hdel_prosp[good_idx]
	hdel_mag = hdel_mag[good_idx]


	### plot comparison
	fig, ax = plt.subplots(2,2, figsize = (15,15))

	ax[0,0].errorbar(dn4000_obs[good_idx], dn4000_prosp[good_idx], fmt='o',alpha=0.6,linestyle=' ',color='black')
	ax[0,0].set_xlabel(r'observed D$_n$(4000)')
	ax[0,0].set_ylabel(r'Prospectr D$_n$(4000)')
	ax[0,0] = threed_dutils.equalize_axes(ax[0,0], dn4000_obs[good_idx], dn4000_prosp[good_idx])
	off,scat = threed_dutils.offset_and_scatter(dn4000_obs[good_idx], dn4000_prosp[good_idx],biweight=True)
	ax[0,0].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat), transform = ax[0,0].transAxes,horizontalalignment='right')
	ax[0,0].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off), transform = ax[0,0].transAxes,horizontalalignment='right')

	ax[0,1].errorbar(dn4000_obs[good_idx], dn4000_mag[good_idx], fmt='o',alpha=0.6,linestyle=' ', color='black')
	ax[0,1].set_xlabel(r'observed D$_n$(4000)')
	ax[0,1].set_ylabel(r'MAGPHYS D$_n$(4000)')
	ax[0,1] = threed_dutils.equalize_axes(ax[0,1], dn4000_obs[good_idx], dn4000_mag[good_idx])
	off,scat = threed_dutils.offset_and_scatter(dn4000_obs[good_idx], dn4000_mag[good_idx],biweight=True)
	ax[0,1].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat), transform = ax[0,1].transAxes,horizontalalignment='right')
	ax[0,1].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off), transform = ax[0,1].transAxes,horizontalalignment='right')

	ax[1,0].errorbar(np.log10(hdel_obs[good_idx]), hdel_prosp[:,0], xerr=hdel_plot_errs,fmt='o',alpha=0.6,linestyle=' ',color='black')
	ax[1,0].set_xlabel(r'observed log(H$_{\delta}$)')
	ax[1,0].set_ylabel(r'Prospectr log(H$_{\delta}$)')
	ax[1,0] = threed_dutils.equalize_axes(ax[1,0], np.log10(hdel_obs[good_idx]), hdel_prosp[:,0])
	off,scat = threed_dutils.offset_and_scatter(np.log10(hdel_obs[good_idx]), hdel_prosp[:,0],biweight=True)
	ax[1,0].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat) + ' dex', transform = ax[1,0].transAxes,horizontalalignment='right')
	ax[1,0].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off) + ' dex', transform = ax[1,0].transAxes,horizontalalignment='right')

	ax[1,1].errorbar(np.log10(hdel_obs[good_idx]), hdel_mag[:,0], xerr=hdel_plot_errs, fmt='o',alpha=0.6,linestyle=' ', color='black')
	ax[1,1].set_xlabel(r'observed log(H$_{\delta}$)')
	ax[1,1].set_ylabel(r'MAGPHYS log(H$_{\delta}$)')
	ax[1,1] = threed_dutils.equalize_axes(ax[1,1], np.log10(hdel_obs[good_idx]), hdel_mag[:,0])
	off,scat = threed_dutils.offset_and_scatter(np.log10(hdel_obs[good_idx]), hdel_mag[:,0],biweight=True)
	ax[1,1].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat), transform = ax[1,1].transAxes,horizontalalignment='right')
	ax[1,1].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off), transform = ax[1,1].transAxes,horizontalalignment='right')

	plt.savefig(outname, dpi=300)
	plt.close()



def obs_vs_model_bdec(alldata,emline_names,outname='test.png'):

	#################
	#### plot observed Balmer decrement versus expected
	#################
	# remove AGNs from this plot

	sn_cut = 5
	eqw_cut = 3

	f_ha = ret_inf(alldata,'flux',model='Prospectr',name='H$\\alpha$')
	err_ha = (ret_inf(alldata,'flux_errup',model='Prospectr',name='H$\\alpha$') - ret_inf(alldata,'flux_errdown',model='Prospectr',name='H$\\alpha$'))/2.
	f_hb = ret_inf(alldata,'flux',model='Prospectr',name='H$\\beta$')
	err_hb = (ret_inf(alldata,'flux_errup',model='Prospectr',name='H$\\beta$') - ret_inf(alldata,'flux_errdown',model='Prospectr',name='H$\\beta$'))/2.
	bdec_measured = f_ha / f_hb

	sn_ha = np.abs(f_ha / err_ha)
	sn_hb = np.abs(f_hb / err_hb)

	eqw_ha = ret_inf(alldata,'eqw_rest',model='Prospectr',name='H$\\alpha$')
	eqw_hb = ret_inf(alldata,'eqw_rest',model='Prospectr',name='H$\\beta$')

	names = np.array([dat['objname'] for dat in alldata])

	#### for now, aggressive S/N cuts
	# S/N(Ha) > 10, S/N (Hbeta) > 10
	keep_idx = np.squeeze((sn_ha > sn_cut) & (sn_hb > sn_cut) & (eqw_ha[:,0] > eqw_cut) & (eqw_hb[:,0] > eqw_cut))

	

	#### calculate expected Balmer decrement for Prospectr, MAGPHYS
	# variable names for Prospectr
	parnames = alldata[0]['pquantiles']['parnames']
	dinx_idx = parnames == 'dust_index'
	dust1_idx = parnames == 'dust1'
	dust2_idx = parnames == 'dust2'

	# variable names for MAGPHYS ()
	mparnames = alldata[0]['model']['parnames']
	mu_idx = mparnames == 'mu'
	tauv_idx = mparnames == 'tauv'

	bdec_magphys, bdec_prospectr, bdec_mod = [],[], []
	ptau1, ptau2, pdindex = [], [], []
	for dat in alldata:
		ptau1.append(dat['bfit']['maxprob_params'][dust1_idx][0])
		ptau2.append(dat['bfit']['maxprob_params'][dust2_idx][0])
		pdindex.append(dat['bfit']['maxprob_params'][dinx_idx][0])
		bdec = threed_dutils.calc_balmer_dec(ptau1[-1], ptau2[-1], -1.0, pdindex[-1],kriek=True)
		bdec_mod.append(dat['bfit']['halpha_flux'] / dat['bfit']['hbeta_flux'])

		bdec_prospectr.append(bdec)
		
		tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
		tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
		bdec = threed_dutils.calc_balmer_dec(tau1, tau2, -1.3, -0.7)
		bdec_magphys.append(np.squeeze(bdec))
	
	sfing, composite, agn = return_agn_str(keep_idx)

	pl_bdec_mod = np.array(bdec_mod)[keep_idx][~agn]
	pl_bdec_magphys = np.array(bdec_magphys)[keep_idx][~agn]
	pl_bdec_prospectr = np.array(bdec_prospectr)[keep_idx][~agn]
	pl_bdec_measured = bdec_measured[keep_idx][~agn]
	pl_bdec_prospectr = pl_bdec_mod

	#### transform from balmer decrement to tau(Hb) - tau(Ha)
	pl_bdec_prospectr = np.log(pl_bdec_prospectr / 2.86)
	pl_bdec_magphys = np.log(pl_bdec_magphys / 2.86)
	pl_bdec_measured = np.log(pl_bdec_measured / 2.86)

	fig, ax = plt.subplots(1,3, figsize = (22,6))

	ax[0].errorbar(pl_bdec_measured, pl_bdec_prospectr, fmt='o',alpha=0.6,linestyle=' ')
	ax[0].set_xlabel(r'observed $\tau_{\mathrm{H}\beta}$ - $\tau_{\mathrm{H}\alpha}$')
	ax[0].set_ylabel(r'Prospectr $\tau_{\mathrm{H}\beta}$ - $\tau_{\mathrm{H}\alpha}$')
	ax[0] = threed_dutils.equalize_axes(ax[0], pl_bdec_measured,pl_bdec_prospectr)
	off,scat = threed_dutils.offset_and_scatter(pl_bdec_measured,pl_bdec_prospectr,biweight=True)
	ax[0].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat),
			  transform = ax[0].transAxes,horizontalalignment='right')
	ax[0].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off),
			      transform = ax[0].transAxes,horizontalalignment='right')

	ax[1].errorbar(pl_bdec_measured, pl_bdec_magphys, fmt='o',alpha=0.6,linestyle=' ')
	ax[1].set_xlabel(r'observed $\tau_{\mathrm{H}\beta}$ - $\tau_{\mathrm{H}\alpha}$')
	ax[1].set_ylabel(r'MAGPHYS $\tau_{\mathrm{H}\beta}$ - $\tau_{\mathrm{H}\alpha}$')
	ax[1] = threed_dutils.equalize_axes(ax[1], pl_bdec_measured,pl_bdec_magphys)
	off,scat = threed_dutils.offset_and_scatter(pl_bdec_measured,pl_bdec_magphys,biweight=True)
	ax[1].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat),
			  transform = ax[1].transAxes,horizontalalignment='right')
	ax[1].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off),
			      transform = ax[1].transAxes,horizontalalignment='right')

	ax[2].errorbar(pl_bdec_prospectr, pl_bdec_magphys, fmt='o',alpha=0.6,linestyle=' ')
	ax[2].set_xlabel(r'Prospectr $\tau_{\mathrm{H}\beta}$ - $\tau_{\mathrm{H}\alpha}$')
	ax[2].set_ylabel(r'MAGPHYS $\tau_{\mathrm{H}\beta}$ - $\tau_{\mathrm{H}\alpha}$')
	ax[2] = threed_dutils.equalize_axes(ax[2], pl_bdec_prospectr,pl_bdec_magphys)
	off,scat = threed_dutils.offset_and_scatter(pl_bdec_prospectr,pl_bdec_magphys,biweight=True)
	ax[2].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat),
			  transform = ax[2].transAxes,horizontalalignment='right')
	ax[2].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off),
			      transform = ax[2].transAxes,horizontalalignment='right')
	
	plt.savefig(outname,dpi=300)
	plt.close()

	pinfo = {}
	pinfo['bdec_magphys'] = np.array(bdec_magphys)
	pinfo['bdec_prospectr'] = np.array(bdec_prospectr)
	pinfo['bdec_measured'] = np.array(bdec_measured)
	pinfo['keep_idx'] = keep_idx

	pinfo['dust1'] = np.array(ptau1)
	pinfo['dust2'] = np.array(ptau2)
	pinfo['dust2_index'] = np.array(pdindex)

	return pinfo

def return_agn_str(idx):

	from astropy.io import fits
	hdulist = fits.open(os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/table1.fits')
	agn_str = hdulist[1].data['Class']
	hdulist.close()

	agn_str = agn_str[idx]
	sfing = (agn_str == 'SF') | (agn_str == '---')
	composite = (agn_str == 'SF/AGN')
	agn = agn_str == 'AGN'

	return sfing, composite, agn

def residual_plots(alldata,obs_info, bdec_info):
	# bdec_info: bdec_magphys, bdec_prospectr, bdec_measured, keep_idx, dust1, dust2, dust2_index
	# obs_info: model_ha, f_ha, f_ha_errup, f_ha_errdown

	fldr = '/Users/joel/code/python/threedhst_bsfh/plots/brownseds/magphys/emlines_comp/residuals/'
	idx = bdec_info['keep_idx']

	sfr_100 = np.log10(np.clip([x['bfit']['sfr_100'] for x in alldata],1e-4,np.inf))[idx]

	#### bdec resid versus ha resid
	bdec_resid = bdec_info['bdec_prospectr'][idx] - bdec_info['bdec_measured'][idx]
	ha_resid = np.log10(obs_info['f_ha'][idx]) - np.log10(obs_info['model_ha'][idx,0])

	fig, ax = plt.subplots(1,1, figsize = (8,8))

	ax.errorbar(ha_resid, bdec_resid, fmt='o',alpha=0.6,linestyle=' ')
	ax.set_xlabel(r'log(Prospectr/obs) [H$_{\alpha}$]')
	ax.set_ylabel(r'Prospectr - obs [Balmer decrement]')
	
	plt.savefig(fldr+'bdec_resid_versus_ha_resid.png', dpi=300)

	sfing, composite, agn = return_agn_str(idx)
	keys = [sfing, composite, agn]
	colors = ['blue', 'purple', 'red']
	labels = ['SF', 'SF/AGN', 'AGN']

	#### dust1 / dust2
	fig, ax = plt.subplots(1,2, figsize = (18,8))
	
	xplot = bdec_info['dust1'][idx]/bdec_info['dust2'][idx]
	yplot = bdec_resid
	for ii in xrange(len(labels)):
		ax[0].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[0].axhline(0, linestyle=':', color='grey')
	ax[0].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))
	ax[0].set_xlabel(r'dust1/dust2')
	ax[0].set_ylabel(r'Prospectr - obs [Balmer decrement]')

	yplot = ha_resid
	for ii in xrange(len(labels)):
		ax[1].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[1].axhline(0, linestyle=':', color='grey')
	ax[1].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))	
	ax[1].set_xlabel(r'dust1/dust2')
	ax[1].set_ylabel(r'log(Prospectr/obs) [H$_{\alpha}$]')
	ax[1].legend()
	
	plt.savefig(fldr+'dust1_dust2_residuals.png', dpi=300)

	#### dust2_index
	fig, ax = plt.subplots(1,2, figsize = (18,8))

	xplot = bdec_info['dust2_index'][idx]
	yplot = bdec_resid
	for ii in xrange(len(labels)):
		ax[0].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[0].set_xlabel(r'dust2_index')
	ax[0].set_ylabel(r'Prospectr - obs [Balmer decrement]')
	ax[0].axhline(0, linestyle=':', color='grey')
	ax[0].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))

	yplot = ha_resid
	for ii in xrange(len(labels)):
		ax[1].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[1].set_xlabel(r'dust2_index')
	ax[1].set_ylabel(r'log(Prospectr/obs) [H$_{\alpha}$]')
	ax[1].legend(loc=3)
	ax[1].axhline(0, linestyle=':', color='grey')
	ax[1].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))
	
	plt.savefig(fldr+'dust_index_residuals.png', dpi=300)

	#### total attenuation at 5500 angstroms
	fig, ax = plt.subplots(1,2, figsize = (18,8))

	xplot = bdec_info['dust1'][idx] + bdec_info['dust2'][idx]
	yplot = bdec_resid
	for ii in xrange(len(labels)):
		ax[0].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[0].set_xlabel(r'total attenuation [5500 $\AA$]')
	ax[0].set_ylabel(r'Prospectr - obs [Balmer decrement]')
	ax[0].legend(loc=4)
	ax[0].axhline(0, linestyle=':', color='grey')
	ax[0].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))

	yplot = ha_resid
	for ii in xrange(len(labels)):
		ax[1].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii],)
	ax[1].set_xlabel(r'total attenuation [5500 $\AA$]')
	ax[1].set_ylabel(r'log(Prospectr/obs) [H$_{\alpha}$]')
	ax[1].axhline(0, linestyle=':', color='grey')
	ax[1].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))
	
	plt.savefig(fldr+'total_attenuation_residuals.png', dpi=300)

	#### sfr_100 residuals
	fig, ax = plt.subplots(1,2, figsize = (18,8))

	xplot = sfr_100
	yplot = bdec_resid
	for ii in xrange(len(labels)):
		ax[0].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii])
	ax[0].set_xlabel(r'SFR$_{100 \mathrm{ Myr}}$ [M$_{\odot}$/yr]')
	ax[0].set_ylabel(r'Prospectr - obs [Balmer decrement]')
	ax[0].legend(loc=3)
	ax[0].axhline(0, linestyle=':', color='grey')
	ax[0].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))

	yplot = ha_resid
	for ii in xrange(len(labels)):
		ax[1].errorbar(xplot[keys[ii]], yplot[keys[ii]], fmt='o',alpha=0.6,linestyle=' ',color=colors[ii],label=labels[ii],)
	ax[1].set_xlabel(r'SFR$_{100 \mathrm{ Myr}}$ [M$_{\odot}$/yr]')
	ax[1].set_ylabel(r'log(Prospectr/obs) [H$_{\alpha}$]')
	ax[1].axhline(0, linestyle=':', color='grey')
	ax[1].set_ylim(-np.max(np.abs(yplot)),np.max(np.abs(yplot)))
	
	plt.savefig(fldr+'sfr_100_residuals.png', dpi=300)

	#### dust2_index vs dust1/dust2
	fig, ax = plt.subplots(1,1, figsize = (8,8))

	ax.errorbar(bdec_info['dust2_index'], bdec_info['dust1']/bdec_info['dust2'], fmt='o',alpha=0.6,linestyle=' ')
	ax.set_xlabel(r'dust2_index')
	ax.set_ylabel(r'dust1/dust2')

	plt.savefig(fldr+'idx_versus_ratio.png', dpi=300)

def plot_emline_comp(alldata,outfolder):
	'''
	emission line luminosity comparisons:
		(1) Observed luminosity, Prospectr vs MAGPHYS continuum subtraction
		(2) Moustakas+10 comparisons
		(3) model Balmer decrement (from dust) versus observed Balmer decrement
		(4) model Halpha (from Kennicutt + dust) versus observed Halpha
	'''

	##### Pull relevant information out of alldata
	emline_names = alldata[0]['residuals']['emlines']['em_name']

	##### load moustakas+10 line flux information
	objnames = objnames = np.array([f['objname'] for f in alldata])
	dat = threed_dutils.load_moustakas_data(objnames = list(objnames))
	
	##### do work
	compare_model_flux(alldata,emline_names,outname = outfolder+'continuum_model_flux_comparison.png')
	compare_moustakas_fluxes(alldata,dat,emline_names,objnames,
							 outname=outfolder+'moustakas_comparison.png',
							 outdec=outfolder+'balmer_dec_comparison.png')
	
	bdec_info = obs_vs_model_bdec(alldata,emline_names, outname=outfolder+'bdec_comparison.png')
	obs_vs_model_hdelta_dn(alldata,outname=outfolder+'hdelta_dn_comp.png')

	obs_info = obs_vs_prosp_ha(alldata,emline_names,bdec_info['keep_idx'],outname=outfolder+'halpha_comparison.png')
	obs_vs_kennicutt_ha(alldata,emline_names,obs_info,
		         outname=outfolder+'kennicutt_comparison.png',
		         outname_cloudy=outfolder+'kennicutt_cloudy.png')
	residual_plots(alldata,obs_info, bdec_info)

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

	##### find prospectr indexes
	parnames = alldata[0]['pquantiles']['parnames']
	idx_mass = parnames == 'mass'
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
			tmp = np.array([data['pquantiles']['q16'][idx_mass][0],
				            data['pquantiles']['q50'][idx_mass][0],
				            data['pquantiles']['q84'][idx_mass][0]])
			promass = np.concatenate((promass,np.atleast_2d(np.log10(tmp))),axis=0)
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

	##### Errors on Prospectr+Magphys quantities
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
		          label='Prospectr',
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

	plt.savefig(outfolder+'relationships.png',dpi=300)
	plt.close

def prospectr_comparison(alldata,outfolder):

	'''
	For Prospectr:
	dust_index versus total attenuation
	dust_index versus SFR
	dust1 versus dust2, everything below -0.45 dust index highlighted
	'''
	
	#### find prospectr indexes
	parnames = alldata[0]['pquantiles']['parnames']
	idx_mass = parnames == 'mass'
	didx_idx = parnames == 'dust_index'
	d1_idx = parnames == 'dust1'
	d2_idx = parnames == 'dust2'

	#### herschel flag
	hflag = np.array([True if np.sum(dat['residuals']['phot']['lam_obs'] > 5e5) else False for dat in alldata])

	#### agn flags
	sfing, composite, agn = return_agn_str(np.ones_like(hflag,dtype=bool))
	agn_flags = [sfing,composite,agn]
	colors = ['blue','purple','red']
	labels = ['SFing/normal','composite', 'AGN']

	#### best-fits
	d1 = np.array([x['bfit']['maxprob_params'][d1_idx][0] for x in alldata])
	d2 = np.array([x['bfit']['maxprob_params'][d2_idx][0] for x in alldata])
	didx = np.array([x['bfit']['maxprob_params'][didx_idx][0] for x in alldata])
	sfr_100 = np.log10(np.clip([x['bfit']['sfr_100'] for x in alldata],1e-4,np.inf))

	#### total attenuation
	dusttot = np.zeros_like(d1)
	for ii in xrange(len(dusttot)): dusttot[ii] = -np.log10(threed_dutils.charlot_and_fall_extinction(5500.,d1[ii],d2[ii],-1.0,didx[ii], kriek=True))

	#### plotting
	fig, ax = plt.subplots(2,3, figsize = (22,12))

	ax[0,0].errorbar(didx,dusttot, fmt='o',alpha=0.6,linestyle=' ',color='0.4')
	ax[0,0].set_ylabel(r'total attenuation [5500 $\AA$]')
	ax[0,0].set_xlabel(r'dust_index')
	ax[0,0].axis((didx.min()-0.1,didx.max()+0.1,dusttot.min()-0.1,dusttot.max()+0.1))

	for flag,col in zip(agn_flags,colors): ax[0,1].errorbar(didx[flag],sfr_100[flag], fmt='o',alpha=0.6,linestyle=' ',color=col)
	ax[0,1].set_xlabel(r'dust_index')
	ax[0,1].set_ylabel(r'SFR$_{100 \mathrm{ Myr}}$ [M$_{\odot}$/yr]')
	ax[0,1].axis((didx.min()-0.1,didx.max()+0.1,sfr_100.min()-0.1,sfr_100.max()+0.1))
	for ii in xrange(len(labels)): ax[0,1].text(0.04,0.92-0.08*ii,labels[ii],color=colors[ii],transform = ax[0,1].transAxes,weight='bold')

	l_idx = didx > -0.45
	h_idx = didx < -0.45
	ax[0,2].errorbar(d1[h_idx],d2[h_idx], fmt='o',alpha=0.6,linestyle=' ',color='0.4')
	ax[0,2].errorbar(d1[l_idx],d2[l_idx], fmt='o',alpha=0.6,linestyle=' ',color='blue')
	ax[0,2].set_xlabel(r'dust1')
	ax[0,2].set_ylabel(r'dust2')
	ax[0,2].axis((d1.min()-0.1,d1.max()+0.1,d2.min()-0.1,d2.max()+0.1))
	ax[1,0].text(0.97,0.05,'dust_index > -0.45',color='blue',transform = ax[0,2].transAxes,weight='bold',ha='right')

	l_idx = didx > -0.45
	h_idx = didx < -0.45
	ax[1,0].errorbar(didx[~hflag],d1[~hflag]/d2[~hflag], fmt='o',alpha=0.6,linestyle=' ', label='no herschel',color='0.4')
	ax[1,0].errorbar(didx[hflag],d1[hflag]/d2[hflag], fmt='o',alpha=0.6,linestyle=' ', label='has herschel',color='red')
	ax[1,0].set_xlabel(r'dust_index')
	ax[1,0].set_ylabel(r'dust1/dust2')
	ax[1,0].axis((didx.min()-0.1,didx.max()+0.1,(d1/d2).min()-0.1,(d1/d2).max()+0.1))
	ax[1,0].text(0.05,0.92,'Herschel-detected',color='red',transform = ax[1,0].transAxes,weight='bold')

	plt.savefig(outfolder+'bestfit_param_comparison.png', dpi=300)
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


	alpha = 0.6
	fmt = 'o'

	##### find prospectr indexes
	parnames = alldata[0]['pquantiles']['parnames']
	idx_mass = parnames == 'mass'
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
			promass = np.concatenate((promass,np.atleast_2d(np.log10(tmp))),axis=0)
			magmass = np.concatenate((magmass,np.atleast_2d(data['magphys']['percentiles']['M*'][1:4])))

	proerrs = [promass[:,1]-promass[:,0],
	           promass[:,2]-promass[:,1]]
	magerrs = [magmass[:,1]-magmass[:,0],
	           magmass[:,2]-magmass[:,1]]
	mass.errorbar(promass[:,1],magmass[:,1],
		          fmt=fmt, alpha=alpha,
			      xerr=proerrs, yerr=magerrs, color='0.4')

	# labels
	mass.set_xlabel(r'log(M$_*$) [Prospectr]',labelpad=13)
	mass.set_ylabel(r'log(M$_*$) [MAGPHYS]')
	mass = threed_dutils.equalize_axes(mass,promass[:,1],magmass[:,1])

	# text
	off,scat = threed_dutils.offset_and_scatter(promass[:,1],magmass[:,1],biweight=True)
	mass.text(0.99,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
			  transform = mass.transAxes,horizontalalignment='right')
	mass.text(0.99,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
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
			      xerr=proerrs, yerr=magerrs, color='0.4')

	# labels
	sfr.set_xlabel(r'log(SFR) [Prospectr]')
	sfr.set_ylabel(r'log(SFR) [MAGPHYS]')
	sfr = threed_dutils.equalize_axes(sfr,prosfr[:,1],magsfr[:,1])

	# text
	off,scat = threed_dutils.offset_and_scatter(prosfr[:,1],magsfr[:,1],biweight=True)
	sfr.text(0.99,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
			  transform = sfr.transAxes,horizontalalignment='right')
	sfr.text(0.99,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
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
			      xerr=proerrs, color='0.4')

	# labels
	met.set_xlabel(r'log(Z/Z$_{\odot}$) [Prospectr]',labelpad=13)
	met.set_ylabel(r'log(Z/Z$_{\odot}$) [best-fit MAGPHYS]')
	met = threed_dutils.equalize_axes(met,promet[:,1],magmet)

	# text
	off,scat = threed_dutils.offset_and_scatter(promet[:,1],magmet,biweight=True)
	met.text(0.99,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
			  transform = met.transAxes,horizontalalignment='right')
	met.text(0.99,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
		      transform = met.transAxes,horizontalalignment='right')
	

	#### Balmer decrement
	bdec_magphys, bdec_prospectr = [],[]
	for ii,dat in enumerate(alldata):
		tau1 = dat['bfit']['maxprob_params'][dust1_idx][0]
		tau2 = dat['bfit']['maxprob_params'][dust2_idx][0]
		dindex = dat['bfit']['maxprob_params'][dinx_idx][0]
		bdec = threed_dutils.calc_balmer_dec(tau1, tau2, -1.0, dindex,kriek=True)
		bdec_prospectr.append(bdec)
		
		tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
		tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
		bdec = threed_dutils.calc_balmer_dec(tau1, tau2, -1.3, -0.7)
		bdec_magphys.append(np.squeeze(bdec))
	
	bdec_magphys = np.array(bdec_magphys)
	bdec_prospectr = np.array(bdec_prospectr)

	balm.errorbar(bdec_prospectr,bdec_magphys,
		          fmt=fmt, alpha=alpha, color='0.4')

	# labels
	balm.set_xlabel(r'Prospectr H$_{\alpha}$/H$_{\beta}$',labelpad=13)
	balm.set_ylabel(r'MAGPHYS H$_{\alpha}$/H$_{\beta}$')
	balm = threed_dutils.equalize_axes(balm,bdec_prospectr,bdec_magphys)

	# text
	off,scat = threed_dutils.offset_and_scatter(bdec_prospectr,bdec_magphys,biweight=True)
	balm.text(0.99,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
			  transform = balm.transAxes,horizontalalignment='right')
	balm.text(0.99,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
		      transform = balm.transAxes,horizontalalignment='right')

	#### Extinction
	tautot_magphys,tautot_prospectr,taudiff_magphys,taudiff_prospectr = [],[], [], []
	for ii,dat in enumerate(alldata):
		tau1 = dat['bfit']['maxprob_params'][dust1_idx][0]
		tau2 = dat['bfit']['maxprob_params'][dust2_idx][0]
		dindex = dat['bfit']['maxprob_params'][dinx_idx][0]

		dust2 = threed_dutils.charlot_and_fall_extinction(5500.,tau1,tau2,-1.0,dindex, kriek=True, nobc=True)
		dusttot = threed_dutils.charlot_and_fall_extinction(5500.,tau1,tau2,-1.0,dindex, kriek=True)
		taudiff_prospectr.append(-np.log(dust2))
		tautot_prospectr.append(-np.log10(dusttot))
		
		tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
		tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
		taudiff_magphys.append(tau2)
		tautot_magphys.append(tau1+tau2)
	
	taudiff_prospectr = np.array(taudiff_prospectr)
	taudiff_magphys = np.array(taudiff_magphys)
	tautot_magphys = np.array(tautot_magphys)
	tautot_prospectr = np.array(tautot_prospectr)

	ext_tot.errorbar(tautot_prospectr,tautot_magphys,
		          fmt=fmt, alpha=alpha, color='0.4')

	# labels
	ext_tot.set_xlabel(r'Prospectr total $\tau_{5500}$',labelpad=13)
	ext_tot.set_ylabel(r'MAGPHYS total $\tau_{5500}$')
	ext_tot = threed_dutils.equalize_axes(ext_tot,tautot_prospectr,tautot_magphys)

	# text
	off,scat = threed_dutils.offset_and_scatter(tautot_prospectr,tautot_magphys,biweight=True)
	ext_tot.text(0.99,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
			  transform = ext_tot.transAxes,horizontalalignment='right')
	ext_tot.text(0.99,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
		      transform = ext_tot.transAxes,horizontalalignment='right',)

	ext_diff.errorbar(taudiff_prospectr,taudiff_magphys,
		          fmt=fmt, alpha=alpha, color='0.4')

	# labels
	ext_diff.set_xlabel(r'Prospectr diffuse $\tau_{5500}$',labelpad=13)
	ext_diff.set_ylabel(r'MAGPHYS diffuse $\tau_{5500}$')
	ext_diff = threed_dutils.equalize_axes(ext_diff,taudiff_prospectr,taudiff_magphys)

	# text
	off,scat = threed_dutils.offset_and_scatter(taudiff_prospectr,taudiff_magphys,biweight=True)
	ext_diff.text(0.99,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
			  transform = ext_diff.transAxes,horizontalalignment='right')
	ext_diff.text(0.99,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
		      transform = ext_diff.transAxes,horizontalalignment='right')


	plt.tight_layout()
	plt.savefig(outfolder+'basic_comparison.png',dpi=300)
	plt.close()

def time_res_incr_comp(alldata_2,alldata_7):

	'''
	compare time_res_incr = 2, 7. input is 7. load 2 separately.
	'''

	mass_2 = np.array([f['bfit']['maxprob_params'][0] for f in alldata_2])
	mass_7 = np.array([f['bfit']['maxprob_params'][0] for f in alldata_7])

	sfr_2 = np.log10(np.clip([f['bfit']['sfr_100'] for f in alldata_2],1e-4,np.inf))
	sfr_7 = np.log10(np.clip([f['bfit']['sfr_100'] for f in alldata_7],1e-4,np.inf))

	fig, ax = plt.subplots(1,2, figsize = (18,8))

	ax[0].errorbar(np.log10(mass_2), np.log10(mass_7), fmt='o',alpha=0.6,linestyle=' ',color='0.4')
	ax[0].set_xlabel(r'log(best-fit M/M$_{\odot}$) [tres = 2]')
	ax[0].set_ylabel(r'log(best-fit M/M$_{\odot}$) [tres = 7]')
	ax[0] = threed_dutils.equalize_axes(ax[0], np.log10(mass_2),np.log10(mass_7))
	off,scat = threed_dutils.offset_and_scatter(np.log10(mass_2),np.log10(mass_7),biweight=True)
	ax[0].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat),
			  transform = ax[0].transAxes,horizontalalignment='right')
	ax[0].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off),
			      transform = ax[0].transAxes,horizontalalignment='right')

	ax[1].errorbar(sfr_2,sfr_7, fmt='o',alpha=0.6,linestyle=' ',color='0.4')
	ax[1].set_xlabel(r'log(best-fit SFR) [tres = 2]')
	ax[1].set_ylabel(r'log(best-fit SFR) [tres = 7]')
	ax[1] = threed_dutils.equalize_axes(ax[1], sfr_2,sfr_7)
	off,scat = threed_dutils.offset_and_scatter(sfr_2,sfr_7,biweight=True)
	ax[1].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat),
			  transform = ax[1].transAxes,horizontalalignment='right')
	ax[1].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off),
			      transform = ax[1].transAxes,horizontalalignment='right')

	plt.savefig(os.getenv('APPS')+'/threedhst_bsfh/plots/brownseds/pcomp/bestfit_mass_sfr_comp.png',dpi=300)
	plt.close()

	nparams = len(alldata_2[0]['bfit']['maxprob_params'])
	parnames = alldata_2[0]['pquantiles']['parnames']
	fig, ax = plt.subplots(4,3, figsize = (23/1.5,30/1.5))
	ax = np.ravel(ax)
	for ii in xrange(nparams):
		
		cent2 = np.array([dat['pquantiles']['q50'][ii] for dat in alldata_2])
		up2 = np.array([dat['pquantiles']['q84'][ii] for dat in alldata_2])
		down2 = np.array([dat['pquantiles']['q16'][ii] for dat in alldata_2])

		cent7 = np.array([dat['pquantiles']['q50'][ii] for dat in alldata_7])
		up7 = np.array([dat['pquantiles']['q84'][ii] for dat in alldata_7])
		down7 = np.array([dat['pquantiles']['q16'][ii] for dat in alldata_7])

		if ii == 0:
			errs2 = threed_dutils.asym_errors(cent2, up2, down2, log=True)
			cent2 = np.log10(cent2)
			errs7 = threed_dutils.asym_errors(cent7, up7, down7, log=True)
			cent7 = np.log10(cent7)
		else:
			errs2 = threed_dutils.asym_errors(cent2, up2, down2, log=False)
			errs7 = threed_dutils.asym_errors(cent7, up7, down7, log=False)

		ax[ii].errorbar(cent2,cent7,xerr=errs2,yerr=errs7,fmt='o',color='0.4',alpha=0.6)
		ax[ii].set_xlabel(parnames[ii]+' tres=2')
		ax[ii].set_ylabel(parnames[ii]+' tres=7')

		ax[ii] = threed_dutils.equalize_axes(ax[ii], cent2,cent7)
		off,scat = threed_dutils.offset_and_scatter(cent2,cent7,biweight=True)
		ax[ii].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat),
			  transform = ax[ii].transAxes,horizontalalignment='right')
		ax[ii].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off),
			      transform = ax[ii].transAxes,horizontalalignment='right')

	plt.tight_layout()
	plt.savefig(os.getenv('APPS')+'/threedhst_bsfh/plots/brownseds/pcomp/model_params_comp.png',dpi=100)
	plt.close()

	nparams = len(alldata_2[0]['pextras']['parnames'])
	parnames = alldata_2[0]['pextras']['parnames']
	fig, ax = plt.subplots(3,3, figsize = (18,18))
	ax = np.ravel(ax)
	for ii in xrange(nparams):
		
		cent2 = np.array([dat['pextras']['q50'][ii] for dat in alldata_2])
		up2 = np.array([dat['pextras']['q84'][ii] for dat in alldata_2])
		down2 = np.array([dat['pextras']['q16'][ii] for dat in alldata_2])

		cent7 = np.array([dat['pextras']['q50'][ii] for dat in alldata_7])
		up7 = np.array([dat['pextras']['q84'][ii] for dat in alldata_7])
		down7 = np.array([dat['pextras']['q16'][ii] for dat in alldata_7])

		errs2 = threed_dutils.asym_errors(cent2, up2, down2, log=True)
		errs7 = threed_dutils.asym_errors(cent7, up7, down7, log=True)
		if 'ssfr' in parnames[ii]:
			cent2 = np.log10(np.clip(cent2,1e-13,np.inf))
			cent7 = np.log10(np.clip(cent7,1e-13,np.inf))
		elif 'sfr' in parnames[ii]:
			cent2 = np.log10(np.clip(cent2,1e-4,np.inf))
			cent7 = np.log10(np.clip(cent7,1e-4,np.inf))
		elif 'emp_ha' in parnames[ii]:
			cent2 = np.log10(np.clip(cent2,1e37,np.inf))
			cent7 = np.log10(np.clip(cent7,1e37,np.inf))
		else:
			cent2 = np.log10(cent2)
			cent7 = np.log10(cent7)


		ax[ii].errorbar(cent2,cent7,xerr=errs2,yerr=errs7,fmt='o',color='0.4',alpha=0.6)
		ax[ii].set_xlabel('log('+parnames[ii]+') tres=2')
		ax[ii].set_ylabel('log('+parnames[ii]+') tres=7')

		ax[ii] = threed_dutils.equalize_axes(ax[ii], cent2,cent7)
		off,scat = threed_dutils.offset_and_scatter(cent2,cent7,biweight=True)
		ax[ii].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat)+' dex',
			  transform = ax[ii].transAxes,horizontalalignment='right')
		ax[ii].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off)+ 'dex',
			      transform = ax[ii].transAxes,horizontalalignment='right')

	plt.tight_layout()
	plt.savefig(os.getenv('APPS')+'/threedhst_bsfh/plots/brownseds/pcomp/derived_params_comp.png',dpi=100)
	plt.close()



