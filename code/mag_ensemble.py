import numpy as np
import matplotlib.pyplot as plt
import os, threed_dutils
import matplotlib as mpl
from astropy import constants
from astropy.cosmology import WMAP9
import magphys_plot_pref

minsfr = 1e-4

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

	# fields
	# flux, flux_errup, flux_errdown, lum, lum_errup, lum_errdown, eqw
	emline_names = alldata[0]['residuals']['emlines']['em_name']
	nlines = len(emline_names)

	# sigh... this is a hack
	if field == 'eqw_rest':
		fillvalue = np.zeros(shape=(3,nlines))
	else:
		fillvalue = np.zeros(nlines)

	if name is None:
		return np.squeeze(np.array([f['residuals']['emlines'][model][field] if f['residuals']['emlines'] is not None else fillvalue for f in alldata]))
	else:
		emline_names = alldata[0]['residuals']['emlines']['em_name']
		idx = emline_names == name
		if field == 'eqw_rest':
			return np.squeeze(np.array([f['residuals']['emlines'][model][field] if f['residuals']['emlines'] is not None else fillvalue for f in alldata])[:,:,idx])
		else:
			return np.squeeze(np.array([f['residuals']['emlines'][model][field] if f['residuals']['emlines'] is not None else fillvalue for f in alldata])[:,idx])

def compare_moustakas_fluxes(alldata,dat,emline_names,objnames,outname='test.png',outdec='bdec.png'):

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

	xplot = remove_doublets(np.transpose(ret_inf(alldata,'flux',model='Prospectr')),emline_names)[:,ind]
	xplot_errup = remove_doublets(np.transpose(ret_inf(alldata,'flux_errup',model='Prospectr')),emline_names)[:,ind]
	xplot_errdown = remove_doublets(np.transpose(ret_inf(alldata,'flux_errdown',model='Prospectr')),emline_names)[:,ind]
	eqw = remove_doublets(np.transpose(ret_inf(alldata,'eqw_rest',model='Prospectr'))[:,0,:],emline_names)[:,ind]

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

		axes[ii].set_ylabel('Moustakas+10 - measured '+emline_names_doubrem[ii])
		axes[ii].set_xlabel('EQW '+emline_names_doubrem[ii])
		off,scat = threed_dutils.offset_and_scatter(np.log10(xp),np.log10(yp),biweight=True)
		axes[ii].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat) +' dex',
				  transform = axes[ii].transAxes,horizontalalignment='right')
		axes[ii].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off) + ' dex',
			      transform = axes[ii].transAxes,horizontalalignment='right')

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
		xplot = ret_inf(alldata,'eqw_rest',model='Prospectr',name=emname)
		idx = np.isfinite(yplot)

		axes[ii].plot(xplot[idx,0], yplot[idx], 
			    	  'o',
			    	  alpha=0.6)
		
		xlabel = r"{0} EQW [Prospectr]"
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

def obs_vs_ks_ha(alldata,emline_names,outname='test.png'):
	
	#################
	#### plot observed Halpha versus model Halpha from KS relationship
	#################
	sn_cut = 10

	f_ha = ret_inf(alldata,'flux',model='Prospectr',name='H$\\alpha$')
	f_ha_errup = ret_inf(alldata,'flux_errup',model='Prospectr',name='H$\\alpha$')
	f_ha_errdown = ret_inf(alldata,'flux_errdown',model='Prospectr',name='H$\\alpha$')
	err_ha = f_ha_errup - f_ha_errdown
	sn_ha = f_ha / err_ha

	keep_idx = np.squeeze(sn_ha > sn_cut)

	##### indexes
	mparnames = alldata[0]['model']['parnames']
	mu_idx = mparnames == 'mu'
	tauv_idx = mparnames == 'tauv'

	##### Pull dust + SFR information
	ha_mag, ha_pro = [], []
	for ii,dat in enumerate(alldata):
		if keep_idx[ii]:
			'''
			tau1 = dat['pquantiles']['maxprob_params'][dust1_idx][0]
			tau2 = dat['pquantiles']['maxprob_params'][dust2_idx][0]
			dindex = dat['pquantiles']['maxprob_params'][dinx_idx][0]
			ext = charlot_and_fall_extinction(6563.0,tau1, tau2, -1.0, dindex)
			'''
			ha_pro.append(dat['pextras']['q50'][-2])
			if ii == 0:
				print 'this should be halpha'
				print dat['pextras']['parnames'][-2]

			tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
			tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
			
			ha_mag.append(threed_dutils.synthetic_halpha(dat['sfr_10'], tau1, tau2, -1.3, -0.7))

			pc2cm = 3.08567758e18
			distance = WMAP9.luminosity_distance(dat['residuals']['phot']['z']).value*1e6*pc2cm
			dfactor = (4*np.pi*distance**2)
			f_ha[ii] = f_ha[ii] * dfactor

	ha_mag = np.log10(np.array(ha_mag))
	ha_pro = np.log10(np.array(ha_pro))

	fig, ax = plt.subplots(1,3, figsize = (22,6))

	ax[0].errorbar(ha_mag, ha_pro, fmt='o',alpha=0.6,linestyle=' ')
	ax[0].set_xlabel(r'log(KS H$_{\alpha}$) [MAGPHYS]')
	ax[0].set_ylabel(r'log(KS H$_{\alpha}$) [Prospectr]')
	ax[0] = threed_dutils.equalize_axes(ax[0], ha_mag, ha_pro)
	off,scat = threed_dutils.offset_and_scatter(ha_mag, ha_pro,biweight=True)
	ax[0].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat) +' dex',
			  transform = ax[0].transAxes,horizontalalignment='right')
	ax[0].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off)+ ' dex',
			      transform = ax[0].transAxes,horizontalalignment='right')

	ax[1].errorbar(ha_mag,np.log10(f_ha[keep_idx]), fmt='o',alpha=0.6,linestyle=' ')
	ax[1].set_xlabel(r'log(KS H$_{\alpha}$) [MAGPHYS]')
	ax[1].set_ylabel(r'log(H$_{\alpha}$) [observed]')
	ax[1] = threed_dutils.equalize_axes(ax[1], ha_mag, np.log10(f_ha[keep_idx]))
	off,scat = threed_dutils.offset_and_scatter(ha_mag, np.log10(f_ha[keep_idx]),biweight=True)
	ax[1].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat) +' dex',
			  transform = ax[1].transAxes,horizontalalignment='right')
	ax[1].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off)+ ' dex',
			      transform = ax[1].transAxes,horizontalalignment='right')

	ax[2].errorbar(ha_pro, np.log10(f_ha[keep_idx]), fmt='o',alpha=0.6,linestyle=' ')
	ax[2].set_xlabel(r'log(KS H$_{\alpha}$) [Prospectr]')
	ax[2].set_ylabel(r'log(H$_{\alpha}$) [observed]')
	ax[2] = threed_dutils.equalize_axes(ax[2], ha_pro, np.log10(f_ha[keep_idx]))
	off,scat = threed_dutils.offset_and_scatter(ha_pro, np.log10(f_ha[keep_idx]),biweight=True)
	ax[2].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat) +' dex',
			  transform = ax[2].transAxes,horizontalalignment='right')
	ax[2].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off)+ ' dex',
			      transform = ax[2].transAxes,horizontalalignment='right')


	plt.savefig(outname,dpi=300)
	plt.close()

def obs_vs_prosp_ha(alldata,emline_names,outname='test.png'):

	#################
	#### plot observed Halpha versus expected (PROSPECTR ONLY)
	#################
	# first pull out observed Halphas
	# add an S/N cut... ? remove later maybe
	sn_cut = 5

	f_ha = ret_inf(alldata,'flux',model='Prospectr',name='H$\\alpha$')
	f_ha_errup = ret_inf(alldata,'flux_errup',model='Prospectr',name='H$\\alpha$')
	f_ha_errdown = ret_inf(alldata,'flux_errdown',model='Prospectr',name='H$\\alpha$')
	err_ha = f_ha_errup - f_ha_errdown
	sn_ha = f_ha / err_ha

	keep_idx = np.squeeze(sn_ha > sn_cut)

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

	fig, ax = plt.subplots(1,1, figsize = (10,10))
	xplot = np.log10(model_ha[keep_idx][:,0])
	yplot = np.log10(f_ha[keep_idx])
	yerr = threed_dutils.asym_errors(f_ha[keep_idx],f_ha_errup[keep_idx], f_ha_errdown[keep_idx],log=True)
	ax.errorbar(xplot, yplot, yerr=yerr,fmt='o',alpha=0.6,linestyle=' ',color='grey')
	ax.set_ylabel(r'log(observed H$_{\alpha}$)')
	ax.set_xlabel(r'log(best-fit Prospectr H$_{\alpha}$)')
	ax = threed_dutils.equalize_axes(ax,xplot,yplot)
	off,scat = threed_dutils.offset_and_scatter(xplot,yplot,biweight=True)
	ax.text(0.99,0.05, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
			  transform = ax.transAxes,horizontalalignment='right')
	ax.text(0.99,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = ax.transAxes,horizontalalignment='right')
	plt.savefig(outname,dpi=300)
	plt.close()

	pinfo = {}
	pinfo['model_ha'] = model_ha
	pinfo['f_ha'] = f_ha
	pinfo['f_ha_errup'] = f_ha_errup
	pinfo['f_ha_errdown'] = f_ha_errdown

	return pinfo

def obs_vs_model_bdec(alldata,emline_names,outname='test.png'):

	#################
	#### plot observed Balmer decrement versus expected
	#################
	sn_cut = 10

	f_ha = ret_inf(alldata,'flux',model='Prospectr',name='H$\\alpha$')
	err_ha = ret_inf(alldata,'flux_errup',model='Prospectr',name='H$\\alpha$') - ret_inf(alldata,'flux_errdown',model='Prospectr',name='H$\\alpha$')
	f_hb = ret_inf(alldata,'flux',model='Prospectr',name='H$\\beta$')
	err_hb = ret_inf(alldata,'flux_errup',model='Prospectr',name='H$\\beta$') - ret_inf(alldata,'flux_errdown',model='Prospectr',name='H$\\beta$')
	
	sn_ha = f_ha / err_ha
	sn_hb = f_hb / err_hb

	#### for now, aggressive S/N cuts
	# S/N(Ha) > 10, S/N (Hbeta) > 10
	keep_idx = np.squeeze((sn_ha > sn_cut) & (sn_hb > sn_cut) & (f_ha > 0) & (f_hb > 0))

	bdec_measured = f_ha / f_hb

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

	bdec_magphys, bdec_prospectr = [],[]
	ptau1, ptau2, pdindex = [], [], []
	for dat in alldata:
		ptau1.append(dat['pquantiles']['maxprob_params'][dust1_idx][0])
		ptau2.append(dat['pquantiles']['maxprob_params'][dust2_idx][0])
		pdindex.append(dat['pquantiles']['maxprob_params'][dinx_idx][0])
		bdec = threed_dutils.calc_balmer_dec(ptau1[-1], ptau2[-1], -1.0, pdindex[-1],kriek=True)
		bdec_prospectr.append(bdec)
		
		tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
		tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
		bdec = threed_dutils.calc_balmer_dec(tau1, tau2, -1.3, -0.7)
		bdec_magphys.append(np.squeeze(bdec))
	
	pl_bdec_magphys = np.array(bdec_magphys)[keep_idx]
	pl_bdec_prospectr = np.array(bdec_prospectr)[keep_idx]
	pl_bdec_measured = bdec_measured[keep_idx]

	
	fig, ax = plt.subplots(1,3, figsize = (22,6))

	ax[0].errorbar(pl_bdec_measured, pl_bdec_prospectr, fmt='o',alpha=0.6,linestyle=' ')
	ax[0].set_xlabel(r'observed H$_{\alpha}$/H$_{\beta}$')
	ax[0].set_ylabel(r'Prospectr H$_{\alpha}$/H$_{\beta}$')
	ax[0] = threed_dutils.equalize_axes(ax[0], pl_bdec_measured,pl_bdec_prospectr)
	off,scat = threed_dutils.offset_and_scatter(pl_bdec_measured,pl_bdec_prospectr,biweight=True)
	ax[0].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat),
			  transform = ax[0].transAxes,horizontalalignment='right')
	ax[0].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off),
			      transform = ax[0].transAxes,horizontalalignment='right')

	ax[1].errorbar(pl_bdec_measured, pl_bdec_magphys, fmt='o',alpha=0.6,linestyle=' ')
	ax[1].set_xlabel(r'observed H$_{\alpha}$/H$_{\beta}$')
	ax[1].set_ylabel(r'MAGPHYS H$_{\alpha}$/H$_{\beta}$')
	ax[1] = threed_dutils.equalize_axes(ax[1], pl_bdec_measured,pl_bdec_magphys)
	off,scat = threed_dutils.offset_and_scatter(pl_bdec_measured,pl_bdec_magphys,biweight=True)
	ax[1].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat),
			  transform = ax[1].transAxes,horizontalalignment='right')
	ax[1].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off),
			      transform = ax[1].transAxes,horizontalalignment='right')

	ax[2].errorbar(pl_bdec_prospectr, pl_bdec_magphys, fmt='o',alpha=0.6,linestyle=' ')
	ax[2].set_xlabel(r'Prospectr H$_{\alpha}$/H$_{\beta}$')
	ax[2].set_ylabel(r'MAGPHYS H$_{\alpha}$/H$_{\beta}$')
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

def residual_plots(alldata,obs_info, bdec_info):
	# bdec_info: bdec_magphys, bdec_prospectr, bdec_measured, keep_idx, dust1, dust2, dust2_index
	# obs_info: model_ha, f_ha, f_ha_errup, f_ha_errdown

	fldr = '/Users/joel/code/python/threedhst_bsfh/plots/brownseds/magphys/emlines_comp/residuals/'
	idx = bdec_info['keep_idx']

	#### bdec resid versus ha resid
	bdec_resid = bdec_info['bdec_prospectr'][idx] - bdec_info['bdec_measured'][idx]
	ha_resid = np.log10(obs_info['f_ha'][idx]) - np.log10(obs_info['model_ha'][idx,0])

	fig, ax = plt.subplots(1,1, figsize = (8,8))

	ax.errorbar(ha_resid, bdec_resid, fmt='o',alpha=0.6,linestyle=' ')
	ax.set_xlabel(r'log(Prospectr/obs) [H$_{\alpha}$]')
	ax.set_ylabel(r'Prospectr - obs [Balmer decrement]')
	
	plt.savefig(fldr+'bdec_resid_versus_ha_resid.png', dpi=300)

	#### dust1 / dust2
	fig, ax = plt.subplots(1,2, figsize = (18,8))

	ax[0].errorbar(bdec_info['dust1'][idx]/bdec_info['dust2'][idx], bdec_resid, fmt='o',alpha=0.6,linestyle=' ')
	ax[0].set_xlabel(r'dust1/dust2')
	ax[0].set_ylabel(r'Prospectr - obs [Balmer decrement]')

	ax[1].errorbar(bdec_info['dust1'][idx]/bdec_info['dust2'][idx], ha_resid, fmt='o',alpha=0.6,linestyle=' ')
	ax[1].set_xlabel(r'dust1/dust2')
	ax[1].set_ylabel(r'log(Prospectr/obs) [H$_{\alpha}$]')
	
	plt.savefig(fldr+'dust1_dust2.png', dpi=300)

	#### dust2_index
	fig, ax = plt.subplots(1,2, figsize = (18,8))

	ax[0].errorbar(bdec_info['dust2_index'][idx], bdec_resid, fmt='o',alpha=0.6,linestyle=' ')
	ax[0].set_xlabel(r'dust2_index')
	ax[0].set_ylabel(r'Prospectr - obs [Balmer decrement]')

	ax[1].errorbar(bdec_info['dust2_index'][idx], ha_resid, fmt='o',alpha=0.6,linestyle=' ')
	ax[1].set_xlabel(r'dust2_index')
	ax[1].set_ylabel(r'log(Prospectr/obs) [H$_{\alpha}$]')
	
	plt.savefig(fldr+'dust2_index.png', dpi=300)

	#### dust2_index vs dust1/dust2
	fig, ax = plt.subplots(1,1, figsize = (8,8))

	ax.errorbar(bdec_info['dust2_index'], bdec_info['dust1']/bdec_info['dust2'], fmt='o',alpha=0.6,linestyle=' ')
	ax.set_xlabel(r'dust2_index')
	ax.set_ylabel(r'dust1/dust2')

	plt.savefig(fldr+'idx_versus_ratio.png', dpi=300)

	print 1/0
def plot_emline_comp(alldata,outfolder):
	'''
	emission line luminosity comparisons:
		(1) Observed luminosity, Prospectr vs MAGPHYS continuum subtraction
		(2) Moustakas+10 comparisons
		(3) model Balmer decrement (from dust) versus observed Balmer decrement
		(4) model Halpha (from KS + dust) versus observed Halpha
	'''

	##### Pull relevant information out of alldata
	emline_names = alldata[0]['residuals']['emlines']['em_name']

	##### load moustakas information
	objnames = objnames = np.array([f['objname'] for f in alldata])
	dat = threed_dutils.load_moustakas_data(objnames = list(objnames))
	

	##### do work
	compare_model_flux(alldata,emline_names,outname = outfolder+'continuum_model_flux_comparison.png')
	compare_moustakas_fluxes(alldata,dat,emline_names,objnames,
							 outname=outfolder+'continuum_model_flux_comparison.png',
							 outdec=outfolder+'balmer_dec_comparison.png')
	
	obs_info = obs_vs_prosp_ha(alldata,emline_names,outname=outfolder+'halpha_comparison.png')
	bdec_info = obs_vs_model_bdec(alldata,emline_names,outname=outfolder+'bdec_comparison.png')
	obs_vs_ks_ha(alldata,emline_names,outname=outfolder+'ks_comparison.png')
	residual_plots(alldata,obs_info, bdec_info)
	print 1/0

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

	plt.savefig(outfolder+'mass_metallicity.png',dpi=300)
	plt.close

def plot_comparison(alldata,outfolder):

	'''
	mass vs mass
	sfr vs sfr
	etc
	'''

	##### set up plots
	fig = plt.figure(figsize=(15,10))
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
			      xerr=proerrs, yerr=magerrs)

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
			      xerr=proerrs, yerr=magerrs)

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
			      xerr=proerrs)

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
		tau1 = dat['pquantiles']['maxprob_params'][dust1_idx][0]
		tau2 = dat['pquantiles']['maxprob_params'][dust2_idx][0]
		dindex = dat['pquantiles']['maxprob_params'][dinx_idx][0]
		bdec = threed_dutils.calc_balmer_dec(tau1, tau2, -1.0, dindex)
		bdec_prospectr.append(bdec)
		
		tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
		tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
		bdec = threed_dutils.calc_balmer_dec(tau1, tau2, -1.3, -0.7)
		bdec_magphys.append(np.squeeze(bdec))
	
	bdec_magphys = np.array(bdec_magphys)
	bdec_prospectr = np.array(bdec_prospectr)

	balm.errorbar(bdec_prospectr,bdec_magphys,
		          fmt=fmt, alpha=alpha)

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
		tau1 = dat['pquantiles']['maxprob_params'][dust1_idx][0]
		
		tau2 = dat['pquantiles']['maxprob_params'][dust2_idx][0]
		taudiff_prospectr.append(tau2)
		tautot_prospectr.append(tau1+tau2)
		
		tau1 = (1-dat['model']['parameters'][mu_idx][0])*dat['model']['parameters'][tauv_idx][0]
		tau2 = dat['model']['parameters'][mu_idx][0]*dat['model']['parameters'][tauv_idx][0]
		taudiff_magphys.append(tau2)
		tautot_magphys.append(tau1+tau2)
	
	taudiff_prospectr = np.array(taudiff_prospectr)
	taudiff_magphys = np.array(taudiff_magphys)
	tautot_magphys = np.array(tautot_magphys)
	tautot_prospectr = np.array(tautot_prospectr)

	ext_tot.errorbar(tautot_prospectr,tautot_magphys,
		          fmt=fmt, alpha=alpha)

	# labels
	ext_tot.set_xlabel(r'Prospectr total $\tau_{5500}$',labelpad=13)
	ext_tot.set_ylabel(r'MAGPHYS total $\tau_{5500}$')
	ext_tot = threed_dutils.equalize_axes(ext_tot,tautot_prospectr,tautot_magphys)

	# text
	off,scat = threed_dutils.offset_and_scatter(tautot_prospectr,tautot_magphys,biweight=True)
	ext_tot.text(0.99,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
			  transform = ext_tot.transAxes,horizontalalignment='right')
	ext_tot.text(0.99,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
		      transform = ext_tot.transAxes,horizontalalignment='right')

	ext_diff.errorbar(taudiff_prospectr,taudiff_magphys,
		          fmt=fmt, alpha=alpha)

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


	plt.savefig(outfolder+'basic_comparison.png',dpi=300)
	plt.close()
