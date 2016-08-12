import threed_dutils
import numpy as np
import matplotlib.pyplot as plt
import os
import magphys_plot_pref
import pickle
from corner import quantile
from astropy import constants
import brown_io
from matplotlib.ticker import MaxNLocator
from extra_output import post_processing

def bdec_to_ext(bdec):
	'''
	shamelessly ripped from mag_ensemble
	'''
	return 2.5*np.log10(bdec/2.86)

def make_plots(runname_nh='brownseds_np_nohersch',runname_h='brownseds_np', recollate_data = False):

	outpickle = '/Users/joel/code/python/threedhst_bsfh/data/'+runname_nh+'_extradat.pickle'
	if not os.path.isfile(outpickle) or recollate_data == True:
		alldata = collate_data(runname_nh=runname_nh,runname_h=runname_h,outpickle=outpickle)
	else:
		with open(outpickle, "rb") as f:
			alldata=pickle.load(f)

	outfolder = '/Users/joel/code/python/threedhst_bsfh/plots/'+runname_nh+'/paper_plots/'
	if not os.path.exists(outfolder):
		os.makedirs(outfolder)

	plot_lir(alldata,outfolder=outfolder)
	plot_halpha(alldata,outfolder=outfolder)
	plot_bdec(alldata,outfolder=outfolder)
	plot_dustpars(alldata,outfolder=outfolder)

def ha_extinction(sample_results):
	parnames = np.array(sample_results['quantiles']['parnames'])

	flatchain = sample_results['flatchain']
	ransamp = flatchain[np.random.choice(flatchain.shape[0], 4000, replace=False),:]
	d1 = ransamp[:,parnames=='dust1']
	d2 = ransamp[:,parnames=='dust2']
	didx = ransamp[:,parnames=='dust_index']
	ha_ext = threed_dutils.charlot_and_fall_extinction(6563.0, d1, d2, -1.0, didx, kriek=True)

	return quantile(ha_ext, [0.16, 0.5, 0.84])

def collate_data(runname_nh=None, runname_h=None,outpickle=None):

	filebase_nh, parm_basename_nh, ancilname_nh=threed_dutils.generate_basenames(runname_nh)
	filebase_h, parm_basename_h, ancilname_h=threed_dutils.generate_basenames(runname_h)
	alldata = brown_io.load_alldata(runname=runname_h) # for observed Halpha
	objname = [dat['objname'] for dat in alldata]

	out = []
	for jj in xrange(len(filebase_nh)):

		### load both 
		outdat_nh = {}
		outdat_h = {}
		obs = {}
		try:
			sample_results_nh, powell_results_nh, model_nh = threed_dutils.load_prospector_data(filebase_nh[jj])
		except:
			print 'failed to load number ' + str(int(jj))
			continue

		### match to filebase_nh
		name = filebase_nh[jj].split('_')[-1]
		match = [s for s in filebase_h if name in s][0]
		sample_results_h, powell_results_h, model_h = threed_dutils.load_prospector_data(match)

		### save CLOUDY-marginalized Halpha

		try: 
			linenames = sample_results_nh['model_emline']['emnames']
		except KeyError:
			param_name = os.getenv('APPS')+'/threed'+sample_results_nh['run_params']['param_file'].split('/threed')[1]
			post_processing(param_name, add_extra=True)
			sample_results_nh, powell_results_nh, model_nh = threed_dutils.load_prospector_data(filebase_nh[jj])

		ha_em = linenames == 'Halpha'
		outdat_nh['ha_q50'] = sample_results_nh['model_emline']['flux']['q50'][ha_em]
		outdat_nh['ha_q84'] = sample_results_nh['model_emline']['flux']['q84'][ha_em]
		outdat_nh['ha_q16'] = sample_results_nh['model_emline']['flux']['q16'][ha_em]

		outdat_h['ha_q50'] = sample_results_h['model_emline']['flux']['q50'][ha_em]
		outdat_h['ha_q84'] = sample_results_h['model_emline']['flux']['q84'][ha_em]
		outdat_h['ha_q16'] = sample_results_h['model_emline']['flux']['q16'][ha_em]

		### save model Balmer decrement
		epars = sample_results_nh['extras']['parnames']
		bdec_idx = epars == 'bdec_cloudy'
		outdat_nh['bdec_q50'] = sample_results_nh['extras']['q50'][bdec_idx]
		outdat_nh['bdec_q84'] = sample_results_nh['extras']['q84'][bdec_idx]
		outdat_nh['bdec_q16'] = sample_results_nh['extras']['q16'][bdec_idx]

		epars = sample_results_h['extras']['parnames']
		bdec_idx = epars == 'bdec_cloudy'
		outdat_h['bdec_q50'] = sample_results_h['extras']['q50'][bdec_idx]
		outdat_h['bdec_q84'] = sample_results_h['extras']['q84'][bdec_idx]
		outdat_h['bdec_q16'] = sample_results_h['extras']['q16'][bdec_idx]

		### save SFRs
		epars = sample_results_nh['extras']['parnames']
		idx = epars == 'sfr_100'
		outdat_nh['sfr100_q50'] = sample_results_nh['extras']['q50'][idx]
		outdat_nh['sfr100_q84'] = sample_results_nh['extras']['q84'][idx]
		outdat_nh['sfr100_q16'] = sample_results_nh['extras']['q16'][idx]

		epars = sample_results_h['extras']['parnames']
		idx = epars == 'sfr_100'
		outdat_h['sfr100_q50'] = sample_results_h['extras']['q50'][idx]
		outdat_h['sfr100_q84'] = sample_results_h['extras']['q84'][idx]
		outdat_h['sfr100_q16'] = sample_results_h['extras']['q16'][idx]

		epars = sample_results_nh['extras']['parnames']
		idx = epars == 'sfr_10'
		outdat_nh['sfr10_q50'] = sample_results_nh['extras']['q50'][idx]
		outdat_nh['sfr10_q84'] = sample_results_nh['extras']['q84'][idx]
		outdat_nh['sfr10_q16'] = sample_results_nh['extras']['q16'][idx]

		epars = sample_results_h['extras']['parnames']
		idx = epars == 'sfr_10'
		outdat_h['sfr10_q50'] = sample_results_h['extras']['q50'][idx]
		outdat_h['sfr10_q84'] = sample_results_h['extras']['q84'][idx]
		outdat_h['sfr10_q16'] = sample_results_h['extras']['q16'][idx]

		### save dust parameters
		outdat_nh['ha_ext_q16'], outdat_nh['ha_ext_q50'], outdat_nh['ha_ext_q84'] = ha_extinction(sample_results_nh)
		outdat_h['ha_ext_q16'],  outdat_h['ha_ext_q50'],  outdat_h['ha_ext_q84'] = ha_extinction(sample_results_h)

		### save fit parameters
		outdat_h['parnames'] = np.array(sample_results_h['quantiles']['parnames'])
		outdat_h['q50'] = sample_results_h['quantiles']['q50']
		outdat_h['q84'] = sample_results_h['quantiles']['q84']
		outdat_h['q16'] = sample_results_h['quantiles']['q16']

		outdat_nh['parnames'] = np.array(sample_results_nh['quantiles']['parnames'])
		outdat_nh['q50'] = sample_results_nh['quantiles']['q50']
		outdat_nh['q84'] = sample_results_nh['quantiles']['q84']
		outdat_nh['q16'] = sample_results_nh['quantiles']['q16']

		### save observed Halpha + Balmer decrement
		match = [i for i, s in enumerate(objname) if name in s][0]

		emline_names = alldata[match]['residuals']['emlines']['em_name']
		idx = emline_names == 'H$\\alpha$'
		obs['ha'] = alldata[match]['residuals']['emlines']['obs']['lum'][idx][0] / constants.L_sun.cgs.value
		obs['ha_up'] = alldata[match]['residuals']['emlines']['obs']['lum_errup'][idx][0] / constants.L_sun.cgs.value
		obs['ha_down'] = alldata[match]['residuals']['emlines']['obs']['lum_errdown'][idx][0] / constants.L_sun.cgs.value

		idx = emline_names == 'H$\\beta$'
		obs['hb'] = alldata[match]['residuals']['emlines']['obs']['lum'][idx][0] / constants.L_sun.cgs.value
		obs['hb_up'] = alldata[match]['residuals']['emlines']['obs']['lum_errup'][idx][0] / constants.L_sun.cgs.value
		obs['hb_down'] = alldata[match]['residuals']['emlines']['obs']['lum_errdown'][idx][0] / constants.L_sun.cgs.value

		### save both L_IRs
		outdat_h['lir'] = sample_results_h['observables']['L_IR']
		outdat_nh['lir'] = sample_results_nh['observables']['L_IR']

		outtemp = {'nohersch':outdat_nh,'hersch':outdat_h,'obs':obs}
		out.append(outtemp)
		
	pickle.dump(out,open(outpickle, "wb"))
	return out

def plot_lir(alldata,outfolder=None):

	'''
	LIR measurements are FUCKED UP
	fix before this plot becomes useful
	'''

	minsfr = 1e-2

	##### LIR
	hersch_lir = np.log10([quantile(dat['hersch']['lir'],[0.5])[0] for dat in alldata])
	good = np.isfinite(hersch_lir)
	hersch_lir = hersch_lir[good]
	hersch_lir_up = np.log10([quantile(dat['hersch']['lir'],[0.84])[0] for dat in alldata])[good]
	hersch_lir_do = np.log10([quantile(dat['hersch']['lir'],[0.16])[0] for dat in alldata])[good]
	hersch_lir_err = threed_dutils.asym_errors(hersch_lir,hersch_lir_up,hersch_lir_do,log=False)

	nohersch_lir = np.log10([quantile(dat['nohersch']['lir'],[0.5])[0] for dat in alldata])[good]
	nohersch_lir_up = np.log10([quantile(dat['nohersch']['lir'],[0.84])[0] for dat in alldata])[good]
	nohersch_lir_do = np.log10([quantile(dat['nohersch']['lir'],[0.16])[0] for dat in alldata])[good]
	nohersch_lir_err = threed_dutils.asym_errors(nohersch_lir,nohersch_lir_up,nohersch_lir_do,log=False)

	##### SFRs
	nh_sfr100 = np.log10(np.clip(np.squeeze([dat['nohersch']['sfr100_q50'] for dat in alldata]),minsfr,np.inf))
	nh_sfr100_up = np.log10(np.clip(np.squeeze([dat['nohersch']['sfr100_q84'] for dat in alldata]),minsfr,np.inf))
	nh_sfr100_do = np.log10(np.clip(np.squeeze([dat['nohersch']['sfr100_q16'] for dat in alldata]),minsfr,np.inf))
	nh_sfr100_err = threed_dutils.asym_errors(nh_sfr100,nh_sfr100_up,nh_sfr100_do,log=False)

	h_sfr100 = np.log10(np.clip(np.squeeze([dat['hersch']['sfr100_q50'] for dat in alldata]),minsfr,np.inf))
	h_sfr100_up = np.log10(np.clip(np.squeeze([dat['hersch']['sfr100_q84'] for dat in alldata]),minsfr,np.inf))
	h_sfr100_do = np.log10(np.clip(np.squeeze([dat['hersch']['sfr100_q16'] for dat in alldata]),minsfr,np.inf))
	h_sfr100_err = threed_dutils.asym_errors(h_sfr100,h_sfr100_up,h_sfr100_do,log=False)

	##### PLOTS
	fig, axes = plt.subplots(1, 2, figsize = (12.5,6))
	ax = np.ravel(axes)

	### LIR
	ax[0].errorbar(hersch_lir,nohersch_lir,xerr=hersch_lir_err,yerr=nohersch_lir_err,fmt='o',alpha=0.8,color='#1C86EE')
	ax[0].set_xlabel('log(L$_{\mathrm{IR}}$ fit) [Herschel]')
	ax[0].set_ylabel('log(L$_{\mathrm{IR}}$ fit) [no Herschel]')

	ax[0].xaxis.set_major_locator(MaxNLocator(5))
	ax[0].yaxis.set_major_locator(MaxNLocator(5))

	off,scat = threed_dutils.offset_and_scatter(hersch_lir,nohersch_lir,biweight=True)
	ax[0].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
			  transform = ax[0].transAxes,horizontalalignment='right')
	ax[0].text(0.96,0.10, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = ax[0].transAxes,horizontalalignment='right')
	ax[0] = threed_dutils.equalize_axes(ax[0], hersch_lir,nohersch_lir)

	### SFR10
	ax[1].errorbar(h_sfr100,nh_sfr100,xerr=h_sfr100_err,yerr=nh_sfr100_err,fmt='o',alpha=0.8,color='#1C86EE')
	ax[1].set_xlabel('log(SFR$_{100 \mathrm{Myr}}$) [Herschel]')
	ax[1].set_ylabel('log(SFR$_{100 \mathrm{Myr}}$)  [no Herschel]')

	ax[1].xaxis.set_major_locator(MaxNLocator(5))
	ax[1].yaxis.set_major_locator(MaxNLocator(5))

	off,scat = threed_dutils.offset_and_scatter(h_sfr100,nh_sfr100,biweight=True)
	ax[1].text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
			  transform = ax[1].transAxes,horizontalalignment='right')
	ax[1].text(0.96,0.10, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = ax[1].transAxes,horizontalalignment='right')
	ax[1] = threed_dutils.equalize_axes(ax[1], h_sfr100,nh_sfr100)

	plt.tight_layout()
	plt.savefig(outfolder+'lir_comparison.png',dpi=150)
	plt.close()

def plot_halpha(alldata,outfolder=None):

	hcolor = '#FF3D0D' #red
	nhcolor = '#1C86EE' #blue

	#### observations
	ha_lum = np.array([dat['obs']['ha'] for dat in alldata])
	ha_lum_up = np.array([dat['obs']['ha_up'] for dat in alldata])
	ha_lum_do = np.array([dat['obs']['ha_down'] for dat in alldata])
	ha_lum_err = (ha_lum_up - ha_lum_do)/2.

	#### model ha
	nohersch_ha = np.squeeze([dat['nohersch']['ha_q50'] for dat in alldata])
	nohersch_ha_up = np.squeeze([dat['nohersch']['ha_q84'] for dat in alldata])
	nohersch_ha_do = np.squeeze([dat['nohersch']['ha_q16'] for dat in alldata])

	hersch_ha = np.squeeze([dat['hersch']['ha_q50'] for dat in alldata])
	hersch_ha_up = np.squeeze([dat['hersch']['ha_q84'] for dat in alldata])
	hersch_ha_do = np.squeeze([dat['hersch']['ha_q16'] for dat in alldata])

	#### observed halpha cuts
	sn_ha = ha_lum / ha_lum_err
	idx = (ha_lum > 0.0) & (sn_ha > 3.0) & (nohersch_ha > 0.0) & (hersch_ha > 0.0)

	#### generate plot quantities
	ha_obs_err = threed_dutils.asym_errors(ha_lum[idx],ha_lum_up[idx],ha_lum_do[idx],log=True)
	ha_obs = np.log10(ha_lum[idx])
	ha_nh_err = threed_dutils.asym_errors(nohersch_ha[idx],nohersch_ha_up[idx],nohersch_ha_do[idx],log=True)
	ha_nh = np.log10(nohersch_ha[idx])
	ha_h_err = threed_dutils.asym_errors(hersch_ha[idx],hersch_ha_up[idx],hersch_ha_do[idx],log=True)
	ha_h = np.log10(hersch_ha[idx])

	fig, ax = plt.subplots(1, 1, figsize = (7,7))

	ax.errorbar(ha_obs,ha_h,xerr=ha_obs_err,yerr=ha_h_err,fmt='o',alpha=0.8,color=hcolor)
	ax.errorbar(ha_obs,ha_nh,xerr=ha_obs_err,yerr=ha_nh_err,fmt='o',alpha=0.8,color=nhcolor)

	ax.set_xlabel(r'log(H$_{\alpha}$) [observed luminosity, L$_{\odot}$]')
	ax.set_ylabel(r'log(H$_{\alpha}$) [model luminosity, L$_{\odot}$]')

	off,scat = threed_dutils.offset_and_scatter(ha_obs,ha_h,biweight=True)
	ax.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
			  transform = ax.transAxes,horizontalalignment='right',color=hcolor)
	ax.text(0.96,0.10, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = ax.transAxes,horizontalalignment='right',color=hcolor)
	off,scat = threed_dutils.offset_and_scatter(ha_obs,ha_nh,biweight=True)
	ax.text(0.96,0.15, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
			  transform = ax.transAxes,horizontalalignment='right',color=nhcolor)
	ax.text(0.96,0.20, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = ax.transAxes,horizontalalignment='right',color=nhcolor)

	ax.text(0.04,0.95, 'Herschel', transform = ax.transAxes,ha='left',color=hcolor)
	ax.text(0.04,0.9, 'No Herschel',transform = ax.transAxes,ha='left',color=nhcolor)

	lo = 5.5
	hi = 9.0
	ax.axis((lo,hi,lo,hi))
	ax.plot([lo,hi],[lo,hi],':',color='0.5',alpha=0.75)

	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(5))

	plt.savefig(outfolder+'herschel_halpha.png',dpi=150)
	plt.close()

def plot_bdec(alldata,outfolder=None):

	hcolor = '#FF3D0D' #red
	nhcolor = '#1C86EE' #blue

	#### observed balmer decrement
	ha_lum = np.array([dat['obs']['ha'] for dat in alldata])
	ha_lum_up = np.array([dat['obs']['ha_up'] for dat in alldata])
	ha_lum_do = np.array([dat['obs']['ha_down'] for dat in alldata])
	ha_lum_err = (ha_lum_up - ha_lum_do)/2.

	hb_lum = np.array([dat['obs']['hb'] for dat in alldata])
	hb_lum_up = np.array([dat['obs']['hb_up'] for dat in alldata])
	hb_lum_do = np.array([dat['obs']['hb_down'] for dat in alldata])
	hb_lum_err = (hb_lum_up - hb_lum_do)/2.

	bdec = ha_lum / hb_lum
	bdec_err = np.abs(bdec) * np.sqrt((ha_lum_err/ha_lum)**2+(hb_lum_err/hb_lum)**2)

	#### model balmer decrement
	nohersch_bdec = bdec_to_ext(np.squeeze([dat['nohersch']['bdec_q50'] for dat in alldata]))
	nohersch_bdec_up = bdec_to_ext(np.squeeze([dat['nohersch']['bdec_q84'] for dat in alldata]))
	nohersch_bdec_do = bdec_to_ext(np.squeeze([dat['nohersch']['bdec_q16'] for dat in alldata]))

	hersch_bdec = bdec_to_ext(np.squeeze([dat['hersch']['bdec_q50'] for dat in alldata]))
	hersch_bdec_up = bdec_to_ext(np.squeeze([dat['hersch']['bdec_q84'] for dat in alldata]))
	hersch_bdec_do = bdec_to_ext(np.squeeze([dat['hersch']['bdec_q16'] for dat in alldata]))

	#### observed halpha cuts
	sn_ha = ha_lum / ha_lum_err
	sn_hb = hb_lum / hb_lum_err
	idx = (ha_lum > 0.0) & (sn_ha > 3.0) & (hb_lum > 0.0) & (sn_hb > 3.0)

	#### generate plot quantities
	bdec_obs_err = threed_dutils.asym_errors(bdec_to_ext(bdec[idx]),
		                                     bdec_to_ext(bdec[idx]+bdec_err[idx]),
		                                     bdec_to_ext(bdec[idx]-bdec_err[idx]),log=False)
	bdec_obs = bdec_to_ext(bdec[idx])
	bdec_nh_err = threed_dutils.asym_errors(nohersch_bdec[idx],nohersch_bdec_up[idx],nohersch_bdec_do[idx],log=False)
	bdec_nh = nohersch_bdec[idx]
	bdec_h_err = threed_dutils.asym_errors(hersch_bdec[idx],hersch_bdec_up[idx],hersch_bdec_do[idx],log=False)
	bdec_h = hersch_bdec[idx]

	### plot!
	fig, ax = plt.subplots(1, 1, figsize = (7,7))

	ax.errorbar(bdec_obs,bdec_h,xerr=bdec_obs_err,yerr=bdec_h_err,fmt='o',alpha=0.8,color=hcolor)
	ax.errorbar(bdec_obs,bdec_nh,xerr=bdec_obs_err,yerr=bdec_nh_err,fmt='o',alpha=0.8,color=nhcolor)

	ax.set_xlabel(r'observed A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$')
	ax.set_ylabel(r'model A$_{\mathrm{H}\beta}$ - A$_{\mathrm{H}\alpha}$')

	off,scat = threed_dutils.offset_and_scatter(bdec_obs,bdec_h,biweight=True)
	ax.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat),
			  transform = ax.transAxes,horizontalalignment='right',color=hcolor)
	ax.text(0.96,0.10, 'mean offset='+"{:.2f}".format(off),
			      transform = ax.transAxes,horizontalalignment='right',color=hcolor)
	off,scat = threed_dutils.offset_and_scatter(bdec_obs,bdec_nh,biweight=True)
	ax.text(0.96,0.15, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
			  transform = ax.transAxes,horizontalalignment='right',color=nhcolor)
	ax.text(0.96,0.20, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = ax.transAxes,horizontalalignment='right',color=nhcolor)
	ax = threed_dutils.equalize_axes(ax, bdec_obs,bdec_h)

	ax.text(0.04,0.95, 'Herschel', transform = ax.transAxes,ha='left',color=hcolor)
	ax.text(0.04,0.9, 'No Herschel',transform = ax.transAxes,ha='left',color=nhcolor)

	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(5))

	plt.savefig(outfolder+'herschel_bdec.png',dpi=150)
	plt.close()

def plot_dustpars(alldata,outfolder=None):

	#### REGULAR PARAMETERS
	pars = np.array(alldata[0]['nohersch']['parnames'])
	to_plot = ['dust1','dust2','dust_index']
	parname = ['birth-cloud dust', 'diffuse dust', 'diffuse dust index']
	fig, axes = plt.subplots(2, 2, figsize = (12,12))
	plt.subplots_adjust(wspace=0.3,hspace=0.3)
	ax = np.ravel(axes)
	for ii,par in enumerate(to_plot):

		#### fit parameter, herschel
		idx = par == pars
		y = np.array([dat['hersch']['q50'][idx] for dat in alldata])
		yup = np.array([dat['hersch']['q84'][idx] for dat in alldata])
		ydown = np.array([dat['hersch']['q16'][idx] for dat in alldata])
		yerr = threed_dutils.asym_errors(y,yup,ydown,log=False)

		#### fit parameter, no herschel
		idx = par == pars
		x = np.array([dat['nohersch']['q50'][idx] for dat in alldata])
		xup = np.array([dat['nohersch']['q84'][idx] for dat in alldata])
		xdown = np.array([dat['nohersch']['q16'][idx] for dat in alldata])
		xerr = threed_dutils.asym_errors(x,xup,xdown,log=False)

		ax[ii].errorbar(x,y,yerr=yerr,xerr=xerr,fmt='o',alpha=0.8,color='#1C86EE')
		ax[ii].set_xlabel(parname[ii]+' [no Herschel]')
		ax[ii].set_ylabel(parname[ii]+' [with Herschel]')

		ax[ii] = threed_dutils.equalize_axes(ax[ii], x,y)
		mean_offset,scat = threed_dutils.offset_and_scatter(x,y)
		ax[ii].text(0.96,0.12, 'scatter='+"{:.2f}".format(scat),transform = ax[ii].transAxes,ha='right')
		ax[ii].text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset), transform = ax[ii].transAxes,ha='right')

		ax[ii].xaxis.set_major_locator(MaxNLocator(5))
		ax[ii].yaxis.set_major_locator(MaxNLocator(5))

	### halpha extinction!
	ha_ext_h, ha_ext_up_h, ha_ext_do_h = 1./np.array([dat['hersch']['ha_ext_q50'] for dat in alldata]), \
										 1./np.array([dat['hersch']['ha_ext_q84'] for dat in alldata]), \
										 1./np.array([dat['hersch']['ha_ext_q16'] for dat in alldata])
	ha_ext_h_errs = threed_dutils.asym_errors(ha_ext_h,ha_ext_up_h,ha_ext_do_h,log=False)

	ha_ext_nh, ha_ext_up_nh, ha_ext_do_nh = 1./np.array([dat['nohersch']['ha_ext_q50'] for dat in alldata]), \
										    1./np.array([dat['nohersch']['ha_ext_q84'] for dat in alldata]), \
										    1./np.array([dat['nohersch']['ha_ext_q16'] for dat in alldata])
	ha_ext_nh_errs = threed_dutils.asym_errors(ha_ext_nh,ha_ext_up_nh,ha_ext_do_nh,log=False)

	ax[ii+1].errorbar(ha_ext_nh,ha_ext_h,xerr=ha_ext_nh_errs,yerr=ha_ext_h_errs,fmt='o',alpha=0.8,color='#1C86EE')
	ax[ii+1].set_xlabel(r'total dust attenuation [6563 $\AA$, no Herschel]')
	ax[ii+1].set_ylabel(r'total dust attenuation [6563 $\AA$, with Herschel]')

	ax[ii+1] = threed_dutils.equalize_axes(ax[ii+1], ha_ext_nh,ha_ext_h)
	mean_offset,scat = threed_dutils.offset_and_scatter(ha_ext_nh,ha_ext_h)
	ax[ii+1].text(0.96,0.12, 'scatter='+"{:.2f}".format(scat),transform = ax[ii+1].transAxes,ha='right')
	ax[ii+1].text(0.96,0.05, 'mean offset='+"{:.2f}".format(mean_offset), transform = ax[ii+1].transAxes,ha='right')

	ax[ii+1].xaxis.set_major_locator(MaxNLocator(5))
	ax[ii+1].yaxis.set_major_locator(MaxNLocator(5))

	plt.savefig(outfolder+'dust_parameters.png',dpi=150)
	plt.close()