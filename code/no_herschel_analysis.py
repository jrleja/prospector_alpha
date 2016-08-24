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
import matplotlib as mpl
import math
import matplotlib.cm as cmx
import matplotlib.colors as colors

dpi = 120

hcolor = '#FF3D0D' #red
nhcolor = '#1C86EE' #blue

global_fs = 12
color = '0.4'


class jLogFormatter(mpl.ticker.LogFormatter):
	'''
	this changes the format from exponential to floating point.
	'''

	def __call__(self, x, pos=None):
		"""Return the format for tick val *x* at position *pos*"""
		vmin, vmax = self.axis.get_view_interval()
		d = abs(vmax - vmin)
		b = self._base
		if x == 0.0:
			return '0'
		sign = np.sign(x)
		# only label the decades
		fx = math.log(abs(x)) / math.log(b)
		isDecade = mpl.ticker.is_close_to_int(fx)
		if not isDecade and self.labelOnlyBase:
			s = ''
		elif x > 10000:
			s = '{0:.3g}'.format(x)
		elif x < 1:
			s = '{0:.3g}'.format(x)
		else:
			s = self.pprint_val(x, d)
		if sign == -1:
			s = '-%s' % s
		return self.fix_minus(s)

#### format those log plots! 
minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

def get_cmap(N):
    '''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
    RGB color.'''
    color_norm  = colors.Normalize(vmin=0, vmax=N-1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='hsv') 
    def map_index_to_rgb_color(index):
        return scalar_map.to_rgba(index)
    return map_index_to_rgb_color

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


	fig, axes = plt.subplots(2, 2, figsize = (10,10))
	ax = np.ravel(axes)
	plot_hfluxes(alldata,outfolder,ax=ax[0])
	plot_lir(alldata,ax=ax[1],ax2=ax[2])
	plot_totalext(alldata,ax=ax[3])
	fig.tight_layout()
	fig.savefig(outfolder+'herschel_comparison.png',dpi=dpi)
	plt.close()

	plot_dustpars(alldata,outfolder=outfolder)

	# if we're not doing this to mocks...
	if 'mock' not in runname_h:
		plot_halpha(alldata,outfolder=outfolder)
		plot_bdec(alldata,outfolder=outfolder)

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
	
	# if we're doing mocks...
	try:
		alldata = brown_io.load_alldata(runname=runname_h) # for observed Halpha
		mock_flag = False
	except:
		mock_flag = True
	objname = np.array([base.split('_')[-1] for base in filebase_nh])

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

		### save Herschel fluxes
		mask = sample_results_h['obs']['phot_mask']
		filtnames = np.array(sample_results_h['obs']['filternames'])[mask]

		# hersch_idx = np.array([True if 'herschel' in filter_name else False for filter_name in filtnames],dtype=bool)
		outdat_h['filtnames'] = filtnames
		outdat_h['wave_effective'] = sample_results_h['obs']['wave_effective'][mask]
		outdat_h['model_fluxes'] = sample_results_h['bfit']['mags'][mask]
		outdat_h['obs_fluxes'] = sample_results_h['obs']['maggies'][mask]
		outdat_h['obs_errs'] = sample_results_h['obs']['maggies_unc'][mask]

		outdat_nh['model_fluxes'] = sample_results_nh['bfit']['mags'][mask]

		### save model Balmer decrement & total extinction
		epars = sample_results_nh['extras']['parnames']
		bdec_idx = epars == 'bdec_cloudy'
		text_idx = epars == 'total_ext5500'
		outdat_nh['bdec_q50'] = sample_results_nh['extras']['q50'][bdec_idx]
		outdat_nh['bdec_q84'] = sample_results_nh['extras']['q84'][bdec_idx]
		outdat_nh['bdec_q16'] = sample_results_nh['extras']['q16'][bdec_idx]

		outdat_nh['text_q50'] = sample_results_nh['extras']['q50'][text_idx]
		outdat_nh['text_q84'] = sample_results_nh['extras']['q84'][text_idx]
		outdat_nh['text_q16'] = sample_results_nh['extras']['q16'][text_idx]

		epars = sample_results_h['extras']['parnames']
		bdec_idx = epars == 'bdec_cloudy'
		text_idx = epars == 'total_ext5500'
		outdat_h['bdec_q50'] = sample_results_h['extras']['q50'][bdec_idx]
		outdat_h['bdec_q84'] = sample_results_h['extras']['q84'][bdec_idx]
		outdat_h['bdec_q16'] = sample_results_h['extras']['q16'][bdec_idx]

		outdat_h['text_q50'] = sample_results_h['extras']['q50'][text_idx]
		outdat_h['text_q84'] = sample_results_h['extras']['q84'][text_idx]
		outdat_h['text_q16'] = sample_results_h['extras']['q16'][text_idx]

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

		if mock_flag is False:
			try: # somehow, the emission line information for ONE GALAXY got deleted? what the actual fuck.
				emline_names = alldata[match]['residuals']['emlines']['em_name']
				idx = emline_names == 'H$\\alpha$'
				obs['ha'] = alldata[match]['residuals']['emlines']['obs']['lum'][idx][0] / constants.L_sun.cgs.value
				obs['ha_up'] = alldata[match]['residuals']['emlines']['obs']['lum_errup'][idx][0] / constants.L_sun.cgs.value
				obs['ha_down'] = alldata[match]['residuals']['emlines']['obs']['lum_errdown'][idx][0] / constants.L_sun.cgs.value

				idx = emline_names == 'H$\\beta$'
				obs['hb'] = alldata[match]['residuals']['emlines']['obs']['lum'][idx][0] / constants.L_sun.cgs.value
				obs['hb_up'] = alldata[match]['residuals']['emlines']['obs']['lum_errup'][idx][0] / constants.L_sun.cgs.value
				obs['hb_down'] = alldata[match]['residuals']['emlines']['obs']['lum_errdown'][idx][0] / constants.L_sun.cgs.value
			except TypeError:
				pass

		### save both L_IRs
		outdat_h['lir'] = sample_results_h['observables']['L_IR']
		outdat_nh['lir'] = sample_results_nh['observables']['L_IR']

		outtemp = {'nohersch':outdat_nh,'hersch':outdat_h,'obs':obs}
		out.append(outtemp)
		
	pickle.dump(out,open(outpickle, "wb"))
	return out

def plot_hfluxes(alldata,outfolder,ax=None):

	##### fluxes
	fnames_all = alldata[0]['hersch']['filtnames'] # this has all of them, nothing is masked!
	nfilter = fnames_all.shape[0]
	nh_flux, h_flux, obs_flux,obs_errs = [np.zeros(shape=(nfilter,len(alldata)))+np.nan for i in xrange(4)]
	for ii, dat in enumerate(alldata):
		fname = dat['hersch']['filtnames']
		for kk, name in enumerate(fname):
			match = name == fnames_all
			if match.sum() != 0:
				nh_flux[match,ii] = dat['nohersch']['model_fluxes'][kk]
				h_flux[match,ii] = dat['hersch']['model_fluxes'][kk]
				obs_flux[match,ii] = dat['hersch']['obs_fluxes'][kk]
				obs_errs[match,ii] = dat['hersch']['obs_errs'][kk]

	#### offsets and 1 sigma
	flux_offset_h, flux_offset_nh = [np.zeros(shape=(nfilter,3)) for i in xrange(2)]
	chi_h, chi_nh = [np.zeros(shape=(nfilter,3)) for i in xrange(2)]
	chi_all_h, chi_all_nh = [], []
	for nn in xrange(nfilter):
		idx = np.isfinite(h_flux[nn,:]) # only include galaxies if they're observed in that band
		flux_offset_h[nn,:] = quantile(np.log10(h_flux[nn,idx]/obs_flux[nn,idx]),[0.5,0.84,0.16])
		flux_offset_nh[nn,:] = quantile(np.log10(nh_flux[nn,idx]/obs_flux[nn,idx]),[0.5,0.84,0.16])

		chi_h_temp = (h_flux[nn,idx]-obs_flux[nn,idx])/obs_errs[nn,idx]
		chi_nh_temp = (nh_flux[nn,idx]-obs_flux[nn,idx])/obs_errs[nn,idx]
		chi_h[nn,:] = quantile(chi_h_temp,[0.5,0.84,0.16])
		chi_nh[nn,:] = quantile(chi_nh_temp,[0.5,0.84,0.16])
		chi_all_h.append(chi_h_temp)
		chi_all_nh.append(chi_nh_temp)

	h_flux_err = threed_dutils.asym_errors(flux_offset_h[:,0],
		                              flux_offset_h[:,1],
		                              flux_offset_h[:,2],log=False)

	nh_flux_err = threed_dutils.asym_errors(flux_offset_nh[:,0],
		                               flux_offset_nh[:,1],
		                               flux_offset_nh[:,2],log=False)

	h_chi_err = threed_dutils.asym_errors(chi_h[:,0],
		                              chi_h[:,1],
		                              chi_h[:,2],log=False)

	nh_chi_err = threed_dutils.asym_errors(chi_nh[:,0],
		                               chi_nh[:,1],
		                               chi_nh[:,2],log=False)

	##### wavelength array
	wave_effective = alldata[0]['hersch']['wave_effective']/1e4 # again, this has all of them!

	### LIR
	ax.errorbar(wave_effective,chi_h[:,0],yerr=h_chi_err,fmt='o',alpha=0.8,color=hcolor)
	ax.errorbar(wave_effective,chi_nh[:,0],yerr=nh_chi_err,fmt='o',alpha=0.8,color=nhcolor)

	ax.set_xlabel(r'observed wavelength [$\mu$m]')
	ax.set_ylabel(r'$\chi$')

	ax.set_xscale('log',nonposx='clip',subsx=(np.atleast_1d(1)))
	ax.xaxis.set_minor_formatter(minorFormatter)
	ax.xaxis.set_major_formatter(majorFormatter)

	### add line
	ax.set_xlim(0.1,800)
	ax.set_ylim(-2.5,2.5)
	ax.plot(ax.get_xlim(),[0.0,0.0],linestyle='--',lw=2,color='black')

	### add text
	fs = 14
	ax.text(0.05,0.92,'fit to Herschel data',color=hcolor,transform=ax.transAxes,fontsize=fs)
	ax.text(0.05,0.86,'fit without Herschel data',color=nhcolor,transform=ax.transAxes,fontsize=fs)

	### UV-MIR CHISQ
	idx = np.array([False if 'herschel' in filter_name else True for filter_name in fnames_all],dtype=bool)
	chisq_h = (chi_h[idx,0]**2).sum()
	chisq_nh = (chi_nh[idx,0]**2).sum()
	ax.text(0.05,0.12,r'UV-MIR $\Sigma \chi^2$='+"{:.2f}".format(chisq_h),color=hcolor,transform=ax.transAxes,fontsize=fs)
	ax.text(0.05,0.06,r'UV-MIR $\Sigma \chi^2$='+"{:.2f}".format(chisq_nh),color=nhcolor,transform=ax.transAxes,fontsize=fs)


	#### DELTA CHI SQUARED PLOT
	fig, ax1 = plt.subplots(1, 1, figsize = (8,8))
	cmap = get_cmap(idx.sum())
	xt, yt = 0.03, 0.95
	for nn in xrange(idx.sum()):

		chisq_nh = chi_all_nh[nn]**2
		chisq_h = chi_all_h[nn]**2
		delta_chisq = chisq_nh-chisq_h
		ax1.plot(np.repeat(wave_effective[nn],delta_chisq.shape[0]),delta_chisq,'o',linestyle=' ',color=cmap(nn),alpha=0.2,mew=0)
		ax1.plot(wave_effective[nn], np.median(delta_chisq),'o',linestyle=' ',color=cmap(nn),alpha=0.8,mew=2,ms=16)

		ax1.text(xt,yt,fnames_all[nn],transform=ax1.transAxes,color=cmap(nn),fontsize=10,weight='bold')

		# text positioning
		if (nn+1) % 8 == 0:
			xt += 0.2
			yt = 0.95
		else:
			yt -= 0.03

	ax1.plot(ax1.get_xlim(),[0.0,0.0],linestyle='--',color='0.6',lw=2,alpha=0.5)

	ax1.set_xlabel(r'observed wavelength [$\mu$m]',labelpad=8)
	ax1.set_ylabel(r'$\chi^2_{\mathrm{no Herschel}}$-$\chi^2_{\mathrm{Herschel}}$')

	ax1.set_xscale('log',nonposx='clip',subsx=(1,2,4))
	ax1.xaxis.set_minor_formatter(minorFormatter)
	ax1.xaxis.set_major_formatter(majorFormatter)

	ax1.set_xlim(0.1,30)
	ax1.set_ylim(-2,2)

	fig.savefig(outfolder+'delta_chisq.png',dpi=150)

	'''
	rms_h = np.sqrt((flux_offset_h[idx,0]**2).sum())
	rms_nh = np.sqrt((flux_offset_nh[idx,0]**2).sum())
	ax.text(0.05,0.12,'UV-MIR RMS '+"{:.3f}".format(rms_h)+' dex',color=hcolor,transform=ax.transAxes,fontsize=fs)
	ax.text(0.05,0.06,'UV-MIR RMS '+"{:.3f}".format(rms_nh)+' dex',color=nhcolor,transform=ax.transAxes,fontsize=fs)
	'''

def plot_totalext(alldata,ax=None):

	'''
	LIR measurements are FUCKED UP
	fix before this plot becomes useful
	'''


	##### total extinction
	hersch_totalext = np.array([dat['hersch']['text_q50'] for dat in alldata])
	hersch_totalext_errup = np.array([dat['hersch']['text_q84'] for dat in alldata])
	hersch_totalext_errdo = np.array([dat['hersch']['text_q16'] for dat in alldata])
	hersch_totalext_err = threed_dutils.asym_errors(hersch_totalext,
		                                            hersch_totalext_errup,
		                                            hersch_totalext_errdo,log=False)

	nohersch_totalext = np.array([dat['nohersch']['text_q50'] for dat in alldata])
	nohersch_totalext_errup = np.array([dat['nohersch']['text_q84'] for dat in alldata])
	nohersch_totalext_errdo = np.array([dat['nohersch']['text_q16'] for dat in alldata])
	nohersch_totalext_err = threed_dutils.asym_errors(nohersch_totalext,
		                                              nohersch_totalext_errup,
		                                              nohersch_totalext_errdo,log=False)

	ax.errorbar(hersch_totalext,nohersch_totalext,xerr=hersch_totalext_err,yerr=nohersch_totalext_err,fmt='o',alpha=0.8,color=color)
	ax.set_xlabel(r'total 5500 $\AA$ optical depth [fit with Herschel]')
	ax.set_ylabel(r'total 5500 $\AA$ optical depth [no Herschel]')

	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(5))

	off,scat = threed_dutils.offset_and_scatter(hersch_totalext,nohersch_totalext,biweight=True)
	ax.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
			  transform = ax.transAxes,horizontalalignment='right',fontsize=global_fs)
	ax.text(0.96,0.10, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = ax.transAxes,horizontalalignment='right',fontsize=global_fs)
	ax = threed_dutils.equalize_axes(ax, hersch_totalext,nohersch_totalext)

def plot_lir(alldata,ax=None,ax2=None):

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

	### LIR
	ax.errorbar(hersch_lir,nohersch_lir,xerr=hersch_lir_err,yerr=nohersch_lir_err,fmt='o',alpha=0.8,color=color)
	ax.set_xlabel('log(model L$_{\mathrm{IR}}$) [fit with Herschel]')
	ax.set_ylabel('log(model L$_{\mathrm{IR}}$) [no Herschel]')

	ax.xaxis.set_major_locator(MaxNLocator(5))
	ax.yaxis.set_major_locator(MaxNLocator(5))

	off,scat = threed_dutils.offset_and_scatter(hersch_lir,nohersch_lir,biweight=True)
	ax.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
			  transform = ax.transAxes,horizontalalignment='right',fontsize=global_fs)
	ax.text(0.96,0.10, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = ax.transAxes,horizontalalignment='right',fontsize=global_fs)
	ax = threed_dutils.equalize_axes(ax, hersch_lir,nohersch_lir)

	### SFR 100
	ax2.errorbar(h_sfr100,nh_sfr100,xerr=h_sfr100_err,yerr=nh_sfr100_err,fmt='o',alpha=0.8,color=color)
	ax2.set_xlabel('log(SFR$_{100 \mathrm{Myr}}$) [fit with Herschel]')
	ax2.set_ylabel('log(SFR$_{100 \mathrm{Myr}}$) [no Herschel]')

	ax2.xaxis.set_major_locator(MaxNLocator(5))
	ax2.yaxis.set_major_locator(MaxNLocator(5))

	off,scat = threed_dutils.offset_and_scatter(h_sfr100,nh_sfr100,biweight=True)
	ax2.text(0.96,0.05, 'biweight scatter='+"{:.2f}".format(scat)+ ' dex',
			  transform = ax2.transAxes,horizontalalignment='right',fontsize=global_fs)
	ax2.text(0.96,0.10, 'mean offset='+"{:.2f}".format(off) + ' dex',
			      transform = ax2.transAxes,horizontalalignment='right',fontsize=global_fs)
	ax2 = threed_dutils.equalize_axes(ax2, h_sfr100,nh_sfr100)

def plot_halpha(alldata,outfolder=None):

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