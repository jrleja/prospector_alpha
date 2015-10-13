import numpy as np
import matplotlib.pyplot as plt
from magphys import read_magphys_output
import os, copy, threed_dutils
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import pickle, math, measure_emline_lum
import magphys_plot_pref
import mag_ensemble
import matplotlib as mpl
from astropy import constants

c = 3e18   # angstroms per second

#### set up colors and plot style
prosp_color = '#e60000'
obs_color = '#95918C'
magphys_color = '#1974D2'

#### where do the pickle files go?
outpickle = '/Users/joel/code/magphys/data/pickles'

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

def median_by_band(x,y,avg=False):

	##### get filter effective wavelengths for sorting
	delz = 0.06
	from brownseds_params import translate_filters
	from translate_filter import calc_lameff_for_fsps
	filtnames = np.array(translate_filters(0,full_list=True))
	wave_effective = calc_lameff_for_fsps(filtnames[filtnames != 'nan'])/1e4
	wave_effective.sort()

	avglam = np.array([])
	outval = np.array([])
	for lam in wave_effective:
		in_bounds = (x < lam) & (x > lam/(1+delz))
		avglam = np.append(avglam, np.mean(x[in_bounds]))
		if avg == False:
			outval = np.append(outval, np.median(y[in_bounds]))
		else:
			outval = np.append(outval, np.mean(y[in_bounds]))

	return avglam, outval

def plot_all_residuals(alldata):

	'''
	show all residuals for spectra + photometry, magphys + prospectr
	'''

	##### set up plots
	fig = plt.figure(figsize=(15,12.5))
	mpl.rcParams.update({'font.size': 13})
	gs1 = mpl.gridspec.GridSpec(4, 1)
	gs1.update(top=0.95, bottom=0.05, left=0.09, right=0.75,hspace=0.22)
	phot = plt.Subplot(fig, gs1[0])
	opt = plt.Subplot(fig, gs1[1])
	akar = plt.Subplot(fig, gs1[2])
	spit = plt.Subplot(fig,gs1[3])
	
	gs2 = mpl.gridspec.GridSpec(4, 1)
	gs2.update(top=0.95, bottom=0.05, left=0.8, right=0.97,hspace=0.22)
	phot_hist = plt.Subplot(fig, gs2[0])
	opt_hist = plt.Subplot(fig, gs2[1])
	akar_hist = plt.Subplot(fig, gs2[2])
	spit_hist = plt.Subplot(fig,gs2[3])
	
	
	##### add plots
	plots = [opt,akar,spit]
	plots_hist = [opt_hist, akar_hist, spit_hist]
	for plot in plots: fig.add_subplot(plot)
	for plot in plots_hist: fig.add_subplot(plot)
	fig.add_subplot(phot)
	fig.add_subplot(phot_hist)

	#### parameters
	alpha_minor = 0.2
	lw_minor = 0.5
	alpha_major = 0.8
	lw_major = 2.5

	##### load and plot photometric residuals
	chi_magphys, chi_prosp, chisq_magphys,chisq_prosp, lam_rest = np.array([]),np.array([]),np.array([]), np.array([]), np.array([])
	for data in alldata:

		if data:
			chi_magphys = np.append(chi_magphys,data['residuals']['phot']['chi_magphys'])
			chi_prosp = np.append(chi_prosp,data['residuals']['phot']['chi_prosp'])
			chisq_magphys = np.append(chisq_magphys,data['residuals']['phot']['chisq_magphys'])
			chisq_prosp = np.append(chisq_prosp,data['residuals']['phot']['chisq_prosp'])
			lam_rest = np.append(lam_rest,data['residuals']['phot']['lam_obs']/(1+data['residuals']['phot']['z'])/1e4)

			phot.plot(data['residuals']['phot']['lam_obs']/(1+data['residuals']['phot']['z'])/1e4, 
				      data['residuals']['phot']['chi_magphys'],
				      alpha=alpha_minor,
				      color=magphys_color,
				      lw=lw_minor
				      )

			phot.plot(data['residuals']['phot']['lam_obs']/(1+data['residuals']['phot']['z'])/1e4, 
				      data['residuals']['phot']['chi_prosp'],
				      alpha=alpha_minor,
				      color=prosp_color,
				      lw=lw_minor
				      )

	##### calculate and plot running median
	nfilters = 33 # calculate this more intelligently?
	magbins, magmedian = median_by_band(lam_rest,chi_magphys)
	probins, promedian = median_by_band(lam_rest,chi_prosp)

	phot.plot(magbins, 
		      magmedian,
		      color='black',
		      lw=lw_major*1.1
		      )

	phot.plot(magbins, 
		      magmedian,
		      color=magphys_color,
		      lw=lw_major
		      )

	phot.plot(probins, 
		      promedian,
		      color='black',
		      lw=lw_major*1.1
		      )
	phot.plot(probins, 
		      promedian,
		      alpha=alpha_major,
		      color=prosp_color,
		      lw=lw_major
		      )
	phot.text(0.99,0.92, 'MAGPHYS',
			  transform = phot.transAxes,horizontalalignment='right',
			  color=magphys_color)
	phot.text(0.99,0.85, 'Prospectr',
			  transform = phot.transAxes,horizontalalignment='right',
			  color=prosp_color)
	phot.text(0.99,0.05, 'photometry',
			  transform = phot.transAxes,horizontalalignment='right')
	phot.set_xlabel(r'$\lambda_{\mathrm{rest}}$ [$\mu$m]')
	phot.set_ylabel(r'$\chi$')
	phot.axhline(0, linestyle=':', color='grey')
	phot.set_xscale('log',nonposx='clip',subsx=(2,4,7))
	phot.xaxis.set_minor_formatter(minorFormatter)
	phot.xaxis.set_major_formatter(majorFormatter)
	phot.set_xlim(0.12,600)

	##### histogram of chisq values
	nbins = 10
	alpha_hist = 0.3
	# first call is color-less, to get bins
	# suitable for both data sets
	histmax = 5
	okmag = chisq_magphys < histmax
	okpro = chisq_prosp < histmax
	n, b, p = phot_hist.hist([chisq_magphys[okmag],chisq_prosp[okpro]],
		                 nbins, histtype='bar',
		                 color=[magphys_color,prosp_color],
		                 alpha=0.0,lw=2)
	n, b, p = phot_hist.hist(chisq_magphys[okmag],
		                 bins=b, histtype='bar',
		                 color=magphys_color,
		                 alpha=alpha_hist,lw=2)
	n, b, p = phot_hist.hist(chisq_prosp[okpro],
		                 bins=b, histtype='bar',
		                 color=prosp_color,
		                 alpha=alpha_hist,lw=2)

	phot_hist.set_ylabel('N')
	phot_hist.xaxis.set_major_locator(MaxNLocator(5))

	phot_hist.set_xlabel(r'$\chi^2_{\mathrm{phot}}/$N$_{\mathrm{phot}}$')

	##### load and plot spectroscopic residuals
	label = ['Optical','Akari', 'Spitzer IRS']
	nbins = [500,50,50]
	for i, plot in enumerate(plots):
		res_magphys, res_prosp, obs_restlam, rms_mag, rms_pro = np.array([]),np.array([]),np.array([]),np.array([]),np.array([])
		for data in alldata:
			if data:
				if data['residuals'][label[i]]:
					res_magphys = np.append(res_magphys,data['residuals'][label[i]]['magphys_resid'])
					res_prosp = np.append(res_prosp,data['residuals'][label[i]]['prospectr_resid'])
					obs_restlam = np.append(obs_restlam,data['residuals'][label[i]]['obs_restlam'])

					plot.plot(data['residuals'][label[i]]['obs_restlam'], 
						      data['residuals'][label[i]]['magphys_resid'],
						      alpha=alpha_minor,
						      color=magphys_color,
						      lw=lw_minor
						      )

					plot.plot(data['residuals'][label[i]]['obs_restlam'], 
						      data['residuals'][label[i]]['prospectr_resid'],
						      alpha=alpha_minor,
						      color=prosp_color,
						      lw=lw_minor
						      )

					rms_mag = np.append(rms_mag,data['residuals'][label[i]]['magphys_rms'])
					rms_pro = np.append(rms_pro,data['residuals'][label[i]]['prospectr_rms'])

		##### calculate and plot running median
		magbins, magmedian = threed_dutils.running_median(obs_restlam,res_magphys,nbins=nbins[i])
		probins, promedian = threed_dutils.running_median(obs_restlam,res_prosp,nbins=nbins[i])

		plot.plot(magbins, 
			      magmedian,
			      color='black',
			      lw=lw_major*1.1
			      )

		plot.plot(magbins, 
			      magmedian,
			      color=magphys_color,
			      lw=lw_major
			      )

		plot.plot(probins, 
			      promedian,
			      color='black',
			      lw=lw_major
			      )

		plot.plot(probins, 
			      promedian,
			      color=prosp_color,
			      lw=lw_major
			      )

		plot.set_xscale('log',nonposx='clip',subsx=(1,2,3,4,5,6,7,8,9))
		plot.xaxis.set_minor_formatter(minorFormatter)
		plot.xaxis.set_major_formatter(majorFormatter)
		plot.set_xlabel(r'$\lambda_{\mathrm{rest}}$ [$\mu$m]')
		plot.set_ylabel(r'log(f$_{\mathrm{obs}}/$f$_{\mathrm{mod}}$)')
		plot.text(0.985,0.05, label[i],
			      transform = plot.transAxes,horizontalalignment='right')
		plot.axhline(0, linestyle=':', color='grey')

		##### histogram of mean offsets
		nbins_hist = 10
		alpha_hist = 0.3
		# first histogram is transparent, to get bins
		# suitable for both data sets
		histmax = 2
		okmag = rms_mag < histmax
		okpro = rms_pro < histmax
		n, b, p = plots_hist[i].hist([rms_mag[okmag],rms_pro[okpro]],
			                 nbins_hist, histtype='bar',
			                 color=[magphys_color,prosp_color],
			                 alpha=0.0,lw=2)
		n, b, p = plots_hist[i].hist(rms_mag[okmag],
			                 bins=b, histtype='bar',
			                 color=magphys_color,
			                 alpha=alpha_hist,lw=2)
		n, b, p = plots_hist[i].hist(rms_pro[okpro],
			                 bins=b, histtype='bar',
			                 color=prosp_color,
			                 alpha=alpha_hist,lw=2)

		plots_hist[i].set_ylabel('N')
		plots_hist[i].set_xlabel(r'RMS [dex]')
		plots_hist[i].xaxis.set_major_locator(MaxNLocator(5))

		plot.set_ylim(-1.2,1.2)
		dynx = (np.max(magbins) - np.min(magbins))*0.05
		plot.set_xlim(np.min(magbins)-dynx,np.max(magbins)+dynx)

	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/brownseds/magphys/'
	
	plt.savefig(outfolder+'median_residuals.png',dpi=300)
	plt.close()

def load_spectra(objname, nufnu=True):
	
	# flux is read in as ergs / s / cm^2 / Angstrom
	# the source key is:
	# 0 = model
	# 1 = optical spectrum
	# 2 = Akari
	# 3 = Spitzer IRS

	foldername = '/Users/joel/code/python/threedhst_bsfh/data/brownseds_data/spectra/'
	rest_lam, flux, obs_lam, source = np.loadtxt(foldername+objname.replace(' ','_')+'_spec.dat',comments='#',unpack=True)

	lsun = 3.846e33  # ergs/s
	flux_lsun = flux / lsun

	# convert to flam * lam
	flux = flux * obs_lam

	# convert to janskys, then maggies * Hz
	flux = flux * 1e23 / 3631

	out = {}
	out['rest_lam'] = rest_lam
	out['flux'] = flux
	out['flux_lsun'] = flux_lsun
	out['obs_lam'] = obs_lam
	out['source'] = source

	return out


def return_sedplot_vars(thetas, sample_results, sps, nufnu=True):

	'''
	if nufnu == True: return in units of nu * fnu (maggies * Hz). Else, return maggies.
	'''

	# observational information
	mask = sample_results['obs']['phot_mask']
	wave_eff = sample_results['obs']['wave_effective'][mask]
	obs_maggies = sample_results['obs']['maggies'][mask]
	obs_maggies_unc = sample_results['obs']['maggies_unc'][mask]

	# model information
	spec, mu ,_ = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)
	mu = mu[mask]

	# output units
	if nufnu == True:
		mu *= c/wave_eff
		spec *= c/sps.wavelengths
		obs_maggies *= c/wave_eff
		obs_maggies_unc *= c/wave_eff

	# here we want to return
	# effective wavelength of photometric bands, observed maggies, observed uncertainty, model maggies, observed_maggies-model_maggies / uncertainties
	# model maggies, observed_maggies-model_maggies/uncertainties
	return wave_eff, obs_maggies, obs_maggies_unc, mu, (obs_maggies-mu)/obs_maggies_unc, spec, sps.wavelengths

def mask_emission_lines(lam,z):

	# OII, Hbeta, OIII, Halpha, NII, SII
	lam_temp = lam*1e4
	mask_lines = np.array([3727, 4861, 4959, 5007, 6563, 6583,6720])*(1.0+z)
	mask_size = 20 # Angstroms
	mask = np.ones_like(lam,dtype=bool)

	for line in mask_lines: mask[(lam_temp > line - mask_size) & (lam_temp < line + mask_size)] = 0.0

	return mask

def calc_rms(lam, z, resid):


	mask = mask_emission_lines(lam,z)
	rms = (np.sum((resid[mask]-resid[mask].mean())**2)/np.sum(mask))**0.5

	return rms

def plot_obs_spec(obs_spec, phot, spec_res, alpha, 
	              modlam, modspec, maglam, magspec,z, 
	              objname, source, sigsmooth,
	              color='black',label=''):

	'''
	standard wrapper for plotting observed + residuals for spectra
	for individual galaxies
	'''

	mask = obs_spec['source'] == source
	obslam = obs_spec['obs_lam'][mask]/1e4
	if np.sum(mask) > 0:

		phot.plot(obslam, 
			      obs_spec['flux'][mask],
			      alpha=0.9,
			      color=color,
			      zorder=-32
			      )

		##### smooth 
		# consider only passing relevant wavelengths
		# if speed becomes an issue
		modspec_smooth = threed_dutils.smooth_spectrum(modlam/(1+z),
		                                        modspec,sigsmooth)
		magspec_smooth = threed_dutils.smooth_spectrum(maglam/(1+z),
		                                        magspec,sigsmooth)

		# interpolate fsps spectra onto observational wavelength grid
		pro_flux_interp = interp1d(modlam,
			                       modspec_smooth, 
			                       bounds_error = False, fill_value = 0)

		prospectr_resid = np.log10(obs_spec['flux'][mask]) - np.log10(pro_flux_interp(obslam))
		spec_res.plot(obslam, 
			          prospectr_resid,
			          color=prosp_color,
			          alpha=alpha,
			          linestyle='-')

		# interpolate magphys onto fsps grid
		mag_flux_interp = interp1d(maglam, magspec_smooth,
		                           bounds_error=False, fill_value=0)
		magphys_resid = np.log10(obs_spec['flux'][mask]) - np.log10(mag_flux_interp(obslam))
		nolines = mask_emission_lines(obslam,z)

		spec_res.plot(obslam[nolines], 
			          magphys_resid[nolines],
			          color=magphys_color,
			          alpha=alpha,
			          linestyle='-')

		#### calculate rms
		# mask emission lines
		magphys_rms = calc_rms(obslam, z, magphys_resid)
		prospectr_rms = calc_rms(obslam, z, prospectr_resid)

		#### write text, add lines
		spec_res.text(0.98,0.16, 'RMS='+"{:.2f}".format(prospectr_rms)+' dex',
			          transform = spec_res.transAxes,ha='right',
			          color=prosp_color,fontsize=14)
		spec_res.text(0.98,0.05, 'RMS='+"{:.2f}".format(magphys_rms)+' dex',
			          transform = spec_res.transAxes,ha='right',
			          color=magphys_color,fontsize=14)
		spec_res.text(0.015,0.05, label,
			          transform = spec_res.transAxes)
		spec_res.axhline(0, linestyle=':', color='grey')
		spec_res.set_xlim(min(obslam)*0.85,max(obslam)*1.15)
		if label == 'Optical':
			spec_res.set_ylim(-np.std(magphys_resid)*5,np.std(magphys_resid)*5)
			spec_res.set_xlim(min(obslam)*0.98,max(obslam)*1.02)

		#### axis scale
		spec_res.set_xscale('log',nonposx='clip', subsx=(1,2,3,4,5,6,7,8,9))
		spec_res.xaxis.set_minor_formatter(minorFormatter)
		spec_res.xaxis.set_major_formatter(majorFormatter)

		# output rest-frame wavelengths + residuals
		out = {
			   'obs_restlam': obslam/(1+z),
			   'magphys_resid': magphys_resid,
			   'magphys_rms': magphys_rms,
			   'obs_obslam': obslam,
			   'prospectr_resid': prospectr_resid,
			   'prospectr_rms': prospectr_rms
			   }

		return out

	else:

		# remove axis
		spec_res.axis('off')

def update_model_info(alldata, sample_results, magphys):

	alldata['objname'] = sample_results['run_params']['objname']
	alldata['magphys'] = magphys['pdfs']
	alldata['model'] = magphys['model']
	alldata['pquantiles'] = sample_results['quantiles']
	alldata['model_emline'] = sample_results['model_emline']
	alldata['lir'] = sample_results['observables']['L_IR']
	alldata['pextras'] = sample_results['extras']
	alldata['pquantiles']['parnames'] = np.array(sample_results['model'].theta_labels())
	alldata['bfit'] = sample_results['bfit']

	return alldata

def sed_comp_figure(sample_results, sps, model, magphys,
                alpha=0.3, samples = [-1],
                maxprob=0, outname=None, fast=False,
                truths = None, agb_off = False,
                **kwargs):
	"""
	Plot the photometry for the model and data (with error bars) for
	a single object, and plot residuals.

	Returns a dictionary called 'residuals', which contains the 
	photometric + spectroscopic residuals for this object, for both
	magphys and prospectr.
	"""


	#### set up plot
	fig = plt.figure(figsize=(12,12))
	gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3,1])
	gs.update(bottom=0.525, top=0.99, hspace=0.00)
	phot, res = plt.Subplot(fig, gs[0]), plt.Subplot(fig, gs[1])

	gs2 = mpl.gridspec.GridSpec(3, 1)
	gs2.update(top=0.475, bottom=0.05, hspace=0.15)
	spec_res_opt,spec_res_akari,spec_res_spit = plt.subplot(gs2[0]),plt.subplot(gs2[1]),plt.subplot(gs2[2])

	# R = 650, 300, 100
	# models have deltav = 1000 km/s in IR, add in quadrature?
	sigsmooth = [450.0, 1000.0, 3000.0]
	ms = 8
	alpha = 0.8

	#### setup output
	residuals={}

	##### Prospectr maximum probability model ######
	# plot the spectrum, photometry, and chi values
	try:
		wave_eff, obsmags, obsmags_unc, modmags, chi, modspec, modlam = \
		return_sedplot_vars(sample_results['bfit']['maxprob_params'], 
			                sample_results, sps)
	except KeyError as e:
		print e
		print "You must run post-processing on the Prospectr " + \
			  "data for " + sample_results['run_params']['objname']
		return None

	phot.plot(wave_eff/1e4, modmags, 
		      color=prosp_color, marker='o', ms=ms, 
		      linestyle=' ', label='Prospectr', alpha=alpha, 
		      markeredgewidth=0.7,**kwargs)
	
	res.plot(wave_eff/1e4, chi, 
		     color=prosp_color, marker='o', linestyle=' ', label='Prospectr', 
		     ms=ms,alpha=alpha,markeredgewidth=0.7,**kwargs)
	
	nz = modspec > 0
	phot.plot(modlam[nz]/1e4, modspec[nz], linestyle='-',
              color=prosp_color, alpha=0.6,**kwargs)

	##### photometric observations, errors ######
	xplot = wave_eff/1e4
	yplot = obsmags
	phot.errorbar(xplot, yplot, yerr=obsmags_unc,
                  color=obs_color, marker='o', label='observed', alpha=alpha, linestyle=' ',ms=ms)

	# plot limits
	phot.set_xlim(min(xplot)*0.4,max(xplot)*1.5)
	phot.set_ylim(min(yplot[np.isfinite(yplot)])*0.4,max(yplot[np.isfinite(yplot)])*2.3)
	res.set_xlim(min(xplot)*0.4,max(xplot)*1.5)
	res.axhline(0, linestyle=':', color='grey')

	##### magphys: spectrum + photometry #####
	# note: we steal the filter effective wavelengths from Prospectr here
	# if filters are mismatched in Prospectr vs MAGPHYS, this will do weird things
	# not fixing it, since it may serve as an "alarm bell"
	m = magphys['obs']['phot_mask']

	# comes out in maggies, change to maggies*Hz
	nu_eff = c / wave_eff
	spec_fac = c / magphys['model']['lam']

	try:
		phot.plot(wave_eff/1e4, 
			      magphys['model']['flux'][m]*nu_eff, 
			      color=magphys_color, marker='o', ms=ms, 
			      linestyle=' ', label='MAGPHYS', alpha=alpha, 
			      markeredgewidth=0.7,**kwargs)
	except:
		print sample_results['obs']['phot_mask']
		print magphys['obs']['phot_mask']
		print sample_results['run_params']['objname']
		print 'Mismatch between Prospectr and MAGPHYS photometry!'
		plt.close()
		print 1/0
		return None
	
	chi_magphys = (magphys['obs']['flux'][m]-magphys['model']['flux'][m])/magphys['obs']['flux_unc'][m]
	res.plot(wave_eff/1e4, 
		     chi_magphys, 
		     color=magphys_color, marker='o', linestyle=' ', label='MAGPHYS', 
		     ms=ms,alpha=alpha,markeredgewidth=0.7,**kwargs)
	
	nz = magphys['model']['spec'] > 0
	phot.plot(magphys['model']['lam'][nz]/1e4, 
		      magphys['model']['spec'][nz]*spec_fac, 
		      linestyle='-', color=magphys_color, alpha=0.6,
		      **kwargs)

	##### observed spectra + residuals #####
	obs_spec = load_spectra(sample_results['run_params']['objname'])

	label = ['Optical','Akari', 'Spitzer IRS']
	resplots = [spec_res_opt, spec_res_akari, spec_res_spit]

	for ii in xrange(3):
		
		if label[ii] == 'Optical':
			residuals['emlines'] = measure_emline_lum.measure(sample_results, obs_spec, magphys,sps)
			sigsmooth[ii] = residuals['emlines']['sigsmooth']

		residuals[label[ii]] = plot_obs_spec(obs_spec, phot, resplots[ii], alpha, modlam/1e4, modspec,
					                         magphys['model']['lam']/1e4, magphys['model']['spec']*spec_fac,
					                         magphys['metadata']['redshift'], sample_results['run_params']['objname'],
		                                     ii+1, color=obs_color, label=label[ii],sigsmooth=sigsmooth[ii])



	# diagnostic text
	textx = 0.98
	texty = 0.15
	deltay = 0.045


	#### SFR and mass
	# calibrated to be to the right of ax_loc = [0.38,0.68,0.13,0.13]
	prosp_sfr = sample_results['extras']['q50'][sample_results['extras']['parnames'] == 'sfr_100'][0]
	prosp_mass = np.log10(sample_results['quantiles']['q50'][np.array(sample_results['model'].theta_labels()) == 'mass'][0])
	mag_mass = np.log10(magphys['model']['parameters'][magphys['model']['parnames'] == 'M*'][0])
	mag_sfr = magphys['model']['parameters'][magphys['model']['parnames'] == 'SFR'][0]
	
	phot.text(0.02, 0.95, r'log(M$_{\mathrm{q50}}$)='+"{:.2f}".format(prosp_mass),
			              fontsize=14, color=prosp_color,transform = phot.transAxes)
	phot.text(0.02, 0.95-deltay, r'SFR$_{\mathrm{q50}}$='+"{:.2f}".format(prosp_sfr),
			                       fontsize=14, color=prosp_color,transform = phot.transAxes)
	phot.text(0.02, 0.95-2*deltay, r'log(M$_{\mathrm{best}}$)='+"{:.2f}".format(mag_mass),
			                     fontsize=14, color=magphys_color,transform = phot.transAxes)
	phot.text(0.02, 0.95-3*deltay, r'SFR$_{\mathrm{best}}$='+"{:.2f}".format(mag_sfr),
			                     fontsize=14, color=magphys_color,transform = phot.transAxes)

	# calculate reduced chi-squared
	chisq=np.sum(chi**2)/np.sum(sample_results['obs']['phot_mask'])
	chisq_magphys=np.sum(chi_magphys**2)/np.sum(sample_results['obs']['phot_mask'])
	phot.text(textx, texty, r'best-fit $\chi^2/$N$_{\mathrm{phot}}$='+"{:.2f}".format(chisq),
			  fontsize=14, ha='right', color=prosp_color,transform = phot.transAxes)
	phot.text(textx, texty-deltay, r'best-fit $\chi^2/$N$_{\mathrm{phot}}$='+"{:.2f}".format(chisq_magphys),
			  fontsize=14, ha='right', color=magphys_color,transform = phot.transAxes)
		
	z_txt = sample_results['model'].params['zred'][0]
		
	# galaxy text
	phot.text(textx, texty-2*deltay, 'z='+"{:.2f}".format(z_txt),
			  fontsize=14, ha='right',transform = phot.transAxes)
		
	# extra line
	phot.axhline(0, linestyle=':', color='grey')
	
	##### add SFH plot
	from threedhst_diag import add_sfh_plot
	ax_loc = [0.38,0.68,0.13,0.13]
	text_size = 1.5
	ax_inset = add_sfh_plot(sample_results,fig,ax_loc,sps,text_size=text_size)
	magmass = magphys['model']['full_parameters'][[magphys['model']['full_parnames'] == 'M*/Msun']]
	magsfr = np.log10(magphys['sfh']['sfr']*magmass)
	magtime = np.abs(np.max(magphys['sfh']['age']) - magphys['sfh']['age'])/1e9
	ax_inset.plot(magtime,magsfr,color=magphys_color,alpha=0.9,linestyle='-')
	ax_inset.text(0.92,0.08, 'MAGPHYS',color=magphys_color, transform = ax_inset.transAxes,fontsize=4*text_size*1.4,ha='right')
	if ax_inset.get_xlim()[0] < np.max(magtime):
		ax_inset.set_xlim(np.max(magtime),0.0)

	# logify
	ax_inset.set_xscale('log',nonposx='clip',subsx=(2,4,7))
	ax_inset.set_xlim(ax_inset.get_xlim()[0],0.005)
	ax_inset.set_xlabel('log(t/Gyr)')


	# legend
	# make sure not to repeat labels
	from collections import OrderedDict
	handles, labels = phot.get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	phot.legend(by_label.values(), by_label.keys(), 
				loc=1, prop={'size':12},
			    frameon=False)
			    
    # set labels
	res.set_ylabel( r'$\chi$')
	for plot in resplots: plot.set_ylabel( r'log(f$_{\mathrm{obs}}/$f$_{\mathrm{mod}}$)')
	phot.set_ylabel(r'$\nu f_{\nu}$')
	spec_res_spit.set_xlabel(r'$\lambda_{obs}$ [$\mathrm{\mu}$m]')
	res.set_xlabel(r'$\lambda_{obs}$ [$\mathrm{\mu}$m]')
	
	# set scales
	phot.set_yscale('log',nonposx='clip')
	phot.set_xscale('log',nonposx='clip')
	res.set_xscale('log',nonposx='clip',subsx=(2,4,7))
	res.xaxis.set_minor_formatter(minorFormatter)
	res.xaxis.set_major_formatter(majorFormatter)

	# chill on the number of tick marks
	#res.yaxis.set_major_locator(MaxNLocator(4))
	allres = resplots+[res]
	for plot in allres: plot.yaxis.set_major_locator(MaxNLocator(4))

	# clean up and output
	fig.add_subplot(phot)
	for res in allres: fig.add_subplot(res)
	
	# set second x-axis
	y1, y2=phot.get_ylim()
	x1, x2=phot.get_xlim()
	ax2=phot.twiny()
	ax2.set_xticks(np.arange(0,10,0.2))
	ax2.set_xlim(x1/(1+z_txt), x2/(1+z_txt))
	ax2.set_xlabel(r'$\lambda_{rest}$ [$\mathrm{\mu}$m]')
	ax2.set_ylim(y1, y2)
	ax2.set_xscale('log',nonposx='clip',subsx=(2,4,7))
	ax2.xaxis.set_minor_formatter(minorFormatter)
	ax2.xaxis.set_major_formatter(majorFormatter)


	# remove ticks
	phot.set_xticklabels([])
    
	if outname is not None:
		fig.savefig(outname, bbox_inches='tight', dpi=500)
		#os.system('open '+outname)
		plt.close()

	# save chi for photometry
	out = {'chi_magphys': chi_magphys,
	       'chi_prosp': chi,
	       'chisq_prosp': chisq,
	       'chisq_magphys': chisq_magphys,
	       'lam_obs': wave_eff,
	       'z': magphys['metadata']['redshift']
	       }
	residuals['phot'] = out
	return residuals
	
def collate_data(filebase=None,
				 outfolder=os.getenv('APPS')+'/threedhst_bsfh/plots/',
				 sample_results=None,
				 sps=None,
				 plt_sed=True):

	'''
	Driver. Loads output, makes residual plots for a given galaxy, saves collated output.
	'''

	# make sure the output folder exists
	if not os.path.isdir(outfolder):
		os.makedirs(outfolder)

	sample_results, powell_results, model = threed_dutils.load_prospectr_data(filebase)

	if not sps:
		# load stellar population, set up custom filters
		if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
			sps = threed_dutils.setup_sps(custom_filter_key=sample_results['run_params'].get('custom_filter_key',None))
		else:
			sps = threed_dutils.setup_sps(zcontinuous=1,
										  custom_filter_key=sample_results['run_params'].get('custom_filter_key',None))

	# load magphys
	objname = sample_results['run_params']['objname']
	magphys = read_magphys_output(objname=objname)

	# BEGIN PLOT ROUTINE
	print 'MAKING PLOTS FOR ' + objname + ' in ' + outfolder

	# sed plot
	# don't cache emission lines, since we will want to turn them on / off
	sample_results['model'].params['add_neb_emission'] = np.array(True)
	print 'MAKING SED COMPARISON PLOT'
	# plot
	residuals = sed_comp_figure(sample_results, sps, copy.deepcopy(sample_results['model']),
					  magphys, maxprob=1,
					  outname=outfolder+objname.replace(' ','_')+'.sed.png')
 		
	# SAVE OUTPUTS
	alldata = {}
	if residuals is not None:
		print 'SAVING OUTPUTS for ' + sample_results['run_params']['objname']
		alldata['residuals'] = residuals
		alldata = update_model_info(alldata, sample_results, magphys)
	else:
		alldata = None

	return alldata

def plt_all(runname=None,startup=True,**extras):

	'''
	for a list of galaxies, make all plots

	startup: if True, then make all the residual plots and save pickle file
			 if False, load previous pickle file
	'''
	if runname == None:
		runname = 'brownseds'

	output = outpickle+'/alldata.pickle'
	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/magphys/sed_residuals/'

	if startup == True:
		filebase, parm_basename, ancilname=threed_dutils.generate_basenames(runname)
		alldata = []

		for jj in xrange(len(filebase)):
			print 'iteration '+str(jj) 


			dictionary = collate_data(filebase=filebase[jj],\
			                           outfolder=outfolder,
			                           **extras)
			alldata.append(dictionary)
		pickle.dump(alldata,open(output, "wb"))
	else:
		with open(output, "rb") as f:
			alldata=pickle.load(f)

	#### herschel flag
	hflag = np.array([True if np.sum(dat['residuals']['phot']['lam_obs'] > 5e5) else False for dat in alldata])

	#### test different time_res_incr prescriptions
	time_res_test = False
	if time_res_test == True:
		with open('/Users/joel/code/magphys/data/pickles/old/alldata.pickle', "rb") as f:
			alldata_low=pickle.load(f)
		mag_ensemble.time_res_incr_comp(alldata_low,alldata)

	mag_ensemble.prospectr_comparison(alldata,os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/pcomp/',hflag)
	mag_ensemble.plot_emline_comp(alldata,os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/magphys/emlines_comp/',hflag)
	mag_ensemble.plot_relationships(alldata,os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/magphys/')
	plot_all_residuals(alldata)
	mag_ensemble.plot_comparison(alldata,os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/magphys/')

def add_sfr_info(runname=None, outfolder=None):

	if runname == None:
		runname = 'brownseds'

	#### load up prospectr results
	filebase, parm_basename, ancilname=threed_dutils.generate_basenames(runname)



	output = outpickle+'/alldata.pickle'
	outname = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/pcomp/sfrcomp.png'

	with open(output, "rb") as f:
		alldata=pickle.load(f)

	sps = threed_dutils.setup_sps(custom_filter_key=None)

	sfr_mips_z2, sfr_mips, sfr_uvir, sfr_prosp = [], [], [], []
	for ii,dat in enumerate(alldata):

		#### load up spec by generating it from model
		sample_results, powell_results, model = threed_dutils.load_prospectr_data(filebase[ii])
		maxprob = sample_results['bfit']['maxprob_params']
		sample_results['model'].params['zred'] = np.array(0.0)
		spec,mags,sm = sample_results['model'].mean_model(maxprob, sample_results['obs'], sps=sps) # Lsun / Hz

		mips_idx = sample_results['obs']['filters'] == 'MIPS_24'

		obs_mips = sample_results['observables']['mags'][mips_idx,0] # in maggies
		obs_mips = obs_mips[0] * 3631*1e-23 # to Jy, to erg/s/cm^2/Hz

		z = dat['residuals']['phot']['z']

		# input angstroms, Lsun/Hz
		# output in erg/s, convert to erg / s
		luv = threed_dutils.return_luv(sps.wavelengths,spec)/constants.L_sun.cgs.value
		lir = threed_dutils.return_lir(sps.wavelengths,spec)/constants.L_sun.cgs.value

		# input angstroms, Lsun/Hz
		# output in apparent magnitude
		# convert to erg/s/cm^2/Hz
		mips,_ = threed_dutils.integrate_mag(sps.wavelengths*(1.+z),spec,'MIPS_24um_AEGIS',z=z)
		mips_fluxdens = 10**((mips+48.60)/(-2.5)) # erg/s/cm^2/Hz
		mips_z2_mag,_ = threed_dutils.integrate_mag(sps.wavelengths*(1+2.0),spec,'MIPS_24um_AEGIS',z=2.0)
		mips_z2_fluxdens = 10**((mips_z2_mag+48.60)/(-2.5)) # erg/s/cm^2/Hz

		# goes in in milliJy, comes out in Lsun
		lir_mips = threed_dutils.mips_to_lir(mips_fluxdens/1e-23/1e-3,z)
		lir_mips_z2 = threed_dutils.mips_to_lir(mips_z2_fluxdens/1e-23/1e-3,2.0)

		# input in Lsun, output in SFR/yr
		sfr_uvir.append(threed_dutils.sfr_uvir(lir,luv))
		sfr_mips.append(threed_dutils.sfr_uvir(lir_mips,luv))
		sfr_mips_z2.append(threed_dutils.sfr_uvir(lir_mips_z2,luv))
		sfr_prosp.append(dat['bfit']['sfr_100'])

		print sfr_uvir[-1]
		print sfr_mips[-1]
		print sfr_mips_z2[-1]
		print sfr_prosp[-1]

	fig, ax = plt.subplots(1,3, figsize = (22,8))

	xplot = np.log10(np.clip(sfr_prosp,1e-3,np.inf))
	yplot = [np.log10(sfr_uvir), np.log10(sfr_mips), np.log10(sfr_mips_z2)]
	xlabel = r'log(SFR$_{100}$) [Prospectr]'
	ylabel = [r'log(SFR$_{UV+IR}$)', r'log(SFR$_{UV+IR[mips]}$)',r'log(SFR$_{UV+IR[mips,z=2]}$)']

	for ii in xrange(3):
		ax[ii].errorbar(xplot, yplot[ii], fmt='o',alpha=0.6,linestyle=' ',color='0.4')
		ax[ii].set_xlabel(xlabel)
		ax[ii].set_ylabel(ylabel[ii])
		ax[ii] = threed_dutils.equalize_axes(ax[ii], xplot, yplot[ii])
		off,scat = threed_dutils.offset_and_scatter(xplot, yplot[ii], biweight=True)
		ax[ii].text(0.99,0.05, 'biweight scatter='+"{:.3f}".format(scat) +' dex',
				  transform = ax[ii].transAxes,horizontalalignment='right')
		ax[ii].text(0.99,0.1, 'mean offset='+"{:.3f}".format(off)+ ' dex',
				      transform = ax[ii].transAxes,horizontalalignment='right')

	plt.tight_layout()
	plt.savefig(outname, dpi=300)


	print 1/0

def add_prosp_mag_info(runname=None):
	
	if runname == None:
		runname = 'brownseds'

	output = outpickle+'/alldata.pickle'
	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/magphys/sed_residuals/'

	with open(output, "rb") as f:
		alldata=pickle.load(f)

	filebase, parm_basename, ancilname=threed_dutils.generate_basenames(runname)
	for ii,dat in enumerate(alldata):
		sample_results, powell_results, model = threed_dutils.load_prospectr_data(filebase[ii])
		magphys = read_magphys_output(objname=dat['objname'])
		dat = update_model_info(dat, sample_results, magphys)
		print str(ii)+' done'

	pickle.dump(alldata,open(output, "wb"))














