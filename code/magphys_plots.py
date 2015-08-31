import numpy as np
import matplotlib.pyplot as plt
from magphys import read_magphys_output
import os, copy, threed_dutils
from bsfh import read_results
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import pickle, math
import matplotlib as mpl

# plotting preferences
mpl.rcParams.update({'font.size': 16})
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 0.5
mpl.rcParams['ytick.minor.width'] = 0.5

c = 3e18   # angstroms per second
minsfr = 1e-4
plt.ioff() # don't pop up a window for each plot

#### set up colors and plot style
prosp_color = '#e60000'
obs_color = '#95918C'
magphys_color = '#1974D2'

#### where do the pickle files go?
outpickle = '/Users/joel/code/magphys/data/pickles'

class jLogFormatter(mpl.ticker.LogFormatter):
	'''
	this just changes the format from exponential to floating point.
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

def equalize_axes(ax, x,y, dynrange=0.1, line_of_equality=True):
	
	''' 
	sets up an equal x and y range that encompasses all of the data
	if line_of_equality, add a diagonal line of equality 
	dynrange represents the % of the data range above and below which
	the plot limits are set
	'''

	dynx, dyny = (np.nanmax(x)-np.nanmin(x))*dynrange,\
                 (np.nanmax(y)-np.nanmin(y))*dynrange
	if np.nanmin(x)-dynx > np.nanmin(y)-dyny:
		min = np.nanmin(y)-dyny
	else:
		min = np.nanmin(x)-dynx
	if np.nanmax(x)+dynx > np.nanmax(y)+dyny:
		max = np.nanmax(x)+dynx
	else:
		max = np.nanmax(y)+dyny

	ax.axis((min,max,min,max))

	if line_of_equality:
		ax.errorbar([-1e3,1e3],[-1e3,1e3],linestyle='--',color='0.1',alpha=0.8)

	return ax
	
def plot_relationships(alldata,outfolder):

	'''
	mass-metallicity
	mass-SFR
	etc
	'''

	##### set up plots
	fig = plt.figure(figsize=(10,5))
	gs1 = mpl.gridspec.GridSpec(1, 2)
	msfr = plt.Subplot(fig, gs1[0])
	mz = plt.Subplot(fig, gs1[1])

	fig.add_subplot(msfr)
	fig.add_subplot(mz)

	alpha = 0.6

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
			      xerr=proerrs_mass, yerr=proerrs_sfr)
	msfr.errorbar(magmass[:,1],magsfr[:,1],
		          fmt='o', alpha=alpha,
		          color=magphys_color,
		          label='MAGPHYS',
			      xerr=magerrs_mass, yerr=magerrs_sfr)

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
			      xerr=proerrs_mass, yerr=proerrs_met)
	mz.errorbar(magmass[:,1],magmet,
		          fmt='o', alpha=alpha,
		          color=magphys_color,
			      xerr=magerrs_mass)	

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
	mz.legend(loc=2, prop={'size':12},
			    frameon=False)

	mz.set_xlabel(r'log(M/M$_{\odot}$)')
	mz.set_ylabel(r'log(Z/Z$_{\odot}$/yr)')

	print 1/0

def plot_comparison(alldata,outfolder):

	'''
	mass vs mass
	sfr vs sfr
	etc
	'''

	##### set up plots
	fig = plt.figure(figsize=(10,10))
	gs1 = mpl.gridspec.GridSpec(2, 2)
	mass = plt.Subplot(fig, gs1[0])
	sfr = plt.Subplot(fig, gs1[1])
	met = plt.Subplot(fig, gs1[2])
	age = plt.Subplot(fig,gs1[3])

	fig.add_subplot(mass)
	fig.add_subplot(sfr)
	fig.add_subplot(met)
	#fig.add_subplot(age)

	alpha = 0.6

	##### find prospectr indexes
	parnames = alldata[0]['pquantiles']['parnames']
	idx_mass = parnames == 'mass'
	idx_met = parnames == 'logzsol'

	eparnames = alldata[0]['pextras']['parnames']
	idx_sfr = eparnames == 'sfr_100'

	##### find magphys indexes
	idx_mmet = alldata[0]['model']['full_parnames'] == 'Z/Zo'

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
	mass = equalize_axes(mass,promass[:,1],magmass[:,1])

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
	sfr = equalize_axes(sfr,prosfr[:,1],magsfr[:,1])

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
	met = equalize_axes(met,promet[:,1],magmet)

	# text
	off,scat = threed_dutils.offset_and_scatter(promet[:,1],magmet,biweight=True)
	met.text(0.99,0.05, 'biweight scatter='+"{:.2f}".format(scat) + ' dex',
			  transform = met.transAxes,horizontalalignment='right')
	met.text(0.99,0.1, 'mean offset='+"{:.2f}".format(off) + ' dex',
		      transform = met.transAxes,horizontalalignment='right')
	

	plt.savefig(outfolder+'basic_comparison.png',dpi=300)
	plt.close()

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
	# first call is transparent, to get bins
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

	if nufnu:

		# convert to flam * lam
		flux = flux * obs_lam

		# convert to janskys, then maggies * Hz
		flux = flux * 1e23 / 3631

	out = {}
	out['rest_lam'] = rest_lam
	out['flux'] = flux
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

def plot_obs_spec(obs_spec, phot, spec_res, alpha, 
	              modlam, modspec, maglam, magspec,z, 
	              objname, source, sigsmooth,
	              color='black',label=''):

	'''
	standard wrapper for plotting observed + residuals for spectra
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
		spec_res.plot(obslam, 
			          magphys_resid,
			          color=magphys_color,
			          alpha=alpha,
			          linestyle='-')

		#### calculate rms
		magphys_rms = (np.sum((magphys_resid-magphys_resid.mean())**2)/len(obslam))**0.5
		prospectr_rms = (np.sum((prospectr_resid-prospectr_resid.mean())**2)/len(obslam))**0.5

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
		return_sedplot_vars(sample_results['quantiles']['maxprob_params'], 
			                sample_results, sps)
	except KeyError:
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

	
	# find most recent output file
	# with the objname
	folder = "/".join(filebase.split('/')[:-1])
	filename = filebase.split("/")[-1]
	files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
	times = [f.split('_')[-2] for f in files]

	# if we found no files, skip this object
	if len(times) == 0:
		print 'Failed to find any files to extract times in ' + folder + ' of form ' + filename
		return 0

	# load results
	mcmc_filename=filebase+'_'+max(times)+"_mcmc"
	model_filename=filebase+'_'+max(times)+"_model"

	# load if necessary
	if not sample_results:
		try:
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
		except (EOFError,ValueError) as e:
			print e
			print 'Failed to open '+ mcmc_filename +','+model_filename
			return 0
	else:
		import pickle
		try:
			mf = pickle.load( open(model_filename, 'rb'))
		except(AttributeError):
			mf = load( open(model_filename, 'rb'))
       
		powell_results = mf['powell']

	if not sps:
		# load stellar population, set up custom filters
		if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
			sps = threed_dutils.setup_sps(custom_filter_key=sample_results['run_params'].get('custom_filter_key',None))
		else:
			sps = threed_dutils.setup_sps(zcontinuous=1,
										  custom_filter_key=sample_results['run_params'].get('custom_filter_key',None))

	# load magphys
	magphys = read_magphys_output(objname=sample_results['run_params']['objname'])

	# BEGIN PLOT ROUTINE
	print 'MAKING PLOTS FOR ' + filename + ' in ' + outfolder
	alldata = {}

	# sed plot
	if plt_sed:
		print 'MAKING SED COMPARISON PLOT'
 		# plot
 		residuals = sed_comp_figure(sample_results, sps, copy.deepcopy(sample_results['model']),
 						  magphys, maxprob=1,
 						  outname=outfolder+filename.replace(' ','_')+'.sed.png')
 		
	# SAVE OUTPUTS
	if residuals is not None:
		print 'SAVING OUTPUTS for ' + sample_results['run_params']['objname']
		alldata['objname'] = sample_results['run_params']['objname']
		alldata['residuals'] = residuals
		alldata['magphys'] = magphys['pdfs']
		alldata['model'] = magphys['model']
		alldata['pquantiles'] = sample_results['quantiles']
		alldata['pextras'] = sample_results['extras']
		alldata['pquantiles']['parnames'] = np.array(sample_results['model'].theta_labels())
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

	plot_relationships(alldata,os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/magphys/')
	plot_all_residuals(alldata)
	plot_comparison(alldata,os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/magphys/')
	