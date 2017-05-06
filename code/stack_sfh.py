import numpy as np
import matplotlib.pyplot as plt
import prosp_dutils
import magphys_plot_pref
from matplotlib.ticker import MaxNLocator
import copy
import matplotlib as mpl
from corner import quantile
import math
import os


### time per bin
# must recalculate if the bins change!
time_per_bin = np.array([1e+08,2.16227766e+08,6.83772234e+08,2.16227766e+09,3.14729578e+09,7.45932607e+09])

mpl.rcParams.update({'font.size': 18})
mpl.rcParams.update({'font.weight': 500})
mpl.rcParams.update({'axes.labelweight': 500})

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

def salim_mainsequence(mass,ssfr=False,**junk):
	'''
	mass in log, returns SFR in log
	'''
	ssfr_salim = -0.35*(mass-10)-9.83
	if ssfr:
		return ssfr_salim
	salim_sfr = np.log10(10**ssfr_salim*10**mass)
	return salim_sfr

def calculate_mean(frac, ndraw=5e5):
	'''
	given list of N PDFs
	calculate the mean by sampling from the PDFs
	'''
	nobj = len(frac)
	mean_array = np.zeros(shape=(ndraw,nobj))
	for i,f in enumerate(frac): mean_array[:,i] = np.random.choice(f,size=ndraw)
	mean_pdf = mean_array.mean(axis=1)
	mean, errup, errdown = quantile(mean_pdf, [0.5,0.84,0.16])

	return mean, errup, errdown

def stack_data(alldata,sigma_sf=None,nbins_horizontal=None,low_mass_cutoff=None,
               show_disp=None,nbins_vertical=None,**junk):

	'''
	stack in fraction for now
	not quite sure what it means physically but it has a ~monotonic relationship with sSFR 
	take the mean
	'''

	### parameter names
	parnames = alldata[0]['pquantiles']['parnames']
	idx_mass = parnames == 'logmass'

	### fractional parameters, for stacking
	fracpars = np.array([True if 'z_fraction' in par else False for par in parnames],dtype=bool)
	frac_parnames = parnames[fracpars]
	nbins_sfh = frac_parnames.shape[0]

	### extra parameters
	eparnames = alldata[0]['pextras']['parnames']
	idx_sfr = eparnames == 'sfr_100'
	idx_time = eparnames == 'half_time'

	### setup dictionary
	outdict = {}

	### grab mass, SFR, SFR_ms for each galaxy
	outdict['logm'] = np.array([dat['pquantiles']['q50'][idx_mass][0] for dat in alldata])
	outdict['logm_eup'] = np.array([dat['pquantiles']['q84'][idx_mass][0] for dat in alldata])
	outdict['logm_edo'] = np.array([dat['pquantiles']['q16'][idx_mass][0] for dat in alldata])
	outdict['logm_err'] = prosp_dutils.asym_errors(outdict['logm'],
			                                        outdict['logm_eup'],
			                                        outdict['logm_edo'])
	outdict['logsfr'] = np.log10([dat['pextras']['q50'][idx_sfr][0] for dat in alldata])
	outdict['halftime'] = np.array([dat['pextras']['q50'][idx_time][0] for dat in alldata])
	outdict['halftime_eup'] = np.array([dat['pextras']['q84'][idx_time][0] for dat in alldata])
	outdict['halftime_edo'] = np.array([dat['pextras']['q16'][idx_time][0] for dat in alldata])
	outdict['halftime_err'] = prosp_dutils.asym_errors(outdict['halftime'],
			                                            outdict['halftime_eup'],
			                                            outdict['halftime_edo'])

	outdict['logsfr_ms'] = salim_mainsequence(outdict['logm'])

	### stack horizontal
	sfr_frac = {}
	sfr_frac['properties'] = outdict

	on_ms = (outdict['logm'] > low_mass_cutoff) & \
	        (np.abs(outdict['logsfr'] - outdict['logsfr_ms']) < sigma_sf)

	sfr_frac['mass_range'] = (outdict['logm'][on_ms].min(),outdict['logm'][on_ms].max())
	sfr_frac['mass_bins'] = np.linspace(sfr_frac['mass_range'][0],sfr_frac['mass_range'][1],nbins_horizontal+1)
	sfr_frac['on_ms'] = on_ms

	percentiles = [0.5,show_disp[1],show_disp[0]]

	### for each main sequence bin
	for j in xrange(nbins_horizontal):
		tdict = {}
		tdict['mean'],tdict['err'],tdict['errup'],tdict['errdown'] = [],[],[],[]

		### what's in the bin?
		in_bin = (outdict['logm'][on_ms] >= sfr_frac['mass_bins'][j]) & (outdict['logm'][on_ms] <= sfr_frac['mass_bins'][j+1])
		tdict['logm'],tdict['logsfr'] = outdict['logm'][on_ms][in_bin],outdict['logsfr'][on_ms][in_bin]

		### calculate sSFR chains (fn / sum(tn*fn))
		outfrac = []
		for dat in np.array(alldata)[on_ms][in_bin]:
			frac = prosp_dutils.transform_zfraction_to_sfrfraction(dat['pquantiles']['sample_chain'][:,fracpars])
			frac = np.concatenate((frac, (1-frac.sum(axis=1))[:,None]),axis=1)
			norm = (frac * time_per_bin).sum(axis=1) ### sum(fn*tn)
			outfrac.append(frac/norm[:,None])

		### calculate statistics for each SFR bin
		for i in xrange(time_per_bin.shape[0]):
			frac = [f[:,i] for f in outfrac]
			mean,errup,errdown = calculate_mean(frac)
			tdict['mean'].append(mean)
			tdict['errup'].append(errup)
			tdict['errdown'].append(errdown)

		tdict['err'] = prosp_dutils.asym_errors(np.array(tdict['mean']),
			                                     np.array(tdict['errup']),
			                                     np.array(tdict['errdown']))

		### name your bin something creative!
		sfr_frac['massbin'+str(j)] = tdict

	### stack vertical
	sfr_vert = {}
	### for each bin
	for j in xrange(nbins_vertical):

		tdict = {}
		tdict['mean'],tdict['err'],tdict['errup'],tdict['errdown'] = [],[],[],[]

		### what's in the bin?
		in_bin = (outdict['logm'] > low_mass_cutoff) & \
        	     ((outdict['logsfr'] - outdict['logsfr_ms']) >= sigma_sf*2*(j-2)) & \
				 ((outdict['logsfr'] - outdict['logsfr_ms']) < sigma_sf*2*(j-1))
		tdict['logm'],tdict['logsfr'] = outdict['logm'][in_bin],outdict['logsfr'][in_bin]

		### calculate sSFR chains (fn / sum(tn*fn))
		outfrac = []
		for dat in np.array(alldata)[in_bin]:
			frac = prosp_dutils.transform_zfraction_to_sfrfraction(dat['pquantiles']['sample_chain'][:,fracpars])
			frac = np.concatenate((frac, (1-frac.sum(axis=1))[:,None]),axis=1)
			norm = (frac * time_per_bin).sum(axis=1) ### sum(fn*tn)
			outfrac.append(frac/norm[:,None])

		### calculate statistics for each SFR bin
		for i in xrange(time_per_bin.shape[0]):
			frac = [f[:,i] for f in outfrac]
			mean,errup,errdown = calculate_mean(frac)
			tdict['mean'].append(mean)
			tdict['errup'].append(errup)
			tdict['errdown'].append(errdown)

		tdict['err'] = prosp_dutils.asym_errors(np.array(tdict['mean']),
			                                     np.array(tdict['errup']),
			                                     np.array(tdict['errdown']))

		### name your bin something creative!
		sfr_vert['massbin'+str(j)] = tdict

	return sfr_frac,sfr_vert
	
def plot_main_sequence(ax,sigma_sf=None,low_mass_cutoff=7, **junk):

	#### ADD SALIM+07
	mass = np.linspace(low_mass_cutoff,12,40)
	sfr = salim_mainsequence(mass)

	ax.plot(mass, sfr,
	          color='green',
	          alpha=0.8,
	          lw=2.5,
	          label='Salim+07',
	          zorder=-1)
	ax.fill_between(mass, sfr-sigma_sf, sfr+sigma_sf, 
	                  color='green',
	                  alpha=0.3)

def plot_mass_v_time(dat,outname):
	
	fig, ax = plt.subplots(1,1, figsize=(7,7))
	
	ms_plot_opts = {
	                  'alpha':0.75,
	                  'fmt':'o',
	                  'mew':1.5,
	                  'ms':10
				   }

	on_ms = dat['on_ms']
	ax.errorbar(dat['properties']['logm'][on_ms], dat['properties']['halftime'][on_ms],
			    xerr=[dat['properties']['logm_err'][0][on_ms],dat['properties']['logm_err'][1][on_ms]],
			    yerr=[dat['properties']['halftime_err'][0][on_ms],dat['properties']['halftime_err'][1][on_ms]],
		        color='0.3',**ms_plot_opts)

	ax.set_xlim(dat['mass_bins'].min()-0.3,dat['mass_bins'].max()+0.3)
	ax.set_xlabel(r'log(M$_{*}$)')
	ax.set_ylabel(r't$_{\mathrm{half-mass}}$ [Gyr]')
	ax.set_ylim(0.2,13.6)
	ax.set_yscale('log',nonposy='clip',subsy=(1,2,4))
	ax.yaxis.set_minor_formatter(minorFormatter)
	ax.yaxis.set_major_formatter(majorFormatter)

	plt.savefig(outname,dpi=150)
	plt.close()

def plot_stacked_sfh(alldata,outfolder,lowmet=True):

	### important parameters
	outname_ms_horizontal = outfolder+'ms_horizontal_stack.png'
	outname_ms_vertical = outfolder+'ms_vertical_stack.png'
	outname_halftime = outfolder+'ms_mass_v_time.png'

	config = {
	          'sigma_sf':0.5,                  # scatter in the star-forming sequence, in dex
	          'nbins_horizontal':3,            # number of bins in horizontal stack
	          'nbins_vertical':3,              # number of bins in vertical stack
	          'horizontal_bin_colors': ['#45ADA8','#FC913A','#FF4E50'],
	          'vertical_bin_colors': ['#45ADA8','#FC913A','#FF4E50'],
	          'low_mass_cutoff':8.7,           # log(M) where we stop stacking and plotting
	          'show_disp':[0.16,0.84]          # percentile of population distribution to show on plot
	         }

	dat,datvert = stack_data(alldata,**config)
	plot_mass_v_time(dat,outname_halftime)
	agelims = np.array([7.0,8.0,8.5,9.0,9.5,9.8,10.13]) # jacked straight from the parameter file, log(Gyr)
	agebins = (agelims[1:]+agelims[:-1])/2. # evenly spaced in log-years

	#### horizontal stack figure
	fig, ax = plt.subplots(1,2, figsize=(14,7))
	plot_main_sequence(ax[0],**config)
	ylim = ax[0].get_ylim()

	ms_plot_opts = {
	                'alpha':0.8,
	                'mew':1.5,
	                'linestyle':' ',
	                'marker':'o',
	                'ms':10
				   }
	ms_line_plot_opts = {
	                     'lw':4,
	                     'linestyle':'-',
	                     'alpha':0.8,
	                     'zorder':-32
				        }
	stack_plot_opts = {
	                  'alpha':0.7,
	                  'fmt':'o',
	                  'mew':1.5,
	                  'linestyle':'-',
	                  'ms':20,
	                  'lw':2,
	                  'elinewidth':2,
	                  'capsize':8,
	                  'capthick':1.5
				      }

	#### plot main sequence data and stacks
	for i in xrange(config['nbins_horizontal']):
		
		### bin designation (very creative)
		idx = 'massbin'+str(i)

		### plot star-forming sequence
		ax[0].plot(dat[idx]['logm'],dat[idx]['logsfr'],
			       color=config['horizontal_bin_colors'][i],
			       **ms_plot_opts)

		### plot SFH stacks
		ax[1].errorbar(10**(agebins-0.04*(i-1)),dat[idx]['mean'],
			            yerr=dat[idx]['err'],
			            color=config['horizontal_bin_colors'][i],
			            **stack_plot_opts)

	### labels and ranges
	ax[0].set_xlabel(r'log(M$_{*}$)')
	ax[0].set_ylabel(r'log(SFR)')
	ax[0].set_ylim(ylim)
	ax[0].set_xlim(dat['mass_bins'].min()-0.3,dat['mass_bins'].max()+0.3)

	ax[1].set_xlim(10**(agebins.min()-0.3),10**(agebins.max()+0.3))
	ax[1].set_xlabel(r'time [yr]')
	ax[1].set_ylabel(r'SFR$_{\mathrm{bin}}$ / M$_{\mathrm{tot}}$ [yr$^{-1}$]')
	ax[1].set_xscale('log',nonposx='clip',subsx=(1,2,4))
	ax[1].set_yscale('log',nonposx='clip',subsx=(1,2,4))

	#### plot mass ranges (AFTER YRANGE IS DETERMINED)
	for i in xrange(config['nbins_horizontal']):
		idx = 'massbin'+str(i)
		if i == 0:
			dlowbin = -0.01
		else:
			dlowbin = 0.017
		if i == config['nbins_horizontal']-1:
			dhighbin = 0.01
		else:
			dhighbin = -0.017

		ax[0].plot(np.repeat(dat['mass_bins'][i]+dlowbin,2),ylim,
			       color=config['horizontal_bin_colors'][i],
			       **ms_line_plot_opts)
		ax[0].plot(np.repeat(dat['mass_bins'][i+1]+dhighbin,2),ylim,
			       color=config['horizontal_bin_colors'][i],
			       **ms_line_plot_opts)

	plt.tight_layout()
	plt.savefig(outname_ms_horizontal,dpi=150)
	plt.close()




	#### vertical stack figure
	fig, ax = plt.subplots(1,2, figsize=(14,7))

	#### plot main sequence position and stacks
	for i in xrange(config['nbins_vertical']):
		
		### bin designation (very creative)
		idx = 'massbin'+str(i)

		### plot star-forming sequence
		ax[0].plot(datvert[idx]['logm'],datvert[idx]['logsfr'],
			       color=config['horizontal_bin_colors'][i],
			       **ms_plot_opts)

		### plot SFH stacks
		ax[1].errorbar(10**(agebins-0.04*(i-1)),datvert[idx]['mean'],
			            yerr=datvert[idx]['err'],
			            color=config['horizontal_bin_colors'][i],
			            **stack_plot_opts)
		if i == 0:
			minmass,maxmass = datvert[idx]['logm'].min(),datvert[idx]['logm'].max()
			minsfr,maxsfr = datvert[idx]['logsfr'].min(),datvert[idx]['logsfr'].max()
		else:
			minmass = np.min([minmass,datvert[idx]['logm'].min()])
			maxmass = np.max([maxmass,datvert[idx]['logm'].max()])
			minsfr = np.min([minsfr,datvert[idx]['logsfr'].min()])
			maxsfr = np.max([maxsfr,datvert[idx]['logsfr'].max()])

	### labels and ranges
	ax[0].set_xlabel(r'log(M$_{*}$)')
	ax[0].set_ylabel(r'log(SFR)')
	xlim = (minmass-0.3,maxmass+0.3)
	ylim = (minsfr-0.3,maxsfr+0.3)
	ax[0].set_xlim(xlim)
	ax[0].set_ylim(ylim)

	ax[1].set_xlim(10**(agebins.min()-0.3),10**(agebins.max()+0.3))
	ax[1].set_xlabel(r'time [yr]')
	ax[1].set_ylabel(r'SFR$_{\mathrm{bin}}$ / M$_{\mathrm{tot}}$ [yr$^{-1}$]')
	ax[1].set_xscale('log',nonposx='clip',subsx=(1,2,4))
	ax[1].set_yscale('log',nonposx='clip',subsx=(1,2,4))

	#### plot mass ranges
	ymid = np.array([salim_mainsequence(xlim[0]),salim_mainsequence(xlim[1])])
	for i in xrange(config['nbins_vertical']):
		idx = 'massbin'+str(i)
		if i == 0:
			dlowbin = -0.01
		else:
			dlowbin = 0.017
		if i == config['nbins_vertical']-1:
			dhighbin = 0.01
		else:
			dhighbin = -0.017

		ax[0].plot(xlim,ymid+config['sigma_sf']*2*(i-1)+dhighbin,
			       color=config['horizontal_bin_colors'][i],
			       **ms_line_plot_opts)
		ax[0].plot(xlim,ymid+config['sigma_sf']*2*(i-2)+dlowbin,
			       color=config['horizontal_bin_colors'][i],
			       **ms_line_plot_opts)

	plt.tight_layout()
	plt.savefig(outname_ms_vertical,dpi=150)
	plt.close()

	mpl.rcParams.update({'font.weight': 400})
	mpl.rcParams.update({'axes.labelweight': 400})










