import numpy as np
import matplotlib.pyplot as plt
import corner, math, copy
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import magphys_plot_pref
import threed_dutils
from no_herschel_analysis import create_step

plt.ioff() # don't pop up a window for each plot

obs_color = '#275ee8'

tiny_number = 1e-3
big_number = 1e90
dpi = 100

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

def histograms(axes, opts, limits, idx, thetas, setup=False, iteration=0,labels=None,ndraw=None):

	### plot options
	alpha = 0.8 # histogram edges
	halpha = 1.0 # histogram filling
	lw = 3 # thickness of histogram edges
	nbins = 10
	ylim = [0,ndraw/3.]

	color = '0.5'
	#nhcolor = '#FC913A'
	#nhweightedcolor = '#45ADA8'
	if setup:
		limits['hist'] = {}

	for i, ax in enumerate(axes):

		### label each axis, store limits, create bins
		if setup:
			ax.set_xlabel(labels[i],weight='bold')
			min, max = thetas[:,idx[i]].min(), thetas[:,idx[i]].max()
			dynrange = (max-min)*.1
			limits['hist'][labels[i]] = (min-dynrange,max+dynrange)
			limits['hist'][labels[i]+'_bins'] = np.linspace(min,max,nbins)
			for tl in ax.get_yticklabels():tl.set_visible(False)
			for tl in ax.get_xticklabels():tl.set_visible(False)

		else:
			### if we're showing the truth
			if iteration == 0:
				ax.plot([thetas[0,idx[i]],thetas[0,idx[i]]],ylim,lw=2.5,color=opts['color'],linestyle='--',zorder=6)

			### normal histogram
			### generate PDFs
			pdf, _ = np.histogram(thetas[:,idx[i]],bins=limits['hist'][labels[i]+'_bins'],density=False)
			plotx = create_step(limits['hist'][labels[i]+'_bins'])
			ploty = create_step(pdf,add_edges=True)

			ax.plot(plotx,ploty,alpha=alpha,lw=lw,color=color)

			ax.fill_between(plotx, np.zeros_like(ploty), ploty, 
							   color=color,
							   alpha=halpha)

		ax.set_xlim(limits['hist'][labels[i]])
		ax.set_ylim(ylim)

	return limits

def sed(ax, popts, limits, obs, setup=False, spec=None, w=None):
	"""
	Plot the photometry for the model and data (with error bars), and
	plot residuals
	#sfh_loc = [0.32,0.35,0.12,0.14],
	good complimentary color for the default one is '#FF420E', a light red
	"""

	#### plot observations, define limits, set labels and scales
	if setup:

		#### plot observations
		mask = obs['phot_mask']
		wave_eff = obs['wave_effective'][mask]
		factor = 3e18/wave_eff
		yplot = obs['maggies'][mask]*factor
		yerr = obs['maggies_unc'][mask]*factor
		xplot = wave_eff/1e4

		ax.errorbar(xplot, yplot, yerr=yerr*2,
	                color=obs_color, marker='o', label='observed', alpha=1.0, linestyle=' ',ms=8,zorder=2)

		#### define limits
		limdict = {}
		limdict['xlim'] = (0.1,25)
		limdict['ylim'] = (yplot.min()*0.4,yplot.max()*6)
		limits['sed'] = limdict

	    ### set labels and scales
		ax.set_ylabel(r'flux',weight='bold')
		ax.set_xlabel(r'wavelength [$\mu$m]',weight='bold')
		ax.set_yscale('log',nonposx='clip')
		ax.set_xscale('log',nonposx='clip',subsx=(2,5))
		ax.xaxis.set_minor_formatter(minorFormatter)
		ax.xaxis.set_major_formatter(majorFormatter)
		
	else:

		#### plot SED
		factor = 3e18/w
		nz = spec > 0
		ax.plot(w[nz]/1e4, spec[nz]*factor, **popts)

	### apply plot limits
	ax.set_xlim(limits['sed']['xlim'])
	ax.set_ylim(limits['sed']['ylim'])
	for tl in ax.get_yticklabels():tl.set_visible(False)

	return limits
			    
def sfh(ax, popts, limits, t=None, sfh=None, setup=False):

	if setup:
		#### set limits and scales for SFH
		axfontsize = 16
		ax.set_ylabel(r'SFR',fontsize=axfontsize,weight='bold',labelpad=3)
		ax.set_xlabel('lookback time [Gyr]',fontsize=axfontsize,weight='bold',labelpad=2)
	
		for tl in ax.get_yticklabels():tl.set_visible(False)
		#for tl in ax.get_xticklabels():tl.set_visible(False)

		#for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(axfontsize) 
		ax.set_xscale('log',nonposx='clip',subsx=([1]))
		ax.xaxis.set_minor_formatter(minorFormatter)
		ax.xaxis.set_major_formatter(majorFormatter)

		for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(axfontsize) 
		ax.set_yscale('log',nonposy='clip',subsy=(1,3))
	else: 
		ax.plot(t,sfh,**popts)

		#### limits
		if 'sfh' not in limits.keys():
			limdict = {}
			limdict['xlim'] = (t.min()*10, t.max())
			limdict['ylim'] = (sfh.min()*.1, sfh.max()*8)
			limits['sfh'] = limdict

		ax.set_xlim(limits['sfh']['xlim'])
		ax.set_ylim(limits['sfh']['ylim'])

	return limits

def make_gif(runname='brownseds_np',objname='Arp 256 N', sample_results = None, outfolder='/Users/joel/talks/2016_cfa_postdoc/gif/',ndraw=10):

	'''
	Driver. Loads output, makes plots for a given galaxy.
	'''

	### LOAD DATA
	if sample_results is None:
		#import threed_dutils
		sample_results, powell_results, model = threed_dutils.load_prospector_data(None, runname=runname, objname=objname)

	### setup plot geometry
	fig = plt.figure(figsize=(12, 10))
	gs1 = mpl.gridspec.GridSpec(4, 26)
	gs1.update(hspace=0.35,top=0.95,bottom=0.1,left=0.00,right=1.0)

	### SED + SFH
	sedax = plt.subplot(gs1[:2,2:-2])
	sfhax = fig.add_axes([0.32,0.61,0.14,0.16],zorder=5)

	### histogram axes
	# first row
	ax2 = plt.subplot(gs1[2,2:6])
	ax3 = plt.subplot(gs1[2,8:12])
	ax4 = plt.subplot(gs1[2,14:18])
	ax5 = plt.subplot(gs1[2,20:24])
	
	# second row
	ax6 = plt.subplot(gs1[3,2:6])
	ax7 = plt.subplot(gs1[3,8:12])
	ax8 = plt.subplot(gs1[3,14:18])
	ax9 = plt.subplot(gs1[3,20:24])
	histax = [ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]

	#### plot options, best-fit versus sample
	popts_bf = {
			   'linestyle': '-',
			   'color': '#FF4E50',
			   'alpha': 0.9,
			   'lw':1.5,
			   'zorder': 0
			   }
	popts = {
			 'linestyle': '-',
			 'color': '0.5',
			 'alpha': 0.15,
			 'lw':1.5,
			 'zorder': -2
			 }

	idx = []
	tnames = ['logmass', 'logzsol', 'dust2', 'dust_index', 'dust1', 'duste_qpah', 'duste_gamma', 'duste_umin']
	for name in tnames: idx.append(sample_results['model'].theta_labels().index(name))
	tpnames = [r'stellar mass',r'stellar metallicity', r'diffuse dust','shape of dust \n attenuation curve',
	           r'birth-cloud dust', r'PAH strength',r'minimum dust heating', 'hot dust fraction']
	limits = {}
	limits = sed(sedax, popts, limits,sample_results['obs'], setup=True)
	limits = sfh(sfhax, popts, limits, setup=True)
	limits = histograms(histax, popts, limits, idx, sample_results['quantiles']['sample_flatchain'][:ndraw,:], setup=True,labels=tpnames,ndraw=ndraw)
 	plt.savefig(outfolder+'setup.png',dpi=dpi)


	for i in xrange(ndraw):

		### plot options
		if i == 0:
			opts = popts_bf
		else:
			opts = popts

		### plots
		limits = sed(sedax, opts, limits, sample_results['obs'],
		             spec=sample_results['observables']['spec'][:,i], w=sample_results['observables']['lam_obs'])
		if i == 0:
			opts['lw'] = 2.5
		limits = sfh(sfhax, opts, limits,
			         t=sample_results['extras']['t_sfh'], sfh=sample_results['extras']['sfh'][:,i])
 		limits = histograms(histax, opts, limits, idx, sample_results['quantiles']['sample_flatchain'][:i+1,:], setup=False, labels=tpnames,iteration=i,ndraw=ndraw)
 		plt.savefig(outfolder+'frame'+str(i)+'.png',dpi=dpi)

	