import numpy as np
import matplotlib.pyplot as plt
from prospect.io import read_results
import sys, os
import magphys_plot_pref
from matplotlib.ticker import MaxNLocator
from prospect.fitting.convergence import make_kl_bins, kl_divergence

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

'''
final to-dos:
	-- update chain plot in threedhst_diag. also plot KL divergence there.
	-- create implementation for prospector
	-- run test for brown sample, make sure it works (inspect chain plots), then push to master
	-- these parameters imply how long of the chain before the end is useful for calculations (n_stable_checks_before_stop, etc)
		-- use this in chain-cutting
		-- also use this to update parameter dispersion floors?
'''

def get_cmap(N,cmap='nipy_spectral'):
	'''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
	RGB color.'''

	import matplotlib.cm as cmx
	import matplotlib.colors as colors

	color_norm  = colors.Normalize(vmin=0, vmax=N-1)
	scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap) 
	def map_index_to_rgb_color(index):
		return scalar_map.to_rgba(index)
	return map_index_to_rgb_color

def show_chain(sample_results,legend=True):
	
	'''
	plot the MCMC chain for all parameters
	'''
	
	### names and dimensions
	parnames = np.array(sample_results['model'].theta_labels())
	chain = sample_results['chain']
	lnprob = sample_results['lnprobability']
	nwalkers, nsteps, npars = chain.shape
	nplot = npars+2 # also plot lnprob

	### plot preferences
	nboldchain = 3
	alpha_bold = 0.9
	alpha = 0.05
	lw_bold = 2
	lw = 1
	cmap = get_cmap(nboldchain,cmap='brg')

	### plot geometry
	ndim = len(parnames)
	#ny = 4
	ny = 3
	nx = int(np.ceil(nplot/float(ny)))
	sz = np.array([nx,ny])
	factor = 3.2           # size of one side of one panel
	lbdim = 0.0 * factor   # size of margins
	whspace = 0.0 * factor         # w/hspace size
	plotdim = factor * sz + factor *(sz-1)* whspace
	dim = 2*lbdim + plotdim

	### create plots
	fig, axarr = plt.subplots(ny, nx, figsize = (dim[0], dim[1]*1.1))
	fig.subplots_adjust(wspace=0.4,hspace=0.3)
	axarr = np.ravel(axarr)
	
	### remove some plots to make room for KL divergence
	#off = [13,14,18,19]
	off = [7,8]
	[axarr[o].axis('off') for o in off] # turn off 
	axarr = np.delete(axarr,off) # remove from array

	### check for stuck walkers
	# must visit at least 10 unique points in lnprobability
	outliers = np.full(nwalkers,False,dtype=bool)
	for k in xrange(nwalkers):
		ncall = np.unique(sample_results['lnprobability'][k,:]).shape[0]
		if ncall <= 10:
			outliers[k] = True

	# remove stuck walkers
	nstuck = outliers.sum()
	print str(nstuck)+' stuck walkers found for '+sample_results['run_params'].get('objname','object')
	if nstuck:
		chain = chain[~outliers,:,:]
		lnprob = lnprob[~outliers,:]
		nwalkers = chain.shape[0]

	### plot chain in each parameter
	for i, ax in enumerate(axarr):
		if i < npars: # we're plotting variables
			for k in xrange(nboldchain,nwalkers): ax.plot(chain[k,:,i],'-', alpha=alpha, lw=lw,zorder=-1)
			for k in xrange(nboldchain): ax.plot(chain[k,:,i],'-', alpha=alpha_bold, lw=lw_bold, color=cmap(k),zorder=1)

			ax.set_ylabel(parnames[i])
			ax.set_ylim(chain[:,:,i].min()*0.95,chain[:,:,i].max()*1.05)

		elif i == npars: # we're plotting lnprob
			for k in xrange(nboldchain,nwalkers): ax.plot(lnprob[k,:],'-', alpha=alpha, lw=lw,zorder=-1)
			for k in xrange(nboldchain): ax.plot(lnprob[k,:],'-', alpha=alpha_bold, lw=lw_bold, color=cmap(k),zorder=1)

			ax.set_ylabel('ln(probability)')
			ax.set_ylim(lnprob.min()*0.95,lnprob.max()*1.05)

		ax.xaxis.set_major_locator(MaxNLocator(4))
		ax.set_xlabel('iteration')

	### add KL divergence
	#kl_ax = fig.add_axes([0.65, 0.1, 0.27, 0.38])
	kl_ax = axarr[6]
	cmap = get_cmap(npars)
	for i in xrange(npars): kl_ax.plot(sample_results['kl_iteration'],sample_results['kl_divergence'][:,i],'o',label=parnames[i],color=cmap(i),lw=1.5,linestyle='-',alpha=0.6)

	kl_ax.set_ylabel('KL divergence')
	kl_ax.set_xlabel('iteration')
	kl_ax.set_xlim(0,nsteps*1.1)

	if legend:
		kl_ax.legend(prop={'size':5},ncol=2,numpoints=1,markerscale=0.7)

	return fig, axarr

def plot_chain():

	datname = '/Users/joel/code/python/bsfh/demo/demo_galphot_1491573461_mcmc.h5'
	sample_results, powell, model = read_results.results_from(datname)

	fig, ax = show_chain(sample_results)

	plt.savefig('test_chain.png',dpi=100)
	plt.close()

if __name__ == "__main__":
	sys.exit(plot_chain())
