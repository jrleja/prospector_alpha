import numpy as np
import matplotlib.pyplot as plt
from brown_io import load_prospector_data
from threed_dutils import generate_basenames
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

def find_subsequence(subseq, seq):
	'''stolen from convergence.py in prospect.fitting
	modified to return the appropriate index (useful to test WHERE a chain is converged)
	'''

	i, n, m = -1, len(seq), len(subseq)
	try:
		while True:
			i = seq.index(subseq[0], i + 1, n - m + 1)
			if subseq == seq[i:i + m]:
				return i+m-1  #(could return "converged" index here)
	except ValueError:
		return False

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

def load_data(filebase,label):
	filebase = filebase.replace('brownseds_agn','brownseds_longrun')
	sample_results, powell_results, model, extra_output = load_prospector_data(filebase,hdf5=True,load_extra_output=False)
	return sample_results, powell_results, model, extra_output

def kl_test(chain,labels,plt_label,fig, niter_chunk=1000, niter_check=600, nhist=30,n_stable_checks_before_stop=4, ax=None,legend=True, chainax=None):
	
	niter = chain.shape[1]
	npars = chain.shape[2]

	niter_check_start = 2*niter_chunk # must run for at least niter_chuck*2 before checking!
	ncheck = np.floor((niter-niter_check_start)/float(niter_check)).astype(int)+1

	### now calculate the K-L divergence in each chunk for each parameter
	if ax is None:
		ax = fig.add_axes([0.65, 0.1, 0.27, 0.38])
	kl = np.zeros(shape=(ncheck,npars))
	xiter = np.arange(ncheck) * niter_check + niter_check_start
	for n in xrange(ncheck):
		for i in xrange(npars):
			### define chains and calculate pdf
			true_chain = chain[:,(xiter[n]-2*niter_chunk):(xiter[n]-niter_chunk),i].flatten()
			pdf_true, bins = make_kl_bins(true_chain,nbins=nhist)
			test_chain = chain[:,(xiter[n]-niter_chunk):xiter[n],i].flatten()
			test_chain = np.clip(test_chain,bins[0],bins[-1]) # clip PDF so that it's all within the TRUE BINS (basically redefining first and last bin to have open edges)
			pdf, _ = np.histogram(test_chain,bins=bins)
			kl[n,i] = kl_divergence(pdf,pdf_true)

	### plotting
	cmap = get_cmap(npars)
	for i in xrange(npars): ax.plot(xiter,kl[:,i],'o',label=labels[i],color=cmap(i),lw=1.5,linestyle='-',alpha=0.6)

	ax.set_ylabel('KL divergence')
	ax.set_xlabel('iteration')
	ax.set_xlim(0,niter*1.1)

	if legend:
		ax.legend(prop={'size':5},ncol=2,numpoints=1,markerscale=0.7)

	### convergence
	# must repeat three times (is there an easier way to do this?)
	converged = np.all(kl < 0.018,axis=1)
	idx = find_subsequence([True]*n_stable_checks_before_stop,converged.tolist())

	if idx != -1:
		iteration = xiter[idx]
		ax.axvline(iteration,linestyle='--',color='red',zorder=0)
		if chainax is not None:
			for a in chainax: a.axvline(iteration,linestyle='--',color='red',zorder=0)
	else:
		iteration = np.nan

	return fig, iteration

def show_chain(sample_results):
	
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
	ny = 4
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
	off = [13,14,18,19]
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
	print str(nstuck)+' stuck walkers found for '+sample_results['run_params']['objname']
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

		elif i == npars: # we're plotting lnprob
			for k in xrange(nboldchain,nwalkers): ax.plot(lnprob[k,:],'-', alpha=alpha, lw=lw,zorder=-1)
			for k in xrange(nboldchain): ax.plot(lnprob[k,:],'-', alpha=alpha_bold, lw=lw_bold, color=cmap(k),zorder=1)

			ax.set_ylabel('ln(probability)')

		else: # save this panel for KL divergence
			continue

		ax.xaxis.set_major_locator(MaxNLocator(4))
		ax.set_xlabel('iteration')

	return fig, axarr

def plot_kl_with_options(plt_label='bseds',cutoff=False):
	'''this assesses the KL divergence for different bin sizes
	and different chunk sizes.
	'''

	niter_chunk = [100,200,400]
	nhist = [10,50,150]
	for u in xrange(20):
		sample_results, powell, model, extra_output = load_data(u,plt_label)

		fig, ax = plt.subplots(1,3, figsize=(15, 5))
		for i in xrange(len(niter_chunk)):
			fig = kl_test(sample_results['chain'], model.theta_labels(),plt_label, fig, ax=ax[i], niter_chunk=niter_chunk[i],legend=False)
			ax[i].text(0.95,0.95,'niter_chunk='+str(niter_chunk[i]),ha='right',transform=ax[i].transAxes)
		outname = sample_results['run_params']['objname']+'_niter.png'
		plt.tight_layout()
		plt.savefig(outname,dpi=100)
		plt.close()

		'''
		fig, ax = plt.subplots(1,3, figsize=(15, 5))
		for i in xrange(len(nhist)):
			fig = kl_test(sample_results['chain'], model.theta_labels(),plt_label, fig, ax=ax[i], nhist=nhist[i], legend=False)
			ax[i].text(0.95,0.95,'nhist='+str(nhist[i]),ha='right',transform=ax[i].transAxes)
		outname = sample_results['run_params']['objname']+'_nhist.png'
		plt.tight_layout()
		plt.savefig(outname,dpi=100)
		plt.close()
		'''
def plot_chain_with_kl(plt_label='bseds'):
	filebase,_,_ = generate_basenames('brownseds_agn')

	ngal = len(filebase)
	converged = np.zeros(ngal)
	for u in xrange(ngal):
		sample_results, powell, model, extra_output = load_data(filebase[u],plt_label)

		fig, ax = show_chain(sample_results)

		fig,converged[u] = kl_test(sample_results['chain'], model.theta_labels(),plt_label, fig,chainax=ax)

		outname = sample_results['run_params']['objname']+'_'+plt_label+'.png'
		plt.savefig(outname,dpi=100)
		plt.close()

		print sample_results['run_params']['objname'] + ': '+ str(converged[u]/2500.)

	bad = np.isnan(converged)
	print str(bad.sum()) + ' not converged'
	print 'average run length would be '+str(converged[~bad].mean()/2500.)+' of current run length'
	print 1/0

if __name__ == "__main__":
	sys.exit(main())
