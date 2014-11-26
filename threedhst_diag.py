import numpy as np
import matplotlib.pyplot as plt
import triangle
import pickle
from bsfh import read_results


def show_chain(sample_results,outname=None,alpha=0.6):
	
	'''
	plot the MCMC chain for all parameters
	'''
	
	parnames = np.array(sample_results['model'].theta_labels())
	nwalkers = sample_results['chain'].shape[0]
	nsteps = sample_results['chain'].shape[1]
	
	# geometry
	ndim = len(parnames)
	nx = 1
	ny = ndim
	sz = np.array([nx,ny])
	factor = 3.0           # size of one side of one panel
	lbdim = 0.0 * factor   # size of margins
	whspace = 0.00*factor         # w/hspace size
	plotdim = factor * sz + factor *(sz-1)* whspace
	dim = 2*lbdim + plotdim
    
	fig, axarr = plt.subplots(ny, nx, figsize = (dim[0], dim[1]))
	fig.subplots_adjust(wspace=0.000,hspace=0.000)
	
	# plot each chain in each parameter
	for jj in xrange(len(parnames)):
		for kk in xrange(nwalkers):
			axarr[jj].plot(sample_results['chain'][kk,:,jj],'-',
						   color='black',
						   alpha=alpha)

		axarr[jj].axis('tight')
		axarr[jj].set_ylabel(parnames[jj])
		if jj == len(parnames)-1:
			axarr[jj].set_xlabel('number of steps')
		else:
			axarr[jj].set_xticklabels([])
			axarr[jj].yaxis.get_major_ticks()[0].label1On = False # turn off bottom ticklabel

	if outname is not None:
		plt.savefig(outname, bbox_inches='tight',dpi=300)
		import os
		os.system("open "+outname)
		plt.close()

#def return_percentiles(sample_results, start=0, thin=1):

	#flatchain = sample_results['chain'][:,start::thin,:]

	#samples = flatchain[:, :, :].reshape((-1, ndim))
	#samples[:, 2] = np.exp(samples[:, 2])
	#m_mcmc, b_mcmc, f_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
    #                         	 zip(*np.percentile(samples, [16, 50, 84],axis=0)))

def phot_figure(sample_results, alpha=0.3, samples = [-1],
                start=0, thin=1,
                **kwargs):
	"""
	Plot the photometry for the model and data (with error bars). Then
	plot residuals
	"""
	from matplotlib import gridspec
	import fsps
	sps = fsps.StellarPopulation(zcontinuous=True)

	fig = plt.figure()
	gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
	gs.update(hspace=0)

	phot, res = plt.Subplot(fig, gs[0]), plt.Subplot(fig, gs[1])
	res.set_ylabel( r'$\chi$')
	phot.set_ylabel('maggies')
    
	# posterior draws
	flatchain = sample_results['chain'][:,start::thin,:]
	flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                  flatchain.shape[2])
	thetas = [flatchain[s,:] for s in samples]
	#mwave, mospec, mounc, specvecs = 
	mu, specvecs, delta, mask, mwave = read_results.model_comp(thetas, sample_results['model'], sps, photflag=1)
	
	#print(mwave, mospec)
	for vecs in specvecs:
		vv = vecs[0], vecs[-1]
		[ax.plot(mwave, v, color='magenta', alpha=alpha, marker='o', **kwargs)
		for ax, v in zip([phot, res], vv) ]
    
	phot.errorbar(mwave, mospec, yerr=mounc,
                  color='black')
	phot.plot(mwave, mospec, label = 'observed',
              color='black', marker='o', **kwargs)
	phot.legend(loc=0)
	res.axhline(0, linestyle=':', color='grey')
    
	fig.add_subplot(phot)
	fig.add_subplot(res)
    
	return fig

def main():

	# define filename and load basics
	file_base = 'threedhst_1416969772'
	file_base = 'threedhst_1416974087'
	filename="/Users/joel/code/python/threedhst_bsfh/results/"+file_base+"_mcmc"
	model_filename="/Users/joel/code/python/threedhst_bsfh/results/"+file_base+"_model"
	sample_results, powell_results, model = read_results.read_pickles(filename, model_file=model_filename,inmod=None)
    
    # chain plot
	show_chain(sample_results, 
			   outname='/Users/joel/code/python/threedhst_bsfh/results/'+file_base+"_chain.png")

	# triangle plot
	read_results.subtriangle(sample_results,
							 outname='/Users/joel/code/python/threedhst_bsfh/results/'+file_base,
							 showpars=None,start=0, thin=1, truths=sample_results['initial_theta'], show_titles=True)
	
	# best-fit model plot
	# sample
	nsample = 5
	ns = sample_results['chain'].shape[0] * sample_results['chain'].shape[1]
	samples = np.random.uniform(0, 1, size=nsample)
	sample = [int(s * ns) for s in samples]
 	
 	# plot
 	pfig = phot_figure(sample_results, samples=sample)
	