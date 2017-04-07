import acor
import numpy as np
import matplotlib.pyplot as plt
from brown_io import load_prospector_data
from threed_dutils import generate_basenames
import sys
import magphys_plot_pref
import pymc
from matplotlib.ticker import MaxNLocator

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

'''
USE K-L criteria to evaluate convergence: https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
	-- calculate integral for each parameter, plot as iteration number
	-- more complex version of the mean + 1sigma point estimators
charlie: likelihood surface? what is the volume of the space, and how does it change with niter?
'''

'''
figure out how to do smart binning
then plot all of the chains with the KL_DIVERGENCE plot below
two-column chain plots, with most chains diffuse but a few highlighted (10?)
'''

nburn = 0
nchunk = 15 # how many places in niter to estimate convergence, etc

def get_cmap(N):
	'''Returns a function that maps each index in 0, 1, ... N-1 to a distinct 
	RGB color.'''

	import matplotlib.cm as cmx
	import matplotlib.colors as colors

	color_norm  = colors.Normalize(vmin=0, vmax=N-1)
	scalar_map = cmx.ScalarMappable(norm=color_norm, cmap='nipy_spectral') 
	def map_index_to_rgb_color(index):
		return scalar_map.to_rgba(index)
	return map_index_to_rgb_color

def load_data(n,label):
	filebase,_,_ = generate_basenames('brownseds_agn')
	filebase[n] = filebase[n].replace('brownseds_agn','brownseds_longrun')
	if 'demo' in label:
		filebase[n] = '/Users/joel/code/python/bsfh/demo/demo_obj0'
	sample_results, powell_results, model, extra_output = load_prospector_data(filebase[n],hdf5=True,load_extra_output=False)
	return sample_results, powell_results, model, extra_output

def pandas_autocorr(chain, labels, plt_label):

	from pandas.tools.plotting import autocorrelation_plot # autocorrelation plot

	npars = chain.shape[1]
	cmap = get_cmap(npars)

	# plot autocorrelation lag
	plt.figure(figsize=(16,6))
	h = [autocorrelation_plot(chain[nburn:,i], color=cmap(i), lw=5, alpha=0.8, label=labels[i])
	     for i in xrange(npars)]

	plt.legend(loc=1, fontsize=14,ncol=3,numpoints=1,markerscale=0.7)
	plt.tight_layout()
	plt.savefig('pandas_autocorrelation_'+plt_label+'.png',dpi=150)
	plt.close()

def autocorrelation(chain,labels,plt_label):

	npars = chain.shape[1]
	maxlags = chain.shape[0]/5.
	nlags = 100
	lags = np.linspace(1,maxlags,nlags).astype(int)

	tau = np.zeros(shape=(lags.shape[0],npars))
	for l, lag in enumerate(lags):
		#print('maxlag:{}').format(lag)
		print l
		for i in xrange(npars):
			tau[l,i] = acor.acor(chain[:,i], maxlag=lag)[0]
			#print('\t '+labels[i]+': {0}'.format(tau[l,i]))

	### emcee version
	from emcee import autocorr
	c = 10
	good = False
	while good == False:
		try:
			emcee_tau = autocorr.integrated_time(chain, c=c)
			good = True
		except:
			if c > 2:
				c -= 0.5
			else:
				c = c ** 0.95
		if (c-1) < 1e-3:
			print 'FAILED TO CALCULATE AUTOCORRELATION TIMES'
			emcee_tau = np.zeros(len(labels))
			break

	print 'AUTOCORRELATION LENGTHS'
	for r, l in zip(emcee_tau,labels): print l+': '+"{:.2f}".format(r)

	### plotting
	fig, ax = plt.subplots(1,1, figsize=(8,8))
	cmap = get_cmap(npars)
	for i in xrange(npars):
		ax.plot(lags,tau[:,i],label=labels[i]+'='+"{:.2f}".format(emcee_tau[i]),color=cmap(i),lw=2)

	ax.set_xlabel('lag')
	ax.set_ylabel('autocorrelation')

	ax.legend(prop={'size':10},title='autocorrelation lengths',ncol=npars / 5,numpoints=1,markerscale=0.7)

	fig.tight_layout()
	plt.savefig('autocorrelation_time_'+plt_label+'.png',dpi=150)
	plt.close()

def gelman_rubin(chain,labels,plt_label, nchunk=10):

	nparameters = chain.shape[2]
	niter = chain.shape[1]
	gelman_rubin_r = np.zeros(shape=(nchunk,nparameters))
	niter_plot = np.zeros(nchunk,dtype='int')
	for i in xrange(nchunk):
		niter_plot[i] = int(niter/nchunk)*i+int(niter/nchunk)
		gelman_rubin_r[i,:] = pymc.gelman_rubin(chain[:,:niter_plot[i],:])

	### plotting
	fig, ax = plt.subplots(1,1, figsize=(8,8))
	cmap = get_cmap(nparameters)
	for i in xrange(nparameters):
		ax.plot(niter_plot,gelman_rubin_r[:,i],'o',label=labels[i],color=cmap(i),lw=1.5,linestyle='-',alpha=0.6)

	ax.set_ylabel('Gelman-Rubin R')
	ax.set_xlabel('iteration')

	### limits
	min, max = gelman_rubin_r.min(), gelman_rubin_r.max()
	ax.set_ylim(min*0.95,max*1.05)

	if max > 10:
		ax.set_yscale('log',nonposy='clip',subsy=(1,2,4))
		ax.yaxis.set_minor_formatter(minorFormatter)
		ax.yaxis.set_major_formatter(majorFormatter)

	ax.axhline(1.2, linestyle='--', color='red',lw=1,zorder=-1)

	ax.legend(prop={'size':10},ncol=nparameters / 5,numpoints=1,markerscale=0.7)

	fig.tight_layout()
	plt.savefig('gelman_rubin_'+plt_label+'.png',dpi=150)
	plt.close()

def intervals(chain,labels,plt_label):

	### calculate
	niter = chain.shape[1]
	nparameters = chain.shape[2]
	intervals = np.zeros(shape=(nchunk,nparameters,5))
	niter_plot = np.zeros(nchunk,dtype='int')
	delta = int(niter/nchunk)
	for i in xrange(nchunk):
		niter_plot[i] = delta*i+delta
		for n in xrange(nparameters):
			fchain = chain[:,delta*i:niter_plot[i],n].flatten()
			intervals[i,n,:] = np.percentile(fchain,[50.0,84.0,16.0,5.0,95.0])

	### plot
	nrows = nparameters / 5
	fig, ax = plt.subplots(nrows,5, figsize=(25,4*(nrows)))
	ax = np.ravel(ax)
	for n in xrange(nparameters):
		ax[n].plot(niter_plot,intervals[:,n,0],lw=3,linestyle='-',alpha=0.9,color='k')
		ax[n].plot(niter_plot,intervals[:,n,1],lw=2,linestyle='--',alpha=0.9,color='k')
		ax[n].plot(niter_plot,intervals[:,n,2],lw=2,linestyle='--',alpha=0.9,color='k')
		ax[n].plot(niter_plot,intervals[:,n,3],lw=1.5,linestyle=':',alpha=0.9,color='k')
		ax[n].plot(niter_plot,intervals[:,n,4],lw=1.5,linestyle=':',alpha=0.9,color='k')
		ax[n].set_xlabel('iteration')
		ax[n].set_ylabel(labels[n])
		ax[n].xaxis.set_major_locator(MaxNLocator(4))
	plt.tight_layout()
	plt.savefig('interval_'+plt_label+'.png',dpi=150)
	plt.close()

def kl_calc(pdf1, pdf2):
	# calculates Kullback-Leibler (KL) divergence
	# https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

	pdf1 = pdf1/float(pdf1.sum())
	pdf2 = pdf2/float(pdf2.sum())
	dl = pdf1 * np.log(pdf1/pdf2)

	### set areas where there is no target density to zero
	no_density = (pdf1 == 0)
	dl[no_density] = 0

	return dl.sum()

def make_kl_bins(chain, nbins=10):

	### make regular bins
	#pdf,bins = np.histogram(chain,bins=nbins)

	### create bins with an equal number of data points in them 
	# when there are empty bins, divergence is undefined
	# this avoids that problem
	sorted = np.sort(chain)
	nskip = np.floor(chain.shape[0]/float(nbins)).astype(int)-1
	bins = sorted[::nskip]
	bins[-1] = sorted[-1] # ensure all points are in the histogram
	assert bins.shape[0] == nbins+1
	pdf,bins = np.histogram(chain,bins=bins)

	'''
	if np.any(pdf == 0):
		sorted = np.sort(chain)
		bins = sorted[::chain.shape[0]/nbins]
		bins[-1] = sorted[-1]
		pdf,bins = np.histogram(chain,bins=bins)
	'''
	return pdf,bins



def kl_test(chain,labels,plt_label):
	
	niter = chain.shape[1]
	npars = chain.shape[2]
	nhist = 50 # how many histograms to split the probability into

	delta = int(niter/nchunk)

	### first assemble the "true" PDF (i.e. the last chunk) 
	### and the corresponding bins 
	pdf_true, bins = [],[]
	for i in xrange(npars): 
		tpdf, tbins = make_kl_bins(chain[:,-delta:,i].flatten(),nbins=nhist)
		bins.append(tbins)
		pdf_true.append(tpdf)

	### now calculate the K-L divergence in each chunk for each parameter
	kl = np.zeros(shape=(nchunk,npars))
	niter_plot = np.zeros(nchunk,dtype='int')
	for n in xrange(nchunk):
		niter_plot[n] = delta*n+delta
		for i in xrange(npars):
			fchain = chain[:,(niter_plot[n]-delta):niter_plot[n],i].flatten()
			fchain = np.clip(fchain,bins[i][0],bins[i][-1]) # clip PDF so that it's all within the TRUE BINS (basically redefining first and last bin to have open edges)
			pdf, _ = np.histogram(fchain,bins=bins[i])
			kl[n,i] = kl_calc(pdf,pdf_true[i])

	### plotting
	fig, ax = plt.subplots(1,1, figsize=(8,8))
	cmap = get_cmap(npars)
	for i in xrange(npars):
		ax.plot(niter_plot,kl[:,i],'o',label=labels[i],color=cmap(i),lw=1.5,linestyle='-',alpha=0.6)

	ax.set_ylabel('KL divergence')
	ax.set_xlabel('iteration')

	ax.legend(prop={'size':12},ncol=npars / 5,numpoints=1,markerscale=0.7)

	fig.tight_layout()
	plt.savefig('kl_divergence_'+plt_label+'.png',dpi=150)
	plt.close()

def plot(plt_label='brownseds_agn_longrun',cutoff=False):

	sample_results, powell, model, extra_output = load_data(0,plt_label)
	chain = sample_results['chain'][:,nburn:,:]
	flatchain = chain.mean(axis=0)

	if cutoff:
		cut = 500
		chain = chain[:,:cut,:]
		flatchain = flatchain[:cut,:]

	#### autocorrelation
	# acorr and emcee
	#autocorrelation(all_walkers, model.theta_labels(),plt_label)
	kl_test(chain, model.theta_labels(),plt_label)
	print 1/0
	pandas_autocorr(flatchain, model.theta_labels(),plt_label)

	#### convergence diagnostics
	intervals(chain,model.theta_labels(),plt_label)
	gelman_rubin(chain,model.theta_labels(),plt_label)

if __name__ == "__main__":
	sys.exit(main())
