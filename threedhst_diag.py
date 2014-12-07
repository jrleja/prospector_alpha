import numpy as np
import matplotlib.pyplot as plt
import triangle, pickle, os
from bsfh import read_results,model_setup
from threedhst_params import return_mwave_custom


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
		os.system("open "+outname)
		plt.close()

def comp_samples(thetas, model, sps, inlog=True, photflag=0):
    specvecs =[]
    obs, _, marker = read_results.obsdict(model.obs, photflag)
    wave, ospec, mask = obs['wave_effective'], obs['spectrum'], obs['mask']
    mwave, mospec = wave[mask], ospec[mask]
    mounc = obs['maggies_unc'][mask]

    if inlog and (photflag == 0):
         mospec = np.exp(mospec)
         mounc *= mospec

    for theta in thetas:
        mu, cal, delta, mask, wave = read_results.model_comp(theta, model, sps,
                                           inlog=True, photflag=1)

        if inlog & (photflag == 0):
            full_cal = np.exp(cal + delta)
            mod = np.exp(mu + cal + delta)
            mu = np.exp(mu)
            cal = np.exp(cal)
            
        elif photflag == 0:
            full_cal = cal + delta/mu
            mod = (mu*cal + delta)

        else:
            mod = mu

        specvecs += [ [mu, cal, delta, mod,mospec/mod, (mospec-mod) / mounc] ]
            
    return wave, mospec, mounc, specvecs

def phot_figure(sample_results, alpha=0.3, samples = [-1],
                start=0, thin=1, maxprob=0, outname=None,
                **kwargs):
	"""
	Plot the photometry for the model and data (with error bars). Then
	plot residuals
	"""
	c = 3e8
	
	from matplotlib import gridspec
	import fsps
	sps = fsps.StellarPopulation(zcontinuous=True, compute_vega_mags=False)
	custom_filter_keys = '$APPS/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load custom model
	model = model_setup.setup_model('$APPS/threedhst_bsfh/threedhst_params.py', sps=sps)
	observables = model.mean_model(sample_results['initial_theta'], sps=sps)
	
	fast_spec, fast_mags = observables[0],observables[1]
	w, spec_throwaway = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
	fast_lam = model.obs['wave_effective']
	
	fig = plt.figure()
	gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
	gs.update(hspace=0)

	phot, res = plt.Subplot(fig, gs[0]), plt.Subplot(fig, gs[1])

	# random posterior draws
	# chain = chain[nwalkers,nsteps,ndim]
	flatchain = sample_results['chain'][:,start::thin,:]
	flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],flatchain.shape[2])	
	thetas = [flatchain[s,:] for s in samples]
	mwave, mospec, mounc, specvecs = comp_samples(thetas, sample_results['model'], sps, photflag=1)
	
	for vecs in specvecs:
		vv = vecs[0], vecs[-1]

		#[ax.plot(np.log10(mwave), np.log10(v), color='grey', alpha=alpha, marker='o', label='random sample', **kwargs)
		#for ax, v in zip([phot, res], vv) ]
    
	phot.errorbar(np.log10(mwave), np.log10(mospec*(c/(mwave/1e10))), yerr=mounc,
                  color='black')
	phot.plot(np.log10(mwave), np.log10(mospec*(c/(mwave/1e10))), label = 'observed',
              color='black', marker='o', **kwargs)
    
	phot.plot(np.log10(fast_lam), np.log10(fast_mags*(c/(fast_lam/1e10))), label = 'FAST fit (phot)', linestyle=' ',color='blue', marker='o', **kwargs)
    
	nz = fast_spec > 0
	phot.plot(np.log10(w[nz]), np.log10(fast_spec[nz]*(c/(w[nz]/1e10))), label = 'FAST fit (spec)',
              color='blue', **kwargs)
	
	# max probability
	if maxprob == 1:
		thetas = sample_results['chain'][sample_results['lnprobability'] == np.max(sample_results['lnprobability']),:]
		if thetas.ndim == 2:			# if multiple chains found the same peak, choose one arbitrarily
			thetas=[thetas[0,:]]
		else:
			thetas = [thetas]
		mwave, mospec, mounc, specvecs = comp_samples(thetas, sample_results['model'], sps, photflag=1)
		
		for vecs in specvecs:
			phot.plot(np.log10(mwave), np.log10(vecs[0]*(c/(mwave/1e10))), color='red', marker='o', label='max lnprob', **kwargs)
			res.plot(np.log10(mwave), vecs[-1], color='red', marker='o', label='max lnprob', **kwargs)
	
	# legend and axes
	phot.legend(loc=0, prop={'size':8},
			    frameon=False)
	res.axhline(0, linestyle=':', color='grey')
    
	res.set_ylabel( r'$\chi$')
	phot.set_ylabel(r'log($\nu f_{\nu}$)')
	res.set_xlabel(r'log($\lambda_{obs}$) [$\AA$]')
	phot.set_xlim(3.5,5.2)
	phot.set_ylim(4,6.5)
	res.set_xlim(3.5,5.2)
	phot.set_xticklabels([])
    
	fig.add_subplot(phot)
	fig.add_subplot(res)
	if outname is not None:
		fig.savefig(outname, bbox_inches='tight', dpi=300)
		os.system("open "+outname)
		plt.close()
	return fig

def sed_test_plot():
	
	"""
	Plot the photometry+spectra for a variety of ages, etc
	"""
	
	
	import fsps
	
	c = 3e8
	
	sps = fsps.StellarPopulation(zcontinuous=True, compute_vega_mags=False)
	custom_filter_keys = '$APPS/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load custom model
	model = model_setup.setup_model('$APPS/threedhst_bsfh/threedhst_params.py', sps=sps)
	
	# setup figure
	fig, axarr = plt.subplots(2, 2, figsize = (8,8))
	fig.subplots_adjust(wspace=0.000,hspace=0.000)
	fast_lam = model.obs['wave_effective']
	init_theta = np.array([10.5, 0, 0, 0.0])
	
	# generate colors
	import pylab
	NUM_COLORS = 10
	cm = pylab.get_cmap('cool')
	plt.rcParams['axes.color_cycle'] = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)] 
	axlim=[3.0,5.5,3.0,8]
	
	# setup delta
	delta = [np.linspace(init_theta[0]-1,init_theta[0]+1, NUM_COLORS),
			 np.linspace(init_theta[1]-0.9,init_theta[1]+1, NUM_COLORS),
			 np.linspace(init_theta[2]-0.9,init_theta[2]+1, NUM_COLORS),
			 np.linspace(init_theta[3]-1,init_theta[3]+0.3, NUM_COLORS)]

	for kk in range(4):
		itone = kk % 2
		ittwo = kk > 1
	
		for jj in range(len(delta[kk])):

			# set model parms
			model_params = np.copy(init_theta)
			model_params[kk] = delta[kk][jj]

			# load data
			observables = model.mean_model(10**model_params, sps=sps)
			fast_spec, fast_mags = observables[0],observables[1]
			w, spec_throwaway = sps.get_spectrum(tage=sps.params['tage'], peraa=False)
    
			nz = fast_spec > 0
			axarr[itone,ittwo].plot(np.log10(w[nz]), np.log10(fast_spec[nz]*(c/(w[nz]/1e10))),label = "{:10.2f}".format(model_params[kk]))
	
		# beautify
		if itone == 1:
			axarr[itone,ittwo].set_xlabel('log(lam)')
		else:
			axarr[itone,ittwo].set_xticklabels([])
			axarr[itone,ittwo].yaxis.get_major_ticks()[0].label1On = False # turn off bottom ticklabel
	
		if ittwo == 0:
			axarr[itone,ittwo].set_ylabel('log(nu*fnu)')
		else:
			axarr[itone,ittwo].set_yticklabels([])
			axarr[itone,ittwo].xaxis.get_major_ticks()[0].label1On = False # turn off bottom ticklabel
	
		axarr[itone,ittwo].legend(loc=0,prop={'size':6},
								  frameon=False,
								  title='log('+str(model.theta_labels()[kk])+')')
		axarr[itone,ittwo].get_legend().get_title().set_size(8)
		axarr[itone,ittwo].axis(axlim)

def make_all_plots(objname, folder='results/'):

	plt_phot_figure = True
	plt_chain_figure = True
	plt_triangle_plot = True

	# find most recent file
	files = [ f for f in os.listdir(folder) if f[-4:] == 'mcmc' ]
	times = [f.split('_')[2] for f in files if f.split('_')[1] == str(objname)]
	file_base = 'threedhst_'+str(objname)+'_'+max(times)

	# load results
	filename=folder+file_base+"_mcmc"
	model_filename=folder+file_base+"_model"
	sample_results, powell_results, model = read_results.read_pickles(filename, model_file=model_filename,inmod=None)
	
    # chain plot
	if plt_chain_figure: show_chain(sample_results,
	                 outname=folder+file_base+"_chain.png",
			         alpha=0.3)

	# triangle plot
	if plt_triangle_plot: read_results.subtriangle(sample_results,
							 outname=folder+file_base,
							 showpars=None,start=0, thin=1, truths=sample_results['initial_theta'], show_titles=True)

	if plt_phot_figure:
		# best-fit model plot
		# sample
		nsample = 5
		ns = sample_results['chain'].shape[0] * sample_results['chain'].shape[1]
		samples = np.random.uniform(0, 1, size=nsample)
		sample = [int(s * ns) for s in samples]

 		# plot
 		pfig = phot_figure(sample_results, samples=sample, maxprob=1, outname=folder+file_base+'_sed.png')
 		

	