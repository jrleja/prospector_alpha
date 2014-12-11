import numpy as np
import matplotlib.pyplot as plt
import triangle, pickle, os, math, copy
from bsfh import read_results,model_setup

def return_mwave_custom(filters):

	"""
	returns effective wavelength based on filter names
	"""

	loc = os.getenv('APPS')+'/threedhst_bsfh/filters/'
	key_str = 'filter_keys_threedhst.txt'
	lameff_str = 'lameff_threedhst.txt'
	
	lameff = np.loadtxt(loc+lameff_str)
	keys = np.loadtxt(loc+key_str, dtype='S20',usecols=[1])
	keys = keys.tolist()
	keys = np.array([keys.lower() for keys in keys], dtype='S20')
	
	lameff_return = [[lameff[keys == filters[i]]][0] for i in range(len(filters))]
	lameff_return = [item for sublist in lameff_return for item in sublist]
	
	return lameff_return

def show_chain(sample_results,outname=None,alpha=0.6):
	
	'''
	plot the MCMC chain for all parameters
	'''
	
	# set + load variables
	parnames = np.array(sample_results['model'].theta_labels())
	nwalkers = sample_results['chain'].shape[0]
	nsteps = sample_results['chain'].shape[1]

	
	# plot geometry
	ndim = len(parnames)
	nwalkers_per_column = 32
	nx = int(math.ceil(nwalkers/float(nwalkers_per_column)))
	ny = ndim+1
	sz = np.array([nx,ny])
	factor = 3.0           # size of one side of one panel
	lbdim = 0.0 * factor   # size of margins
	whspace = 0.00*factor         # w/hspace size
	plotdim = factor * sz + factor *(sz-1)* whspace
	dim = 2*lbdim + plotdim

	fig, axarr = plt.subplots(ny, nx, figsize = (dim[0], dim[1]))
	fig.subplots_adjust(wspace=0.000,hspace=0.000)
	
	# plot chain in each parameter
	# sample_results['chain']: nwalkers, nsteps, nparams
	for ii in xrange(nx):
		walkerstart = nwalkers_per_column*ii
		walkerend   = nwalkers_per_column*(ii+1)
		for jj in xrange(len(parnames)):
			for kk in xrange(walkerstart,walkerend):
				axarr[jj,ii].plot(sample_results['chain'][kk,:,jj],'-',
						   	   color='black',
						   	   alpha=alpha)
				
			# fiddle with x-axis
			axarr[jj,ii].axis('tight')
			axarr[jj,ii].set_xticklabels([])
				
			# fiddle with y-axis
			if ii == 0:
				axarr[jj,ii].set_ylabel(parnames[jj])
			else:
				axarr[jj,ii].set_yticklabels([])
			axarr[jj,ii].set_ylim(np.amin(sample_results['chain'][:,:,jj]), np.amax(sample_results['chain'][:,:,jj]))
			axarr[jj,ii].yaxis.get_major_ticks()[0].label1On = False # turn off bottom ticklabel

		# plot lnprob
		for kk in xrange(walkerstart,walkerend): 
			axarr[jj+1,ii].plot(sample_results['lnprobability'][kk,:],'-',
						      color='black',
						      alpha=alpha)
		# axis
		axarr[jj+1,ii].axis('tight')
		axarr[jj+1,ii].set_xlabel('number of steps')
		# fiddle with y-axis
		if ii == 0:
			axarr[jj+1,ii].set_ylabel('lnprob')
		else:
			axarr[jj+1,ii].set_yticklabels([])

		testable = np.isfinite(sample_results['lnprobability'])
		max = np.amax(sample_results['lnprobability'][testable])
		stddev = np.std(sample_results['lnprobability'][testable])
		axarr[jj+1,ii].set_ylim(max-stddev, max)
		
		axarr[jj+1,ii].yaxis.get_major_ticks()[0].label1On = False # turn off bottom ticklabel


	if outname is not None:
		plt.savefig(outname, bbox_inches='tight',dpi=300)
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

        specvecs += [ [mu, cal, delta, mu,mospec/mu, (mospec-mu) / mounc] ]
            
    return wave, mospec, mounc, specvecs	

def sed_figure(sample_results, alpha=0.3, samples = [-1],
                start=0, thin=1, maxprob=0, outname=None, plot_init = 0,
                parm_file = os.getenv('APPS')+'/threedhst_bsfh/parameter_files/threedhst_params.py',
                chop_chain=1,**kwargs):
	"""
	Plot the photometry for the model and data (with error bars), and
	plot residuals
	"""
	c = 3e8
	
	from matplotlib import gridspec
	import fsps
	
	# load stellar population, set up custom filters
	sps = fsps.StellarPopulation(zcontinuous=True, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load parameter file
	model = model_setup.setup_model(parm_file, sps=sps)

	# set up plot
	fig = plt.figure()
	gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
	gs.update(hspace=0)
	phot, res = plt.Subplot(fig, gs[0]), plt.Subplot(fig, gs[1])

	# FLATTEN AND CHOP CHAIN
	# chain = chain[nwalkers,nsteps,ndim]
	chopped_chain = sample_results['chain'][:,int(sample_results['chain'].shape[1]/chop_chain):,:]
	flatchain = chopped_chain[:,start::thin,:]
	flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],flatchain.shape[2])

	# MAKE RANDOM POSTERIOR DRAWS
	nsample = 5
	ns = flatchain.shape[0] * flatchain.shape[1]
	samples = np.random.uniform(0, 1, size=nsample)
	sample = [int(s * ns) for s in samples]
		
	thetas = [flatchain[s,:] for s in samples]
	mwave, mospec, mounc, specvecs = comp_samples(thetas, sample_results['model'], sps, photflag=1)

	# define observations
	xplot = np.log10(mwave)
	yplot = np.log10(mospec*(c/(mwave/1e10)))
	linerr_down = np.clip(mospec-mounc, 1e-80, 1e80)*(c/(mwave/1e10))
	linerr_up = np.clip(mospec+mounc, 1e-80, 1e80)*(c/(mwave/1e10))
	yerr = [yplot - np.log10(linerr_down), np.log10(linerr_up)-yplot]

	# set up plot limits
	phot.set_xlim(min(xplot)*0.96,max(xplot)*1.04)
	phot.set_ylim(min(yplot[np.isfinite(yplot)])*0.96,max(yplot[np.isfinite(yplot)])*1.04)
	res.set_xlim(min(xplot)*0.96,max(xplot)*1.04)

	# PLOT RANDOM DRAWS
	for vecs in specvecs:		
		phot.plot(np.log10(mwave), np.log10(vecs[0]*(c/(mwave/1e10))), color='grey', alpha=alpha, marker='o', label='posterior sample', **kwargs)
		res.plot(np.log10(mwave), vecs[-1], color='grey', alpha=alpha, marker='o', **kwargs)
		
    # PLOT OBSERVATIONS + ERRORS 
	phot.errorbar(xplot, yplot, yerr=yerr,
                  color='black', marker='o', label='observed')

    # PLOT INITIAL PARAMETERS
	if plot_init:
		observables = model.mean_model(sample_results['initial_theta'], sps=sps)
		fast_spec, fast_mags = observables[0],observables[1]
		fast_lam = model.obs['wave_effective']
		phot.plot(np.log10(fast_lam), np.log10(fast_mags*(c/(fast_lam/1e10))), label = 'initial parms', linestyle=' ',color='blue', marker='o', **kwargs)
    
		#nz = fast_spec > 0
		#phot.plot(np.log10(w[nz]), np.log10(fast_spec[nz]*(c/(w[nz]/1e10))), label = 'init fit (spec)',
    	#          color='blue', **kwargs)
	
	# plot max probability
	maxprob = np.max(sample_results['lnprobability'])
	probind = sample_results['lnprobability'] == maxprob
	thetas = sample_results['chain'][probind,:]

	print 'maxprob: ' + str(maxprob)

	if thetas.ndim == 2:			# if multiple chains found the same peak, choose one arbitrarily
		thetas=[thetas[0,:]]
	else:
		thetas = [thetas]
	mwave, mospec, mounc, specvecs = comp_samples(thetas, sample_results['model'], sps, photflag=1)
		
	phot.plot(np.log10(mwave), np.log10(specvecs[0][0]*(c/(mwave/1e10))), color='red', marker='o', label='max lnprob', **kwargs)
	res.plot(np.log10(mwave), specvecs[0][-1], color='red', marker='o', label='max lnprob', **kwargs)
		
	# diagnostic text
	textx = (phot.get_xlim()[1]-phot.get_xlim()[0])*0.975+phot.get_xlim()[0]
	texty = (phot.get_ylim()[1]-phot.get_ylim()[0])*0.2+phot.get_ylim()[0]
	deltay = (phot.get_ylim()[1]-phot.get_ylim()[0])*0.04

	phot.text(textx, texty, r'best-fit $\chi^2$='+"{:.2f}".format(np.sum(specvecs[0][-1]**2)),
			  fontsize=12, ha='right')
	phot.text(textx, texty-deltay, r'avg acceptance='+"{:.2f}".format(np.mean(sample_results['acceptance'])),
				 fontsize=12, ha='right')
		
	# load fast data
	filename=model.run_params['fastname']
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	fields = [f for f in dat.dtype.names]
	id_ind = fields.index('id')
	uvj_ind = fields.index('uvj_flag')
	sn_ind = fields.index('sn_F160W')
	z_ind = fields.index('z')
	z_txt = [f[z_ind] for f in dat if f[id_ind] == float(model.run_params['objname'])][0]
	uvj_txt = [f[uvj_ind] for f in dat if f[id_ind] == float(model.run_params['objname'])][0]
	sn_txt = [f[sn_ind] for f in dat if f[id_ind] == float(model.run_params['objname'])][0]
		
	# galaxy text
	phot.text(textx, texty-2*deltay, 'z='+"{:.2f}".format(z_txt),
			  fontsize=12, ha='right')
	phot.text(textx, texty-3*deltay, 'uvj_flag='+str(uvj_txt),
			  fontsize=12, ha='right')
	phot.text(textx, texty-4*deltay, 'S/N(F160W)='+"{:.2f}".format(sn_txt),
			  fontsize=12, ha='right')
		
		
	# extra line
	res.axhline(0, linestyle=':', color='grey')
	
	# legend
	# make sure not to repeat labels
	from collections import OrderedDict
	handles, labels = phot.get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))
	phot.legend(by_label.values(), by_label.keys(), 
				loc=1, prop={'size':8},
			    frameon=False)
			    
    # set labels
	res.set_ylabel( r'$\chi$')
	phot.set_ylabel(r'log($\nu f_{\nu}$)')
	res.set_xlabel(r'log($\lambda_{obs}$) [$\AA$]')
	
	# remove ticks
	phot.set_xticklabels([])
    
    # clean up and output
	fig.add_subplot(phot)
	fig.add_subplot(res)
	if outname is not None:
		fig.savefig(outname, bbox_inches='tight', dpi=300)
		plt.close()

def make_all_plots(objname, parm_file=None, 
				   outfolder=os.getenv('APPS')+'/threedhst_bsfh/plots/',
				   infolder =os.getenv('APPS')+'/threedhst_bsfh/results/'):

	'''
	Driver. Loads output, makes all plots.
	'''

	plt_chain_figure = 1
	plt_triangle_plot = 1
	plt_sed_figure = 1
	
	# thin and chop the chain?
	thin=2
	chop_chain=2

	# find most recent output file
	# with the objname
	files = [ f for f in os.listdir(infolder) if f[-4:] == 'mcmc' ]
	times = [f.split('_')[2] for f in files if f.split('_')[1] == str(objname)]
	file_base = 'threedhst_'+str(objname)+'_'+max(times)

	# load results
	filename=infolder+file_base+"_mcmc"
	model_filename=infolder+file_base+"_model"
	sample_results, powell_results, model = read_results.read_pickles(filename, model_file=model_filename,inmod=None)
	
    # chain plot
	if plt_chain_figure: 
		print 'MAKING CHAIN PLOT'
		show_chain(sample_results,
	                 outname=outfolder+file_base+"_chain.png",
			         alpha=0.3)

	# triangle plot
	if plt_triangle_plot: 
		print 'MAKING TRIANGLE PLOT'
		chopped_sample_results = copy.deepcopy(sample_results)
		chopped_sample_results['chain'] = sample_results['chain'][:,int(sample_results['chain'].shape[1]/chop_chain):,:]

		read_results.subtriangle(chopped_sample_results,
							 outname=outfolder+file_base,
							 showpars=None,start=0, thin=thin, #truths=sample_results['initial_theta'],
							 show_titles=True)

	# sed plot
	if plt_sed_figure:
		print 'MAKING SED PLOT'
 		# plot
 		pfig = sed_figure(sample_results, 
 						  maxprob=1, 
 						  outname=outfolder+file_base+'_sed.png',
 						  parm_file=parm_file,
 						  thin=thin,
 						  chop_chain=chop_chain)
 		

	
	
	
	
	
	
	
	
	





''' TEST PLOT PLEASE IGNORE '''
def sed_test_plot():
	
	"""
	Plot the photometry+spectra for a variety of ages, etc
	"""
	
	
	import fsps
	
	c = 3e8
	
	sps = fsps.StellarPopulation(zcontinuous=True, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load custom model
	model = model_setup.setup_model(os.getenv('APPS')+'/threedhst_bsfh/threedhst_params.py', sps=sps)
	
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