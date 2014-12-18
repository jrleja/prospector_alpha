import numpy as np
import matplotlib.pyplot as plt
import triangle, pickle, os, math, copy
from bsfh import read_results,model_setup
import matplotlib.image as mpimg
from astropy.cosmology import WMAP9
import fsps

def integrate_sfh(t1,t2,tage,tau,sf_start,tburst,fburst):
	
	if t2 < sf_start:
		return 0.0
	elif t2 > tage:
		return 1.0
	else:
		intsfr = (np.exp(-t1/tau)*(1+t1/tau) - 
		          np.exp(-t2/tau)*(1+t2/tau))
		norm=(1.0-fburst)/(np.exp(-sf_start/tau)*(sf_start/tau+1)-
				  np.exp(-tage    /tau)*(tage    /tau+1))
		intsfr=intsfr*norm
		
	if t2 > tburst:
		intsfr=intsfr+fburst

	return intsfr

def plot_sfh(sfh_parms):

	'''
	create sfh for plotting purposes
	input: str_sfh_parms = ['tage','tau','tburst','fburst','sf_start']
	'''
	tage = sfh_parms[0]
	tau = sfh_parms[1]
	tburst = sfh_parms[2]
	fburst = sfh_parms[3]
	sf_start = sfh_parms[4]

	#tage = 1.2*tuniv
	# evenly spaced times for output
	t=np.linspace(0,tage,num=50)
	
	# integrated te^-t/tau, with t = t-tstart
	# dropped normalization in eqn
	# re-normalize at end
	intsfr = np.zeros(len(t))
	for jj in xrange(len(t)): intsfr[jj] = integrate_sfh(sf_start,t[jj],tage,tau,sf_start,tburst,fburst)

	return t, intsfr

def create_plotquant(sample_results, logplot = ['mass', 'tau', 'tage', 'tburst', 'sf_start']):
    
    '''
    creates plottable versions of chain and sets up new plotnames
    '''
    
    # set up plot chain, parnames
    plotchain = copy.deepcopy(sample_results['chain'])
    parnames = np.array(sample_results['model'].theta_labels())
    
    # properly define timescales in certain parameters
    # will have to redefine after inserting p(z)
    # note that we switch prior min/max here!!
    tuniv = WMAP9.age(sample_results['model'].config_list[0]['init']).value*1.2
    redefine = ['tburst','sf_start']
    for par in redefine:

    	priors = [f['prior_args'] for f in sample_results['model'].config_list if f['name'] == par][0]

    	min = priors['mini']
    	max = priors['maxi']
    	priors['mini'] = np.clip(tuniv-max,1e-10,1e80)
    	priors['maxi'] = tuniv-min

    	plotchain[:,:,list(parnames).index(par)] = tuniv - plotchain[:,:,list(parnames).index(par)]

	# define plot quantities and plot names
	# primarily converting to log or not
    if logplot is not None:
    	
    	# if we're interested in logging it...
    	# change the plotname, chain values, and priors
    	plotnames=[]
    	for ii in xrange(len(parnames)): 
    		if parnames[ii] in logplot: 
    			plotnames.append('log'+parnames[ii])
    			plotchain[:,:,ii] = np.log10(plotchain[:,:,ii])
    			priors = [f['prior_args'] for f in sample_results['model'].config_list if f['name'] == parnames[ii]][0]
    			for k,v in priors.iteritems(): priors[k]=np.log10(v)
    		else:
    			plotnames.append(parnames[ii])
    else:
    	plotnames = [f for f in parnames]
    
    sample_results['plotnames'] = plotnames
    sample_results['plotchain'] = plotchain

    return sample_results

def return_extent(sample_results):    
    
	'''
	sets plot range for chain plot and triangle plot for each parameter
	'''
    
	# set range
	extents = []
	parnames = np.array(sample_results['model'].theta_labels())
	for ii in xrange(len(parnames)):
		
		# set min/max
		extent = [np.min(sample_results['plotchain'][:,:,ii]),np.max(sample_results['plotchain'][:,:,ii])]
		
		# is the chain stuck at one point? if so, set the range equal to the prior range
		# else check if we butting up against the prior? if so, extend by 10%
		priors = [f['prior_args'] for f in sample_results['model'].config_list if f['name'] == parnames[ii]][0]
		if extent[0] == extent[1]:
			extent = (priors['mini'],priors['maxi'])
		else:
			extend = (extent[1]-extent[0])*0.12
			if np.abs((priors['mini']-extent[0])/priors['mini']) < 1e-4:
				extent[0]=extent[0]-extend
			if np.abs((priors['maxi']-extent[1])/priors['maxi']) < 1e-4:
				extent[1]=extent[1]+extend
    	
		extents.append((extent[0],extent[1]))
	
	return extents

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

def load_ancil_data(filename,objnum):

	'''
	loads ancillary plotting information
	'''
	
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	fields = [f for f in dat.dtype.names]
	id_ind = fields.index('id')

	# search for ID, pull out object info
	obj_ind = [int(x[id_ind]) for x in dat].index(int(objnum))
	values = dat[fields].view(float).reshape(len(dat),-1)[obj_ind]

	return values, fields

def load_obs_3dhst(filename, objnum, min_error = None):
	"""
	Load 3D-HST photometry file, return photometry for a particular object.
	min_error: set the minimum photometric uncertainty to be some fraction
	of the flux. if not set, use default errors.
	"""
	obs ={}
	fieldname=filename.split('/')[-1].split('_')[0].upper()
	with open(filename, 'r') as f:
		hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',
					 dtype = np.dtype([(n, np.float) for n in hdr[1:]]))
	obj_ind = np.where(dat['id'] == int(objnum))[0][0]
	
	# extract fluxes+uncertainties for all objects and all filters
	flux_fields = [f for f in dat.dtype.names if f[0:2] == 'f_']
	unc_fields = [f for f in dat.dtype.names if f[0:2] == 'e_']
	filters = [f[2:] for f in flux_fields]

	# extract fluxes for particular object, converting from record array to numpy array
	flux = dat[flux_fields].view(float).reshape(len(dat),-1)[obj_ind]
	unc  = dat[unc_fields].view(float).reshape(len(dat),-1)[obj_ind]

	# define all outputs
	filters = [filter.lower()+'_'+fieldname.lower() for filter in filters]
	wave_effective = np.array(return_mwave_custom(filters))
	phot_mask = np.logical_or(np.logical_or((flux != unc),(flux > 0)),flux != -99.0)
	maggies = flux/(10**10)
	maggies_unc = unc/(10**10)
	
	# set minimum photometric error
	if min_error is not None:
		under = maggies_unc < min_error*maggies
		maggies_unc[under] = min_error*maggies[under]
	
	# sort outputs based on effective wavelength
	points = zip(wave_effective,filters,phot_mask,maggies,maggies_unc)
	sorted_points = sorted(points)

	# build output dictionary
	obs['wave_effective'] = np.array([point[0] for point in sorted_points])
	obs['filters'] = np.array([point[1] for point in sorted_points])
	obs['phot_mask'] =  np.array([point[2] for point in sorted_points])
	obs['maggies'] = np.array([point[3] for point in sorted_points])
	obs['maggies_unc'] =  np.array([point[4] for point in sorted_points])
	obs['wavelength'] = None
	obs['spectrum'] = None

	return obs

def load_fast_3dhst(filename, objnum):
	"""
	Load FAST output for a particular object
	Returns a dictionary of inputs for BSFH
	"""

	# filter through header junk, load data
	fieldname=filename.split('/')[-1].split('_')[0].upper()
	with open(filename, 'r') as f:
		for jj in range(1): hdr = f.readline().split()
	dat = np.loadtxt(filename, comments = '#',dtype = np.dtype([(n, np.float) for n in hdr[1:]]))

	# extract field names, search for ID, pull out object info
	fields = [f for f in dat.dtype.names]
	id_ind = fields.index('id')
	obj_ind = [int(x[id_ind]) for x in dat].index(int(objnum))
	values = dat[fields].view(float).reshape(len(dat),-1)[obj_ind]

	return values, fields

def show_chain(sample_results,outname=None,alpha=0.6):
	
	'''
	plot the MCMC chain for all parameters
	'''
	
	# set + load variables
	parnames = np.array(sample_results['model'].theta_labels())
	nwalkers = sample_results['plotchain'].shape[0]
	nsteps = sample_results['plotchain'].shape[1]

	
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
		walkerend   = np.clip(nwalkers_per_column*(ii+1),0,sample_results['plotchain'].shape[0])
		for jj in xrange(len(parnames)):
			for kk in xrange(walkerstart,walkerend):
				axarr[jj,ii].plot(sample_results['plotchain'][kk,:,jj],'-',
						   	   color='black',
						   	   alpha=alpha)
				
			# fiddle with x-axis
			axarr[jj,ii].axis('tight')
			axarr[jj,ii].set_xticklabels([])
				
			# fiddle with y-axis
			if ii == 0:
				axarr[jj,ii].set_ylabel(sample_results['plotnames'][jj])
			else:
				axarr[jj,ii].set_yticklabels([])
			axarr[jj,ii].set_ylim(sample_results['extents'][jj])
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

def sed_figure(sample_results, sps, model,
                alpha=0.3, samples = [-1],
                start=0, thin=1, maxprob=0, outname=None, plot_init = 0,
                #parm_file = os.getenv('APPS')+'/threedhst_bsfh/parameter_files/threedhst_params.py',
                chop_chain=1,**kwargs):
	"""
	Plot the photometry for the model and data (with error bars), and
	plot residuals
	"""
	c = 3e8
	
	from matplotlib import gridspec

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
	phot.set_xlim(min(xplot)*0.9,max(xplot)*1.04)
	phot.set_ylim(min(yplot[np.isfinite(yplot)])*0.7,max(yplot[np.isfinite(yplot)])*1.04)
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
	
	# add SFH plot
	str_sfh_parms = ['tau','tburst','fburst','sf_start']
	sfh_parms = [thetas[0][i] for i in xrange(len(thetas[0])) if model.theta_labels()[i] in str_sfh_parms]
	sfh_parms = [model.params['tage'][0]]+sfh_parms
	t, intsfh = plot_sfh(sfh_parms)

	axfontsize=4
	ax_inset=fig.add_axes([0.17,0.36,0.12,0.14],zorder=32)
	ax_inset.axis([np.min(t),np.max(t)+0.08*(np.max(t)-np.min(t)),0,1.05])
	ax_inset.plot(t, intsfh,'-')
	ax_inset.set_ylabel('cum SFH',fontsize=axfontsize,weight='bold')
	ax_inset.set_xlabel('t [Gyr]',fontsize=axfontsize,weight='bold')
	ax_inset.tick_params(labelsize=axfontsize)
	
	# add RGB
	try:
		field=sample_results['run_params']['photname'].split('/')[-1].split('_')[0]
		objnum="%05d" % int(sample_results['run_params']['objname'])
		img=mpimg.imread(os.getenv('APPS')+
		                 '/threedhst_bsfh/data/RGB_v4.0_field/'+
		                 field.lower()+'_'+objnum+'_vJH_6.png')
		ax_inset2=fig.add_axes([0.155,0.51,0.15,0.15],zorder=32)
		ax_inset2.imshow(img)
		ax_inset2.set_axis_off()
	except:
		print 'no RGB image'
	
	# calculate reduced chi-squared
	chisq=np.sum(specvecs[0][-1]**2)
	degoffreedom = len(yplot)-len(model.theta_labels())-1
	reduced_chisq = chisq/degoffreedom
	
	# diagnostic text
	textx = (phot.get_xlim()[1]-phot.get_xlim()[0])*0.975+phot.get_xlim()[0]
	texty = (phot.get_ylim()[1]-phot.get_ylim()[0])*0.2+phot.get_ylim()[0]
	deltay = (phot.get_ylim()[1]-phot.get_ylim()[0])*0.038

	phot.text(textx, texty, r'best-fit $\chi^2$='+"{:.2f}".format(reduced_chisq),
			  fontsize=10, ha='right')
	phot.text(textx, texty-deltay, r'avg acceptance='+"{:.2f}".format(np.mean(sample_results['acceptance'])),
				 fontsize=10, ha='right')
		
	# load ancil data
	if 'ancilname' not in model.run_params.keys():
		model.run_params['ancilname'] = os.getenv('APPS')+'/threedhst_bsfh/data/COSMOS_testsamp.dat'
	ancilvals,ancilfields = load_ancil_data(model.run_params['ancilname'],model.run_params['objname'])
	sn_txt = ancilvals[ancilfields.index('sn_F160W')]
	uvj_txt = ancilvals[ancilfields.index('uvj_flag')]
	z_txt = ancilvals[ancilfields.index('z')]
		
	# galaxy text
	phot.text(textx, texty-2*deltay, 'z='+"{:.2f}".format(z_txt),
			  fontsize=10, ha='right')
	phot.text(textx, texty-3*deltay, 'uvj_flag='+str(uvj_txt),
			  fontsize=10, ha='right')
	phot.text(textx, texty-4*deltay, 'S/N(F160W)='+"{:.2f}".format(sn_txt),
			  fontsize=10, ha='right')
		
		
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
	
	# clean up and output
	fig.add_subplot(phot)
	fig.add_subplot(res)
	
	# set second x-axis
	y1, y2=phot.get_ylim()
	x1, x2=phot.get_xlim()
	ax2=phot.twiny()
	ax2.set_xticks(np.arange(0,10,0.2))
	ax2.set_xlim(np.log10((10**(x1))/(1+z_txt)), np.log10((10**(x2))/(1+z_txt)))
	ax2.set_xlabel(r'log($\lambda_{rest}$) [$\AA$]')
	ax2.set_ylim(y1, y2)

	# remove ticks
	phot.set_xticklabels([])
    
	if outname is not None:
		fig.savefig(outname, bbox_inches='tight', dpi=300)
		plt.close()

def make_all_plots(filebase, parm_file=None, 
				   outfolder=os.getenv('APPS')+'/threedhst_bsfh/plots/'):

	'''
	Driver. Loads output, makes all plots for a given galaxy.
	'''

	plt_chain_figure = 1
	plt_triangle_plot = 1
	plt_sed_figure = 1
	
	# thin and chop the chain?
	thin=2
	chop_chain=2

	# find most recent output file
	# with the objname
	folder = "/".join(filebase.split('/')[:-1])
	filename = filebase.split("/")[-1]
	files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
	times = [f.split('_')[-2] for f in files]

	# if we found no files, skip this object
	if len(times) == 0:
		print 'Failed to open '+ mcmc_filename +','+model_filename
		return 0

	# load results
	mcmc_filename=filebase+'_'+max(times)+"_mcmc"
	model_filename=filebase+'_'+max(times)+"_model"
	
	try:
		sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
	except:
		print 'Failed to open '+ mcmc_filename +','+model_filename
		return 0
	print 'MAKING PLOTS FOR ' + filename
	
	# define nice plotting quantities
	sample_results = create_plotquant(sample_results)
	sample_results['extents'] = return_extent(sample_results)
	
	# load stellar population, set up custom filters
	sps = fsps.StellarPopulation(zcontinuous=True, compute_vega_mags=False)
	custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

	# load parameter file
	model = model_setup.setup_model(parm_file, sps=sps)
	
    # chain plot
	if plt_chain_figure: 
		print 'MAKING CHAIN PLOT'
		show_chain(sample_results,
	               outname=outfolder+filename+'_'+max(times)+".chain.png",
			       alpha=0.3)

	# triangle plot
	if plt_triangle_plot: 
		print 'MAKING TRIANGLE PLOT'
		chopped_sample_results = copy.deepcopy(sample_results)
		chopped_sample_results['plotchain'] = sample_results['plotchain'][:,int(sample_results['plotchain'].shape[1]/chop_chain):,:]
		chopped_sample_results['chain'] = sample_results['chain'][:,int(sample_results['chain'].shape[1]/chop_chain):,:]

		read_results.subtriangle(chopped_sample_results, sps, model,
							 outname=outfolder+filename+'_'+max(times),
							 showpars=None,start=0, thin=thin, #truths=sample_results['initial_theta'],
							 show_titles=True)

	# sed plot
	if plt_sed_figure:
		print 'MAKING SED PLOT'
 		# plot
 		pfig = sed_figure(sample_results, sps, model,
 						  maxprob=1, 
 						  outname=outfolder+filename+'_'+max(times)+'.sed.png',
 						  thin=thin,
 						  chop_chain=chop_chain)
 		
def plot_all_driver():

	'''
	for a list of galaxies, make all plots
	'''

	id_list = os.getenv('APPS')+"/threedhst_bsfh/data/COSMOS_testsamp.ids"
	basename = "threedhst_nebon"
	basename = "threedhst"
	parm_basename = "threedhst_params_nebon"
	parm_basename = "threedhst_params"
	ids = np.loadtxt(id_list, dtype='|S20')
	
	for jj in xrange(len(ids)):
		filebase = os.getenv('APPS')+"/threedhst_bsfh/results/"+basename+'_'+ids[jj]
		make_all_plots(filebase, 
		parm_file=os.getenv('APPS')+"/threedhst_bsfh/parameter_files/"+parm_basename+'_'+str(jj+1)+'.py')
	
	
	
	
	
	
	





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