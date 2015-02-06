import numpy as np
import matplotlib.pyplot as plt
import triangle, os, math, copy, threed_dutils
from bsfh import model_setup, read_results
import matplotlib.image as mpimg
from astropy.cosmology import WMAP9
import fsps

tiny_number = 1e-90
big_number = 1e90
plt_chain_figure = 1
plt_triangle_plot = 1
plt_sed_figure = 1

def plot_sfh(sample_results,nsamp=1000):

	'''
	create sfh for plotting purposes
	input: str_sfh_parms = ['tage','tau','tburst','fburst','sf_start']
	'''

	# find SFH parameters that are variables in the chain
    # save their indexes for future use
	str_sfh_parms = ['mass','tau','tburst','fburst','sf_start','tage']
	parnames = sample_results['model'].theta_labels()
	indexes = []
	for string in str_sfh_parms:
		indexes.append(np.char.find(parnames,string) > -1)
	indexes = np.array(indexes,dtype='bool')

	# sample randomly from SFH
	import copy
	flatchain = copy.copy(sample_results['flatchain'])
	np.random.shuffle(flatchain)

	# initialize output variables
	nt = 50
	intsfr = np.zeros(shape=(nsamp,nt))

	for mm in xrange(nsamp):

		# combine into an SFH vector that includes non-parameter values
		sfh_parms = []
		for ll in xrange(len(str_sfh_parms)):
			if np.sum(indexes[ll]) > 0:
				sfh_parms.append(flatchain[mm,indexes[ll]])
			else:
				_ = [x['init'] for x in sample_results['model'].config_list if x['name'] == str_sfh_parms[ll]][0]
				if len(np.atleast_1d(_)) != np.sum(indexes[0]):
					_ = np.zeros(np.sum(indexes[0]))+_
				sfh_parms.append(_)
		mass,tau,tburst,fburst,sf_start,tage = sfh_parms

		if mm == 0:
			t=np.linspace(0,np.max(tage),num=50)

		for jj in xrange(nt): intsfr[mm,jj] = threed_dutils.integrate_sfh(sf_start,t[jj],mass,
		                                                                      tage,tau,sf_start,tburst,fburst)

	q = np.zeros(shape=(nt,3))
	for jj in xrange(nt): q[jj,:] = np.percentile(intsfr[:,jj],[16.0,50.0,84.0])
	return t, q

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
	
	# check for multiple stellar populations
	for ii in xrange(len(parnames)):    	
    	if parnames[ii] in redefine:
	    	priors = [f['prior_args'] for f in sample_results['model'].config_list if f['name'] == parnames[ii]][0]
	    	min = priors['mini']
	    	max = priors['maxi']
	    	priors['mini'] = np.clip(tuniv-max,tiny_number,big_number)
	    	priors['maxi'] = tuniv-min

	    	plotchain[:,:,list(parnames).index(parnames[ii])] = np.clip(tuniv - plotchain[:,:,list(parnames).index(parnames[ii])],tiny_number,big_number)
	    
	    elif parnames[ii][:-2] in redefine:
	    	priors = [f['prior_args'] for f in sample_results['model'].config_list if f['name'] == parnames[ii][:-2]][0]
	    	min = priors['mini']
	    	max = priors['maxi']
	    	priors['mini'] = np.clip(tuniv-max,tiny_number,big_number)
	    	priors['maxi'] = tuniv-min

	    	plotchain[:,:,list(parnames).index(parnames[ii][:-2])] = np.clip(tuniv - plotchain[:,:,list(parnames).index(parnames[ii][:-2])],tiny_number,big_number)	    	

	# define plot quantities and plot names
	# primarily converting to log or not
    if logplot is not None:
    	
    	# if we're interested in logging it...
    	# change the plotname, chain values, and priors
    	plotnames=[]
    	for ii in xrange(len(parnames)): 
    		
    		# check for multiple stellar populations
			if (parnames[ii] in logplot): 
				plotnames.append('log('+parnames[ii]+')')
				plotchain[:,:,ii] = np.log10(plotchain[:,:,ii])
				priors = [f['prior_args'] for f in sample_results['model'].config_list if f['name'] == parnames[ii]][0]
				for k,v in priors.iteritems(): priors[k]=np.log10(v)
			elif (parnames[ii][:-2] in logplot):
				plotnames.append('log('+parnames[ii]+')')
				plotchain[:,:,ii] = np.log10(plotchain[:,:,ii])
				priors = [f['prior_args'] for f in sample_results['model'].config_list if f['name'] == parnames[ii][:-2]][0]
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
		priors = [f['prior_args'] for f in sample_results['model'].config_list if f['name'] == parnames[ii]]

		# check for multiple stellar populations
		if len(priors) == 0:
			priors = [f['prior_args'] for f in sample_results['model'].config_list if f['name'] == parnames[ii][:-2]][0]
			
			# separate priors for each component?
			if len(np.atleast_1d(priors['mini'])) > 1:
				mini = priors['mini'][int(parnames[ii][-1])-1]
				maxi = priors['maxi'][int(parnames[ii][-1])-1]
			else:
				mini = priors['mini']
				maxi = priors['maxi']
		
		elif len(priors) == 1:
			priors = priors[0]
			mini = priors['mini']
			maxi = priors['maxi']

		if extent[0] == extent[1]:
			extent = (mini,maxi)
		else:
			extend = (extent[1]-extent[0])*0.12
			if np.abs(0.5*(mini-extent[0])/(mini+extent[0])) < 1e-4:
				extent[0]=extent[0]-extend
			if np.abs(0.5*(maxi-extent[1])/(maxi+extent[1])) < 1e-4:
				extent[1]=extent[1]+extend
    	
		extents.append((extent[0],extent[1]))
	
	return extents

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
	nwalkers_per_column = 64
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

def comp_samples(thetas, sample_results, sps, inlog=True, photflag=0):
    specvecs =[]
    obs, _, marker = read_results.obsdict(sample_results['obs'], photflag)
    wave, ospec, mask = obs['wave_effective'], obs['spectrum'], obs['mask']
    mwave, mospec = wave[mask], ospec[mask]
    mounc = obs['maggies_unc'][mask]

    if inlog and (photflag == 0):
         mospec = np.exp(mospec)
         mounc *= mospec

    for theta in thetas:
        mu, cal, delta, mask, wave = read_results.model_comp(theta, sample_results, sps,
                                           					 photflag=1)

        specvecs += [ [mu, cal, delta, mu,mospec/mu, (mospec-mu) / mounc] ]
    
    # wave = effective wavelength of photometric bands
    # mospec LOOKS like the observed spectrum, but if phot_flag=1, it's the observed maggies
    # mounc is the observed photometric uncertainty
    # specvecs is: model maggies, nothing, nothing, model maggies, observed maggies/model maggies, observed maggies-model maggies / uncertainties
    return wave, mospec, mounc, specvecs	

def sed_figure(sample_results, sps, model,
                alpha=0.3, samples = [-1],
                maxprob=0, outname=None, plot_init = 0,
                **kwargs):
	"""
	Plot the photometry for the model and data (with error bars), and
	plot residuals
	"""
	c = 3e8
	ms = 5
	alpha = 0.8
	
	from matplotlib import gridspec

	# set up plot
	fig = plt.figure()
	gs = gridspec.GridSpec(2,1, height_ratios=[3,1])
	gs.update(hspace=0)
	phot, res = plt.Subplot(fig, gs[0]), plt.Subplot(fig, gs[1])

	# FLATTEN AND CHOP CHAIN
	# chain = chain[nwalkers,nsteps,ndim]
	flatchain = sample_results['flatchain']

	# MAKE RANDOM POSTERIOR DRAWS
	nsample = 5
	ns = flatchain.shape[0] * flatchain.shape[1]
	samples = np.random.uniform(0, 1, size=nsample)
	sample = [int(s * ns) for s in samples]
		
	thetas = [flatchain[s,:] for s in samples]
	mwave, mospec, mounc, specvecs = comp_samples(thetas, sample_results, sps, photflag=1)

	# define observations
	xplot = np.log10(mwave)
	yplot = np.log10(mospec*(c/(mwave/1e10)))
	linerr_down = np.clip(mospec-mounc, 1e-80, 1e80)*(c/(mwave/1e10))
	linerr_up = np.clip(mospec+mounc, 1e-80, 1e80)*(c/(mwave/1e10))
	yerr = [yplot - np.log10(linerr_down), np.log10(linerr_up)-yplot]

	# set up plot limits
	phot.set_xlim(min(xplot)*0.9,max(xplot)*1.04)
	phot.set_ylim(min(yplot[np.isfinite(yplot)])*0.7,max(yplot[np.isfinite(yplot)])*1.04)
	res.set_xlim(min(xplot)*0.9,max(xplot)*1.04)

	# plot max probability model
	mwave, mospec, mounc, specvecs = comp_samples([sample_results['quantiles']['maxprob_params']], sample_results, sps, photflag=1)
		
	phot.plot(np.log10(mwave), np.log10(specvecs[0][0]*(c/(mwave/1e10))), 
		      color='#e60000', marker='o', ms=ms, linestyle=' ', label='max lnprob', 
		      alpha=alpha, markeredgewidth=0.7,**kwargs)
	
	res.plot(np.log10(mwave), specvecs[0][-1], 
		     color='#e60000', marker='o', linestyle=' ', label='max lnprob', 
		     ms=ms,alpha=alpha,markeredgewidth=0.7,**kwargs)
	
	# add most likely spectrum
	spec,_,w = model.mean_model(sample_results['quantiles']['maxprob_params'], sample_results['obs'], sps=sps)
	nz = spec > 0

	phot.plot(np.log10(w[nz]), np.log10(spec[nz]*(c/(w[nz]/1e10))), linestyle='-',
              color='red', alpha=0.6,**kwargs)

    # PLOT OBSERVATIONS + ERRORS 
	phot.errorbar(xplot, yplot, yerr=yerr,
                  color='#545454', marker='o', label='observed', alpha=alpha, linestyle=' ',ms=ms)
	
	# add SFH plot
	t, perc = plot_sfh(sample_results)
	axfontsize=4
	ax_inset=fig.add_axes([0.17,0.36,0.12,0.14],zorder=32)
	axlim_sfh=[np.min(t),np.max(t)+0.08*(np.max(t)-np.min(t)),0,1.05]
	ax_inset.axis(axlim_sfh)
	ax_inset.plot(t, perc[:,1],'-',color='black')
	ax_inset.fill_between(t, perc[:,0], perc[:,2], color='0.75')
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
	ndof = np.sum(sample_results['obs']['phot_mask']) - len(sample_results['model'].free_params)-1
	reduced_chisq = chisq/(ndof-1)
	
	# diagnostic text
	textx = (phot.get_xlim()[1]-phot.get_xlim()[0])*0.975+phot.get_xlim()[0]
	texty = (phot.get_ylim()[1]-phot.get_ylim()[0])*0.2+phot.get_ylim()[0]
	deltay = (phot.get_ylim()[1]-phot.get_ylim()[0])*0.038

	phot.text(textx, texty, r'best-fit $\chi^2_n$='+"{:.2f}".format(reduced_chisq),
			  fontsize=10, ha='right')
	phot.text(textx, texty-deltay, r'avg acceptance='+"{:.2f}".format(np.mean(sample_results['acceptance'])),
				 fontsize=10, ha='right')
		
	# load ancil data
	if 'ancilname' not in sample_results['run_params'].keys():
		sample_results['run_params']['ancilname'] = os.getenv('APPS')+'/threedhst_bsfh/data/COSMOS_testsamp.dat'
	ancildat = threed_dutils.load_ancil_data(os.getenv('APPS')+'/threedh'+sample_results['run_params']['ancilname'].split('/threedh')[1],sample_results['run_params']['objname'])
	sn_txt = ancildat['sn_F160W'][0]
	uvj_txt = ancildat['uvj_flag'][0]
	z_txt = ancildat['z'][0]
		
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

def make_all_plots(filebase=None, parm_file=None, 
				   outfolder=os.getenv('APPS')+'/threedhst_bsfh/plots/',
				   sample_results=None,
				   sps=None):

	'''
	Driver. Loads output, makes all plots for a given galaxy.
	'''
	
	# thin and chop the chain?
	#thin=1
	#chop_chain=1.666

	# make sure the output folder exists
	if not os.path.isdir(outfolder):
		os.makedirs(outfolder)

	
	# find most recent output file
	# with the objname
	folder = "/".join(filebase.split('/')[:-1])
	filename = filebase.split("/")[-1]
	files = [f for f in os.listdir(folder) if "_".join(f.split('_')[:-2]) == filename]	
	times = [f.split('_')[-2] for f in files]

	if not sample_results:
		# if we found no files, skip this object
		if len(times) == 0:
			print 'Failed to find any files to extract times in ' + folder + ' of form ' + filename
			return 0

		# load results
		mcmc_filename=filebase+'_'+max(times)+"_mcmc"
		model_filename=filebase+'_'+max(times)+"_model"
		
		try:
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
		except:
			print 'Failed to open '+ mcmc_filename +','+model_filename
			return 0

	
	if not sps:
	# load stellar population, set up custom filters
		sps = threed_dutils.setup_sps()

	# BEGIN PLOT ROUTINE
	print 'MAKING PLOTS FOR ' + filename + ' in ' + outfolder
	
	# define nice plotting quantities
	sample_results = create_plotquant(sample_results)
	sample_results['extents'] = return_extent(sample_results)

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
		#chopped_sample_results['plotchain'] = threed_dutils.chop_chain(sample_results['plotchain'])
		#chopped_sample_results['chain'] = threed_dutils.chop_chain(sample_results['chain'])

		read_results.subtriangle(sample_results, sps, copy.deepcopy(sample_results['model']),
							 outname=outfolder+filename+'_'+max(times),
							 showpars=None,start=0,
							 show_titles=True)

	# sed plot
	# MANY UNNECESSARY CALCULATIONS
	if plt_sed_figure:
		print 'MAKING SED PLOT'
 		# plot
 		pfig = sed_figure(sample_results, sps, copy.deepcopy(sample_results['model']),
 						  maxprob=1, 
 						  outname=outfolder+filename+'_'+max(times)+'.sed.png')
 		
def plot_all_driver():

	'''
	for a list of galaxies, make all plots
	'''

	runname = "neboff"
	
	runname = "photerr"

	filebase, parm_basename=threed_dutils.generate_basenames(runname)
	for jj in xrange(len(filebase)):
		make_all_plots(filebase=filebase[jj],\
		               parm_file=parm_basename[jj],\
		               outfolder=os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/')
	