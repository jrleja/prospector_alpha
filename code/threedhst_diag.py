import numpy as np
import matplotlib.pyplot as plt
import triangle, os, math, copy, threed_dutils
from bsfh import read_results
import matplotlib.image as mpimg
from astropy.cosmology import WMAP9
import fsps
plt.ioff() # don't pop up a window for each plot

tiny_number = 1e-3
big_number = 1e90
plt_chain_figure = 1
plt_triangle_plot = 1
plt_sed_figure = 1

def add_sfh_plot(sample_results,fig,ax_loc,truths=None,fast=None):
	
	'''
	add a small SFH plot at ax_loc
	'''

	# minimum and maximum SFRs to plot
	minsfr = 0.005
	maxsfr = 10000

	t, perc = plot_sfh(sample_results, ncomp=sample_results.get('ncomp',2))
	perc = np.log10(np.clip(perc,minsfr,maxsfr))
	axfontsize=4
	
	# set up plotting
	if fig is not None:
		ax_inset=fig.add_axes(ax_loc,zorder=32)
	else:
		ax_inset = ax_loc
	
	# plot whole SFH
	ax_inset.plot(t, perc[:,0,1],'-',color='black')
	
	##### FAST + normal fit SFH #####
	if truths is None:
	
		colors=['blue','red']
		for aa in xrange(1,perc.shape[1]):
			ax_inset.plot(t, perc[:,aa,1],'-',color=colors[aa-1],alpha=0.4)
			ax_inset.text(0.08,0.83-0.07*(aa-1), 'tau'+str(aa),transform = ax_inset.transAxes,color=colors[aa-1],fontsize=axfontsize*1.4)

		# FAST SFH
		if fast:
			ltau = 10**fastparms[fastfields=='ltau'][0]/1e9
			lage = 10**fastparms[fastfields=='lage'][0]/1e9
			fmass = 10**fastparms[fastfields=='lmass'][0]
			tuniv = WMAP9.age(fastparms[fastfields=='z'][0]).value

			tf, pf = plot_sfh_fast(ltau,lage,fmass,tuniv)
			pf = np.log10(np.clip(pf,minsfr,maxsfr))
			ax_inset.plot(tf, pf,'-',color='k')
			ax_inset.plot(tf, pf,'-',color=fastcolor,alpha=1.0,linewidth=0.75)
			ax_inset.text(0.08,0.83-0.07*(perc.shape[1]-1), 'fast',transform = ax_inset.transAxes,color=fastcolor,fontsize=axfontsize*1.4)
			
			# for setting limits
			# WARNING: THIS IS A HACK
			# possible since not using q16 part of perc
			perc[:,0,0] = pf

	##### TRUTHS + 50th percentile SFH #####
	else:
		
		ax_inset.fill_between(t, perc[:,0,0], perc[:,0,2], color='0.75')

		parnames = sample_results['model'].theta_labels()
		tage = sample_results['model'].params['tage'][0]
		tt,pt = plot_sfh_single(truths['truths'],tage,parnames,ncomp=sample_results.get('ncomp',2))

		pt = np.log10(np.clip(pt,minsfr,maxsfr))
		#tcolors=['steelblue','maroon']
		#for aa in xrange(sample_results.get('ncomp',2)):
		#	ax_inset.plot(tt, pt[:,aa+1],'-',color=tcolors[aa],alpha=0.5,linewidth=0.75)
		ax_inset.plot(tt, pt[:,0],'-',color='blue')
		ax_inset.text(0.08,0.83, 'truth',transform = ax_inset.transAxes,color='blue',fontsize=axfontsize*1.4)

	dynrange = (np.max(perc)-np.min(perc))*0.1
	axlim_sfh=[np.min(t),
	           np.max(t)+0.08*(np.max(t)-np.min(t)),
	           np.min(perc)-dynrange,np.max(perc)+dynrange]
	ax_inset.axis(axlim_sfh)

	ax_inset.set_ylabel('log(SFH)',fontsize=axfontsize,weight='bold')
	ax_inset.set_xlabel('t [Gyr]',fontsize=axfontsize,weight='bold')
	ax_inset.tick_params(labelsize=axfontsize)

	# label
	ax_inset.text(0.08,0.9, 'tot',transform = ax_inset.transAxes,fontsize=axfontsize*1.4)

def plot_sfh_fast(tau,tage,mass,tuniv=None):

	'''
	version of plot_sfh, but only for FAST outputs
	this means no chain sampling, and simple tau rather than delayed tau models
	if we specify tuniv, return instead (tuniv-t)
	'''
	
	t=np.linspace(0,tage,num=50)
	sfr = np.exp(-t/tau)
	sfr_int = mass/np.sum(sfr * (t[1]-t[0])*1e9)  # poor man's integral, integrate f*yrs
	sfr = sfr * sfr_int

	if tuniv:
		t = tuniv-t
		t = t[::-1]

	return t,sfr

def plot_sfh_single(truths,tage,parnames,ncomp=2):

	parnames = np.array(parnames)

	# initialize output variables
	nt = 50
	intsfr = np.zeros(shape=(nt,ncomp+1))
	deltat=0.001
	t=np.linspace(0,np.max(tage),num=50)

	# define input variables
	mass = truths[np.array([True if 'mass' in x else False for x in parnames])]
	tau = truths[np.array([True if 'tau' in x else False for x in parnames])]
	sf_start = truths[np.array([True if 'sf_start' in x else False for x in parnames])]
	tage = np.zeros(ncomp)+tage

	for jj in xrange(nt): intsfr[jj,0] = threed_dutils.integrate_sfh(t[jj]-deltat,t[jj],mass,
		                                                                    tage,tau,sf_start)*np.sum(mass)/(deltat*1e9)
	for kk in xrange(ncomp):
		for jj in xrange(nt): intsfr[jj,kk+1] = threed_dutils.integrate_sfh(t[jj]-deltat,t[jj],mass[kk],
                                                                    tage[kk],tau[kk],sf_start[kk])*mass[kk]/(deltat*1e9)

	return t, intsfr

def plot_sfh(sample_results,nsamp=1000,ncomp=2):

	'''
	create sfh for plotting purposes
	input: str_sfh_parms = ['tage','tau','sf_start']
	returns: time-vector, plus SFR(t), where SFR is in units of Gyr^-1 (normalize by mass to get true SFR)
	SFR(t) = [len(TIME),len(COMPONENTS)+1,3]
	the 0th vector is total sfh, 1st is 1st component, etc
	third dimension is [q16,q50,q84]
	'''

	# find SFH parameters that are variables in the chain
    # save their indexes for future use
	str_sfh_parms = ['mass','tau','sf_start','tage']
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
	intsfr = np.zeros(shape=(nsamp,nt,ncomp+1))
	deltat=0.001

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
		mass,tau,sf_start,tage = sfh_parms

		if mm == 0:
			t=np.linspace(0,np.max(tage),num=50)

		totmass = np.sum(mass)
		for jj in xrange(nt): intsfr[mm,jj,0] = threed_dutils.integrate_sfh(t[jj]-deltat,t[jj],mass,
		                                                                    tage,tau,sf_start)*totmass/(deltat*1e9)
		for kk in xrange(ncomp):
			for jj in xrange(nt): intsfr[mm,jj,kk+1] = threed_dutils.integrate_sfh(t[jj]-deltat,t[jj],mass[kk],
	                                                                    tage[kk],tau[kk],sf_start[kk])*mass[kk]/(deltat*1e9)

	q = np.zeros(shape=(nt,ncomp+1,3))
	for kk in xrange(ncomp+1):
		for jj in xrange(nt): q[jj,kk,:] = np.percentile(intsfr[:,jj,kk],[16.0,50.0,84.0])

	# check to see if (SFH1 + SFH2) == SFHTOT [should it?]
	#if (np.abs(intsfr[:,0,1] - intsfr[:,1,1]-intsfr[:,2,1]) > intsfr[:,0,1]*0.001).any():
	#	print 1/0

	return t, q

def create_plotquant(sample_results, logplot = ['mass', 'tau', 'tage'], truths=None):
    
	'''
	creates plottable versions of chain and sets up new plotnames
	'''

	# set up plot chain, parnames
	plotchain = copy.deepcopy(sample_results['chain'])
	parnames = np.array(sample_results['model'].theta_labels())

	# properly define timescales in certain parameters
	# will have to redefine after inserting p(z)
	# note that we switch prior min/max here!!
	tuniv = WMAP9.age(sample_results['model'].params['zred'][0]).value
	if truths is not None:
		tuniv = 14.0
	redefine = ['sf_start']

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
			min = np.zeros(2)+priors['mini']
			max = np.zeros(2)+priors['maxi']
			priors['mini'] = np.clip(tuniv-max,tiny_number,big_number)
			priors['maxi'] = tuniv-min

			plotchain[:,:,ii] = np.clip(tuniv - plotchain[:,:,ii],tiny_number,big_number)	    	

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
				for k,v in priors.iteritems(): priors[k]=np.log10(np.clip(v,1e-30,1e99))
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

def show_chain(sample_results,outname=None,alpha=0.6,truths=None):
	
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

			# add truths
			if truths is not None:
				axarr[jj,ii].axhline(truths['plot_truths'][jj], linestyle='-',color='r')

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
		axarr[jj+1,ii].set_ylim(max-4*stddev, max)
		
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
                alpha=0.3, samples = [-1], powell=None,
                maxprob=0, outname=None, plot_init = 0, fast=True,
                truths = None,
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

	# flatchain
	flatchain = sample_results['flatchain']

	# plot max probability model
	mwave, mospec, mounc, specvecs = comp_samples([sample_results['quantiles']['maxprob_params']], sample_results, sps, photflag=1)
		
	phot.plot(np.log10(mwave), np.log10(specvecs[0][0]*(c/(mwave/1e10))), 
		      color='#e60000', marker='o', ms=ms, linestyle=' ', label='max lnprob', 
		      alpha=alpha, markeredgewidth=0.7,**kwargs)
	
	res.plot(np.log10(mwave), specvecs[0][-1], 
		     color='#e60000', marker='o', linestyle=' ', label='max lnprob', 
		     ms=ms,alpha=alpha,markeredgewidth=0.7,**kwargs)
	
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

	# add most likely spectrum
	spec,_,w = model.mean_model(sample_results['quantiles']['maxprob_params'], sample_results['obs'], sps=sps)
	nz = spec > 0

	phot.plot(np.log10(w[nz]), np.log10(spec[nz]*(c/(w[nz]/1e10))), linestyle='-',
              color='red', alpha=0.6,**kwargs)

    # PLOT OBSERVATIONS + ERRORS 
	phot.errorbar(xplot, yplot, yerr=yerr,
                  color='#545454', marker='o', label='observed', alpha=alpha, linestyle=' ',ms=ms)
	

	# plot best-fit FAST model
	if fast:
		fastcolor='#00CCFF'
		f_filename = os.getenv('APPS')+'/threedhst_bsfh'+sample_results['run_params']['fastname'].split('/threedhst_bsfh')[1]
		f_objname  = sample_results['run_params']['objname']
		fspec,fmags,fw,fastparms,fastfields=threed_dutils.return_fast_sed(f_filename,f_objname, sps=sps, obs=sample_results['obs'], dustem = False)
		nz = fspec > 0

		phot.plot(np.log10(fw[nz]), np.log10(fspec[nz]*(c/(fw[nz]/1e10))), linestyle='-',
	              color=fastcolor, alpha=0.6,zorder=-1)

		fwave = sample_results['obs']['wave_effective']
		phot.plot(np.log10(fwave), np.log10(fmags*(c/(fwave/1e10))), 
			     color=fastcolor, marker='o', linestyle=' ', label='fast', 
			     ms=ms,alpha=alpha,markeredgewidth=0.7,zorder=-1)

	# plot truths
	if truths is not None:
		mwave_truth, mospec_truth, mounc_truth, specvecs_truth = comp_samples([truths['truths']], sample_results, sps, photflag=1)
		
		#phot.plot(np.log10(mwave_truth), np.log10(specvecs_truth[0][0]*(c/(mwave_truth/1e10))), 
		#	      color='blue', marker='o', ms=ms, linestyle=' ', label='truths', 
		#	      alpha=alpha, markeredgewidth=0.7,**kwargs)
	
		res.plot(np.log10(mwave_truth), specvecs_truth[0][-1], 
			     color='blue', marker='o', linestyle=' ', label='truths', 
			     ms=ms,alpha=0.3,markeredgewidth=0.7,**kwargs)

	# plot Powell minimization answer
	if powell is not None:
		mwave_powell, mospec_powell, mounc_powell, specvecs_powell = comp_samples([powell], sample_results, sps, photflag=1)
		
		phot.plot(np.log10(mwave_powell), np.log10(specvecs_powell[0][0]*(c/(mwave_powell/1e10))), 
			      color='purple', marker='o', ms=ms, linestyle=' ', label='powell', 
			      alpha=alpha, markeredgewidth=0.7,**kwargs)

		#res.plot(np.log10(mwave_powell), specvecs_powell[0][-1], 
		#	     color='purple', marker='o', linestyle=' ', label='powell', 
		#	     ms=ms,alpha=0.3,markeredgewidth=0.7,**kwargs)

	# add SFH plot
	ax_loc = [0.2,0.35,0.12,0.14]
	add_sfh_plot(sample_results,fig,ax_loc,truths=truths,fast=fast)

	# add RGB
	try:
		field=sample_results['run_params']['photname'].split('/')[-1].split('_')[0]
		objnum="%05d" % int(sample_results['run_params']['objname'])
		img=mpimg.imread(os.getenv('APPS')+
		                 '/threedhst_bsfh/data/RGB_v4.0_field/'+
		                 field.lower()+'_'+objnum+'_vJH_6.png')
		ax_inset2=fig.add_axes([0.31,0.34,0.15,0.15],zorder=32)
		ax_inset2.imshow(img)
		ax_inset2.set_axis_off()
	except:
		print 'no RGB image'

	# diagnostic text
	textx = (phot.get_xlim()[1]-phot.get_xlim()[0])*0.975+phot.get_xlim()[0]
	texty = (phot.get_ylim()[1]-phot.get_ylim()[0])*0.2+phot.get_ylim()[0]
	deltay = (phot.get_ylim()[1]-phot.get_ylim()[0])*0.038

	# calculate reduced chi-squared
	chisq=np.sum(specvecs[0][-1]**2)
	ndof = np.sum(sample_results['obs']['phot_mask']) - len(sample_results['model'].free_params)-1
	reduced_chisq = chisq/(ndof-1)

	# also calculate for truths if truths exist
	if truths is not None:
		chisq_truth=np.sum(specvecs_truth[0][-1]**2)
		reduced_chisq_truth = chisq_truth/(ndof-1)
		phot.text(textx, texty, r'best-fit $\chi^2_n$='+"{:.2f}".format(reduced_chisq)+' (true='
			      +"{:.2f}".format(reduced_chisq_truth)+')',
			      fontsize=10, ha='right')
	else:
		phot.text(textx, texty, r'best-fit $\chi^2_n$='+"{:.2f}".format(reduced_chisq),
			  fontsize=10, ha='right')
	
	phot.text(textx, texty-deltay, r'avg acceptance='+"{:.2f}".format(np.mean(sample_results['acceptance'])),
				 fontsize=10, ha='right')
		
	# load ancil data
	if 'ancilname' in sample_results['run_params'].keys():
		ancildat = threed_dutils.load_ancil_data(os.getenv('APPS')+'/threedh'+sample_results['run_params']['ancilname'].split('/threedh')[1],sample_results['run_params']['objname'])
		sn_txt = ancildat['sn_F160W'][0]
		uvj_txt = ancildat['uvj_flag'][0]
		z_txt = sample_results['model'].params['zred'][0]
	else:
		sn_txt = -99.0
		uvj_txt = -99.0
		z_txt = -99.0

	# FAST text
	if fast:
		textx_f = (phot.get_xlim()[1]-phot.get_xlim()[0])*0.42+phot.get_xlim()[0]
		totmass = np.log10(np.sum(sample_results['quantiles']['maxprob_params'][0:2]))
		av    = threed_dutils.av_to_dust2(fastparms[fastfields=='Av'])[0]
		zf    = fastparms[fastfields=='z'][0]

		taucolor='black'
		tagecolor='black'
		if ltau < 0.1:
			taucolor='red'
		if lage < 0.1:
			tagecolor='red'

		phot.text(textx_f, texty, r'M$_{fast}$='+"{:.3f}".format(np.log10(fmass))+' ('+"{:.3f}".format(totmass)+')',
			  fontsize=10)
		phot.text(textx_f, texty-deltay, 'Av='+"{:.3f}".format(av),
			  fontsize=10)
		phot.text(textx_f, texty-deltay*2, r'$\tau$='+"{:.3f}".format(ltau),
			  fontsize=10,color=taucolor)
		phot.text(textx_f, texty-deltay*3, 'tage='+"{:.3f}".format(lage),
			  fontsize=10,color=tagecolor)
		phot.text(textx_f, texty-deltay*4, 'z='+"{:.3f}".format(zf),
			  fontsize=10)
		
		try:
			# get structural parameters (km/s, kpc)
			sigmaRe = ancildat['sigmaRe'][0]
			e_sigmaRe = ancildat['e_sigmaRe'][0]
			Re      = ancildat['Re'][0]*1e3
			nserc   = ancildat['n'][0]
			G      = 4.302e-3 # pc Msun**-1 (km/s)**2

			# dynamical masses
			# bezanson 2014, eqn 13+14
			k              = 8.87 - 0.831*nserc + 0.0241*nserc**2
			mdyn_serc      = k*Re*sigmaRe**2/G
			phot.text(textx_f, texty+deltay, 'Mdyn='+"{:.3f}".format(np.log10(mdyn_serc)),
			  		  fontsize=10)
		except:
			pass
		
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
	#os.system('open '+outname)

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
		except (EOFError,ValueError) as e:
			print e
			print 'Failed to open '+ mcmc_filename +','+model_filename
			return 0

	if not sps:
		# load stellar population, set up custom filters
		if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
			sps = threed_dutils.setup_sps()
		else:
			sps = threed_dutils.setup_sps(zcontinuous=1)

	# BEGIN PLOT ROUTINE
	print 'MAKING PLOTS FOR ' + filename + ' in ' + outfolder
	
	# do we know the truths?
	try:
		truths = threed_dutils.load_truths(os.getenv('APPS')+'/threed'+sample_results['run_params']['truename'].split('/threed')[1],
			                              sample_results['run_params']['objname'],
			                              sample_results)
	except KeyError:
		truths=None

	# define nice plotting quantities
	sample_results = create_plotquant(sample_results, truths=truths)
	sample_results['extents'] = return_extent(sample_results)
    # chain plot
	if plt_chain_figure: 
		print 'MAKING CHAIN PLOT'
		show_chain(sample_results,
	               outname=outfolder+filename+'_'+max(times)+".chain.png",
			       alpha=0.3,truths=truths)

	# triangle plot
	if plt_triangle_plot: 
		print 'MAKING TRIANGLE PLOT'
		chopped_sample_results = copy.deepcopy(sample_results)

		read_results.subtriangle(sample_results, sps, copy.deepcopy(sample_results['model']),
							 outname=outfolder+filename+'_'+max(times),
							 showpars=None,start=0,
							 show_titles=True, truths=truths)

	# sed plot
	if plt_sed_figure:
		print 'MAKING SED PLOT'
		
		# FAST fit?
		try:
			sample_results['run_params']['fastname']
			fast=1
		except:
			fast=0
		
		# powell guess?
		#bestpowell=np.argmin([p.fun for p in powell_results])
		#pguess = powell_results[bestpowell].x
		pguess = None

 		# plot
 		pfig = sed_figure(sample_results, sps, copy.deepcopy(sample_results['model']),
 						  maxprob=1,fast=fast,truths=truths,powell=pguess,
 						  outname=outfolder+filename+'_'+max(times)+'.sed.png')
 		
def plot_all_driver():

	'''
	for a list of galaxies, make all plots
	'''

	runname = 'dtau_intmet'
	runname = 'dtau_genpop'
	#runname = 'dtau_nonir'
	#runname = 'dtau_genpop_fixedmet'
	#runname = 'dtau_ha_zperr'
	runname = 'dtau_ha_plog'
	runname = 'testsed_burst'

	filebase, parm_basename, ancilname=threed_dutils.generate_basenames(runname)
	for jj in xrange(len(filebase)):
		print 'iteration '+str(jj) 
		make_all_plots(filebase=filebase[jj],\
		               parm_file=parm_basename[jj],\
		               outfolder=os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/')
	