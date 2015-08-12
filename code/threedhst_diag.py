import numpy as np
import matplotlib.pyplot as plt
import triangle, os, math, copy, threed_dutils
from bsfh import read_results
import matplotlib.image as mpimg
from astropy.cosmology import WMAP9
import fsps
from matplotlib.ticker import MaxNLocator

plt.ioff() # don't pop up a window for each plot

tiny_number = 1e-3
big_number = 1e90

def subtriangle(sample_results,  sps, model,
                outname=None, showpars=None,
                start=0, thin=1, truths=None,
                powell_results=None, 
                **kwargs):
    """
    Make a triangle plot of the (thinned, latter) samples of the posterior
    parameter space.  Optionally make the plot only for a supplied subset
    of the parameters.
    """
    import triangle
    # pull out the parameter names and flatten the thinned chains
    parnames = np.array(sample_results['model'].theta_labels())
    plotflatchain = threed_dutils.chop_chain(sample_results['plotchain'])

    # restrict to parameters you want to show
    if showpars is not None:
        ind_show = np.array([p in showpars for p in parnames], dtype= bool)
        plotflatchain = plotflatchain[:,ind_show]
        truths = truths[ind_show]
        parnames= parnames[ind_show]

    # plot truths
    if truths is not None:
        ptruths = [truths['plot_truths'][truths['parnames'][ii] == parnames][0] if truths['parnames'][ii] in parnames \
                   else None \
                   for ii in xrange(len(truths['plot_truths']))]
    else:
        ptruths = None

    fig = triangle.corner(plotflatchain, labels = sample_results['plotnames'],
                          quantiles=[0.16, 0.5, 0.84], verbose=False,
                          truths = ptruths, extents=sample_results['extents'],truth_color='red',**kwargs)
    
    fig = add_to_corner(fig, sample_results, sps, model, truths=truths, powell_results=powell_results)

    if outname is not None:
        fig.savefig('{0}.triangle.png'.format(outname))
        plt.close(fig)
    else:
        return fig

def add_to_corner(fig, sample_results, sps, model,truths=None,maxprob=True,powell_results=None):

    '''
    adds in posterior distributions for 'select' parameters
    if we have truths, list them as text
    '''

	# pull information from triangle to replicate plots
	# will want to put them in axes[6-8] or something
    axes = fig.get_axes()
	
    plotquant = np.log10(sample_results['extras'].get('flatchain',None))
    plottit   = sample_results['extras'].get('parnames',None)

    to_show = ['half_time','ssfr_100','sfr_100']
    if sample_results['ncomp'] > 1:
        toshow = ['half_time','ssfr_100','sfr_100','totmass']
    showing = np.array([x in to_show for x in plottit])

    # extra text
    scale    = len(sample_results['model'].theta_labels())
    ttop     = 0.88-0.02*(12-scale)
    fs       = 24-(12-scale)
    
    if truths is not None:
        parnames = np.append(truths['parnames'],'lnprob')
        parnames = np.append(parnames, truths['extra_parnames'])
        tvals    = np.append(truths['plot_truths'],truths['truthprob'])
        tvals    = np.append(tvals, truths['extra_truths'])

        plt.figtext(0.73, ttop, 'truths',weight='bold',
                       horizontalalignment='right',fontsize=fs)
        for kk in xrange(len(tvals)):
            plt.figtext(0.73, ttop-0.02*(kk+1), parnames[kk]+'='+"{:.2f}".format(tvals[kk]),
                       horizontalalignment='right',fontsize=fs)

    # show maximum probability
    if maxprob:
        maxprob_parnames = np.append(sample_results['model'].theta_labels(),'lnprob')
        plt.figtext(0.75, ttop, 'pmax',weight='bold',
                       horizontalalignment='left',fontsize=fs)
        for kk in xrange(len(maxprob_parnames)):
            if maxprob_parnames[kk] == 'mass':
               yplot = np.log10(sample_results['quantiles']['maxprob_params'][kk])
            elif maxprob_parnames[kk] == 'lnprob':
                yplot = sample_results['quantiles']['maxprob']
            else:
                yplot = sample_results['quantiles']['maxprob_params'][kk]

            # add parameter names if not covered by truths
            if truths is None:
            	plt.figtext(0.8, ttop-0.02*(kk+1), maxprob_parnames[kk]+'='+"{:.2f}".format(yplot),
                       horizontalalignment='right',fontsize=fs)
            else:
           		plt.figtext(0.75, ttop-0.02*(kk+1), "{:.2f}".format(yplot),
                       horizontalalignment='left',fontsize=fs)

    # show burn-in results
    burn_params = np.append(sample_results['post_burnin_center'],sample_results['post_burnin_prob'])
    burn_names = np.append(sample_results['model'].theta_labels(), 'lnprobability')

    plt.figtext(0.82, ttop, 'burn-in',weight='bold',
                   horizontalalignment='left',fontsize=fs)

    for kk in xrange(len(burn_names)):
        if burn_names[kk] == 'mass':
           yplot = np.log10(burn_params[kk])
        else:
            yplot = burn_params[kk]

        plt.figtext(0.82, ttop-0.02*(kk+1), "{:.2f}".format(yplot),
                   horizontalalignment='left',fontsize=fs)

    # show powell results
    if powell_results:
        best = np.argmin([p.fun for p in powell_results])
        powell_params = np.append(powell_results[best].x,-1*powell_results[best]['fun'])
        powell_names = np.append(sample_results['model'].theta_labels(),'lnprob')

        plt.figtext(0.89, ttop, 'powell',weight='bold',
                       horizontalalignment='left',fontsize=fs)

        for kk in xrange(len(powell_names)):
            if powell_names[kk] == 'mass':
               yplot = np.log10(powell_params[kk])
            else:
                yplot = powell_params[kk]

            plt.figtext(0.89, ttop-0.02*(kk+1), "{:.2f}".format(yplot),
                       horizontalalignment='left',fontsize=fs)


    plotloc   = 2
    for jj in xrange(len(plottit)):
		
        if showing[jj] == 0:
            continue
        plotloc += 1

        axes[plotloc].set_visible(True)
        axes[plotloc].set_frame_on(True)

        if plottit[jj] == 'half_time':
            plotquant[:,jj] = 10**plotquant[:,jj]

        # Plot the histograms.
        try:
            n, b, p = axes[plotloc].hist(plotquant[:,jj], bins=50,
                              histtype="step",color='k',
                              range=[np.min(plotquant[:,jj]),np.max(plotquant[:,jj])])
        except:
            plt.close()
            print np.min(plotquant[:,jj])
            print np.max(plotquant[:,jj])
            print 1/0


        # plot quantiles
        qvalues = np.log10([sample_results['extras']['q16'][jj],
        		            sample_results['extras']['q50'][jj],
        		            sample_results['extras']['q84'][jj]])
        if plottit[jj] == 'half_time':
            qvalues = 10**qvalues

        for q in qvalues:
        	axes[plotloc].axvline(q, ls="dashed", color='k')

        # display quantiles
        q_m = qvalues[1]-qvalues[0]
        q_p = qvalues[2]-qvalues[1]

        # format quantile display
        fmt = "{{0:{0}}}".format(".2f").format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(qvalues[1]), fmt(q_m), fmt(q_p))
        if plottit[jj] != 'half_time':
            title = "{0} = {1}".format('log'+plottit[jj],title)
        else:
            title = "{0} = {1}".format(plottit[jj],title)
        axes[plotloc].set_title(title)

        # axes
        axes[plotloc].set_ylim(0, 1.1 * np.max(n))
        axes[plotloc].set_yticklabels([])
        axes[plotloc].xaxis.set_major_locator(MaxNLocator(5))
	
        # truths
        if truths is not None:
            if plottit[jj] in parnames:
                plottruth = tvals[parnames == plottit[jj]]
                axes[plotloc].axvline(x=plottruth,color='r')

    return fig

def add_sfh_plot(sample_results,fig,ax_loc,sps,truths=None,fast=None):
	
	'''
	add a small SFH plot at ax_loc
	'''

	t, perc = plot_sfh(sample_results, ncomp=sample_results['ncomp'],sps)
	perc = np.log10(perc)
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
	
		ax_inset.fill_between(t, perc[:,0,0], perc[:,0,2], color='0.75')

		#colors=['blue','red']
		#for aa in xrange(1,perc.shape[1]):
		#	ax_inset.plot(t, perc[:,aa,1],'-',color=colors[aa-1],alpha=0.4)
		#	ax_inset.text(0.08,0.83-0.07*(aa-1), 'tau'+str(aa),transform = ax_inset.transAxes,color=colors[aa-1],fontsize=axfontsize*1.4)

		# set up plotting range
		plotmax_y = np.max(perc[:,0,1])
		plotmin_y = np.min(perc[:,0,1])

		# FAST SFH
		if fast:
			ltau = 10**fastparms[fastfields=='ltau'][0]/1e9
			lage = 10**fastparms[fastfields=='lage'][0]/1e9
			fmass = 10**fastparms[fastfields=='lmass'][0]
			tuniv = WMAP9.age(fastparms[fastfields=='z'][0]).value

			tf, pf = plot_sfh_fast(ltau,lage,fmass,tuniv)
			pf = np.log10(pf)
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
		tt,pt = plot_sfh_single(truths,truths['parnames'],ncomp=sample_results['ncomp'])
		pt = np.log10(pt)
		#tcolors=['steelblue','maroon']
		#for aa in xrange(sample_results.get('ncomp',2)):
		#	ax_inset.plot(tt, pt[:,aa+1],'-',color=tcolors[aa],alpha=0.5,linewidth=0.75)
		ax_inset.plot(tt, pt[:,0],'-',color='blue')
		ax_inset.text(0.08,0.83, 'truth',transform = ax_inset.transAxes,color='blue',fontsize=axfontsize*1.4)

		# set up plotting range
		plotmax_y = np.maximum(np.max(perc[:,0,1]),np.max(pt[:,0]))
		plotmin_y = np.maximum(np.min(perc[:,0,1]),np.min(pt[:,0]))

	# the minimum time for which the upper percentile is equal to the minimum SFR
	# exclude any times before the 50th percentile of tage, since those are 
	# probably after quenching
	plotmax_x = t[perc[:,0,2] == np.min(perc[:,0,2])]
	plotmax_x = np.min(plotmax_x[plotmax_x > sample_results['quantiles']['q50'][np.array(sample_results['model'].theta_labels()) == 'tage']])
	if truths is not None:
		plotmax_x = np.max(np.append(plotmax_x,truths['sfh_params']['tage']))

	dynrange = (plotmax_y-plotmin_y)*0.1
	axlim_sfh=[plotmax_x,
	           np.min(t),
	           plotmin_y,
	           plotmax_y+dynrange]
	ax_inset.axis(axlim_sfh)

	ax_inset.set_ylabel('log(SFR)',fontsize=axfontsize,weight='bold')
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

def plot_sfh_single(truths,parnames,ncomp=1):

	parnames = np.array(parnames)

	# initialize output variables
	nt = 50
	intsfr = np.zeros(shape=(nt,ncomp+1))
	deltat=0.0001
	t=np.linspace(0,np.max(truths['sfh_params']['tage']),num=50)

	for jj in xrange(nt): intsfr[jj,0] = threed_dutils.calculate_sfr(truths['sfh_params'], deltat, tcalc = t[jj])

	for kk in xrange(ncomp):
		iterable = [(key,value[kk]) for key, value in truths['sfh_params'].iteritems()]
		newdict = {key: value for (key, value) in iterable}
		for jj in xrange(nt): intsfr[jj,kk+1] = threed_dutils.calculate_sfr(newdict, deltat, tcalc = t[jj])

	return t[::-1], intsfr

def plot_sfh(sample_results,nsamp=1000,ncomp=1):

	'''
	create sfh for plotting purposes
	input: str_sfh_parms = ['tage','tau','sf_start']
	returns: time-vector, plus SFR(t), where SFR is in units of Gyr^-1 (normalize by mass to get true SFR)
	SFR(t) = [len(TIME),len(COMPONENTS)+1,3]
	the 0th vector is total sfh, 1st is 1st component, etc
	third dimension is [q16,q50,q84]
	'''

	# sample randomly from SFH
	import copy
	flatchain = copy.copy(sample_results['flatchain'])
	np.random.shuffle(flatchain)

	# calculate minimum SFR for plotting purposes
	minsfr = sample_results['quantiles']['q50'][np.array(sample_results['model'].theta_labels()) == 'mass'] /  \
	        (sample_results['quantiles']['q50'][np.array(sample_results['model'].theta_labels()) == 'tage']*1e9*10000)

	# initialize output variables
	nt = 100
	intsfr = np.zeros(shape=(nsamp,nt,ncomp+1))
	deltat=0.0001

	for mm in xrange(nsamp):

		# SFH parameter vector
		sfh_params = threed_dutils.find_sfh_params(sample_results['model'],flatchain[mm,:],sample_results['obs'],sps)

		if mm == 0:
			# set up time vector
			idx = np.array(sample_results['model'].theta_labels()) == 'tage'
			maxtime = np.max(flatchain[:nsamp,idx])

			t,step=np.linspace(0,maxtime,num=nt,retstep=True)

		# calculate new time vector such that
		# the spacing from tage back to zero
		# is identical for each SFH model
		tcalc = t-sfh_params['tage']
		tcalc = tcalc[tcalc < 0]*-1
		#tcalc = tcalc[::-1]

		totmass = np.sum(sfh_params['mass'])
		for jj in xrange(len(tcalc)): 

			intsfr[mm,jj,0] = threed_dutils.calculate_sfr(sfh_params, deltat, tcalc = tcalc[jj],minsfr=minsfr)

		# now for each component
		for kk in xrange(ncomp):
			iterable = [(key,value[kk]) for key, value in sfh_params.iteritems()]
			newdict = {key: value for (key, value) in iterable}
			for jj in xrange(nt): intsfr[mm,jj,kk+1] = threed_dutils.calculate_sfr(newdict, deltat, tcalc = t[jj],minsfr=minsfr)

		# clip to minimum sfr, at 0.01% of average SFR over lifetime
		# this is only for values that were unfilled by the loop over tcalc
		intsfr = np.clip(intsfr,minsfr,np.inf)

	q = np.zeros(shape=(nt,ncomp+1,3))
	for kk in xrange(ncomp+1):
		for jj in xrange(nt): q[jj,kk,:] = np.percentile(intsfr[:,jj,kk],[16.0,50.0,84.0])

	return t, q

def create_plotquant(sample_results, logplot = ['mass'], truths=None):
    
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
	# COMMENTED OUT FOR NOW, FOR TRUTH TESTS
	# MAYBE REINTRODUCE LATER FOR REAL DATA
	#redefine = ['sf_start','sf_trunc']
	redefine = []

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
		extent = [np.percentile(sample_results['plotchain'][:,:,ii],0.5),
		          np.percentile(sample_results['plotchain'][:,:,ii],99.5)]

		# is the chain stuck at one point? if so, set the range equal to param*0.8,param*1.2
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
			extent = (extent[0]*0.8,extent[0]*1.2)
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
			if truths is not None and parnames[jj] == truths['parnames'][jj]:
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

		finite = np.isfinite(sample_results['lnprobability'])
		max = np.max(sample_results['lnprobability'][finite])
		min = np.percentile(sample_results['lnprobability'][finite],10)
		axarr[jj+1,ii].set_ylim(min, max+np.abs(max)*0.01)
		
		axarr[jj+1,ii].yaxis.get_major_ticks()[0].label1On = False # turn off bottom ticklabel


	if outname is not None:
		plt.savefig(outname, bbox_inches='tight',dpi=300)
		plt.close()

def return_sedplot_vars(thetas, sample_results, sps, nufnu=True):

	'''
	if nufnu == True: return in units of nu * fnu. Else, return maggies.
	'''

	# observational information
	mask = sample_results['obs']['phot_mask']
	wave_eff = sample_results['obs']['wave_effective'][mask]
	obs_maggies = sample_results['obs']['maggies'][mask]
	obs_maggies_unc = sample_results['obs']['maggies_unc'][mask]

	# model information
	spec, mu ,_ = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)
	mu = mu[mask]

	# output units
	if nufnu == True:
		c = 3e8
		factor = c*1e10
		mu *= factor/wave_eff
		spec *= factor/sps.wavelengths
		obs_maggies *= factor/wave_eff
		obs_maggies_unc *= factor/wave_eff

	# here we want to return
	# effective wavelength of photometric bands, observed maggies, observed uncertainty, model maggies, observed_maggies-model_maggies / uncertainties
	# model maggies, observed_maggies-model_maggies/uncertainties
	return wave_eff, obs_maggies, obs_maggies_unc, mu, (obs_maggies-mu)/obs_maggies_unc, spec, sps.wavelengths

def sed_figure(sample_results, sps, model,
                alpha=0.3, samples = [-1],
                maxprob=0, outname=None, fast=False,
                truths = None, agb_off = False,
                **kwargs):
	"""
	Plot the photometry for the model and data (with error bars), and
	plot residuals
	"""

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

	# for maximum probability model, plot the spectrum,
	# photometry, and chi values
	wave_eff, obsmags, obsmags_unc, modmags, chi, modspec, modlam = return_sedplot_vars(sample_results['quantiles']['maxprob_params'], sample_results, sps)

	phot.plot(np.log10(wave_eff), np.log10(modmags), color='#e60000', marker='o', ms=ms, linestyle=' ', label='max lnprob', alpha=alpha, markeredgewidth=0.7,**kwargs)
	
	res.plot(np.log10(wave_eff), chi, 
		     color='#e60000', marker='o', linestyle=' ', label='max lnprob', 
		     ms=ms,alpha=alpha,markeredgewidth=0.7,**kwargs)
	
	nz = modspec > 0
	phot.plot(np.log10(modlam[nz]), np.log10(modspec[nz]), linestyle='-',
              color='red', alpha=0.6,**kwargs)

	# plot AGB-off for Charlie
	if agb_off:
		sample_results['model'].params['add_agb_dust_model'] = np.array(False)
		_, _, _, _, _, modspec_off, modlam_off = return_sedplot_vars(sample_results['quantiles']['maxprob_params'], sample_results, sps)

		nz = modspec > 0
		phot.plot(np.log10(modlam_off[nz]), np.log10(modspec_off[nz]), linestyle='-',
              color='blue', alpha=0.6,label='AGB dust off',**kwargs)

	# define observations for later use
	xplot = np.log10(wave_eff)
	yplot = np.log10(obsmags)
	linerr_down = np.clip(obsmags-obsmags_unc, 1e-80, np.inf)
	linerr_up = np.clip(obsmags+obsmags_unc, 1e-80, np.inf)
	yerr = [yplot - np.log10(linerr_down), np.log10(linerr_up)-yplot]

	# set up plot limits
	phot.set_xlim(min(xplot)*0.9,max(xplot)*1.04)
	phot.set_ylim(min(yplot[np.isfinite(yplot)])*0.7,max(yplot[np.isfinite(yplot)])*1.04)
	res.set_xlim(min(xplot)*0.9,max(xplot)*1.04)

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
		
		# if truths are made with a different model than they are fit with,
		# then this will be passing parameters to the wrong model. pass.
		# in future, attach a model to the truths file!
		try:
			wave_eff, _, _, _, chi_truth, _, _ = return_sedplot_vars(sample_results['quantiles']['maxprob_params'], sample_results, sps)

			res.plot(np.log10(wave_eff_truth), chi_truth, 
				     color='blue', marker='o', linestyle=' ', label='truths', 
				     ms=ms,alpha=0.3,markeredgewidth=0.7,**kwargs)
		except AssertionError:
			pass

	# add SFH plot
	ax_loc = [0.2,0.35,0.12,0.14]
	add_sfh_plot(sample_results,fig,ax_loc,sps,truths=truths,fast=fast)

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
	chisq=np.sum(chi**2)
	ndof = np.sum(sample_results['obs']['phot_mask']) - len(sample_results['model'].free_params)-1
	reduced_chisq = chisq/(ndof-1)

	# also calculate for truths if truths exist
	if truths is not None:
		try:
			chisq_truth=np.sum(chi_truth**2)
			reduced_chisq_truth = chisq_truth/(ndof-1)
			phot.text(textx, texty, r'best-fit $\chi^2_n$='+"{:.2f}".format(reduced_chisq)+' (true='
				      +"{:.2f}".format(reduced_chisq_truth)+')',
				      fontsize=10, ha='right')
		except NameError:
			pass
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
	else:
		sn_txt = -99.0
		uvj_txt = -99.0
	z_txt = sample_results['model'].params['zred'][0]

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

def make_all_plots(filebase=None,
				   outfolder=os.getenv('APPS')+'/threedhst_bsfh/plots/',
				   sample_results=None,
				   sps=None,plt_chain=True,
				   plt_triangle=True,
				   plt_sed=True):

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

	# if we found no files, skip this object
	if len(times) == 0:
		print 'Failed to find any files to extract times in ' + folder + ' of form ' + filename
		return 0

	# load results
	mcmc_filename=filebase+'_'+max(times)+"_mcmc"
	model_filename=filebase+'_'+max(times)+"_model"

	# load if necessary
	if not sample_results:
		try:
			sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)
		except (EOFError,ValueError) as e:
			print e
			print 'Failed to open '+ mcmc_filename +','+model_filename
			return 0
	else:
		import pickle
		try:
			mf = pickle.load( open(model_filename, 'rb'))
		except(AttributeError):
			mf = load( open(model_filename, 'rb'))
       
		powell_results = mf['powell']

	if not sps:
		# load stellar population, set up custom filters
		if np.sum([1 for x in sample_results['model'].config_list if x['name'] == 'pmetals']) > 0:
			sps = threed_dutils.setup_sps(custom_filter_key=sample_results['run_params'].get('custom_filter_key',None))
		else:
			sps = threed_dutils.setup_sps(zcontinuous=1,
										  custom_filter_key=sample_results['run_params'].get('custom_filter_key',None))

	# BEGIN PLOT ROUTINE
	print 'MAKING PLOTS FOR ' + filename + ' in ' + outfolder
	
	# do we know the truths?
	try:
		truths = threed_dutils.load_truths(os.getenv('APPS')+'/threed'+sample_results['run_params']['truename'].split('/threed')[1],
			                              sample_results['run_params']['objname'],
			                              sample_results, sps=sps)
	except KeyError:
		truths=None

	# define nice plotting quantities
	sample_results = create_plotquant(sample_results, truths=truths)
	sample_results['extents'] = return_extent(sample_results)
    # chain plot
	if plt_chain: 
		print 'MAKING CHAIN PLOT'
		show_chain(sample_results,
	               outname=outfolder+filename+'_'+max(times)+".chain.png",
			       alpha=0.3,truths=truths)

	# triangle plot
	if plt_triangle: 
		print 'MAKING TRIANGLE PLOT'
		chopped_sample_results = copy.deepcopy(sample_results)

		subtriangle(sample_results, sps, copy.deepcopy(sample_results['model']),
							 outname=outfolder+filename+'_'+max(times),
							 showpars=None,start=0,
							 show_titles=True, truths=truths, powell_results=powell_results)

	# sed plot
	if plt_sed:
		print 'MAKING SED PLOT'
		
		# FAST fit?
		try:
			sample_results['run_params']['fastname']
			fast=1
		except:
			fast=0

 		# plot
 		pfig = sed_figure(sample_results, sps, copy.deepcopy(sample_results['model']),
 						  maxprob=1,fast=fast,truths=truths,
 						  outname=outfolder+filename+'_'+max(times)+'.sed.png')
 		
def plot_all_driver(runname=None,**extras):

	'''
	for a list of galaxies, make all plots
	'''
	if runname == None:
		runname = 'testsed_simha_truth'

	filebase, parm_basename, ancilname=threed_dutils.generate_basenames(runname)
	for jj in xrange(len(filebase)):
		print 'iteration '+str(jj) 
		make_all_plots(filebase=filebase[jj],\
		               outfolder=os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/',
		               **extras)
	