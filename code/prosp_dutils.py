import numpy as np
import os, fsps, sys, copy
from prospect.models import model_setup
from scipy.interpolate import interp1d
from scipy.integrate import simps
from astropy import constants
from dynesty.plotting import _quantile as wquant

def return_lir(lam,spec,z=None):
    """ returns IR luminosity (8-1000 microns) in erg/s
    input spectrum must be Lsun/Hz, wavelength in \AA
    """
    botlam = np.atleast_1d(8e4-1)
    toplam = np.atleast_1d(1000e4+1)
    edgetrans = np.atleast_1d(0)
    lir_filter = [[np.concatenate((botlam,np.linspace(8e4, 1000e4, num=100),toplam))],
                  [np.concatenate((edgetrans,np.ones(100),edgetrans))]]

    # calculate integral
    _,lir     = integrate_mag(lam,spec,lir_filter, z=z) # comes out in ergs/s

    return lir

def return_luv(lam,spec,z=None):
    """ returns UV luminosity (1216-3000 \AA) in erg/s
    input spectrum must be Lsun/Hz, wavelength in \AA
    """

    luv_filter = [[np.array([1215]+np.linspace(1216, 3000, num=100).tolist()+[3001])],
                  [np.array([0]+np.ones(100).tolist()+[0])]]

    # calculate integral
    _,luv     = integrate_mag(lam,spec,luv_filter, z=z) # comes out in ergs/s

    return luv


def return_lmir(lam,spec,z=None):
    """ returns MIR luminosity (4-20 microns) in erg/s
    input spectrum must be Lsun/Hz, wavelength in \AA
    """
    botlam, toplam = 4e4-1, 20e4+1
    lmir_filter = [[np.array([botlam]+np.linspace(botlam+1, toplam-1, num=100).tolist()+[toplam])],
                  [np.array([0]+np.ones(100).tolist()+[0])]]

    # calculate integral
    _,lmir     = integrate_mag(lam,spec,lmir_filter, z=z) # comes out in ergs/s

    return lmir

def sfr_uvir(lir,luv):
    """inputs in Lsun. Calculates UV+IR SFR from Whitaker+14
    output is Msun/yr, in Chabrier IMF"""
    return 1.09e-10*(lir + 2.2*luv)

def smooth_spectrum(lam,spec,sigma,
                    minlam=0.0,maxlam=1e50):     

    '''
    ripped from Charlie Conroy's smoothspec.f90
    the 'fast way'
    integration is truncated at +/-4*sigma
    '''
    c_kms = 2.99e5
    int_trunc=4
    spec_out = copy.copy(spec)

    for ii in xrange(len(lam)):
        if lam[ii] < minlam or lam[ii] > maxlam:
            spec_out[ii] = spec[ii]
            continue

        dellam = lam[ii]*(int_trunc*sigma/c_kms+1)-lam[ii]
        integrate_lam = (lam > lam[ii]-dellam) & (lam < lam[ii]+dellam)

        if np.sum(integrate_lam) <= 1:
            spec_out[ii] = spec[ii]
        else:
            vel = (lam[ii]/lam[integrate_lam]-1)*c_kms
            func = 1/np.sqrt(2*np.pi)/sigma * np.exp(-vel**2/2./sigma**2)
            dx=np.abs(np.diff(vel)) # we want absolute value
            func = func / np.trapz(func,dx=dx)
            spec_out[ii] = np.trapz(func*spec[integrate_lam],dx=dx)

    return spec_out

def asym_errors(center, up, down, log=False):

    if log:
        errup = np.log10(up)-np.log10(center)
        errdown = np.log10(center)-np.log10(down)
        errarray = [errdown,errup]
    else:
        errarray = [center-down,up-center]

    return errarray

def running_sigma(x,y,nbins=10,bins=None):

    if bins is None:
        bins = np.linspace(x.min(),x.max()*1.001, nbins+1)
    else:
        nbins = len(bins)-1
        
    idx  = np.digitize(x,bins)

    # RUNNING SIGMA
    siglow, sighigh = np.empty(nbins), np.empty(nbins)
    for k in range(nbins): siglow[k], sighigh[k] = np.percentile(y[idx-1==k],[16,84])
    outbins = (bins[:-1]+bins[1:])/2.

    return outbins,np.array(siglow),np.array(sighigh)

def running_median(x,y,nbins=10,avg=False,weights=None,bins=None,return_bincount=False):

    if bins is None:
        bins = np.linspace(x.min(),x.max()*1.001, nbins+1)
    else:
        nbins = len(bins)-1
        
    idx  = np.digitize(x,bins)

    # median
    if not avg:

        if weights is None:
            running_median = np.array([np.median(y[idx-1==k]) for k in range(nbins)])
        else:
            running_median = []
            for k in range(nbins): 
                index = (idx-1==k)
                if index.sum() == 0:
                    running_median += [[0.0,0.0,0.0]]
                elif index.sum() == 1:
                    running_median += [[y[index][0],y[index][0],y[index][0]]]
                else:
                    weight = weights[index] / weights[index].sum()
                    running_median += [wquant(y[index],np.array([0.5, 0.84, 0.16]),weights=weight)]
            running_median = np.array(running_median)

    # average
    if avg: 
        running_median= []
        for k in range(nbins): # for loop because we can't pass an empty array to np.average!
            if (idx-1==k).sum() == 0:
                running_median.append(0.0)
            else:
                if weights is None:
                    weight = None
                else:
                    weight = weights[idx-1==k]
                running_median.append(np.average(y[idx-1==k],weights=weight))
        running_median = np.array(running_median)
    outbins = (bins[:-1]+bins[1:])/2.

    if return_bincount:
        return outbins,running_median, np.array([(idx-1 == i).sum() for i in range(nbins)])
    return outbins,running_median

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

def normalize_error_asym(obs,mod):
    
    # define output
    out = np.zeros_like(obs[:,0])
    
    # define errors
    edown_mod = mod[:,0] - mod[:,2]
    eup_mod = mod[:,1] - mod[:,0]
    edown_obs = obs[:,0] - obs[:,2]
    eup_obs = obs[:,1] - obs[:,0]

    # find out which side of error bar to use
    undershot = obs[:,0] > mod[:,0]
    
    # create output
    out[undershot] = (obs[undershot,0] - mod[undershot,0]) / np.sqrt(eup_mod[undershot]**2+edown_obs[undershot]**2)
    out[~undershot] = (obs[~undershot,0] - mod[~undershot,0]) / np.sqrt(edown_mod[~undershot]**2+eup_obs[~undershot]**2)

    return out

def equalize_axes(ax, x,y, dynrange=0.1, line_of_equality=True, axlims=None, log_in_linear=False):
    """ defines a plot range that encompasses all of the data
    line_of_equality: add a dashed diagonal line of equality 
    dynrange: the percent of the data range above and below which the plot limits are set
    log_in_linear: we are passed LOG variables but want ranges in LINEAR space
    """

    dynx, dyny = (np.nanmax(x)-np.nanmin(x))*dynrange,\
                 (np.nanmax(y)-np.nanmin(y))*dynrange
    
    if axlims is None:
        if np.nanmin(x)-dynx > np.nanmin(y)-dyny:
            min = np.nanmin(y)-dyny
        else:
            min = np.nanmin(x)-dynx
        if np.nanmax(x)+dynx > np.nanmax(y)+dyny:
            max = np.nanmax(x)+dynx
        else:
            max = np.nanmax(y)+dyny
    else:
        min = axlims[0]
        max = axlims[1]

    if log_in_linear:
        min = 10**min
        max = 10**max

    ax.set_xlim(min,max)
    ax.set_ylim(min,max)

    if line_of_equality:
        ax.plot([min,max],[min,max],linestyle='--',color='0.1',alpha=0.8)
    return ax

def offset_and_scatter(x,y,biweight=False,mad=False):

    n = len(x)
    mean_offset = np.nanmean(y-x)

    if biweight:
        diff = y-x
        Y0  = np.median(diff)

        # calculate MAD
        MAD = np.median(np.abs(diff-Y0))/0.6745

        # biweighted value
        U   = (diff-Y0)/(6.*MAD)
        UU  = U*U
        Q   = UU <= 1.0
        if np.sum(Q) < 3:
            print 'distribution is TOO WEIRD, returning -1'
            scat=-1

        N = len(diff)
        numerator = np.sum( (diff[Q]-Y0)**2 * (1-UU[Q])**4)
        den1      = np.sum( (1.-UU[Q])*(1.-5.*UU[Q]))
        siggma    = N*numerator/(den1*(den1-1.))

        scat      = np.sqrt(siggma)

    elif mad:
        from astropy.stats import median_absolute_deviation
        diff = y-x
        scat = median_absolute_deviation(diff)

    else:
        scat=np.sqrt(np.sum((y-x-mean_offset)**2.)/(n-2))

    return mean_offset,scat

def find_sfh_params(model,theta,obs,sps,sm=None):

    # pass theta to model
    model.set_parameters(theta)

    # find all variables in `str_sfh_parms`
    str_sfh_params = ['sfh','mass','tau','sf_start','tage','sf_trunc','sf_slope','agebins','sfr_fraction','logsfr']
    sfh_out = []
    for string in str_sfh_params:
        if string in model.params:
            sfh_out.append(np.atleast_1d(model.params[string]))
        else:
            sfh_out.append(np.array([]))

    # turn into a dictionary
    iterable = [(str_sfh_params[ii],sfh_out[ii]) for ii in range(len(sfh_out))]
    out = {key: value for (key, value) in iterable}

    # if we pass stellar mass from a prior model call,
    # we don't have to calculate it here
    if sm is None and out['mass'].shape[0] == 1:
        _,_,sm_new=model.sed(theta, obs, sps=sps)
        try:
            sm = sps.csp.stellar_mass
        except AttributeError as e: 
            sm = sm_new

    ### create mass fractions for nonparametric SFHs
    # first old-style
    if (out['sfr_fraction'].shape[0] != 0):
        out['sfr_fraction_full'] = np.concatenate((out['sfr_fraction'],np.atleast_1d(1-out['sfr_fraction'].sum())))
        time_per_bin = []
        for (t1, t2) in sps.params['agebins']: time_per_bin.append(10**t2-10**t1)
        out['mass_fraction'] = out['sfr_fraction_full']*np.array(time_per_bin)
        out['mass_fraction'] /= out['mass_fraction'].sum()
        out['mformed'] = copy.copy(out['mass'])
        out['mass'] *= sm
    # now new-style nonparametric
    elif out['mass'].shape[0] > 1:
        out['mass_fraction'] = out['mass']/out['mass'].sum()
        out['mformed'] = out['mass'].sum()
        out['mass'] = out['mass'].sum()*sm
    else:
        # Need this because mass is 
        # current mass, not total mass formed!
        out['mformed'] = out['mass']
        out['mass'] *= sm

    return out

def test_likelihood(sps,model,obs,thetas,param_file):

    '''
    skeleton:
    load up some model, instantiate an sps, and load some observations
    generate spectrum, compare to observations, assess likelihood
    can be run in different environments as a test
    '''

    from prospect.likelihood import lnlike_spec,lnlike_phot

    run_params = model_setup.get_run_params(param_file=param_file)

    if sps is None:
        sps = model_setup.load_sps(**run_params)

    if model is None:
        model = model_setup.load_model(**run_params)

    if obs is None:
        obs = model_setup.load_obs(**run_params)

    if thetas is None:
        thetas = np.array(model.initial_theta)

    spec_noise, phot_noise = model_setup.load_gp(**run_params)

    # generate photometry
    mu, phot, x = model.mean_model(thetas, obs, sps=sps)

    # Noise modeling
    if spec_noise is not None:
        spec_noise.update(**model.params)
    if phot_noise is not None:
        phot_noise.update(**model.params)
    vectors = {'spec': mu, 'unc': obs['unc'],
               'sed': model._spec, 'cal': model._speccal,
               'phot': phot, 'maggies_unc': obs['maggies_unc']}

    # Calculate likelihoods
    lnp_prior = model.prior_product(thetas)
    lnp_spec = lnlike_spec(mu, obs=obs, spec_noise=spec_noise, **vectors)
    lnp_phot = lnlike_phot(phot, obs=obs, phot_noise=phot_noise, **vectors)

    return lnp_prior + lnp_phot + lnp_spec

def synthetic_halpha(sfr,dust1,dust2,dust1_index,dust2_index,kriek=False):

    '''
    SFR in Msun/yr
    mass in Msun
    '''

    # calculate Halpha luminosity from Kennicutt relationship
    # comes out in units of [ergs/s]
    # correct from Chabrier to Salpeter with a factor of 1.7
    flux = 1.26e41 * (sfr*1.7)
    lam     = 6563.0

    # correct for dust
    # if dust_index == None, use Calzetti
    if dust2_index is not None:
        flux=flux*charlot_and_fall_extinction(lam,dust1,dust2,dust1_index,dust2_index,kriek=kriek)
    else:
        Rv   = 4.05
        klam = 2.659*(-2.156+1.509/(lam/1e4)-0.198/(lam/1e4)**2+0.011/(lam/1e4)**3)+Rv
        A_lam = klam/Rv*dust2

        flux = flux*10**(-0.4*A_lam)

    # comes out in ergs/s
    return flux

def bdec_to_ext(bdec):
    return np.log10(bdec/2.86)

def chev_extinction(tau_v, lam, ebars=False):
    '''
    return optical depth based on Eqn 7+9,10 in Chevallard et al. 2013
    lambda must be in angstroms
    if ebars==True, also return the +/- 1 sigma
    this only works perfectly at lambda = 5500!
    must include error in b_tauv if you do it elsewhere!

    returns (center, up, down)
    '''

    n_v = 2.8 / (1. + 3.*np.sqrt(tau_v))
    b_tauv = 0.3-0.05*tau_v
    n_lam = n_v + b_tauv*(lam/1e4-0.55)
    tau_return = tau_v * (lam/5500.)**(-n_lam)

    if ebars:
        n_v_up = n_v*1.25
        n_v_do = n_v*0.75
        n_lam_up = n_v_up + b_tauv*(lam/1e4-0.55)
        n_lam_do = n_v_do + b_tauv*(lam/1e4-0.55)

        return np.array([tau_return, tau_v*(lam/5500.)**(-n_lam_up),tau_v*(lam/5500.)**(-n_lam_do)])
    else:
        return tau_return


def charlot_and_fall_extinction(lam,dust1,dust2,dust1_index,dust2_index, kriek=False, nobc=False, nodiff=False):

    '''
    returns F(obs) / F(emit) for a given attenuation curve + dust1 + dust2
    '''

    dust1_ext = np.exp(-dust1*(lam/5500.)**dust1_index)
    dust2_ext = np.exp(-dust2*(lam/5500.)**dust2_index)

    # sanitize inputs
    lam = np.atleast_1d(lam)

    # are we using Kriek & Conroy 13?
    if kriek == True:
        dd63=6300.00
        lamv=5500.0
        dlam=350.0
        lamuvb=2175.0

        #Calzetti curve, below 6300 Angstroms, else no addition
        cal00 = np.zeros_like(lam)
        gt_dd63 = lam > dd63
        le_dd63 = ~gt_dd63
        if np.sum(gt_dd63) > 0:
            cal00[gt_dd63] = 1.17*( -1.857+1.04*(1e4/lam[gt_dd63]) ) + 1.78
        if np.sum(le_dd63) > 0:
            cal00[le_dd63]  = 1.17*(-2.156+1.509*(1e4/lam[le_dd63])-0.198*(1E4/lam[le_dd63])**2 + 0.011*(1E4/lam[le_dd63])**3) + 1.78
        cal00 = cal00/0.44/4.05 

        eb = 0.85 - 1.9 * dust2_index  #KC13 Eqn 3

        #Drude profile for 2175A bump
        drude = eb*(lam*dlam)**2 / ( (lam**2-lamuvb**2)**2 + (lam*dlam)**2 )

        attn_curve = dust2*(cal00+drude/4.05)*(lam/lamv)**dust2_index
        dust2_ext = np.exp(-attn_curve)

    if nobc:
        ext_tot = dust2_ext
    elif nodiff:
        ext_tot = dust1_ext
    else:
        ext_tot = dust2_ext*dust1_ext

    return ext_tot

def calc_balmer_dec(tau1, tau2, ind1, ind2,kriek=False):

    ha_lam = 6562.801
    hb_lam = 4861.363
    balm_dec = 2.86*charlot_and_fall_extinction(ha_lam,tau1,tau2,ind1,ind2,kriek=kriek) / \
                    charlot_and_fall_extinction(hb_lam,tau1,tau2,ind1,ind2,kriek=kriek)
    return balm_dec

def exp_decl_sfh_half_time(tage,tau):
    ''' integrate SFR = Ae^(-t/tau)
    note that this returns YEARS AGO that the half mass was reached
    so a larger half-mass time is an OLDER galaxy
    '''
    return tage-tau*np.log(2./(1+np.exp(-tage/tau)))

def g05_prior(size=1000):

    from prosp_dutils import exp_decl_sfh_avg_age_with_burst as avg_age

    # random tage
    tmin, tmax = 1, 13.5
    tage = (tmax - tmin) * np.random.random_sample(size) + tmin

    # random tau
    gmin, gmax = 0.0, 1
    gamma = (gmax-gmin) * np.random.random_sample(size) + gmin
    tau = 1./gamma

    # random burst amplitude (Mburst / Mnotburst)
    amin, amax = 0.03,4
    aburst = (amax-amin) * np.random.random_sample(size) + amin

    # random burst duration
    #dtburst_min, dtburst_max = 0.03,0.3
    # dtburst = (dtburst_max-dtburst_min) * np.random.random_sample(size) + dtburst_min

    # each 2 Gyr has a 10% chance of a burst happening
    # so assume a random burst happened in the last 20 gyr
    # if burst > tage, then it does NOT happen
    tburst = np.random.random_sample(size)*20

    tavg = np.array([avg_age(tage[i],tau[i],aburst[i],tburst[i]) for i in range(size)])

    return tavg

def exp_decl_sfh_avg_age(tage,tau):
    ''' this is done numerically
    '''
    t = np.linspace(0,tage,1000)
    tavg = np.trapz(t*np.exp(-t/tau),x=t) / np.trapz(np.exp(-t/tau),x=t)
    return tage-tavg

def linexp_decl_sfh_avg_age(tage,tau):
    ''' this is done numerically
    '''
    t = np.linspace(0,tage,1000)
    tavg = np.trapz((t**2)*np.exp(-t/tau),x=t) / np.trapz(t*np.exp(-t/tau),x=t)
    return tage-tavg

def sfh_half_time(x,sfh_params,c):
    """wrapper for use with halfmass assembly time
    """

    # check for nonparametric
    if sfh_params['sf_start'].shape[0] == 0:
        sf_start = 0.0
    else:
        sf_start = sfh_params['sf_start']
    return integrate_sfh(sf_start,x,sfh_params)-c

def all_ages(theta,mod,sps):
    """calculates light-weighted (L_bol and r-band) ages and mass-weighted ages
    all in Gyr
    """
    # zred=0
    zsave = mod.params['zred']
    mod.params['zred'] = 0.0

    # no dust
    ndust_thetas = copy.copy(theta)
    for par in ['dust1','dust1_fraction','dust2']:
        if par in mod.theta_labels():
            ndust_thetas[mod.theta_index[par]] = 0.0

    # fake obs
    fake_obs = {'maggies': None, 'phot_mask': None, 'wavelength': None,  'filters': []}

    # Lbol light-weighted and mass-weighted ages
    sps.ssp.params['compute_light_ages'] = True
    spec, mags, sm = mod.mean_model(ndust_thetas, fake_obs, sps=sps)
    lwa_lbol = sps.ssp.log_lbol
    mwa = sps.ssp.stellar_mass

    # and this gets us the r-band light weighted age
    # this must come after the model call above, which sets tabular SFH etc
    if hasattr(sps,'convert_sfh'):
        _, _, tmax = sps.convert_sfh(sps.params['agebins'], sps.params['mass'])
    else:
        tmax = sps.ssp.params['tage']
    lwa_rband = sps.ssp.get_mags(bands=['SDSS_r'],tage=tmax)[0]

    # restore defaults and return
    mod.params['zred'] = zsave
    sps.ssp.params['compute_light_ages'] = False

    return mwa, lwa_lbol, lwa_rband

def massweighted_age(sfh_params):
    """calculate mass-weighted age.
    currently only works for nonparameteric SFH
    """
    avg_age_per_bin = (10**sfh_params['agebins']).sum(axis=1)/2.
    mwa = (sfh_params['mass_fraction'] * avg_age_per_bin).sum()/1e9
    return mwa

def halfmass_assembly_time(sfh_params):

    from scipy.optimize import brentq

    # calculate half-mass assembly time
    # c = 0.5 if half-mass assembly time occurs before burst
    try:
        half_time = brentq(sfh_half_time, 0,14,
                           args=(sfh_params,0.5),
                           rtol=1.48e-08, maxiter=1000)
    except ValueError:
        # big problem
        warnings.warn("You've passed SFH parameters that don't allow t_half to be calculated. Check for bugs.", UserWarning)
        half_time = np.nan

    # define age of galaxy
    tgal = sfh_params['tage']
    if tgal.shape[0] == 0:
        tgal = np.max(10**sfh_params['agebins']/1e9)

    return tgal-half_time

def generate_basenames(runname,ancilname=None):

    filebase, parm = [], []

    if runname == 'brownseds_agn':

        id_list = os.getenv('APPS')+"/prospector_alpha/data/"+runname+".ids"
        ids = np.loadtxt(id_list, dtype='|S20',delimiter=',')
        ngals = len(ids)

        parm_basename = runname+"_params"
        ancilname=None

        for jj in xrange(ngals):
            filebase.append(os.getenv('APPS')+"/prospector_alpha/results/"+runname+'/'+runname+'_'+ids[jj])
            parm.append(os.getenv('APPS')+"/prospector_alpha/parameter_files/"+runname+'/'+parm_basename+'_'+str(jj+1)+'.py') 

    elif 'brownseds' in runname:

        id_list = os.getenv('APPS')+'/prospector_alpha/data/brownseds_data/photometry/namelist.txt'
        ids = np.loadtxt(id_list, dtype='|S20',delimiter=',')
        ngals = len(ids)

        basename = runname
        parm_basename = basename+"_params"
        ancilname=None

        for jj in xrange(ngals): filebase.append(os.getenv('APPS')+"/prospector_alpha/results/"+runname+'/'+ids[jj])
        
        # check for multiple parameter files
        parbase = os.getenv('APPS')+"/prospector_alpha/parameter_files/"+runname+'/'+parm_basename
        if os.path.isfile(parbase+'_1.py'):
            for jj in xrange(ngals): parm.append(parbase+'_'+str(jj+1)+'.py')
        else:
            parm = parbase+'.py'

    elif runname=='td_delta':

        id_list = os.getenv('APPS')+"/prospector_alpha/data/3dhst/td_new.ids"
        ids = np.loadtxt(id_list, dtype='|S60',delimiter=',')
        ngals = len(ids)

        parm_basename = runname+"_params"
        ancilname=None

        for jj in xrange(ngals):
            filebase.append(os.getenv('APPS')+'/prospector_alpha/results/'+runname+'/'+ids[jj].split('_')[0]+'/'+ids[jj])
            parm.append(os.getenv('APPS')+"/prospector_alpha/parameter_files/"+parm_basename+'.py') 

    elif 'td' in runname:

        id_list = os.getenv('APPS')+"/prospector_alpha/data/3dhst/"+runname+".ids"
        ids = np.loadtxt(id_list, dtype='|S60',delimiter=',')
        ngals = len(ids)

        parm_basename = runname+"_params"
        ancilname=None

        for jj in xrange(ngals):
            filebase.append(os.getenv('APPS')+'/prospector_alpha/results/'+runname+'/'+ids[jj])
            parm.append(os.getenv('APPS')+"/prospector_alpha/parameter_files/"+parm_basename+'.py') 

    elif 'vis' in runname:

        ids = ['vis_'+str(id) for id in np.arange(10)+1]
        filebase = [os.getenv('APPS')+'/prospector_alpha/results/'+runname+'/'+id for id in ids]
        ancilname = '_'

    else:

        # read file names directly from the results folder
        results_dir = os.getenv('APPS')+"/prospector_alpha/results/"+runname
        all_ids = ["_".join(f.split('_')[:-2]) for f in os.listdir(results_dir)]
        ids = list(set(all_ids))

        # construct outputs
        parm = os.getenv('APPS')+"/prospector_alpha/parameter_files/"+runname+"_params.py"
        filebase = [os.getenv('APPS')+"/prospector_alpha/results/"+runname+'/'+id for id in ids]
        ancilname = None

    return filebase,parm,ancilname

def chop_chain(chain, convergence_check_interval=None, convergence_chunks=325,
               convergence_stable_points_criteria=3, nchop=1.15, weights=None, size=3e5,**extras):
    '''
    if we used emcee, either (a) use the final 1/4th of the chain, or (b) use KL divergence convergence criteria
    if we used nestle, sample chain nestle_nsample times according to weights
    '''

    if weights is not None:
        flatchain = np.random.choice(chain, size=size, p=weights)
        return flatchain
    
    if convergence_check_interval is None:
        nchop= int(chain.shape[1]/nchop)
    else:
        nchop = -convergence_check_interval*convergence_stable_points_criteria

    if len(chain.shape) == 3:
        flatchain = chain[:,nchop:,:]
        flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1],
                                      flatchain.shape[2])
    else:
        flatchain = chain[:,nchop:]
        flatchain = flatchain.reshape(flatchain.shape[0] * flatchain.shape[1])

    return flatchain

def gaussian(x, mu, sig):
    '''
    can't believe there's not a package for this
    x is x-values, mu is mean, sig is sigma
    '''
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def gas_met(nii_ha,oiii_hb):
    # return gas-phase metallicity based on NII / Halpha ratio
    # 
    # uses the following relationships from Maiolino et al. 2008:
    #       x = log (Z/Zsol) = 12 + log(O/H) - 8.69, Allende Priete et al. 2001
    #       c0: -0.7732 c1: 1.2357  c2: -0.2811 c3: -0.7201 c4:-0.3330  [NII/Halpha]
    #       c0: 0.1549  c1: -1.5031 c2: -0.9790 c3: -0.0297 [OIII 5007 / Hbeta] 
    #       log R = c0 + c1 x + c2 x**2 + c3 x**3 + c4 x**4

    #### find solutions
    c1 = [-0.7732 - np.log10(nii_ha), 1.2357, -0.2811, -0.7201, -0.3330]
    c2 = [0.1549 - np.log10(oiii_hb), -1.5031, -0.9790, -0.0297]

    #### dummy metallicity array, NII / Ha
    # used to determine where to look with OIII + Hb
    # since that is double-valued
    z_solution = np.linspace(-3.0,0.5,500)
    zeros_guess = c1[0] + c1[1]*z_solution + c1[2]*z_solution**2 + c1[3]*z_solution**3 + c1[4]*z_solution**4
    closest_niiha = z_solution[np.abs(zeros_guess).argmin()]

    #### dummy metallicity array, OIII / Hb
    inflection_point = 7.9-8.69 # estimated from Fig 5 of Maiolino+08
    if closest_niiha < inflection_point:
        z_solution = np.linspace(-4.0,inflection_point,500)
    else:
        z_solution = np.linspace(inflection_point,0.7,500)

    zeros_guess = c2[0] + c2[1]*z_solution + c2[2]*z_solution**2 + c2[3]*z_solution**3
    closest = z_solution[np.abs(zeros_guess).argmin()]

    return closest
    #return (closest+closest_niiha)/2.

def integral_average(x,y,x0,x1):
    """to do a definite integral over a given x,y array
    you have to redefine the x,y array to only exist over
    the relevant range
    """

    xarr_new = np.linspace(x0, x1, 400)
    bad = xarr_new == 0.0
    if np.sum(bad) > 0:
        xarr_new[bad]=1e-10
    yarr_new = np.interp(xarr_new,x,y)

    I1 = simps(yarr_new, xarr_new) / (x1 - x0)

    return I1

def av_to_dust2(av):
    return av/1.086

def integrate_mag(spec_lam,spectra,filter, z=None):

    '''
    borrowed from calc_ml
    given a filter name and spectrum, calculate magnitude/luminosity in filter (see alt_file for filter names)
    INPUT: 
        SPEC_LAM: must be in angstroms. this will NOT BE corrected for reddening even if redshift is specified. this
        allows you to calculate magnitudes in rest or observed frame.
        SPECTRA: must be in Lsun/Hz (FSPS standard). if redshift is specified, the normalization will be taken care of.
    OUTPUT:
        MAG: comes out as absolute magnitude
        LUMINOSITY: comes out in erg/s
            NOTE: if redshift is specified, INSTEAD RETURN apparent magnitude and flux [erg/s/cm^2]
    '''

    resp_lam = filter[0][0]
    res      = filter[1][0]

    # physical units, in CGS
    pc2cm = 3.08568E18
    lsun  = 3.839E33
    c     = 2.99E10

    # interpolate filter response onto spectral lambda array
    # when interpolating, set response outside wavelength range to be zero.
    response_interp_function = interp1d(resp_lam,res, bounds_error = False, fill_value = 0)
    resp_interp = response_interp_function(spec_lam)
    
    # integrate spectrum over filter response
    # first calculate luminosity: convert to flambda (factor of c/lam^2, with c converted to AA/s)
    # then integrate over flambda [Lsun/AA] to get Lsun
    spec_flam = spectra*(c*1e8/(spec_lam**2))
    luminosity = simps(spec_flam*resp_interp,spec_lam)
    
    # now calculate luminosity density [erg/s/Hz] in filter
    # this involves normalizing the filter response by integrating over wavelength
    norm = simps(resp_interp/spec_lam,spec_lam)
    luminosity_density = simps(spectra*(resp_interp/norm)/spec_lam,spec_lam)

    # if redshift is specified, convert to flux and apparent magnitude
    if z is not None:
        from astropy.cosmology import WMAP9

        dfactor = (WMAP9.luminosity_distance(z).value*1e5)**(-2)*(1+z)
        luminosity = luminosity*dfactor
        luminosity_density = luminosity_density*dfactor

    # convert luminosity density to flux density
    # the units of the spectra are Lsun/Hz; convert to
    # erg/s/cm^2/Hz, at 10pc for absolute mags
    flux_density = luminosity_density*lsun/(4.0*np.pi*(pc2cm*10)**2)
    luminosity   = luminosity*lsun

    # convert flux density to magnitudes in AB system
    mag = -2.5*np.log10(flux_density)-48.60

    #print 'maggies: {0}'.format(10**(-0.4*mag)*1e10)
    return mag, luminosity

def load_filter_response(filter, alt_file=None):
    '''READS FILTER RESPONSE CURVES FOR FSPS'''
    
    if not alt_file:
        filter_response_curve = os.getenv('SPS_HOME')+'/data/allfilters.dat'
    else:
        filter_response_curve = alt_file

    # initialize output arrays
    lam,res = (np.zeros(0) for i in range(2))

    # upper case?
    if filter.lower() == filter:
        lower_case = True
    else:
        lower_case = False

    # open file
    with open(filter_response_curve, 'r') as f:
        # Skips text until we find the correct filter
        for line in f:
            if lower_case:
                if line.lower().find(filter) != -1:
                    break
            else:
                if line.find(filter) != -1:
                    break
        # Reads text until we hit the next filter
        for line in f:  # This keeps reading the file
            if line.find('#') != -1:
                break
            # read line, extract data
            data = line.split()
            lam = np.append(lam,float(data[0]))
            res = np.append(res,float(data[1]))

    if len(lam) == 0:
        print "Couldn't find filter " + filter + ': STOPPING'
        print 1/0

    return lam, res

def return_full_sfh(t, sfh_params,**kwargs):

    '''
    returns full SFH given a time vector [in Gyr] and a
    set of SFH parameters
    '''

    deltat=1e-8 # Gyr

    # calculate new time vector such that
    # the spacing from tage back to zero
    # is identical for each SFH model
    try:
        tcalc = t-sfh_params['tage']
    except ValueError:
        tcalc = t-np.max(10**sfh_params['agebins'])/1e9
    tcalc = tcalc[tcalc < 0]*-1

    sfr = np.zeros(len(t))
    for mm in range(len(tcalc)): 
        sfr[mm] = calculate_sfr(sfh_params, deltat, tcalc = tcalc[mm], **kwargs)

    return sfr

def calculate_sfr(sfh_params, timescale, tcalc = None, 
                  minsfr = None, maxsfr = None):

    '''
    standardized SFR calculator. returns SFR averaged over timescale.

    SFH_PARAMS: standard input
    TIMESCALE: timescale over which to calculate SFR. timescale must be in Gyr.
    TCALC: at what point in the SFH do we want the SFR? If not specified, TCALC is set to sfh_params['tage']
    MINSFR: minimum returned SFR. if not specified, minimum is 0.01% of average SFR over lifetime
    MAXSFR: maximum returned SFR. if not specified, maximum is infinite.

    returns in [Msun/yr]

    '''

    if sfh_params['sfh'] > 0:
        tage = sfh_params['tage']
    else:
        tage = np.max(10**sfh_params['agebins']/1e9)

    if tcalc is None:
        tcalc = tage

    sfr = integrate_sfh(tcalc-timescale, tcalc, sfh_params) * \
                        sfh_params['mformed'].sum()/(timescale*1e9)

    if minsfr is None:
        minsfr = sfh_params['mformed'].sum() / (tage*1e9*10000)

    if maxsfr is None:
        maxsfr = np.inf

    sfr = np.clip(sfr, minsfr, maxsfr)

    return sfr

def transform_zfraction_to_sfrfraction(zfraction):
    '''vectorized and without I/O keywords
    '''
    if zfraction.ndim == 1:
        zfraction = np.atleast_2d(zfraction).transpose()
    sfr_fraction = np.zeros_like(zfraction)
    sfr_fraction[:,0] = 1-zfraction[:,0]
    for i in xrange(1,sfr_fraction.shape[1]): sfr_fraction[:,i] = np.prod(zfraction[:,:i],axis=1)*(1-zfraction[:,i])
    #sfr_fraction[:,-1] = np.prod(zfraction,axis=1)  #### THIS IS SET IMPLICITLY
    return sfr_fraction

def integrate_exp_tau(t1,t2,sfh):

    return sfh['tau'][0]*(np.exp(-t1/sfh['tau'][0])-np.exp(-t2/sfh['tau'][0]))

def integrate_delayed_tau(t1,t2,sfh):

    return (np.exp(-t1/sfh['tau'])*(1+t1/sfh['tau']) - \
           np.exp(-t2/sfh['tau'])*(1+t2/sfh['tau']))*sfh['tau']**2

def integrate_linramp(t1,t2,sfh):

    # integration constant: SFR(sf_trunc-sf_start)
    cs = (sfh['sf_trunc']-sfh['sf_start'])*(np.exp(-(sfh['sf_trunc']-sfh['sf_start'])/sfh['tau']))

    # enforce positive SFRs
    # by limiting integration to where SFR > 0
    t_zero_cross = -1.0/sfh['sf_slope'] + sfh['sf_trunc']
    if t_zero_cross > sfh['sf_trunc']-sfh['sf_start']:
        t1           = np.clip(t1,sfh['sf_trunc']-sfh['sf_start'],t_zero_cross)
        t2           = np.clip(t2,sfh['sf_trunc']-sfh['sf_start'],t_zero_cross)

    # initial integral: SFR = SFR[t=t_trunc] * [1 + s(t-t_trunc)]
    intsfr = cs*(t2-t1)*(1-sfh['sf_trunc']*sfh['sf_slope']) + cs*sfh['sf_slope']*0.5*((t2+sfh['sf_start'])**2-(t1+sfh['sf_start'])**2)

    return intsfr

def integrate_sfh(t1,t2,sfh_params):
    """ integrate star formation history from t1 to t2
    returns fraction of *total* (not stellar) mass formed in time inteval
    """

    # copy so we don't overwrite values
    sfh = sfh_params.copy()

    # if we're using a parameterized SFH
    if sfh_params['sfh'] > 0:

        # make sure we have an sf_start
        if (sfh['sf_start'].shape[0] == 0):
            sfh['sf_start'] = np.atleast_1d(0.0)

        # here is our coordinate transformation to match fsps
        t1 = t1-sfh['sf_start'][0]
        t2 = t2-sfh['sf_start'][0]

        # match dimensions, if two-tau model
        ndim = len(np.atleast_1d(sfh['mass']))
        if len(np.atleast_1d(t2)) != ndim:
            t2 = np.zeros(ndim)+t2
        if len(np.atleast_1d(t1)) != ndim:
            t1 = np.zeros(ndim)+t1

        # redefine sf_trunc, if not being used for sfh=5 purposes
        if (sfh['sf_trunc'] == 0.0) or sfh['sf_trunc'].shape[0] == 0:
            sfh['sf_trunc'] = sfh['tage']

        # if we're outside of the time boundaries, clip to boundary values
        # this only affects integrals which would have been questionable in the first place
        t1 = np.clip(t1,0,float(sfh['tage']-sfh['sf_start']))
        t2 = np.clip(t2,0,float(sfh['tage']-sfh['sf_start']))

        # if we're using normal tau
        if (sfh['sfh'] == 1):

            # add tau model
            intsfr = integrate_exp_tau(t1,t2,sfh)
            norm = sfh['tau'][0]*(1-np.exp(-(sfh['sf_trunc'][0]-sfh['sf_start'][0])/sfh['tau'][0]))
            intsfr = intsfr/norm

        # if we're using delayed tau
        elif (sfh['sfh'] == 4):

            # add tau model
            intsfr =  integrate_delayed_tau(t1,t2,sfh)
            norm =    1.0-np.exp(-(sfh['sf_trunc'][0]-sfh['sf_start'][0])/sfh['tau'][0])*(1+(sfh['sf_trunc'][0]-sfh['sf_start'][0])/sfh['tau'][0])
            intsfr = intsfr/(norm*sfh['tau'][0]**2)

        # else, add lin-ramp
        elif (sfh['sfh'] == 5):

            # by-hand calculation
            norm1 = integrate_delayed_tau(0,sfh['sf_trunc']-sfh['sf_start'],sfh)
            norm2 = integrate_linramp(sfh['sf_trunc']-sfh['sf_start'],sfh['tage']-sfh['sf_start'],sfh)

            if (t1 < sfh['sf_trunc']-sfh['sf_start']) and \
               (t2 < sfh['sf_trunc']-sfh['sf_start']):
                intsfr = integrate_delayed_tau(t1,t2,sfh) / (norm1+norm2)
            elif (t1 > sfh['sf_trunc']-sfh['sf_start']) and \
                 (t2 > sfh['sf_trunc']-sfh['sf_start']):
                intsfr = integrate_linramp(t1,t2,sfh) / (norm1+norm2)
            else:
                intsfr = (integrate_delayed_tau(t1,sfh['sf_trunc']-sfh['sf_start'],sfh) + \
                          integrate_linramp(sfh['sf_trunc']-sfh['sf_start'],t2,sfh)) / \
                          (norm1+norm2)

        else:
            sys.exit('no such SFH implemented')

        # return sum of SFR components
        tot_mformed = np.sum(intsfr*sfh['mass'])/np.sum(sfh['mass'])
    
    #### nonparametric SFH
    else:

        ### make sure we got what we need
        if ('agebins' not in sfh_params):
            sys.exit('missing parameters!')

        ### put bins in proper units
        to_linear_bins = 10**sfh_params['agebins']/1e9
        time_per_bin = to_linear_bins[:,1] - to_linear_bins[:,0]
        time_bins = np.max(to_linear_bins) - to_linear_bins

        ### if it's outside the SFH bins, clip it
        t1 = np.clip(t1,np.min(time_bins),np.max(time_bins))
        t2 = np.clip(t2,np.min(time_bins),np.max(time_bins))

        # annoying edge cases
        if t1 == t2:
            return 0.0
        if (t2 > time_bins[0,0]) & (t1 > time_bins[0,0]): 
            sys.exit('SFR is undefined in this time region, outside youngest bin!')

        ### which bins to integrate?
        in_range = (time_bins >= t1) & (time_bins <= t2)
        bin_ids = in_range.sum(axis=1)

        ### this doesn't work if we're fully inside a single bin...
        if in_range.sum() == 0:
            bin_ids = (time_bins[:,1] <= t1) & (time_bins[:,0] >= t2)

        ### weights
        weights = np.zeros(sfh_params['mass_fraction'].shape)
        # if we're all in one bin
        if np.sum(bin_ids) == 1:
            weights[bin_ids == 1] = t2-t1
        # else do the whole thing
        else:
            for i in xrange(bin_ids.shape[0]):
                if bin_ids[i] == 2: # bins that are entirely in t1,t2.
                    weights[i] = time_per_bin[i]
                if bin_ids[i] == 1: # edge cases
                    if t2 < time_bins[i,0]: # this is the most recent edge
                        weights[i] = t2 - time_bins[i,1]
                    else: # this is the oldest edge
                        weights[i] = time_bins[i,0]-t1
                if bin_ids[i] == 0: # no contribution
                    continue

        ### bug catch
        try:
            np.testing.assert_approx_equal(np.sum(weights),t2-t1,significant=5)
        except AssertionError:
            print 1/0
            sys.exit('weights do not sum to 1')

        tot_mformed = np.sum((weights/time_per_bin)*sfh_params['mass_fraction'])

    return tot_mformed

def measure_lbol(sps,mass):

    '''
    requires mformed
    return in Lsun
    '''

    ## get SPS lbol, weighted by SSP weights
    # two options due to very different meanings of ssp.log_lbol when using
    # tabular or "regular" SSPs
    # THIRD OPTION: access csp
    try:
        if np.isscalar(sps.ssp.log_lbol):
            weighted_lbol = 10**sps.ssp.log_lbol
        else:
            ssp_lbol = np.insert(10**sps.ssp.log_lbol, 0, 10**sps.ssp.log_lbol[0])
            weights = sps.all_ssp_weights
            weighted_lbol = (ssp_lbol * weights).sum() / weights.sum() * mass
    except AttributeError:
        weighted_lbol = 10**sps.csp.log_lbol
    return weighted_lbol

def measure_agn_luminosity(fagn,sps,mass):

    '''
    requires mformed
    calculate L_AGN for a given F_AGN, SPS
    return in erg / s
    '''

    ## get SPS lbol, weighted by SSP weights
    # two options due to very different meanings of ssp.log_lbol when using
    # tabular or "regular" SSPs
    if np.isscalar(sps.ssp.log_lbol):
        weighted_lbol = 10**sps.ssp.log_lbol
        lagn = weighted_lbol*float(fagn)*constants.L_sun.cgs.value
    else:
        ssp_lbol = np.insert(10**sps.ssp.log_lbol, 0, 10**sps.ssp.log_lbol[0])
        weights = sps.all_ssp_weights
        weighted_lbol = (ssp_lbol * weights).sum() / weights.sum()

        ## calculate L_AGN
        lagn_sps = weighted_lbol*fagn
        lagn = lagn_sps * mass * constants.L_sun.cgs.value

    return lagn

def estimate_xray_lum(sfr):
    '''
    requires SFR in Msun/year
    L[0.5-8 keV] in erg / s = CONSTANT * SFR  (FROM MINEO+14)
    SFR is corrected into a Salpeter IMF
    '''

    sfr_chabrier = 10**(np.log10(sfr)+0.24)
    return 4e39*sfr_chabrier

def measure_restframe_properties(sps, model = None, thetas = None, emlines = False,
                                 measure_ir = False, measure_luv = False, measure_mir = False,
                                 abslines = False, measure_rf = False, measure_uvj=False):
    """measures a variety of rest-frame properties: total IR luminosity,
    total UV luminosity, total MIR luminosity, emission line luminosity, etc.
    absorption line depths can be measured as well, via fixed definitions.

    as written, this will NOT work if zred is a variable.
    """
    out = {}

    # constants
    pc = 3.085677581467192e18  # in cm
    dfactor_10pc = 4*np.pi*(10*pc)**2
    to_ergs = 3631e-23

    ### save redshift, lumdist
    z      = model.params.get('zred', np.array(0.0))
    lumdist = model.params.get('lumdist', np.array(0.0))
    nebinspec = model.params.get('nebemlineinspec', True)
    model.params['zred'] = np.array(0.0)
    if lumdist:
        model.params['lumdist'] = np.array(1e-5)
    if nebinspec == False:
        model.params['nebemlineinspec'] = True

    ### if we want restframe optical photometry, generate fake obs file
    ### else generate NO obs file (don't do extra filter convolutions if not necessary)
    obs = {'filters': [], 'wavelength': None}
    if measure_uvj:
        from sedpy.observate import load_filters
        filters = ['bessell_U','bessell_V','twomass_J']# ,'bessell_B','bessell_R','twomass_Ks']
        if measure_rf:
            filters += ['sdss_g0','sdss_r0','sdss_i0']
        obs['filters'] += load_filters(filters)

    ### calculate SED. comes out as maggies per Hz, @ 10pc
    spec,mags,sm = model.mean_model(thetas, obs, sps=sps)
    w = sps.wavelengths

    ### reset model
    model.params['zred'] = z
    if lumdist:
        model.params['lumdist'] = lumdist
    if nebinspec == False:
        model.params['nebemlineinspec'] = False

    ### convert to Lsun / hz
    spec *= dfactor_10pc / constants.L_sun.cgs.value * to_ergs

    ### calculate in fnu for filter calculations
    to_flam = 3e18 / w**2
    spec_flam = spec * to_flam

    ##### do we need a smooth spectrum?
    if (abslines):
        smooth_spec = smooth_spectrum(w,spec_flam,250.0,minlam=3e3,maxlam=7e3)
        out['abslines'] = measure_abslines(w,smooth_spec) # comes out in Lsun and rest-frame EQW

    ##### measure emission lines
    if emlines:
        out['emlines'] = measure_emlines(spec_flam,sps,enames=emlines)
    out['dn4000'] = measure_Dn4000(w,spec_flam)

    if measure_ir:
        out['lir'] = return_lir(w,spec, z=None)/constants.L_sun.cgs.value # comes out in ergs/s, convert to Lsun
    if measure_luv:
        out['luv'] = return_luv(w,spec, z=None)/constants.L_sun.cgs.value # comes out in ergs/s, convert to Lsun
    if measure_mir:
        out['lmir'] = return_lmir(w,spec, z=None)/constants.L_sun.cgs.value # comes out in ergs/s, convert to Lsun
    if measure_uvj:
        out['uvj'] = mags[:3]
        out['photname'] = filters[:3]
    if measure_rf:
        out['rf'] = mags[3:]

    return out

def measure_Dn4000(lam,flux,ax=None):
    """ defined as average flux ratio between
    [4050,4250] and [3750,3950] (Bruzual 1983; Hamilton 1985)
    blue: 3850-3950 . . . 4000-4100 (Balogh 1999)
    """
    blue = (lam > 3850) & (lam < 3950)
    red  = (lam > 4000) & (lam < 4100)
    dn4000 = np.mean(flux[red])/np.mean(flux[blue])

    if ax is not None:
        ax.plot(lam,flux,color='black',drawstyle='steps-mid')
        ax.plot(lam[blue],flux[blue],color='blue',drawstyle='steps-mid')
        ax.plot(lam[red],flux[red],color='red',drawstyle='steps-mid')

        ax.text(0.96,0.05, 'D$_{n}$4000='+"{:.2f}".format(dn4000), transform = ax.transAxes,horizontalalignment='right')

        ax.set_xlim(3800,4150)
        plt_lam_idx = (lam > 3800) & (lam < 4150)
        minplot = np.min(flux[plt_lam_idx])*0.9
        maxplot = np.max(flux[plt_lam_idx])*1.1
        ax.set_ylim(minplot,maxplot)

    return dn4000

def measure_emlines(smooth_spec,sps,enames=None):
    """ emission line fluxes are part of SPS output now. this is
    largely present to measure the continuum for EQW calculations
    """

    ### load fsps emission line list
    loc = os.getenv('SPS_HOME')+'/data/emlines_info.dat'
    dat = np.loadtxt(loc, delimiter=',',
                     dtype = {'names':('lam','name'),'formats':('f16','S40')})

    ### define emission lines
    # legacy code compatible
    if type(enames) == bool:
        lines = np.array(['Hdelta','Hbeta','[OIII]1','[OIII]2','Halpha','[NII]'])
        fsps_name = np.array(['H delta 4102','H beta 4861','[OIII]4960','[OIII]5007','H alpha 6563','[NII]6585'])
    else:
        lines = enames
        fsps_name = enames

    ##### measure emission line flux + EQW
    out = {}
    for jj in xrange(len(lines)):

        # if we don't do nebular emission, zero this out
        if not hasattr(sps, 'get_nebline_luminosity'):
            out[lines[jj]] = {'flux':0.0,'eqw':0.0}
            continue

        ### calculate luminosity (in Lsun)
        idx = fsps_name[jj] == dat['name']
        eflux = float(sps.get_nebline_luminosity[idx]*sps.params['mass'].sum())
        elam = float(sps.emline_wavelengths[idx])

        # simple continuum estimation
        tidx = np.abs(sps.wavelengths-elam) < 100
        eqw = eflux/np.median(smooth_spec[tidx])

        out[lines[jj]] = {'flux':eflux,'eqw':eqw}

    return out

def measure_abslines(lam,flux,plot=False, alt_plot=False):

    '''
    Nelan et al. (2005)
    Halpha wide: 6515-6540, 6554-6575, 6575-6585
    Halpha narrow: 6515-6540, 6554-6568, 6568-6575

    Worthey et al. 1994
    Hbeta: 4827.875-4847.875, 4847.875-4876.625, 4876.625-4891

    Worthey et al. 1997
    hdelta wide: 4041.6-4079.75, 4083.5-4122.25, 4128.5-4161.0
    hdelta narrow: 4057.25-4088.5, 4091-4112.25, 4114.75-4137.25

    WIDENED HALPHA AND HBETA BECAUSE OF SMOOTHING
    '''

    out = {}

    # define lines and indexes
    lines = np.array(['halpha_wide', 'halpha_narrow',
                      'hbeta',
                      'hdelta_wide', 'hdelta_narrow'])

    # improved hdelta narrow
    index = [(6540.,6586.),(6542.,6584.),(4842.875,4884.625),(4081.5,4124.25),(4095.0,4113.75)]
    up =    [(6590.,6610.),(6585.,6595.),(4894.625,4910.000),(4124.25,4151.00),(4113.75,4130.25)]
    down =  [(6515.,6540.),(6515.,6540.),(4817.875,4835.875),(4041.6,4081.5),(4072.25,4094.50)]

    # if we want to plot but don't have fig + ax set up, set them up
    # else if we have them, unpack them
    if plot == True:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2,3, figsize = (18.75,12))
        ax = np.ravel(ax)
    elif plot is not False:
        fig, ax = plot[0],plot[1]

    # measure the absorption flux
    for ii in xrange(len(lines)):
        dic = {}

        #### only alt_plot for hdelta!
        if alt_plot & (ii <=2):
            continue

        if plot: 
            dic['flux'], dic['eqw'], dic['lam'] = measure_idx(lam,flux,index[ii],up[ii],down[ii],ax=ax[ii],alt_plot=alt_plot)
        else: 
            dic['flux'], dic['eqw'], dic['lam'] = measure_idx(lam,flux,index[ii],up[ii],down[ii],ax=None)
        out[lines[ii]] = dic

    if plot is not False:
        out['ax'] = ax
        out['fig'] = fig

    return out

def measure_idx(lam,flux,index,up,down,ax=None,alt_plot=False):

    '''
    measures absorption depths
    '''

    continuum_line_color='red'
    observed_flux_color = 'black'
    line_index_color = 'blue'
    continuum_index_color = 'cyan'
    eqw_yloc = 0.05
    alpha = 1.0

    if alt_plot:
        eqw_yloc = 0.1
        line_index_color = 'green'
        alpha = 0.5


    ##### identify average flux, average wavelength
    low_cont = (lam > down[0]) & (lam < down[1])
    high_cont = (lam > up[0]) & (lam < up[1])
    abs_idx = (lam > index[0]) & (lam < index[1])

    low_flux = np.mean(flux[low_cont])
    high_flux = np.mean(flux[high_cont])

    low_lam = np.mean(down)
    high_lam = np.mean(up)

    ##### draw straight line between midpoints
    # y = mx + b
    # m = (y2 - y1) / (x2 - x1)
    # b = y0 - mx0
    m = (high_flux-low_flux)/(high_lam-low_lam)
    b = high_flux - m*high_lam

    ##### integrate the flux and the straight line, take the difference
    yline = m*lam[abs_idx]+b
    absflux = np.trapz(yline,lam[abs_idx]) - np.trapz(flux[abs_idx], lam[abs_idx])

    lamcont = np.mean(lam[abs_idx])
    abseqw = absflux/(m*lamcont+b)

    ##### plot if necessary
    if ax is not None:
        ax.plot(lam,flux,color=observed_flux_color, drawstyle='steps-mid',alpha=alpha)
        ax.plot(lam[abs_idx],yline,color=continuum_line_color,alpha=alpha)
        ax.plot(lam[abs_idx],flux[abs_idx],color=line_index_color,alpha=alpha)
        ax.plot(lam[low_cont],flux[low_cont],color=continuum_index_color,alpha=alpha)
        ax.plot(lam[high_cont],flux[high_cont],color=continuum_index_color,alpha=alpha)

        ax.text(0.96,eqw_yloc, 'EQW='+"{:.2f}".format(abseqw), transform = ax.transAxes,horizontalalignment='right',color=line_index_color)
        ax.set_xlim(np.min(down)-50,np.max(up)+50)

        plt_lam_idx = (lam > down[0]) & (lam < up[1])
        
        if alt_plot:
            minplot = np.min([ np.min(flux[plt_lam_idx])*0.9, ax.get_ylim()[0] ])
            maxplot = np.max([ np.max(flux[plt_lam_idx])*1.1, ax.get_ylim()[1] ])
        else:
            minplot = np.min(flux[plt_lam_idx])*0.9
            maxplot = np.max(flux[plt_lam_idx])*1.1
        ax.set_ylim(minplot,maxplot)

    return absflux, abseqw, lamcont
