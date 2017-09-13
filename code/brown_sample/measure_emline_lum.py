import numpy as np
from astropy.cosmology import WMAP9 as cosmo
from astropy.modeling import core, fitting, Parameter, functional_models
import prosp_dutils
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy import constants
from matplotlib.ticker import MaxNLocator
import os, time

c_kms = 2.99e5
maxfev = 2000
dpi=150

emline = np.array(['[OIII] 4959','[OIII] 5007',r'H$\beta$','[NII] 6549', '[NII] 6583', r'H$\alpha$','[OII] 3726','[OII] 3728'])
em_wave = np.array([4958.92,5006.84,4861.33,6548.03,6583.41,6562.80,3726.1,3728.8])
em_bbox = [(4800,5050),(6500,6640),(3680,3760)]

class tLinear1D(core.Fittable1DModel):

    slope_low = Parameter(default=0)
    intercept_low = Parameter(default=0)
    slope_mid = Parameter(default=0)
    intercept_mid = Parameter(default=0)
    slope_high = Parameter(default=0)
    intercept_high = Parameter(default=0)
    linear = True

    @staticmethod
    def evaluate(x, slope_low, intercept_low, slope_mid, intercept_mid, slope_high, intercept_high):
        """One dimensional Line model function"""

        out = np.zeros_like(x)
        low = (np.array(x) < 4000) & (np.array(x) > 3650)
        mid = (np.array(x) > 4000) & (np.array(x) < 5600)
        high = np.array(x) > 5600
        out[low] = slope_low*x[low]+intercept_low
        out[mid] = slope_mid*x[mid]+intercept_mid
        out[high] = slope_high*x[high]+intercept_high

        return out

    @staticmethod
    def fit_deriv(x, slope_low, intercept_low, slope_mid, intercept_mid, slope_high, intercept_high):
        """One dimensional Line model derivative with respect to parameters"""

        low = (np.array(x) < 4000) & (np.array(x) > 3650)
        mid = (np.array(x) > 4000) & (np.array(x) < 5600)
        high = np.array(x) > 5600
        
        d_lslope = np.zeros_like(x)
        d_lint = np.zeros_like(x)
        d_mslope = np.zeros_like(x)
        d_mint = np.zeros_like(x)
        d_hslope = np.zeros_like(x)
        d_hint = np.zeros_like(x)
        
        d_lslope[low] = x[low]
        d_mslope[mid] = x[mid]
        d_hslope[high] = x[high]
        d_lint[low] = np.ones_like(x[low])
        d_mint[mid] = np.ones_like(x[mid])
        d_hint[high] = np.ones_like(x[high])

        return [d_lslope, d_lint, d_mslope, d_mint, d_hslope, d_hint]

def bootstrap(obslam, obsflux, model, fitter, noise, line_lam, 
              flux_flag=True, nboot=100):

    ### count lines
    nlines = len(np.atleast_1d(line_lam))

    ### desired output
    params = np.zeros(shape=(len(model.parameters),nboot))
    flux_chain = np.zeros(shape=(nboot,nlines))

    ### random data sets are generated and fitted
    nsuccess, nrand, successflag = 0, 0, True
    while nsuccess < nboot:

        ### if we can't get a successful fit after many tries
        ### fix the emission line sigmas!
        # don't use OII for this purpose 
        if nrand >= 10:
            stddev = np.array([getattr(fit, 'stddev_'+str(j)).value for j in xrange(nlines)])
            bad = (np.abs(stddev) >= 12)
            good = (~bad) & ('[OII]' not in emline)
            try:
                fixed_sig = np.max(stddev[good])
            except ValueError:
                fixed_sig = 10

            for i,flag in enumerate(bad):
                if not flag:
                    continue
                
                ### fix it to the average of all other sigmas
                getattr(model, 'stddev_'+str(i)).fixed = True
                getattr(model, 'stddev_'+str(i)).value = fixed_sig
                print 'FAILURE: fixing '+emline[i]+' to ' "{:.2f}".format(fixed_sig/em_wave[i]*c_kms)

        ### do we have a range of continuua?
        if type(obsflux[0]) == np.float64:
            oflux = obsflux
        else:
            oflux = obsflux[nsuccess]

        randomDelta = np.random.normal(0., noise, len(oflux))
        randomFlux = oflux + randomDelta
        fit = fitter(model, obslam, randomFlux,maxiter=1000)
        params[:,nsuccess] = fit.parameters

        ### catch error: what happens if sigma is at boundary?
        stddev = np.array([getattr(fit, 'stddev_'+str(j)).value for j in xrange(nlines)])
        if np.all(np.abs(stddev) < 12):
            nsuccess += 1
        elif nrand == 30:
            print 'BIG FAILURE, skipping'
            nsuccess += 1
        else:
            nrand +=1
            continue
        
        nrand = 0
        for j in xrange(nlines): getattr(model, 'stddev_'+str(j)).fixed = False

        if flux_flag:
            ### calculate emission line flux
            for j in xrange(nlines): 
                amp = getattr(fit, 'amplitude_'+str(j)).value
                stddev = getattr(fit, 'stddev_'+str(j)).value
                flux_chain[nsuccess-1,j] = amp*np.sqrt(2*np.pi*stddev**2) * constants.L_sun.cgs.value

    ### now get median + percentiles
    medianpar = np.percentile(params, 50,axis=1)
    fit.parameters = medianpar

    # we want errors for flux and eqw
    flux_point_estimates = np.percentile(flux_chain, [50,84,16],axis=0)

    return fit, flux_point_estimates, flux_chain

def tiedfunc_oii(g1):
    amp = 0.35 * g1.amplitude_6
    return amp

def tiedfunc_oiii(g1):
    amp = 2.98 * g1.amplitude_0
    return amp

def tiedfunc_nii(g1):
    amp = 2.93 * g1.amplitude_3
    return amp

def loii_2(g1):
    zadj = g1.mean_6 / 3726.1 - 1
    return (1+zadj) *  3728.8

def loii_1(g1):
    zadj = g1.mean_0 / 4958.92 - 1
    return (1+zadj) *  3726.1

def loiii_2(g1):
    zadj = g1.mean_0 / 4958.92 - 1
    return (1+zadj) *  5006.84

def lhbeta(g1):
    zadj = g1.mean_5 / 6562.80 - 1
    return (1+zadj) *  4861.33

def lnii_1(g1):
    zadj = g1.mean_0 / 4958.92 - 1
    return (1+zadj) *  6548.03

def lnii_2(g1):
    zadj = g1.mean_3 / 6548.03 - 1
    return (1+zadj) *  6583.41

def lhalpha(g1):
    zadj = g1.mean_0 / 4958.92 - 1
    return (1+zadj) *  6562.80

def soii(g1):
    return g1.stddev_6

def soiii(g1):
    return g1.stddev_0

def snii(g1):
    return g1.stddev_3

def sig_ret(model):
    n=0
    sig=[]
    while True:
        try:
            sig.append(getattr(model, 'stddev_'+str(n)).value)
        except:
            break
        n+=1

    return sig

def umbrella_model(lams, amp_tied, lam_tied, sig_tied,continuum_6400):
    '''
    return model for [OIII], Hbeta, [NII], Halpha, + constant
    centers are all tied to redshift parameter
    '''

    #### EQW initial
    # OIII 1, OIII 2, Hbeta, NII 1, NII 2, Halpha
    eqw_init = np.array([3.,9.,4.,1.,3.,12.,3.,3.])*4   
    stddev_init = 3.5

    #### ADD ALL MODELS FIRST
    for ii in xrange(len(lams)):
        amp_init = continuum_6400*eqw_init[ii] / (np.sqrt(2.*np.pi*stddev_init**2))
        if ii == 0:
            model = functional_models.Gaussian1D(amplitude=amp_init, mean=lams[ii], stddev=stddev_init)
        else: 
            model += functional_models.Gaussian1D(amplitude=amp_init, mean=lams[ii], stddev=stddev_init)

    # add slope + constant
    model += tLinear1D(intercept_low=continuum_6400,intercept_mid=continuum_6400,intercept_high=continuum_6400)

    #### NOW TIE THEM TOGETHER
    for ii in xrange(len(lams)):
        # position and widths
        if lam_tied[ii] is not None:
            getattr(model, 'mean_'+str(ii)).tied = lam_tied[ii]

        # amplitudes, if necessary
        if amp_tied[ii] is not None:
            getattr(model, 'amplitude_'+str(ii)).tied = amp_tied[ii]

        # sigmas, if necessary
        if sig_tied[ii] is not None:
            getattr(model, 'stddev_'+str(ii)).tied = sig_tied[ii]

        getattr(model, 'stddev_'+str(ii)).max = 12


    return model

def measure(sample_results, extra_output, obs_spec, sps, runname='brownseds_np', sigsmooth=None):
    
    '''
    measure emission line luminosities using prospector continuum model
    '''

    '''
    ##### FIRST, CHECK RESOLUTION IN REGION OF HALPHA
    ##### IF IT'S JUNK, THEN RETURN NOTHING!
    idx = (np.abs(obs_spec['rest_lam'] - 6563)).argmin()
    dellam = obs_spec['rest_lam'][idx+1] - obs_spec['rest_lam'][idx]
    if dellam > 14:
        print 'low resolution, not measuring fluxes for '+sample_results['run_params']['objname']
        return None
    '''
    #### smoothing in km/s
    smooth       = 200

    #### output names
    objname_short = sample_results['run_params']['objname'].replace(' ','_')
    base = '/Users/joel/code/python/prospector_alpha/plots/'+runname+'/magphys/line_fits/'
    if not os.path.isdir(base):
        os.makedirs(base)
    out_em = base+objname_short+'_em_prosp.png'
    out_abs = base+objname_short+'_abs_prosp.png'
    out_absobs = base+objname_short+'_absobs.png'

    #### define emission lines to measure
    # we do this in sets of lines, which are unpacked at the end
    amp_tied = [None, tiedfunc_oiii, None, None, tiedfunc_nii, None, None, tiedfunc_oii]
    #lam_tied = [None, loiii_2, lhbeta, lnii_1, lnii_2, lhalpha, loii_1, loii_2]
    lam_tied = [None, loiii_2, None, None, lnii_2, None, None, loii_2]
    sig_tied = [None, soiii, None, None, snii, None, None, soii]
    nline = len(emline)

    #### now define sets of absorption lines to measure
    abslines = np.array(['halpha_wide', 'halpha_narrow', 'hbeta', 'hdelta_wide', 'hdelta_narrow'])
    nabs = len(abslines)             

    #### mapping between em_bbox and abslines
    mod_abs_mapping = [np.where(abslines == 'hbeta')[0][0],np.where(abslines == 'halpha_wide')[0][0]]

    #### put all spectra in proper units (Lsun/AA, rest wavelength in Angstroms)
    # first, get rest-frame Prospector spectrum
    # with z=0.0, it arrives in erg/s/cm^2/AA @ 10pc @ rest wavelength
    # save redshift, restore at end
    lumdist = sample_results['model'].params['lumdist']
    z = sample_results['model'].params['zred']
    sample_results['model'].params['zred'] = np.array(0.0)
    sample_results['model'].params['lumdist'] = np.array(1e-5)
    sample_results['model'].params['add_neb_emission'] = np.array(False)
    sample_results['model'].params['add_neb_continuum'] = np.array(False)
    sample_results['model'].params['peraa'] = np.array(True)

    model_lam = sps.wavelengths
    pc = 3.085677581467192e18  # cm

    # observed spectra arrive in Lsun/cm^2/AA
    # convert distance factor
    dfactor = 4*np.pi*(pc*lumdist[0] * 1e6)**2 * (1+z) 
    obsflux = obs_spec['flux_lsun']*dfactor
    obslam = obs_spec['rest_lam']

    ##### define fitter
    fitter = fitting.LevMarLSQFitter()

    #######################
    ##### NOISE ###########
    #######################
    #### define emission line model and fitting region
    # use continuum_6400 to set up a good guess for starting positions
    continuum_6400 = obsflux[(np.abs(obslam-6400)).argmin()]
    emmod =  umbrella_model(em_wave, amp_tied, lam_tied, sig_tied, continuum_6400=continuum_6400)
    p_idx = np.zeros_like(obslam,dtype=bool)
    for bbox in em_bbox:p_idx[(obslam > bbox[0]) & (obslam < bbox[1])] = True

    ### initial fit to emission
    # use to find sigma
    gauss_fit,_,_ = bootstrap(obslam[p_idx], obsflux[p_idx], emmod, fitter, np.min(np.abs(obsflux[p_idx]))*0.001, em_wave, 
                              flux_flag=False, nboot=1)

    ### find proper smoothing
    # by taking two highest EQW lines, and using the width
    # this assumes no scaling w/ wavelength, and that gas ~ stellar velocity dispersion
    tmpflux = np.zeros(nline)
    for j in xrange(nline):tmpflux[j] = getattr(gauss_fit, 'amplitude_'+str(j)).value*np.sqrt(2*np.pi*getattr(gauss_fit, 'stddev_'+str(j)).value**2)
    mlow = (em_wave < 4000) & (em_wave > 3650)
    mmid = (em_wave >= 4000) & (em_wave < 5600)
    mhigh = em_wave >= 5600

    emnorm =  np.concatenate((gauss_fit.intercept_low_8+gauss_fit.slope_low_8*em_wave[mlow],
                              gauss_fit.intercept_mid_8+gauss_fit.slope_mid_8*em_wave[mmid],
                              gauss_fit.intercept_high_8+gauss_fit.slope_high_8*em_wave[mhigh]))
    tmpeqw = tmpflux / emnorm

    # take two highest EQW lines
    # not OII due to resolution!
    oii_mask = (emline != '[OII] 3726') & (emline != '[OII] 3728')
    high_eqw = tmpeqw[oii_mask].argsort()[-2:]
    sigma_spec = np.mean(np.array(sig_ret(gauss_fit))[high_eqw]/em_wave[high_eqw])*c_kms
    sigma_spec = np.clip(sigma_spec,10.0,250)

    test_plot = False
    if test_plot:
        fig, ax = plt.subplots(1,1)
        plt.plot(obslam[p_idx],obsflux[p_idx], color='black')
        plt.plot(obslam[p_idx],gauss_fit(obslam[p_idx]),color='red')
        plt.show()
        print 1/0

    remove_10pc_dfactor = 4*np.pi*(pc*10)**2

    #### adjust model wavelengths for the slight difference with published redshifts
    top, bot = 0, 0
    for i, (tiefunc, w_em) in enumerate(zip(lam_tied,em_wave)):
        if tiefunc is None:
            top += (getattr(gauss_fit, 'mean_'+str(i))/w_em - 1)
            bot += 1
    zadj = top / float(bot)
    print 'z adjust: '+'{:.4f}'.format(zadj)
    #zadj = gauss_fit.mean_0 / 4958.92 - 1
    #model_newlam = (1+zadj)*model_lam
    model_newlam = (1+zadj)*model_lam

    nboot = 100
    emline_noise, residuals = [], []
    for ii in range(nboot):
        model_flux,mags,sm = sample_results['model'].mean_model(extra_output['quantiles']['sample_chain'][ii,:], sample_results['obs'], sps=sps)
        model_flux *= remove_10pc_dfactor/constants.L_sun.cgs.value

        #### normalize continuum model to observations
        for kk in xrange(len(em_bbox)):

            ### average in bounding box. emission lines masked.
            mask_size = 30
            mod_idx = (model_newlam > em_bbox[kk][0]-30) & (model_newlam < em_bbox[kk][1]+30)
            obs_idx = (obslam > em_bbox[kk][0]-30) & (obslam < em_bbox[kk][1]+30)
            for line in em_wave: 
                mod_idx[(model_newlam > line - mask_size) & (model_newlam < line + mask_size)] = False
                obs_idx[(obslam > line - mask_size) & (obslam < line + mask_size)] = False

            norm_factor = np.mean(obsflux[obs_idx])/np.mean(model_flux[mod_idx])

            mod_idx = (model_newlam > em_bbox[kk][0]-30) & (model_newlam < em_bbox[kk][1]+30)
            obs_idx = (obslam > em_bbox[kk][0]-30) & (obslam < em_bbox[kk][1]+30)

            model_flux[mod_idx] = model_flux[mod_idx]*norm_factor

        #### absorption model versus emission model
        if False:
            p_idx_mod = ((model_newlam > em_bbox[0][0]) & (model_newlam < em_bbox[0][1])) | \
                        ((model_newlam > em_bbox[1][0]) & (model_newlam < em_bbox[1][1])) | \
                        ((model_newlam > em_bbox[2][0]) & (model_newlam < em_bbox[2][1]))
            fig, ax = plt.subplots(1,1)
            plt.plot(model_newlam[p_idx_mod], model_flux[p_idx_mod], color='black')
            plt.plot(obslam[p_idx], gauss_fit(obslam[p_idx]), color='red')
            plt.show()
            print 1/0

        #### interpolate model onto observations
        flux_interp = interp1d(model_newlam, model_flux, bounds_error = False, fill_value = 0)
        modflux = flux_interp(obslam[p_idx])

        #### smooth model to match observations
        smoothed_absflux = prosp_dutils.smooth_spectrum(obslam[p_idx], modflux, sigma_spec)

        #### subtract model from observations
        residuals.append(obsflux[p_idx] - smoothed_absflux)

        #### mask emission lines
        masklam = 30
        mask = np.ones_like(obslam[p_idx],dtype=bool)
        for lam in em_wave: mask[(obslam[p_idx] > lam-masklam) & (obslam[p_idx] < lam+masklam)] = 0

        #### find 1sigma of residuals
        sig = np.percentile(residuals[-1][mask],[16,84])
        emline_noise.append((sig[1]-sig[0])/2.)
        continuum = gauss_fit[8](em_wave)

    ### take minimum for noise
    emline_noise = np.array(emline_noise).min()
    print emline[0]+' noise: '"{:.4f}".format(emline_noise/continuum[0])

    ###############################################
    ##### MEASURE FLUXES FROM OBSERVATIONS ########
    ###############################################
    # get errors in parameters by bootstrapping
    # use lower of two error estimates
    # flux come out of bootstrap as (nlines,[median,errup,errdown])

    #### now measure emission line fluxes
    t1 = time.time()
    bfit_mod, emline_flux, emline_chain = bootstrap(obslam[p_idx],residuals,emmod,fitter,emline_noise,em_wave,nboot=nboot)
    print('main bootstrapping took {0}s'.format(time.time()-t1))

    #############################
    #### PLOT ALL THE THINGS ####
    #############################
    # set up figure
    # loop over models at some point? including above fit?
    fig, axarr = plt.subplots(1, 3, figsize = (25,6))
    for ii, bbox in enumerate(em_bbox):

        p_idx_em = ((obslam[p_idx] > bbox[0]) & (obslam[p_idx] < bbox[1]))

        # observations
        axarr[ii].plot(obslam[p_idx][p_idx_em],obsflux[p_idx][p_idx_em],color='black',drawstyle='steps-mid')
        # emission model + continuum model
        axarr[ii].plot(obslam[p_idx][p_idx_em],bfit_mod(obslam[p_idx][p_idx_em])+smoothed_absflux[p_idx_em],color='red')
        # continuum model + zeroth-order emission continuum
        axarr[ii].plot(obslam[p_idx][p_idx_em],smoothed_absflux[p_idx_em]+bfit_mod[8](obslam[p_idx][p_idx_em]),color='#1E90FF')

        axarr[ii].set_ylabel(r'flux [L$_{\odot}/\AA$]')
        axarr[ii].set_xlabel(r'$\lambda$ [$\AA$]')

        axarr[ii].xaxis.set_major_locator(MaxNLocator(5))
        axarr[ii].yaxis.get_major_formatter().set_powerlimits((0, 1))

        e_idx = (em_wave > bbox[0]) & (em_wave < bbox[1])
        nplot=0
        for kk in xrange(len(emline)):
            if e_idx[kk]:
                fmt = "{{0:{0}}}".format(".2e").format
                emline_str = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                emline_str = emline_str.format(fmt(emline_flux[0,kk]), 
                                               fmt(emline_flux[1,kk]-emline_flux[0,kk]), 
                                               fmt(emline_flux[0,kk]-emline_flux[2,kk])) + ' erg/s, '
                axarr[ii].text(0.03, 0.93-0.085*nplot, 
                               emline[kk]+': '+emline_str,
                               fontsize=16, transform = axarr[ii].transAxes)
                nplot+=1
        for kk in xrange(len(emline)):
            if e_idx[kk]:
                axarr[ii].text(0.03, 0.93-0.085*nplot, emline[kk]+r'$\sigma$='+"{:.2f}".format(getattr(bfit_mod, 'stddev_'+str(kk)).value/em_wave[kk]*c_kms), transform = axarr[ii].transAxes)
                nplot+=1
        ylim = axarr[ii].get_ylim()
        axarr[ii].set_ylim(ylim[0],ylim[1]*1.4)

        axarr[ii].text(0.98, 0.93, 'em+abs model', fontsize=16, transform = axarr[ii].transAxes,ha='right',color='red')
        axarr[ii].text(0.98, 0.85, 'abs model', fontsize=16, transform = axarr[ii].transAxes,ha='right',color='#1E90FF')
        axarr[ii].text(0.98, 0.77, 'observations', fontsize=16, transform = axarr[ii].transAxes,ha='right')

    plt.tight_layout()
    plt.savefig(out_em, dpi=dpi)
    plt.close()

    ##############################################
    ##### MEASURE OBSERVED ABSORPTION LINES ######
    ##############################################
    # use redshift from emission line fit
    # currently, NO spectral smoothing; inspect to see if it's important for HDELTA ONLY

    # if sigma_spec < 200:
    if sigma_spec < 250:
        to_convolve = (250.**2 - sigma_spec**2)**0.5
        obsflux = prosp_dutils.smooth_spectrum(obslam/(1+zadj),obsflux,to_convolve,minlam=3e3,maxlam=1e4)

    #### bootstrap
    nboot = 100
    tobs_abs_flux,tobs_abs_eqw = [np.zeros(shape=(nabs,nboot)) for i in xrange(2)]

    for i in xrange(nboot):
        randomDelta = np.random.normal(0.,emline_noise,len(obsflux))
        randomFlux = obsflux + randomDelta
        out = prosp_dutils.measure_abslines(obslam/(1+zadj),randomFlux,plot=False)

        for kk in xrange(nabs): tobs_abs_flux[kk,i] = out[abslines[kk]]['flux']
        for kk in xrange(nabs): tobs_abs_eqw[kk,i] = out[abslines[kk]]['eqw']

    ### we want errors for flux and eqw
    obs_abs_flux,obs_abs_eqw = [np.zeros(shape=(nabs,3)) for i in xrange(2)]

    for kk in xrange(nabs): obs_abs_flux[kk,:] = np.array([np.percentile(tobs_abs_flux[kk,:],50),
                                                           np.percentile(tobs_abs_flux[kk,:],84),
                                                           np.percentile(tobs_abs_flux[kk,:],16)])

    for kk in xrange(nabs): obs_abs_eqw[kk,:] =  np.array([np.percentile(tobs_abs_eqw[kk,:],50), 
                                                           np.percentile(tobs_abs_eqw[kk,:],84), 
                                                           np.percentile(tobs_abs_eqw[kk,:],16)])

    # bestfit, for plotting purposes
    out = prosp_dutils.measure_abslines(obslam/(1+zadj),obsflux,plot=True)
    obs_lam_cont = np.zeros(nabs)
    for kk in xrange(nabs): obs_lam_cont[kk] = out[abslines[kk]]['lam']

    dn4000_obs = prosp_dutils.measure_Dn4000(obslam,obsflux,ax=out['ax'][5])

    plt.tight_layout()
    plt.savefig(out_absobs, dpi=dpi)
    plt.close()

    out = {}
    # SAVE OBSERVATIONS
    obs = {}

    obs['lum'] = emline_flux[0,:]
    obs['lum_errup'] = emline_flux[1,:]
    obs['lum_errdown'] = emline_flux[2,:]
    obs['lum_chain'] = emline_chain

    obs['flux'] = emline_flux[0,:]  / dfactor / (1+z)
    obs['flux_errup'] = emline_flux[1,:]  / dfactor / (1+z)
    obs['flux_errdown'] = emline_flux[2,:]  / dfactor / (1+z)

    obs['dn4000'] = dn4000_obs
    obs['balmer_lum'] = obs_abs_flux
    obs['balmer_eqw_rest'] = obs_abs_eqw
    obs['balmer_flux'] = obs_abs_flux / dfactor / (1+z)
    obs['balmer_names'] = abslines
    obs['balmer_eqw_rest_chain'] = tobs_abs_eqw

    obs['continuum_obs'] = obs_abs_flux[:,0] / obs_abs_eqw[:,0] 
    obs['continuum_lam'] = obs_lam_cont

    out['obs'] = obs

    out['em_name'] = emline
    out['em_lam'] = em_wave
    out['sigsmooth'] = sigma_spec

    return out