import numpy as np
import os
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis
from sedpy import observate
from astropy.cosmology import WMAP9
from td_io import load_zp_offsets
from scipy.stats import truncnorm
from astropy.io import ascii

lsun = 3.846e33
pc = 3.085677581467192e18  # in cm

lightspeed = 2.998e18  # AA/s
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)
jansky_mks = 1e-26

#############
# RUN_PARAMS
#############
APPS = os.getenv('APPS')
run_params = {'verbose':True,
              'debug': False,
              'outfile': APPS+'/prospector_alpha/results/td_new/AEGIS_13',
              'nofork': True,
              # dynesty params
              'nested_bound': 'multi', # bounding method
              'nested_sample': 'rwalk', # sampling method
              'nested_walks': 50, # MC walks
              'nested_nlive_batch': 200, # size of live point "batches"
              'nested_nlive_init': 200, # number of initial live points
              'nested_weight_kwargs': {'pfrac': 1.0}, # weight posterior over evidence by 100%
              'nested_dlogz_init': 0.01,
              # Model info
              'zcontinuous': 2,
              'compute_vega_mags': False,
              'initial_disp':0.1,
              'interp_type': 'logarithmic',
              'agelims': [0.0,7.4772,8.0,8.5,9.0,9.5,9.8,10.0],
              # Data info (phot = .cat, dat = .dat, fast = .fout)
              'datdir':APPS+'/prospector_alpha/data/3dhst/',
              'runname': 'td_new',
              'objname':'AEGIS_13'
              }
############
# OBS
#############
def gauss(x,mu,sigma):
    return 1./np.sqrt(2*np.pi*sigma**2) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sigma, 2.)))

def sharp_quench(logm,agebins):
    """ sharp quenching event 1 Gyr ago
    SFR(star-forming) / SFR(quenched) = 50
    """
    dt = (10**agebins[:,1]-10**agebins[:,0])
    dt_q, sfr_contrast = 1e9, 50
    dt_s = (10**agebins.max()-10**agebins.min())-dt_q
    sfr_q = 10**logm/(sfr_contrast*dt_s+dt_q)
    sfr_s = sfr_q*sfr_contrast
    
    sfr = np.full(agebins.shape[0],sfr_s)
    amin = np.abs((10**agebins[:,1])-1e9).argmin()
    sfr[:amin+1] = sfr_q
    return sfr*dt

def recent_burst(logm, agebins):
    """ declining SFH + 20% of mass in burst @ 0.5 Gyr
    """

    # separate mass into components
    logm_smooth, m_burst = np.log10((10**logm)*0.8), (10**logm)*0.2
    mass_smooth = constant_sfr(logm_smooth,agebins)

    # define time bins
    t = 10**agebins.mean(axis=1)
    dt = (10**agebins[:,1]-10**agebins[:,0])
    mu, sigma = 0.5e9, 0.2e9
    sfr_burst = gauss(t,mu,sigma)*m_burst

    return mass_smooth + sfr_burst*dt

def intermediate_burst(logm, agebins):
    """ declining SFH + 20% of mass in burst at 2 Gyr
    """

    # separate mass into components
    logm_smooth, m_burst = np.log10((10**logm)*0.7), (10**logm)*0.3
    mass_smooth = declining_sfr(logm_smooth,agebins)

    # define time bins
    t = 10**agebins.mean(axis=1)
    dt = (10**agebins[:,1]-10**agebins[:,0])
    mu, sigma = 2e9, 0.3e9
    sfr_burst = gauss(t,mu,sigma)*m_burst

    return mass_smooth + sfr_burst*dt

def very_steep_declining_sfr(logm,agebins):
    """ use tau model with tage = max(agebins), tau = tage/2
    then put SFR(t_mid) * delta(t) mass in each bin 
    """
    # import matplotlib.pyplot as plt
    #plt.plot(t,sfr)

    # set tage, tau
    tage = 10**agebins.max()
    tau = tage/10.

    # define time bins
    t = 10**agebins.mean(axis=1)
    dt = (10**agebins[:,1]-10**agebins[:,0])

    # do the integral
    A = 10**logm / (tau*(1-np.exp(-tage/tau)))
    sfr = A*np.exp((t-tage)/tau)

    return sfr * dt

def steeply_declining_sfr(logm,agebins):
    """ use tau model with tage = max(agebins), tau = tage/2
    then put SFR(t_mid) * delta(t) mass in each bin 
    """
    # import matplotlib.pyplot as plt
    #plt.plot(t,sfr)

    # set tage, tau
    tage = 10**agebins.max()
    tau = tage/4.

    # define time bins
    t = 10**agebins.mean(axis=1)
    dt = (10**agebins[:,1]-10**agebins[:,0])

    # do the integral
    A = 10**logm / (tau*(1-np.exp(-tage/tau)))
    sfr = A*np.exp((t-tage)/tau)

    return sfr * dt

def declining_sfr(logm,agebins):
    """ use tau model with tage = max(agebins), tau = tage/2
    then put SFR(t_mid) * delta(t) mass in each bin 
    """
    # import matplotlib.pyplot as plt
    #plt.plot(t,sfr)

    # set tage, tau
    tage = 10**agebins.max()
    tau = tage/2.

    # define time bins
    t = 10**agebins.mean(axis=1)
    dt = (10**agebins[:,1]-10**agebins[:,0])

    # do the integral
    A = 10**logm / (tau*(1-np.exp(-tage/tau)))
    sfr = A*np.exp((t-tage)/tau)

    return sfr * dt

def constant_sfr(logm,agebins):
    dt = (10**agebins[:,1]-10**agebins[:,0])
    sfr = 10**logm/(10**agebins.max()-10**agebins.min())
    return sfr*dt

def rising_sfr(logm,agebins):
    
    # set tage, tau
    tage = 10**agebins.max()
    tau = tage/2.

    # define time bins
    t = 10**agebins.mean(axis=1)
    dt = (10**agebins[:,1]-10**agebins[:,0])

    # do the integral
    A = 10**logm / (tau*(np.exp(tage/tau)-1))
    sfr = A*np.exp((tage-t)/tau)

    return sfr*dt

def steeply_rising_sfr(logm,agebins):
    
    # set tage, tau
    tage = 10**agebins.max()
    tau = tage/4.

    # define time bins
    t = 10**agebins.mean(axis=1)
    dt = (10**agebins[:,1]-10**agebins[:,0])

    # do the integral
    A = 10**logm / (tau*(np.exp(tage/tau)-1))
    sfr = A*np.exp((tage-t)/tau)

    return sfr*dt

def mock_params(mock_key,agebins):

    # set mass and SFH
    logm = 10
    if mock_key == 1:
        masses = declining_sfr(logm,agebins)
    if mock_key == 2:
        masses = constant_sfr(logm,agebins)
    if mock_key == 3:
        masses = rising_sfr(logm,agebins)
    if mock_key == 4:
        masses = steeply_declining_sfr(logm,agebins)
    if mock_key == 5:
        masses = steeply_rising_sfr(logm,agebins)
    if mock_key == 6:
        masses = recent_burst(logm,agebins)
    if mock_key == 7:
        masses = intermediate_burst(logm,agebins)
    if mock_key == 8:
        masses = sharp_quench(logm,agebins)
    if mock_key == 9:
        masses = very_steep_declining_sfr(logm,agebins)
    mass,zfrac = masses_to_zfrac(mass=masses,agebins=agebins)

    # then set dust, metallicity parameters and return theta vector
    # logzsol, dust2, dust_index, dust1_fraction
    theta = np.array([np.log10(mass)]+zfrac.tolist()+[0.0,0.3])

    return theta

def load_obs(mock_key=1, **extras):

    # generate filters and mask
    fnames  = ['galex_FUV','galex_NUV'] + \
              ['sdss_u0','sdss_g0','sdss_r0','sdss_i0','sdss_z0'] + \
              ['twomass_J','twomass_H','twomass_Ks'] + \
              ['spitzer_irac_ch1','spitzer_irac_ch2','spitzer_irac_ch3','spitzer_irac_ch4']
    filters = observate.load_filters(fnames)
    phot_mask = np.ones_like(filters,dtype=bool)

    obs = {}
    obs['filters'] = filters
    obs['phot_mask'] = phot_mask
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['logify_spectrum'] = False

    # generate mock parameters
    model = load_model(**extras)
    model.params['nebemlineinspec'] = True
    theta = mock_params(mock_key,model.params['agebins'])
    sps = load_sps(**extras)
    spec,maggies,sm = model.mean_model(theta, obs, sps=sps)

    ### build output dictionary
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['maggies'] = maggies
    obs['spec_true'] = spec
    obs['lam_true'] = sps.wavelengths

    return obs

##########################
# TRANSFORMATION FUNCTIONS
##########################
def load_gp(**extras):
    return None, None

def tie_gas_logz(logzsol=None, **extras):
    return logzsol

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2

def massmet_to_logmass(massmet=None,**extras):
    return massmet[0]

def massmet_to_logzsol(massmet=None,**extras):
    return massmet[1]

def zfrac_to_sfrac(z_fraction=None, **extras):
    """This transforms from latent, independent `z` variables to sfr
    fractions. The transformation is such that sfr fractions are drawn from a
    Dirichlet prior.  See Betancourt et al. 2010
    """
    sfr_fraction = np.zeros(len(z_fraction) + 1)
    sfr_fraction[0] = 1.0 - z_fraction[0]
    for i in range(1, len(z_fraction)):
        sfr_fraction[i] = np.prod(z_fraction[:i]) * (1.0 - z_fraction[i])
    sfr_fraction[-1] = 1 - np.sum(sfr_fraction[:-1])

    return sfr_fraction

def zfrac_to_masses(logmass=None, z_fraction=None, agebins=None, **extras):
    """This transforms from latent, independent `z` variables to sfr fractions
    and then to bin mass fractions. The transformation is such that sfr
    fractions are drawn from a Dirichlet prior.  See Betancourt et al. 2010
    :returns masses:
        The stellar mass formed in each age bin.
    """
    # sfr fractions (e.g. Leja 2017)
    sfr_fraction = zfrac_to_sfrac(z_fraction)
    # convert to mass fractions
    time_per_bin = np.diff(10**agebins, axis=-1)[:,0]
    sfr_fraction *= np.array(time_per_bin)
    sfr_fraction /= sfr_fraction.sum()
    masses = 10**logmass * sfr_fraction

    return masses

def masses_to_zfrac(mass=None, agebins=None, **extras):
    """The inverse of zfrac_to_masses, for setting mock parameters based on
    real bin masses.
    """
    total_mass = mass.sum()
    time_per_bin = np.diff(10**agebins, axis=-1)[:,0]
    sfr_fraction = mass / time_per_bin
    sfr_fraction /= sfr_fraction.sum()
    z_fraction = np.zeros(len(sfr_fraction) - 1)
    z_fraction[0] = 1 - sfr_fraction[0]
    for i in range(1, len(z_fraction)):
        z_fraction[i] = 1.0 - sfr_fraction[i] / np.prod(z_fraction[:i])

    return total_mass, z_fraction

#############
# MODEL_PARAMS
#############
model_params = []

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

model_params.append({'name': 'add_igm_absorption', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'add_agb_dust_model', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': None,
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'pmetals', 'N': 1,
                        'isfree': False,
                        'init': -99,
                        'units': '',
                        'prior_function': None,
                        'prior_args': {'mini':-3, 'maxi':-1}})

model_params.append({'name': 'logmass', 'N': 1,
                        'isfree': True,
                        'init': 10.0,
                        'units': 'Msun',
                        'prior': priors.TopHat(mini=6, maxi=14)})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -0.5,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior': priors.TopHat(mini=-1.98, maxi=0.19)})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N':1,
                        'isfree': False,
                        'init': 0,
                        'units': None})

model_params.append({'name': 'mass', 'N': 1,
                     'isfree': False,
                     'depends_on': zfrac_to_masses,
                     'init': 1.,
                     'units': r'M$_\odot$',})

model_params.append({'name': 'agebins', 'N': 1,
                        'isfree': False,
                        'init': [],
                        'units': 'log(yr)',
                        'prior': None})

model_params.append({'name': 'z_fraction', 'N': 1,
                        'isfree': True,
                        'init': [],
                        'units': '',
                        'prior': priors.Beta(alpha=1.0, beta=1.0,mini=0.0,maxi=1.0)})

########    IMF  ##############
model_params.append({'name': 'imf_type', 'N': 1,
                             'isfree': False,
                             'init': 1, #1 = chabrier
                             'units': None,
                             'prior': None})

######## Dust Absorption ##############
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 2,
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})
                        
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior': None})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=3.0)})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': '',
                        'prior': priors.TopHat(mini=-1.0, maxi=0.4)})

model_params.append({'name': 'dust1_index', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'prior': None})

model_params.append({'name': 'dust_tesc', 'N': 1,
                        'isfree': False,
                        'init': 7.0,
                        'units': 'log(Gyr)',
                        'prior_function_name': None,
                        'prior_args': None})

###### Dust Emission ##############
model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior': None})

model_params.append({'name': 'duste_gamma', 'N': 1,
                        'isfree': False,
                        'init': 0.01,
                        'init_disp': 0.2,
                        'disp_floor': 0.15,
                        'units': None,
                        'prior': priors.TopHat(mini=0.0, maxi=1.0)})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': False,
                        'init': 1.0,
                        'init_disp': 5.0,
                        'disp_floor': 4.5,
                        'units': None,
                        'prior': priors.TopHat(mini=0.1, maxi=25.0)})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': False,
                        'init': 2.0,
                        'init_disp': 3.0,
                        'disp_floor': 3.0,
                        'units': 'percent',
                        'prior': priors.TopHat(mini=0.0, maxi=7.0)})

###### Nebular Emission ###########
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': r'log Z/Z_\odot',
                        'prior': None})

model_params.append({'name': 'add_neb_continuum', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': r'log Z/Z_\odot',
                        'prior': None})

model_params.append({'name': 'nebemlineinspec', 'N': 1,
                        'isfree': False,
                        'init': False,
                        'prior': None})

model_params.append({'name': 'gas_logz', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'units': r'log Z/Z_\odot',
                        'prior': priors.TopHat(mini=-2.0, maxi=0.5)})

model_params.append({'name': 'gas_logu', 'N': 1, # scale with sSFR?
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'prior': priors.TopHat(mini=-4.0, maxi=-1.0)})

##### AGN dust ##############
model_params.append({'name': 'add_agn_dust', 'N': 1,
                        'isfree': False,
                        'init': False,
                        'units': '',
                        'prior': None})

model_params.append({'name': 'fagn', 'N': 1,
                        'isfree': False,
                        'init': 0.01,
                        'init_disp': 0.03,
                        'disp_floor': 0.02,
                        'units': '',
                        'prior': priors.LogUniform(mini=1e-5, maxi=3.0)})

model_params.append({'name': 'agn_tau', 'N': 1,
                        'isfree': False,
                        'init': 20.0,
                        'init_disp': 5,
                        'disp_floor': 2,
                        'units': '',
                        'prior': priors.LogUniform(mini=5.0, maxi=150.0)})

####### Calibration ##########
model_params.append({'name': 'phot_jitter', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'init_disp': 0.5,
                        'units': 'fractional maggies (mags/1.086)',
                        'prior': priors.TopHat(mini=0.0, maxi=0.5)})

####### Units ##########
model_params.append({'name': 'peraa', 'N': 1,
                     'isfree': False,
                     'init': False})

model_params.append({'name': 'mass_units', 'N': 1,
                     'isfree': False,
                     'init': 'mformed'})

#### resort list of parameters 
# because we can
parnames = [m['name'] for m in model_params]
fit_order = ['logmass','z_fraction', 'logzsol', 'dust2']
tparams = [model_params[parnames.index(i)] for i in fit_order]
for param in model_params: 
    if param['name'] not in fit_order:
        tparams.append(param)
model_params = tparams

###### Redefine SPS ######
class NebSFH(FastStepBasis):
    
    @property
    def emline_wavelengths(self):
        return self.ssp.emline_wavelengths

    @property
    def get_nebline_luminosity(self):
        """Emission line luminosities in units of Lsun per solar mass formed
        """
        return self.ssp.emline_luminosity/self.params['mass'].sum()

    def nebline_photometry(self,filters,z):
        """analytically calculate emission line contribution to photometry
        """
        emlams = self.emline_wavelengths * (1+z)
        elums = self.get_nebline_luminosity # Lsun / solar mass formed
        flux = np.empty(len(filters))
        for i,filt in enumerate(filters):
            # calculate transmission at nebular emission
            trans = np.interp(emlams, filt.wavelength, filt.transmission, left=0., right=0.)
            idx = (trans > 0)
            if True in idx:
                flux[i] = (trans[idx]*emlams[idx]*elums[idx]).sum()/filt.ab_zero_counts
            else:
                flux[i] = 0.0
        return flux

    def get_spectrum(self, outwave=None, filters=None, peraa=False, **params):
        """Get a spectrum and SED for the given params.
        ripped from SSPBasis
        addition: check for flag nebeminspec. if not true,
        add emission lines directly to photometry
        """

        # Spectrum in Lsun/Hz per solar mass formed, restframe
        wave, spectrum, mfrac = self.get_galaxy_spectrum(**params)

        # Redshifting + Wavelength solution
        # We do it ourselves.
        a = 1 + self.params.get('zred', 0)
        af = a
        b = 0.0

        if 'wavecal_coeffs' in self.params:
            x = wave - wave.min()
            x = 2.0 * (x / x.max()) - 1.0
            c = np.insert(self.params['wavecal_coeffs'], 0, 0)
            # assume coeeficients give shifts in km/s
            b = chebval(x, c) / (lightspeed*1e-13)

        wa, sa = wave * (a + b), spectrum * af  # Observed Frame
        if outwave is None:
            outwave = wa
        
        spec_aa = lightspeed/wa**2 * sa # convert to perAA
        # Observed frame photometry, as absolute maggies
        if filters is not None:
            mags = observate.getSED(wa, spec_aa * to_cgs, filters)
            phot = np.atleast_1d(10**(-0.4 * mags))
        else:
            phot = 0.0

        ### if we don't have emission lines, add them
        if (not self.params['nebemlineinspec']) and self.params['add_neb_emission']:
            phot += self.nebline_photometry(filters,a-1)*to_cgs

        # Spectral smoothing.
        do_smooth = (('sigma_smooth' in self.params) and
                     ('sigma_smooth' in self.reserved_params))
        if do_smooth:
            # We do it ourselves.
            smspec = self.smoothspec(wa, sa, self.params['sigma_smooth'],
                                     outwave=outwave, **self.params)
        elif outwave is not wa:
            # Just interpolate
            smspec = np.interp(outwave, wa, sa, left=0, right=0)
        else:
            # no interpolation necessary
            smspec = sa

        # Distance dimming and unit conversion
        zred = self.params.get('zred', 0.0)
        if (zred == 0) or ('lumdist' in self.params):
            # Use 10pc for the luminosity distance (or a number
            # provided in the dist key in units of Mpc)
            dfactor = (self.params.get('lumdist', 1e-5) * 1e5)**2
        else:
            lumdist = WMAP9.luminosity_distance(zred).value
            dfactor = (lumdist * 1e5)**2
        if peraa:
            # spectrum will be in erg/s/cm^2/AA
            smspec *= to_cgs / dfactor * lightspeed / outwave**2
        else:
            # Spectrum will be in maggies
            smspec *= to_cgs / dfactor / 1e3 / (3631*jansky_mks)

        # Convert from absolute maggies to apparent maggies
        phot /= dfactor

        # Mass normalization
        mass = np.sum(self.params.get('mass', 1.0))
        if np.all(self.params.get('mass_units', 'mstar') == 'mstar'):
            # Convert from current stellar mass to mass formed
            mass /= mfrac

        return smspec * mass, phot * mass, mfrac

def load_sps(**extras):

    sps = NebSFH(**extras)
    return sps

def load_model(alpha_sfh=0.2, **extras):

    # we'll need this to access specific model parameters
    n = [p['name'] for p in model_params]

    # create SFH bins
    nbins = 2000
    zred = model_params[n.index('zred')]['init']
    tuniv = WMAP9.age(zred).value
    agelims = np.linspace(6,np.log10(tuniv*1e9),nbins+1)
    agebins = np.array([agelims[:-1], agelims[1:]])

    # load into `agebins` in the model_params dictionary
    model_params[n.index('agebins')]['N'] = nbins
    model_params[n.index('agebins')]['init'] = agebins.T

    # z-fraction setup
    model_params[n.index('mass')]['N'] = nbins
    model_params[n.index('z_fraction')]['N'] = nbins-1
    if type(alpha_sfh) != type(np.array([])):
        alpha = np.repeat(alpha_sfh,nbins-1)
    else:
        alpha = alpha_sfh
    tilde_alpha = np.array([alpha[i-1:].sum() for i in xrange(1,nbins)])
    model_params[n.index('z_fraction')]['prior'] = priors.Beta(alpha=tilde_alpha, beta=alpha, mini=0.0, maxi=1.0)
    model_params[n.index('z_fraction')]['init'] = np.array([(i-1)/float(i) for i in range(nbins,1,-1)])

    return sedmodel.SedModel(model_params)

