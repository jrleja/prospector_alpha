import numpy as np
import os
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis
from sedpy import observate
from astropy.cosmology import WMAP9
from astropy.io import fits

lsun = 3.846e33
pc = 3.085677581467192e18  # in cm

lightspeed = 2.998e18  # AA/s
to_cgs = lsun/(4.0 * np.pi * (pc*10)**2)
jansky_mks = 1e-26

#############
# RUN_PARAMS
#############

run_params = {'verbose':True,
              'debug': False,
              'outfile': os.getenv('APPS')+'/prospector_alpha/results/edo/bns',
              'nofork': True,
              # Optimizer params
              'ftol':0.5e-5, 
              'maxfev':5000,
              # MCMC params
              'nwalkers':620,
              'nburn':[150,200,200], 
              'niter': 10000,
              'interval': 0.2,
              # Convergence parameters
              'convergence_check_interval': 50,
              'convergence_chunks': 325,
              'convergence_kl_threshold': 0.016,
              'convergence_stable_points_criteria': 8, 
              'convergence_nhist': 50,
              # Model info
              'zcontinuous': 2,
              'compute_vega_mags': False,
              'initial_disp':0.1,
              'interp_type': 'logarithmic',
              'agelims': [0.0,8.0,8.5,9.0,9.5,9.75,9.95,10.04,10.1],
              # Data info
              'objname':'galaxy',
              }
run_params['outfile'] = run_params['outfile']+'_'+run_params['objname']

############
# OBS
#############
ftrans = {
          'GALEX FUV': 'galex_FUV', 
          'GALEX NUV': 'galex_NUV', 
          'PS1 g': 'PS1.g', 
          'PS1 r': 'PS1.r', 
          'PS1 i': 'PS1.i', 
          'PS1 z': 'PS1.z', 
          'PS1 y': 'PS1.y', 
          '2MASS J': 'twomass_J', 
          '2MASS H': 'twomass_H', 
          '2MASS K': 'twomass_Ks', 
          'WISE W1': 'wise_w1', 
          'WISE W2': 'wise_w2',
          'WISE W3': 'wise_w3', 
          'WISE W4': 'wise_w4'
         }



def load_obs(**extras):
    """
    let's do this
    """

    ### from email
    dat = {
           'GALEX FUV': (18.86, np.nan), # limit
           'GALEX NUV': (17.82, 0.09),
           'PS1 g': (12.80, 0.02),
           'PS1 r': (12.16, 0.01),
           'PS1 i': (11.81, 0.01),
           'PS1 z': (11.57, 0.01),
           'PS1 y': (11.36, 0.02),
           '2MASS J': (10.98, 0.02),
           '2MASS H': (10.82, 0.02),
           '2MASS K': (11.02, 0.02),
           'WISE W1': (11.92, 0.01),
           'WISE W2': (12.59, 0.01),
           'WISE W3': (13.70, 0.04),
           'WISE W4': (13.86, 0.19)
          }

    ### load fluxes, convert from AB magnitudes to maggies
    flux = 10**((-2./5)*np.array([dat[key][0] for key in dat.keys()]))

    # convert uncertainty to maggies
    unc = np.array([dat[key][1] for key in dat.keys()])*flux/1.086
    
    # do 3sigma floor properly
    idx = np.isnan(unc)
    unc[idx] = flux[idx]/3.
    flux[idx] = 0.0

    # create filter names
    fnames = [ftrans[f] for f in dat.keys()]

    ### implement 5% error floor
    unc = np.clip(unc, flux*0.05, np.inf)

    ### build output dictionary
    obs = {}
    obs['filters'] = observate.load_filters(fnames)
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = np.ones_like(flux,dtype=bool)
    obs['maggies'] = flux
    obs['maggies_unc'] =  unc
    obs['wavelength'] = None
    obs['spectrum'] = None

    '''
    import matplotlib.pyplot as plt
    uvot = np.array(['UVOT' in f for f in fnames],dtype=bool)
    plt.errorbar(obs['wave_effective'][~uvot],obs['maggies'][~uvot],yerr=obs['maggies_unc'][~uvot],fmt='o',ms=5,linestyle=' ',color='black')
    plt.errorbar(obs['wave_effective'][uvot],obs['maggies'][uvot],yerr=obs['maggies_unc'][uvot],fmt='o',ms=5,linestyle=' ',color='red')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('log(wavelength)')
    plt.ylabel(r'f$_{\nu}$')
    plt.show()
    print 1/0
    '''
    return obs

######################
# GENERATING FUNCTIONS
######################
def transform_logmass_to_mass(mass=None, logmass=None, **extras):
    return 10**logmass

def load_gp(**extras):
    return None, None

def tie_gas_logz(logzsol=None, **extras):
    return logzsol

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2

def transform_zfraction_to_sfrfraction(sfr_fraction=None, z_fraction=None, **extras):
    sfr_fraction[0] = 1-z_fraction[0]
    for i in xrange(1,sfr_fraction.shape[0]): sfr_fraction[i] =  np.prod(z_fraction[:i])*(1-z_fraction[i])
    return sfr_fraction

#############
# MODEL_PARAMS
#############

model_params = []

###### BASIC PARAMETERS ##########
model_params.append({'name': 'zred', 'N': 1,
                        'isfree': False,
                        'init': 0.00973,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

model_params.append({'name': 'lumdist', 'N': 1,
                        'isfree': False,
                        'init': 39.5,
                        'units': 'Mpc',
                        'prior_function': None,
                        'prior_args': None})

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

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': True,
                        'init': -0.5,
                        'init_disp': 0.25,
                        'disp_floor': 0.2,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior': priors.TopHat(mini=-1.98, maxi=0.19)})
                        
###### SFH   ########
model_params.append({'name': 'sfh', 'N':1,
                        'isfree': False,
                        'init': 0,
                        'units': None})

model_params.append({'name': 'logmass', 'N': 1,
                        'isfree': True,
                        'init': 10.0,
                        'units': 'Msun',
                        'prior': priors.TopHat(mini=5.0, maxi=13.0)})

model_params.append({'name': 'mass', 'N': 1,
                        'isfree': False,
                        'init': 1e10,
                        'depends_on': transform_logmass_to_mass,
                        'units': 'Msun',
                        'prior': priors.TopHat(mini=1e5, maxi=1e13)})

model_params.append({'name': 'agebins', 'N': 1,
                        'isfree': False,
                        'init': [],
                        'units': 'log(yr)',
                        'prior': None})

model_params.append({'name': 'sfr_fraction', 'N': 1,
                        'isfree': False,
                        'init': [],
                        'depends_on': transform_zfraction_to_sfrfraction,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=1.0)})

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
                             'prior_function_name': None,
                             'prior_args': None})

######## Dust Absorption ##############
model_params.append({'name': 'dust_type', 'N': 1,
                        'isfree': False,
                        'init': 4,
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})
                        
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'depends_on': to_dust1,
                        'init': 1.0,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=6.0)})

model_params.append({'name': 'dust1_fraction', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.8,
                        'disp_floor': 0.8,
                        'units': '',
                        'prior': priors.ClippedNormal(mini=0.0, maxi=2.0, mean=1.0, sigma=0.3)})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=3.0)})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior': priors.TopHat(mini=-2.2, maxi=0.4)})

model_params.append({'name': 'dust1_index', 'N': 1,
                        'isfree': False,
                        'init': -1.0,
                        'units': '',
                        'prior': priors.TopHat(mini=-1.5, maxi=-0.5)})

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
                        'prior_function': None,
                        'prior_args': None})

model_params.append({'name': 'duste_gamma', 'N': 1,
                        'isfree': True,
                        'init': 0.01,
                        'init_disp': 0.4,
                        'disp_floor': 0.3,
                        'units': None,
                        'prior': priors.TopHat(mini=0.0, maxi=1.0)})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 10.0,
                        'disp_floor': 5.0,
                        'units': None,
                        'prior': priors.TopHat(mini=0.1, maxi=25.0)})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': True,
                        'init': 3.0,
                        'init_disp': 3.0,
                        'disp_floor': 3.0,
                        'units': 'percent',
                        'prior': priors.TopHat(mini=0.0, maxi=10.0)})

###### Nebular Emission ###########
model_params.append({'name': 'add_neb_emission', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': r'log Z/Z_\odot',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'add_neb_continuum', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': r'log Z/Z_\odot',
                        'prior_function_name': None,
                        'prior_args': None})
                        
model_params.append({'name': 'nebemlineinspec', 'N': 1,
                        'isfree': False,
                        'init': False,
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'gas_logz', 'N': 1,
                        'isfree': False,
                        'init': 0.0,
                        'depends_on': tie_gas_logz,
                        'units': r'log Z/Z_\odot',
                        'prior': priors.TopHat(mini=-2.0, maxi=0.5)})

model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': False,
                        'init': -2.0,
                        'units': '',
                        'prior': priors.TopHat(mini=-4.0, maxi=-1.0)})

##### AGN dust ##############
model_params.append({'name': 'add_agn_dust', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': '',
                        'prior_function_name': None,
                        'prior_args': None})

model_params.append({'name': 'fagn', 'N': 1,
                        'isfree': True,
                        'init': 0.05,
                        'init_disp': 0.05,
                        'disp_floor': 0.01,
                        'units': '',
                        'prior': priors.LogUniform(mini=1e-5, maxi=3.0)})

model_params.append({'name': 'agn_tau', 'N': 1,
                        'isfree': True,
                        'init': 10.0,
                        'init_disp': 10,
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
#### so that major ones are fit first
parnames = [m['name'] for m in model_params]
fit_order = ['logmass','z_fraction', 'dust2', 'logzsol', 'dust_index', 'dust1_fraction', 'duste_qpah', 'duste_gamma', 'duste_umin']
tparams = [model_params[parnames.index(i)] for i in fit_order]
for param in model_params: 
    if param['name'] not in fit_order:
        tparams.append(param)
model_params = tparams
        
class FracSFH(FastStepBasis):
    
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

    def get_galaxy_spectrum(self, **params):
        self.update(**params)

        #### here's the custom fractional stuff
        fractions = np.array(self.params['sfr_fraction'])
        bin_fractions = np.append(fractions,(1-np.sum(fractions)))
        time_per_bin = []
        for (t1, t2) in self.params['agebins']: time_per_bin.append(10**t2-10**t1)
        bin_fractions *= np.array(time_per_bin)
        bin_fractions /= bin_fractions.sum()
        
        mass = bin_fractions*self.params['mass']
        mtot = self.params['mass'].sum()

        time, sfr, tmax = self.convert_sfh(self.params['agebins'], mass)
        self.ssp.params["sfh"] = 3 #Hack to avoid rewriting the superclass
        self.ssp.set_tabular_sfh(time, sfr)
        wave, spec = self.ssp.get_spectrum(tage=tmax, peraa=False)

        return wave, spec / mtot, self.ssp.stellar_mass / mtot

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

    sps = FracSFH(**extras)
    return sps

def load_model(agelims=[], **extras):

    #### CALCULATE TUNIV #####
    n = [p['name'] for p in model_params]
    zred = model_params[n.index('zred')]['init']
    tuniv = WMAP9.age(zred).value

    #### NONPARAMETRIC SFH ######
    agelims[-1] = np.log10(tuniv*1e9)
    agebins = np.array([agelims[:-1], agelims[1:]])
    ncomp = len(agelims) - 1

    #### SET UP AGEBINS
    model_params[n.index('agebins')]['N'] = ncomp
    model_params[n.index('agebins')]['init'] = agebins.T

    #### FRACTIONAL MASS INITIALIZATION
    # N-1 bins, last is set by x = 1 - np.sum(sfr_fraction)
    model_params[n.index('z_fraction')]['N'] = ncomp-1
    tilde_alpha = np.array([ncomp-i for i in xrange(1,ncomp)])
    model_params[n.index('z_fraction')]['prior'] = priors.Beta(alpha=tilde_alpha, beta=np.ones_like(tilde_alpha),mini=0.0,maxi=1.0)
    model_params[n.index('z_fraction')]['init'] =  model_params[n.index('z_fraction')]['prior'].sample()
    model_params[n.index('z_fraction')]['init_disp'] = 0.02

    model_params[n.index('sfr_fraction')]['N'] = ncomp-1
    model_params[n.index('sfr_fraction')]['prior'] = priors.TopHat(maxi=np.full(ncomp-1,1.0), mini=np.full(ncomp-1,0.0))
    model_params[n.index('sfr_fraction')]['init'] =  np.zeros(ncomp-1)+1./ncomp
    model_params[n.index('sfr_fraction')]['init_disp'] = 0.02

    #### CREATE MODEL
    model = sedmodel.SedModel(model_params)

    return model

model_type = sedmodel.SedModel

