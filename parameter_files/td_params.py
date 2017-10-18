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
              'outfile': APPS+'/prospector_alpha/results/td/GOODSS_47721',
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
              'agelims': [0.0,8.0,8.5,9.0,9.5,9.8,10.0],
              # Data info (phot = .cat, dat = .dat, fast = .fout)
              'datdir':APPS+'/prospector_alpha/data/3dhst/',
              'runname': 'td',
              'objname':'GOODSS_47721'
              }
############
# OBS
#############

def load_obs(objname=None, datdir=None, runname=None, err_floor=0.05, zperr=True, no_zp_corrs=False, **extras):

    ''' 
    objname: number of object in the 3D-HST COSMOS photometric catalog
    err_floor: the fractional error floor (0.05 = 5% floor)
    zp_err: inflate the errors by the zeropoint offsets from Skelton+14
    '''

    ### open file, load data
    photname = datdir + objname.split('_')[0] + '_' + runname + '.cat'
    with open(photname, 'r') as f:
        hdr = f.readline().split()
    dtype = np.dtype([(hdr[1],'S20')] + [(n, np.float) for n in hdr[2:]])
    dat = np.loadtxt(photname, comments = '#', delimiter=' ', dtype = dtype)

    ### extract filters, fluxes, errors for object
    # from ReadMe: "All fluxes are normalized to an AB zeropoint of 25, such that: magAB = 25.0-2.5*log10(flux)
    obj_idx = (dat['id'] == objname.split('_')[-1])
    filters = np.array([f[2:] for f in dat.dtype.names if f[0:2] == 'f_'])
    flux = np.squeeze([dat[obj_idx]['f_'+f] for f in filters])
    unc = np.squeeze([dat[obj_idx]['e_'+f] for f in filters])

    ### add correction to MIPS magnitudes (only MIPS 24 right now!)
    # due to weird MIPS filter conventions
    dAB_mips_corr = np.array([-0.03542,-0.07669,-0.03807]) # 24, 70, 160, in AB magnitudes
    dflux = 10**(-dAB_mips_corr/2.5)

    mips_idx = np.array(['mips_24um' in f for f in filters],dtype=bool)
    flux[mips_idx] *= dflux[0]
    unc[mips_idx] *= dflux[0]

    ### define photometric mask, convert to maggies
    phot_mask = (flux != unc) & (flux != -99.0) & (unc > 0)
    maggies = flux/(1e10)
    maggies_unc = unc/(1e10)

    # deal with the zeropoint errors
    # ~5% to ~20% effect
    # either REMOVE them or INFLATE THE ERRORS by them
    # in general, we don't inflate errors of space-based bands
    # if use_zp is set, RE-APPLY these offsets
    if (zperr) or (no_zp_corrs):
        no_zp_correction = ['irac1','irac2','irac3','irac4','f435w','f606w','f606wcand','f775w','f814w',
                            'f814wcand','f850lp','f850lpcand','f125w','f140w','f160w']
        zp_offsets = load_zp_offsets(None)
        band_names = np.array([x['Band'].lower()+'_'+x['Field'].lower() for x in zp_offsets])
        for ii,f in enumerate(filters):
            match = band_names == f
            if match.sum():
                in_exempt = len([s for s in no_zp_correction if s in f])
                if (no_zp_corrs) & (not in_exempt):
                    maggies[ii] /= zp_offsets[match]['Flux-Correction'][0]
                    maggies_unc[ii] /= zp_offsets[match]['Flux-Correction'][0]
                if zperr & (not in_exempt):
                    maggies_unc[ii] = ( (maggies_unc[ii]**2) + (maggies[ii]*(1-zp_offsets[match]['Flux-Correction'][0]))**2 ) **0.5

    ### implement error floor
    maggies_unc = np.clip(maggies_unc, maggies*err_floor, np.inf)

    ### if we have super negative flux, then mask it !
    ### where super negative is <0 with 95% certainty
    neg = (maggies < 0) & (np.abs(maggies/maggies_unc) > 2)
    phot_mask[neg] = False

    ### build output dictionary
    obs = {}
    obs['filters'] = observate.load_filters(filters)
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = phot_mask
    obs['maggies'] = maggies
    obs['maggies_unc'] =  maggies_unc
    obs['wavelength'] = None
    obs['spectrum'] = None
    obs['logify_spectrum'] = False

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

model_params.append({'name': 'massmet', 'N': 2,
                        'isfree': True,
                        'init': np.array([10,-0.5]),
                        'prior': None})

model_params.append({'name': 'logmass', 'N': 1,
                        'isfree': False,
                        'depends_on': massmet_to_logmass,
                        'init': 10.0,
                        'units': 'Msun',
                        'prior': None})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': False,
                        'init': -0.5,
                        'depends_on': massmet_to_logzsol,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior': None})
                        
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
                        'init': 4,
                        'units': 'index',
                        'prior_function_name': None,
                        'prior_args': None})
                        
model_params.append({'name': 'dust1', 'N': 1,
                        'isfree': False,
                        'depends_on': to_dust1,
                        'init': 1.0,
                        'units': '',
                        'prior': None})

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
                        'depends_on': tie_gas_logz,
                        'units': r'log Z/Z_\odot',
                        'prior': priors.TopHat(mini=-2.0, maxi=0.5)})

model_params.append({'name': 'gas_logu', 'N': 1, # scale with sSFR?
                        'isfree': False,
                        'init': -2.0,
                        'units': '',
                        'prior': priors.TopHat(mini=-4.0, maxi=-1.0)})

##### AGN dust ##############
model_params.append({'name': 'add_agn_dust', 'N': 1,
                        'isfree': False,
                        'init': True,
                        'units': '',
                        'prior': None})

model_params.append({'name': 'fagn', 'N': 1,
                        'isfree': True,
                        'init': 0.01,
                        'init_disp': 0.03,
                        'disp_floor': 0.02,
                        'units': '',
                        'prior': priors.LogUniform(mini=1e-5, maxi=3.0)})

model_params.append({'name': 'agn_tau', 'N': 1,
                        'isfree': True,
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
fit_order = ['massmet','z_fraction', 'dust2', 'dust_index', 'dust1_fraction', 'fagn', 'agn_tau']
tparams = [model_params[parnames.index(i)] for i in fit_order]
for param in model_params: 
    if param['name'] not in fit_order:
        tparams.append(param)
model_params = tparams

##### Mass-metallicity prior ######
class MassMet(priors.Prior):
    """A Gaussian prior designed to approximate the Gallazzi et al. 2005 
    stellar mass--stellar metallicity relationship.

    Must be updated to have relevant functions of `distribution` in `priors.py`
    in order to be run with a nested sampler.
    """

    prior_params = ['mass_mini', 'mass_maxi', 'z_mini', 'z_maxi']
    distribution = truncnorm
    massmet = np.loadtxt(os.getenv('APPS')+'/prospector_alpha/data/gallazzi_05_massmet.txt')

    def scale(self,mass):
        upper_84 = np.interp(mass, self.massmet[:,0], self.massmet[:,3]) 
        lower_16 = np.interp(mass, self.massmet[:,0], self.massmet[:,2])
        return (upper_84-lower_16)

    def loc(self,mass):
        return np.interp(mass, self.massmet[:,0], self.massmet[:,1])

    def get_args(self,mass):
        a = (self.params['z_mini'] - self.loc(mass)) / self.scale(mass)
        b = (self.params['z_maxi'] - self.loc(mass)) / self.scale(mass)
        return [a, b]

    @property
    def range(self):
        return ((self.params['mass_mini'], self.params['mass_maxi']),\
                (self.params['z_mini'], self.params['z_maxi']))

    def bounds(self, **kwargs):
        if len(kwargs) > 0:
            self.update(**kwargs)
        return self.range

    def __call__(self, x, **kwargs):
        """Compute the value of the probability density function at x and
        return the ln of that.

        :params x:
            x[0] = mass, x[1] = metallicity. Used to calculate the prior

        :param kwargs: optional
            All extra keyword arguments are used to update the `prior_params`.

        :returns lnp:
            The natural log of the prior probability at x, scalar or ndarray of
            same length as the prior object.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        p = np.atleast_2d(np.zeros_like(x))
        a, b = self.get_args(x[...,0])
        p[...,1] = self.distribution.pdf(x[...,1], a, b, loc=self.loc(x[...,0]), scale=self.scale(x[...,0]))
        with np.errstate(invalid='ignore'):
            p[...,1] = np.log(p[...,1])
        return p

    def sample(self, nsample=None, **kwargs):
        """Draw a sample from the prior distribution.

        :param nsample: (optional)
            Unused
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = np.random.uniform(low=self.params['mass_mini'],high=self.params['mass_maxi'])
        a, b = self.get_args(mass)
        met = self.distribution.rvs(a, b, loc=self.loc(mass), scale=self.scale(mass))
        return np.array([mass, met])

    def unit_transform(self, x, **kwargs):
        """Go from a value of the CDF (between 0 and 1) to the corresponding
        parameter value.

        :param x:
            A scalar or vector of same length as the Prior with values between
            zero and one corresponding to the value of the CDF.

        :returns theta:
            The parameter value corresponding to the value of the CDF given by
            `x`.
        """
        if len(kwargs) > 0:
            self.update(**kwargs)
        mass = x[0]*(self.params['mass_maxi'] - self.params['mass_mini']) + self.params['mass_mini']
        a, b = self.get_args(mass)
        met = self.distribution.ppf(x[1], a, b, loc=self.loc(mass), scale=self.scale(mass))
        return np.array([mass,met])

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

def load_model(objname=None, datdir=None, runname=None, agelims=[], zred=None, alpha_sfh=0.2, **extras):

    # we'll need this to access specific model parameters
    n = [p['name'] for p in model_params]

    # first calculate redshift and corresponding t_universe
    # if no redshift is specified, read from file
    if zred is None:
        datname = datdir + objname.split('_')[0] + '_' + runname + '.dat'
        dat = ascii.read(datname)
        idx = dat['phot_id'] == int(objname.split('_')[-1])
        zred = float(dat['z_best'][idx])
    tuniv = WMAP9.age(zred).value

    # now construct the nonparametric SFH
    # current scheme: six bins, four spaced equally in logarithmic 
    # space AFTER t=100 Myr + BEFORE tuniv-1 Gyr
    if tuniv > 5:
        tbinmax = (tuniv-2)*1e9
    else:
        tbinmax = (tuniv-1)*1e9
    agelims = agelims[:1] + np.linspace(agelims[1],np.log10(tbinmax),len(agelims)-2).tolist() + [np.log10(tuniv*1e9)]
    agebins = np.array([agelims[:-1], agelims[1:]])
    ncomp = len(agelims) - 1

    # load into `agebins` in the model_params dictionary
    model_params[n.index('agebins')]['N'] = ncomp
    model_params[n.index('agebins')]['init'] = agebins.T

    # now we do the computational z-fraction setup
    # number of zfrac variables = (number of SFH bins - 1)
    # set initial with a constant SFH
    # if alpha_SFH is a vector, use this as the alpha array
    # else assume all alphas are the same
    model_params[n.index('mass')]['N'] = ncomp
    model_params[n.index('z_fraction')]['N'] = ncomp-1
    if type(alpha_sfh) != type(np.array([])):
        alpha = np.repeat(alpha_sfh,ncomp-1)
    else:
        alpha = alpha_sfh
    tilde_alpha = np.array([alpha[i-1:].sum() for i in xrange(1,ncomp)])
    model_params[n.index('z_fraction')]['prior'] = priors.Beta(alpha=tilde_alpha, beta=alpha, mini=0.0, maxi=1.0)
    model_params[n.index('z_fraction')]['init'] = np.array([(i-1)/float(i) for i in range(ncomp,1,-1)])
    model_params[n.index('z_fraction')]['init_disp'] = 0.02

    # set mass-metallicity prior
    # insert redshift into model dictionary
    model_params[n.index('massmet')]['prior'] = MassMet(z_mini=-1.98, z_maxi=0.19, mass_mini=7, mass_maxi=12.5)
    model_params[n.index('zred')]['init'] = zred

    return sedmodel.SedModel(model_params)

