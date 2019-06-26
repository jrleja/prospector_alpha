import numpy as np
import os
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis
from sedpy import observate
from astropy.cosmology import WMAP9
from td_io import load_zp_offsets
from scipy.stats import truncnorm
from astropy.io import fits

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
              'outfile': APPS+'/prospector_alpha/results/cosmos_delta/575118',
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
              'nbins_sfh': 7,
              'sigma': 0.3,
              'df': 2,
              'agelims': [0.0,7.4772,8.0,8.5,9.0,9.5,9.8,10.0],
              # Data info (phot = .cat, dat = .dat, fast = .fout)
              'datdir':APPS+'/prospector_alpha/data/cosmos/',
              'objname': 575118
              }
############
# OBS
#############
# from Table 3 of Laigle+16
# 'offset' is offset in AB magnitudes from photo-z's
# 'extinct' == galactic extinction, factor to multiply by E(B-V) and subtract from magnitudes
ftrans = {
    '100': {'name': 'herschel_pacs_100', 'offset': 0.0,'extinct': 0.0},
    '160': {'name': 'herschel_pacs_160', 'offset': 0.0,'extinct': 0.0},
    '24': {'name': 'mips_24um_cosmos', 'offset': 0.0,'extinct': 0.0},
    '250': {'name': 'herschel_spire_250', 'offset': 0.0,'extinct': 0.0},
    '350': {'name': 'herschel_spire_350', 'offset': 0.0,'extinct': 0.0},
    '500': {'name': 'herschel_spire_500', 'offset': 0.0,'extinct': 0.0},
    '814W': {'name': 'f814w_cosmos', 'offset': np.nan,'extinct': np.nan}, # no extinction provided
    'B': {'name': 'b_cosmos', 'offset': 0.146,'extinct': 4.020},
    'GALEX_FUV': {'name': 'galex_FUV', 'offset': np.nan,'extinct': np.nan}, # no extinction provided
    'GALEX_NUV': {'name': 'galex_NUV', 'offset': 0.128,'extinct': 8.621},
    'H': {'name': 'uvista_h_cosmos', 'offset': 0.055,'extinct': 0.563},
    'Hw': {'name': 'wircam_H', 'offset': -0.031,'extinct': 0.563},
    'IA484': {'name': 'ia484_cosmos', 'offset': -0.002,'extinct': 3.621},
    'IA527': {'name': 'ia527_cosmos', 'offset': 0.025,'extinct': 3.264},
    'IA624': {'name': 'ia624_cosmos', 'offset': -0.010,'extinct': 2.694},
    'IA679': {'name': 'ia679_cosmos', 'offset': -0.194,'extinct': 2.430},
    'IA738': {'name': 'ia738_cosmos', 'offset': 0.020,'extinct': 2.150},
    'IA767': {'name': 'ia767_cosmos', 'offset': 0.024,'extinct': 1.996},
    'IB427': {'name': 'ia427_cosmos', 'offset': 0.050,'extinct': 4.260},
    'IB464': {'name': 'ia464_cosmos', 'offset': -0.014,'extinct': 3.843},
    'IB505': {'name': 'ia505_cosmos', 'offset': -0.013,'extinct': 3.425},
    'IB574': {'name': 'ia574_cosmos', 'offset': 0.065,'extinct': 2.937},
    'IB709': {'name': 'ia709_cosmos', 'offset': 0.017,'extinct': 2.289},
    'IB827': {'name': 'ia827_cosmos', 'offset': -0.005,'extinct': 1.747},
    'J': {'name': 'uvista_j_cosmos', 'offset': 0.017,'extinct': 0.871},
    'Ks': {'name': 'uvista_ks_cosmos', 'offset': -0.001,'extinct': 0.364},
    'Ksw': {'name': 'wircam_Ks', 'offset': 0.068,'extinct': 0.364},
    'NB711': {'name': 'NB711.SuprimeCam', 'offset': 0.040,'extinct': 2.268},
    'NB816': {'name': 'NB816.SuprimeCam', 'offset': -0.035,'extinct': 1.787},
    'V': {'name': 'v_cosmos', 'offset': -0.117,'extinct': 3.117},
    'Y': {'name': 'uvista_y_cosmos', 'offset': 0.001,'extinct': 1.211},
    'ip': {'name': 'ip_cosmos', 'offset': 0.020,'extinct': 1.991},
    'r': {'name': 'r_cosmos', 'offset': -0.012,'extinct': 2.660},
    'u': {'name': 'u_cosmos', 'offset': 0.010,'extinct': 4.660},
    'yHSC': {'name': 'hsc_y', 'offset': -0.014,'extinct': 1.298},
    'zp': {'name': 'zp', 'offset': -0.084,'extinct': 1.461},
    'zpp': {'name': 'zpp', 'offset': -0.084,'extinct': 1.461},
    'SPLASH_1': {'name': 'irac1_cosmos', 'offset': -0.025,'extinct': 0.162},
    'SPLASH_2': {'name': 'irac2_cosmos', 'offset': -0.005,'extinct': 0.111},
    'SPLASH_3': {'name': 'irac3_cosmos', 'offset': -0.061,'extinct': 0.075},
    'SPLASH_4': {'name': 'irac4_cosmos', 'offset': -0.025,'extinct': 0.045}
}

def load_obs(objname=None, datdir=None, err_floor=0.05, zperr=True, no_zp_corrs=False, **extras):
    """
    -- load catalog, match object name
    -- pull out 3" aperture fluxes + errors and full correction, 'total' fluxes + errors
    -- calculate fluxes & errors, accounting for:
        -- galactic extinction
        -- aperture to total correction
        -- inflate errors by zero-point offsets
    -- connect to filter definitions from Laigle
    -- convert to maggies
    -- phot_mask
    -- error floor
    -- Ly-a masking
    """

    # load data, find object
    hdu = fits.open(datdir + 'laigle_catalog.fits')[1]
    oidx = (hdu.data['NUMBER'] == int(objname))

    # pull out filters (sigh @ lack of convention)
    names = hdu.data.dtype.names
    fnames = []
    for name in names:
        split_name = name.split('_')
        if (len(split_name) == 1): continue
        if split_name[1] == 'FLUX':
            fnames += [split_name[0]]
        elif (split_name[0] == 'FLUX') & (split_name[1] not in ['RADIUS','CHANDRA','XMM','NUSTAR']):
            fnames += ['_'.join(split_name[1:])]
        elif (split_name[0] == 'SPLASH'):
            fnames += ['_'.join(split_name[:2])]
    
    # throw out duplicates + filters w/ missing info
    fnames = np.unique(fnames)
    fidx = np.array([True if np.isfinite(ftrans[f]['offset']) else False for f in fnames],dtype=bool) 
    fnames = fnames[fidx]

    # pull out fluxes + errors, correcting for aperture
    # we INFLATE errors by zero-point corrections & do not apply!
    flux, unc = [], []
    aper_to_tot = hdu.data['OFFSET_MAG'][oidx]
    ebv = hdu.data['EBV'][oidx]
    for fname in fnames:
        dmag = -ebv*ftrans[fname]['extinct']
        ecorr = 10**(ftrans[fname]['offset']/-2.5)
        if (fname+'_FLUX_APER3' in names):
            dmag += aper_to_tot 
            corr = 10**(dmag[0]/-2.5)
            flux += [hdu.data[fname+'_FLUX_APER3'][oidx][0]*corr]
            unc  += [hdu.data[fname+'_FLUXERR_APER3'][oidx][0]*corr*ecorr]
        elif 'SPLASH' in fname:
            corr = 10**(dmag[0]/-2.5)
            flux += [hdu.data[fname+'_FLUX'][oidx][0]*corr]
            unc  += [hdu.data[fname+'_FLUX_ERR'][oidx][0]*corr*ecorr]
        else:
            corr = 10**(dmag[0]/-2.5)
            flux += [hdu.data['FLUX_'+fname][oidx][0]*corr]
            try:
                unc  += [hdu.data['FLUXERRTOT_'+fname][oidx][0]*corr*ecorr]
            except KeyError:
                unc  += [hdu.data['FLUXERR_'+fname][oidx][0]*corr*ecorr]

    # convert to standard filter names & load filters
    filters = np.array([ftrans[f]['name'] for f in fnames])
    ofilters = observate.load_filters(filters)

    # convert to maggies
    # the MIR / FIR fluxes are in mJy; special conversion here...
    fconv = 1e-6 / 3631
    flux = np.array(flux)*fconv
    unc = np.array(unc)*fconv
    diff_units = ['herschel_pacs_100', 'herschel_pacs_160', 'mips_24um_cosmos',\
                  'herschel_spire_250', 'herschel_spire_350', 'herschel_spire_500']
    tidx = np.in1d(filters,diff_units)
    flux[tidx] *= 1e3
    unc[tidx] *= 1e3

    ### add correction to MIPS magnitudes (only MIPS 24 right now!)
    # due to weird MIPS filter conventions
    dAB_mips_corr = np.array([-0.03542,-0.07669,-0.03807]) # 24, 70, 160, in AB magnitudes
    dflux = 10**(-dAB_mips_corr/2.5)
    mips_idx = np.array(['mips_24um' in f for f in filters],dtype=bool)
    flux[mips_idx] *= dflux[0]
    unc[mips_idx] *= dflux[0]

    ### define photometric mask
    phot_mask = np.isfinite(flux)

    ### implement error floor
    unc = np.clip(unc, flux*err_floor, np.inf)

    ### mask anything touching or bluewards of Ly-a
    zred = float(hdu.data['zpdf'][oidx])
    wavemax = np.array([f.wavelength[f.transmission > (f.transmission.max()*0.1)].max() for f in ofilters]) / (1+zred)
    wavemin = np.array([f.wavelength[f.transmission > (f.transmission.max()*0.1)].min() for f in ofilters]) / (1+zred)
    filtered = [1230]
    for f in filtered: phot_mask[(wavemax > f) & (wavemin < f)] = False
    phot_mask[wavemin < 1200] = False

    ### build output dictionary
    obs = {}
    obs['filters'] = ofilters
    obs['wave_effective'] = np.array([filt.wave_effective for filt in obs['filters']])
    obs['phot_mask'] = phot_mask
    obs['maggies'] = flux
    obs['maggies_unc'] =  unc
    obs['wavelength'] = None
    obs['spectrum'] = None

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

def logmass_to_masses(massmet=None, logsfr_ratios=None, agebins=None, **extras):
    logsfr_ratios = np.clip(logsfr_ratios,-10,10) # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**logsfr_ratios
    dt = (10**agebins[:,1]-10**agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**massmet[0]) / coeffs.sum()

    return m1 * coeffs

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
                     'depends_on': logmass_to_masses,
                     'init': 1.,
                     'units': r'M$_\odot$',})

model_params.append({'name': 'agebins', 'N': 1,
                        'isfree': False,
                        'init': [],
                        'units': 'log(yr)',
                        'prior': None})

model_params.append({'name': 'logsfr_ratios', 'N': 7,
                        'isfree': True,
                        'init': [],
                        'units': '',
                        'prior': None})

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
                        'prior': priors.ClippedNormal(mini=0.0, maxi=4.0, mean=0.3, sigma=1)})

model_params.append({'name': 'dust_index', 'N': 1,
                        'isfree': True,
                        'init': 0.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
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
                        'isfree': True,
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

#### resort list of parameters for later display purposes
parnames = [m['name'] for m in model_params]
fit_order = ['massmet','logsfr_ratios', 'dust2', 'dust_index', 'dust1_fraction', 'fagn', 'agn_tau', 'gas_logz']
tparams = [model_params[parnames.index(i)] for i in fit_order]
for param in model_params: 
    if param['name'] not in fit_order:
        tparams.append(param)
model_params = tparams

##### Mass-metallicity prior ######
class MassMet(priors.Prior):
    """A Gaussian prior designed to approximate the Gallazzi et al. 2005 
    stellar mass--stellar metallicity relationship.
    """

    prior_params = ['mass_mini', 'mass_maxi', 'z_mini', 'z_maxi']
    distribution = truncnorm
    massmet = np.loadtxt(os.getenv('APPS')+'/prospector_alpha/data/gallazzi_05_massmet.txt')

    def __len__(self):
        """ Hack to work with Prospector 0.3
        """
        return 2

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
        mass = np.random.uniform(low=self.params['mass_mini'],high=self.params['mass_maxi'],size=nsample)
        a, b = self.get_args(mass)
        met = self.distribution.rvs(a, b, loc=self.loc(mass), scale=self.scale(mass), size=nsample)

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
        check for flag nebeminspec. if not true,
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

def load_model(nbins_sfh=7, sigma=0.3, df=2, agelims=[], objname=None, datdir=None, zred=None, **extras):

    # we'll need this to access specific model parameters
    n = [p['name'] for p in model_params]

    # first calculate redshift and corresponding t_universe
    # if no redshift is specified, read from file
    if zred is None:
        hdu = fits.open(datdir + 'laigle_catalog.fits')[1]
        oidx = (hdu.data['NUMBER'] == int(objname))
        zred = float(hdu.data['zpdf'][oidx])
    tuniv = WMAP9.age(zred).value*1e9

    # now construct the nonparametric SFH
    # current scheme: last bin is 15% age of the Universe, first two are 0-30, 30-100
    # remaining N-3 bins spaced equally in logarithmic space
    tbinmax = (tuniv*0.85)
    agelims = agelims[:2] + np.linspace(agelims[2],np.log10(tbinmax),nbins_sfh-2).tolist() + [np.log10(tuniv)]
    agebins = np.array([agelims[:-1], agelims[1:]])

    # load nvariables and agebins
    model_params[n.index('agebins')]['N'] = nbins_sfh
    model_params[n.index('agebins')]['init'] = agebins.T
    model_params[n.index('mass')]['N'] = nbins_sfh
    model_params[n.index('logsfr_ratios')]['N'] = nbins_sfh-1
    model_params[n.index('logsfr_ratios')]['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    model_params[n.index('logsfr_ratios')]['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
                                                                      scale=np.full(nbins_sfh-1,sigma),
                                                                      df=np.full(nbins_sfh-1,df))
    # set mass-metallicity prior
    # insert redshift into model dictionary
    model_params[n.index('massmet')]['prior'] = MassMet(z_mini=-1.98, z_maxi=0.19, mass_mini=7, mass_maxi=12.5)
    model_params[n.index('zred')]['init'] = zred

    return sedmodel.SedModel(model_params)

