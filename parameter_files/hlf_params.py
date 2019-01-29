import numpy as np
import os
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis
from sedpy import observate
from astropy.cosmology import WMAP9
from td_io import load_zp_offsets
from scipy.stats import truncnorm
from astropy.io import ascii
from astropy import units as u

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
              'outfile': APPS+'/prospector_alpha/results/hlf/hlf_params',
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
              'objname': 1
              }
############
# OBS
#############

trans_phot = {
    'f225w': 'HST_WF3_UVIS2.f225w',
    'f275w': 'wfc3_uvis_f275w',
    'f336w': 'wfc3_uvis_f336w',
    'f390w': 'f390w',
    'f435w': 'ACS_f435W',
    'f475w': 'wfc3_uvis_f475w',
    'f555w': 'acs_wfc_f555w',
    'f606w': 'f606w_aegis',
    'f625w': 'acs_wfc_f625w',
    'f775w': 'ACS_f775W',
    'f814w': 'f814w_aegis',
    'f850lp': 'acs_wfc_f850lp',
    'f098m': 'HST_WFC3_IR.F098M',
    'f105w': 'wfc3_ir_f105w',
    'f110w': 'wfc3_ir_f110w',
    'f125w': 'f125w_aegis',
    'f140w': 'f140w_aegis',
    'f160w': 'f160w_aegis',
    'irac1': 'irac1_aegis',
    'irac2': 'irac2_aegis'
}

def load_obs(objname=None, err_floor=0.05, **extras):

    # open data
    dloc = os.getenv('APPS')+'/prospector_alpha/data/hlsp_hlf_hst_60mas_goodss.v1.5.nzpcat'
    with open(dloc, 'r') as f:
        hdr = f.readline().split()[1:]
    dtype = np.dtype([(n, np.float) for n in hdr])
    dat = np.loadtxt(dloc,dtype=dtype)
    
    ### extract filters, fluxes, errors for object
    obj_idx = (dat['id'] == objname)
    filters = np.array([f[2:] for f in dat.dtype.names if f[0:2] == 'f_'])
    flux = np.squeeze([dat[obj_idx]['f_'+f] for f in filters])
    unc = np.squeeze([dat[obj_idx]['e_'+f] for f in filters])

    # normalized to zeropoint of ABmag=25, convert to maggies
    maggies = flux / 1e10
    maggies_unc = unc / 1e10

    ### define photometric mask, clip
    phot_mask = flux != -99
    maggies_unc = np.clip(maggies_unc, maggies*err_floor, np.inf)

    # need filter objects
    ofilters = observate.load_filters([trans_phot[f.lower()] for f in filters])

    # remove very blue filters
    zred = load_z(objname)
    wavemin = np.array([f.wavelength[f.transmission > (f.transmission.max()*0.1)].min() for f in ofilters]) / (1+zred)
    idx = wavemin < 1250
    phot_mask[idx] = 0

    ### build output dictionary
    obs = {}
    obs['filters'] = ofilters
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

def logmass_to_masses(logmass=None, logsfr_ratios=None, agebins=None, **extras):
    logsfr_ratios = np.clip(logsfr_ratios,-100,100) # numerical issues...
    nbins = agebins.shape[0]
    sratios = 10**logsfr_ratios
    dt = (10**agebins[:,1]-10**agebins[:,0])
    coeffs = np.array([ (1./np.prod(sratios[:i])) * (np.prod(dt[1:i+1]) / np.prod(dt[:i])) for i in range(nbins)])
    m1 = (10**logmass) / coeffs.sum()

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

model_params.append({'name': 'logmass', 'N': 1,
                        'isfree': True,
                        'init': 10.0,
                        'units': 'Msun',
                        'prior': priors.TopHat(mini=7, maxi=13)})

model_params.append({'name': 'logzsol', 'N': 1,
                        'isfree': False,
                        'init':0.16,
                        'units': r'$\log (Z/Z_\odot)$',
                        'prior': priors.TopHat(mini=-2, maxi=0.19)})
                        
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
fit_order = ['logmass','logsfr_ratios', 'dust2', 'dust_index', 'dust1_fraction', 'gas_logz']
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

def load_z(objname):

    dat = np.array([[36777,53.21632767,-27.90732384,5.653],
                    [57174,53.26964951,-27.86892319,5.599],
                    [69908,53.26034927,-27.84705353,5.617],
                    [73289,53.22440338,-27.84117508,5.685],
                    [76225,53.27734375,-27.83659172,5.455],
                    [90806,53.17047882,-27.81241989,5.785],
                    [96204,53.16573715,-27.80338097,5.556],
                    [96736,53.13042450,-27.80231285,5.729],
                    [100974,53.13927841,-27.79579544,5.953],
                    [101638,53.14606094,-27.79448700,6.185],
                    [104667,53.13898468,-27.79031754,5.464],
                    [104831,53.16219711,-27.78996468,4.921],
                    [107797,53.17784882,-27.78535271,5.917],
                    [110501,53.20392609,-27.78096199,5.504],
                    [115171,53.15441513,-27.77325630,4.367],
                    [117628,53.16769409,-27.76810646,5.840],
                    [118957,53.15404510,-27.76599884,6.054],
                    [64429, -1, -1, 6.172],
                    [89007, -1, -1, 6.844],
                    [72870, -1, -1, 6.316],
                    [88419, -1, -1, 6.244],
                    [95245, -1, -1, 7.927],
                    [109421, -1, -1, 6.844]])
    idx = dat[:,0] == objname
    return dat[idx,3]

def load_model(objname=None, datdir=None, runname=None, agelims=[], zred=0.0, nbins_sfh=7, sigma=0.3,df=2, **extras):

    # we'll need this to access specific model parameters
    n = [p['name'] for p in model_params]

    # redshift
    zred = load_z(objname)
    tuniv = WMAP9.age(zred).value

    # now construct the nonparametric SFH
    # current scheme: six bins, four spaced equally in logarithmic 
    # last bin is 15% age of the Universe, first two are 0-30, 30-100
    tbinmax = (tuniv*0.85)*1e9
    agelims = agelims[:2] + np.linspace(agelims[2],np.log10(tbinmax),len(agelims)-3).tolist() + [np.log10(tuniv*1e9)]
    agebins = np.array([agelims[:-1], agelims[1:]])
    ncomp = len(agelims) - 1

    # load into `agebins` in the model_params dictionary
    model_params[n.index('agebins')]['N'] = ncomp
    model_params[n.index('agebins')]['init'] = agebins.T

    # load nvariables and agebins
    model_params[n.index('agebins')]['N'] = ncomp
    model_params[n.index('agebins')]['init'] = agebins.T
    model_params[n.index('mass')]['N'] = ncomp
    model_params[n.index('logsfr_ratios')]['N'] = nbins_sfh-1
    model_params[n.index('logsfr_ratios')]['init'] = np.full(nbins_sfh-1,0.0) # constant SFH
    model_params[n.index('logsfr_ratios')]['prior'] = priors.StudentT(mean=np.full(nbins_sfh-1,0.0),
                                                                      scale=np.full(nbins_sfh-1,sigma),
                                                                      df=np.full(nbins_sfh-1,df))
    # insert redshift into model dictionary
    model_params[n.index('zred')]['init'] = zred

    return sedmodel.SedModel(model_params)
