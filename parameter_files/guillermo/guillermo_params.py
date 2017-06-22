import numpy as np
import os
from prospect.models import priors, sedmodel
from prospect.sources import FastStepBasis
from sedpy import observate
from astropy.cosmology import WMAP9
from td_io import load_zp_offsets

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
              'outfile': APPS+'/threedhst_bsfh/results/guillermo/guillermo',
              'nofork': True,
              # Optimizer params
              'ftol':0.5e-5, 
              'maxfev':5000,
              # MCMC params
              'nwalkers':434,
              'nburn':[150,200,400], 
              'niter': 10000,
              'interval': 0.2,
              # Convergence parameters
              'convergence_check_interval': 50,
              'convergence_chunks': 650,
              'convergence_kl_threshold': 0.0175,
              'convergence_stable_points_criteria': 8, 
              'convergence_nhist': 50,
              # Model info
              'zcontinuous': 2,
              'compute_vega_mags': False,
              'initial_disp':0.1,
              'interp_type': 'logarithmic',
              'agelims': [0.0,8.0,8.5,9.0,9.5,9.8],
              # Data info (phot = .cat, dat = .dat, fast = .fout)
              'cat':APPS+'/threedhst_bsfh/data/gds14876_allphotometry.cat',
              'objinfo': APPS+'/threedhst_bsfh/data/allfilters.dat'
              }

############
# OBS
#############
fname_map = {'VIMOS_U': 'bessell_U', 
             'U_ctio': 'bessell_U',
             'ACS_bccd':'ACS_f435W',
             'ACS_vccd':'ACS_f606W',
             'ACS_iccd':'ACS_f775W',
             'ACS_f814w':'ACS_f814W',
             'ACS_zccd':'ACS_f850lp',
             'Subaru_IAL427ccd':'ia427_goodss',
             'Subaru_IAL445ccd':'ia445_goodss',
             'Subaru_IAL464ccd':'ia464_cosmos',
             'Subaru_IAL484ccd':'ia484_cosmos',
             'Subaru_IAL505ccd':'ia505_cosmos',
             'Subaru_IAL527ccd':'ia527_cosmos',
             'Subaru_IAL550ccd':'ia550_goodss',
             'Subaru_IAL574ccd':'ia574_cosmos',
             'Subaru_IAL598ccd':'ia598_goodss',
             'Subaru_IAL624ccd':'ia624_cosmos',
             'Subaru_IAL651ccd':'ia651_goodss',
             'Subaru_IAL679ccd':'ia679_cosmos',
             'Subaru_IAL709ccd':'ia709_cosmos',
             'Subaru_IAL738ccd':'ia738_goodss',
             'Subaru_IAL767ccd':'ia767_goodss',
             'Subaru_IAL797ccd':'ia797_goodss',
             'Subaru_IAL827ccd':'ia827_cosmos',
             'Subaru_IAL856ccd':'ia856_goodss',
             'WFC3_F098M':'wfc3_ir_f098m',
             'WFC3_F105W':'WFC3_f105W',
             'WFC3_F125W':'f125w_cosmos',
             'WFC3_F140W':'f140w_cosmos',
             'WFC3_F160W':'f160w_goodsn',
             'ISAAC_K':'isaac_k',
             'HawkI_K':'hawki_k',
             'IRAC_36':'spitzer_irac_ch1',
             'IRAC_45':'spitzer_irac_ch2',
             'IRAC_58':'spitzer_irac_ch3',
             'IRAC_80':'spitzer_irac_ch4',
             'MIPS24':'spitzer_mips_24', 
             'MIPS70':'spitzer_mips_70',  ### DPE?
             'PACS_100':'herschel_pacs_100',
             'PACS_160':'herschel_pacs_160', 
             'SPIRE_250':'herschel_spire_250', ## ext? 
             'SPIRE_350':'herschel_spire_350', ## ext?
             'SPIRE_500':'herschel_spire_500', ## ext?
             'PACS_70':'herschel_pacs_70', 
             'PACS_100':'herschel_pacs_100', 
             'PACS_160':'herschel_pacs_160',
             'ESOWFI_Iccd': 'bessell_U' ## use U band b/c it is masked below anyway
             }

def load_obs(cat=None, objinfo=None,**extras):
    
    ### load data and filter file
    dat = np.genfromtxt(cat)
    dtype = np.dtype([('filter', 'S20'),
                      ('fluxcol', np.int), ('errcol', np.int),
                      ('flag', np.int)])
    info = np.genfromtxt(objinfo, dtype=dtype)
    info = info[2:] # drop the header + redshift

    ### extract fluxes, filternames
    use = np.ones_like(info['flag'],dtype=bool)
    mjy = dat[info[use]['fluxcol']-1]
    mjy_unc = dat[info[use]['errcol']-1]
    filters = info[use]['filter']
    filternames = [fname_map[f] for f in filters]

    ### add ALMA filters
    mjy = np.concatenate((mjy, np.array([5.43e03, 3.1e+03])))
    mjy_unc = np.concatenate((mjy_unc, np.array([0.19e+03, 0.1e+03])))
    filternames = filternames + ['ALMAshort', 'ALMAlong']

    ### build obs dictionary
    obs = {}
    obs['redshift'] =  2.3091
    obs['maggies'] = mjy / 1e6/3631.
    obs['maggies_unc'] = np.clip(mjy_unc / 1e6/3631., obs['maggies']*0.05, np.inf) # implement 5% error floor
    obs['filters'] = observate.load_filters(filternames)
    obs['wave_effective'] = np.array([f.wave_effective for f in obs['filters']])
    obs['phot_mask'] = np.ones(len(mjy), dtype=bool)
    obs['phot_mask'] = obs['phot_mask'] & \
                       ((obs['wave_effective'] / (obs['redshift'] + 1)) > 1300) & \
                       (obs['maggies'] > 0)
    obs['spectrum'] = None
    obs['wavelength'] = None
    obs['unc'] = None

    ### correct MIPS magnitudes due to weird MIPS conventions
    '''
    fnames = [f.name for f in obs['filters']]
    mips_corr = np.array([-0.03542,-0.07669,-0.03807]) # 24, 70, 160
    mips_names = ['spitzer_mips_24','spitzer_mips_70']
    for i,name in enumerate(mips_names):
      idx = fnames.index(name)
      mab = -(5./2)*np.log10(obs['maggies'][idx]) + mips_corr[i]
    # mag_adj[mag_fields.index('[24]') ] += mips_corr[0]
    '''
    '''
    # make simple SED plot for debugging purposes
    import matplotlib.pyplot as plt
    idx = obs['phot_mask']
    yerr = np.log10(obs['maggies'])[idx] - np.log10(np.array(obs['maggies'])[idx]-np.array(obs['maggies_unc'])[idx])
    lam = np.log10(obs['wave_effective']/1e4)[idx]
    nufnu = np.log10(obs['maggies']*3e18/obs['wave_effective'])[idx]
    plt.errorbar(lam,nufnu,yerr=yerr,linestyle=' ',fmt='o',ms=6)
    plt.xlabel(r'log(wavelength/$\mu$m)')
    plt.ylabel(r'log($\nu$ f$_{\nu}$)')
    plt.xlim(-1,2.2)
    plt.show()
    '''
    return obs

##########################

# TRANSFORMATION FUNCTIONS
##########################
def transform_logmass_to_mass(mass=None, logmass=None, **extras):
    return 10**logmass

def load_gp(**extras):
    return None, None

def tie_gas_logz(logzsol=None, **extras):
    return logzsol

def to_dust1(dust1_fraction=None, dust1=None, dust2=None, **extras):
    return dust1_fraction*dust2

def transform_zfraction_to_sfrfraction(sfr_fraction=None, z_fraction=None, **extras):
    """This transforms from latent, independent `z` variables to sfr
    fractions [see Leja+16 for definition of sfr_fractions]. 
    The transformation is such that sfr fractions are drawn from a
    Dirichlet prior (Betancourt et al. 2010)
    """

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
                        'init': 2.3091,
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
                        'prior': priors.TopHat(mini=0.0, maxi=6.0)})

model_params.append({'name': 'dust1_fraction', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.8,
                        'disp_floor': 0.8,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

model_params.append({'name': 'dust2', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 0.25,
                        'disp_floor': 0.15,
                        'units': '',
                        'prior': priors.TopHat(mini=0.0, maxi=4.0)})

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

model_params.append({'name': 'frac_obrun', 'N': 1,
                        'isfree': True,
                        'init': 0.1,
                        'units': 'fraction',
                        'prior': priors.TopHat(mini=0.0, maxi=1.0)})

###### Dust Emission ##############
model_params.append({'name': 'add_dust_emission', 'N': 1,
                        'isfree': False,
                        'init': 1,
                        'units': None,
                        'prior': None})

model_params.append({'name': 'duste_gamma', 'N': 1,
                        'isfree': True,
                        'init': 0.01,
                        'init_disp': 0.2,
                        'disp_floor': 0.15,
                        'units': None,
                        'prior': priors.TopHat(mini=0.0, maxi=1.0)})

model_params.append({'name': 'duste_umin', 'N': 1,
                        'isfree': True,
                        'init': 1.0,
                        'init_disp': 5.0,
                        'disp_floor': 4.5,
                        'units': None,
                        'prior': priors.TopHat(mini=0.1, maxi=25.0)})

model_params.append({'name': 'duste_qpah', 'N': 1,
                        'isfree': True,
                        'init': 2.0,
                        'init_disp': 3.0,
                        'disp_floor': 3.0,
                        'units': 'percent',
                        'prior': priors.TopHat(mini=0.0, maxi=10.0)})

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

model_params.append({'name': 'gas_logu', 'N': 1,
                        'isfree': False,
                        'init': -2.0,
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
                        'init': 0.00,
                        'init_disp': 0.03,
                        'disp_floor': 0.02,
                        'units': '',
                        'prior': priors.LogUniform(mini=1e-5, maxi=3.0)})

model_params.append({'name': 'agn_tau', 'N': 1,
                        'isfree': False,
                        'init': 4.0,
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
#### so that major ones are fit first
parnames = [m['name'] for m in model_params]
fit_order = ['logmass','z_fraction', 'dust2', 'logzsol', 'dust_index', 'dust1_fraction', 'duste_gamma', 'duste_qpah', 'duste_umin', 'frac_obrun']
tparams = [model_params[parnames.index(i)] for i in fit_order]
for param in model_params: 
    if param['name'] not in fit_order:
        tparams.append(param)
model_params = tparams

###### Redefine SPS ######
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

def load_model(objname=None, datname=None, agelims=[], **extras):
    
    ###### REDSHIFT ######
    n = [p['name'] for p in model_params]
    zred = model_params[n.index('zred')]['init']

    #### CALCULATE TUNIV #####
    tuniv = WMAP9.age(zred).value
    agelims[-1] = np.log10(tuniv*1e9)

    #### NONPARAMETRIC SFH #####
    # six bins, four spaced equally in logarithmic space AFTER t=100 Myr + BEFORE tuniv-1 Gyr
    if tuniv > 5:
        tbinmax = (tuniv-2)*1e9
    else:
        tbinmax = (tuniv-1)*1e9
    agelims = [agelims[0]] + np.linspace(agelims[1],np.log10(tbinmax),5).tolist() + [np.log10(tuniv*1e9)]
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

