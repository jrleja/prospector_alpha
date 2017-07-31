import numpy as np
from astropy import constants
from bsfh import read_results
import fsps, os
import prosp_dutils

def integrate_sfh(t1,t2,tage,tau,sf_start,tburst,fburst):
    
    '''
    integrate a delayed tau SFH between t1 and t2
    '''
    
    # if we're outside of the boundaries, return boundary values
    if t2 < sf_start:
        return 0.0
    elif t2 > tage:
        return 1.0

    intsfr = (np.exp(-t1/tau)*(1+t1/tau) - 
              np.exp(-t2/tau)*(1+t2/tau))
    norm=(1.0-fburst)/(np.exp(-sf_start/tau)*(sf_start/tau+1)-
              np.exp(-tage    /tau)*(tage    /tau+1))
    intsfr=intsfr*norm
    
    # if burst occured in this time
    if (t2 > tburst) & (t1 < tburst):
        intsfr=intsfr+fburst

    return intsfr

sps = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=0)
w = sps.wavelengths

sps.params['sfh'] = 4 #composite
sps.params['imf_type'] = 0 # Salpeter
sps.params['tage'] = 1.0
sps.params['tau'] = 0.5
sps.params['smooth_velocity'] = True

k98 = 7.9e-42 #(M_sun/yr) / (erg/s)

if len(sps.zlegend) > 10:
    # We are using BaSeL with it's poorly resolved grid,
    # so we need to broaden the lines to see them
    sps.params['sigma_smooth'] = 1e3
    halpha = (w > 6500) & (w < 6650)
    sps.params['zmet'] = 22
    sps.params['logzsol'] = 0.0
else:
    # we are using MILES with better optical resolution
    sps.params['sigma_smooth'] = 0.0
    halpha = (w > 6556) & (w < 6573)
    sps.params['zmet'] = 4
    sps.params['logzsol'] = 0.0

ns = 40
Cha = np.zeros(ns)
gtage = np.random.uniform(7.0, 14.0, ns)
gtau = np.random.uniform(0.1, 10.0, ns)

mcmc_filename = os.getenv('APPS')+'/prospector_alpha/results/dtau_intmet/dtau_intmet_0181_15268_1424608305_mcmc'
model_filename = os.getenv('APPS')+'/prospector_alpha/results/dtau_intmet/dtau_intmet_0181_15268_1424608305_model'
sample_results, powell_results, model = read_results.read_pickles(mcmc_filename, model_file=model_filename,inmod=None)

# set redshift
sample_results['model'].params['zred'] = np.atleast_1d(0.00)

#params = sample_results['model'].params
#print 1/0
#for kk,nn in params.iteritems():
#    if kk in sps.params.all_params:
#        if kk == 'zmet':
#            vv = np.abs(nn - (np.arange( len(sps.zlegend))+1)).argmin()+1
#        else:
#            vv = nn.copy()
#        sps.params[kk] = vv

#get the spectrum with neb for a variety of neb parameters, compute line luminosity
deltat=0.1 # in Gyr

# remove filters
sample_results['obs']['filters'] = ['U']

flatchain = prosp_dutils.chop_chain(sample_results['chain'])
np.random.shuffle(flatchain)
for jj in xrange(ns):
    
    thetas = flatchain[jj,:]

    # get neboff
    sample_results['model'].params['add_neb_emission'] = np.atleast_1d(False)
    sample_results['model'].params['add_neb_continuum'] = np.atleast_1d(False)
    spec,mags_neboff,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)

    # get nebon
    sample_results['model'].params['add_neb_emission'] = np.atleast_1d(True)
    sample_results['model'].params['add_neb_continuum'] = np.atleast_1d(True)
    nebspec,mags_neboff,w = sample_results['model'].mean_model(thetas, sample_results['obs'], sps=sps)

    sfr_100 = integrate_sfh(sps.params['tage']-deltat,
                            sps.params['tage'],
                            sps.params['tage'],
                            np.array(thetas[3:5]),
                            np.array(thetas[5:7]),
                            sps.params['tburst'],
                            sps.params['fburst'])/(deltat*1e9)
    print 1/0
    ha_lum = constants.L_sun.cgs.value * np.trapz((nebspec - spec)[halpha], w[halpha])
    print ha_lum,sfr_100
    Cha[jj] = ha_lum/sfr_100
    
print(Cha.mean(), Cha.std()/Cha.mean(), Cha.mean()*k98 )
print Cha
