import numpy as np
from scipy.special import gamma, gammainc
import threed_dutils

def int_sfh_2(t1, t2, tage, tau, sf_start, tburst=0, fburst=0):
    """Use Gamma functions
    """
    normalized_times = (np.array([t1, t2, tage]) - sf_start) / tau
    mass = gammainc(2, normalized_times)
    intsfr = (mass[1] - mass[0]) / mass[2]
    return intsfr

if __name__ == "__main__":

    tage = np.array([13.0,13.0])
    tau  = np.array([1.0,1.0])

    delta_t = np.atleast_1d(0.1)
    ntest = 100
    sf_start = np.tile(np.linspace(0.0,tage[0]-delta_t, ntest), (2,1))

    #instantaneous SFR
    sfr_instant = (tage[0]-sf_start[0,:]) * np.exp(-(tage[0]-sf_start[0,:])/tau[0])*2 / 1e9
    
    #normalize by total mass formed since sf_start
    sfr_instant /= tau[0]**2 * gammainc(2, (tage[0]-sf_start[0,:])/tau[0]) * gamma(2, (tage[0]-sf_start[0,:])/tau[0])

    mf1 = np.zeros(ntest)
    mf2 = np.zeros(ntest)
    
    # calculate sfr from SFH integration
    for i in xrange(ntest):
        mf1[i] = threed_dutils.integrate_sfh(tage-delta_t, 
                                             tage, 
                                             np.array([1.0,1.0]), 
                                             tage, 
                                             tau, 
                                             sf_start[:,i])/(1e9*delta_t)
        mf2[i] = threed_dutils.integrate_sfh_old(tage-delta_t, 
                                                 tage, 
                                                 np.array([1.0,1.0]), 
                                                 tage, 
                                                 tau, 
                                                 sf_start[:,i])/(1e9*delta_t)

    print mf1-mf2
    print mf1-sfr_instant