import numpy as np
import td_new_params as pfile
import matplotlib.pyplot as plt
from mass_function_model import sfr_ms

def draw_ssfr_from_prior(ndraw=2e5, alpha_sfh=0.2):

    # simulation information
    ndraw = int(ndraw)
    zred = np.array([1.5])
    logmass = np.array([10.])
    smass_factor = 0.8 # add in a not-unreasonable stellar mass <--> total mass conversion
    minssfr, maxssfr = 1e-14, 1e-7

    # figure information
    fig, ax = plt.subplots(1,1, figsize=(3.4, 3.4))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    fs = 10
    colors = ['red', 'black']
    labels = ['new SFH prior', 'old SFH prior']
    
    for i, alpha_sfh in enumerate([0.2,1.0]):    

        # new redshift, new model
        model = pfile.load_model(zred=zred, alpha_sfh=alpha_sfh, **pfile.run_params)
        prior = model._config_dict['z_fraction']['prior']
        agebins = model.params['agebins']

        # draw from the prior
        mass = np.zeros(shape=(agebins.shape[0],ndraw))
        for n in range(ndraw): mass[:,n] = pfile.zfrac_to_masses(logmass=logmass, z_fraction=prior.sample(), agebins=agebins)

        # convert to sSFR
        time_per_bin = np.diff(10**agebins, axis=-1)[:,0]
        ssfr = np.log10(np.clip(mass[0,:] / time_per_bin[0] / 10**logmass,minssfr/10,maxssfr))

        # histogram
        ax.hist(ssfr,bins=30,histtype='step',color=colors[i],normed=True,label=labels[i],
                lw=1.5,range=(np.log10(minssfr),np.log10(maxssfr)))
    
    ax.set_xlim(-14,-7)
    ax.set_ylim(0,1)
    ax.set_xlabel('instantaneous log(sSFR/yr)')
    ax.set_ylabel('normalized probability')
    ax.legend(loc=2, prop={'size':10}, scatterpoints=1,fancybox=True)

    # add sSFR(main sequence)
    plt.tight_layout()
    plt.savefig('/Users/joel/my_papers/td_sfrd/figures/sfh_prior.png',dpi=190)    
    plt.show()
    print 1/0