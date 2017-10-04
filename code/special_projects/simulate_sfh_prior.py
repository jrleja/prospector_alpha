import numpy as np
import td_params as pfile
import matplotlib.pyplot as plt
from mass_function_model import sfr_ms

def draw_ssfr_from_prior(ndraw=1e4, alpha_sfh=0.2):

    # let's do it
    ndraw = int(ndraw)
    zred = np.array([0.0, 0.5, 1.5, 2.5]) # where do we measure?
    logmass = np.array([10.])
    smass_factor = 0.8 # add in a not-unreasonable stellar mass <--> total mass conversion
    minssfr, maxssfr = 1e-14, 1e-7

    # figure stuff
    fig, axes = plt.subplots(2,2, figsize=(7, 7))
    fig.subplots_adjust(hspace=0.0, wspace=0.0)
    axes = np.ravel(axes)
    fs = 10

    for i, z in enumerate(zred):
        
        # new redshift, new model
        model = pfile.load_model(zred=z, alpha_sfh=alpha_sfh, **pfile.run_params)
        prior = model._config_dict['z_fraction']['prior']
        agebins = model.params['agebins']

        # draw from the prior
        mass = np.zeros(shape=(agebins.shape[0],ndraw))
        for n in range(ndraw): mass[:,n] = pfile.zfrac_to_masses(logmass=logmass, z_fraction=prior.sample(), agebins=agebins)

        # convert to sSFR
        time_per_bin = np.diff(10**agebins, axis=-1)[:,0]
        ssfr = np.log10(np.clip(mass[0:2,:].sum(axis=0) / time_per_bin[0:2].sum() / 10**logmass,minssfr,maxssfr))

        # histogram
        axes[i].hist(ssfr,bins=50,histtype="step",color='k',normed=True)
        axes[i].set_xlim(-14,-7)
        axes[i].set_ylim(0,1)

        if i > 1:
            axes[i].set_xlabel('sSFR (100 Myr)')
        else:
            axes[i].set_xticks([])

        if (i % 2) == 1:
            axes[i].set_yticks([])

        axes[i].text(0.02,0.94,'z = '+"{:.1f}".format(z), transform=axes[i].transAxes, fontsize=fs)
        axes[i].text(0.02,0.88,'<SFR>='+"{:.1f}".format(ssfr.mean()), transform=axes[i].transAxes, fontsize=fs)

        # add sSFR(main sequence)
        try:
            ssfr_ms = sfr_ms(z,logmass) / (10**logmass * smass_factor)
            axes[i].axvline(np.log10(ssfr_ms),linestyle='--', color='red',lw=1.5,alpha=0.8,zorder=5)
            prob = (ssfr > np.log10(ssfr_ms)).sum() / float(ssfr.shape[0])
            axes[i].text(0.02,0.82,'ln(p(SFR>=MS))='+"{:.2f}".format(np.log(prob)), transform=axes[i].transAxes, fontsize=fs)
        except ZeroDivisionError:
            pass
    plt.show()
    print 1/0