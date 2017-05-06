import numpy as np
import matplotlib.pyplot as plt
import magphys_plot_pref


from brownseds_agn_params import load_obs, run_params

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)


def plot():

    fig, ax = plt.subplots(1,1, figsize=(14, 6))
    obs = load_obs(**run_params)

    for f in obs['filters']:

        ax.plot(f.wavelength/1e4,f.transmission/f.transmission.max(),lw=2,color='0.3',alpha=0.8)

        ax.text(f.wave_effective/1e4, 1.05, f.name, weight='semibold',fontsize=8)

    ax.set_xscale('log',nonposx='clip',subsx=(1,3))
    ax.xaxis.set_minor_formatter(minorFormatter)
    ax.xaxis.set_major_formatter(majorFormatter)

    ax.set_ylim(0,3)
    ax.set_xlim(1,25)

    ax.set_ylabel('Transmission')
    ax.set_xlabel(r'wavelength ($\mu$m)')

    print 1/0
