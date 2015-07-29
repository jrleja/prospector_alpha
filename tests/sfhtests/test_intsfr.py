import numpy as np
import matplotlib.pyplot as plt
from bsfh import model_setup
import threed_dutils

if __name__ == "__main__":

    # load model,sps 
    #model = model_setup.load_model('/Users/joel/code/python/threedhst_bsfh/parameter_files/testsed_simha/testsed_simha_params.py')
    model = model_setup.load_model('/Users/joel/code/python/threedhst_bsfh/parameter_files/testsed_simha/testsed_simha_params_2.py')
    sps   = threed_dutils.setup_sps(zcontinuous=2,compute_vega_magnitudes=False)

    # custom-set parameters
    initial_theta_true = np.array([10**11.37,-1.35,-0.31,5.93,2.93,1.07,3.74,-0.4])
    model.initial_theta = initial_theta_true
    model.set_parameters(model.initial_theta)

    # set up arrays
    #sfh_params = threed_dutils.find_sfh_params(model,model.initial_theta)
    #tcalc = np.linspace(0,5.93,100)
    #mf1 = np.zeros(len(tcalc))
    
    ###### calculate FSPS quantities ######
    # pass parameters to sps object
    #sm = np.empty(0)
    #for tt in tcalc:
    model.params['tage'] = np.atleast_1d(0)
    for k, v in model.params.iteritems():
        if k in sps.params.all_params:
            if k == 'zmet':
                vv = np.abs(v - (np.arange( len(sps.zlegend))+1)).argmin()+1
            elif k == 'dust1':
                # temporary! replace with smarter function soon
                vv = model.params['dust2']*1.86+0.0
            elif k == 'tau':
                vv = 10**v
            else:
                vv = v.copy()
            sps.params[k] = vv
        if k == 'mass':
            mass = v
    sfr_true = sps.sfr
    lage_true = sps.log_age

    # now the best-fit
    initial_theta_fit = np.array([10**11.34,-1.38,0.43,7.95,7.45,0.92,3.71,-0.4])
    model.initial_theta = initial_theta_fit
    model.set_parameters(model.initial_theta)

    model.params['tage'] = np.atleast_1d(0)
    for k, v in model.params.iteritems():
        if k in sps.params.all_params:
            if k == 'zmet':
                vv = np.abs(v - (np.arange( len(sps.zlegend))+1)).argmin()+1
            elif k == 'dust1':
                # temporary! replace with smarter function soon
                vv = model.params['dust2']*1.86+0.0
            elif k == 'tau':
                vv = 10**v
            else:
                vv = v.copy()
            sps.params[k] = vv
        if k == 'mass':
            mass = v
    sfr_fit = sps.sfr
    lage_fit = sps.log_age


    fig, ax = plt.subplots()
    ax.plot(10**lage_true, sfr_true, 'bo', alpha=0.3)
    ax.plot(10**lage_fit, sfr_fit, 'ro', alpha=0.3)
    print 1/0

    y_offset = 0.00
    for k,v in sfh_params.iteritems():
        if k != 'tage':
            ax.text(0.95,0.9-y_offset,k+'='+"{:.2f}".format(v[0]), horizontalalignment='right',transform = ax.transAxes)
            y_offset+=0.05


    ax.set_ylabel('sps.sfr')
    ax.set_xlabel('tage')
    ax.set_ylim(-0.1,5)
    fig.show()
    print 1/0
    plt.savefig('sfh=5_test1.png',dpi=300)

def test_sfh5():

    # load model,sps 
    model = model_setup.load_model('/Users/joel/code/python/threedhst_bsfh/parameter_files/testsed_simha/testsed_simha_params.py')
    #model = model_setup.load_model('/Users/joel/code/python/threedhst_bsfh/parameter_files/testsed_nonoise_fastgen/testsed_nonoise_fastgen_params.py')
    sps   = threed_dutils.setup_sps(zcontinuous=2,compute_vega_magnitudes=False)

    # custom-set parameters
    #initial_theta = np.array([10**9.95,0.18,10**1.40,14.0-7.17,13.94,-4.67,1.84,-0.42])
    initial_theta = np.array([10**10.68,-0.81,10**-0.25,14.0-4.83,12.89,-3.08,2.17,-0.40])
    model.initial_theta = initial_theta
    sf_start = 14.0-7.17
    model.initial_theta[np.array(model.theta_labels()) == 'sf_start'] = sf_start
    model.set_parameters(model.initial_theta)

    # set up arrays
    tcalc = np.linspace(sf_start,14.0, 100)
    mf1 = np.zeros(len(tcalc))
    
    ###### calculate FSPS quantities ######
    # pass parameters to sps object
    model.params['tage'] = np.array(0.0)
    for k, v in model.params.iteritems():
        if k in sps.params.all_params:
            if k == 'zmet':
                vv = np.abs(v - (np.arange( len(sps.zlegend))+1)).argmin()+1
            elif k == 'dust1':
                # temporary! replace with smarter function soon
                vv = model.params['dust2']*1.86+0.0
            else:
                vv = v.copy()
            sps.params[k] = vv
        if k == 'mass':
            mass = v

    # FSPS sfr
    tcalc_fsps = 10**sps.log_age/1e9
    sfr_fsps = sps.sfr*mass

    ###### calculate handwritten quantities ######
    model.params['tage'] = np.array(tcalc_fsps[-1])
    sfh_params = threed_dutils.find_sfh_params(model,model.initial_theta)
    delta_t = 0.01
    for i,tt in enumerate(tcalc):
        
        # use my SFH integral, averaged over 1 Myr
        mf1[i] = threed_dutils.calculate_sfr(sfh_params, delta_t, minsfr=0.0, maxsfr=None,tcalc=tt)

    fig, ax = plt.subplots()
    ax.plot(tcalc, mf1, 'o', label='Joel', alpha=0.3)
    ax.plot(tcalc_fsps, sfr_fsps, 'o', label='FSPS', alpha=0.3)

    ax.set_yscale('log')
    ax.set_ylabel('SFR')
    ax.set_xlabel('time')
    ax.legend(loc=0)
    fig.show()
    print 1/0

