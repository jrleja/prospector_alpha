import numpy as np
import matplotlib.pyplot as plt
from bsfh import model_setup
import fsps

def setup_sps(zcontinuous=2,compute_vega_magnitudes=False):

    '''
    easy way to define an SPS
    '''

    # load stellar population, set up custom filters
    sps = fsps.StellarPopulation(zcontinuous=zcontinuous, compute_vega_mags=compute_vega_magnitudes)
    #custom_filter_keys = os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
    #fsps.filters.FILTERS = model_setup.custom_filter_dict(custom_filter_keys)

    return sps

def find_sfh_params(model,theta):

    str_sfh_parms = ['sfh','mass','tau','sf_start','tage','sf_trunc','sf_slope']
    parnames = model.theta_labels()
    sfh_out = []

    for string in str_sfh_parms:
        
        # find SFH parameters that are variables in the chain
        index = np.char.find(parnames,string) > -1

        # if not found, look in fixed parameters
        if np.sum(index) == 0:
            sfh_out.append(np.atleast_1d(model.params.get(string,0.0)))
        else:
            sfh_out.append(theta[index])

    iterable = [(str_sfh_parms[ii],sfh_out[ii]) for ii in xrange(len(sfh_out))]
    out = {key: value for (key, value) in iterable}

    return out

if __name__ == "__main__":

    # load model, set initial theta
    # SFH = 5 test
    model = model_setup.load_model('testsed_simha_params.py')
    initial_theta = np.array([1e10,-0.81,10**-0.25,14.0-10.83,12.89,-3.08,2.17,-0.40])

    # SFH = 4 test
    # model = model_setup.load_model('testsed_nonoise_fastgen_params.py')
    # initial_theta = np.array([1e10,-0.81,10**-0.25,14.0-10.83,2.17,-0.40])

    # set up SPS
    sps   = setup_sps(zcontinuous=2,compute_vega_magnitudes=False)

    # make sure parameters are set
    model.initial_theta = initial_theta
    model.set_parameters(model.initial_theta)

    # set up arrays
    sfh_params = find_sfh_params(model,model.initial_theta)
    tcalc = np.linspace(sfh_params['sf_start'],14.0, 100)
    sm = np.empty(0)

    ###### calculate FSPS quantities ######
    # pass parameters to sps object
    for tt in tcalc:
        model.params['tage'] = tt
        for k, v in model.params.iteritems():
            if k in sps.params.all_params:
                if k == 'zmet':
                    vv = np.abs(v - (np.arange( len(sps.zlegend))+1)).argmin()+1
                else:
                    vv = v.copy()
                sps.params[k] = vv
            if k == 'mass':
                mass = v
        sm = np.append(sm,sps.stellar_mass)

    print sm
    fig, ax = plt.subplots()
    ax.plot(tcalc, sm, 'o', alpha=0.3)

    y_offset = 0.00
    for k,v in sfh_params.iteritems():
        if k != 'tage':
            ax.text(0.95,0.9-y_offset,k+'='+"{:.2f}".format(v[0]), horizontalalignment='right',transform = ax.transAxes)
            y_offset+=0.05


    ax.set_ylabel('sps.stellar_mass')
    ax.set_xlabel('tage')
    ax.set_ylim(-0.1,1)
    fig.show()
    plt.savefig('sfh=5_test1.png',dpi=300)
