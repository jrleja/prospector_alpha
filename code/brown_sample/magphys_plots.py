import numpy as np
import matplotlib.pyplot as plt
import os, copy, prosp_dutils
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import math, measure_emline_lum, prospector_io
import magphys_plot_pref
import mag_ensemble
import matplotlib as mpl
from prospect.models import model_setup
from astropy import constants
from allpar_plot import allpar_plot
import stack_sfh
import stack_irs_spectra
import time
import brownseds_highz_params as pfile

c = 3e18   # angstroms per second
dpi = 150

#### set up colors and plot style
prosp_color = '#1974D2'
median_err_color = '0.75'
obs_color = '#FF420E'
magphys_color = '#e60000'

class jLogFormatter(mpl.ticker.LogFormatter):
    '''
    this changes the format from exponential to floating point.
    '''

    def __call__(self, x, pos=None):
        """Return the format for tick val *x* at position *pos*"""
        vmin, vmax = self.axis.get_view_interval()
        d = abs(vmax - vmin)
        b = self._base
        if x == 0.0:
            return '0'
        sign = np.sign(x)
        # only label the decades
        fx = math.log(abs(x)) / math.log(b)
        isDecade = mpl.ticker.is_close_to_int(fx)
        if not isDecade and self.labelOnlyBase:
            s = ''
        elif x > 10000:
            s = '{0:.3g}'.format(x)
        elif x < 1:
            s = '{0:.3g}'.format(x)
        else:
            s = self.pprint_val(x, d)
        if sign == -1:
            s = '-%s' % s
        return self.fix_minus(s)

#### format those log plots! 
minorFormatter = jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = jLogFormatter(base=10, labelOnlyBase=True)

def read_magphys_sfh(idx,met):

  sfh_file = os.getenv('magphys')+'/OptiLIB_cb07_sfh.txt'

  age,sfr = [],[]
  with open(sfh_file, 'r') as f:
    found = False
    for line in f:
      if not found:
        if line.find('#') != -1:
          if int(line.split()[1]) == idx and \
             float(line.split()[2]) == met:
            found = True
        continue
      else:
        if line.find('#') == -1:
          _ = line.split()
          age.append(float(_[0]))
          sfr.append(float(_[1]))
        else:
          break

  if len(age) == 0:
    print "couldn't match metallicity and index! something is wrong."
    print 1/0

  return np.array(age),np.array(sfr)

def slice_name(objname):
  # retain up to 2 letters in front of name
  for i, c in enumerate(objname):
    if c.isdigit() and i <= 2:
      break
    elif i == 2:
      objname=objname[:i] + objname[i+1:]
  return objname

def read_magphys_output(objname=None):

  '''
  two output files

  GALAXY_ID.fit has observed fluxes+uncertainties,
  index of best-fit IR + optical models, chi^2, values of
  best-fit parameters, fluxes in each filter, and likelihood
  distribution of all parameters

  GALAXY_ID.sed has main parameters of the best-fit model,
  and the SED of the best-fit model.
  '''
  from astropy.cosmology import WMAP9
  project_info = {
                  'idfile': os.getenv('APPS')+"/threedhst_bsfh/data/brownseds_data/photometry/namelist.txt",
                  'datname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/table1.fits',
                  'photname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/table3.fits',
                  'extinctname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/table4.fits',
                  'herschname':os.getenv('APPS')+'/threedhst_bsfh/data/brownseds_data/photometry/kingfish.brownapertures.flux.fits',
                  'output': '/Users/joel/code/magphys/output'
                 }

  if objname is None:
    objname = np.loadtxt(project_info['idfile'],delimiter='#',dtype = str)[0]

  #### first file ####
  file1 = slice_name(objname.replace(' ',''))[:8]+'.fit'
  with open(project_info['output']+'/'+file1[:10], 'r') as f:

    # skip first line
    f.readline()

    ##### observations #####
    # flux in Lsun/Hz
    obs = {}
    obs['filters'] = np.array(f.readline().split()[1:],dtype=str)
    obs['flux'] = np.array(f.readline().split(),dtype=float)
    obs['flux_unc'] = np.array(f.readline().split(),dtype=float)
    obs['phot_mask'] = (obs['flux'] != 0) | (obs['flux_unc'] != 0)

    # skip four lines
    for kk in xrange(4): f.readline()

    ##### metadata #####
    metadata = {}
    line = np.array(f.readline().split(),dtype=float)
    metadata['bestfit_opt_ind'] = line[0]
    metadata['bestfit_ir_ind'] = line[1]
    metadata['chisq'] = line[2]
    metadata['redshift'] = line[3]

    ##### best-fit model #####
    # in Lsun/Hz
    model = {}
    model['parnames'] = np.array(f.readline().replace('.',' ').split()[1:],dtype=str)
    model['parameters'] = np.array(f.readline().split(),dtype=float)
    f.readline()
    model['flux'] = np.array(f.readline().split(),dtype=float)

    # skip three lines
    for kk in xrange(3): f.readline()

    ##### parameter PDFs #####
    pdfs = {'x_interp_perc': np.array([0.025,0.16,0.5,0.84,0.975]),
            'percentiles': {},
            'likelihood_distr': {},
            'likelihood_x': {}
            }
    for par in model['parnames']:

      for line in f:
        # ignore comments
        # if it's the last comment, switch to new parname
        if line.find('#') != -1 and line.find('....') != -1:
          continue
        elif line.find('#') != -1:
          break

        _ = np.array(line.split(),dtype=float)

        if _.shape[0] == 2:
          pdfs['likelihood_x'][par] = np.append(pdfs['likelihood_x'].get(par,[]),_[0])
          pdfs['likelihood_distr'][par] = np.append(pdfs['likelihood_distr'].get(par,[]),_[1])
        else:
          pdfs['percentiles'][par] = _
      
  ##### second file #####
  file2 = slice_name(objname.replace(' ',''))[:8]+'.sed'
  with open(project_info['output']+'/'+file2[:10], 'r') as f:

    # skip two lines
    for i in xrange(2): f.readline()

    ##### read model parameters
    model['full_parnames'] = np.array(f.readline().replace('.',' ').split(),dtype=str)[1:]
    model['full_parameters'] = np.array(f.readline().split(),dtype=float)
    f.readline()
    model['full_parnames'] = np.append(model['full_parnames'],np.array(f.readline().replace('.',' ').split(),dtype=str)[1:])
    model['full_parameters'] = np.append(model['full_parameters'],np.array(f.readline().split(),dtype=float))

    # skip three lines
    for i in xrange(3): f.readline()

    ##### read spectrum ##### 
    # starts in obslam
    for line in f:
      _ = np.array(line.split(),dtype=float)
      model['lam'] = np.append(model.get('lam',[]),10**_[0])#*(1+metadata['redshift']))
      model['spec'] = np.append(model.get('spec',[]),_[1])
      model['spec_nodust'] = np.append(model.get('spec_nodust',[]),_[2])

  ##### units of [log(Lsun/angstrom)], change to maggies
  # first save output as speclum
  model['speclum'] = 10**model['spec']
  model['speclum_nodust'] = 10**model['spec_nodust']

  # now convert to Lsun/Hz
  c = 3e18 # Angstroms / s
  model['spec'] = (10**model['spec']) * model['lam']**2 / c
  model['spec_nodust'] = (10**model['spec_nodust']) * model['lam']**2 / c

  # from Lsun/Hz to maggies
  lsun = 3.846e33  # ergs/s
  jansky_cgs = 1e-23
  to_maggies = lsun / (3631*jansky_cgs)

  # cm^2
  pc = 3.085677581467192e18  # cm
  dfactor = 4*np.pi*(pc*WMAP9.luminosity_distance(metadata['redshift']).value *
                    1e6)**2 / (1+metadata['redshift'])

  model['spec'] *= to_maggies / dfactor
  model['spec_nodust'] *= to_maggies / dfactor
  obs['flux'] *= to_maggies / dfactor
  obs['flux_unc'] *= to_maggies / dfactor
  model['flux']  *= to_maggies / dfactor

  ##### load SFH
  metind = model['full_parnames'] == 'Z/Zo'
  age,sfr = read_magphys_sfh(int(metadata['bestfit_opt_ind']),
                           float(model['full_parameters'][metind]))
  sfh = {'age':age,'sfr':sfr}

  ##### add SFR_10 from best-fitting SFH
  # in Msun/yr
  magmass = model['full_parameters'][[model['full_parnames'] == 'M*/Msun']]
  magsfr = sfh['sfr']*magmass
  magtime = np.abs(np.max(sfh['age']) - sfh['age'])

  sfr_10 = prosp_dutils.integral_average(magtime,magsfr,0,1e7)
  if np.isfinite(sfr_10) == False:
    print 1/0
  model['full_parameters'] = np.append(model['full_parameters'],sfr_10)
  model['full_parnames'] = np.append(model['full_parnames'],'SFR_10')

  magphys = {'model': model,
             'obs': obs, 
             'metadata': metadata,
             'pdfs': pdfs,
             'sfh': sfh}

  return magphys

def median_by_band(x,y,avg=False):

    ##### get filter effective wavelengths for sorting
    delz = 0.15
    from brownseds_np_params import translate_filters
    from sedpy import observate
    filtnames = np.array(translate_filters(0,full_list=True))
    filts = observate.load_filters(filtnames[filtnames != 'nan'])
    wave_effective = np.array([filt.wave_effective for filt in filts])/1e4
    wave_effective.sort()

    avglam = np.array([])
    outval = np.array([])
    for lam in wave_effective:
        in_bounds = (x <= lam) & (x > lam/(1+delz))
        avglam = np.append(avglam, np.mean(x[in_bounds]))
        if avg == False:
            outval = np.append(outval, np.median(y[in_bounds]))
        else:
            outval = np.append(outval, np.mean(y[in_bounds]))

    return avglam, outval

def plot_all_residuals(alldata,runname):

    '''
    show all residuals for spectra + photometry, magphys + prospector
    split into star-forming and quiescent
    by an arbitrary sSFR cut
    '''
    qucolor = '#FF3D0D' #red
    sfcolor = '#1C86EE' #blue
    ssfr_limit = 1e-11


    ##### set up plots
    fig = plt.figure(figsize=(15,12.5))
    #mpl.rcParams.update({'font.size': 13})
    gs1 = mpl.gridspec.GridSpec(4, 1)
    gs1.update(top=0.95, bottom=0.05, left=0.09, right=0.75,hspace=0.3)
    phot = plt.Subplot(fig, gs1[0])
    opt = plt.Subplot(fig, gs1[1])
    akar = plt.Subplot(fig, gs1[2])
    spit = plt.Subplot(fig,gs1[3])
    
    gs2 = mpl.gridspec.GridSpec(4, 1)
    gs2.update(top=0.95, bottom=0.05, left=0.8, right=0.97,hspace=0.3)
    phot_hist = plt.Subplot(fig, gs2[0])
    opt_hist = plt.Subplot(fig, gs2[1])
    akar_hist = plt.Subplot(fig, gs2[2])
    spit_hist = plt.Subplot(fig,gs2[3])
    
    ##### add plots
    plots = [opt,akar,spit]
    plots_hist = [opt_hist, akar_hist, spit_hist]
    for plot in plots: fig.add_subplot(plot)
    for plot in plots_hist: fig.add_subplot(plot)
    fig.add_subplot(phot)
    fig.add_subplot(phot_hist)

    #### parameters
    alpha_minor = 0.2
    lw_minor = 0.5
    alpha_major = 0.8
    lw_major = 2.5

    ##### load and plot photometric residuals
    chi_magphys, chi_prosp, chisq_magphys, chisq_prosp, lam_rest, frac_magphys,frac_prosp, \
    ssfr_100,lam_rest_sf,lam_rest_qu,frac_prosp_sf,frac_prosp_qu,chisq_prosp_sf, \
    chisq_prosp_qu = [np.array([]) for i in range(14)]
    for data in alldata:

        if data:

            #### save chi
            chi_prosp = np.append(chi_prosp,data['residuals']['phot']['chi_prosp'])

            #### save goodness of fit
            chisq_prosp = np.append(chisq_prosp,data['residuals']['phot']['chisq_prosp'])
            lam_rest = np.append(lam_rest,data['residuals']['phot']['lam_obs']/(1+data['residuals']['phot']['z'])/1e4)

            #### star-forming or quiescent?
            ssfr_100 = np.append(ssfr_100,data['extras']['ssfr_100']['q50'])
            if ssfr_100[-1] > ssfr_limit:
                pcolor = sfcolor
                lam_rest_sf = np.append(lam_rest_sf,data['residuals']['phot']['lam_obs']/(1+data['residuals']['phot']['z'])/1e4)
                frac_prosp_sf = np.append(frac_prosp_sf,np.log10(1./(1-data['residuals']['phot']['frac_prosp'])))
                chisq_prosp_sf = np.append(chisq_prosp_sf,data['residuals']['phot']['chisq_prosp'])
            else:
                pcolor = qucolor
                lam_rest_qu = np.append(lam_rest_qu,data['residuals']['phot']['lam_obs']/(1+data['residuals']['phot']['z'])/1e4)
                frac_prosp_qu = np.append(frac_prosp_qu,np.log10(1./(1-data['residuals']['phot']['frac_prosp'])))
                chisq_prosp_qu = np.append(chisq_prosp_qu,data['residuals']['phot']['chisq_prosp'])


            #### plot
            phot.plot(data['residuals']['phot']['lam_obs']/(1+data['residuals']['phot']['z'])/1e4, 
                      np.log10(1./(1-data['residuals']['phot']['frac_prosp'])),
                      alpha=alpha_minor,
                      color=pcolor,
                      lw=lw_minor
                      )

    sfflag = ssfr_100 > ssfr_limit

    ##### calculate and plot running median
    pro_sf_bins, pro_sf_median = median_by_band(lam_rest_sf,frac_prosp_sf)
    pro_qu_bins, pro_qu_median = median_by_band(lam_rest_qu,frac_prosp_qu)

    phot.plot(pro_qu_bins, 
              pro_qu_median,
              color='black',
              lw=lw_major*1.1
              )
    phot.plot(pro_qu_bins, 
              pro_qu_median,
              color=qucolor,
              lw=lw_major
              )

    phot.plot(pro_sf_bins, 
              pro_sf_median,
              color='black',
              lw=lw_major*1.1
              )
    phot.plot(pro_sf_bins, 
              pro_sf_median,
              color=sfcolor,
              lw=lw_major
              )

    phot.text(0.98,0.85, 'star-forming',
              transform = phot.transAxes,horizontalalignment='right',
              color=sfcolor)
    phot.text(0.98,0.75, 'quiescent',
              transform = phot.transAxes,horizontalalignment='right',
              color=qucolor)
    phot.text(0.98,0.05, 'photometry',
              transform = phot.transAxes,horizontalalignment='right')
    phot.set_xlabel(r'$\lambda_{\mathrm{rest}}$ [$\mu$m]')
    phot.set_ylabel(r'log(f$_{\mathrm{obs}}/$f$_{\mathrm{mod}}$)')
    phot.axhline(0, linestyle=':', color='grey')
    phot.set_xscale('log',nonposx='clip',subsx=(2,4,7))
    phot.xaxis.set_minor_formatter(minorFormatter)
    phot.xaxis.set_major_formatter(majorFormatter)
    phot.set_xlim(0.12,600)
    phot.set_ylim(-0.4,0.4)

    ##### histogram of chisq values
    nbins = 10
    alpha_hist = 0.6
    # first call is color-less, to get bins
    # suitable for both data sets
    histmax = np.log10(100.0)
    chisq_prosp_sf = np.log10(chisq_prosp_sf)
    chisq_prosp_qu = np.log10(chisq_prosp_qu)

    okqu = chisq_prosp_qu < histmax
    oksf = chisq_prosp_sf < histmax
    n, b, p = phot_hist.hist([chisq_prosp_qu[okqu],chisq_prosp_sf[oksf]],
                         nbins, histtype='bar',
                         alpha=0.0,lw=2)
    n, b, p = phot_hist.hist(chisq_prosp_sf[oksf],
                         bins=b, histtype='bar',
                         color=sfcolor,
                         alpha=alpha_hist,lw=2)
    n, b, p = phot_hist.hist(chisq_prosp_qu[okqu],
                         bins=b, histtype='bar',
                         color=qucolor,
                         alpha=alpha_hist,lw=2)
    phot_hist.set_ylabel('N')
    phot_hist.xaxis.set_major_locator(MaxNLocator(4))

    phot_hist.set_xlabel(r'log($\chi^2_{\mathrm{phot}}$/N$_{\mathrm{phot}}$)')

    ##### load and plot spectroscopic residuals
    label = ['Optical','Akari', 'Spitzer IRS']
    nbins = [100,100,100]
    pmax = 0.0
    pmin = 0.0
    for i, plot in enumerate(plots):
        res_prosp_qu, rms_pro_qu, obs_restlam_qu, \
        res_prosp_sf, rms_pro_sf, obs_restlam_sf = [np.array([]) for nn in range(6)]
        for kk,data in enumerate(alldata):
            if data:
                if label[i] in data['residuals'].keys():

                    xplot_prosp = data['residuals'][label[i]]['obs_restlam']
                    yplot_prosp = data['residuals'][label[i]]['prospector_resid']

                    # color?
                    if sfflag[kk]:
                        pcolor=sfcolor
                        res_prosp_sf = np.append(res_prosp_sf,data['residuals'][label[i]]['prospector_resid'])
                        obs_restlam_sf = np.append(obs_restlam_sf,data['residuals'][label[i]]['obs_restlam'])
                        rms_pro_sf = np.append(rms_pro_sf,data['residuals'][label[i]]['prospector_rms'])
                    else:
                        pcolor=qucolor
                        res_prosp_qu = np.append(res_prosp_qu,data['residuals'][label[i]]['prospector_resid'])
                        obs_restlam_qu = np.append(obs_restlam_qu,data['residuals'][label[i]]['obs_restlam'])
                        rms_pro_qu = np.append(rms_pro_qu,data['residuals'][label[i]]['prospector_rms'])

                    plot.plot(xplot_prosp, 
                              yplot_prosp,
                              alpha=alpha_minor,
                              color=pcolor,
                              lw=lw_minor
                              )

                    pmin = np.min((pmin,np.nanmin(yplot_prosp)))
                    pmax = np.max((pmax,np.nanmax(yplot_prosp)))

        ##### calculate and plot running median
        # wrap quiescent galaxies in try-except clause
        # as there are no Akari spectra for these
        try:
            probins_qu, promedian_qu = prosp_dutils.running_median(obs_restlam_qu,res_prosp_qu,nbins=nbins[i])
            plot.plot(probins_qu, 
                      promedian_qu,
                      color='black',
                      lw=lw_major*1.1
                      )
            plot.plot(probins_qu, 
                      promedian_qu,
                      color=qucolor,
                      lw=lw_major
                      )
        except ValueError:
            pass

        try:
            probins_sf, promedian_sf = prosp_dutils.running_median(obs_restlam_sf,res_prosp_sf,nbins=nbins[i])
            plot.plot(probins_sf, 
                      promedian_sf,
                      color='black',
                      lw=lw_major*1.1
                      )
            plot.plot(probins_sf, 
                      promedian_sf,
                      color=sfcolor,
                      lw=lw_major
                      )
        except ValueError:
            pass

        plt_ylim = np.max((np.abs(pmin*0.8),np.abs(pmax*1.2)))
        plt_xlim_lo = np.min(np.concatenate((obs_restlam_sf,obs_restlam_qu)))*0.99
        plt_xlim_up = np.max(np.concatenate((obs_restlam_sf,obs_restlam_qu)))*1.01

        plot.set_ylim(-0.8,0.8)
        plot.set_xlim(plt_xlim_lo,plt_xlim_up)
        if i == 2:
            plot.set_xscale('log',nonposx='clip',subsx=(1,2,3,4,5,6,7,8,9))
        else:
            plot.set_xscale('log',nonposx='clip',subsx=(1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,8,9))
        plot.xaxis.set_minor_formatter(minorFormatter)
        plot.xaxis.set_major_formatter(majorFormatter)
        plot.set_xlabel(r'$\lambda_{\mathrm{rest}}$ [$\mu$m]')
        plot.set_ylabel(r'log(f$_{\mathrm{obs}}/$f$_{\mathrm{mod}}$)')
        plot.text(0.985,0.05, label[i],
                  transform = plot.transAxes,horizontalalignment='right')
        plot.axhline(0, linestyle=':', color='grey')

        ##### histogram of mean offsets
        nbins_hist = 10
        alpha_hist = 0.6
        # first histogram is transparent, to get bins
        # suitable for both data sets
        histmax = 2
        okproqu = rms_pro_qu < histmax
        okprosf = rms_pro_sf < histmax

        n, b, p = plots_hist[i].hist([rms_pro_sf[okprosf],rms_pro_qu[okproqu]],
                             nbins_hist, histtype='bar',
                             alpha=0.0,lw=2)
        n, b, p = plots_hist[i].hist(rms_pro_sf[okprosf],
                             bins=b, histtype='bar',
                             color=sfcolor,
                             alpha=alpha_hist,lw=2)
        n, b, p = plots_hist[i].hist(rms_pro_qu[okproqu],
                             bins=b, histtype='bar',
                             color=qucolor,
                             alpha=alpha_hist,lw=2)

        plots_hist[i].set_ylabel('N')
        plots_hist[i].set_xlabel(r'RMS [dex]')
        plots_hist[i].xaxis.set_major_locator(MaxNLocator(4))

    outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/magphys/'
    
    plt.savefig(outfolder+'median_residuals.png',dpi=dpi)
    plt.close()

def return_sedplot_vars(spec, mu, obs, sps, nufnu=True):

    '''
    if nufnu == True: return in units of nu * fnu (maggies * Hz). Else, return maggies.
    '''

    # observational information
    mask = obs['phot_mask']
    wave_eff = obs['wave_effective'][mask]
    obs_maggies = obs['maggies'][mask]
    obs_maggies_unc = obs['maggies_unc'][mask]
    mu = mu[mask]

    # output units
    if nufnu == True:
        mu *= c/wave_eff
        spec *= c/sps.wavelengths
        obs_maggies *= c/wave_eff
        obs_maggies_unc *= c/wave_eff

    # here we want to return
    # effective wavelength of photometric bands, observed maggies, observed uncertainty, model maggies, observed_maggies-model_maggies / uncertainties
    # model maggies, observed_maggies-model_maggies/uncertainties
    return wave_eff, obs_maggies, obs_maggies_unc, mu, (obs_maggies-mu)/obs_maggies_unc, (obs_maggies-mu)/obs_maggies, spec, sps.wavelengths

def mask_emission_lines(lam,z):

    # OII, Hbeta, OIII, Halpha, NII, SII
    lam_temp = lam*1e4
    mask_lines = np.array([3727, 4102, 4341, 4861, 4959, 5007, 6563, 6583,6720])*(1.0+z)
    mask_size = 20 # Angstroms
    mask = np.ones_like(lam,dtype=bool)

    for line in mask_lines: mask[(lam_temp > line - mask_size) & (lam_temp < line + mask_size)] = 0.0

    return mask

def calc_rms(lam, z, resid):

    mask = mask_emission_lines(lam,z)
    rms = (np.sum((resid[mask]-resid[mask].mean())**2)/np.sum(mask))**0.5

    return rms

def plot_obs_spec(obs_spec, phot, spec_res, alpha, 
                  modlam, modspec, maglam, magspec,z, 
                  objname, source, sigsmooth,
                  color='black',label=''):

    '''
    standard wrapper for plotting observed + residuals for spectra
    for individual galaxies
    '''

    mask = obs_spec['source'] == source
    obslam = obs_spec['obs_lam'][mask]/1e4
    lw = 2.5

    ##### smooth the model, or the observations
    if label != 'Spitzer IRS':
        modspec_smooth = prosp_dutils.smooth_spectrum(modlam,
                                                modspec,sigsmooth)
        obs_flux = obs_spec['flux'][mask]
    else: # observations!
        modspec_smooth = modspec
        obs_flux = prosp_dutils.smooth_spectrum(obslam, obs_spec['flux'][mask], 2500)

    # interpolate fsps spectra onto observational wavelength grid
    pro_flux_interp = interp1d(modlam*(1+z),
                               modspec_smooth, 
                               bounds_error = False, fill_value = 0)

    prospector_resid = np.log10(obs_flux) - np.log10(pro_flux_interp(obslam))

    obs_spec_plot = np.log10(obs_flux)
    prosp_spec_plot = np.log10(pro_flux_interp(obslam))

    #### calculate rms
    # mask emission lines
    prospector_rms = calc_rms(obslam, z, prospector_resid)

    if spec_res is not None:
        spec_res.plot(obslam, 
                      obs_spec_plot,
                      color=obs_color,
                      alpha=alpha,
                      linestyle='-',
                      label='observed',
                      lw=lw)

        spec_res.plot(obslam, 
                      prosp_spec_plot,
                      color=prosp_color,
                      alpha=alpha,
                      linestyle='-',
                      label='Prospector (predicted)',
                      lw=lw)

        #### write text, add lines
        spec_res.text(0.98,0.05, 'RMS='+"{:.2f}".format(prospector_rms)+' dex',
                      transform = spec_res.transAxes,ha='right',
                      color='black',fontsize=14)
        spec_res.text(0.015,0.05, label,
                      transform = spec_res.transAxes)
        spec_res.axhline(0, linestyle=':', color='grey')
        spec_res.set_xlim(min(obslam)*0.85,max(obslam)*1.15)
        if label == 'Optical':
            spec_res.set_xlim(min(obslam)*0.98,max(obslam)*1.02)

        plt_max=np.array([np.nanmax(np.abs(obs_spec_plot)),np.nanmax(np.abs(prosp_spec_plot))]).max()
        plt_min=np.array([np.nanmin(np.abs(obs_spec_plot)),np.nanmin(np.abs(prosp_spec_plot))]).min()
        spec_res.set_ylim(plt_min-0.35,plt_max+0.35)

        spec_res.set_xscale('log',nonposx='clip', subsx=(1,2,3,4,5,6,7,8,9))
        spec_res.xaxis.set_minor_formatter(minorFormatter)
        spec_res.xaxis.set_major_formatter(majorFormatter)

        if label == 'Optical':
            from collections import OrderedDict
            handles, labels = spec_res.get_legend_handles_labels()
            by_label = OrderedDict(zip(labels, handles))
            spec_res.legend(by_label.values(), by_label.keys(), 
                            loc=2, prop={'size':12},
                            frameon=False)

    # output rest-frame wavelengths + residuals
    out = {
           'obs_restlam': obslam/(1+z),
           'obs_obslam': obslam,
           'prospector_resid': prospector_resid,
           'prospector_rms': prospector_rms
           }

    return out

def update_model_info(alldata, sample_results, extra_output, magphys):

    alldata['objname'] = sample_results['run_params']['objname']
    alldata['thetas'] = extra_output['thetas']
    npars = sample_results['chain'].shape[-1]
    
    alldata['spec_info'] = extra_output['obs']
    alldata['weights'] = extra_output['weights']
    alldata['phot_mask'] = sample_results['obs']['phot_mask']

    alldata['obs_maggies'] = sample_results['obs']['maggies'][alldata['phot_mask']]
    alldata['filters'] = np.array(sample_results['obs']['filternames'])[alldata['phot_mask']]

    alldata['extras'] = extra_output['extras']
    
    return alldata

def sed_comp_figure(sample_results, extra_output, sps, magphys,
                    alpha=0.3, samples = [-1],
                    maxprob=0, outname=None, fast=False,
                    truths = None, agb_off = False, runname=None,
                    **kwargs):
    """
    Plot the photometry for the model and data (with error bars) for
    a single object, and plot residuals.

    Returns a dictionary called 'residuals', which contains the 
    photometric + spectroscopic residuals for this object, for both
    magphys and prospector.
    """

    z_txt = copy.copy(sample_results['model'].params['zred'])
    lumdist = copy.copy(sample_results['model'].params['lumdist'])
    sigsmooth = [450.0, 1000.0, 1.0]
    residuals = {}
    alpha = 0.65
    ms = 8

    # generate best-fit model
    spec,mags,sm = sample_results['model'].mean_model(sample_results['chain'][extra_output['sample_idx'][0]], sample_results['obs'], sps=sps)

    ##### Prospector maximum probability model ######
    wave_eff, obsmags, obsmags_unc, modmags, chi, frac_prosp, modspec, modlam = return_sedplot_vars(spec, mags, sample_results['obs'], sps)

    if outname is not None:
        #### set up plot
        fig = plt.figure(figsize=(12,12))
        gs = mpl.gridspec.GridSpec(2, 1, height_ratios=[3,1])
        gs.update(bottom=0.525, top=0.99, hspace=0.00)
        phot, res = plt.Subplot(fig, gs[0]), plt.Subplot(fig, gs[1])

        gs2 = mpl.gridspec.GridSpec(3, 1)
        gs2.update(top=0.465, bottom=0.05, hspace=0.3)
        spec_res_opt,spec_res_akari,spec_res_spit = plt.subplot(gs2[0]),plt.subplot(gs2[1]),plt.subplot(gs2[2])

        phot.plot(wave_eff/1e4, np.log10(modmags), 
                  color=prosp_color, marker='o', ms=ms, 
                  linestyle=' ', label='Prospector (fit)', alpha=alpha, 
                  markeredgewidth=0.7,**kwargs)
        
        res.plot(wave_eff/1e4, chi, 
                 color=prosp_color, marker='o', linestyle=' ', label='Prospector (fit)', 
                 ms=ms,alpha=alpha,markeredgewidth=0.7,**kwargs)
        
        nz = modspec > 0
        phot.plot(modlam[nz]/1e4, np.log10(modspec[nz]), linestyle='-',
                  color=prosp_color, alpha=0.6,**kwargs)

        ###### spectra for q50 + 5th, 95th percentile
        w = extra_output['observables']['lam_obs']
        spec_pdf = np.zeros(shape=(len(w),3))
        for jj in xrange(len(w)): spec_pdf[jj,:] = np.percentile(extra_output['observables']['spec'][jj,:],[16.0,50.0,84.0])
        sfactor = 3e18/w    # must convert to nu fnu

        nz = spec_pdf[:,1] > 0
        phot.fill_between(w/1e4, np.log10(spec_pdf[:,0]*sfactor), 
                             np.log10(spec_pdf[:,2]*sfactor),
                             color=median_err_color,
                             zorder=-48)


        ##### photometric observations, errors ######
        xplot = wave_eff/1e4
        yplot = np.log10(obsmags)
        phot.errorbar(xplot, yplot, yerr=np.log10(obsmags_unc+obsmags) - np.log10(obsmags),
                      color=obs_color, marker='o', label='observed', alpha=alpha, linestyle=' ',ms=ms)

        # plot limits
        phot.set_xlim(min(xplot)*0.4,max(xplot)*1.5)
        phot.set_ylim(min(yplot[np.isfinite(yplot)])-0.4,max(yplot[np.isfinite(yplot)])+0.4)
        res.set_xlim(min(xplot)*0.4,max(xplot)*1.5)
        res.axhline(0, linestyle=':', color='grey')

        resplots = [spec_res_opt, spec_res_akari, spec_res_spit]

    else:
        phot = None
        resplots = [None, None, None]

    ##### observed spectra + residuals #####
    obs_spec = prospector_io.load_spectra(sample_results['run_params']['objname'])

    label = ['Optical','Akari', 'Spitzer IRS']

    nplot = 0
    for ii in xrange(3):
        
        if label[ii] == 'Optical':
            residuals['emlines'] = measure_emline_lum.measure(sample_results, extra_output, obs_spec, sps,runname=runname)
            if residuals.get('emlines',None) is not None:
                sigsmooth[ii] = residuals['emlines']['sigsmooth']
            else:
                sigsmooth[ii] = 450.0

        source = ii+1
        mask = obs_spec['source'] == source
        if np.sum(mask) > 0:
            residuals[label[ii]] = plot_obs_spec(obs_spec, phot, resplots[nplot], alpha, modlam/1e4, modspec,
                                                 None, None,
                                                 extra_output['zred'], sample_results['run_params']['objname'],
                                                 source, color=obs_color, label=label[ii],sigsmooth=sigsmooth[ii])
            nplot += 1

    # calculate reduced chi-squared
    chisq=np.sum(chi**2)/(np.sum(sample_results['obs']['phot_mask']))

    if outname is not None:

        for kk in xrange(nplot,3): resplots[kk].axis('off')

        # diagnostic text
        textx = 0.02
        texty = 0.97
        deltay = 0.05


        #### SFR and mass
        # calibrated to be to the right of ax_loc = [0.38,0.68,0.13,0.13]
        prosp_sfr = extra_output['extras']['sfr_100']['q50']
        prosp_mass = np.log10(extra_output['extras']['stellar_mass']['q50'])

        phot.text(textx, texty-deltay, r'best-fit $\chi^2$/N$_{\mathrm{phot}}$='+"{:.2f}".format(chisq),
                  fontsize=14, ha='left', color=prosp_color,transform = phot.transAxes)
                        
        # galaxy text
        phot.text(textx, texty-2*deltay, 'z='+"{:.2f}".format(z_txt),
                  fontsize=14, ha='left',transform = phot.transAxes)
            
        # extra line
        phot.axhline(0, linestyle=':', color='grey')
        
        ##### add SFH plot
        from prosp_diagnostic_plots import add_sfh_plot
        ax_loc = [0.38,0.68,0.13,0.13]
        ax_inset = fig.add_axes(ax_loc,zorder=32)
        text_size = 1.5

        add_sfh_plot([extra_output],fig,sps,text_size=text_size,ax_inset=ax_inset,main_color=['black'])

        ##### add MAGPHYS SFH
        ax_inset.set_xlabel('time [Gyr]')


        # legend
        # make sure not to repeat labels
        from collections import OrderedDict
        handles, labels = phot.get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        phot.legend(by_label.values(), by_label.keys(), 
                    loc=1, prop={'size':12},
                    frameon=False)
                    
        # set labels
        res.set_ylabel( r'$\chi$')
        for plot in resplots: plot.set_ylabel(r'log($\nu$f$_{\nu}$)')
        phot.set_ylabel(r'log($\nu$f$_{\nu}$)')
        spec_res_spit.set_xlabel(r'$\lambda_{obs}$ [$\mathrm{\mu}$m]')
        res.set_xlabel(r'$\lambda_{obs}$ [$\mathrm{\mu}$m]')
        
        # set scales
        phot.set_xscale('log',nonposx='clip')
        res.set_xscale('log',nonposx='clip',subsx=(2,5))
        res.xaxis.set_minor_formatter(minorFormatter)
        res.xaxis.set_major_formatter(majorFormatter)

        # chill on the number of tick marks
        allres = resplots+[res]
        for plot in allres: plot.yaxis.set_major_locator(MaxNLocator(4))

        # clean up and output
        fig.add_subplot(phot)
        for res in allres: fig.add_subplot(res)
        
        # set second x-axis
        y1, y2=phot.get_ylim()
        x1, x2=phot.get_xlim()
        ax2=phot.twiny()
        ax2.set_xticks(np.arange(0,10,0.2))
        ax2.set_xlim(x1/(1+z_txt), x2/(1+z_txt))
        ax2.set_xlabel(r'$\lambda_{rest}$ [$\mathrm{\mu}$m]')
        ax2.set_ylim(y1, y2)
        ax2.set_xscale('log',nonposx='clip',subsx=(2,5))
        ax2.xaxis.set_minor_formatter(minorFormatter)
        ax2.xaxis.set_major_formatter(majorFormatter)

        # remove ticks
        phot.set_xticklabels([])
    
        fig.savefig(outname, bbox_inches='tight', dpi=dpi)
        #os.system('open '+outname)
        plt.close()

    # save chi+fractional difference for photometry
    out = {'chi_prosp': chi,
           'frac_prosp': frac_prosp,
           'chisq_prosp': chisq,
           'lam_obs': wave_eff,
           'z': extra_output['zred'],
           'lumdist': lumdist
           }
    residuals['phot'] = out
    return residuals
    
def collate_data(filebase=None,
                 outfolder=os.getenv('APPS')+'/prospector_alpha/plots/',
                 sample_results=None,
                 sps=None, elines_only=True,
                 runname=None,
                 plt_sed=True):

    '''
    Driver. Loads output, makes residual plots for a given galaxy, saves collated output.
    '''

    # make sure the output folder exists
    if not os.path.isdir(outfolder):
        os.makedirs(outfolder)

    # attempt to load data
    try:
         sample_results, powell_results, model, extra_output = prospector_io.load_prospector_data(filebase)
    except AttributeError:
        print 'failed to load ' + filebase
        return None

    if not sps:
        sps = pfile.load_sps(**sample_results['run_params'])

    # load magphys
    objname = sample_results['run_params']['objname']
    #magphys = read_magphys_output(objname=objname)

    # BEGIN PLOT ROUTINE
    print 'MAKING PLOTS FOR ' + objname + ' in ' + outfolder

    # do we have a model?
    for key in sample_results['run_params']:
      if type(sample_results['run_params'][key]) == unicode:
          if 'prospector_alpha' in sample_results['run_params'][key]:
              sample_results['run_params'][key] = os.getenv('APPS')+'/prospector_alpha'+sample_results['run_params'][key].split('prospector_alpha')[-1]
    sample_results['model'] = pfile.load_model(**sample_results['run_params'])

    if elines_only:
        print 'analyzing spectrum ONLY'
        outname = None
    else:
        print 'MAKING SED COMPARISON PLOT'
        outname = outfolder+objname.replace(' ','_')+'.sed.png'
    # plot
    t1 = time.time()
    residuals = sed_comp_figure(sample_results, extra_output, sps,
                                None, maxprob=1,runname=runname, outname=outname)
    print('line measurement took {0}s'.format(time.time()-t1))
    print 'SAVING OUTPUTS for ' + sample_results['run_params']['objname']
    print ' '

    # SAVE OUTPUTS
    alldata = {}
    if residuals is not None:
        alldata['residuals'] = residuals
        alldata = update_model_info(alldata, sample_results, extra_output, None)
    else:
        alldata = None

    return alldata, sps

def plt_all(runname=None,startup=True,**extras):

    '''
    for a list of galaxies, make all plots

    startup: if True, then make all the residual plots and save pickle file
             if False, load previous pickle file
    '''
    if runname == None:
        runname = 'brownseds'

    outfolder = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/magphys/sed_residuals/'

    if startup == True:
        filebase, parm_basename, ancilname = prosp_dutils.generate_basenames(runname)
        alldata = []
        sps = None
        for jj in range(len(filebase)):

            dictionary, sps = collate_data(filebase=filebase[jj],\
                                           outfolder=outfolder,
                                           runname=runname,
                                           sps=sps,
                                           **extras)

            alldata.append(dictionary)
        prospector_io.save_alldata(alldata,runname=runname)
    else:
        alldata = prospector_io.load_alldata(runname=runname)

    ha_idx = alldata[0]['residuals']['emlines']['em_name'] == 'H$\\alpha$'
    hb_idx = alldata[0]['residuals']['emlines']['em_name'] == 'H$\\beta$'

    with open('brown_optical_info.dat', "w") as f:
      f.write('# name ha_lum ha_lum_errup ha_lum_errdown hb_lum hb_lum_errup hb_lum_errdown dn4000\n')
      for dat in alldata:
        objname = str(dat['objname'])
        halum = dat['residuals']['emlines']['obs']['lum'][ha_idx][0]
        halum_errup = dat['residuals']['emlines']['obs']['lum_errup'][ha_idx][0]
        halum_errdown = dat['residuals']['emlines']['obs']['lum_errdown'][ha_idx][0]

        hblum = dat['residuals']['emlines']['obs']['lum'][hb_idx][0]
        hblum_errup = dat['residuals']['emlines']['obs']['lum_errup'][hb_idx][0]
        hblum_errdown = dat['residuals']['emlines']['obs']['lum_errdown'][hb_idx][0]

        try:
          dn4000 = dat['residuals']['emlines']['obs']['dn4000']
        except:
          dn4000 = -99

        f.write(objname+" {:.4e}".format(halum)+" {:.4e}".format(halum_errup)+" {:.4e}".format(halum_errdown) + \
                " {:.4e}".format(hblum)+" {:.4e}".format(hblum_errup)+" {:.4e}".format(hblum_errdown)+\
                " {:.4f}".format(dn4000)+'\n')


    plot_all_residuals(alldata,runname)


    #### herschel flag
    hflag = np.array([True if np.sum(dat['residuals']['phot']['lam_obs'] > 5e5) else False for dat in alldata])
    stack_sfh.plot_stacked_sfh(alldata,os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/pcomp/')
    mag_ensemble.plot_emline_comp(alldata,os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/magphys/emlines_comp/',hflag)
    mag_ensemble.prospector_comparison(alldata,os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/pcomp/',hflag)
    mag_ensemble.plot_relationships(alldata,os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/magphys/')
    mag_ensemble.plot_comparison(alldata,os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/magphys/')


    prospector_io.write_results(alldata,os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/pcomp/')
    allpar_plot(alldata,hflag,os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/pcomp/')
    plot_all_residuals(alldata,runname)
    stack_irs_spectra.plot_stacks(alldata=alldata,outfolder=os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/pcomp/')

def perform_wavelength_cal(spec_dict, objname):
    '''
    (1) fit a polynomial to ratio
    (2) apply polynomial correction to obs_spec
    '''

    spec_cal = prospector_io.load_spec_cal(runname=None)

    #### find matching galaxy by loading basenames for BROWNSEDS
    filebase, parm_basename, ancilname=prosp_dutils.generate_basenames('brownseds')
    match = np.array([f.split('_')[-1] for f in filebase]) == objname

    #### find ratio, calculate polynomial
    # u, g, r pivot wavelengths
    lam = np.array([3556.52396991,4702.49527923,6175.5788808])
    ratio = spec_cal['obs_phot'][:,match] / spec_cal['spec_phot'][:,match]
    co = np.polyfit(lam, ratio, 2)

    #### correct optical spectra
    opt_idx = spec_dict['source'] == 1
    correction = co[0]*spec_dict['obs_lam'][opt_idx]**2+co[1]*spec_dict['obs_lam'][opt_idx]+co[2]
    spec_dict['flux'][opt_idx] = spec_dict['flux'][opt_idx]*correction

    return spec_dict

def compute_specmags(runname=None, outfolder=None):

    '''
    step 1: load observed spectra, model spectra, photometry
    step 2: normalize model spectra to observed spectra (at red end, + blue if using u-band)
    step 3: meld them together
    step 4: calculate u, g, r mags from spectra
    step 5: plot spectral u, g, r versus photometric u, g, r

    dump into pickle file, call perform_wavelength_cal() to apply

    future steps: don't use super low-res spectra
    '''


    print 'THIS FUNCTION IS BROKEN, UNTIL YOU LET THREED_DUTILS USE LUMDIST'
    from bsfh import model_setup

    if runname == None:
        runname = 'brownseds'

    #### load up prospector results
    filebase, parm_basename, ancilname=prosp_dutils.generate_basenames(runname)
    outname = os.getenv('APPS')+'/prospector_alpha/plots/'+runname+'/pcomp/sfrcomp.png'
    alldata = prospector_io.load_alldata(runname=runname)
    sps = prosp_dutils.setup_sps(custom_filter_key=None)

    optphot = np.zeros(shape=(3,len(alldata)))
    obsphot = np.zeros(shape=(3,len(alldata)))
    for ii,dat in enumerate(alldata):

        #### load up model spec
        z = dat['residuals']['phot']['z']
        mod_spec = dat['bfit']['spec']
        mod_wav = sps.wavelengths

        #### load up observed spec
        # arrives in maggies * Hz
        # change to maggies
        spec_dict = prospector_io.load_spectra(dat['objname'])
        opt_idx = spec_dict['source'] == 1
        obs_wav = spec_dict['obs_lam'][opt_idx]
        obs_spec = spec_dict['flux'][opt_idx] / (3e18 / obs_wav)

        #### find adjoining sections in model
        minwav = np.min(obs_wav)
        maxwav = np.max(obs_wav)

        lamlim = (2800,7500)
        lower_join = (mod_wav > lamlim[0]) & (mod_wav < minwav)
        upper_join = (mod_wav < lamlim[1]) & (mod_wav > maxwav)

        #### normalize and combine
        # take 300 angstrom slices on either side
        dellam = 300
        low_obs = obs_wav < minwav+dellam
        low_mod = (mod_wav < minwav + dellam) & (mod_wav > minwav)
        up_obs = obs_wav > maxwav-dellam
        up_mod = (mod_wav > maxwav - dellam) & (mod_wav < maxwav)

        # avoid known emission lines: [OII], [NII], Halpha, [SII]
        elines = np.array([3727,6549,6563,6583,6717,6731])
        lambuff = 22
        for line in elines:
            low_obs[(obs_wav > line - lambuff) & (obs_wav < line + lambuff)] = False
            low_mod[(mod_wav > line - lambuff) & (mod_wav < line + lambuff)] = False
            up_obs[(obs_wav > line - lambuff) & (obs_wav < line + lambuff)] = False
            up_mod[(mod_wav > line - lambuff) & (mod_wav < line + lambuff)] = False

        # calculate mean
        lownorm = np.mean(obs_spec[low_obs]) / np.mean(mod_spec[low_mod])
        upnorm = np.mean(obs_spec[up_obs]) / np.mean(mod_spec[up_mod])

        # combine
        comblam = np.concatenate((mod_wav[lower_join],obs_wav,mod_wav[upper_join]))
        combspec = np.concatenate((mod_spec[lower_join]*lownorm,obs_spec,mod_spec[upper_join]*upnorm))

        # plot conjoined spectra
        if True:
            # observed spectra
            plt.plot(obs_wav,np.log10(obs_spec),color='black')
            #plt.plot(obs_wav[up_obs],np.log10(obs_spec[up_obs]),color='purple')
            #plt.plot(mod_wav[up_mod],np.log10(mod_spec[up_mod]),color='green')

            # post-break model
            plt.plot(mod_wav[lower_join],np.log10(mod_spec[lower_join]*lownorm),color='red')
            plt.plot(mod_wav[upper_join],np.log10(mod_spec[upper_join]*upnorm),color='red')
            plt.plot(mod_wav[lower_join],np.log10(mod_spec[lower_join]),color='grey')
            plt.plot(mod_wav[upper_join],np.log10(mod_spec[upper_join]),color='grey')

            # limits, save
            plt.xlim(2800,7200)
            plt.savefig('/Users/joel/code/python/prospector_alpha/plots/'+runname+'/pcomp/specnorm/'+dat['objname']+'.png',dpi=100)
            plt.close()

        #### convert combspec from maggies to Lsun/Hz
        pc2cm =  3.08568E18
        dfactor = (1+z) / ( 4.0 * np.pi * (dat['residuals']['phot']['lumdist']*1e6*pc2cm)**2)
        combspec *= 3631*1e-23 / constants.L_sun.cgs.value / dfactor # to Jy, to erg/s/cm^2/Hz, to Lsun/cm^2/Hz, to Lsun/Hz

        #### integrate spectra, save mags
        optphot[0,ii],_ = prosp_dutils.integrate_mag(comblam,combspec,'SDSS Camera u',alt_file='/Users/joel/code/fsps/data/allfilters.dat',z=z)
        optphot[1,ii],_ = prosp_dutils.integrate_mag(comblam,combspec,'SDSS Camera g',alt_file='/Users/joel/code/fsps/data/allfilters.dat',z=z)
        optphot[2,ii],_ = prosp_dutils.integrate_mag(comblam,combspec,'SDSS Camera r',alt_file='/Users/joel/code/fsps/data/allfilters.dat',z=z)
        optphot[:,ii] =  10**(-0.4*optphot[:,ii])

        #### save observed mags
        run_params = model_setup.get_run_params(param_file=parm_basename[ii])
        obs = model_setup.load_obs(**run_params)
        obsphot[0,ii] = obs['maggies'][obs['filters'] == 'SDSS_u']
        obsphot[1,ii] = obs['maggies'][obs['filters'] == 'SDSS_g']
        obsphot[2,ii] = obs['maggies'][obs['filters'] == 'SDSS_r']

    ##### plot
    kwargs = {'color':'0.5','alpha':0.8,'histtype':'bar','lw':2,'normed':1,'range':(0.5,2.0)}
    nbins = 80

    x = obsphot / optphot
    tits = [r'(photometric flux / spectral flux) [u-band]',
            r'(photometric flux / spectral flux) [g-band]',
            r'(photometric flux / spectral flux) [r-band]']

    #### ALL GALAXIES
    fig, axes = plt.subplots(1, 3, figsize = (18.75,6))
    for ii, ax in enumerate(axes):
        num, b, p = ax.hist(x[ii,:],nbins,**kwargs)
        save_xlim = ax.get_xlim()
        b = (b[:-1] + b[1:])/2.
        ax.set_ylabel('N')
        ax.set_xlabel(tits[ii])
        ax.set_xlim(save_xlim)
        ax.xaxis.set_major_locator(MaxNLocator(5))
    plt.savefig('/Users/joel/code/python/prospector_alpha/plots/'+runname+'/pcomp/spectral_integration.png',dpi=dpi)
    plt.close()

    out = {'obs_phot':obsphot,'spec_phot':optphot}
    prospector_io.save_spec_cal(out,runname=runname)

def add_prosp_mag_info(runname='brownseds_np'):

    alldata = prospector_io.load_alldata(runname=runname)

    filebase, parm_basename, ancilname=prosp_dutils.generate_basenames(runname)
    for ii,dat in enumerate(alldata):
        sample_results, powell_results, model, extra_output = prospector_io.load_prospector_data(filebase[ii], hdf5=True)
        dat = update_model_info(dat, sample_results, extra_output)
        print str(ii)+' done'

    prospector_io.save_alldata(alldata,runname=runname)















