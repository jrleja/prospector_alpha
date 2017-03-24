import numpy as np
from threedhst_diag import add_sfh_plot
import os
import matplotlib.pyplot as plt
import magphys_plot_pref
import matplotlib as mpl
from magphys_plots import median_by_band
import threed_dutils
from astropy import constants
from matplotlib.ticker import MaxNLocator

'''
plot RMS for spectral quantities as a function of f_agn
'''

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)

def collate_data(alldata, alldata_noagn):

	'''
	return observed + model Halpha + Hbeta luminosities, Balmer decrement, and Dn4000, + errors
	also return f_agn
	'''

	### package up information
	the_data = [alldata, alldata_noagn]
	data_label = ['agn','no_agn']
	output = {}
	for ii in xrange(2):

		### build containers
		rms = {}
		labels = ['halpha','hbeta','bdec','dn4000']
		obs_eline_names = alldata[0]['residuals']['emlines']['em_name']
		mod_eline_names = alldata[0]['model_emline']['emnames']
		mod_extra_names = alldata[0]['pextras']['parnames']
		for d in ['obs','mod']:
			rms[d] = {}
			for l in labels: rms[d][l] = []

		#### model parameters
		objname = []
		model_pars = {}
		pnames = ['fagn', 'agn_tau']
		for p in pnames: 
			model_pars[p] = {'q50':[],'q84':[],'q16':[]}
		parnames = alldata[0]['pquantiles']['parnames']

		#### load model information
		for dat in the_data[ii]:

			# if we don't measure spectral parameters, we don't want that shit here
			if dat['residuals']['emlines'] is None:
				continue

			#### model parameters [NEW MODEL ONLY]
			objname.append(dat['objname'])
			if data_label[ii] == 'agn':
				for key in model_pars.keys():
					model_pars[key]['q50'].append(dat['pquantiles']['q50'][parnames==key][0])
					model_pars[key]['q84'].append(dat['pquantiles']['q84'][parnames==key][0])
					model_pars[key]['q16'].append(dat['pquantiles']['q16'][parnames==key][0])

			#### pull the chain for spectral quantities
			rms['obs']['halpha'].append(np.log10(dat['residuals']['emlines']['obs']['lum_chain'][:,obs_eline_names=='H$\\alpha$'].squeeze() / constants.L_sun.cgs.value))
			rms['obs']['hbeta'].append(np.log10(dat['residuals']['emlines']['obs']['lum_chain'][:,obs_eline_names=='H$\\beta$'].squeeze() / constants.L_sun.cgs.value))
			rms['obs']['bdec'].append(threed_dutils.bdec_to_ext(10**rms['obs']['halpha'][-1]/10**rms['obs']['hbeta'][-1]))
			rms['obs']['dn4000'].append(np.array(dat['residuals']['emlines']['obs']['dn4000']))

			rms['mod']['halpha'].append(np.log10(dat['model_emline']['flux']['chain'][:,mod_eline_names=='Halpha'].squeeze()))
			rms['mod']['hbeta'].append(np.log10(dat['model_emline']['flux']['chain'][:,mod_eline_names=='Hbeta'].squeeze()))
			rms['mod']['bdec'].append(threed_dutils.bdec_to_ext(dat['pextras']['flatchain'][:,mod_extra_names=='bdec_calc'].squeeze()))
			rms['mod']['dn4000'].append(dat['spec_info']['dn4000']['chain'].squeeze())

		#### numpy arrays
		for key in model_pars.keys(): 
			for key2 in model_pars[key].keys():
				model_pars[key][key2] = np.array(model_pars[key][key2])

		output[data_label[ii]] = {}
		output[data_label[ii]]['objname'] = objname
		output[data_label[ii]]['model_pars'] = model_pars
		output[data_label[ii]]['rms'] = rms

	return output

def plot_comparison(runname='brownseds_agn',runname_noagn='brownseds_np',alldata=None,alldata_noagn=None,outfolder=None):

	#### load alldata
	if alldata is None:
		import brown_io

		alldata = brown_io.load_alldata(runname=runname)
		alldata_noagn = brown_io.load_alldata(runname=runname_noagn)

	#### make output folder if necessary
	if outfolder is None:
		outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)
	
	### collate data
	### choose galaxies with largest 10 F_AGN
	pdata = collate_data(alldata,alldata_noagn)

	plot_rms(pdata,outfolder)

def plot_rms(pdata,outfolder):

	#### plot geometry
	fig, ax = plt.subplots(2,2, figsize=(10, 10))
	fig2, ax2 = plt.subplots(4,2, figsize=(12, 20))

	ax = ax.ravel()
	ax2 = ax2.ravel()
	red = '#FF3D0D'
	blue = '#1C86EE' 

	### titles
	trans = {
	         'halpha': r'log(H$_{\alpha}$ flux)',
	         'hbeta': r'log(H$_{\beta}$ flux)',
	         'dn4000': r'D$_{\mathrm{n}}$4000',
	         'bdec':r'log(F/F$_0$)$_{\mathrm{H}\alpha}$ - log(F/F$_0$)$_{\mathrm{H}\beta}$'
	        }
	lims = [(-2,2), (-2,2), (-0.3,0.3), (-0.15,0.15)]
	lims2 = [(3,10),(4.5,9),(-0.2,0.8),(0.6,2.0)]
	
	### for each observable, pull out RMS
	ndraw = int(1e5)
	fagn = np.log10(pdata['agn']['model_pars']['fagn']['q50'])
	ordered_keys = ['halpha','hbeta','bdec','dn4000']
	for ii,key in enumerate(ordered_keys):
		### for each galaxy
		q50, q84, q16 = [], [], []
		obs_agn, obs_noagn, mod_agn, mod_noagn = [], [], [], []
		ngal = len(pdata['no_agn']['rms']['mod'][key])
		for jj in xrange(ngal):
			obs_agn.append(np.random.choice(np.atleast_1d(pdata['agn']['rms']['obs'][key][jj]),size=ndraw))
			mod_agn.append(np.random.choice(pdata['agn']['rms']['mod'][key][jj],size=ndraw))
			rms_agn = np.sqrt((obs_agn[-1]-mod_agn[-1])**2)

			obs_noagn.append(np.random.choice(np.atleast_1d(pdata['no_agn']['rms']['obs'][key][jj]),size=ndraw))
			mod_noagn.append(np.random.choice(pdata['no_agn']['rms']['mod'][key][jj],size=ndraw))
			rms_noagn = np.sqrt((obs_noagn[-1]-mod_noagn[-1])**2)

			### if we have more than 1/6 NaNs (i.e. places where obs < 0), then dump it
			diff = rms_noagn-rms_agn
			if np.isnan(diff).sum() > diff.shape[0]/6.:
				cent,up,down = np.nan,np.nan,np.nan
			else:
				cent,up,down = np.nanpercentile(diff,[50.0,84.0,16.0])
			q50.append(cent)
			q84.append(up)
			q16.append(down)
		
		q50, q84, q16 = np.array(q50), np.array(q84), np.array(q16)
		idx = np.isfinite(q50)

		### plot
		errs = threed_dutils.asym_errors(q50[idx], q84[idx], q16[idx])
		ax[ii].errorbar(fagn[idx],q50[idx],yerr=errs,fmt='o', ecolor=blue, capthick=1,elinewidth=1,ms=8,alpha=0.5,zorder=-5)
		ax[ii].set_title(trans[key])
		ax[ii].set_xlabel(r'log(f$_{\mathrm{MIR}}$)')
		ax[ii].set_ylabel(r'RMS(NO AGN) - RMS(AGN)')
		ax[ii].axhline(0, linestyle='--', color='0.2')
		ax[ii].xaxis.set_major_locator(MaxNLocator(5))

		### dynamic ylimits
		#ymax = np.abs(np.concatenate((q16[idx],q84[idx]))).max()*1.05
		#ax[ii].set_ylim(-ymax,ymax)
		ax[ii].set_ylim(lims[ii][0],lims[ii][1])

		### running median
		x, y = threed_dutils.running_median(fagn[idx],q50[idx],nbins=9,weights=2./(q84[idx]-q16[idx])**2,avg=True)
		ax[ii].plot(x,y,color=red,lw=4,alpha=0.6)
		ax[ii].plot(x,y,color=red,lw=4,alpha=0.6)

		### plot both relations
		# AGN first
		q16o, q50o, q84o, q16m, q50m, q84m = [np.zeros(ngal) for i in range(6)]
		for jj in xrange(ngal):
			q50o[jj],q84o[jj],q16o[jj] = np.nanpercentile(obs_agn[jj],[50.0,84.0,16.0])
			q50m[jj],q84m[jj],q16m[jj] = np.nanpercentile(mod_agn[jj],[50.0,84.0,16.0])
		
		errs_obs = threed_dutils.asym_errors(q50o[idx], q84o[idx], q16o[idx])
		errs_mod = threed_dutils.asym_errors(q50m[idx], q84m[idx], q16m[idx])

		p2 = 2*ii
		ax2[p2].errorbar(q50o[idx],q50m[idx],xerr=errs_obs,yerr=errs_mod,fmt='o', 
			            ecolor=blue, capthick=1,elinewidth=1,ms=0.0,alpha=0.5,zorder=-5)
		pts = ax2[p2].scatter(q50o[idx], q50m[idx], marker='o', c=fagn[idx], cmap=plt.cm.plasma,s=70,zorder=10)

		ax2[p2].set_title(trans[key])
		ax2[p2].set_xlabel(r'observed')
		ax2[p2].set_ylabel(r'model (AGN)')
		ax2[p2].xaxis.set_major_locator(MaxNLocator(5))
		ax2[p2].yaxis.set_major_locator(MaxNLocator(5))
		ax2[p2].set_xlim(lims2[ii][0],lims2[ii][1])
		ax2[p2].set_ylim(lims2[ii][0],lims2[ii][1])
		ax2[p2].plot(lims2[ii],lims2[ii],'--',color='0.5',alpha=0.5)
		off,scat = threed_dutils.offset_and_scatter(q50o[idx],q50m[idx],biweight=True)
		ax2[p2].text(0.05,0.94, 'biweight scatter='+"{:.2f}".format(scat),
                     transform = ax2[p2].transAxes,horizontalalignment='left')
		ax2[p2].text(0.05,0.89, 'mean offset='+"{:.2f}".format(off),
                     transform = ax2[p2].transAxes,horizontalalignment='left')

		#### now no-AGN (same code)
		q16o, q50o, q84o, q16m, q50m, q84m = [np.zeros(ngal) for i in range(6)]
		for jj in xrange(ngal):
			q50o[jj],q84o[jj],q16o[jj] = np.nanpercentile(obs_noagn[jj],[50.0,84.0,16.0])
			q50m[jj],q84m[jj],q16m[jj] = np.nanpercentile(mod_noagn[jj],[50.0,84.0,16.0])
		
		errs_obs = threed_dutils.asym_errors(q50o[idx], q84o[idx], q16o[idx])
		errs_mod = threed_dutils.asym_errors(q50m[idx], q84m[idx], q16m[idx])

		p2+=1
		ax2[p2].errorbar(q50o[idx],q50m[idx],xerr=errs_obs,yerr=errs_mod,fmt='o', 
			            ecolor=blue, capthick=1,elinewidth=1,ms=0.0,alpha=0.5,zorder=-5)
		pts = ax2[p2].scatter(q50o[idx], q50m[idx], marker='o', c=fagn[idx], cmap=plt.cm.plasma,s=70,zorder=10)
		ax2[p2].set_title(trans[key])
		ax2[p2].set_xlabel(r'observed')
		ax2[p2].set_ylabel(r'model (no AGN)')
		ax2[p2].xaxis.set_major_locator(MaxNLocator(5))
		ax2[p2].yaxis.set_major_locator(MaxNLocator(5))
		ax2[p2].set_xlim(lims2[ii][0],lims2[ii][1])
		ax2[p2].set_ylim(lims2[ii][0],lims2[ii][1])
		ax2[p2].plot(lims2[ii],lims2[ii],'--',color='0.5',alpha=0.5)
		off,scat = threed_dutils.offset_and_scatter(q50o[idx],q50m[idx],biweight=True)
		ax2[p2].text(0.05,0.94, 'biweight scatter='+"{:.2f}".format(scat),
                     transform = ax2[p2].transAxes,horizontalalignment='left')
		ax2[p2].text(0.05,0.89, 'mean offset='+"{:.2f}".format(off),
                     transform = ax2[p2].transAxes,horizontalalignment='left')

		### label, add colorbar
		cb = fig.colorbar(pts, ax=ax2[p2], aspect=10)
		cb.set_label(r'f$_{\mathrm{MIR}}$')
		cb.solids.set_rasterized(True)
		cb.solids.set_edgecolor("face")

	fig.tight_layout()
	fig.savefig(outfolder+'delta_observables.png',dpi=120)

	fig2.tight_layout()
	fig2.savefig(outfolder+'obs_vs_mod.png',dpi=120)

	plt.close()








