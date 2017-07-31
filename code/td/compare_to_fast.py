import numpy as np
import matplotlib.pyplot as plt
import os
import hickle
import td_io
from prosp_dutils import generate_basenames,av_to_dust2,asym_errors,equalize_axes,offset_and_scatter,exp_decl_sfh_half_time
from brown_io import load_prospector_extra
import magphys_plot_pref
from astropy.cosmology import WMAP9
plt.ioff()

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)
popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'ms':9,'alpha':0.8,'color':'0.3'}
red = '#FF3D0D'
dpi = 120

minlogssfr = -15
minssfr = 1e-13
minsfr = 0.1

def collate_data(runname, runname_fast, filename=None, regenerate=False):
	
	### if it's already made, load it and give it back
	# else, start with the making!
	if os.path.isfile(filename) and regenerate == False:
		with open(filename, "r") as f:
			outdict=hickle.load(f)
			return outdict

	### define output containers
	parlabels = [r'log(M$_{\mathrm{stellar}}$/M$_{\odot}$)', 'SFR [M$_{\odot}$/yr]',
	             r'diffuse dust optical depth', r'log(sSFR) [yr$^-1$]',
	             r"t$_{\mathrm{half-mass}}$ [Gyr]", r'log(Z/Z$_{\odot}$)', r'Q$_{\mathrm{PAH}}$',
	             r'f$_{\mathrm{AGN}}$', 'dust index']
	pnames = ['stellar_mass','sfr_100','dust2','ssfr_100','half_time','logzsol', 'duste_qpah', 'fagn', 'dust_index']
	source = ['FAST', 'FAST', 'FAST', 'FAST', 'FAST']
	outprosp, outprosp_fast, outfast, outlabels = {},{},{},{}
	sfr_100_uvir = []
	outfast['z'] = []
	for i,par in enumerate(parlabels):
		
		### look for it in FAST
		try:
			x = source[i]
			outfast[pnames[i]] = []
		except IndexError:
			pass

		### if it's in FAST, it's in Prospector-FAST
		if pnames[i] in outfast.keys():
			outprosp_fast[pnames[i]] = {}
			outprosp_fast[pnames[i]]['q50'],outprosp_fast[pnames[i]]['q84'],outprosp_fast[pnames[i]]['q16'] = [],[],[]

		### it's always in Prospector
		outprosp[pnames[i]] = {}
		outprosp[pnames[i]]['q50'],outprosp[pnames[i]]['q84'],outprosp[pnames[i]]['q16'] = [],[],[]
	
	### fill output containers
	basenames, _, _ = generate_basenames(runname)
	basenames_fast, _, _ = generate_basenames(runname_fast)
	field = [name.split('/')[-1].split('_')[0] for name in basenames]

	fastlist, uvirlist = [], []
	allfields = np.unique(field).tolist()
	for f in allfields:
		fastlist.append(td_io.load_fast_v41(f))
		uvirlist.append(td_io.load_mips_data(f))
	for i, name in enumerate(basenames):

		print 'loading '+name.split('/')[-1]
		### make sure all files exist
		try:
			prosp = load_prospector_extra(name)
			prosp_fast = load_prospector_extra(basenames_fast[i])
		except:
			print name.split('/')[-1]+' failed to load. skipping.'
			continue

		### prospector first
		for par in pnames:
			### switch to tell between 'quantiles' and 'extras'
			pidx = prosp['quantiles']['parnames'] == par
			loc = 'quantiles'
			if pidx.sum() == 0:
				pidx = prosp['extras']['parnames'] == par
				loc = 'extras'
			### fill it up
			for q in ['q16','q50','q84']:

				# if we don't have that variable in the Prospector file,
				# zero it out
				if np.sum(pidx) == 0:
					outprosp[par][q].append(0.0)
					continue

				x = prosp[loc][q][pidx][0]
				if par == 'stellar_mass' or par == 'ssfr_100':
					x = np.log10(x)
				outprosp[par][q].append(x)

		### prospector-fast
		for par in pnames:
			### switch to tell between 'quantiles', 'extras', and 'NOT THERE'
			pidx = prosp_fast['quantiles']['parnames'] == par
			loc = 'quantiles'
			if pidx.sum() == 0:
				pidx = prosp_fast['extras']['parnames'] == par
				loc = 'extras'
			if np.sum(pidx) == 0:
					continue
			### fill it up
			for q in ['q16','q50','q84']:
				x = prosp_fast[loc][q][pidx][0]
				if par == 'stellar_mass' or par == 'ssfr_100':
					x = np.log10(x)
				outprosp_fast[par][q].append(x)

		### now FAST, UV+IR SFRs
		# find correct field, find ID match
		fidx = allfields.index(field[i])
		fast = fastlist[fidx]
		uvir = uvirlist[fidx]
		f_idx = fast['id'] == int(name.split('_')[-1])
		u_idx = uvir['id'] == int(name.split('_')[-1])

		# fill it up
		outfast['stellar_mass'].append(fast['lmass'][f_idx][0])
		outfast['dust2'].append(av_to_dust2(fast['Av'][f_idx][0]))
		outfast['sfr_100'].append(10**fast['lsfr'][f_idx][0])
		outfast['half_time'].append(exp_decl_sfh_half_time(10**fast['lage'][f_idx][0],10**fast['ltau'][f_idx][0])/1e9)
		outfast['z'].append(fast['z'][f_idx][0])
		sfr_100_uvir.append(uvir['sfr'][u_idx][0])

	### turn everything into numpy arrays
	for k1 in outprosp.keys():
		for k2 in outprosp[k1].keys():
			outprosp[k1][k2] = np.array(outprosp[k1][k2])
	for k1 in outprosp_fast.keys():
		for k2 in outprosp_fast[k1].keys():
			outprosp_fast[k1][k2] = np.array(outprosp_fast[k1][k2])
	for key in outfast: outfast[key] = np.array(outfast[key])

	out = {
		   'fast':outfast,
	       'prosp':outprosp,
	       'prosp_fast': outprosp_fast,
	       'labels':np.array(parlabels),
	       'pnames':np.array(pnames),
	       'uv_ir_sfr': np.array(sfr_100_uvir)
	      }

	### dump files and return
	hickle.dump(out,open(filename, "w"))
	return out

def do_all(runname='td_massive', runname_fast='fast_mimic',outfolder=None,**opts):

	if outfolder is None:
		outfolder = os.getenv('APPS') + '/prospector_alpha/plots/'+runname+'/fast_plots/'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)
			os.makedirs(outfolder+'data/')

	data = collate_data(runname,runname_fast,filename=outfolder+'data/fastcomp.h5',**opts)
	fast_comparison(data['fast'],data['prosp_fast'],data['labels'],data['pnames'],outfolder+'fast_to_fast_comparison.png')
	fast_comparison(data['prosp_fast'],data['prosp'],data['labels'],data['pnames'],outfolder+'fast_to_palpha_comparison.png')

	prospector_versus_z(data,outfolder+'prospector_versus_z.png')
	uvir_comparison(data,outfolder+'uvir_comparison.png')

def fast_comparison(fast,prosp,parlabels,pnames,outname):
	
	fig, axes = plt.subplots(2, 2, figsize = (10,10))
	axes = np.ravel(axes)

	for i,par in enumerate(['stellar_mass','sfr_100','dust2','half_time']):

		### clip SFRs
		if par[:3] == 'sfr':
			minimum = minsfr
		else:
			minimum = -np.inf

		### grab data
		try:
			xfast = np.clip(fast[par]['q50'],minimum,np.inf)
			xfast_up = np.clip(fast[par]['q84'],minimum,np.inf)
			xfast_down = np.clip(fast[par]['q16'],minimum,np.inf)
			xerr = asym_errors(xfast, xfast_up, xfast_down, log=False)
		except:
			xfast = np.clip(fast[par],minimum,np.inf)
			xerr = None

		yprosp = np.clip(prosp[par]['q50'],minimum,np.inf)
		yprosp_up = np.clip(prosp[par]['q84'],minimum,np.inf)
		yprosp_down = np.clip(prosp[par]['q16'],minimum,np.inf)
		yerr = asym_errors(yprosp, yprosp_up, yprosp_down, log=False)

		### plot
		axes[i].errorbar(xfast,yprosp,xerr=xerr,yerr=yerr,**popts)

		### if we have some enforced minimum, don't include in scatter calculation
		if ((xfast == xfast.min()).sum()-1 != 0) | ((yprosp == yprosp.min()).sum()-1 != 0):
			good = (xfast != xfast.min()) & (yprosp != yprosp.min())
		else:
			good = np.ones_like(xfast,dtype=bool)

		## log axes & range
		if par[:3] == 'sfr' or par == 'half_time':
			sub = (1,2,4)
			if par[:3] == 'sfr':
				sub = ([1])
				
			axes[i].set_yscale('log',nonposy='clip',subsy=sub)
			axes[i].yaxis.set_major_formatter(majorFormatter)
			axes[i].yaxis.set_minor_formatter(minorFormatter)
			axes[i].set_xscale('log',nonposy='clip',subsx=sub)
			axes[i].xaxis.set_major_formatter(majorFormatter)
			axes[i].xaxis.set_minor_formatter(minorFormatter)

			axes[i] = equalize_axes(axes[i], np.log10(xfast), np.log10(yprosp), dynrange=0.1, line_of_equality=True, log_in_linear=True)
			off,scat = offset_and_scatter(np.log10(xfast[good]),np.log10(yprosp[good]),biweight=True)

		else:
			axes[i] = equalize_axes(axes[i], xfast, yprosp, dynrange=0.1, line_of_equality=True)
			off,scat = offset_and_scatter(xfast[good],yprosp[good],biweight=True)
		
		if par == 'dust2':
			scatunits = ''
		else:
			scatunits = ' dex'

		### labels
		axes[i].text(0.95,0.12, 'offset='+"{:.2f}".format(off)+scatunits,
                     transform = axes[i].transAxes,horizontalalignment='right')
		axes[i].text(0.95,0.06, 'biweight scatter='+"{:.2f}".format(scat)+scatunits,
                     transform = axes[i].transAxes,horizontalalignment='right')
		axes[i].set_xlabel('FAST '+ parlabels[pnames==par][0])
		axes[i].set_ylabel('Prospector '+  parlabels[pnames==par][0])

	plt.tight_layout()
	plt.savefig(outname,dpi=dpi)
	plt.close()

def prospector_versus_z(data,outname):
	
	fig, axes = plt.subplots(2, 3, figsize = (15,10))
	axes = np.ravel(axes)

	toplot = ['stellar_mass','sfr_100','ssfr_100','half_time','logzsol','dust2']
	for i,par in enumerate(toplot):
		zred = data['fast']['z']
		yprosp, yprosp_up, yprosp_down = data['prosp'][par]['q50'], data['prosp'][par]['q84'], data['prosp'][par]['q16']

		### clip SFRs
		if par[:3] == 'sfr':
			minimum = minsfr
		elif par == 'ssfr_100':
			minimum = minlogssfr
		else:
			minimum = -np.inf
		yprosp = np.clip(yprosp,minimum,np.inf)
		yprosp_up = np.clip(yprosp_up,minimum,np.inf)
		yprosp_down = np.clip(yprosp_down,minimum,np.inf)

		yerr = asym_errors(yprosp, yprosp_up, yprosp_down, log=False)
		axes[i].errorbar(zred,yprosp,yerr=yerr,**popts)
		axes[i].set_xlabel('redshift')
		axes[i].set_ylabel('Prospector '+data['labels'][data['pnames'] == par][0])

		# add tuniv
		if par == 'half_time':
			n = 50
			zred = np.linspace(zred.min(),zred.max(),n)
			tuniv = WMAP9.age(zred).value
			axes[i].plot(zred,tuniv,'--',lw=2,zorder=-1, color=red)
			axes[i].text(zred[n/2]*1.1,tuniv[n/2]*1.1, r't$_{\mathrm{univ}}$',rotation=-50,color=red,weight='bold')

		# logscale
		if par == 'sfr_100':
			axes[i].set_yscale('log',nonposy='clip',subsy=(1,3))
			axes[i].yaxis.set_major_formatter(majorFormatter)
			axes[i].yaxis.set_minor_formatter(minorFormatter)
	plt.tight_layout()
	plt.savefig(outname,dpi=dpi)
	plt.close()

def uvir_comparison(data, outname):

	### data
	sfr_prosp, sfr_prosp_up, sfr_prosp_down = data['prosp']['sfr_100']['q50'], data['prosp']['sfr_100']['q84'], data['prosp']['sfr_100']['q16']
	ssfr_prosp, ssfr_prosp_up, ssfr_prosp_down = 10**data['prosp']['ssfr_100']['q50'], 10**data['prosp']['ssfr_100']['q84'], 10**data['prosp']['ssfr_100']['q16']

	mass = 10**data['prosp']['stellar_mass']['q50']
	logzsol = data['prosp']['logzsol']['q50']
	halftime = data['prosp']['half_time']['q50']

	sfr_uvir = data['uv_ir_sfr']

	sfr_prosp = np.clip(sfr_prosp,minssfr,np.inf)
	sfr_prosp_up = np.clip(sfr_prosp_up,minssfr,np.inf)
	sfr_prosp_down = np.clip(sfr_prosp_down,minssfr,np.inf)
	sfr_uvir = np.clip(sfr_uvir,minssfr,np.inf)

	ssfr_prosp = np.clip(ssfr_prosp,minssfr,np.inf)
	ssfr_prosp_up = np.clip(ssfr_prosp_up,minssfr,np.inf)
	ssfr_prosp_down = np.clip(ssfr_prosp_down,minssfr,np.inf)
	ssfr_uvir = np.clip(sfr_uvir/mass,minssfr,np.inf)

	### flag for enforced minimum
	good = (sfr_uvir != minsfr) & (sfr_prosp != minsfr)
	good = (ssfr_uvir != minssfr) & (ssfr_prosp != minssfr)

	qpah, qpah_up, qpah_down = data['prosp']['duste_qpah']['q50'], data['prosp']['duste_qpah']['q84'], data['prosp']['duste_qpah']['q16']
	fagn, fagn_up, fagn_down = data['prosp']['fagn']['q50'], data['prosp']['fagn']['q84'], data['prosp']['fagn']['q16']
	sfr_ratio, sfr_ratio_up, sfr_ratio_down = np.log10(sfr_prosp/sfr_uvir), np.log10(sfr_prosp_up/sfr_uvir), np.log10(sfr_prosp_down/sfr_uvir)

	### errors
	sfr_err = asym_errors(sfr_prosp, sfr_prosp_up, sfr_prosp_down, log=False)
	ssfr_err = asym_errors(ssfr_prosp, ssfr_prosp_up, ssfr_prosp_down, log=False)
	sfr_ratio_err = asym_errors(sfr_ratio[good], sfr_ratio_up[good], sfr_ratio_down[good], log=False)
	qpah_err = asym_errors(qpah[good], qpah_up[good], qpah_down[good], log=False)
	fagn_err = asym_errors(fagn[good], fagn_up[good], fagn_down[good], log=False)

	### plot geometry
	if qpah.sum() != 0:
		fig, ax = plt.subplots(2, 2, figsize = (12,12))
	else:
		fig, ax = plt.subplots(1,2,figsize=(14,6))
	ax = np.ravel(ax)

	### UV_IR SFR plot
	ax[0].errorbar(ssfr_uvir, ssfr_prosp, yerr=ssfr_err, **popts)

	sub = ([1])
	ax[0].set_xlabel('sSFR$_{\mathrm{UVIR}}$')
	ax[0].set_ylabel('sSFR$_{\mathrm{Prosp}}$')
	ax[0].set_yscale('log',nonposy='clip',subsy=sub)
	ax[0].yaxis.set_major_formatter(majorFormatter)
	ax[0].yaxis.set_minor_formatter(minorFormatter)
	ax[0].set_xscale('log',nonposy='clip',subsx=sub)
	ax[0].xaxis.set_major_formatter(majorFormatter)
	ax[0].xaxis.set_minor_formatter(minorFormatter)

	off,scat = offset_and_scatter(np.log10(ssfr_uvir[good]),np.log10(ssfr_prosp[good]),biweight=True)
	scatunits = ' dex'
	ax[0].text(0.05,0.94, 'offset='+"{:.2f}".format(off)+scatunits,
                 transform = ax[0].transAxes)
	ax[0].text(0.05,0.89, 'biweight scatter='+"{:.2f}".format(scat)+scatunits,
                 transform = ax[0].transAxes)
	lim = ax[0].get_xlim()
	ax[0].plot([lim[0],lim[1]],[lim[0],lim[1]],'--', color='red', zorder=2)

	ax[1].errorbar(ssfr_prosp[good], np.log10(ssfr_prosp[good]/ssfr_uvir[good]), **popts)
	pts = ax[1].scatter(ssfr_prosp[good], np.log10(ssfr_prosp[good]/ssfr_uvir[good]), marker='o', c=halftime[good],
				  cmap=plt.cm.plasma,s=75,zorder=10)
	cbar = fig.colorbar(pts, ax=ax[1])
	cbar.set_label(r'half-mass time [Gyr]')
	ax[1].set_ylim(-1.0,0.5)
	ax[1].set_xlim(1e-11,5e-9)

	ax[1].set_xlabel('sSFR$_{\mathrm{Prosp}}$')
	ax[1].set_ylabel('log(sSFR$_{\mathrm{Prosp}}$/sSFR$_{\mathrm{UVIR}}$)')
	ax[1].axhline(0, linestyle='--', color='red',lw=2,zorder=-1)

	ax[1].set_xscale('log',nonposy='clip',subsx=([1]))
	ax[1].xaxis.set_major_formatter(majorFormatter)
	ax[1].xaxis.set_minor_formatter(minorFormatter)

	if ax.shape[0] > 2:
		ax[2].errorbar(qpah[good], sfr_ratio[good], xerr=qpah_err, yerr=sfr_ratio_err, **popts)
		ax[3].errorbar(fagn[good], sfr_ratio[good], xerr=fagn_err, yerr=sfr_ratio_err, **popts)

		### plot formatting
		# ratio plots
		subsx = [(1,3),([1])]
		for i, a in enumerate(ax[2:]):
			a.set_xscale('log',nonposy='clip',subsx=subsx[i])
			a.xaxis.set_major_formatter(majorFormatter)
			a.xaxis.set_minor_formatter(minorFormatter)	
			a.set_ylabel(r'log(SFR$_{\mathrm{Prosp}}$/SFR$_{\mathrm{UVIR}}$)')
			a.axhline(0, linestyle='--', color='red',lw=2,zorder=-1)

		ax[2].set_xlabel(data['labels'][data['pnames'] == 'duste_qpah'][0])
		ax[3].set_xlabel(data['labels'][data['pnames'] == 'fagn'][0])

	plt.tight_layout()
	plt.savefig(outname,dpi=dpi)
	plt.close()






