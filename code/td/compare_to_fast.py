import numpy as np
import matplotlib.pyplot as plt
import os
import hickle
import td_io
from threed_dutils import generate_basenames,av_to_dust2,asym_errors,equalize_axes,offset_and_scatter,exp_decl_sfh_half_time
from brown_io import load_prospector_extra
import magphys_plot_pref
from astropy.cosmology import WMAP9

minorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=False)
majorFormatter = magphys_plot_pref.jLogFormatter(base=10, labelOnlyBase=True)
popts = {'fmt':'o', 'capthick':1.5,'elinewidth':1.5,'ms':9,'alpha':0.8,'color':'0.6'}
red = '#FF3D0D'
dpi = 120

def collate_data(runname, filename=None, regenerate=False):
	
	### if it's already made, load it and give it back
	# else, start with the making!
	if os.path.isfile(filename) and regenerate == False:
		with open(filename, "r") as f:
			outdict=hickle.load(f)
			return outdict

	### define output containers
	fnames = ['']
	parlabels = [r'log(M$_{\mathrm{stellar}}$/M$_{\odot}$)', 'SFR [M$_{\odot}$/yr]',
	             r'diffuse dust optical depth', r"t$_{\mathrm{half-mass}}$ [Gyr]",
	             r'log(sSFR) [yr$^-1$]', r'log(Z/Z$_{\odot}$)']
	pnames = ['stellar_mass','sfr_100','dust2','half_time','ssfr_100','logzsol']
	source = ['FAST', 'FAST', 'FAST', 'FAST']
	outprosp, outlabels, outfast = {},{},{}
	for i,par in enumerate(parlabels):
		try:
			x = source[i]
			outfast[pnames[i]] = []
		except IndexError:
			pass
		outprosp[pnames[i]] = {}
		outprosp[pnames[i]]['q50'],outprosp[pnames[i]]['q84'],outprosp[pnames[i]]['q16'] = [],[],[]
	
	outfast['sfr_100_uvir'] = [] # FAST-only
	outfast['z'] = []

	### fill output containers
	basenames, _, _ = generate_basenames(runname)
	fast = td_io.load_fast_v41('COSMOS')
	uvir = td_io.load_mips_data('COSMOS')
	for i, name in enumerate(basenames):

		# prospector first
		prosp = load_prospector_extra(name)
		for par in pnames:
			pidx = prosp['quantiles']['parnames'] == par
			loc = 'quantiles'
			### switch to tell between 'quantiles' and 'extras'
			if pidx.sum() == 0:
				pidx = prosp['extras']['parnames'] == par
				loc = 'extras'
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


		# now fast and UVIR SFRs
		# haven't calculated half-mass time yet...
		f_idx = fast['id'] == int(name.split('_')[-1])
		u_idx = uvir['id'] == int(name.split('_')[-1])
		outfast['stellar_mass'].append(fast['lmass'][f_idx][0])
		outfast['dust2'].append(av_to_dust2(fast['Av'][f_idx][0]))
		outfast['sfr_100'].append(10**fast['lsfr'][f_idx][0])
		outfast['half_time'].append(exp_decl_sfh_half_time(10**fast['lage'][f_idx][0],10**fast['ltau'][f_idx][0])/1e9)
		outfast['sfr_100_uvir'].append(uvir['sfr'][u_idx][0])
		outfast['z'].append(fast['z'][f_idx][0])

	### put them into master container
	for k1 in outprosp.keys():
		for k2 in outprosp[k1].keys():
			outprosp[k1][k2] = np.array(outprosp[k1][k2])
	for key in outfast: outfast[key] = np.array(outfast[key])

	### add fast-only parts
	pnames.append('sfr_100_uvir')
	source.append('UV+IR')
	parlabels.append('log(SFR) [M$_{\odot}$/yr]')

	out = {
		   'fast':outfast,
	       'prosp':outprosp,
	       'labels':np.array(parlabels),
	       'pnames':np.array(pnames),
	       'source':np.array(source)
	      }

	### dump files and return
	hickle.dump(out,open(filename, "w"))
	return out

def do_all(runname='td_massive',outfolder=None,**opts):

	if outfolder is None:
		outfolder = os.getenv('APPS') + '/threedhst_bsfh/plots/'+runname+'/fast_plots/'
		if not os.path.isdir(outfolder):
			os.makedirs(outfolder)
			os.makedirs(outfolder+'data/')

	data = collate_data(runname,filename=outfolder+'data/fastcomp.h5',**opts)
	fast_comparison(data,outfolder+'fast_comparison.png')
	prospector_versus_z(data,outfolder+'prospector_versus_z.png')

def fast_comparison(data,outname):
	
	fig, axes = plt.subplots(2, 3, figsize = (15,10))
	axes = np.ravel(axes)

	i = -1
	for par in data['pnames']:

		try:
			xfast = data['fast'][par]
		except KeyError:
			print par + ' is not in FAST, skipping'
			continue
		i+=1
		### clip FAST SFRs
		if par[:3] == 'sfr':
			xfast = np.clip(xfast,0.01,np.inf)

		### hack to plot UV+IR SFRs
		if par == 'sfr_100_uvir':
			par = 'sfr_100'
		yprosp, yprosp_up, yprosp_down = data['prosp'][par]['q50'], data['prosp'][par]['q84'], data['prosp'][par]['q16']

		yerr = asym_errors(yprosp, yprosp_up, yprosp_down, log=False)

		axes[i].errorbar(xfast,yprosp,yerr=yerr,**popts)

		## log axes & range
		if par[:3] == 'sfr' or par == 'half_time':
			sub = (1,3)
			if par[:3] == 'sfr':
				sub = ([1])
				
			axes[i].set_yscale('log',nonposy='clip',subsy=sub)
			axes[i].yaxis.set_major_formatter(majorFormatter)
			axes[i].yaxis.set_minor_formatter(minorFormatter)
			axes[i].set_xscale('log',nonposy='clip',subsx=sub)
			axes[i].xaxis.set_major_formatter(majorFormatter)
			axes[i].xaxis.set_minor_formatter(minorFormatter)

			axes[i] = equalize_axes(axes[i], np.log10(xfast), np.log10(yprosp), dynrange=0.1, line_of_equality=True, log_in_linear=True)
			off,scat = offset_and_scatter(np.log10(xfast),np.log10(yprosp),biweight=True)
			scatunits = ' dex'

		else:
			axes[i] = equalize_axes(axes[i], xfast, yprosp, dynrange=0.2, line_of_equality=True)
			off,scat = offset_and_scatter(xfast,yprosp,biweight=True)
			scatunits = ''
			if par == 'stellar_mass':
				scatunits = ' dex'

		
		### labels
		
		axes[i].text(0.95,0.12, 'offset='+"{:.2f}".format(off)+scatunits,
                     transform = axes[i].transAxes,horizontalalignment='right')
		axes[i].text(0.95,0.06, 'biweight scatter='+"{:.2f}".format(scat)+scatunits,
                     transform = axes[i].transAxes,horizontalalignment='right')
		axes[i].set_xlabel(data['source'][i]+' '+data['labels'][data['pnames'] == par][0])
		axes[i].set_ylabel('Prospector '+data['labels'][data['pnames'] == par][0])
	axes[-1].axis('off')
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
