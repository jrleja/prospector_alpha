import wise_colors,optical_color_color,bpt,plot_delta_pars,property_comparison,xray_luminosity,plot_spec_rms
import delta_mass_met
import os
import brown_io

def plot(runname='brownseds_agn',runname_noagn='brownseds_np',
	     alldata=None,alldata_noagn=None,open_all=True,outfolder=None,fmir=False):

	#### load alldata
	if alldata is None:
		alldata = brown_io.load_alldata(runname=runname)
	if alldata_noagn is None:
		alldata_noagn = brown_io.load_alldata(runname=runname_noagn)

	#### outfolder
	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/'
	if not os.path.isdir(outfolder):
		os.makedirs(outfolder)

	from copy import deepcopy
	if fmir:
		alldata_sub = deepcopy(alldata)
		for dat in alldata_sub:
			fidx = dat['pquantiles']['parnames'] == 'fagn'
			midx = dat['pextras']['parnames'] == 'fmir'

			dat['pquantiles']['q50'][fidx] = dat['pextras']['q50'][midx]
			dat['pquantiles']['q84'][fidx] = dat['pextras']['q84'][midx]
			dat['pquantiles']['q16'][fidx] = dat['pextras']['q16'][midx]
			dat['pquantiles']['sample_chain'][:,fidx] = dat['pextras']['flatchain'][:,midx]
	else:
		alldata_sub = deepcopy(alldata)

	#### PLOT ALL
	print 'PLOTTING WISE COLORS'
	wise_colors.plot_mir_colors(runname=runname,alldata=alldata_sub,outfolder=outfolder)
	print 'PLOTTING BPT DIAGRAM'
	bpt.plot_bpt(runname=runname,alldata=alldata_sub,outfolder=outfolder)
	print 'PLOTTING MASS-METALLICITY DIAGRAM'
	delta_mass_met.plot_comparison(runname=runname,alldata=alldata_sub,alldata_noagn=alldata_noagn,outfolder=outfolder)
	#print 'PLOTTING OPTICAL COLOR COLOR DIAGRAM'
	#optical_color_color.plot(runname=runname,alldata=alldata_sub,outfolder=outfolder)
	print 'PLOTTING DELTA PARS'
	plot_delta_pars.plot(runname=runname,runname_noagn=runname_noagn,alldata=alldata_sub,alldata_noagn=alldata_noagn,outfolder=outfolder)
	print 'PLOTTING PROPERTY COMPARISON'
	property_comparison.plot_comparison(runname=runname,runname_noagn=runname_noagn,alldata=alldata_sub,alldata_noagn=alldata_noagn,outfolder=outfolder)
	print 'PLOTTING XRAY LUMINOSITY'
	xray_luminosity.make_plot(runname=runname,alldata=alldata_sub,outfolder=outfolder)
	print 'PLOTTING DELTA OBSERVABLES'
	plot_spec_rms.plot_comparison(runname=runname,alldata=alldata_sub,alldata_noagn=alldata_noagn,outfolder=outfolder)

	### check out what you've made
	if open_all:
		os.system('open '+outfolder+'*.png')