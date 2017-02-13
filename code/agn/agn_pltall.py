import wise_colors,optical_color_color,bpt,plot_delta_pars,property_comparison,xray_luminosity,plot_spec_rms
import os
import brown_io

def plot(runname='brownseds_agn',runname_noagn='brownseds_np',
	     alldata=None,alldata_noagn=None,open_all=True,outfolder=None):

	#### load alldata
	if alldata is None:
		alldata = brown_io.load_alldata(runname=runname)
	if alldata_noagn is None:
		alldata_noagn = brown_io.load_alldata(runname=runname_noagn)

	#### outfolder
	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/'
	if not os.path.isdir(outfolder):
		os.makedirs(outfolder)

	#### PLOT ALL
	print 'PLOTTING WISE COLORS'
	wise_colors.plot_mir_colors(runname=runname,alldata=alldata,outfolder=outfolder)
	print 'PLOTTING BPT DIAGRAM'
	bpt.plot_bpt(runname=runname,alldata=alldata,outfolder=outfolder)
	print 'PLOTTING OPTICAL COLOR COLOR DIAGRAM'
	optical_color_color.plot(runname=runname,alldata=alldata,outfolder=outfolder)
	print 'PLOTTING DELTA PARS'
	plot_delta_pars.plot(runname=runname,runname_noagn=runname_noagn,alldata=alldata,alldata_noagn=alldata_noagn,outfolder=outfolder)
	print 'PLOTTING PROPERTY COMPARISON'
	property_comparison.plot_comparison(runname=runname,runname_noagn=runname_noagn,alldata=alldata,alldata_noagn=alldata_noagn,outfolder=outfolder)
	print 'PLOTTING XRAY LUMINOSITY'
	xray_luminosity.make_plot(runname=runname,alldata=alldata,outfolder=outfolder)
	print 'PLOTTING DELTA OBSERVABLES'
	plot_spec_rms.plot_comparison(runname=runname,alldata=alldata,alldata_noagn=alldata_noagn,outfolder=outfolder)

	### check out what you've made
	if open_all:
		os.system('open '+outfolder+'*.png')