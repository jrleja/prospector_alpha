import wise_colors
import bpt


def plot(runname='brownseds_agn',alldata=None,open_all=True,outfolder=none):

	#### load alldata
	if alldata is None:
		import brown_io
		alldata = brown_io.load_alldata(runname=runname)

	#### outfolder
	outfolder = os.getenv('APPS')+'/threedhst_bsfh/plots/'+runname+'/agn_plots/'
	if not os.path.isdir(outfolder):
		os.makedirs(outfolder)

	#### PLOT ALL
	wise_colors.plot_mir_colors(runname=runname,alldata=alldata,outfolder=outfolder)
	bpt.plot_bpt(runname=runname,alldata=alldata,outfolder=outfolder)

	### check out what you've made
	if open_all:
		os.system('open '+outfolder+'*.png')