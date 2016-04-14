import os
import numpy as np
from astropy.io import ascii
from calc_ml import load_filter_response

def read_translate(field):
	
	trans_filename=os.getenv('APPS')+'/threedhst_bsfh/filters/translate/'+field.lower()+"_3dhst.v4.1.translate"
	dat = np.loadtxt(trans_filename, dtype=np.dtype('|S16'))
	
	return dat

def read_threedhst_filters(filtnum):
	'''READS FILTER RESPONSE CURVES FOR FAST FILTERS'''
	
	if filtnum == 0:
		print "ERROR"
		sys.exit()
		
	filter_response_curve = '/Users/joel/code/python/threedhst_bsfh/filters/FILTER.RES.latest'

	# initialize output arrays
	lam,res = (np.zeros(0) for i in range(2))

	# open file
	with open(filter_response_curve, 'r') as f:

    	# skip to correct filter
		for jj in xrange(filtnum-1):
    		
			# find how many lines are next
			line = f.readline()
			data = line.split()
			nlines = data[0]

			# skip that many lines
			for kk in xrange(int(nlines)): f.readline()
    	
    	# how many lines long is the filter definition?
		line = f.readline()
		data = line.split()
		nlines = data[0]
    	
		# Reads text until we hit the next filter
		for kk in xrange(int(nlines)):
			
			# read line, extract data
			line = f.readline()
			data_inloop = line.split()
			lam = np.append(lam,float(data_inloop[1]))
			res = np.append(res,float(data_inloop[2]))

	return lam, res


def calc_lameff(lam,res):

	''' calculates effective lambda given input response curve'''
	dellam = lam[1]-lam[0]
	lam_sum_up = np.sqrt(np.sum(lam*res*dellam)/np.sum(res*dellam/lam))
	return lam_sum_up

def calc_lameff_for_fsps(fsps_filter_list):

	'''
	read in FSPS filter names
	loop over FSPS filters
		load effective WAVELENGTHS
		calculate lameff
	write out to file
	'''
	lameff = np.empty(0)
	for filter in fsps_filter_list:
		lam,res = load_filter_response(filter, alt_file=None)
		lameff = np.append(lameff,calc_lameff(lam,res))

	return lameff

def translate_fsps_to_sedpy(fsps_filtname,sedpy_filtname,
					        outfolder='/Users/joel/code/python/sedpy/sedpy/data/filters'):

	'''
	used to translate FSPS filter curves into sedpy-style filter files
	INPUT: FSPS filter name
	OUTPUT: sedpy-style filter definition file at outfolder/sedpy_filtname.par
	'''
	lam,res = load_filter_response(fsps_filtname, alt_file=None)

	outfile=outfolder+'/'+sedpy_filtname+'.par'
	with open(outfile, 'w') as f:
		for l,r in zip(lam,res): f.write("{:.6e}".format(l)+'  '+"{:.6e}".format(r)+'\n')

	print 'created '+outfile

def main(field=False):

	''' TRANSLATES FAST FILTER DEFINITIONS TO FSPS '''
	if not field:
		field = ['AEGIS','COSMOS','GOODS-N','GOODS-S','UDS']
	
	
	filters=[]
	for kk in xrange(0,len(field)):
	
		''' LOAD FAST FILTER DEFINITIONS '''
		translate=read_translate(field[kk])
		ntemp_filters = len(translate[:,0])/2
		temp_filters = [dict() for x in range(ntemp_filters)]
		print('Translating {0} filters for {1}'.format(ntemp_filters, field[kk]))
		for jj in xrange(ntemp_filters):
			temp_filters[jj]['filt_name'] = "_".join(translate[2*jj][0].split('_')[1:])+'_'+field[kk]
			print ('\t Translating {0}'.format(temp_filters[jj]['filt_name']))
			line_num = translate[2*jj][1][1:]
			temp_filters[jj]['lam'], temp_filters[jj]['response'] = read_threedhst_filters(int(line_num))
		filters = filters + temp_filters
		
		''' ADD IN EXTRA FILTER DEFINITIONS RIPPED FROM FSPS'''
		fsps_filters = ['MIPS 24um']
		temp_filters = [dict() for x in range(len(fsps_filters))]
		for jj in xrange(len(fsps_filters)):
			temp_filters[jj]['filt_name']="_".join(fsps_filters[jj].split(' '))+'_'+field[kk]
			temp_filters[jj]['lam'],temp_filters[jj]['response']=load_filter_response(fsps_filters[jj], 
			                     alt_file='/Users/joel/code/python/threedhst_bsfh/filters/extra_filters.txt')
			print ('\t Translating {0}'.format(temp_filters[jj]['filt_name']))		
		filters = filters + temp_filters
	
	''' SORT BY EFFECTIVE WAVELENGTHS '''
	nfilters = len(filters)
	lameff=np.array([])
	for jj in xrange(nfilters):
		lameff = np.append(lameff,calc_lameff(filters[jj]['lam'],filters[jj]['response']))
	points = zip(lameff,filters)
	points.sort(key=lambda tup: tup[0])	
	
	lameff = [point[0] for point in points]
	filters = [point[1] for point in points]
	
	''' WRITE OUT FSPS FILTER DEFINITIONS '''
	print('{0} total filters'.format(nfilters))
	outfile="allfilters_threedhst.dat"
	with open(outfile, 'w') as f:
		for jj in xrange(nfilters):
			f.write('# ' + filters[jj]['filt_name']+'\n')
			for ll in xrange(len(filters[jj]['lam'])):
				f.write(str(filters[jj]['lam'][ll])+'\t'+str(filters[jj]['response'][ll])+'\n')
		
	''' WRITE OUT FILTER KEYS '''
	outfile=os.getenv('APPS')+'/threedhst_bsfh/filters/filter_keys_threedhst.txt'
	with open(outfile, 'w') as f:
		for jj in xrange(nfilters): f.write(str(jj+1).strip()+'\t'+filters[jj]['filt_name'].strip()+'\n')
	
	''' WRITE OUT EFFECTIVE WAVELENGTHS '''
	outfile=os.getenv('APPS')+'/threedhst_bsfh/filters/lameff_threedhst.txt'
	with open(outfile,'w') as f:
		for jj in xrange(nfilters): f.write("{0}\n".format(lameff[jj]))
		
if __name__ == "__main__":
    main()