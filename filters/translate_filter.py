import calc_ml
import numpy as np

def read_translate(field):
	
	trans_filename=field.lower()+"_3dhst.v4.1.translate"
	dat = np.loadtxt(trans_filename, dtype=np.dtype('|S16'))
	
	return dat

def read_threedhst_filters(filtnum):
	'''READS FILTER RESPONSE CURVES FOR FAST FILTERS'''
	
	if filtnum == 0:
		print "ERROR"
		print 1/0
	
	from astropy.io import ascii
	import numpy as np
	
	filter_response_curve = 'FILTER.RES.latest'

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

def main(field=False):

	''' TRANSLATES FAST FILTER DEFINITIONS TO FSPS '''
	if not field:
		#field = ['COSMOS']
		field = ['AEGIS','COSMOS','GOODS-N','GOODS-S','UDS']
	
	
	filters=[]
	for kk in xrange(0,len(field)):
	
		''' LOAD FAST FILTER DEFINITIONS '''
		translate=read_translate(field[kk])
		ntemp_filters = len(translate[:,0])/2
		temp_filters = [dict() for x in range(ntemp_filters)]
		print('Translating {0} filters for {1}'.format(ntemp_filters, field[kk]))
		for jj in xrange(0,ntemp_filters):
			temp_filters[jj]['filt_name'] = "_".join(translate[2*jj][0].split('_')[1:])+'_'+field[kk]
			print ('\t Translating {0}'.format(temp_filters[jj]['filt_name']))
			if temp_filters[jj]['filt_name'] == 'UVISTA_COSMOS':
				print 1/0
			line_num = translate[2*jj][1][1:]
			temp_filters[jj]['lam'], temp_filters[jj]['response'] = read_threedhst_filters(int(line_num))
		filters = filters + temp_filters
		
	''' WRITE OUT FSPS FILTER DEFINITIONS '''
	nfilters = len(filters)
	print('{0} total filters'.format(nfilters))
	outfile="allfilters_threedhst.dat"
	with open(outfile, 'w') as f:
		for jj in xrange(nfilters):
			f.write('# ' + filters[jj]['filt_name']+'\n')
			for ll in xrange(len(filters[jj]['lam'])):
				f.write(str(filters[jj]['lam'][ll])+'\t'+str(filters[jj]['response'][ll])+'\n')
		
	''' WRITE OUT FILTER KEYS '''
	outfile='filter_keys_threedhst.txt'
	with open(outfile, 'w') as f:
		for jj in xrange(nfilters): f.write(str(jj+1).strip()+'\t'+filters[jj]['filt_name'].strip()+'\n')
	
	''' WRITE OUT EFFECTIVE WAVELENGTHS '''
	outfile='lameff_threedhst.txt'
	with open(outfile,'w') as f:
		for jj in xrange(nfilters):
			lameff = calc_lameff(filters[jj]['lam'],filters[jj]['response'])
			f.write("{0}\n".format(lameff))
if __name__ == "__main__":
    main()