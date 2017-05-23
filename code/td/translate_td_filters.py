import os
import numpy as np

APPS = os.getenv('APPS')

def generate_resp_curves_from_photometry_file(photfile=APPS+'/threedhst_bsfh/data/3dhst/COSMOS_td_massive.cat',
	                                          outfolder=APPS+'/sedpy/sedpy/data/filters'):

	'''
	given a 3D-HST photometry file
	generate sedpy filter response files
	'''

	### load filter names
	with open(photfile, 'r') as f:
		hdr = f.readline().split()
	dtype = np.dtype([(hdr[1],'S20')] + [(n, np.float) for n in hdr[2:]])
	filters = np.array([f[2:] for f in hdr if f[0:2] == 'f_'])

	### write out sedpy filters
	for f in filters:

		### load 3D-HST filter response
		# pull out field and filter name, then load translate file for that field
		field, filtname = f.split('_')[-1], "_".join(f.split('_')[:-1])
		trans = read_translate(field)

		# find filter number in FILTER.RES.LATEST
		# we have a few more matches than we need here, ugly
		try: # if it's in the EAZY filter file, pull it!
			match = np.array(['f_'+filtname == t.lower() for t in trans[:,0]],dtype=bool)
			filtnum = int(trans[match,1][0][1:])
			lam, res = read_threedhst_filters(filtnum)
		except IndexError: # it's not in the EAZY filter file. pull it from FSPS.
			if filtname == 'mips_24um':
				fsps_filtname = 'MIPS 24um'
			else:
				print 'need to figure out what '+filtname+' is named in the FSPS file!'
				print 1/0

			lam, res = load_fsps_filter(fsps_filtname)

		### now write to sedpy format
		outfile=outfolder+'/'+f+'.par'
		with open(outfile, 'w') as f:
			for l,r in zip(lam,res): f.write("{:.6e}".format(l)+'  '+"{:.6e}".format(r)+'\n')
		print 'created '+outfile

def read_translate(field):
	
	trans_filename=os.getenv('APPS')+'/threedhst_bsfh/filters/translate/'+field.lower()+"_3dhst.v4.1.translate"
	dat = np.loadtxt(trans_filename, dtype=np.dtype('|S16'))
	
	return dat

def read_threedhst_filters(filtnum):
	'''READS FILTER RESPONSE CURVES FOR FAST FILTERS'''
	
	if filtnum == 0:
		print "ERROR"
		sys.exit()
		
	filter_response_curve = APPS+'/threedhst_bsfh/filters/FILTER.RES.latest'

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

def load_fsps_filter(filter, alt_file=None):
	'''READS FILTER RESPONSE CURVES FOR FSPS'''
		
	if not alt_file:
		filter_response_curve = os.getenv('SPS_HOME')+'/data/allfilters.dat'
	else:
		filter_response_curve = alt_file

	# initialize output arrays
	lam,res = (np.zeros(0) for i in range(2))

	# upper case?
	if filter.lower() == filter:
		lower_case = True
	else:
		lower_case = False

	# open file
	with open(filter_response_curve, 'r') as f:
    	# Skips text until we find the correct filter
		for line in f:
			if lower_case:
				if line.lower().find(filter) != -1:
					break
			else:
				if line.find(filter) != -1:
					break
		# Reads text until we hit the next filter
		for line in f:  # This keeps reading the file
			if line.find('#') != -1:
				break
			# read line, extract data
			data = line.split()
			lam = np.append(lam,float(data[0]))
			res = np.append(res,float(data[1]))

	if len(lam) == 0:
		print "Couldn't find filter " + filter + ': STOPPING'
		print 1/0

	return lam, res

def translate_txt_to_sedpy(txt_name,sedpy_filtname,
					       outfolder='/Users/joel/code/python/sedpy/sedpy/data/filters'):

	'''
	used to translate text filter curves into yanny-style filter files
	INPUT: text filter file
	OUTPUT: sedpy-style filter definition file at outfolder/sedpy_filtname.par
	'''
	dat = np.loadtxt(txt_name, comments = '#',
					 dtype = {'names':(['lambda','transmission']),
					         'formats':('f16','f16')})

	outfile=outfolder+'/'+sedpy_filtname+'.par'
	with open(outfile, 'w') as f:

		# header
		f.write('\n')
		f.write('typedef struct {\n')
		f.write('  double lambda;\n')
		f.write('  double pass;\n')
		f.write('} KFILTER;\n')
		f.write('\n')

		# data
		for l,t in zip(dat['lambda'],dat['transmission']): f.write('KFILTER  '+"{:.1f}".format(l*1e4)+'  '+"{:.6f}".format(t)+'\n')

	print 'created '+outfile

def translate_fsps_to_sedpy(fsps_filtname,sedpy_filtname,
					        outfolder='/Users/joel/code/python/sedpy/sedpy/data/filters'):

	'''
	used to translate FSPS filter curves into sedpy-style filter files
	INPUT: FSPS filter name
	OUTPUT: sedpy-style filter definition file at outfolder/sedpy_filtname.par
	'''
	lam,res = load_fsps_filter(fsps_filtname, alt_file=None)

	outfile=outfolder+'/'+sedpy_filtname+'.par'
	with open(outfile, 'w') as f:
		for l,r in zip(lam,res): f.write("{:.6e}".format(l)+'  '+"{:.6e}".format(r)+'\n')

	print 'created '+outfile