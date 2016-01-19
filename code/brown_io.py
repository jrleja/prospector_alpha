import pickle

#### where do alldata pickle files go?
outpickle = '/Users/joel/code/magphys/data/pickles'

def save_spec_cal(spec_cal,runname='brownseds'):
	output = outpickle+'/spec_calibration.pickle'
	pickle.dump(spec_cal,open(output, "wb"))

def load_spec_cal(runname='brownseds'):
	with open(outpickle+'/spec_calibration.pickle', "rb") as f:
		spec_cal=pickle.load(f)
	return spec_cal

def load_alldata(runname='brownseds'):

	output = outpickle+'/'+runname+'_alldata.pickle'
	with open(output, "rb") as f:
		alldata=pickle.load(f)
	return alldata

def save_alldata(alldata,runname='brownseds'):

	output = outpickle+'/'+runname+'_alldata.pickle'
	pickle.dump(alldata,open(output, "wb"))