import numpy as np
from prosp_dutils import asym_errors

def halpha_cuts(e_pinfo,sn=False):

	sn_ha = np.abs(e_pinfo['obs']['f_ha'][:,0] / e_pinfo['obs']['err_ha'])
	sn_hb = np.abs(e_pinfo['obs']['f_hb'][:,0] / e_pinfo['obs']['err_hb'])

	if sn:
		sn_cut = sn
	else:
		sn_cut = e_pinfo['obs']['sn_cut']

	keep_idx = np.squeeze((sn_ha > sn_cut) & \
		                  (sn_hb > sn_cut) & \
		                  (e_pinfo['obs']['eqw_ha'][:,0] > e_pinfo['obs']['eqw_cut']) & \
		                  (e_pinfo['obs']['eqw_hb'][:,0] > e_pinfo['obs']['eqw_cut']) & \
		                  (e_pinfo['obs']['f_ha'][:,0] > 0) & \
		                  (e_pinfo['obs']['f_hb'][:,0] > 0) & \
		                  (e_pinfo['prosp']['cloudy_ha'][:,0] > 0))
	print 'halpha cuts returned {0} galaxies'.format(keep_idx.sum())
	return keep_idx

def hdelta_cuts(e_pinfo, eqw=False):

	if eqw:
		hdel_sn = np.abs((e_pinfo['obs']['hdel_eqw'][:,0]/ e_pinfo['obs']['hdel_eqw_err']))
	else:
		hdel_sn = np.abs((e_pinfo['obs']['hdel'][:,0]/ e_pinfo['obs']['hdel_err']))

	### define limits
	# need this for stupid reasons in pdf_dist
	good_idx = (hdel_sn > e_pinfo['obs']['hdelta_sn_cut']) & \
			   (e_pinfo['obs']['hdel_eqw'][:,0] > e_pinfo['obs']['hdelta_eqw_cut']) & \
	           (e_pinfo['obs']['hdel'][:,0] > 0) & \
	           (e_pinfo['prosp']['hdel_eqw_elineon_marg'][:,0] > 0)
	print 'hdelta cuts returned {0} galaxies'.format(good_idx.sum())

	return good_idx

def dn4000_cuts(e_pinfo):

	dn_idx = e_pinfo['obs']['dn4000'] > 0.5
	return dn_idx

def load_atlas3d(e_pinfo):

	#### load up atlas3d info
	dtype={'names': ('name', 'hbeta_ang','hbeta_ang_err', 'fe5015_ang','fe5015_ang_err','mgb_ang','mgb_ang_err','fe5270_ang','fe5270_ang_err','age_ssp','age_ssp_err','z_h_ssp','z_h_ssp_err','a_fe_ssp','a_fe_ssp_err','quality'), \
	       'formats': ('S16', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i4')}
	a3d = np.loadtxt('/Users/joel/code/python/prospector_alpha/data/brownseds_data/atlas_3d_abundances.dat',dtype=dtype,comments='#')

	#### matching
	objnames = e_pinfo['objnames']
	a3d_met,a3d_met_err, a3d_alpha, prosp_met,prosp_met_errup,prosp_met_errdo = [[] for i in range(6)]
	obj_idx = np.zeros(objnames.shape[0],dtype=bool)
	for i, obj in enumerate(objnames): 
		match = a3d['name'] == obj.replace(' ','')
		
		# no match
		if np.sum(match) == 0:
			continue

		# match, save metallicity
		prosp_met.append(e_pinfo['prosp']['met'][i,0])
		prosp_met_errup.append(e_pinfo['prosp']['met'][i,1])
		prosp_met_errdo.append(e_pinfo['prosp']['met'][i,2])
		a3d_met.append(a3d['z_h_ssp'][match])
		a3d_met_err.append(a3d['z_h_ssp_err'][match])
		a3d_alpha.append(a3d['a_fe_ssp'][match])
		obj_idx[i] = True

	prosp_met = np.array(prosp_met)
	prosp_met_err = asym_errors(prosp_met, prosp_met_errup,prosp_met_errdo,log=False)
	a3d_met = np.array(a3d_met)
	a3d_met_err = np.array(a3d_met_err)
	a3d_alpha = np.array(a3d_alpha)

	return prosp_met, prosp_met_err, a3d_met, a3d_met_err, a3d_alpha, obj_idx