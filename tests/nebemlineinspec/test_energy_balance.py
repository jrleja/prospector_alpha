import td_huge_params as pfile
from sedpy import observate
import numpy as np

# load up the machinery
run_params = pfile.run_params
obs = pfile.load_obs(**run_params)
sps = pfile.load_sps(**run_params)
mod = pfile.load_model(**run_params)

# load up a couple of Herschel filters
hfilters = ['herschel_pacs_70', 'herschel_pacs_160', 'herschel_spire_350']
obs['filters'] = observate.load_filters(hfilters)
obs['phot_mask'] = np.ones_like(obs['filters'],dtype=bool)

# test the difference with and without nebemlineinspec
mod.params['nebemlineinspec'] = np.atleast_1d(False)
spec,mags_false,sm = mod.mean_model(mod.initial_theta, obs, sps=sps)
mod.params['nebemlineinspec'] = np.atleast_1d(True)
spec,mags_true,sm = mod.mean_model(mod.initial_theta, obs, sps=sps)
print (mags_true-mags_false)/mags_false