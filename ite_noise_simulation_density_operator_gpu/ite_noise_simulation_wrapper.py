
"this code performs an example of the noisy ITE  saving the output in the .jls format"

import pickle
import sys
import ite_noise_simulation_main_class
from ite_noise_simulation_main_class import ite_noise_cp

"setting parameters for the calculation (see ite_noise_simulation_main_class.py for a description of paraemters)"
L = 11
g = float(sys.argv[1])
p = 0.01
dtau = 0.1
n_steps = 200
noise_type = '1q_local_X'
BC = 'open'

'initializing an instance of the class'
ite_noise_obj = ite_noise_cp(L=L)

'performing the calculation'
obs_avg_std = ite_noise_obj.ite_many_runs(g=g, p=p, dtau=dtau, n_steps=n_steps, noise_type=noise_type, BC=BC)

'saving data'
with open('data/ite_noise_rho_L%s_g%s_p%s_dtau%s_nsteps%s_noise%s_%s_DIS.pkl' % (L,g,p,dtau,n_steps,noise_type,BC) , 'wb') as outp:
    pickle.dump(obs_avg_std, outp, pickle.HIGHEST_PROTOCOL)
