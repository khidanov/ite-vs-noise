"""
This script utilizes module "ite_noise_simulation_main_class.py" to
perform noisy Trotterized ITE simulations for the 1D TFIM.

Running "python ite_noise_simulation_run.py" would run the code on the head node
with default parameters (see below).
(Note that in the case of GPU simulations the head node should be a GPU node.)
Otherwise, the parameters can be specified in the command line or in the job
generating script to run the code on computing nodes
(see "ite_noise_simulation_cluster_submission.py").
"""

import argparse
import pickle
import os

import ite_noise_simulation_main_class
from ite_noise_simulation_main_class import ITE_noise_cp


def none_or_str(value):
    """
    Function used to assign the type None to a command line variable if the
    command line reads "None", str otherwise.
    """
    if value == 'None':
        return None
    return value


"""
Reading out parameters from the command line.
"""
parser = argparse.ArgumentParser(
    description = "Perform noisy Trotterized ITE simulations for 1D TFIM"
)
parser.add_argument(
    "-L",
    "--L",
    type = int,
    default = 10,
    metavar = '\b',
    help = "number of sites"
)
parser.add_argument(
    "-g",
    "--g",
    type = float,
    default = 0.1,
    metavar = '\b',
    help = "transverse field"
)
parser.add_argument(
    "-p",
    "--p",
    type = float,
    default = 0.01,
    metavar = '\b',
    help = "noise strength (error probability)"
)
parser.add_argument(
    "-dt",
    "--dtau",
    type = float,
    default = 0.1,
    metavar = '\b',
    help = "imaginary time Trotter step size"
)
parser.add_argument(
    "--n_steps",
    type = int,
    default = 200,
    metavar = '\b',
    help = "number of imaginary time steps"
)
parser.add_argument(
    "--noise_type",
    type = none_or_str,
    choices = [None,
        '1q_local_X',
        '1q_local_Y',
        '1q_local_Z',
        '1q_local_depolarizing',
        '1q_local_AD',
        '2q_local_XX',
        '2q_local_YY',
        '2q_local_ZZ'
        ],
    default = '1q_local_X',
    help = "noise type"
)
parser.add_argument(
    "-bc",
    "--BC",
    type = str,
    choices = ['open', 'periodic'],
    default = 'open',
    metavar='\b',
    help = "boubdary conditions"
)
parser.add_argument(
    "--num_cpus",
    type = int,
    default = 1,
    metavar = '\b',
    help = "number of CPUs"
)
args = parser.parse_args()


"""
Initializing an instance of the class.
"""
ite_noise_obj = ITE_noise_cp(args.L)

"""
Performing noisy ITE.
"""
u4_data = ite_noise_obj.ite_density_matrix(
    args.g,
    args.p,
    args.dtau,
    args.n_steps,
    args.noise_type,
    args.BC,
)

file_dir = os.path.dirname(os.path.abspath(__file__))

"""
Saving u4_data to a file.
"""
with open(file_dir +
    '/data/ite_noise_rho_L%s_g%s_p%s_dtau%s_nsteps%s_noise%s_%s.pkl' % (
        args.L,
        args.g,
        args.p,
        args.dtau,
        args.n_steps,
        args.noise_type,
        args.BC,
        ), 'wb') as outp:
    pickle.dump(u4_data, outp, pickle.HIGHEST_PROTOCOL)
