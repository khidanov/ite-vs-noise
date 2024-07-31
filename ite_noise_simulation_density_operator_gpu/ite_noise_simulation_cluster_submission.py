"""
This script generates job scripts to run "ite_noise_simulation_run.py" on
multiple computing nodes on a cluster.
The parameters for "ite_noise_simulation_run.py" are specified in the
"set_of_params" list below.
Separate job is created and executed for each parameter set.
"""

import os
import numpy as np

"""
Specifying parameter values for "ite_noise_simulation_run.py"
"""
set_of_params=[(L,
                round(g,3),
                p,
                dtau,
                n_steps,
                noise_type,
                BC)
                for L in [10]
                for g in [0.1]
                for p in [0.01]
                for dtau in [0.1]
                for n_steps in [200]
                for noise_type in ['1q_local_X']
                for BC in ['open']
                ]

# set_of_params=[(L,
#                 round(g,3),
#                 p,
#                 dtau,
#                 n_steps,
#                 noise_type,
#                 BC)
#                 for num_qubits in [10]
#                 for g in np.linspace(0.4,1.4,21)
#                 for p in [0.01]
#                 for dtau in [0.1]
#                 for n_steps in [200]
#                 for noise_type in ['1q_local_X']
#                 for BC in ['open']
#                 ]


file_dir = os.path.dirname(os.path.abspath(__file__))

"""
For each parameter value, creating an sbatch job to execute.
Execution logs are saved into a directory logs/.
For GPU calculations, the specified partition should be a GPU partition.
"""
CPUs = 24
for params in set_of_params:
    (L,
    g,
    p,
    dtau,
    n_steps,
    noise_type,
    BC) = params
    job_file = "ite_noise_simulation_cluster_submission_job.sbatch"
    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -t 20:00:00\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --ntasks-per-node=1\n")
        fh.writelines("#SBATCH --cpus-per-task="+str(CPUs)+"\n")
        fh.writelines("#SBATCH --mem-per-cpu=4G\n")
        fh.writelines("#SBATCH --partition=volta\n") #specify partition
        fh.writelines("#SBATCH --hint=compute_bound\n")
        fh.writelines("#SBATCH -o " + file_dir + "/ite_noise_simulation_out\n")
        fh.writelines("#SBATCH -e " + file_dir + "/ite_noise_simulation_err\n")
        fh.writelines("#SBATCH --job-name=ite_noise_simulation\n")
        # fh.writelines("#SBATCH --mail-user=\n")
        # fh.writelines("#SBATCH --mail-type=FAIL\n")
        fh.writelines("export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        fh.writelines("export OPENBLAS_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        fh.writelines("export MKL_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        fh.writelines("{ time python " +
                        file_dir +
                        ("/ite_noise_simulation_run.py "
                        "-L %s "
                        "-g %s "
                        "-p %s "
                        "-dt %s "
                        "--n_steps %s "
                        "--noise_type %s "
                        "-bc %s "
                        "--num_cpus %s ")
                        % (params + (CPUs,)) +
                        (" ; } 2> logs/time_ite_noise_rho_L%s_g%s_p%s_dtau%s_"
                        "nsteps%s_noise%s_%s.txt \n") % params
                    )
    """
    Executing the job.
    """
    os.system("sbatch %s" %job_file)

"""
Deleting the .sbatch file in the end.
"""
os.remove(job_file)
