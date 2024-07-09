
import os
import numpy as np

g_array = [round(g,3) for g in np.linspace(0.4,1.4,21)]

"creating sbatch job to run for each parameter value"
CPUs = 24
for g in g_array:
    job_file = "ite_noise_loop_submission_job.sbatch"
    with open(job_file, "w") as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH -t 20:00:00\n")
        fh.writelines("#SBATCH --nodes=1\n")
        fh.writelines("#SBATCH --ntasks-per-node=1\n") #half of available CPUs for a given partition; CPU allocation is x2 of this number for swift, biocrunch, x4 for legion, x1 for speedy
        fh.writelines("#SBATCH --cpus-per-task="+str(CPUs)+"\n")  #half of available CPUs for a given partition; CPU allocation is x2 of this number for swift, biocrunch, x4 for legion, x1 for speedy
        fh.writelines("#SBATCH --mem-per-cpu=4G\n")
        fh.writelines("#SBATCH --partition=volta\n")
        fh.writelines("#SBATCH --hint=compute_bound\n")
        fh.writelines("#SBATCH -o /home/akhin/ite_noise_code_for_paper_test/ite_noise_simulation_density_operator_gpu/ite_noise_simulation_out\n")
        fh.writelines("#SBATCH -e /home/akhin/ite_noise_code_for_paper_test/ite_noise_simulation_density_operator_gpu/ite_noise_simulation_out\n")
        fh.writelines("#SBATCH --job-name=ite_noise_simulation\n")
        fh.writelines("#SBATCH --mail-user=akhin@iastate.edu\n")
        fh.writelines("#SBATCH --mail-type=FAIL\n")
        fh.writelines("export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        fh.writelines("export OPENBLAS_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        fh.writelines("export MKL_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
        fh.writelines("python /home/akhin/ite_noise_code_for_paper_test/ite_noise_simulation_density_operator_gpu/ite_noise_simulation_wrapper.py %s" % g)

    "running the job"
    os.system("sbatch %s" %job_file)

os.remove(job_file)
