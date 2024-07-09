import os
import numpy as np

CPUs = 24
job_file = "ite_noise_loop_submission_job.sbatch"
with open(job_file, "w") as fh:
    fh.writelines("#!/bin/bash\n")
    fh.writelines("#SBATCH -t 24:00:00\n")
    fh.writelines("#SBATCH --nodes=1\n")
    fh.writelines("#SBATCH --ntasks-per-node=1\n") #half of available CPUs for a given partition; CPU allocation is x2 of this number for swift, biocrunch, x4 for legion, x1 for speedy
    fh.writelines("#SBATCH --cpus-per-task="+str(CPUs)+"\n")  #half of available CPUs for a given partition; CPU allocation is x2 of this number for swift, biocrunch, x4 for legion, x1 for speedy
    fh.writelines("#SBATCH --mem-per-cpu=4G\n")
    fh.writelines("#SBATCH --partition=dense\n")
    fh.writelines("#SBATCH --hint=compute_bound\n")
    fh.writelines("#SBATCH -o /home/akhin/ite_noise_code_for_paper_test/dmrg/dmrg_out\n")
    fh.writelines("#SBATCH -e /home/akhin/ite_noise_code_for_paper_test/dmrg/dmrg_err\n")
    fh.writelines("#SBATCH --job-name=dmrg\n")
    fh.writelines("#SBATCH --mail-user=akhin@iastate.edu\n")
    fh.writelines("#SBATCH --mail-type=FAIL\n")
    fh.writelines("export OMP_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
    fh.writelines("export OPENBLAS_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
    fh.writelines("export MKL_NUM_THREADS=$SLURM_JOB_CPUS_PER_NODE\n")
    fh.writelines("julia /home/akhin/ite_noise_code_for_paper_test/dmrg/dmrg_wrapper.jl")

"running the job"
os.system("sbatch %s" %job_file)

os.remove(job_file)
