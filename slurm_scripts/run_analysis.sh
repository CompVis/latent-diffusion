#!/bin/bash
#SBATCH -p <partition_name> 
#SBATCH --gres gpu:<num_gpus>
#SBATCH --nodes <num_nodes> 
#SBATCH --ntasks-per-node <tasks_per_node>
#SBATCH -t <days>-<hours>:<minutes>:<seconds>

# activate Conda environment
source ~/.bashrc
conda activate ldm

cd ..

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

# print start time
echo $(date)
echo $SLURM_NTASKS
echo $MASTER_ADDR
echo $MASTER_PORT

# start command
srun python3 analysis.py
srun python3 analysis_wav.py
