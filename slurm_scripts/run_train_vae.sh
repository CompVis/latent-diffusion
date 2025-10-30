#!/bin/bash
#SBATCH -p <partition_name> 
#SBATCH --gres gpu:<num_gpus>
#SBATCH --nodes <num_nodes> 
#SBATCH --ntasks-per-node <tasks_per_node>
#SBATCH -t <days>-<hours>:<minutes>:<seconds>
#SBATCH -o train_vae_output_%j.log
#SBATCH -e train_vae_error_%j.log

# activate Conda environment
source ~/.bashrc
conda activate ldm

cd ..

cmd="python3 main.py --base configs/autoencoder/autoencoder_kl_16x16x16.yaml \
 --num_nodes $SLURM_JOB_NUM_NODES --max_ep 120 \
 --name ch1-1-2-2-4_baseline \
 --resume logs/ch1-1-2-2-4_baseline/last.ckpt --wandb_id <wandb_run_id>"

# print start time and command to log
echo $(date)
echo $cmd

# start command
srun $cmd
