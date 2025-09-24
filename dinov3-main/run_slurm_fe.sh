#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100_full
#SBATCH --nodelist=agpu004

#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR

#SBATCH --mail-type=ALL
#SBATCH --mail-user=r02sw23@abdn.ac.uk
#SBATCH --time=24:00:00

nvidia-smi
module load anaconda3
source activate dinov3

srun python dinov3FE_images.py