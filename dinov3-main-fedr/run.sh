#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --nodelist=hmem011

#SBATCH -o slurm.%j.out # STDOUT
#SBATCH -e slurm.%j.err # STDERR

#SBATCH --mail-type=ALL
#SBATCH --mail-user=r02sw23@abdn.ac.uk

#SBATCH --time=24:00:00

module load anaconda3

source activate SMP

srun python segemntation_inference_dinov3.py