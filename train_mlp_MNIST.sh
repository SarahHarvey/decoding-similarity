#!/bin/bash
#SBATCH --job-name=trainnetwork          # Job name
#SBATCH --output=.out/result_%j.out      # Standard output file (%j expands to jobId)
#SBATCH --error=.out/result_%j.err       # Standard error file
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem=2GB                        # Memory per node
#SBATCH --time=0-01:00:00                # Time limit days-hrs:min:sec
#SBATCH --gres=gpu:1                     # Request 1 GPU
#SBATCH --partition=gpu                  # GPU partition 

N_HIDDEN=${1:-6}   # Default to 6 if not provided
N_PROBES=${2:-1000}  # Default to 1000 if not provided

module load python
srun hostname

# Build the command with optional flags
CMD="python train_mlp_MNIST.py --n_hidden $N_HIDDEN --n_probes $N_PROBES"

# Add --seed if $3 is provided
if [ -n "$3" ]; then
    CMD="$CMD --seed $3"
fi

srun $CMD