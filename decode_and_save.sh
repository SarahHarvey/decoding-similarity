#!/bin/bash
#SBATCH --job-name=trainnetwork          # Job name
#SBATCH --output=.out/result_%j.out      # Standard output file (%j expands to jobId)
#SBATCH --error=.out/result_%j.err       # Standard error file
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=10                # Number of CPU cores per task
#SBATCH --mem=256GB                      # Memory per node
#SBATCH --time=0-12:00:00                # Time limit days-hrs:min:sec


MODELNAME=${1:-resnet50}   # Default to resnet50 if not provided
IMAGENET_SAMPLE=${2:-imagenet_random_sample_5000_v1}  # Default to imagenet_random_sample_5000_v1 if not provided
P=${3:-5001}  # Default to 5001 dimensions if not provided
KERNEL=${4:-linear}  # Default to linear kernel if not provided

module load python
srun hostname

# Build the command with optional flags
CMD="python decode_and_save.py --model_name $MODELNAME --imagenet_sample $IMAGENET_SAMPLE --P $P --kernel $KERNEL"

srun $CMD