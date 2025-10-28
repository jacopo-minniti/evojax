#!/bin/bash
#SBATCH --job-name=evojax
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=l40s:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/jacopo04/evojax/jobs_logs/evojax-%j.out
#SBATCH --error=/scratch/jacopo04/evojax/jobs_logs/evojax-%j.err
#SBATCH -D /scratch/jacopo04/evojax

# --- Environment Setup ---
# Load any necessary modules
module --force purge
module load StdEnv/2023
module load python
module load gcc opencv arrow

# Activate your venv
source .venv/bin/activate

pip install --upgrade "jax[cuda12]"
python scripts/benchmarks/train.py --config_fname scripts/benchmarks/configs/NEAT/slimevolley.yaml
