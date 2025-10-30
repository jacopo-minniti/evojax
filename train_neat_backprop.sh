#!/bin/bash
#SBATCH --job-name=neat_backprop
#SBATCH --time=00:10:00
#SBATCH --gpus-per-node=l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --output=/scratch/jacopo04/evojax/jobs_logs/neat_backprop-%j.out
#SBATCH --error=/scratch/jacopo04/evojax/jobs_logs/neat_backprop-%j.err
#SBATCH -D /scratch/jacopo04/evojax

set -euo pipefail

# --- Environment Setup ---
module --force purge
module load StdEnv/2023
module load python
module load gcc opencv arrow

source .venv/bin/activate

pip install --upgrade "jax[cuda12]"

python examples/train_classifier.py --output-dir log/NEAT_backprop/binary_classification
