#!/bin/bash
#SBATCH --job-name=neat_backprop
#SBATCH --time=00:20:00
#SBATCH --gpus-per-node=l40s:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
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

CONFIG_FNAME="${1:-scripts/benchmarks/configs/NEAT_backprop/binary_classification.yaml}"
shift || true

python examples/train_classifier.py --config_fname "${CONFIG_FNAME}" "$@"
