#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=gnn        # Job name
#SBATCH --time=0-02:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=cpu           # Select the ai partition
## --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=30000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# Optional directives
#SBATCH --mail-type=ALL         # Email notification for job status changes
#SBATCH --mail-user=abis28891@gmail.com  # Email address for notifications

# Your job commands here
python tsp_solving.py

