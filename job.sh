#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=gnn2          # Job name
#SBATCH --time=0-40:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=cpu           # Select the ai partition
## --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=30000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# Optional directives
#SBATCH --mail-type=ALL         # Email notification for job status changes
#SBATCH --mail-user=abis28891@gmail.com  # Email address for notifications

# Your job commands here
python tsp_solving.py
python generate_instances.py 3 64 ../tsplib95_3_instances_64_node/all_instances_adj_tour_cost.txt ../walid
python preprocess_dataset.py ../walid
python train.py ../walid ../dodo --use_gpu
#
#python test_me.py