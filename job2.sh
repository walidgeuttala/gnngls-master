#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=tsp          # Job name
#SBATCH --time=0-01:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=ai          # Select the ai partition
#SBATCH --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=30000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# Optional directives
#SBATCH --mail-type=ALL         # Email notification for job status changes
#SBATCH --mail-user=abis28891@gmail.com  # Email address for notifications

# Your job commands here
#python tsp_solving.py
#python generate_instances.py 5000 64 ../tsplib95_10000_instances_64_node/all_instances_adj_tour_cost.txt ../generated_v2
#python preprocess_dataset.py ../generated
#python train.py ../generated ../model_result --use_gpu
python test.py ../generated/test.txt ../gnngls/models/tsp100/checkpoint_best_val.pt ../runs2 regret_pred --use_gpu
#python test_me.py