#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=tsp2          # Job name
#SBATCH --time=1-00:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=cpu           # Select the ai partition
# --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=30000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# Optional directives
#SBATCH --mail-type=ALL         # Email notification for job status changes
#SBATCH --mail-user=abis28891@gmail.com  # Email address for notifications

# Your job commands here
#python tsp_solving.py
#python generate_instances.py 110 128 ../tsplib95_110_instances_128_node/all_instances_adj_tour_cost.txt ../generatedv4
#python preprocess_dataset.py ../generatedv4
#python train.py ../generated ../model_result --use_gpu
python test_save.py ../generatedv4/test.txt ../gnngls/models/tsp100/checkpoint_best_val.pt ../runs4 regret_pred ../saved_data
#python test_me.py