#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=tsp3          # Job name
#SBATCH --time=0-01:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=cpu           # Select the ai partition
# --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=30000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# Optional directives
#SBATCH --mail-type=ALL         # Email notification for job status changes
#SBATCH --mail-user=walidgeuttala@gmail.com  # Email address for notifications

# Your job commands here
#python generate_instances.py
# python tsp_solving.py
# python generate_instances.py 2000 128 ../tsplib95_10000_instances_64_node/tsp_all_instances_adj_tour_cost.txt ../generatedn2000

# python train.py ../generatedn2000 ../model_result
# python test_save.py ../generatedn2000/test.txt ../gnngls/models/tsp100/checkpoint_best_val.pt ../runsn2000 regret_pred ../outn2000
#python test_me.py
# python graph_transform.py
python preprocess_dataset.py ../atsp_n5900