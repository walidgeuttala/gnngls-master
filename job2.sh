#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=tsp3          # Job name
#SBATCH --time=0-01:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=gpu           # Select the ai partition
#SBATCH --gres=gpu:1        # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=40000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# Optional directives
#SBATCH --mail-type=ALL         # Email notification for job status changes
#SBATCH --mail-user=walidgeuttala@gmail.com  # Email address for notifications

# Your job commands here
#python generate_instances.py
# python tsp_solving.py
#python generate_instances.py 2000 128 ../tsplib95_10000_instances_64_node/tsp_all_instances_adj_tour_cost.txt ../generatedn2000
#python preprocess_dataset.py ../generatedn2000
python train2.py ../atsp_n5900 ../model_result_try --use_gpu
#python test.py ../tsp_lib_test_with_regret2/test.txt ../model_result_try/Jul13_16-09-51_6937d04a2f2f4c90b92ad923ed0d8304/checkpoint_best_val.pt ../runs_lib_19_regret2 regret_pred ../out_lib_19_regret3
#python test_me.py
