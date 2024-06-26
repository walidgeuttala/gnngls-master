#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=tspsp2        # Job name
#SBATCH --time=0-12:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=gpu           # Select the ai partition
#SBATCH --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=30000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# Optional directives
#SBATCH --mail-type=ALL         # Email notification for job status changes
#SBATCH --mail-user=abis28891@gmail.com  # Email address for notifications

# Your job commands here
#python tsp_solving.py
#python generate_instances.py 110 128 ../tsplib95_110_instances_128_node/all_instances_adj_tour_cost.txt ../generatedv4
#python preprocess_dataset.py ../cleaned_data_n5900
#python train.py ../generated ../model_result --use_gpu
#python test_me.py
#python preprocess_dataset2.py ../generatedn2000
python train3.py ../tsp_n5900 ../model_result_n5900 --use_gpu
#python test.py ../tsp_n5900/test3.txt ../model_result_n5900/Apr25_14-21-43_167db68588d5493fbd47a7a145fa3e02/checkpoint_best_val.pt ../run_result_n5900 regret_pred ../test_model4 --use_gpu
#python generate_instances.py 8000 128 ../tsplib95_10000_instances_64_node/tsp_all_instances_adj_tour_cost.txt ../generatedn8000
#python test_regret.py
#python remove_extra_edges.py

#Apr25_14-21-43_167db68588d5493fbd47a7a145fa3e02
