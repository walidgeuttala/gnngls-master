#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=tsp22          # Job name
#SBATCH --time=0-12:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=ai           # Select the ai partition
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
python train2.py ../atsp_n5900 ../model_result_n5900 --use_gpu
#python test.py ../atsp_n5900/test3.txt ../model_result_n5900/Apr25_07-23-29_c9d520ac762949bca09ff5df493ede10/checkpoint_best_val.pt ../run_result_n5900 regret_pred ../test_model2 --use_gpu
#python generate_instances.py 8000 128 ../tsplib95_10000_instances_64_node/tsp_all_instances_adj_tour_cost.txt ../generatedn8000
#python test_regret.py
#python remove_extra_edges.py
#python models_test.py


#Apr24_20-47-23_b79a061deed3469484a4bb05fcc9183c   hetrov1 RGB
#Apr25_07-23-29_c9d520ac762949bca09ff5df493ede10    hetrov2 RGB4