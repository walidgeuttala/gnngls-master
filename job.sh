#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=tspsp          # Job name
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
#python train2.py ../tsp_n5900 ../model_result_n5900 --use_gpu
#python test.py ../tsp_n5900/test3.txt ../model_result_n5900/Apr25_16-11-20_e588e067396a45fa859199221cd0fc06/checkpoint_best_val.pt ../run_result_n5900 regret_pred ../test_model4 --use_gpu
#python generate_instances.py 8000 128 ../tsplib95_10000_instances_64_node/tsp_all_instances_adj_tour_cost.txt ../generatedn8000
#python test_regret.py
#python remove_extra_edges.py

#Apr25_16-10-12_b03c35e339a6441587bb33e81885c08c model1  4569076   9
#Apr25_16-16-10_f644d0bc8edc474fa5d2641c6ae89d73   model3     4566184
#Apr25_16-11-20_e588e067396a45fa859199221cd0fc06  model1      4568360   8
#Apr25_17-08-24_7d8ed1e72d5d44faabfa5847f6a5f2c4  model2 
# model2 results 4569084

#model1 4570011 ../test_model4
#model2 4570003 ../test_model5
#model3 4570008 ../test_model6


#Apr25_16-10-12_b03c35e339a6441587bb33e81885c08c model1  4569076  
#Apr25_17-08-24_7d8ed1e72d5d44faabfa5847f6a5f2c4  model2    4570003            4570003        
#Apr25_16-16-10_f644d0bc8edc474fa5d2641c6ae89d73   model3     4566184
# model2 results 4569084


python test_cal2.py