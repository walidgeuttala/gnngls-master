#!/bin/bash
#SBATCH -A p_gnn001               # Account name to be debited
#SBATCH --job-name=tsp3          # Job name
#SBATCH --time=0-03:00:00        # Maximum walltime (30 minutes)
#SBATCH --partition=gpu           # Select the ai partition
#SBATCH --gres=gpu:1          # Request 1 to 4 GPUs per node
#SBATCH --mem-per-cpu=30000       # Memory per CPU core (16 GB)
#SBATCH --nodes=1               # Request 1 node

# Optional directives
#SBATCH --mail-type=ALL         # Email notification for job status changes
#SBATCH --mail-user=walidgeuttala@gmail.com  # Email address for notifications

# Your job commands here
#python generate_instances.py
# python tsp_solving.py
#python generate_instances.py 2000 128 ../tsplib95_10000_instances_64_node/tsp_all_instances_adj_tour_cost.txt ../generatedn2000
#python preprocess_dataset.py ../generatedn2000
#python train.py ../atsp_n5900 ../model_result_try
# python test_save.py ../model_result_try/test.txt ../gnngls/models/tsp100/checkpoint_best_val.pt ../runsn2000 regret_pred ../outn2000
#python test_me.py

python train2.py ../generated_insatnces_3000_size_50 ../model_result_try_size_50_samples_1000 --use_gpu

#python test.py ../generated_insatnces_1000_size_100/test.txt ../model_result_try_size_50_samples_1000/Sep30_21-08-41_e549c15adf494579b25fb67d7836c7af/checkpoint_best_val.pt ../runs_lib_19_regret2 regret_pred ../output_ATSP_trained_50_size_samples_1000_size_100_model2 --use_gpu
# python test.py ../generated_insatnces_100_size_150/test.txt ../model_result_try_size_50_samples_1000/Sep30_21-08-41_e549c15adf494579b25fb67d7836c7af/checkpoint_best_val.pt ../runs_lib_19_regret2 regret_pred ../output_ATSP_trained_50_size_samples_100_size_150_model2 --use_gpu
# python test.py ../generated_insatnces_100_size_250/test.txt ../model_result_try_size_50_samples_1000/Sep30_21-08-41_e549c15adf494579b25fb67d7836c7af/checkpoint_best_val.pt ../runs_lib_19_regret2 regret_pred ../output_ATSP_trained_50_size_samples_100_size_250_model2 --use_gpu
#python test.py ../generated_insatnces_100_size_1000/test.txt ../model_result_try_size_50_samples_1000/Sep30_21-08-41_e549c15adf494579b25fb67d7836c7af/checkpoint_best_val.pt ../runs_lib_19_regret2 regret_pred ../output_ATSP_trained_50_size_samples_100_size_1000_model2 --use_gpu
