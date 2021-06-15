export PYTHONDONTWRITEBYTECODE=1

# MAML office-home supervised task 1-shot 5-way (Finalized Arguments)
python -u run_homeoffice_maml_supervised.py --test_domain Clipart --shots 1 --meta-shots 15 --inner-iters 5 --meta-step 0.0001 --meta-batch 3 --meta-iters 60000 --eval-iters 10 --learning-rate 0.01 --eval-samples 600 --order 2 --dropout 0.10 --augment --pre_train_wts_enable | tee logs/June_14/officehome_maml_supervised_task_test-domain_Clipart_seed_0_shots_1_ways_5_14_06_2021_0254.txt

# Just using pre-trained wts only
#python -u run_homeoffice_maml_supervised.py --test_domain Product --shots 1 --meta-shots 15 --inner-iters 5 --meta-step 0.0001 --meta-batch 3 --meta-iters 60000 --eval-iters 10 --learning-rate 0.01 --eval-samples 600 --order 2 --pre_train_wts_enable | tee logs/June/officehome_maml_supervised_task_seed_0_shots_1_pre-trained-wts_ways_5_11_06_2021_1906.txt

#python -u run_homeoffice_maml_supervised.py --shots 1 --meta-shots 15 --inner-iters 5 --meta-step 0.0001 --meta-batch 3 --meta-iters 60000 --eval-iters 10 --learning-rate 0.01 --eval-samples 600 --order 2 --only-evaluation --checkpoint /home/sankha/Surjayan/Few_Shot_learning_Domain_Adaptation/results_auto/domain_adaptation/maml_supervised/officehome_maml_checkpoint_order_2_seed_0_datetime_20_04_2021_14_32_07_shots_1_ways_5/final_model.pt

# # MO-MAML office-home supervised task 1-shot 5-way
# python -u run_homeoffice_MO_maml_supervised.py --shots 1 --meta-shots 15 --inner-iters 5 --meta-step 0.001 --meta-batch 3 --meta-iters 60000 --eval-iters 10 --learning-rate 0.01 --eval-samples 600 --order 2 | tee logs/output_homeoffice_mo_maml_supervised_task_seed_0_shots_1_ways_5_12_04_2021_1305.txt

# gs-maml
#python -u run_homeoffice_gs_maml_supervised.py --shots 1 --meta-shots 15 --inner-iters 5 --meta-step 0.001 --meta-batch 3 --meta-iters 60000 --eval-iters 10 --learning-rate 0.01 --eval-samples 600 --order 2 | tee logs/output_homeoffice_gs_maml_supervised_task_seed_0_shots_1_ways_5_13_04_2021_1940.txt