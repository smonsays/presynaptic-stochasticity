#!/bin/bash
# Ablation experiments on permuted MNIST.
# Hyperparameters for each baseline are individually tuned for a fair comparison

source .myenv/bin/activate
printf -v date '%(%Y%m%d)T'
dir="${date}_ablation_perm-mnist"
seeds=( 2019 2020 2021 )

for seed in "${seeds[@]}"
do

    python run_dyn_continual.py --log_dir ${dir}\/full --seed ${seed} --task perm_mnist --prob_drift_down 0.04021639 --prob_drift_up 0.04460148 --prob_freeze 0.94791622 &
    python run_dyn_continual.py --log_dir ${dir}\/no_freeze --seed ${seed} --task perm_mnist --prob_freeze 1.0 --prob_drift_down 0.04378614 --prob_drift_up 0.04522727 &
    python run_dyn_continual.py --log_dir ${dir}\/no_freeze_no_learning_mod --seed ${seed} --task perm_mnist --prob_freeze 1.0 --no_lr_modulation --prob_drift_down 0.02954858 --prob_drift_up 0.09202317 &
    python run_dyn_continual.py --log_dir ${dir}\/no_learning_mod --seed ${seed} --task perm_mnist --no_lr_modulation --prob_drift_down 0.03559312 --prob_drift_up 0.08535058 --prob_freeze 0.92272659 &
    python run_dyn_continual.py --log_dir ${dir}\/fixed_probs --seed ${seed} --task perm_mnist --fixed_probs 0.5 --prob_drift_down 0.05188978 --prob_drift_up 0.0571483 --prob_freeze 0.92943784 &
    python run_dyn_continual.py --log_dir ${dir}\/no_normalise --seed ${seed} --task perm_mnist --no_normalise --prob_drift_down 0.02752443 --prob_drift_up 0.09712197 --prob_freeze 0.91279887 &

    # python run_dyn_continual.py --log_dir ${dir}\/hard_freeze --seed ${seed} --task perm_mnist --hard_freeze true &
    # python run_dyn_continual.py --log_dir ${dir}\/grad_hard_reset --seed ${seed} --task perm_mnist --grad_hard_reset &

    # Only run 4 processes in parallel
    wait

done
