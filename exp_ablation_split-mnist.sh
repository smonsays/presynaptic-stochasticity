#!/bin/bash
# Ablation experiments on split MNIST
# Hyperparameters for each baseline are individually tuned for a fair comparison

source .myenv/bin/activate
printf -v date '%(%Y%m%d)T'
dir="${date}_ablation_split-mnist"
seeds=( 2019 2020 2021 )

for seed in "${seeds[@]}"
do

    python run_dyn_continual.py --log_dir ${dir}\/full --seed ${seed} --task split_mnist --prob_drift_down 0.0624703 --prob_drift_up 0.05835393 --prob_freeze 0.94452875 &
    python run_dyn_continual.py --log_dir ${dir}\/no_freeze --seed ${seed} --task split_mnist --prob_freeze 1.0 --prob_drift_down 0.08590142 --prob_drift_up 0.06543912 &
    python run_dyn_continual.py --log_dir ${dir}\/no_freeze_no_learning_mod --seed ${seed} --task split_mnist --prob_freeze 1.0 --no_lr_modulation --prob_drift_down 0.09587958 --prob_drift_up 0.01798108 &
    python run_dyn_continual.py --log_dir ${dir}\/no_learning_mod --seed ${seed} --task split_mnist --no_lr_modulation --prob_drift_down 0.0506991 --prob_drift_up 0.04199989 --prob_freeze 0.94187781 &
    python run_dyn_continual.py --log_dir ${dir}\/fixed_probs --seed ${seed} --task split_mnist --fixed_probs 0.5 --prob_drift_down 0.09833934 --prob_drift_up 0.09764242 --prob_freeze 0.9466155 &
    python run_dyn_continual.py --log_dir ${dir}\/no_normalise --seed ${seed} --task split_mnist --no_normalise --prob_drift_down 0.02838338 --prob_drift_up 0.09717001 --prob_freeze 0.9499824 &

    # python run_dyn_continual.py --log_dir ${dir}\/hard_freeze --seed ${seed} --task split_mnist --hard_freeze true &
    # python run_dyn_continual.py --log_dir ${dir}\/grad_hard_reset --seed ${seed} --task split_mnist --grad_hard_reset &

    # Only run 4 processes in parallel
    wait

done
