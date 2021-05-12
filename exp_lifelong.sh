#!/bin/bash
# Main experiments

source .myenv/bin/activate
printf -v date '%(%Y%m%d)T'
dir="${date}lifelong"

seeds=( 2019 2020 2021 )
tasks=( "perm_fmnist" "perm_mnist" "split_fmnist" "split_mnist")

for seed in "${seeds[@]}"
do
	for task in "${tasks[@]}"
	do
		python run_dyn_continual.py --log_dir ${dir}\/${task} --seed ${seed} --task ${task} &
		python run_mlp_continual.py --log_dir ${dir}\/${task} --seed ${seed} --task ${task} &
		python run_mlp_multitask.py --log_dir ${dir}\/${task} --seed ${seed} --task ${task} --optimizer adam --learning_rate 0.001 --weight_decay 0.0 &
		wait
	done
done
