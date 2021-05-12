#!/bin/bash
# Energy experiments

source .myenv/bin/activate
printf -v date '%(%Y%m%d)T'
dir="${date}_energy"

seeds=( 2019 2020 2021 )
weightdecays=( 0.0 0.001 0.005 0.01 0.05 0.1)
task=mnist_energy

for seed in "${seeds[@]}"
do
	for weightdecay in "${weightdecays[@]}"
	do
		python run_dyn_continual.py --log_dir ${dir}\/dyn --seed ${seed} --task ${task} --weight_decay ${weightdecay} --experiment_lesion &
		python run_mlp_continual.py --log_dir ${dir}\/mlp --seed ${seed} --task ${task} --weight_decay ${weightdecay} &
		python run_mlp_continual.py --log_dir ${dir}\/mlp-prob --seed ${seed} --task ${task} --weight_decay ${weightdecay} --prob_release 0.4 --learning_rate 0.05 &

		wait
	done
done
