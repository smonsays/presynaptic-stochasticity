"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import copy
import json
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch


from lib import config, data, ddc, test, metric, nets, train, utils


def add_parameter_noise(cfg, noise_ratio):
    """
    Add noise to the hyperparameters of the Dynamic DropConnect model.
    This changes the input dictionary in-place.

    Args:
        cfg: Dictionary containing the configuration with all hyperparameters
        noise_ratio: Ratio of noise relative to magnitude of parameter base value

    Returns:
        Perturbed hyperparameter dictionary
    """
    # Add uniform noise scaled relative to magnitude of input float.
    def _add_relative_uniform_noise(x, noise_ratio):
        noise_range = noise_ratio * abs(x)
        return random.uniform(x - noise_range, x + noise_range)

    # Noise only applies to the hyperparemeters of the model that we tune
    cfg['prob_drift_down'] = _add_relative_uniform_noise(cfg['prob_drift_down'], noise_ratio)
    cfg['prob_drift_up'] = _add_relative_uniform_noise(cfg['prob_drift_up'], noise_ratio)
    cfg['prob_freeze'] = _add_relative_uniform_noise(cfg['prob_freeze'], noise_ratio)
    cfg['grad_threshold'] = _add_relative_uniform_noise(cfg['grad_threshold'], noise_ratio)

    return cfg


def generate_random_hyperparameter(task):
    """
    Randomly sample task-specific hyperparameters.

    Args:
        task: Task name

    Returns:
        Dictionary with random hyperparameters
    """
    if task == "mnist_energy":
        # Tie prob_up and prob_down
        prob_up_down = np.random.uniform(0.01, 0.1)
        random_hyperparameter = {
            "prob_drift_down": prob_up_down,
            "prob_drift_up": prob_up_down,
            "weight_decay": np.random.choice([0.001, 0.005, 0.01, 0.05]),
        }

    elif task == "perm_mnist" or task == "perm_fmnist":
        random_hyperparameter = {
            "prob_drift_down": np.random.uniform(0.01, 0.1),
            "prob_drift_up": np.random.uniform(0.01, 0.1),
            "prob_freeze": np.random.uniform(0.9, 0.96),
        }

    elif task == "split_mnist" or task == "split_fmnist":
        random_hyperparameter = {
            "prob_drift_down": np.random.uniform(0.01, 0.1),
            "prob_drift_up": np.random.uniform(0.01, 0.1),
            "prob_freeze": np.random.uniform(0.9, 0.96),
        }

    else:
        raise ValueError("Task \"{}\" not defined.".format(task))

    return random_hyperparameter


def load_default_config(task):
    """
    Load default parameter configuration from file.

    Args:
        tasks: String with the task name

    Returns:
        Dictionary of default parameters for the given task
    """
    if task == "mnist":
        default_config = "etc/dyn_mnist.json"
    elif task == "mnist_energy":
        default_config = "etc/dyn_mnist_energy.json"
    elif task == "mnist_fisher":
        default_config = "etc/dyn_mnist_fisher.json"
    elif task == "perm_fmnist":
        default_config = "etc/dyn_perm_fmnist.json"
    elif task == "perm_mnist":
        default_config = "etc/dyn_perm_mnist.json"
    elif task == "split_fmnist":
        default_config = "etc/dyn_split_fmnist.json"
    elif task == "split_mnist":
        default_config = "etc/dyn_split_mnist.json"
    elif task == "perm_mnist_cont":
        default_config = "etc/dyn_perm_mnist_cont.json"
    else:
        raise ValueError("Task \"{}\" not defined.".format(task))

    with open(default_config) as config_json_file:
        cfg = json.load(config_json_file)

    return cfg


def parse_shell_args(args):
    """
    Parse shell arguments for this script

    Args:
        args: Command line arguments passed through sys.argv[1:]

    Returns:
        Dictionary with configuration
    """
    parser = argparse.ArgumentParser(description="Run experiments with the Dynamic DropConnect model.")

    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS,
                        help="Size of mini batches during training.")

    parser.add_argument("--dimensions", type=int, nargs="+",
                        default=argparse.SUPPRESS, help="Dimensions of the neural network.")

    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS,
                        help="Number of epochs to train.")

    parser.add_argument("--experiment_clamp", action='store_true', default=False,
                        help="Flag to conduct the probability clamp experiment.")

    parser.add_argument("--experiment_fisher", action='store_true', default=False,
                        help="Flag to conduct the Fisher Information experiment.")

    parser.add_argument("--experiment_lesion", action='store_true', default=False,
                        help="Flag to conduct the importance lesion experiment.")

    parser.add_argument("--fixed_probs", type=float, default=argparse.SUPPRESS,
                        help="Fixed forward transmission probabilities decoupled from importance parameter.")

    parser.add_argument("--grad_hard_reset", action='store_true', default=argparse.SUPPRESS,
                        help="Flag to enable resetting gradient cache after every training step (effectively turning it into a simple threshold).")
    parser.add_argument("--no_grad_hard_reset", dest="grad_hard_reset", action='store_false', default=argparse.SUPPRESS,
                        help="Flag to disable resetting gradient cache after every training step (effectively turning it into a simple threshold).")

    parser.add_argument("--hard_freeze", action='store_true', default=argparse.SUPPRESS,
                        help="Flag to enable complete freezing of weights when transmission probability is frozen.")
    parser.add_argument("--no_hard_freeze", dest="hard_freeze", action='store_false', default=argparse.SUPPRESS,
                        help="Flag to disable complete freezing of weights when transmission probability is frozen.")

    parser.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS,
                        help="Learning rate for the parameters.")

    parser.add_argument("--log_dir", type=str, default="",
                        help="Subdirectory within ./log/ where to store logs.")

    parser.add_argument("--lr_modulation", action='store_true', default=argparse.SUPPRESS,
                        help="Flag to enable the learning rate modulation.")
    parser.add_argument("--no_lr_modulation", dest="lr_modulation", action='store_false', default=argparse.SUPPRESS,
                        help="Flag to disable the learning rate modulation.")

    parser.add_argument("--nonlinearity", choices=["leaky_relu", "relu", "sigmoid", "tanh"],
                        default=argparse.SUPPRESS, help="Nonlinearity between network layers.")

    parser.add_argument("--normalise", action='store_true', default=argparse.SUPPRESS,
                        help="Flag to enable normalisation of sampled weights by their transmission probability.")
    parser.add_argument("--no_normalise", dest="normalise", action='store_false', default=argparse.SUPPRESS,
                        help="Flag to disable normalisation of sampled weights by their transmission probability.")

    parser.add_argument("--prob_drift_down", type=float, default=argparse.SUPPRESS,
                        help="Weight probability decrease parameter.")

    parser.add_argument("--prob_drift_up", type=float, default=argparse.SUPPRESS,
                        help="Weight probability increase parameter.")

    parser.add_argument("--prob_freeze", type=float, default=argparse.SUPPRESS,
                        help="Weight probability at which proabilities are frozen.")

    parser.add_argument("--random_hyperparameter", action='store_true', default=False,
                        help="Use random hyperparameters.")

    parser.add_argument("--relative_parameter_noise", type=float, default=None,
                        help="Relative noise to be added to the parameters.")

    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for pytorch")

    parser.add_argument("--task", choices=["mnist", "mnist_energy", "mnist_fisher", "perm_fmnist", "perm_mnist", "perm_mnist_cont", "split_fmnist", "split_mnist"],
                        default="mnist_energy", help="Continual task to be solved.")

    parser.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS,
                        help="Weight decay (~l2-regularization strength).")

    return vars(parser.parse_args(args))


def run_dyn_continual(cfg, verbose=True):
    """
    Main routine of this script.

    Args:
        cfg: Dictionary containing the configuration with all hyperparameters
        verbose: Boolean controlling the verbosity of the logger

    Returns:
        Results dictionary, trained dyn_mlp model, test_loaders
    """
    # Initialize seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    # Load specified dataset
    train_loaders, test_loaders = data.create_dataloader(cfg["dataset"]["name"], **cfg["dataset"]["kwargs"])

    # Initialize nonlinearity used as activation function
    nonlinearity = utils.create_nonlinearity(cfg['nonlinearity'])

    mlp = nets.MultilayerPerceptron(cfg['dimensions'], nonlinearity, cfg['fixed_bias'])
    dyn_mlp = ddc.DynamicDropConnect(mlp, cfg['learning_rate'], cfg['grad_threshold'],
                                     cfg['prob_drift_down'], cfg['prob_drift_up'], cfg['prob_freeze'],
                                     cfg['prob_max'], cfg['prob_min'], cfg['fixed_bias'], cfg['fixed_probs'],
                                     cfg['hard_freeze'], cfg['manual_freeze'], cfg['lr_modulation'],
                                     cfg['weight_decay'], cfg['normalise']).to(config.device)

    logging.info("Start training with parametrization:\n{}".format(
        json.dumps(cfg, indent=4, sort_keys=True)))

    # Store results in a dicitonary of tensors
    results = {
        "energy_neuron": torch.zeros(len(train_loaders), cfg['epochs']),
        "energy_weight": torch.zeros(len(train_loaders), cfg['epochs']),
        "frozen_prob": torch.zeros(len(train_loaders), cfg['epochs']),
        "mean_prob": torch.zeros(len(train_loaders), cfg['epochs']),
        "mutual_inf": torch.zeros(len(train_loaders), cfg['epochs']),
        "prob_angles": torch.zeros(len(train_loaders), cfg['epochs'], len(dyn_mlp.weight_probs)),
        "perm_angles": torch.zeros(len(train_loaders), cfg['epochs'], len(dyn_mlp.weight_probs)),
        "task_acc": torch.zeros(len(train_loaders), len(train_loaders)),
        "task_mutual_inf": torch.zeros(len(train_loaders), len(train_loaders)),
        "test_acc": torch.zeros(len(train_loaders), cfg['epochs']),
        "train_acc": torch.zeros(len(train_loaders), cfg['epochs']),
        "train_loss": torch.zeros(len(train_loaders), cfg['epochs']),
    }

    # Train each task in a continual fashion
    for task, (trainer, tester) in enumerate(zip(train_loaders, test_loaders)):
        logging.info("Starting training of task {}".format(task + 1))

        for epoch in range(cfg['epochs']):
            # Store weights probs to quantify the change of important weights in the end
            weight_probs_prev = copy.deepcopy(dyn_mlp.weight_probs)

            # Train the model for a single epoch
            train_acc, train_loss = train.train(dyn_mlp, trainer, method="ddc", verbose=verbose)

            # Test the model on the current task
            test_acc = test.accuracy(dyn_mlp, tester, cfg['test_samples'])

            # Compute the mutual information between true and learned labels and the energy
            mutual_inf = metric.mutual_information(dyn_mlp, tester)
            energy_weight = dyn_mlp.energy
            energy_neuron = test.activation(dyn_mlp, tester, cfg['test_samples'])

            # Compute the mean transmission probabilities of all weights and ratio of frozen weights
            mean_prob = utils.compute_mean_probs(dyn_mlp.weight_probs)
            frozen_prob = utils.compute_frozen_probs(dyn_mlp.frozen_weight_mask)

            # Compute the layerwise angles between weight_probs now and beginning of the epoch
            angles = torch.stack([
                utils.vector_angle(p_prev.view(-1), p_curr.view(-1))
                for p_prev, p_curr in zip(weight_probs_prev, dyn_mlp.weight_probs)
            ])

            # Compute permuted angles as a control
            perm_angles = torch.stack([
                utils.vector_angle(p_prev.view(-1), p_curr.view(-1)[torch.randperm(p_curr.numel())])
                for p_prev, p_curr in zip(weight_probs_prev, dyn_mlp.weight_probs)
            ])

            # Store results
            results['energy_weight'][task, epoch] = energy_weight
            results['energy_neuron'][task, epoch] = energy_neuron
            results['frozen_prob'][task, epoch] = frozen_prob
            results['mean_prob'][task, epoch] = mean_prob
            results['mutual_inf'][task, epoch] = mutual_inf
            results['prob_angles'][task, epoch] = angles
            results['perm_angles'][task, epoch] = perm_angles
            results['test_acc'][task, epoch] = train_acc
            results['train_acc'][task, epoch] = train_acc
            results['train_loss'][task, epoch] = train_loss

            # Logging
            if verbose:
                logging.info("epoch {}/{}: train_acc {:.4f} \t test_acc {:.4f} \t mean_p: {:.4f} \t frozen_p: {:.6f} \t angle_p: {:.4f} \t mutual_inf {:.4f} \t energy: {:.4f} \t bit/energy: {:.4f}".format(epoch + 1, cfg['epochs'], train_acc, test_acc, mean_prob, frozen_prob, angles.mean(), mutual_inf, energy_weight, mutual_inf / energy_weight))

            config.writer.add_scalars('task{}/accuracy'.format(task + 1), {'train': train_acc, 'test': test_acc}, epoch)
            config.writer.add_scalar('task{}/information'.format(task + 1), mutual_inf, epoch)
            config.writer.add_scalar('task{}/info_per_energy'.format(task + 1), mutual_inf / energy_weight, epoch)
            config.writer.add_scalar('task{}/energy_weight'.format(task + 1), energy_weight, epoch)
            config.writer.add_scalar('task{}/energy_neuron'.format(task + 1), energy_neuron, epoch)
            config.writer.add_scalar('task{}/train_loss'.format(task + 1), train_loss, epoch)
            config.writer.add_scalars('task{}/probability'.format(task + 1), {'mean': mean_prob, 'frozen': frozen_prob}, epoch)

            for l, p in enumerate(dyn_mlp.weight_probs):
                config.writer.add_histogram('task{}/probabilities/layer{}'.format(task + 1, l), p.view(-1), epoch)

            for l, w in enumerate(dyn_mlp.weights):
                config.writer.add_histogram('task{}/weights/layer{}'.format(task + 1, l), w.view(-1), epoch)

            for l, (p, w) in enumerate(zip(dyn_mlp.weight_probs, dyn_mlp.weights)):
                config.writer.add_histogram('task{}/weightxprob/layer{}'.format(task + 1, l), (p * w).view(-1), epoch)

        # If manual freezing is enabled, use task boundaries to freeze weights
        if cfg['manual_freeze']:
            dyn_mlp.update_frozen_weight_mask()
            frozen_prob = utils.compute_frozen_probs(dyn_mlp.frozen_weight_mask)

        # Compute metrics to be logged
        task_accuracies = torch.tensor([
            test.accuracy(dyn_mlp, tester, cfg['test_samples'])
            for tester in test_loaders
        ])
        task_mutual_inf = torch.tensor([
            metric.mutual_information(dyn_mlp, tester)
            for tester in test_loaders
        ])

        # Store results
        results['task_acc'][task] = task_accuracies
        results['task_mutual_inf'][task] = task_mutual_inf

        # Logging
        task_accuracies_dict = {"task{}".format(task + 1): task_acc.item() for task, task_acc in enumerate(task_accuracies)}
        if verbose:
            logging.info("Task accuracies: {}".format(json.dumps(task_accuracies_dict, indent=4, sort_keys=True)))
            logging.info("Mean task accuracy: {:.4f}".format(torch.mean(task_accuracies)))

        config.writer.add_scalar('continual/mean_accuracy', torch.mean(task_accuracies), task + 1)
        config.writer.add_scalars('continual/task_accuracies', task_accuracies_dict, task + 1)
        config.writer.add_scalars('continual/probability', {'mean': mean_prob, 'frozen': frozen_prob}, task + 1)

        # Early stopping for random search
        if cfg["random_hyperparameter"] and (task == 0) and task_accuracies[0] < 0.5:
            break

    # Compute the full mutual information on all test sets if there is multiple tasks
    if len(test_loaders) > 1:
        test_loader_all = data._create_dataloader(
            torch.utils.data.ConcatDataset([
                tester.dataset for tester in test_loaders
            ]), is_train=False, batch_size=cfg['dataset']['kwargs']['batch_size'])

        results['mutual_inf_all'] = metric.mutual_information(dyn_mlp, test_loader_all)

    return results, dyn_mlp, test_loaders


if __name__ == '__main__':
    # Load configuration (Priority: 1 User, 2 Random, 3 Default)
    user_config = parse_shell_args(sys.argv[1:])
    cfg = load_default_config(user_config["task"])

    # Draw random dynamics parameters if specified
    if user_config['random_hyperparameter']:
        cfg.update(generate_random_hyperparameter(user_config['task']))

    cfg.update(user_config)

    # Optionally add noise to the model hyperparameters to test sensitivity
    if cfg['relative_parameter_noise'] is not None:
        logging.info("Added noise to the parameters")
        cfg.update(add_parameter_noise(cfg, noise_ratio=user_config['relative_parameter_noise']))

    # Setup global logger and logging directory
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_dyn_" + cfg['dataset']['name']
    config.setup_logging(run_id, dir=cfg['log_dir'])

    # Run the continual learning task with complete logging enabled
    results, dyn_mlp, test_loaders = run_dyn_continual(cfg, verbose=True)

    # Conduct the importance ablation experiment
    if cfg['experiment_lesion']:
        weight_parameters = [value for (key, value) in dyn_mlp.named_parameters() if '.weight_raw' in key]

        # Lesion according to weight probabilities
        weight_probabilities = [value for (key, value) in dyn_mlp.named_buffers() if '.weight_prob' in key]
        results['lesion_prob'] = test.importance_lesion(dyn_mlp, weight_parameters, weight_probabilities, test_loaders[0], cfg['test_samples'])
        logging.info("Weight probability ablation accuracies:\n{}".format(json.dumps(results['lesion_prob'], indent=4, sort_keys=True)))

        # Lesion according to weight magnitudes
        weight_magnitudes = [torch.abs(value) for (key, value) in dyn_mlp.named_parameters() if '.weight_raw' in key]
        results['lesion_magn'] = test.importance_lesion(dyn_mlp, weight_parameters, weight_magnitudes, test_loaders[0], cfg['test_samples'])
        logging.info("Weight magnitude ablation accuracies:\n{}".format(json.dumps(results['lesion_magn'], indent=4, sort_keys=True)))

        weight_magnxprob = [m * p for m, p in zip(weight_magnitudes, weight_probabilities)]
        results['lesion_magnxprob'] = test.importance_lesion(dyn_mlp, weight_parameters, weight_magnxprob, test_loaders[0], cfg['test_samples'])
        logging.info("Weight magnxprob ablation accuracies:\n{}".format(json.dumps(results['lesion_magn'], indent=4, sort_keys=True)))

        # Lesion randomly
        weight_random = [torch.rand_like(value) for (key, value) in dyn_mlp.named_buffers() if '.weight_prob' in key]
        results['lesion_rand'] = test.importance_lesion(dyn_mlp, weight_parameters, weight_random, test_loaders[0], cfg['test_samples'])
        logging.info("Random ablation accuracies:\n{}".format(json.dumps(results['lesion_rand'], indent=4, sort_keys=True)))

    # Conduct the Fisher information importance correlation experiment
    if cfg['experiment_fisher']:
        corr_coeff, p_value, fisher = test.fisher_importance_correlation(dyn_mlp, test_loaders)
        logging.info("Transmission probabilities and fisher information correlate with {:.4f} (p={:.4f})".format(corr_coeff, p_value))
        results['fisher_correlation'] = (corr_coeff, p_value)
        results['params_fisher'] = fisher

    # Conduct the probability clamp experiment
    if cfg['experiment_clamp']:
        clamped_accuracies = {}
        for prob_clamp in torch.arange(0.1, 1.1, 0.1):
            clamped_accuracies["{:2f}".format(prob_clamp)] = test.probability_clamp(
                prob_clamp, dyn_mlp, test_loaders[0], cfg['test_samples'])

        results['clamped_accuracies'] = clamped_accuracies
        logging.info("Clamped probability accuracies:\n{}".format(json.dumps(clamped_accuracies, indent=4, sort_keys=True)))

    # Save the configuration as json
    utils.save_dict_as_json(cfg, run_id, config.log_dir)

    # Store results, configuration and model state as pickle
    results['cfg'], results['state_dict'] = cfg, dyn_mlp.state_dict()
    torch.save(results, os.path.join(config.log_dir, run_id + "_results.pt"))

    # Zip the tensorboard logging results and remove the folder to save space
    config.writer.close()
    path_tensorboard = os.path.join(config.log_dir, run_id + "_tensorboard")
    shutil.make_archive(path_tensorboard, 'zip', path_tensorboard)
    shutil.rmtree(path_tensorboard)
