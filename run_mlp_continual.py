"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time

import torch

from lib import config, data, metric, nets, test, train, utils


def load_default_config(task):
    """
    Load default parameter configuration from file.

    Args:
        tasks: String with the task name

    Returns:
        Dictionary of default parameters for the given task
    """
    if task == "mnist_energy":
        default_config = "etc/mlp_mnist_energy.json"
    elif task == "perm_fmnist":
        default_config = "etc/mlp_perm_fmnist.json"
    elif task == "perm_mnist":
        default_config = "etc/mlp_perm_mnist.json"
    elif task == "split_fmnist":
        default_config = "etc/mlp_split_fmnist.json"
    elif task == "split_mnist":
        default_config = "etc/mlp_split_mnist.json"
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
    parser = argparse.ArgumentParser(description="Run experiments with a (probabilistic) Multilayer Perceptron.")

    parser.add_argument("--batch_size", type=int, default=argparse.SUPPRESS,
                        help="Size of mini batches during training.")
    parser.add_argument("--dimensions", type=int, nargs="+",
                        default=argparse.SUPPRESS, help="Dimensions of the neural network.")
    parser.add_argument("--epochs", type=int, default=argparse.SUPPRESS,
                        help="Number of epochs to train.")
    parser.add_argument("--learning_rate", type=float, default=argparse.SUPPRESS,
                        help="Learning rate of the optimizer.")
    parser.add_argument("--log_dir", type=str, default="",
                        help="Subdirectory within ./log/ where to store logs.")
    parser.add_argument("--nonlinearity", choices=["leaky_relu", "relu", "sigmoid", "tanh"],
                        default=argparse.SUPPRESS, help="Nonlinearity between network layers.")
    parser.add_argument("--optimizer", choices=["adam", "adagrad", "sgd"],
                        default=argparse.SUPPRESS, help="Optimizer used to train the model.")
    parser.add_argument("--prob_release", type=float, default=argparse.SUPPRESS,
                        help="Transmission probability of the weights.")
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for pytorch")
    parser.add_argument("--task", choices=["mnist_energy", "perm_fmnist", "perm_mnist", "split_fmnist", "split_mnist"],
                        default="mnist_energy", help="Continual task to be solved.")
    parser.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS,
                        help="Weight decay (~l2-regularization strength) of the optimizer.")

    # Parse arguments to dictionary
    return vars(parser.parse_args(args))


def run_mlp_continual(cfg, verbose=True):
    """
    Main routine of this script.

    Args:
        cfg: Dictionary containing the configuration with all hyperparameters
        verbose: Boolean controlling the verbosity of the logger

    Returns:
        Results dictionary, trained model, test_loaders
    """
    # Initialize seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    # Load specified dataset
    train_loaders, test_loaders = data.create_dataloader(cfg["dataset"]["name"], **cfg["dataset"]["kwargs"])

    # Initialize nonlinearity used as activation function
    nonlinearity = utils.create_nonlinearity(cfg['nonlinearity'])

    # Initialize the model (need to(device) for adagrad)
    mlp = nets.MultilayerPerceptron(cfg['dimensions'], nonlinearity, cfg['fixed_bias'], cfg['prob_release']).to(config.device)

    # Define optimizer (may include l2 regularization via weight_decay)
    optimizer = utils.create_optimizer(cfg['optimizer'], mlp, lr=cfg['learning_rate'],
                                       weight_decay=cfg['weight_decay'])

    logging.info("Start training with parametrization:\n{}".format(json.dumps(cfg, indent=4)))

    # Store results in a dicitonary of tensors
    results = {
        "energy_neuron": torch.zeros(len(train_loaders), cfg['epochs']),
        "energy_weight": torch.zeros(len(train_loaders), cfg['epochs']),
        "mutual_inf": torch.zeros(len(train_loaders), cfg['epochs']),
        "task_acc": torch.zeros(len(train_loaders), len(train_loaders)),
        "task_mutual_inf": torch.zeros(len(train_loaders), len(train_loaders)),
        "test_acc": torch.zeros(len(train_loaders), cfg['epochs']),
        "train_acc": torch.zeros(len(train_loaders), cfg['epochs']),
        "train_loss": torch.zeros(len(train_loaders), cfg['epochs']),
    }

    # Train each task in a continual fashion
    for task, (trainer, tester) in enumerate(zip(train_loaders, test_loaders)):
        logging.info("Starting training of task {}".format(task + 1))

        # Train for the specified amount of epochs
        for epoch in range(cfg['epochs']):
            # Train the model for a single epoch
            train_acc, train_loss = train.train(mlp, trainer, method="standard", optimizer=optimizer, verbose=verbose)

            # Test the model on the current task
            test_acc = test.accuracy(mlp, tester, cfg['test_samples'])

            # Compute the mutual information between true and learned labels
            mutual_inf = metric.mutual_information(mlp, tester)
            energy_neuron = test.activation(mlp, tester, cfg['test_samples'])
            energy_weight = mlp.energy

            # Store results
            results['energy_neuron'][task, epoch] = energy_neuron
            results['energy_weight'][task, epoch] = energy_weight
            results['mutual_inf'][task, epoch] = mutual_inf
            results['test_acc'][task, epoch] = test_acc
            results['train_acc'][task, epoch] = train_acc
            results['train_loss'][task, epoch] = train_loss

            # Logging
            if verbose:
                logging.info("epoch {}: train_acc {:.4f} \t test_acc {:.4f} \t mutual_inf {:.4f} \t energy: {:.4f} \t bit/energy: {:.4f}".format(epoch + 1, train_acc, test_acc, mutual_inf, energy_weight, mutual_inf / energy_weight))

            config.writer.add_scalars('task{}/accuracy'.format(task + 1), {'train': train_acc, 'test': test_acc}, epoch)
            config.writer.add_scalar('task{}/information'.format(task + 1), mutual_inf, epoch)
            config.writer.add_scalar('task{}/info_per_energy'.format(task + 1), mutual_inf / energy_weight, epoch)
            config.writer.add_scalar('task{}/energy_neuron'.format(task + 1), energy_neuron, epoch)
            config.writer.add_scalar('task{}/energy_weight'.format(task + 1), energy_weight, epoch)
            config.writer.add_scalar('task{}/train_loss'.format(task + 1), train_loss, epoch)

        # Compute the test accuracies on all tasks and the mean accuracy over all tasks
        task_accuracies = torch.tensor([
            test.accuracy(mlp, tester, cfg['test_samples'])
            for tester in test_loaders
        ])
        task_mutual_inf = torch.tensor([
            metric.mutual_information(mlp, tester)
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

    # Compute the full mutual information on all test sets if there is multiple tasks
    if len(test_loaders) > 1:
        test_loader_all = data._create_dataloader(
            torch.utils.data.ConcatDataset([
                tester.dataset for tester in test_loaders
            ]), is_train=False, batch_size=cfg['dataset']['kwargs']['batch_size'])

        results['mutual_inf_all'] = metric.mutual_information(mlp, test_loader_all)

    return results, mlp, test_loaders


if __name__ == '__main__':
    # Load configuration
    user_config = parse_shell_args(sys.argv[1:])
    cfg = load_default_config(user_config["task"])
    cfg.update(user_config)

    # Setup global logger and logging directory
    if cfg['prob_release'] is None:
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_mlp_" + cfg['dataset']['name']
    else:
        run_id = time.strftime("%Y%m%d_%H%M%S") + "_prob-mlp_" + cfg['dataset']['name']

    config.setup_logging(run_id, dir=cfg['log_dir'])

    # Run the script using the created paramter configuration
    results, mlp, test_loaders = run_mlp_continual(cfg, verbose=True)

    # Save the configuration as json
    utils.save_dict_as_json(cfg, run_id, config.log_dir)

    # Store results as pickle
    results['cfg'], results['state_dict'] = cfg, mlp.state_dict()
    torch.save(results, os.path.join(config.log_dir, run_id + "_results.pt"))

    # Zip the tensorboard logging results and remove the folder to save space
    config.writer.close()
    path_tensorboard = os.path.join(config.log_dir, run_id + "_tensorboard")
    shutil.make_archive(path_tensorboard, 'zip', path_tensorboard)
    shutil.rmtree(path_tensorboard)
