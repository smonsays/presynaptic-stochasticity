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

from lib import config, data, test, metric, nets, train, utils


def create_multitask_dataloader(name, **kwargs):
    """
    Load multitask datasets
    """
    if name == "split_fmnist":
        train_loader, test_loaders = data.create_multitask_split_fmnist_loader(**kwargs)
    elif name == "split_mnist":
        train_loader, test_loaders = data.create_multitask_split_mnist_loader(**kwargs)
    elif name == "perm_fmnist":
        train_loader, test_loaders = data.create_multitask_perm_fmnist_loader(**kwargs)
    elif name == "perm_mnist":
        train_loader, test_loaders = data.create_multitask_perm_mnist_loader(**kwargs)
    else:
        raise ValueError("Dataset \"{}\" undefined".format(name))

    return train_loader, test_loaders


def load_default_config(task):
    """
    Load default parameter configuration from file.
    Args:
        tasks: String with the task name

    Returns:
        Dictionary of default parameters for the given task
    """
    if task == "perm_fmnist":
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
    Parse shell arguments for this script and return as dictionary
    """
    parser = argparse.ArgumentParser(description="Train a MultilayerPerceptron on a continual task as a multitask.")

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
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS, help="Random seed for pytorch")
    parser.add_argument("--task", choices=["perm_fmnist", "perm_mnist", "split_fmnist", "split_mnist"],
                        default="perm_mnist", help="Continual task to be solved.")
    parser.add_argument("--weight_decay", type=float, default=argparse.SUPPRESS,
                        help="Weight decay (~l2-regularization strength) of the optimizer.")

    # Parse arguments to dictionary
    return vars(parser.parse_args(args))


def run_mlp_multitask(cfg, verbose=True):

    # Initialize seed if specified (might slow down the model)
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])

    # Load multitask datasets
    train_loader, test_loaders = create_multitask_dataloader(cfg['dataset']['name'], **cfg['dataset']['kwargs'])

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
        "energy": torch.zeros(cfg['epochs']),
        "task_mutual_inf": torch.zeros(cfg['epochs'], len(test_loaders)),
        "task_acc": torch.zeros(cfg['epochs'], len(test_loaders)),
        "test_acc": torch.zeros(cfg['epochs']),
        "train_acc": torch.zeros(cfg['epochs']),
        "train_loss": torch.zeros(cfg['epochs']),
    }

    # Train for the specified amount of epochs
    for epoch in range(cfg['epochs']):
        train_acc, train_loss = train.train(mlp, train_loader, method="standard", optimizer=optimizer, verbose=verbose)

        # Compute the test accuracies and mutual inf on all tasks and the energy
        task_accuracies = torch.tensor([
            test.accuracy(mlp, tester, cfg['test_samples'])
            for tester in test_loaders
        ])
        task_mutual_inf = torch.tensor([
            metric.mutual_information(mlp, tester)
            for tester in test_loaders
        ])

        # Store results
        results['energy'][epoch] = mlp.energy
        results['task_acc'][epoch] = task_accuracies
        results['task_mutual_inf'][epoch] = task_mutual_inf
        results['train_acc'][epoch] = train_acc
        results['train_loss'][epoch] = train_loss

        # Logging
        if verbose:
            logging.info("epoch {}: train_acc {:.4f} \t test_acc {:.4f} \t mutual_inf {:.4f} \t energy: {:.4f} \t bit/energy: {:.4f}".format(epoch + 1, train_acc, torch.mean(task_accuracies), torch.mean(task_mutual_inf), mlp.energy, torch.mean(task_mutual_inf) / mlp.energy))

        config.writer.add_scalars('accuracy', {'train': train_acc, 'test': torch.mean(task_accuracies)}, epoch)
        config.writer.add_scalar('information', torch.mean(task_mutual_inf), epoch)
        config.writer.add_scalar('info_per_energy', torch.mean(task_mutual_inf) / mlp.energy, epoch)
        config.writer.add_scalar('energy', mlp.energy, epoch)
        config.writer.add_scalar('train_loss', train_loss, epoch)

        task_accuracies_dict = {"task{}": task_acc.item() for task, task_acc in enumerate(task_accuracies)}
        config.writer.add_scalars('task_accuracies', task_accuracies_dict, epoch)

    # Compute the joint mutual information on all test sets
    test_loader_all = data._create_dataloader(
        torch.utils.data.ConcatDataset([
            tester.dataset for tester in test_loaders
        ]), is_train=False, batch_size=cfg['dataset']['kwargs']['batch_size'])

    results['mutual_inf_all'] = metric.mutual_information(mlp, test_loader_all)

    return results, mlp, test_loaders


if __name__ == '__main__':
    # Parse shell arguments as input configuration
    user_config = parse_shell_args(sys.argv[1:])

    # Load default parameter configuration from file
    cfg = load_default_config(user_config["task"])

    # Overwrite default parameters with user configuration where specified
    cfg.update(user_config)

    # Setup global logger and logging directory
    run_id = time.strftime("%Y%m%d_%H%M%S") + "_mlp_multitask_" + cfg['dataset']['name']
    config.setup_logging(run_id, dir=cfg['log_dir'])

    # Run the script using the created paramter configuration
    results, mlp, test_loaders = run_mlp_multitask(cfg)

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
