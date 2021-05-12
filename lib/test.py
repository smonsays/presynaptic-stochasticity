"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import copy
import logging

import numpy as np
import torch
from scipy.stats import pearsonr


from lib import config, data, metric


@torch.no_grad()
def accuracy(model, test_loader, num_samples):
    """
    Compute accuracy on test set by sampling multiple models per batch

    Args:
        model: Trained model derived from nets.NeuralNetwork
        test_loader: torch.utils.data.DataLoader containing the test samples
        num_samples: Number of times to sample the network per test sample

    Returns:
        Accuracy
    """

    # Prepare the model for testing
    model.train(mode=False)
    model = model.to(config.device)

    tot_test, tot_acc = 0.0, 0.0

    # Count correct predictions for all data points in test set
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

        # Compute average output probabilities over multiple runs
        out = torch.zeros(x_batch.size(0), model.output_dim, device=config.device)
        for s in range(num_samples):
            out += torch.nn.functional.softmax(model(x_batch), dim=1)
        out /= num_samples

        # Batch accuracy
        pred = torch.max(out, dim=1)[1]
        acc = pred.eq(y_batch).sum().item()

        tot_acc += acc
        tot_test += x_batch.size()[0]

    return tot_acc / tot_test


@torch.no_grad()
def activation(model, test_loader, num_samples):
    """
    Compute the summed activation of all neurons averaged over multiple inputs and samples.

    Args:
        model: Trained model derived from nets.NeuralNetwork
        test_loader: torch.utils.data.DataLoader containing the test samples
        num_samples: Number of times to sample the network per test sample

    Returns:
        Average activation (neuron energy)
    """
    # Prepare the model for testing
    model.train(mode=False)
    model = model.to(config.device)
    energy = torch.zeros(len(test_loader), num_samples, test_loader.batch_size, device=config.device)

    for idx_batch, (x_batch, y_batch) in enumerate(test_loader):
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

        for s in range(num_samples):
            energy[idx_batch][s] = model.activation(x_batch)

    return torch.mean(energy)


def fisher_importance_correlation(dyn_mlp, test_loaders):
    """
    Evaluating the Fisher Information of model parameters and correlating
    it with the corresponding importance parameters for a DynamicDropConnect network.

    Args:
        dyn_mlp: ddc.DynamicDropConnect object
        test_loaders: List of test loaders

    Returns:
        Correlation between fisher and release probabilities, p-value, fisher diagonal as vector
    """
    # Store the original model state to be restored after the experiment
    original_model_state = copy.deepcopy(dyn_mlp.state_dict())

    # Concatenate all datasets into single data loader with batch_size=1
    test_sets = torch.utils.data.ConcatDataset([
        tester.dataset for tester in test_loaders
    ])
    fisher_loader = data._create_dataloader(test_sets, False, 1)

    # Extract the transmission probabilities
    probs_long = torch.cat([
        val.reshape(-1) for key, val in dyn_mlp.network.named_buffers() if "weight_prob" in key
    ])

    # Set all transmission probabilities manually to 1
    for key, val in dyn_mlp.network.named_buffers():
        if "_prob" in key:
            val.fill_(1.0)

    # Compute the diagonal of the fisher information for all model parameters
    fisher_diag = metric.fisher_information(dyn_mlp, fisher_loader)

    # Correlate the fisher information diagonal with the learned probabilities
    fisher_diag_long = torch.cat([f.reshape(-1) for f in fisher_diag]).detach()
    corr_coeff, p_value = pearsonr(fisher_diag_long.cpu().numpy(), probs_long.cpu().numpy())

    # Restore the original model parameters
    dyn_mlp.load_state_dict(original_model_state)

    return corr_coeff, p_value, fisher_diag_long


@torch.no_grad()
def importance_lesion(model, parameters, importances, test_loader, num_samples, step_size=10):
    """
    Ablation experiment progressively removing unimportant parameters and
    evaluating the ensuing accuracy on `test_loader`.

    Args:
        model: torch.nn.Module
        parameters: List of parameters of the model to be lesioned
        importances: List of importances with same shape as parameters
        test_loader: torch.utils.data.DataLoader of test data
        num_samples: Number of network samples to evaluate the accuracy
        step_size: Step size of percentage of parameters to be lesioned

    Returns:
        List of accuracies for each lesion step
    """
    # Store the original model state to be restored after the ablation
    original_model_state = copy.deepcopy(model.state_dict())

    # Draw a small, random perturbation for the importances to make the ordering unique
    perturbations = [1e-4 * (torch.rand_like(imp) - 0.5) for imp in importances]
    importances_perturbed = [imp + pert for imp, pert in zip(importances, perturbations)]

    # Create dictionary of percentiles within all importance measures to determine shutoff thresholds
    percentile_steps = range(step_size, 100, step_size)
    percentiles = np.percentile(torch.cat([imp.view(-1).cpu() for imp in importances_perturbed]), percentile_steps)
    accuracies = {}

    for min_imp, step in zip(percentiles, percentile_steps):

        # Determine the number of parameters to be lesioned
        num_total = sum([torch.numel(p) for p in parameters])
        num_lesioned = int((step / 100) * num_total)

        # Try the naive approach of lesioning
        masks = [torch.ge(imp, min_imp) for imp in importances_perturbed]

        # Check that this did indeed work
        num_mask = sum([torch.sum(torch.logical_not(m)) for m in masks])
        if num_mask != num_lesioned:
            logging.warning("num_mask={} != num_lesioned={}".format(num_mask, num_lesioned))

        # Set parameters whose importance is below the threshold to zero
        for p, m in zip(parameters, masks):
            p.copy_(p * m)

        # Evaluate the accuracy of the lesioned model
        accuracies["{:2f}".format(step / 100)] = accuracy(model, test_loader, num_samples)

        # Restore the original model parameters
        model.load_state_dict(original_model_state)

    return accuracies


def probability_clamp(prob_clamp, dyn_mlp, test_loader, num_samples):
    """
    Clamp all transmission probabilities to a fixed value and evaluate test accuracy.

    Args:
        prob_clamp: Value to clamp transmission probabilities to
        dyn_mlp: ddc.DynamicDropConnect object
        test_loader: torch.utils.data.DataLoader of test data
        num_samples: Number of network samples to evaluate the accuracy

    Returns:
        Test accuracy
    """
    # Store the original model state to be restored after the ablation
    original_model_state = copy.deepcopy(dyn_mlp.state_dict())

    # Clamp the probabilities
    for key, val in dyn_mlp.network.named_buffers():
        if "weight_prob" in key:
            val.fill_(prob_clamp)

    test_acc = accuracy(dyn_mlp, test_loader, num_samples)

    # Restore the original model parameters
    dyn_mlp.load_state_dict(original_model_state)

    return test_acc
