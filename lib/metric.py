"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch

from lib import config


def fisher_information(model, dataloader):
    """
    Compute the (empirical) Fisher information of model weights on a given dataloader.

    Args:
        model: Object derived from ddc.DynamicDropConnect
        dataloader: Dataloader to be used to compute the empirical Fisher

    Returns:
        Diagonal of the Fisher Information matrix
    """
    # NOTE: Only dataloader with batch_size=1 allowed to guarantee correctness
    assert dataloader.batch_size == 1

    # Prepare model for testing
    model.train(mode=False)
    model = model.to(config.device)

    # Pre-allocate list of tensors to collect squared log-likelihoods for each parameter
    d_log_squared = [
        torch.zeros_like(value)
        for key, value in model.named_parameters() if "weight" in key
    ]

    for idx, (x, y) in enumerate(dataloader):
        x, y = x.to(config.device), y.to(config.device)

        # Compute the log_likelihood of the current data point w.r.t to the observed label
        log_probs = torch.nn.functional.log_softmax(model(x), dim=1)
        log_likelihood = log_probs.squeeze()[y.data]

        # Only determine fisher information for weights, not biases
        weights = [value for key, value in model.named_parameters() if "weight_raw" in key]

        # Compute (d/ dtheta log(f(x)))**2
        grads = torch.autograd.grad(log_likelihood, weights)
        for d_log, g in zip(d_log_squared, grads):
            d_log += g**2

    # Take the expected value to get the diagonal entries of the Fisher information matrix
    fisher_diag = [d_log / len(dataloader) for d_log in d_log_squared]

    return fisher_diag


@torch.no_grad()
def mutual_information(model, dataloader):
    """
    Compute the (empirical) mutual information I(X,Y) where X is the r.v. of the true
    labels and Y is the r.v. of the learned labels.

    Args:
        model: nets.NeuralNetwork encoding the empirical learned labels given the examples from the dataloader
        dataloader: torch.utils.data.DataLoader containing samples from the true label distributions

    Returns:
        Mutual information
    """
    # P(Y|X) where rows denote X (true labels), columns denote Y (learned labels)
    # NOTE: Technically the dimensionality should be (len(dataloader.dataset.classes), model.output_dim)
    # but as ConcatDataset does not expose the `classes` attribute, we cannot use it here
    cond_dist = torch.zeros((model.output_dim, model.output_dim)).to(config.device)
    cond_dist_count = torch.zeros(model.output_dim).to(config.device)

    for batch_idx, (x_batch, y_batch) in enumerate(dataloader):
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

        # BUG: This doesn't work as intended, only one of the entries is added properly to the view for some reason
        # NOTE: This could potentially be solved by using the accumlate flag in index_put_ or put_
        # cond_dist[y_batch] += torch.nn.functional.softmax(model(x_batch), dim=1)
        # cond_dist_count[y_batch] += 1

        y_pred = torch.nn.functional.softmax(model(x_batch), dim=1)

        for label in torch.unique(y_batch):
            cond_dist[label] += y_pred[y_batch == label].sum(dim=0)
            cond_dist_count[label] += (y_batch == label).sum()

    # Normalise the conditional distribution P(Y|X)
    # NOTE: Need to divide EACH ROW by the total number of samples (hence the transposes)
    cond_dist = (cond_dist.t() / cond_dist_count).t()

    true_dist = cond_dist_count / cond_dist_count.sum()  # P(X)
    joint_dist = (cond_dist.t() * true_dist).t()  # P(X,Y)
    learned_dist = joint_dist.sum(dim=0)  # P(Y)

    # I(X,Y) = ∑_y ∑_x P(y,x) * log2(P(y|x) / P(y))
    mutual_inf = torch.sum(joint_dist * torch.log2(cond_dist / learned_dist))

    return mutual_inf
