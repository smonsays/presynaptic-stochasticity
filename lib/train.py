"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import logging

import torch

from lib import config


def train(model, train_loader, method, optimizer=None, verbose=True):
    """
    Train model for a single epoch using the specified training method.

    Args:
        model: torch.nn.Module
        train_loader: torch.utils.data.DataLoader with the training dataset
        method: String specifying the training method ("standard" or "ddc")
        optimizer: torch.optim.Optimizer required when using training method "standard"
        verbose: Logging verbosity

    Returns:
        Training accuracy, cumulative training loss
    """
    # Validate inputs
    if method == "default":
        assert isinstance(optimizer, torch.optim.Optimizer)

    # Prepare model for training
    model.train(mode=True)
    model = model.to(config.device)

    correct_pred, loss = 0.0, 0.0

    # Run the training loop over the whole dataset
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(config.device), y_batch.to(config.device)

        if method == "standard":
            output, batch_loss = _train_step_mlp(x_batch, y_batch, model, optimizer)
        elif method == "ddc":
            output, batch_loss = _train_step_ddc(x_batch, y_batch, model)
        else:
            raise ValueError("Training method \"{}\" undefined".format(method))

        if torch.isnan(batch_loss):
            raise ValueError("Loss diverged to nan")

        # Compute training accuracy
        with torch.no_grad():
            loss += batch_loss
            pred = torch.max(output, dim=1)[1]
            correct_pred += torch.sum(torch.eq(pred, y_batch)).item()

            if verbose and (batch_idx % (len(train_loader) // 10) == 0):
                batch_acc = torch.sum(torch.eq(pred, y_batch)).item() / x_batch.size()[0]
                logging.info('{:.0f}%: batch_loss: {:.4f} \t batch_acc: {:.2f}'.format(100. * batch_idx / len(train_loader), batch_loss, batch_acc))

    model.train(mode=False)

    return correct_pred / len(train_loader.dataset), loss


def _train_step_ddc(x_batch, y_batch, model):
    """
    Single training step for DDC.

    Args:
        x_batch:    Batch of inputs
        y_batch:    Batch of labels
        model:      ddc.DynamicDropConnect object

    Returns:
        Model output, batch loss
    """

    # Compute forward pass on current batch
    output = model(x_batch)

    # Compute the batch loss as the cross entropy loss
    batch_loss = torch.nn.functional.cross_entropy(output, y_batch)

    # Compute gradients of model parameters wrt current batch and detach
    gradients = torch.autograd.grad(batch_loss, model.parameters())
    gradients = [g.detach() for g in gradients]

    # Optimize all parameters
    model.update_params(gradients)

    # Apply weight probability update
    model.update_probs(gradients)

    return output, batch_loss


def _train_step_mlp(x_batch, y_batch, model, optimizer):
    """
    Perform single standard training step.

    Args:
        x_batch:    Batch of inputs
        y_batch:    Batch of labels
        model:      torch.nn.Module
        optimizer:  torch.optim.Optimizer for model parameters

    Returns:
        Model output, batch loss
    """
    # Compute forward pass on current batch
    output = model(x_batch)

    # Compute batch loss as cross entropy loss
    batch_loss = torch.nn.functional.cross_entropy(output, y_batch)

    # Compute gradients wrt current batch loss and perform parameter update
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()

    return output, batch_loss
