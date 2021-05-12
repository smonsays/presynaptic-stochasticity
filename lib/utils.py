"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import csv
import json
import math
import os

import torch
import torchvision


def compute_frozen_probs(mask_list):
    """
    Compute the ratio of weight probabilities exceeding the freezing threshold.

    Args:
        mask_list (List[Tensor]): list of binary tensors determining if weight is frozen

    Returns:
        Scalar Tensor
    """
    num_frozen = sum(torch.sum(m) for m in mask_list)
    num_total = sum(torch.numel(m) for m in mask_list)

    return num_frozen.float() / num_total


def compute_mean_probs(probs_list):
    """
    Compute the mean weight probabilities.

    Args:
        probs_list (List[Tensor]): list of tensors containing probabilities

    Returns:
        Scalar Tensor
    """
    # Concatenate all tensors into a single 1D tensor
    probs_cat = torch.cat([p.view(-1) for p in probs_list])

    return torch.mean(probs_cat)


def create_nonlinearity(name):
    """
    Return nonlinearity function given its name.
    """
    if name == "leaky_relu":
        return torch.nn.functional.leaky_relu
    elif name == "relu":
        return torch.nn.functional.relu
    elif name == "sigmoid":
        return torch.sigmoid
    elif name == "tanh":
        return torch.nn.functional.tanh
    else:
        raise ValueError("Nonlinearity \"{}\" undefined".format(name))


def create_optimizer(name, model, **kwargs):
    """
    Return optimizer for the given model.
    """
    if name == "adagrad":
        return torch.optim.Adagrad(model.parameters(), **kwargs)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), **kwargs)
    elif name == "sgd":
        return torch.optim.SGD(model.parameters(), **kwargs)
    else:
        raise ValueError("Optimizer \"{}\" undefined".format(name))


def list_to_csv(mylist, filepath):
    """
    Save list as csv file.
    """
    with open(filepath, 'w', newline='') as f:
        wr = csv.writer(f)
        wr.writerow(mylist)


def save_dict_as_json(config, name, dir):
    """
    Store a dictionary as a json text file.
    """
    with open(os.path.join(dir, name + ".json"), 'w') as file:
        json.dump(config, file, sort_keys=True, indent=4)


def show_tensor(input):
    """
    Transform tensor into PIL object and show in separate window.
    """
    image = torchvision.transforms.functional.to_pil_image(input)
    image.show()


def vector_angle(a, b):
    """
    Compute the angle between two vectors.
    """
    # Numerically stable implementation of vector angle
    # torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
    cos_theta = torch.nn.functional.cosine_similarity(a, b, dim=0)

    angle_radians = torch.acos(cos_theta)

    return 180 * (angle_radians / math.pi)
