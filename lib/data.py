"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import logging
import os

import torch
from torchvision import datasets, transforms

from lib import sampler

# NOTE: Change the path here, to use a different folder for datasets
# The default store data in a folder next to the project folder
__DATAPATH__ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data")


def create_dataloader(name, **kwargs):
    """
    Return test and train dataloader given name of dataset.

    Args:
        name: Name of the dataset
        kwargs: Additional keyword arguments specific to the dataset

    Returns:
        train_loaders: List of torch.utils.data.DataLoader for all training sets
        test_loaders: List of torch.utils.data.DataLoader for all test sets
    """
    if name == "mnist":
        train_loader, test_loader = create_mnist_loader(**kwargs)
        # Wrap in list to mimic a T=1 continual task
        train_loaders, test_loaders = [train_loader], [test_loader]
    elif name == "perm_mnist":
        train_loaders, test_loaders = create_perm_mnist_loader(**kwargs)
    elif name == "perm_mnist_cont":
        train_loaders, test_loaders = create_continuous_perm_mnist_loader(**kwargs)
    elif name == "perm_fmnist":
        train_loaders, test_loaders = create_perm_fmnist_loader(**kwargs)
    elif name == "split_mnist":
        train_loaders, test_loaders = create_split_mnist_loader(**kwargs)
    elif name == "split_fmnist":
        train_loaders, test_loaders = create_split_fmnist_loader(**kwargs)
    else:
        raise ValueError("Dataset \"{}\" undefined".format(name))

    return train_loaders, test_loaders


def _create_dataloader(dataset, is_train, batch_size, num_workers=4):
    """
    Create a dataloader for a given dataset.

    Args:
        dataset: torch.utils.data.Dataset used in the dataloader
        is_train: Boolean indicating whether to use the train or test set
        batch_size: Number of training samples per batch
        num_workers: Optional integer of how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        torch.utils.data.DataLoader
    """
    # For GPU acceleration store dataloader in pinned (page-locked) memory
    pin_memory = True if torch.cuda.is_available() else False

    # Create dataloader objects
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=is_train,
                                             num_workers=num_workers, pin_memory=pin_memory)

    return dataloader


def _create_multitask_loader(train_datasets, test_datasets, batch_size, num_workers=4):
    """
    Create multitask train- and testloaders.

    Args:
        train_datasets: list of torch.utils.data.Dataset for training
        test_datasets: list of torch.utils.data.Dataset for testing
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: Single torch.utils.data.DataLoader for all training tasks
        test_loaders: List of torch.utils.data.DataLoader for all test tasks
    """
    # Combine all train datasets into a single big data set
    train_dataset_all = torch.utils.data.ConcatDataset(train_datasets)

    # Create a single big train loader containing all the tasks
    train_loader = _create_dataloader(train_dataset_all, True, batch_size, num_workers)

    # Create a list of data loaders for each test set
    test_loaders = [
        _create_dataloader(test, False, batch_size, num_workers)
        for test in test_datasets
    ]

    return train_loader, test_loaders


def _create_split_datasets(dataset, permutation):
    """
    Create a list of datasets split into pairs by target.

    Args:
        dataset: torch.utils.data.Dataset used in the dataloader
        permutation: Tensor of permuted indeces used to permute the images

    Returns:
        split_datasets_tasks: List of datasets grouped by label pairs
    """
    # Get the indices for samples from the different classes
    split_indices = [torch.nonzero(dataset.targets == label).squeeze() for label in range(len(dataset.classes))]

    # Adapt the targets such that even task have a 0 label and odd tasks have 1 label
    dataset.targets = permutation[dataset.targets] % 2

    # Create Subsets of the whole dataset
    split_datasets = [torch.utils.data.Subset(dataset, indices) for indices in split_indices]

    # Shuffle tasks given specified permutation
    split_datasets = [split_datasets[i] for i in torch.argsort(permutation)]

    # Re-concatenate the datasets in pairs
    split_datasets_tasks = [
        torch.utils.data.ConcatDataset([even, odd])
        for even, odd in zip(split_datasets[:-1:2], split_datasets[1::2])
    ]

    return split_datasets_tasks


def _create_split_loader(dataset, is_train, batch_size, permutation, num_workers=4):
    """
    Create a dataloader for a continual split task given a dataset.

    Args:
        dataset: torch.utils.data.Dataset to be split in pairs by target
        is_train: Boolean indicating whether to use the train or test set
        batch_size: Number of training samples per batch
        permutation: Tensor of permuted indeces used to permute the images
        num_workers: Integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        List of torch.utils.data.DataLoader
    """
    # Re-concatenate the datasets in pairs
    split_datasets_tasks = _create_split_datasets(dataset, permutation)

    # Create the dataloader
    split_loaders = [
        _create_dataloader(dataset, is_train, batch_size, num_workers)
        for dataset in split_datasets_tasks
    ]

    return split_loaders


def _permute_tensor(input, permutation):
    """
    Permute elements of tensor given a matrix of permutation indices

    Args:
        input: torch.Tensor to be permuted
        permuation: Permutation of the indeces of all elements of input (flat)

    Returns:
        Permuted input
    """
    # Cache the original dimensions
    dimensions = input.size()

    # Apply the permutation to the flattened tensor
    output_flat = torch.index_select(input.view(-1), 0, permutation)

    # Restore original dimensions
    output = output_flat.view(dimensions)

    return output


def create_continuous_perm_mnist_loader(num_tasks, epochs, batch_size, transition_portion, num_workers=4):
    """
    Create a single big dataloader for the contninuous permuted MNIST-n task with multiple test loaders.

    Args:
        num_tasks: Number of permuted MNIST tasks to generate
        epochs: Number of epochs to train per task
        batch_size: Number of training samples per batch
        transition_portion: Float between 0 and 1 determining the portion
            of the task steps that transitions to the next task
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: Single big training torch.utils.data.DataLoader for all tasks and epochs
        test_loaders: List of test torch.utils.data.DataLoader for the individual tasks
    """
    # Generate permuted MNIST datasets
    permutations = [torch.randperm(28 * 28) for i in range(num_tasks)]
    train_datasets = [get_perm_mnist_dataset(p, is_train=True) for p in permutations]
    test_datasets  = [get_perm_mnist_dataset(p, is_train=False) for p in permutations]

    # Combine all train datasets into a single big data set
    train_dataset_all = torch.utils.data.ConcatDataset(train_datasets)

    # For GPU acceleration store dataloader in pinned (page-locked) memory
    pin_memory = True if torch.cuda.is_available() else False

    # Create the sampling schedule in a continuous setting with overlap
    task_probs = sampler.create_task_probs(num_tasks, epochs * len(train_datasets[0]), transition_portion)

    # Split the schedule into distinct tasks
    task_probs_splits = torch.split(task_probs, epochs * len(train_datasets[0]))
    train_samplers = [
        sampler.ContinualMultinomialSampler(len(train_datasets[0]), num_tasks, t)
        for t in task_probs_splits
    ]

    # Create distinct trainloaders per task although there is overlap and a single task
    # contains transitions in and out of other tasks
    train_loaders = [
        torch.utils.data.DataLoader(train_dataset_all, batch_size=batch_size, sampler=t, pin_memory=pin_memory)
        for t in train_samplers
    ]

    # Create a list of data loaders for each test set
    test_loaders = [
        _create_dataloader(test, False, batch_size, num_workers)
        for test in test_datasets
    ]

    return train_loaders, test_loaders


def create_fashion_mnist_loader(batch_size, num_workers=4):
    """
    Create train- and testloader for fashion MNIST.

    Args:
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: List of training torch.utils.data.DataLoader for the individual tasks
        test_loader: List of test torch.utils.data.DataLoader for the individual tasks
    """
    # Get fashion MNIST
    train_dataset = get_fmnist_dataset(True)
    test_dataset  = get_fmnist_dataset(False)

    # Create dataloader objects
    train_loader = _create_dataloader(train_dataset, True, batch_size, num_workers)
    test_loader  = _create_dataloader(test_dataset, False, batch_size, num_workers)

    return train_loader, test_loader


def create_mnist_loader(batch_size, num_workers=4):
    """
    Create a single train- and testloader for standard MNIST

    Args:
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: List of training torch.utils.data.DataLoader for the individual tasks
        test_loader: List of test torch.utils.data.DataLoader for the individual tasks
    """
    # Get permuted MNIST
    train_dataset = get_mnist_dataset(True)
    test_dataset  = get_mnist_dataset(False)

    # Create dataloader objects
    train_loader = _create_dataloader(train_dataset, True, batch_size, num_workers)
    test_loader  = _create_dataloader(test_dataset, False, batch_size, num_workers)

    return train_loader, test_loader


def create_multitask_split_fmnist_loader(shuffle_tasks, batch_size, num_workers=4):
    """
    Create Multitask split fashion MNIST train- and testloader.

    Args:
        shuffle_tasks: Bool determining if task order should be shuffled
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: Single torch.utils.data.DataLoader for all training tasks
        test_loaders: List of torch.utils.data.DataLoader for all test tasks
    """
    if shuffle_tasks:
        permutation = torch.randperm(10)
        logging.info("Permutation for split fashionMNIST:{}".format(permutation))
    else:
        permutation = torch.arange(10)

    # Generate split fashionMNIST datsets
    train_datasets = _create_split_datasets(get_fmnist_dataset(True), permutation)
    test_datasets  = _create_split_datasets(get_fmnist_dataset(False), permutation)

    return _create_multitask_loader(train_datasets, test_datasets, batch_size, num_workers)


def create_multitask_split_mnist_loader(shuffle_tasks, batch_size, num_workers=4):
    """
    Create Multitask split MNIST train- and testloader.

    Args:
        shuffle_tasks: Bool determining if task order should be shuffled
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: Single torch.utils.data.DataLoader for all training tasks
        test_loaders: List of torch.utils.data.DataLoader for all test tasks
    """
    if shuffle_tasks:
        permutation = torch.randperm(10)
        logging.info("Permutation for split MNIST:{}".format(permutation))
    else:
        permutation = torch.arange(10)

    # Generate split fashionMNIST datsets
    train_datasets = _create_split_datasets(get_mnist_dataset(True), permutation)
    test_datasets  = _create_split_datasets(get_mnist_dataset(False), permutation)

    return _create_multitask_loader(train_datasets, test_datasets, batch_size, num_workers)


def create_multitask_perm_fmnist_loader(num_tasks, batch_size, num_workers=4):
    """
    Create Multitask permuted fashion MNIST train- and testloader.

    Args:
        num_tasks: Number of permuted fashion MNIST tasks to generate
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: Single torch.utils.data.DataLoader for all training tasks
        test_loaders: List of torch.utils.data.DataLoader for all test tasks
    """
    # Generate permuted fashion MNIST datasets
    permutations = [torch.randperm(28 * 28) for i in range(num_tasks)]
    train_datasets = [get_perm_fmnist_dataset(p, is_train=True) for p in permutations]
    test_datasets  = [get_perm_fmnist_dataset(p, is_train=False) for p in permutations]

    return _create_multitask_loader(train_datasets, test_datasets, batch_size, num_workers)


def create_multitask_perm_mnist_loader(num_tasks, batch_size, num_workers=4):
    """
    Create Multitask permuted MNIST train- and testloader.

    Args:
        num_tasks: Number of permuted MNIST tasks to generate
        batch_size: Number of training samples per batch
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: Single torch.utils.data.DataLoader for all training tasks
        test_loaders: List of torch.utils.data.DataLoader for all test tasks
    """
    # Generate pMNIST datasets
    permutations = [torch.randperm(28 * 28) for i in range(num_tasks)]
    train_datasets = [get_perm_mnist_dataset(p, is_train=True) for p in permutations]
    test_datasets  = [get_perm_mnist_dataset(p, is_train=False) for p in permutations]

    return _create_multitask_loader(train_datasets, test_datasets, batch_size, num_workers)


def create_perm_fmnist_loader(batch_size, num_tasks, num_workers=4):
    """
    Create lists of train- and testloaders for permuted fashion MNIST.

    Args:
        batch_size: Number of training samples per batch
        num_tasks: Number of permuted fashion MNIST tasks to generate
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: List of training torch.utils.data.DataLoader for the individual tasks
        test_loader: List of test torch.utils.data.DataLoader for the individual tasks
    """
    # Create list of permutations
    permutations = [torch.randperm(28 * 28) for _ in range(num_tasks)]

    # Create a list of tuples of train, test loaders
    loaders = [
        create_perm_fmnist_loader_single(batch_size, p, num_workers)
        for p in permutations
    ]
    # Unpack the list of tuples into two separate lists
    train_loaders, test_loaders = map(list, zip(*loaders))

    return train_loaders, test_loaders


def create_perm_fmnist_loader_single(batch_size, permutation, num_workers=4):
    """
    Create a single train- and testloader for permuted fashion MNIST.

    Args:
        batch_size: Number of training samples per batch
        permutation: Tensor of permuted indeces used to permute the images
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: torch.utils.data.DataLoader of training data
        test_loader: torch.utils.data.DataLoader of test data
    """
    # Get permuted MNIST
    train_dataset = get_perm_fmnist_dataset(permutation, True)
    test_dataset  = get_perm_fmnist_dataset(permutation, False)

    # Create dataloader objects
    train_loader = _create_dataloader(train_dataset, True, batch_size, num_workers)
    test_loader  = _create_dataloader(test_dataset, False, batch_size, num_workers)

    return train_loader, test_loader


def create_perm_mnist_loader(batch_size, num_tasks, num_workers=4):
    """
    Create lists of train- and testloaders for permuted MNIST.

    Args:
        batch_size: Number of training samples per batch
        num_tasks: Number of permuted MNIST tasks to generate
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: List of training torch.utils.data.DataLoader for the individual tasks
        test_loader: List of test torch.utils.data.DataLoader for the individual tasks
    """

    # Create list of permutations
    permutations = [torch.randperm(28 * 28) for _ in range(num_tasks)]

    # Create a list of tuples of train, test loaders
    loaders = [
        create_perm_mnist_loader_single(batch_size, p, num_workers)
        for p in permutations
    ]
    # Unpack the list of tuples into two separate lists
    train_loaders, test_loaders = map(list, zip(*loaders))

    return train_loaders, test_loaders


def create_perm_mnist_loader_single(batch_size, permutation, num_workers=4):
    """
    Create a single train- and testloader for permuted MNIST.

    Args:
        batch_size: Number of training samples per batch
        permutation: Tensor of permuted indeces used to permute the images
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: torch.utils.data.DataLoader of training data
        test_loader: torch.utils.data.DataLoader of test data
    """
    # Get permuted MNIST
    train_dataset = get_perm_mnist_dataset(permutation, True)
    test_dataset  = get_perm_mnist_dataset(permutation, False)

    # Create dataloader objects
    train_loader = _create_dataloader(train_dataset, True, batch_size, num_workers)
    test_loader  = _create_dataloader(test_dataset, False, batch_size, num_workers)

    return train_loader, test_loader


def create_split_mnist_loader(batch_size, shuffle_tasks, num_workers=4):
    """
    Create train and test_loaders for split MNIST.

    Args:
        batch_size: Number of training samples per batch
        shuffle_tasks: Bool determining if task order should be shuffled
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: List of torch.utils.data.DataLoader for all training tasks
        test_loader: List of torch.utils.data.DataLoader for all test tasks
    """
    if shuffle_tasks:
        permutation = torch.randperm(10)
        logging.info("Permutation for splitMNIST:{}".format(permutation))
    else:
        permutation = torch.arange(10)

    train_loader = _create_split_loader(get_mnist_dataset(True), True, batch_size, permutation, num_workers)
    test_loader  = _create_split_loader(get_mnist_dataset(False), False, batch_size, permutation, num_workers)

    return train_loader, test_loader


def create_split_fmnist_loader(batch_size, shuffle_tasks, num_workers=4):
    """
    Create train and test_loaders for split fashion MNIST.

    Args:
        batch_size: Number of training samples per batch
        shuffle_tasks: Bool determining if task order should be shuffled
        num_workers: Optional integer defining how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process.

    Returns:
        train_loader: List of torch.utils.data.DataLoader for all training tasks
        test_loader: List of torch.utils.data.DataLoader for all test tasks
    """
    if shuffle_tasks:
        permutation = torch.randperm(10)
        logging.info("Permutation for split fashionMNIST:{}".format(permutation))
    else:
        permutation = torch.arange(10)

    train_loader = _create_split_loader(get_fmnist_dataset(True), True, batch_size, permutation, num_workers)
    test_loader  = _create_split_loader(get_fmnist_dataset(False), False, batch_size, permutation, num_workers)

    return train_loader, test_loader


def get_cifar10_dataset(is_train):
    """
    Get the CIFAR10 dataset.

    Args:
        is_train: Boolean indicating whether to use the train or test set

    Returns:
        dataset: CIFAR 10 torch.utils.data.Dataset
    """
    if is_train:
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    dataset = datasets.CIFAR10(root=__DATAPATH__, train=is_train, download=True, transform=transform)

    return dataset


def get_fmnist_dataset(is_train):
    """
    Create fashion MNIST data set using the letters split.

    Args:
        is_train: Boolean indicating whether to use the train or test set

    Returns:
        dataset: fashion MNIST torch.utils.data.Dataset
    """
    dataset = datasets.FashionMNIST(__DATAPATH__, train=is_train, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(0.2861,), std=(0.3530,)),
                                    ]))

    return dataset


def get_mnist_dataset(is_train):
    """
    Create MNIST data set without permutating the images.

    Args:
        is_train: Boolean indicating whether to use the train or test set

    Returns:
        dataset: MNIST torch.utils.data.Dataset
    """
    dataset = datasets.MNIST(__DATAPATH__, train=is_train, download=True, transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                             ]))
    return dataset


def get_perm_mnist_dataset(permutation, is_train):
    """
    Create MNIST data set using the specified permutation on the images.
    Note: Validate using plt.imshow(mnist.data[0])

    Args:
        permutation: Tensor of permuted indeces used to permute the images
        is_train: Boolean indicating whether to use the train or test set

    Returns:
        dataset: permuted MNIST torch.utils.data.Dataset
    """
    dataset = datasets.MNIST(__DATAPATH__, train=is_train, download=True, transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
                             lambda img: _permute_tensor(img, permutation),
                             ]))
    return dataset


def get_perm_fmnist_dataset(permutation, is_train):
    """
    Create MNIST data set using the specified permutation on the images.

    Args:
        permutation: Tensor of permuted indeces used to permute the images
        is_train: Boolean indicating whether to use the train or test set

    Returns:
        dataset: permuted fashion MNIST torch.utils.data.Dataset
    """
    dataset = datasets.FashionMNIST(__DATAPATH__, train=is_train, download=True, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.2861,), std=(0.3530,)),
                                    lambda img: _permute_tensor(img, permutation),
                                    ]))
    return dataset
