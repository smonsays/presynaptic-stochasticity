"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import numpy as np
import torch


class ContinualMultinomialSampler(torch.utils.data.Sampler):
    """
    Samples elements in a continual setting given the multinomial
    probability distribution specified in task_probs.
    """

    def __init__(self, task_samples, num_tasks, task_probs):
        """
        Args:
            task_samples: Number of samples per task (assuming equal taks sizes)
            num_tasks: Number of tasks contained in the dataset (equal size is assumed)
            task_probs: Multinomial probability distribution specifying the probability of a sample for each iteration in a new row
        """
        # Determine the range of indeces per task in the data source
        self.task_ranges = torch.tensor([
            (t * task_samples, (t + 1) * task_samples - 1) for t in range(num_tasks)
        ])
        self.task_probs = task_probs

    def __iter__(self):
        # Sample the task id per iteration
        sampled_tasks = torch.multinomial(self.task_probs, 1).squeeze()

        # Sample the sample indeces given the sampled tasks
        # NOTE: Need to use numpy since torch.randint is not vectorized
        sample_indeces = np.random.randint(
            low=self.task_ranges[sampled_tasks][:, 0], high=self.task_ranges[sampled_tasks][:, 1]
        )

        return iter(sample_indeces)

    def __len__(self):
        # The number of samples is implictly defined by the dimension of task_probs
        return self.task_probs.size(0)


def create_task_probs(num_tasks, task_steps, transition_portion):
    """
    Create the task probabilities over the course of the whole continual task.
    NOTE: The first and last task effectively have more samples since they don't
          have an in- or out-transition

    Args:
        num_tasks: Number of tasks
        task_steps: Steps to be taken per task
        transition_portion: Portion of the task steps that transitions to other task (in or out) in [0,1]

    Returns:
        Multinomial distribution for each time step defining the probability to sample a given task
    """
    # Compute the number of transition steps for both the in- and the out-transition (hence divided by 2)
    transition_steps = int(task_steps * transition_portion / 2)

    # Initialize the whole tensor of evolving task probabilities with zeros
    probs_task = torch.zeros((task_steps * num_tasks, num_tasks))

    for task_id in range(num_tasks):
        # Initialise the 3 distinct parts: transition_in /, single -- , transition_out \
        probs_single = torch.zeros((task_steps - 2 * transition_steps, num_tasks))
        probs_transition_in = torch.zeros((transition_steps, num_tasks))
        probs_transition_out = torch.zeros((transition_steps, num_tasks))

        # Create the deterministic section where only a single task is presented as a one-hot encoding
        probs_single[:, task_id] = 1.0

        # Create the in-transition probabilities with a linear transition between two classes
        probs_transition_in[:, (task_id - 1) % num_tasks] = torch.linspace(0.5, 0.0, steps=transition_steps)
        probs_transition_in[:, task_id] = torch.linspace(0.5, 1.0, steps=transition_steps)

        # Create the out-transition probabilities with a linear transition between two classes
        probs_transition_out[:, task_id] = torch.linspace(1.0, 0.5, steps=transition_steps)
        probs_transition_out[:, (task_id + 1) % num_tasks] = torch.linspace(0.0, 0.5, steps=transition_steps)

        # Concatenate the three phases of probabilities to a single tensor and insert
        probs_task[task_id * task_steps:(task_id + 1) * task_steps, :] = torch.cat((probs_transition_in, probs_single, probs_transition_out))

    # Overwrite the first transition in and the last transition out
    probs_task[0: transition_steps, :] = torch.zeros((transition_steps, num_tasks))
    probs_task[0: transition_steps, 0] = 1.0

    probs_task[-transition_steps:, :] = torch.zeros((transition_steps, num_tasks))
    probs_task[-transition_steps:, -1] = 1.0

    return probs_task
