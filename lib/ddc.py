"""
Copyright (c) Simon Schug
All rights reserved.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import torch


def _register_dropconnect_pre_hook(module, prob_init, normalise, fixed_probs=False):
    """
    Monkey patches a torch.nn.Module applying DropConnect to its weights by registering a forward_pre_hook.

    Args:
        module:     torch.nn.Module whose weights should be Bernoulli sampled
        prob_init:  Initial value for all transmission probabilities
        normalise:  Normalise sampled weights such that gradients correspond to expected values
        fixed_prob: Boolean that if enabled uses fixed transmission probabilities during sampling
                    but keeps a buffer of (virtual) transmission probabilities to be used for learning rate modulation.
    """
    # Obtain a list of all weight parameters of the module
    weight_names = [name for name, _ in module.named_parameters() if "weight" in name]

    # Remove the registered weight parameters and register new raw parameters
    # and buffers for the transmission probabilities
    for name in weight_names:
        param = getattr(module, name)
        del module._parameters[name]

        module.register_parameter(name + "_raw", torch.nn.Parameter(param))
        # module.register_buffer(name + "_prob", prob_init + (torch.rand_like(param)) * (1.0 - prob_init))
        module.register_buffer(name + "_prob", torch.full_like(param, prob_init))

    # Define the forward_pre_hook
    def dropconnect_hook(module, input):
        for name in weight_names:
            raw_param = getattr(module, name + "_raw")
            prob_param = getattr(module, name + "_prob")

            # Sample a binary mask, apply and normalise
            if fixed_probs:
                mask = torch.bernoulli(prob_init * torch.ones_like(raw_param))
            else:
                mask = torch.bernoulli(prob_param)

            param = (mask * raw_param)
            if normalise:
                param /= prob_param

            setattr(module, name, param)

        return None

    # Register the forward_pre_hook
    module.register_forward_pre_hook(dropconnect_hook)


class DynamicDropConnect(torch.nn.Module):
    """
    Wraps a neural network model in the DynamicDropconnect algorithm.
    """
    def __init__(self, network, learning_rate, grad_threshold, prob_drift_down, prob_drift_up,
                 prob_freeze, prob_max=1.0, prob_min=0.25, fixed_bias=True, fixed_probs=None,
                 hard_freeze=False, manual_freeze=False, lr_modulation=True, weight_decay=0.0, normalise=True):
        """
        Args:
            network: Neural network derived from nets.NeuralNetwork class
            learning_rate: SGD learning rate for the weights
            grad_threshold: Scalar threshold for when gradients are considered to be "large"
            prob_drift_down: Base amount by which transmission probabilities are decreased when gradients surpass the threshold
            prob_drift_up: Base amount by which transmission probabilities are increased when gradients surpass the threshold
            prob_freeze: Probability threshold after which probabilities are kept frozen
            prob_max: Maximum transmission probability (default: 1.0)
            prob_min: Minimum transmission probability, used for initialisation (default: 0.25)

            fixed_bias: Boolean flag if bias is kept fixed during learning
            fixed_probs: Optional float keeping transmission probabilities fixed in the forward pass
            hard_freeze: Boolean flag additionally freezing weights once transmission probabilities are frozen
            manual_freeze: Boolean flag to control freezing manually instead of using the threshold mechanism
            lr_modulation: Boolean flag to control whether learning rate is modulated by transmission probabilities
            weight_decay: Optional float to enable weight decay in SGD parameter updates
            normalise: Boolean flag to control whether sampled parameters are normalised by transmission probabilities
        """
        super().__init__()
        self.network = network

        # Probability update dynamics
        self.learning_rate = learning_rate
        self.grad_threshold = grad_threshold
        self.prob_drift_down = prob_drift_down
        self.prob_drift_up = prob_drift_up
        self.prob_freeze = prob_freeze
        self.prob_max = prob_max
        self.prob_min = prob_min

        # Configuration
        self.fixed_bias = fixed_bias
        self.fixed_probs = fixed_probs
        self.hard_freeze = hard_freeze
        self.manual_freeze = manual_freeze
        self.lr_modulation = lr_modulation
        self.weight_decay = weight_decay
        self.normalise = normalise

        if self.manual_freeze or not(self.fixed_bias):
            raise NotImplementedError

        # Apply dropconnect forward hook to every layer
        for module in self.network.layers:
            if self.fixed_probs:
                _register_dropconnect_pre_hook(module, self.fixed_probs, self.normalise, fixed_probs=True)
            else:
                _register_dropconnect_pre_hook(module, self.prob_min, self.normalise)

        # In the case of no normalisation, weight initialisations must be scaled for a fair comparison
        with torch.no_grad():
            if not self.normalise:
                if self.fixed_probs is None:
                    for w in self.weights:
                        w.multiply_(1.0 / self.prob_min)
                else:
                    for w in self.weights:
                        w.multiply_(1.0 / self.fixed_probs)

    @property
    @torch.no_grad()
    def energy(self):
        if self.normalise:
            # In case of normalisation weights already contain expected values
            return torch.sum(torch.cat([torch.abs(w).view(-1) for w in self.weights]))
        else:
            # In case of no normalisation expected weight magnitude is given by p * |w|
            return torch.sum(torch.cat([(p * torch.abs(w)).view(-1) for p, w in zip(self.weight_probs, self.weights)]))

    @property
    def output_dim(self):
        return self.network.output_dim

    @property
    def weight_probs(self):
        return [value for key, value in self.named_buffers(recurse=True) if "weight_prob" in key]

    @property
    def weights(self):
        return [value for key, value in self.named_parameters(recurse=True) if "weight_raw" in key]

    @property
    def frozen_weight_mask(self):
        if self.manual_freeze:
            # Weights are frozen manually when self.update_weight_freezing() is invoked
            raise NotImplementedError
        else:
            # All weights with a probability greater equal the freeze probability are automatically frozen
            return [torch.ge(value, self.prob_freeze) for key, value in self.named_buffers(recurse=True) if "_prob" in key]

    def forward(self, x):
        return self.network(x)

    def activation(self, x):
        return self.network.activation(x)

    @torch.no_grad()
    def update_params(self, gradients):
        """
        Update parameters using learning rate modulation based on transmission probability.

        Args:
            gradients: list of gradients for all parameters of self.network

        NOTE:
            See discussion [here](https://stackoverflow.com/questions/59018085/can-i-specify-kernel-weight-specific-learning-rates-in-pytorch)
            on how a parameter-specific learning rate can be handled by torch.optim.Optimizer
            Alternatively, ddc could be thought of to be an optimizer itself (probably the most Pytorch-way of doing things).
            However, this also adds substantial complexity.
        """
        grad_idx = 0
        for module in self.network.layers:
            for param_name, param in module.named_parameters():
                if "weight" in param_name:
                    d_p = gradients[grad_idx]
                    probs = getattr(module, param_name.replace("_raw", "") + "_prob")

                    # Hard freezing forbids any changes to frozen weights
                    if self.hard_freeze:
                        frozen_mask = torch.ge(probs, self.prob_freeze)
                        d_p = d_p * (~frozen_mask)

                    # Modulate the learning rate of weights according to their transmission probabilities
                    if self.lr_modulation:
                        d_p = d_p * (self.prob_max - probs)

                    # Weight decay
                    if self.weight_decay != 0:
                        d_p = d_p.add(param, alpha=self.weight_decay)

                    # Update the weight
                    param.add_(d_p, alpha=-self.learning_rate)

                elif ('bias' in param_name):
                    if not(self.fixed_bias):
                        # Update the bias
                        param.add_(gradients[grad_idx], alpha=-self.learning_rate)

                else:
                    raise ValueError("There is a parameter that is neither a weight nor a bias: {}".format(param_name))

                grad_idx += 1

    def update_probs(self, gradients):
        """
        Update transmission probabilities for all weights.

        Args:
            gradients: List of gradients for _all_ network modules
        """
        grad_idx = 0
        for module in self.network.layers:
            for param_name, _ in module.named_parameters():
                if "weight" in param_name:
                    probs = getattr(module, param_name.replace("_raw", "") + "_prob")

                    frozen_mask = torch.ge(probs, self.prob_freeze)
                    update_mask = torch.ge(torch.abs(gradients[grad_idx]), self.grad_threshold)

                    delta = self.prob_drift_up * update_mask.float() - self.prob_drift_down * (~update_mask).float()
                    scaler = (self.prob_max - probs) * (~frozen_mask).float()
                    new_probs = torch.clamp(probs + scaler * delta, min=self.prob_min, max=self.prob_max)

                    setattr(module, param_name.replace("_raw", "") + "_prob", new_probs)

                grad_idx += 1
