import math
import numpy as np


class Linear(object):
    """
    A linear layer that performs:
    y = Wx + b,
    where W is the weight matrix and b is the bias.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = np.empty(shape=(out_features, in_features))
        self.bias = np.empty(shape=out_features) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        """
        Randomly initialize the parameters of layer
        The specific method implemented in here is called Kaiming Uniform initialization.
        """
        negative_slope = math.sqrt(5)
        fan_in = self.weight.shape[1]
        gain = math.sqrt(2.0 / (1 + negative_slope ** 2))
        std = gain / math.sqrt(fan_in)
        bound = math.sqrt(3.0) * std

        self.weight[:] = np.random.uniform(low=-bound,
                                           high=bound,
                                           size=self.weight.shape)[:]
        if self.bias is not None:
            bound = 1 / math.sqrt(fan_in)
            self.bias[:] = np.random.uniform(low=-bound,
                                             high=bound,
                                             size=self.bias.shape)[:]

    def forward(self, x):
        """
        Parameters:
        -------------------------
        x: input features of shape [batch size, feature size]

        Returns:
        -------------------------
        out = Wx + b
        """
        out = x @ self.weight.T

        if self.bias is not None:
            out += self.bias

        return out
