import numpy as np
from act_func import sigmoid, derivative_sigmoid, softmax
from linear import Linear
from util import onehot_encoding


class MLP(object):
    def __init__(self, layer_sizes=[784, 512, 512, 10]):
        # number of layers in the network
        self.num_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        self.layers = []

        for in_feat, out_feat in zip(layer_sizes[:-1], layer_sizes[1:]):
            self.layers.append(Linear(in_feat, out_feat, bias=True))

        # list objects to store the activations and pre-activations of neural network
        self.activations = []
        self.pre_activations = []

    def forward(self, x):
        """
        Perform the forward operation of network
        Parameters:
        ------------------------------
        x: a numpy array containing (flattened) images

        Returns:
        ------------------------------
        out: a numpy array containing the estimated class probabilities
        """
        self._clear()
        x = np.atleast_2d(x)

        out = x
        self.activations.append(out)

        for layer in self.layers[:-1]:
            # linear transformation
            out = layer.forward(out)
            self.pre_activations.append(out)
            # applying the activation function
            out = sigmoid(out)
            self.activations.append(out)

        output_layer = self.layers[-1]
        out = output_layer.forward(out)
        self.pre_activations.append(out)
        out = softmax(out)
        self.activations.append(out)

        return out

    def _clear(self):
        """
        Empty the lists
        """
        self.activations.clear()
        self.pre_activations.clear()

    def backward(self, y):
        """
        Perform the backward step (i.e., backpropagation) to compute the
        gradient (w.r.t.) parmaeters

        Parameters:
        ---------------------------
        y: a 1D numpy array containing the labels. The number of labels provided in
           the array y should match the number of examples used in the forward phase.
           For instance, consider the following example.

           out = model.forward(x)
           grad_w, grad_b, _ = model.backward(y)

           In the above, the number of images in x should match the number of labels in the
           variable y. In other words, the i-th label in y is the correct label for
           the i-th image in x.
        """
        if len(self.activations) == 0 or len(self.pre_activations) == 0:
            raise ValueError('Call backward() function after the feed-forward step')

        assert len(self.pre_activations) == self.num_layers

        batch_size = y.shape[0]
        grad_w = [None for _ in self.layers]
        grad_b = [None for _ in self.layers]
        delta = [None for _ in self.layers]

        # compute the error at the top layer
        class_prob = self.activations[-1]
        delta[-1] = class_prob - onehot_encoding(y)
        grad_w[-1] = np.dot(delta[-1].T, self.activations[-2]) / batch_size
        grad_b[-1] = delta[-1].mean(0)

        # propagate error towards bottom layers
        for ll in range(2, self.num_layers + 1):
            act_der = derivative_sigmoid(self.pre_activations[-ll])
            delta[-ll] = np.dot(delta[-ll + 1], self.layers[-ll + 1].weight) * act_der
            grad_w[-ll] = np.dot(delta[-ll].T, self.activations[-ll - 1]) / batch_size
            grad_b[-ll] = delta[-ll].mean(0)

        # clean up
        self._clear()
        return grad_w, grad_b, delta

    def grad_wrt_input(self, x, y):
        """
        This function computes the graident of loss function w.r.t.
        the input x (it's different from that w.r.t. parmaeters).
        Parmaeters:
        ----------------------------------
        x: a numpy array containing image(s)
        y: a numpy array containing label(s) for image(s) in x

        Returns:
        ----------------------------------
        A numpy array containing the gradient(s) w.r.t. input x
        """
        _, _, delta = self.backward(y)

        return np.dot(delta[0], self.layers[0].weight)
