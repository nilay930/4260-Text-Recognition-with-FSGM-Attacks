import numpy as np
import matplotlib.pyplot as plt


mnist_mean = 0.1307
mnist_std = 0.3081


def visualize_example(x_img, y_probs, b_unnormalize=True, label=-1,
                      filename=None):
    """
    Parameters:
    ------------------------------
    x_img: 1D numpy array of length 784 containing the image to display
    b_unnormalize: boolean, If set true, the image will be unnormalized
                   (i.e., inverse of standardization will be applied)
    label: an integer value representing the class of given handwritten digit image
    filename: string, when provided, the resulting plot will be saved with
              the given name

    Returns:
    ------------------------------
    None
    """
    img = x_img.reshape(28, 28)

    if b_unnormalize:
        x_img = unnormalize(x_img)

    if y_probs.ndim > 1:
        y_probs = y_probs.ravel()

    fig, ax = plt.subplots(ncols=2, figsize=(6.6,3))
    ax[0].imshow(img, cmap='Greys')
    ax[0].set_axis_off()
    ax[0].set_title('Generated Image')
    
    x_class = np.arange(10)
    max_prob = np.amax(y_probs)
    mask = y_probs < max_prob
    
    ax[1].bar(x_class[mask], y_probs[mask], align='center', color='C0', alpha=0.8)
    ax[1].bar(x_class[~mask], y_probs[~mask], align='center', color='C1', alpha=0.8)
    ax[1].set_xticks(x_class, [str(c) for c in x_class])
    ax[1].set_xlabel("Classes", fontsize=13)
    ax[1].set_ylabel("Class Probability", fontsize=13)

    if label >= 0:
        ax[1].set_title(f'Class label: {label}')

    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(top=0.9, bottom=0.18, wspace=0.3)

    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)
    plt.close(fig)


def onehot_encoding(y, num_classes=10):
    """
    Convert the given array of integers into their corresponding one-hot vectors
    For example, 1 is represented by [0, 1, 0, 0, ..., 0] and 2 by [0, 0, 1, 0, ..., 0, 0].
    """
    encoded_y = np.eye(num_classes)[y]

    return encoded_y


def normalize(X):
    """
    standardize the given input image
    """
    X = X / 255.0
    X = (X - mnist_mean) / mnist_std
    return X


def unnormalize(X):
    """
    apply the inverse operations of standardization
    """
    X = (X * mnist_std) + mnist_mean
    X *= 255

    return X.astype(np.uint8)


def cross_entropy_loss(y_hat, y):
    """
    The cross-entropy loss function
    """
    batch_size = y.shape[0]
    # to prevent the log from taking 0 as input
    eps = np.finfo(float).eps
    y_ = onehot_encoding(y)
    entropy = -np.sum(y_ * np.log(y_hat + eps))

    return entropy / batch_size


def minibatches(X, Y, batch_size):
    """
    A python generator to split a dataset into mini-batches of the given size
    """
    n_samples = X.shape[0]

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_x, batch_y = X[start_idx:end_idx], Y[start_idx:end_idx]

        yield batch_x, batch_y


if __name__ == "__main__":
    x = np.random.uniform(-0.1, 0.1, size=784)
    y_prob = np.zeros(10)
    y_prob[1] = 1
    visualize_example(x, y_prob)
