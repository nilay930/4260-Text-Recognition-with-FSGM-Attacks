import numpy as np
import pickle
import matplotlib.pyplot as plt

from act_func import sigmoid, derivative_sigmoid, softmax
from linear import Linear
from neural_net import MLP
from util import onehot_encoding, unnormalize, cross_entropy_loss, visualize_example

# Question 4 

def generate_image_for_class(model, target_class):
    alpha=0.1 
    lr=0.1
    max_iters=1000
    
    x = np.random.uniform(-alpha, alpha, size=784)
    for _ in range(max_iters):
        y_pred = model.forward(x)        
        grad_input = model.grad_wrt_input(x, np.array([target_class]))
    
        x = x - (lr * grad_input.ravel())
                
        if np.argmax(y_pred) == target_class:
            break

    visualize_example(x, y_pred, b_unnormalize = True, label = target_class, filename = f'targeted_random_img_class_{target_class}.png')

# Question 5 

def fgsm(x_test, y_test, model, eps=0.05):

    xorig = x_test.ravel()    
    x_adv = x_test.ravel()
        
    while True:
                
        y_pred = model.forward(x_adv)
        if np.argmax(y_pred) != y_test:
            print("dne")
            break
        
        grad_input = model.grad_wrt_input(x_test, np.array([y_test]))        
        x_adv = x_adv + eps * np.sign(grad_input) 
        
    return x_adv

# Driver that runs both Q4 and Q5

def main():
    model = None
    with open('trained_model.pkl', 'rb') as fid:
        model = pickle.load(fid)
    
    for c in range(10):
        generate_image_for_class(model, c)

    mnist = None
    with open('mnist.pkl', 'rb') as fid:
        mnist = pickle.load(fid)
    
    mnist_image_num = 0
    x_test = mnist['test_images'][mnist_image_num]
    y_test = mnist['test_labels'][mnist_image_num]
    
    x_adv = fgsm(x_test, y_test, model)
    y_pred2 = model.forward(x_adv)
    
    visualize_example(x_adv, y_pred2, b_unnormalize = True, label = y_test, filename = "FGSM_untargeted.png")

if __name__ == "__main__":
    main()