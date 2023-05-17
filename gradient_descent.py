import numpy as np
import math


#returns index of the class with highest probablity 
def highest(output_layer):
    for i in output_layer:
        if output_layer[i] > max:
            max = output_layer[i]

#returns relu of x
def leaky_relu(x):
    if x > 0:
        return x
    else:
        return x*0.01
    
def leaky_relu_derivative(x):
    if x > 0:
        return 1
    else:
        return 0.01
    
def weighted_sum(last_layer, all_weights, biases, layer_length):
    layer = [0]*layer_length
    for i in range(layer_length):
        layer[i] = biases[i]
        products = np.dot(last_layer, all_weights[:,i])
        layer[i] += products
    return layer

def hidden_layer(weighted_sum, layer_length):
    
    layer = [0]*layer_length
    for i in range(layer_length):
        layer[i] = leaky_relu(weighted_sum[i])
    return layer


def softmax(layer, layer_length):
    sum = 0
    max_val = np.max(layer)
    for i in range(layer_length):
        sum += math.exp(layer[i] - max_val)
    
    for i in range(layer_length):
        layer[i]= math.exp(layer[i] - max_val)/sum

    return layer


#expected_prob = gradient_descent.output_layer(hidden_layer_2_output, output_weights, output_biases, labels_length, y_train[h]) 
def output_layer(sum, layer_length):
    layer = softmax(sum, layer_length)
    return layer
    
    
