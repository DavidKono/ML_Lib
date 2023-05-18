import numpy as np
import math


class input:
    def __init__(self, image_array):
        self.length = len(image_array[1])


class layer:
    #i will need to somehow fill in the last layer's length into this
    def __init__(self, activ_func, length, last_layer_length):
        self.activ_func = activ_func
        self.length = length

        #Glorot uniform initialization = sqrt(6/(nodes in and nodes out)), input_size is last_layer.length
        self.initial_range = math.sqrt(6/(self.length + last_layer_length))

        self.weights = np.random.uniform(low=-self.initial_range, high=self.initial_range, size=(last_layer_length, self.length))
        self.biases = np.random.uniform(low=-self.initial_range, high=self.initial_range, size=(length))

        self.weights_grad_total = np.zeros((last_layer_length, self.length))
        self.bias_grad_total = np.zeros((self.length))

        self.delta = np.zeros(self.length)


    #finds weighted sum of node inputs and the weights and biases
    def weights_and_biases(self, last_layer):
        self.weighted_sum = [0]*self.length
        for i in range(self.length):
            self.weighted_sum[i] = self.biases[i]
            products = np.dot(last_layer, self.weights[:,i])
            self.weighted_sum[i] += products
        return self.weighted_sum

    #takes weighted sum and performs activation function to get layer output
    def activate(self, weighted_sum):
        self.layer_output = self.activ_func(weighted_sum)
        return self.layer_output



       

#applies leaky relu to each element in a layer and returns the whole layer
def leaky_relu(weighted_sum):
    layer = [0] * len(weighted_sum)
    for i in range(len(weighted_sum)):
        if weighted_sum[i] > 0:
            layer[i] = weighted_sum[i]
        else:
            layer[i] = weighted_sum[i]*0.01

    return layer


def softmax(layer, layer_length):
    sum = 0
    max_val = np.max(layer)
    for i in range(layer_length):
        sum += math.exp(layer[i] - max_val)
    
    for i in range(layer_length):
        layer[i]= math.exp(layer[i] - max_val)/sum

    return layer


#this function adds layer's details to list so that it can be initialised in the sgd function
L = []
num = 1
def add(activ_func, size):
    L[num] = layer(activ_func, size, L[num-1].length)

    num +=1 #idk somehow count up each instance


def sgd(epochs, batch_size, learn_rate):

    L[0] = 


    for h in range(epochs):

        for i in range(batch_size):


            for m in range(1, num_layers):
                L[m].weighted = layer.weights_and_biases(L[m-1].outputs) 

            #zero hot array
            actual_probabality = np.zeros((labels_length,))
            actual_probabality[int(y_train[image_no])] = 1

            #output is the first layer done, it has different calcs
            L[num_layers].delta = np.zeros(labels_length)
            L[num_layers].delta = L[num_layers].output - actual_probabality

            L[m].biases_grad = L[num_layers].delta
            L[m].weights_grad = np.outer(L[num_layers-1].output, L[num_layers].delta)

            for m in range(num_layers - 1 - m, 1): #idk about syntax make sure does not drop below 1 which would lead to input layer which has no weights and biases

                L[m].delta = deriv(L[m].weighted_sum) * np.dot(L[m+1].delta, L[m+1].weights.T)

                L[m].biases_grad = L[m].delta 
                L[m].weights_grad = np.outer(L[m-1].output, L[m].delta)



