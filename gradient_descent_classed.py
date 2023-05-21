import numpy as np
import math


#this class is used with input() function, which creates L[0], a list of all the input layers. L[0].output is called to get the i'th input layer in the list
#should also include label data in this
class input_layers:
    def __init__(self, image_array_list, labels_list):
        self.image_array_list = image_array_list
        #initialise length as length of first image, assumes images are normalised
        self.length = len(image_array_list[0,:])
        #initialise output as zero array so instance of input_layers can be used as instance of layer class
        self.output = np.zeros(self.length)

        self.labels_length = int(max(labels_list) + 1)
        self.label = labels_list

class layer:
    #i will need to somehow fill in the last layer's length into this
    def __init__(self, activ_func, length, last_layer_length):
        self.activ_func = activ_func
        self.activ_func_deriv = self.get_deriv(activ_func)

        self.length = length

        #Glorot uniform initialization = sqrt(6/(nodes in and nodes out)), input_size is last_layer.length
        self.initial_range = math.sqrt(6/(self.length + last_layer_length))
        #self.initial_range = 0.2 #for some reason this breaks the algorithm lol, look into this

        self.weights = np.random.uniform(low=-self.initial_range, high=self.initial_range, size=(last_layer_length, self.length))
        self.biases = np.random.uniform(low=-self.initial_range, high=self.initial_range, size=(length)) 

        self.weights_grad = np.zeros((last_layer_length, self.length))
        self.bias_grad = np.zeros((self.length))

        self.delta = np.zeros(self.length)


    #finds weighted sum of node inputs and the weights and biases
    def weights_and_biases(self, last_layer):
        self.weighted_sum = np.dot(last_layer, self.weights) + self.biases
        return self.weighted_sum


#these activ funcs are applied to whole layer to utilise numpy
    def leaky_relu(self, weighted_sum):
        self.output = np.where(weighted_sum > 0, weighted_sum, weighted_sum * 0.1)
        return self.output
    
    def leaky_relu_deriv(self, weighted_sum):
        self.output = np.where(weighted_sum < 0, weighted_sum, 1)
        self.output = np.where(weighted_sum > 0, weighted_sum, 0.1)
        return self.output

    def softmax(self, weighted_sum):
        max_val = np.max(weighted_sum)
        exp_vals = np.exp(weighted_sum - max_val)
        sum = np.sum(exp_vals)
        
        self.output = exp_vals/sum
        return self.output

    #takes weighted sum and performs activation function to get layer output
    def activate(self, weighted_sum):
        self.output = self.activ_func(self, weighted_sum) 
        return self.output
    
    def deriv(self, weighted_sum):
        self.output = self.activ_func_deriv(weighted_sum)
        return self.output
    
    def get_deriv(self, activ_func):
        if activ_func == layer.leaky_relu:
            return self.leaky_relu_deriv


L = []
def input(image_array_list, labels_list):
    #L[0] should only have its output object called 
    L.append(input_layers(image_array_list, labels_list))



def add(activ_func, size):
    L.append(layer(activ_func, size, L[len(L) - 1].length))



def sgd(epochs, batch_size, learn_rate):
    num_layers = len(L)

    for h in range(epochs):
        expected_prob_total = 0
        cost_total = 0 

        rand_list = np.random.choice(range(0, 60000), size=batch_size, replace=False)
        for i in range(batch_size):
            image_no = int(rand_list[i])

            L[0].output = L[0].image_array_list[image_no,:]

        #forward prop
            for m in range(1, num_layers):
                L[m].weighted = L[m].weights_and_biases(L[m-1].output) 
                L[m].output = L[m].activate(L[m].weighted)

            #zero hot array
            actual_probabality = np.zeros((L[0].labels_length))
            actual_probabality[int(L[0].label[image_no])] = 1

            expected_prob = L[num_layers-1].output[int(L[0].label[image_no])]
            cost_total += -math.log(1e-9 + expected_prob)
            expected_prob_total += expected_prob

        #backprop
            #output L[num_layers-1] is the first layer done, it has different calcs
            L[num_layers-1].delta = np.zeros(L[0].labels_length)
            L[num_layers-1].delta = L[num_layers-1].output - actual_probabality

            L[m].biases_grad = L[num_layers-1].delta
            L[m].weights_grad = np.outer(L[num_layers-2].output, L[num_layers-1].delta)

            m = num_layers - 2
            while m > 0 : #idk about syntax make sure does not drop below 1 which would lead to input layer which has no weights and biases

                L[m].delta = L[m].deriv(L[m].weighted) * np.dot(L[m+1].delta, L[m+1].weights.T)

                L[m].biases_grad = L[m].delta 
                L[m].weights_grad = np.outer(L[m-1].output, L[m].delta)

                m -= 1

            for m in range(1, num_layers):
                L[m].weights -= learn_rate * L[m].weights_grad
                L[m].biases -= learn_rate * L[m].bias_grad
            
        learn_rate = learn_rate/1.2

        print("Epoch:", h + 1, "Accuracy:", expected_prob_total/batch_size, "Cost:", cost_total/batch_size)
            



