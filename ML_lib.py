import numpy as np
import pickle
import time

class checks:
    def __init__(self):
        #bool for check
        self.learn_rate_annealing = 0
        self.check_loss_annealing = 0
        self.check_loss = 0

        #
        self.subsequent_decreases = 0
        self.max_decreases = 1

        #
        self.train_accuracy = 0
        self.last_train_accuracy = 0
        self.test_accuracy = 0
        self.last_test_accuracy = 0

global_truths = checks()

#this class is used with input() function, which creates L[0], a list of all the input layers. L[0].output is called to get the i'th input layer in the list
#should also include label data in this
class input_layers:
    def __init__(self, image_array_list, labels_list):
        self.image_array_list = image_array_list
        #initialise length as length of first image, assumes images are normalised. needs to be called "length" to work with other layers' lengths
        self.length = len(image_array_list[0,:])
        #initialise output as zero array so instance of input_layers can be used as instance of layer class
        self.output = np.zeros(self.length)

        #length param used as number of images for xtrain
        self.length_param = len(image_array_list[:,0])

        self.labels_length = int(max(labels_list) + 1)
        self.label = labels_list

class layer:
    #i will need to somehow fill in the last layer's length into this
    def __init__(self, activ_func, length, last_layer_length):
        self.activ_func = activ_func
        self.activ_func_deriv = self.get_deriv(activ_func)

        self.length = length

        #Glorot uniform initialization = sqrt(6/(nodes in and nodes out)), input_size is last_layer.length
        #self.initial_range = np.sqrt(6/(self.length + last_layer_length))

        #he initialisation
        self.weights = np.random.randn(last_layer_length, self.length) * np.sqrt(2 / last_layer_length)
        self.biases = np.zeros((length))

        
        self.weights_grad = np.zeros((last_layer_length, self.length))
        self.bias_grad = np.zeros((self.length))

        self.delta = np.zeros(self.length)

    #finds weighted sum of node inputs and the weights and biases
    def weights_and_biases(self, last_layer):
        self.weighted_sum = np.dot(last_layer, self.weights) + self.biases
        return self.weighted_sum

#these activ funcs are applied to whole layer to utilise numpy
    def leaky_relu(self, weighted_sum):
        self.output = np.where(weighted_sum > 0, weighted_sum, weighted_sum * 0.001)
        return self.output
    
    def leaky_relu_deriv(self, weighted_sum):
        self.output = np.where(weighted_sum > 0, 1, 0.001)
        return self.output
    
    def relu(self, weighted_sum):
        self.output = np.where(weighted_sum > 0, weighted_sum, 0)
        return self.output
    
    def relu_deriv(self, weighted_sum):
        self.output = np.where(weighted_sum > 0, 1, 0)
        return self.output
    
    def sigmoid(self, weighted_sum):
        return 1 / (1 + np.exp(-weighted_sum))
    
    def sigmoid_deriv(self, weighted_sum):
        sigmoid_x = self.sigmoid(weighted_sum)
        return sigmoid_x * (1 - sigmoid_x)

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
        if activ_func == layer.relu:
            return self.relu_deriv
        if activ_func == layer.sigmoid:
            return self.sigmoid_deriv



L = []
#L[0] seperate because it doesnt have weighted sum, etc
def input(image_array_list, labels_list):
    L.append(input_layers(image_array_list, labels_list))

def add(activ_func, size):
    L.append(layer(activ_func, size, L[len(L) - 1].length))

H = []
def test_input(x_test, y_test):
    H.append(input_layers(x_test, y_test))




def forward_prop(num_layers):
    for m in range(1, num_layers):
        L[m].weighted = L[m].weights_and_biases(L[m-1].output) 
        L[m].output = L[m].activate(L[m].weighted)

def back_prop(image_no, learn_rate):
    num_layers = len(L)

    #output L[num_layers-1] is the first layer done, it has different calcs
    #zero hot array for delta
    actual_probabality = np.zeros((L[0].labels_length))
    actual_probabality[int(L[0].label[image_no])] = 1
    L[num_layers-1].delta = L[num_layers-1].output - actual_probabality

    L[num_layers-1].bias_grad = learn_rate * L[num_layers-1].delta
    L[num_layers-1].weights_grad = learn_rate * np.outer(L[num_layers-2].output, L[num_layers-1].delta)

    #then from second last layer
    m = num_layers - 2
    while m > 0 : 
        L[m].delta = L[m].deriv(L[m].weighted) * np.dot(L[m+1].delta, L[m+1].weights.T)
        L[m].bias_grad = learn_rate * L[m].delta 
        L[m].weights_grad = learn_rate * np.outer(L[m-1].output, L[m].delta) #+ (1/85000 * L[m].weights) + (1/60000 * np.square(L[m].weights))
        
        m -= 1

    for m in range(1, num_layers):
        L[m].weights -= L[m].weights_grad
        L[m].biases -= L[m].bias_grad




def sgd(epochs, batch_size, learn_rate):

    num_layers = len(L)

    for h in range(epochs):
        start_time = time.time()
        expected_prob_total = 0
        cost_total = 0 

        rand_list = np.random.choice(range(0, L[0].length_param), size=batch_size, replace=False)
        for i in range(batch_size):

        #initialise image input
            image_no = int(rand_list[i])
            L[0].output = L[0].image_array_list[image_no,:]

        #forward prop
            forward_prop(num_layers)

        #accuracy and cost for display
            expected_prob = L[num_layers-1].output[int(L[0].label[image_no])]
            cost_total += -np.log(1e-9 + expected_prob)
            expected_prob_total += expected_prob

        #backprop
            back_prop(image_no, learn_rate)
            
        #print
        global_truths.train_accuracy = expected_prob_total/batch_size
        print("Epoch:", h + 1, "Accuracy:", global_truths.train_accuracy, "Cost:", cost_total/batch_size)

        perform_early_stop_or_anneal(learn_rate)
            
        end_time = time.time()
        epoch_time = int((end_time - start_time) * 1000)
        print("Time taken this epoch = ", epoch_time, " ms")
        print("")


def perform_early_stop_or_anneal(learn_rate):
    #early stop
        if global_truths.check_loss == 1:
            early_stop_action()  
        
    #learn rate annealing
        if global_truths.learn_rate_annealing == 1:
            learn_rate_anneal_action(learn_rate)
            
    #test of both
        if global_truths.check_loss_annealing == 1:
            check_annealing_action(learn_rate)

def decreases_function(this_accuracy, last_accuracy):
    #if accuracy gone up, save
    if this_accuracy > last_accuracy:
        global_truths.subsequent_decreases = 0
        return 0
    
    #else if gone down, increase subsequent decreases
    else:
        global_truths.subsequent_decreases += 1
        #if has exceeded max decreases, activate
        if global_truths.subsequent_decreases >= global_truths.max_decreases:
            global_truths.subsequent_decreases = 0
            return 1
        
        return 2
    
def check_annealing(max_subsequent_losses):
    global_truths.check_loss_annealing = 1
    global_truths.max_decreases = max_subsequent_losses

def check_annealing_action(learn_rate):   
    action = decreases_function(global_truths.train_accuracy, global_truths.last_train_accuracy)
    if action == 1:
        print("train accuracy has gone down; loading back and reducing learn rate")
        learn_rate = learn_rate/2
        load_back()
    if action == 0:
        global_truths.last_train_accuracy = global_truths.train_accuracy
        save_model()

def learn_rate_anneal():
    global_truths.learn_rate_annealing = 1

def learn_rate_anneal_action(learn_rate):
    global_truths.test_accuracy = test()
    action = decreases_function(global_truths.test_accuracy, global_truths.last_test_accuracy)
    global_truths.last_test_accuracy = global_truths.test_accuracy
    if action == 1:
        print("test accuracy has gone down; reducing learn rate")
        learn_rate = learn_rate/2
            
def early_stop(max_subsequent_losses):
    global_truths.check_loss = 1
    global_truths.max_decreases = max_subsequent_losses

def early_stop_action():
    global_truths.test_accuracy = test()
    action = decreases_function(global_truths.test_accuracy, global_truths.last_test_accuracy) 
    if action == 1:
        print("test accuracy has gone down; early stopping")
        exit(0)
    if action == 0:
        global_truths.last_test_accuracy = global_truths.test_accuracy
        save_model()



def test():
    expected_prob_total = 0
    cost_total = 0 
    
    #change to work on H[0]
    batch_size = H[0].length_param
    
    num_layers = len(L)

    for i in range(batch_size):

        L[0].output = H[0].image_array_list[i,:]

    #forward prop
        forward_prop(num_layers)

        #zero hot array
        actual_probabality = np.zeros((L[0].labels_length))
        actual_probabality[int(H[0].label[i])] = 1

        expected_prob = L[num_layers-1].output[int(H[0].label[i])]
        cost_total += -np.log(1e-9 + expected_prob)
        expected_prob_total += expected_prob

    print("Test Accuracy:", expected_prob_total/batch_size, "Test Cost:", cost_total/batch_size)
    return expected_prob_total/batch_size

def save_model():
    for i in range(1, len(L)):
        with open(f"L{i}biases.npy", "wb") as file:
            pickle.dump(L[i], file)
    print("Model Saved")

def load_back():
    #no append
    for i in range(1, len(L)):
        with open(f"L{i}biases.npy", "rb") as file: 
            L[i] = pickle.load(file)
    print("Model Loaded")

def load_model(length):
    for i in range(1, length):
        with open(f"L{i}biases.npy", "rb") as file: 
            L.append([])
            L[i] = pickle.load(file)
    print("Model Loaded")
