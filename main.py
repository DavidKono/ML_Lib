
import numpy as np
import math
import os
import scipy.io
import gradient_descent


def load_mnist_fashion_data(filename):
    # Load the CSV file
    data = np.genfromtxt(filename, delimiter=',', skip_header=1)

    # Extract features and labels
    x = data[:, 1:]/255  # Features (pixel values)
    y = data[:, 0]   # Labels (target values)

    return x, y

""" x_train, y_train = load_mnist_fashion_data('C:/Users/Daithi/Documents/Coding/fashion/fashion-mnist_train.csv')
np.save('x_train.npy', x_train)
np.save('y_train.npy', y_train)

x_test, y_test = load_mnist_fashion_data('C:/Users/Daithi/Documents/Coding/fashion/fashion-mnist_test.csv')
np.save('x_test.npy', x_test)
np.save('y_test.npy', y_test)  """

#quick exit code
import signal
def exit_gracefully(signal, frame):
        print("Exiting gracefully...")
        exit(0)

# Register the exit function to be called on SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, exit_gracefully) 

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

# Print the shape of the data
print("X_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

input_size = 28 * 28


#preprocessing
#preprocessing.preprocess_files(folder_path, input_size, new_size)


#quick exit code
import signal
def exit_gracefully(signal, frame):
        print("Exiting gracefully...")
        exit(0)

# Register the exit function to be called on SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, exit_gracefully) 

hidden_1_size = 256
hidden_2_size = 128
labels_length = 10

def train():
#this is all create functions
#randomise weights and biases once for hidden 1 and 2
    
    hidden_1_weights = np.random.uniform(low=-0.3, high=0.3, size=(input_size, hidden_1_size))
    hidden_1_biases = np.random.uniform(low=-0.3, high=0.3, size=(hidden_1_size))
    
    hidden_2_weights = np.random.uniform(low=-0.3, high=0.3, size=(hidden_1_size, hidden_2_size))
    hidden_2_biases = np.random.uniform(low=-0.3, high=0.3, size=(hidden_2_size))

    output_weights = np.random.uniform(low=-0.3, high=0.3, size=(hidden_2_size, labels_length))
    output_biases = np.random.uniform(low=-0.3, high=0.3, size=(labels_length))

    #last_cost = 0 #for performance based learn rate

    #this is all train function
    epochs = 10 #after 20 epochs there is a 90% chance for each image that it has been included in a training batch, if the batch size was 50. min_trials = log(1-wanted_prob) / log(1-prob_per_trial)
    learn_rate = 0.005
    for h in range(epochs):
        
        cost_total = 0

        output_grad_total = np.zeros((hidden_2_size,labels_length))
        output_bias_grad_total = np.zeros((labels_length,))
        hidden_2_grad_total = np.zeros((hidden_1_size,hidden_2_size))
        hidden_2_bias_grad_total = np.zeros((hidden_2_size,))
        hidden_1_grad_total = np.zeros((input_size,hidden_1_size))
        hidden_1_bias_grad_total = np.zeros((hidden_1_size,))

        #random batch
        batch_size = 128
        rand_list = np.random.choice(range(0, 60000), size=batch_size, replace=False)

        prob_total = 0

        #make i range thru random vals
        for i in range(batch_size):
            image_no = int(rand_list[i])
            #image_no = i
            hidden_layer_1_weighted = gradient_descent.weighted_sum(x_train[image_no,:], hidden_1_weights, hidden_1_biases, hidden_1_size)
            hidden_layer_1_output = gradient_descent.hidden_layer(hidden_layer_1_weighted, hidden_1_size)

            hidden_layer_2_weighted = gradient_descent.weighted_sum(hidden_layer_1_output, hidden_2_weights, hidden_2_biases, hidden_2_size)
            hidden_layer_2_output = gradient_descent.hidden_layer(hidden_layer_2_weighted, hidden_2_size)

            output_layer_weighted = gradient_descent.weighted_sum(hidden_layer_2_output, output_weights, output_biases, labels_length) 
            output_layer_output = gradient_descent.output_layer(output_layer_weighted, labels_length)
            #expected_prob = gradient_descent.softmax(output_layer_output, labels_length, int(y_train[image_no] - 1))
            expected_prob = output_layer_output[int(y_train[image_no])]

            prob_total += expected_prob

            cost_total += -math.log(1e-9 + expected_prob)
            

            actual_probabality = np.zeros((labels_length,))
            actual_probabality[int(y_train[image_no])] = 1


            """ output_delta = np.zeros(labels_length)
            #output weights and biases grad
            output_delta = output_layer_output - actual_probabality
            output_bias_grad_total += output_delta
                                #outer for h[x] * m[y]
            output_grad_total = np.outer(hidden_layer_2_output,output_delta)

            hidden_2_delta = np.zeros(hidden_2_size)
            
            leaky_relu_deriv = [0] * hidden_2_size
            for y in range(hidden_2_size):
                leaky_relu_deriv[y] = gradient_descent.leaky_relu_derivative(hidden_layer_2_weighted[y])
            dot = np.dot(output_delta, output_weights.T)

            hidden_2_delta = leaky_relu_deriv * dot
            hidden_2_bias_grad_total += hidden_2_delta
            hidden_2_grad_total += np.outer(hidden_layer_1_output, hidden_2_delta)

            hidden_1_delta = np.zeros(hidden_1_size)

            leaky_relu_deriv = [0] * hidden_1_size
            for y in range(hidden_1_size):
                leaky_relu_deriv[y] = gradient_descent.leaky_relu_derivative(hidden_layer_1_weighted[y])
            dot = np.dot(hidden_2_delta, hidden_2_weights.T)

            hidden_1_delta = leaky_relu_deriv * dot
            hidden_1_bias_grad_total += hidden_1_delta

            hidden_1_grad_total += np.outer(x_train[image_no], hidden_1_delta) """

            output_delta = np.zeros(labels_length)
            #output weights and biases grad
            output_delta = output_layer_output - actual_probabality
            output_bias_grad_total = output_delta
                                #outer for h[x] * m[y]
            output_grad_total = np.outer(hidden_layer_2_output,output_delta)

            hidden_2_delta = np.zeros(hidden_2_size)
            
            leaky_relu_deriv = [0] * hidden_2_size
            for y in range(hidden_2_size):
                leaky_relu_deriv[y] = gradient_descent.leaky_relu_derivative(hidden_layer_2_weighted[y])
            dot = np.dot(output_delta, output_weights.T)

            hidden_2_delta = leaky_relu_deriv * dot
            hidden_2_bias_grad_total = hidden_2_delta
            hidden_2_grad_total = np.outer(hidden_layer_1_output, hidden_2_delta)

            hidden_1_delta = np.zeros(hidden_1_size)

            leaky_relu_deriv = [0] * hidden_1_size
            for y in range(hidden_1_size):
                leaky_relu_deriv[y] = gradient_descent.leaky_relu_derivative(hidden_layer_1_weighted[y])
            dot = np.dot(hidden_2_delta, hidden_2_weights.T)

            hidden_1_delta = leaky_relu_deriv * dot
            hidden_1_bias_grad_total = hidden_1_delta

            hidden_1_grad_total = np.outer(x_train[image_no], hidden_1_delta)



            output_weights -= learn_rate * (output_grad_total)
            output_biases -= learn_rate * (output_bias_grad_total)

            hidden_2_weights -= learn_rate * (hidden_2_grad_total)
            hidden_2_biases -= learn_rate * (hidden_2_bias_grad_total)

            hidden_1_weights -= learn_rate * (hidden_1_grad_total)
            hidden_1_biases -= learn_rate * (hidden_1_bias_grad_total) 



        """ output_weights -= learn_rate * (output_grad_total/batch_size)
        output_biases -= learn_rate * (output_bias_grad_total/batch_size)

        hidden_2_weights -= learn_rate * (hidden_2_grad_total/batch_size)
        hidden_2_biases -= learn_rate * (hidden_2_bias_grad_total/batch_size)

        hidden_1_weights -= learn_rate * (hidden_1_grad_total/batch_size)
        hidden_1_biases -= learn_rate * (hidden_1_bias_grad_total/batch_size)  """

        avg_cost = cost_total/batch_size

        avg_prob = prob_total/batch_size
        print(h, ": ", avg_prob, ": ", avg_cost)

        """ if (abs(avg_cost - last_cost)/avg_cost < 0.1):
            learn_rate = learn_rate/1.01

        last_cost = avg_cost """

        learn_rate = learn_rate/1.2 #note this is based on the avg cost of the data being used in the training data, therefore this data is often new but is then incorporated into the model, so the cost of the data used in the model itself is not very useful

        np.savez('my_arrays.npz', 
            hidden_1_weights=hidden_1_weights,
            hidden_1_biases=hidden_1_biases,
            hidden_2_weights=hidden_2_weights,
            hidden_2_biases=hidden_2_biases,
            output_weights=output_weights,
            output_biases=output_biases)

def test(number_of_tests): 
    rand_list = np.random.choice(range(0, 10000), size=number_of_tests, replace=False)

    prob_total = 0

    loaded_arrays = np.load('my_arrays.npz')
    for i in range(rand_list.size):

        # Assign each loaded array to its respective variable
        hidden_1_weights = loaded_arrays['hidden_1_weights']
        hidden_1_biases = loaded_arrays['hidden_1_biases']
        hidden_2_weights = loaded_arrays['hidden_2_weights']
        hidden_2_biases = loaded_arrays['hidden_2_biases']
        output_weights = loaded_arrays['output_weights']
        output_biases = loaded_arrays['output_biases'] 

        hidden_layer_1_weighted = gradient_descent.weighted_sum(x_test[rand_list[i],:], hidden_1_weights, hidden_1_biases, hidden_1_size)
        hidden_layer_1_output = gradient_descent.hidden_layer(hidden_layer_1_weighted, hidden_1_size)

        hidden_layer_2_weighted = gradient_descent.weighted_sum(hidden_layer_1_output, hidden_2_weights, hidden_2_biases, hidden_2_size)
        hidden_layer_2_output = gradient_descent.hidden_layer(hidden_layer_2_weighted, hidden_2_size)

        output_layer_weighted = gradient_descent.weighted_sum(hidden_layer_2_output, output_weights, output_biases, labels_length) 
        output_layer_output = gradient_descent.output_layer(output_layer_weighted, labels_length)
        expected_prob = output_layer_output[int(y_test[rand_list[i]])]
        #print(expected_prob)

        prob_total += expected_prob
    
    print("avg accuracy ", prob_total/number_of_tests)
    
train()
test(300)