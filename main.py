import ML_lib
import numpy as np

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy') 

# Print the shape of the data
print("X_train shape:", x_train.shape)
print("y_train shape:", y_train.shape) 

def train():
    ML_lib.input(x_train, y_train)
    ML_lib.test_input(x_test, y_test)

    #ML_lib.load_model(4)
    ML_lib.add(ML_lib.layer.leaky_relu, 256)
    ML_lib.add(ML_lib.layer.leaky_relu, 128)
    ML_lib.add(ML_lib.layer.softmax, 10)  

    #ML_lib.early_stop(1)
    #ML_lib.learn_rate_anneal()
    #ML_lib.check_annealing(1)

    ML_lib.sgd(10, 2000, 0.005)

    #ML_lib.test()
    #ML_lib.save_model()

#quick exit code
import signal
def exit_gracefully(signal, frame):
    ML_lib.save_model()
    exit(0)

# Register the exit function to be called on SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, exit_gracefully) 

train()