import gradient_descent_classed
import numpy as np

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


def train():
    gradient_descent_classed.input(x_train)

    gradient_descent_classed.add(gradient_descent_classed.leaky_relu, 256)
    gradient_descent_classed.add(gradient_descent_classed.leaky_relu, 128)
    gradient_descent_classed.add(gradient_descent_classed.leaky_relu, 128)

    gradient_descent_classed.add(gradient_descent_classed.softmax, 10)

    gradient_descent_classed.sgd(10, 1024, 0.005)

#def test():


train()
#test()
