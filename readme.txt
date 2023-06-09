ML_lib instructions

to load data:
ideally save as numpy array eg. x_train = np.load('x_train.npy')

to train:

define the input data
ML_lib.input(x_train, y_train)
ML_lib.test_input(x_test, y_test)

add your layer with activation function, and node count
ML_lib.add(ML_lib.layer.leaky_relu, 256)
ML_lib.add(ML_lib.layer.leaky_relu, 128)
ML_lib.add(ML_lib.layer.softmax, 10)

add early stop, regularisation, learn rate decay funcs, at any point before training algortithm, eg.
ML_lib.early_stop(3)

finally, declare training algorithm with params. for sgd the order is epochs, batch size, and learn rate
ML_lib.sgd(10, 2000, 0.005)


note, the above code creates a new model. to save and load a model, the following can be done
to save:
add ML_lib.save_model() at any point after declaring your training algorithm to save the model

to load:
use ML_lib.load_model(number_layers). note, you will still need to define the xtrain and ytrain inputs, and any addlayer functions should be removed. also, the number of layers including the input layers needs to be read into the function
eg.
ML_lib.input(x_train, y_train)
ML_lib.test_input(x_test, y_test)
ML_lib.load_model(4)
ML_lib.test()


-list of functions
input(x_train, y_train)
test_input(x_test, y_test)
load_model(num_layers)
save_model()
test()
add(ML_lib.layer.activ_func, node_count)

-activation funcs
relu
leaky relu
sigmoid
softmax

-algorithms
sgd

-misc
early_stop(max_subsequent_decreases)
learn_rate_anneal()    - halves learn rate when test accuracy decreases
check_annealing(max_subsequent_decreases)   - experimental feature to load and modify learn rate as accuracy decreases
