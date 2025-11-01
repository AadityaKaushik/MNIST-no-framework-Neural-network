import numpy as np

def initialize_parameters(input_size, hidden_size, output_size):
    limit1 = np.sqrt(6 / (input_size + hidden_size))
    limit2 = np.sqrt(6 / (hidden_size + output_size))
    parameters = {
        "w1": np.random.uniform(-limit1, limit1, size=(hidden_size, input_size)),
        "b1": np.zeros((hidden_size, 1)),
        "w2": np.random.uniform(-limit2, limit2, size=(output_size, hidden_size)),
        "b2": np.zeros((output_size, 1))
    }
    return parameters

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def compute_loss(A2, Y):
    m = Y.shape[1]
    return -np.sum(Y * np.log(A2 + 1e-8)) / m

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forwardprop(X_train, parameters):
    w1, w2, b1, b2 = parameters['w1'], parameters['w2'], parameters['b1'], parameters['b2']
    z1 = np.dot(w1, X_train)+b1
    a1 = relu(z1)
    z2 = np.dot(w2, a1)+b2
    a2 = softmax(z2)
    cache = {
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "a2": a2
    }
    return cache

def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    one_hot = np.zeros((num_classes, m))
    one_hot[y, np.arange(m)] = 1
    return one_hot

def backprop(X_train, Y_train, parameters, cache):
    w1, w2 = parameters["w1"], parameters["w2"]
    z1, a1, a2 = cache["z1"], cache["a1"], cache["a2"]
    m = X_train.shape[1]

    dz2 = a2 - Y_train
    dw2 = (1/m) * np.dot(dz2, a1.T)
    db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

    da1 = np.dot(w2.T, dz2)
    dz1 = da1 * relu_derivative(z1)
    dw1 = (1/m) * np.dot(dz1, X_train.T)
    db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

    grads = {
        "dw1": dw1,
        "dw2": dw2,
        "dz1": dz1,
        "dz2": dz2,
        "db1": db1,
        "db2": db2
    }
    return grads

