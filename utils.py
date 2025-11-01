from model import backprop, one_hot_encode, forwardprop, softmax, compute_loss, relu_derivative, relu, initialize_parameters
import numpy as np
import pickle as pickle

def predict(X, parameters):
    cache = forwardprop(X, parameters)
    probs = cache["a2"]
    prediction = np.argmax(probs, axis=0)
    return prediction

def save_parameters(parameters, filename="parameters.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(parameters, f)
    print(f"Parameters saved to {filename}")

def load_parameters(filename="parameters.pkl"):
    with open(filename, "rb") as f:
        parameters = pickle.load(f)
    print(f"Parameters loaded from {filename}")
    return parameters