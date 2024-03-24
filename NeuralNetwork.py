import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

training_outputs = np.array([[0, 1, 1, 0]]).T

np.random.seed(1)
weights = 2 * np.random.random((3, 1)) - 1

print("Random starting weights:")
print(weights)

for i in range(10000):  # Increase the number of iterations for better convergence
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, weights))

    errors = training_outputs - outputs
    adjustments = errors * sigmoid_derivative(outputs)
    weights += np.dot(input_layer.T, adjustments)

print("Synaptic weights after training:")
print(weights)
print("Outputs after training:")
print(outputs)
