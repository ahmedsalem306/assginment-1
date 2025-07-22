import numpy as np


np.random.seed(0)

def tanh(x):
    return np.tanh(x)


inputs = np.array([0.6, 0.9])


weights_input_hidden = np.random.uniform(-0.5, 0.5, (2, 2))  
weights_hidden_output = np.random.uniform(-0.5, 0.5, (2, 1))  


bias_hidden = 0.5
bias_output = 0.7


hidden_input = np.dot(inputs, weights_input_hidden) + bias_hidden
hidden_output = tanh(hidden_input)


final_input = np.dot(hidden_output, weights_hidden_output) + bias_output
final_output = tanh(final_input)


print("Weights Input-Hidden:\n", weights_input_hidden)
print("Hidden Layer Output:", hidden_output)
print("Weights Hidden-Output:\n", weights_hidden_output)
print("Final Output:", final_output)
