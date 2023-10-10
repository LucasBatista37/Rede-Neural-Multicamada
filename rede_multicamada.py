import numpy as np

def sigmoid(soma):
    return 1 / (1 + np.exp(-soma))

b = sigmoid(50)
print(b)