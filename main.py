import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=100, centers=2, random_state=0)

# plt.scatter(X[:, 0], X[:, 1], c=y, cmap='summer')
# plt.show()

class NeuronComputation:

    def __init__(self, X, y, n_iter = 100, alpha = 0.01):

        self.W = []
        self.b = []
        self.Z = []
        self.A = []
        self.loss = []

        for i in range(n_iter):

            if i == 0:

                self.W.append(self.initialisation(X)[0])
                self.b.append(self.initialisation(X)[1])

            self.Z.append(self.get_output(X, self.W[-1], self.b[-1]))
            self.A.append(self.activation(self.Z[-1]))

            self.loss.append(self.compute_loss(y, self.A[-1]))

            self.W.append(self.update_params(self.W[-1], self.b[-1], X, y, self.A[-1], alpha)[0])
            self.b.append(self.update_params(self.W[-1], self.b[-1], X, y, self.A[-1], alpha)[1])

    def initialisation(self, X):

        W = np.random.rand(X.shape[1])
        b = np.random.rand(1)
        return W, b

    def get_output(self, X, W, b):

        return np.dot(X, W) + b

    def activation(self, Z):

        return 1 / (1 + np.exp(-Z))

    def compute_loss(self, y, A):

        m = y.shape[0]
        return -np.sum(y * np.log(A) + (1 - y) * np.log(1 - A)) / m
    
    def update_params(self, W, b, X, y, A, alpha):

        m = y.shape[0]
        
        W -= alpha / m * np.dot(X.T, A - y)
        b -= alpha / m * np.sum(A - y)
        return W, b

neuron = NeuronComputation(X, y, n_iter=1000)
plt.plot(neuron.loss)
for i in range(X.shape[0]):
    print("y :", y[i], " A : ", neuron.A[-1][i])
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss over iterations')
plt.show()