'''
Author: Selva Subramani Damodaran
Assignment 9 | CSC 425/525 Artificial Intelligence

This program shows the implementation of neural network using Python
The NN used contains 3-dimensional input
Goal is to predict a test sample

'''
import numpy as np
import matplotlib.pyplot as plt
#import math as Math

class NeuralNetwork():
    def __init__(self):
        self.weights = np.zeros(range(3))

    def sign(self, z):
        return np.sign(z)

    def forward_process(self, X):
        return self.sign(np.dot(X, self.weights))

    def backpropogation_process(self, X, y):
        for i, x in enumerate(X):
            if(np.dot(X[i], self.weights)*y[i]) <= 0:
                self.weights += X[i]*y[i]
        return self.weights
    
    def calculate_loss(self, X, y, epochs):
        errors = []
        for epoch in range(epochs):
            total_error = 0

            for i, x in enumerate(X):
                if(np.dot(X[i], self.weights)*y[i]) <= 0:
                    total_error += np.dot(X[i], self.weights)*y[i]
            errors.append(total_error*-1)
        
        plt.plot(errors)
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.show()

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            predict = self.forward_process(X)
            self.weights = self.backpropogation_process(X, y)
        
        self.calculate_loss(X, y, epochs)
        # display predicted results
        print("Predicted results: ", self.forward_process(X))

    def predict(self, test):
        predict = self.forward_process(test)
        return np.where(predict <= 0, -1, 1)

NN = NeuralNetwork()
print("Initial random weights: ", NN.weights)

# training dataset -- input train and label pairs
training_X = np.array([[1, 1, 1], [1, 9.4, 6.4], [1, 2.5, 2.1], [1, 8.0, 7.7], [1, 0.5, 2.2], [1, 7.9, 8.4], [1, 7.0, 7.0], [1, 2.8, 0.8], [1, 1.2, 3.0], [1, 7.8, 6.1]])
training_y = np.array([[1, -1, 1, -1, 1, -1, -1, 1, 1, -1]]).T

# start to training the neural network
NN.train(training_X, training_y, 20)

print("Weights after training: ", NN.weights)

# training accuracy performed on the training set
print("The predicted result for training samples are ", NN.predict(training_X))

# prediction a test sample [-2,4,-1],[4,1,1]
test_X = np.array([[-2,4,-1],[4,1,1]])
print("The predicted result for test sample is ", NN.predict(test_X))