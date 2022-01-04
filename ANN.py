import math

import numpy as np
import numpy.matlib as npm
import matplotlib.pyplot as plt
from Activation_Functions import *


class ANN:
    def __init__(self, learning_rate, number_of_epochs, batch_size,
                 train_set, test_set, number_of_samples):
        self.learning_rate = learning_rate
        self.number_of_epochs = number_of_epochs
        self.batch_size = batch_size
        self.set = train_set
        self.test = test_set

        # initializing wights
        w1 = np.random.normal(npm.zeros((150, 102)), npm.ones((150, 102)))
        w2 = np.random.normal(npm.zeros((60, 150)), npm.ones((60, 150)))
        w3 = np.random.normal(npm.zeros((4, 60)), npm.ones((4, 60)))
        w = [w1, w2, w3]

        # initializing biases
        b1 = np.zeros((150, 1))
        b2 = np.zeros((60, 1))
        b3 = np.zeros((4, 1))
        b = [b1, b2, b3]

        self.biases = b
        self.weights = w

        self.number_of_samples = number_of_samples

    def feedforward(self, img):
        z1 = np.add((self.weights[0] @ img), self.biases[0])
        a1 = np.asarray([sigmoid(z) for z in z1]).reshape((150, 1))
        z2 = np.add((self.weights[1] @ a1), self.biases[1])
        a2 = np.asarray([sigmoid(z) for z in z2]).reshape((60, 1))
        z3 = np.add((self.weights[2] @ a2), self.biases[2])
        a3 = np.asarray([sigmoid(z) for z in z3]).reshape((4, 1))
        #a3 = self.softmax(a3)
        return [a1, a2, a3], [z1, z2, z3]

    def back_propagation(self, grad_w, grad_b, grad_a, a, z, img):
        for x in range(4):
            for y in range(60):
                grad_w[2][x, y] += a[1][y, 0] * sigmoid_derivative(z[2][x, 0]) * (
                        2 * a[2][x, 0] - 2 * img[1][x, 0])

        for x in range(4):
            grad_b[2][x, 0] += sigmoid_derivative(z[2][x, 0]) * (2 * a[2][x, 0] - 2 * img[1][x, 0])

        for x in range(60):
            for y in range(4):
                grad_a[1][x, 0] += self.weights[2][y, x] * sigmoid_derivative(z[2][y, 0]) * (
                        2 * a[2][y, 0] - 2 * img[1][y])

        for x in range(60):
            for y in range(150):
                grad_w[1][x, y] += grad_a[1][x, 0] * sigmoid_derivative(z[1][x, 0]) * a[0][y, 0]

        for x in range(60):
            grad_b[1][x, 0] += sigmoid_derivative(z[1][x, 0]) * grad_a[1][x, 0]

        for x in range(150):
            for y in range(60):
                grad_a[0][x, 0] += self.weights[1][y, x] * sigmoid_derivative(z[1][y, 0]) * \
                                   grad_a[1][y, 0]

        for x in range(150):
            for y in range(102):
                grad_w[0][x, y] += grad_a[0][x, 0] * sigmoid_derivative(z[0][x, 0]) * img[0][y]

        for x in range(150):
            grad_b[0][x, 0] += sigmoid_derivative(z[0][x, 0]) * grad_a[0][x, 0]

        return grad_w, grad_b, grad_a

    def back_propagation_vectorized(self, grad_w, grad_b, grad_a, a, z, y):
        temp = (np.asarray([sigmoid_derivative(z) for z in z[2]]).reshape((4, 1)) * (2 * a[2] - 2 * y[1]))
        grad_w[2] += temp @ (np.transpose(a[1]))
        grad_b[2] += temp

        grad_a[1] += np.transpose(self.weights[2]) @ temp
        grad_w[1] += (np.asarray([sigmoid_derivative(z) for z in z[1]]).reshape(60, 1) * a[1]) @ (np.transpose(a[0]))
        temp = (np.asarray([sigmoid_derivative(z) for z in z[1]]).reshape((60, 1)) * grad_a[1])
        grad_b[1] += temp

        grad_a[0] += np.transpose(self.weights[1]) @ temp
        temp = (np.asarray([sigmoid_derivative(z) for z in z[0]]).reshape(150, 1) * grad_a[0])
        grad_w[0] += temp @ np.transpose(y[0])
        grad_b[0] += temp

        return grad_w, grad_b, grad_a

    def calculate_accuracy(self):
        number_of_correct_guesses = 0
        np.random.shuffle(self.set)
        for image in range(self.number_of_samples):
            guess = np.argmax(self.feedforward(self.set[image][0])[0][-1])
            label = np.argmax(self.set[image][1])
            number_of_correct_guesses = number_of_correct_guesses + 1 if guess == label else number_of_correct_guesses
        return number_of_correct_guesses / self.number_of_samples

    def softmax(self, a):
        new_a = [math.exp(a[i]) for i in range(len(a))]
        for i in range(len(a)):
            new_a[i] = new_a[i]/sum(new_a)
        return new_a

    def train_network(self, flag):
        print("\nSecond Step: ")
        print("The accuracy is %d percent." % (self.calculate_accuracy() * 100))

        print("\nThird Step: ")
        print('Training the network: \n')
        cost = []
        accuracy = []
        for i in range(self.number_of_epochs):
            epoch_cost = 0
            np.random.shuffle(self.set)
            batch_count = 0
            for j in range(0, self.number_of_samples, self.batch_size):  # range(start, stop, step)
                batch_count += 1
                if batch_count > self.number_of_samples // self.batch_size:
                    break
                batch = self.set[j:j + self.batch_size]

                for img in range(self.batch_size):
                    grad_w1 = np.zeros((150, 102))
                    grad_w2 = np.zeros((60, 150))
                    grad_w3 = np.zeros((4, 60))
                    grad_w = [grad_w1, grad_w2, grad_w3]

                    grad_b1 = np.zeros((150, 1))
                    grad_b2 = np.zeros((60, 1))
                    grad_b3 = np.zeros((4, 1))
                    grad_b = [grad_b1, grad_b2, grad_b3]

                    grad_a1 = np.zeros((150, 1))
                    grad_a2 = np.zeros((60, 1))
                    grad_a3 = np.zeros((4, 1))
                    grad_a = [grad_a1, grad_a2, grad_a3]
                    # for img in batch:
                    print('  EPOCH: %02d/%02d\t\tBATCH: %04d/%04d\t\tIMAGE: %04d/%04d\t\t\n' % (
                        i + 1, self.number_of_epochs, batch_count, self.number_of_samples // self.batch_size, img + 1,
                        self.batch_size), end='')
                    a, z = self.feedforward(batch[img][0])
                    if flag:
                        grad_w, grad_b, grad_a = self.back_propagation(grad_w, grad_b, grad_a, a, z, batch[img])
                    else:
                        grad_w, grad_b, grad_a = self.back_propagation_vectorized(grad_w, grad_b, grad_a, a, z,
                                                                                  batch[img])
                    c = 0
                    for x in range(4):
                        c += math.pow((batch[img][1][x, 0] - a[2][x, 0]), 2)
                    epoch_cost += c

                for x in range(3):
                    self.weights[x] -= (grad_w[x] / self.batch_size) * self.learning_rate
                    self.biases[x] -= (grad_b[x] / self.batch_size) * self.learning_rate

                print("-----------------------------------------------------------------")
            cost.append(epoch_cost / self.number_of_samples)
            accuracy.append(self.calculate_accuracy())
            print('EPOCH COMPLETED!\n')

        # plt.plot(cost, 'g')
        # plt.xlabel("Epoch", color='black')
        # plt.ylabel("Cost", color='black')
        # plt.grid(color='gray', linestyle='--', linewidth=0.5)
        # plt.show()
        return accuracy, cost
