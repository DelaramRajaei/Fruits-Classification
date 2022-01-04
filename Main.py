# Principles of Computational Intelligence
# Project 1
# Delaram Rajaei 9731084

import matplotlib.pyplot as plt
from Loading_Datasets import *
from ANN import *
import time

if __name__ == '__main__':
    start_time = time.time()
    learning_rate = 0.5
    number_of_epochs = 5
    number_of_run = 5
    batch_size = 10
    number_of_samples = 1962
    neural_network = ANN(learning_rate=learning_rate,
                         number_of_epochs=number_of_epochs,
                         batch_size=batch_size,
                         train_set=train_set,
                         test_set=test_set,
                         number_of_samples=number_of_samples)
    accuracy = []
    costs = []
    for i in range(number_of_run):
        a, c = neural_network.train_network(False)  # True: back_propagation   ,False: back_propagation_vectorized
        accuracy.append(sum(a) / len(a))
        costs.append(sum(c) / len(c))

    stop_time = time.time()
    print('\n\tTraining process completed in {}s'.format(round(stop_time - start_time)).upper())

    plt.plot(costs, 'g')
    plt.xlabel("Number of runs", color='black')
    plt.ylabel("Cost", color='black')
    plt.title("Average cost", color='black')
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.show()

    print(accuracy)
    accuracy_avg = sum(accuracy) / len(accuracy)
    print('\tThe accuracy of the network for train set:\t{}%'.format(accuracy_avg * 100).upper())
