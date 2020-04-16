


import numpy as np
import matplotlib.pyplot as plt

from Optimiser import Optimiser

def evaluate(x):
    x1 = x.flatten()
    return sum(x1**2)/len(x)


def test():

    shape = (3,20)

    min_bound = np.zeros(shape)
    max_bound = np.ones(shape)

    bounds = (min_bound,max_bound)

    op = Optimiser(shape,1,bounds)

    costs = []

    while not op.isNNTrained():

        trial = op.getTrial()
        cost = evaluate(trial)
        
        print(cost)

        op.setFitness(cost)

        costs.append(cost)

    print('NN Trained')

    for ii in range(1000):

        print('Trial: ',ii)
        
        trial = op.getTrial()
        cost = evaluate(trial)
        
        print(cost)

        op.setFitness(cost)

        costs.append(cost)




    y = np.array(costs)

    plt.figure()
    plt.plot(y,'.')
    plt.xlabel('Trial')
    plt.ylabel('Cost')
    plt.title('Cost over Time of Optimiser')
    plt.show()


if __name__ == "__main__":
    test()
