


import numpy as np
import matplotlib.pyplot as plt

from Optimiser import Optimiser

def evaluate(x):
    x1 = x.flatten()
    m = 0.5*np.ones(len(x1))
    return sum((x1-m)**2)/len(x1)


def test():

    shape = (1,10)

    min_bound = np.zeros(shape)
    max_bound = np.ones(shape)

    bounds = (min_bound,max_bound)

    op = Optimiser(shape,1,bounds)

    costs = []

    while not op.isNNTrained():

        trial = op.getTrial()
        cost = evaluate(trial)
        
        print('Trial: ',trial)

        print('Cost: ',cost)

        op.setFitness(cost)

        costs.append(cost)

    print('NN Trained')

    for ii in range(100):

        print('Trial: ',ii)
        
        trial = op.getTrial()
        cost = evaluate(trial)
        
        print('Trial: ',trial)

        print('Cost: ',cost)

        op.setFitness(cost)

        costs.append(cost)




    y = np.array(costs)

    plt.figure()
    plt.plot(y,'.')
    plt.xlabel('Trial')
    plt.ylabel('Cost')
    plt.title('Cost over Time of Optimiser')
    plt.show()

    plt.savefig('optimier_test.png')


if __name__ == "__main__":
    test()
