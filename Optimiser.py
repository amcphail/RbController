#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:00:16 2020

@author: amcp011
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
import keras.backend as K

import numpy as np
import scipy as sp
import scipy.optimize as op
import matplotlib.pyplot as plt

from Either import *

tolerance=0e-6

# build the neural net
#  Coil current 0 -> 60
#  MOT power 0 -> 10000
#  MOT detuning 74000 -> 86000
#  Repump detuning 74000 -> 86000


class SingleMLP(Object):
    
    def __init__(self, input_length,hidden_layers,output_length,weights=None):

        self.input_length=input_length
        self.hidden_layers=hidden_layers
        self.output_length=output_length

        minp = np.zeros((input_length))
        maxp = np.ones((input_length))
        self.bounds = list(zip(minp,maxp))

        self.model = keras.Sequential()
        self.model.add(keras.layers.Flatten(input_shape=(input_length,)))
        for ii in range(1,len(hidden_layers)+1):
            self.model.add(keras.layers.Dense(hidden_layers[ii-1], activation='relu'))
            if weights:
                self.model.get_layer(index=ii).set_weights([weights[ii]])             

        self.model.add(keras.layers.Dense(output_length))
        index = len(hidden_layers)+1
        if weights:
            self.model.get_layer(index=index).set_weights([weights[index]])             
                                                
        self.model.compile(optimizer='adam',
                           metrics=['accuracy'])
        
    def fit(self, x_train, y_train, epochs=3, batch_size=32):
        self.model.fit(x_train, y_train, epochs=3, batch_size=32)

    def cost(self,x):
        return self.model.predict(x)

    def jacobian(self,x):    
        jacobian_matrix = []
        for m in range(self.output_size):
            # We iterate over the M elements of the output vector
            grad_func = tf.gradients(self.model.output[:, m], self.model.input)
        
            with K.get_session() as sess:
                gradients = sess.run(grad_func, feed_dict={self.model.input: x.reshape((1, x.size))})

            jacobian_matrix.append(gradients[0][0,:])
        
        return np.array(jacobian_matrix)

    
    def minimise(self, start_params, tolerance):
        '''
        Runs scipy.optimize.minimize() on the network.
        '''
        res = so.minimize(fun = self.cost,
                x0 = start_params,
                jac = self.jacobian,
                bounds = self.bounds,
                tol = tolerance)

        return res

    def getNextTrial(self,guess,tolerance):
        result = self.minimise(guess,tolerance)
        return result.x

    def getWeights(self):
        return self.model.get_weights
    
class DifferentialEvolver(Object):

    def __init__(self, dimensions, mutation_rate=0.8, crossp=0.7, popsize=15):
        
        self.dimensions=dimensions
        self.mutation_rate=mutation_rate
        self.crossp=crossp
        self.popsize=popsize

        self.population = np.random.rand(popsize,dimensions)
        self.fitness = np.ones((popsize,dimensions))
        self.best_index = -1
        self.best = np.zeros(dimensions)
        
        self.current_trial_index = -1
        self.trial = np.zeros(dimensions)

    def getPopulation(self):
        return self.population

    def setPopulation(self,population):
        self.population = population

    def getFitnesses(self):
        return self.fitness
        
    def setFitnesses(self,fitness):
        self.fitness = fitness

        self.best_index = np.argmin(self.fitness)
        self.best = self.population[self.best_index]

    def getNextTrial(self):
        self.current_trial_index = np.random.randint(0,self.popsize)

        idxs = [idx for idx in range(self.popsize) if idx != self.current_trial_index]
        # rand/2
        a, b, c, d, e = self.population[np.random.choice(idxs,5,replace=False)]
        # for best not rand use best instead of a
        mutant = np.clip(a+self.mutation_rate*((b-c)+(d-e)),0,1)
        cross_points = np.random.rand(dimensions) < self.crossp
        if not np.any(cross_points):
            cross_points[np.random.randint(0,dimensions)] = True
        self.trial = np.where(cross_points, mutant, self.population[self.current_trial_index])
        return self.trial

    def setThisTrial(self,trial):
        self.current_trial_index = np.random.randint(0,self.popsize)

        self.trial = trial

    def setFitness(self,fitness):
        if fitness < self.fitness[self.current_trial_index]:
            self.fitness[self.current_trial_index] = fitness
            self.population[self.current_trial_index] = self.trial
            if fitness < self.fitness[self.best_index]:
                self.best_index = self.current_trial_index
                self.best = self.trial

    def getBest(self):
        return self.best

class Optimiser(Object):
    
    def __init__(self,inputs,outputs,bounds,filename=None):

        self.inputs = inputs
        self.outputs = outputs
        self.bounds = bounds.flatten()

        self.anns = {}
        self.des = {}

        self.de_initialised = False
        self.de_init_trials = 0 # hard coding population size to 15
        self.de_init_pop = Nothing()
        self.de_init_fit = Nothing()
        self.de_training_set = (np.zeros((2*self.inputs,self.inputs)),np.ones((2*self.inputs,1)))
        
        self.de_trials = -1
        self.nn_trained = False

        self.nn_trials = []
        self.nn_fitnesses = []
        
        self.active_trial = -1
        
        if filename:
            self.loadFile(filename)
        else:
            self.input_length = self.inputs[0]*self.inputs[1]
            self.hidden_layers = [64,64,64,64,64]
            self.output_length = 1
            
            self.anns[1] = SingleMLP(self.input_length,self.hidden_layers,self.output_length)
            self.anns[2] = SingleMLP(self.input_length,self.hidden_layers,self.output_length)
            self.anns[3] = SingleMLP(self.input_length,self.hidden_layers,self.output_length)

            self.des[1] = DifferentialEvolver(self.inputs)
    
        self.min_bound, self.max_bound = np.asarray(self.bounds).T
        self.difference=np.fabs(self.min_bound-self.max_bound)
            
    def loadFile(self,filename):
    
        self.optimiser_file = np.load(filename).item()
    
        self.inputs = self.optimiser_file.get('inputs')
        self.outputs = self.optimiser_file.get('outputs')
        self.bounds = self.optimiser_file.get('bounds')
        
        self.input_length = self.optimiser_file.get('input_length')
        self.hidden_layers = self.optimiser_file.get('hidden_layers')
        self.output_length = self.optimiser_file.get('output_length')
        
        ann1_weights = self.optimiser_file.get('ann1_weights')
        ann2_weights = self.optimiser_file.get('ann2_weights')
        ann3_weights = self.optimiser_file.get('ann3_weights')
        
        self.anns[1] = SingleMLP(self.input_length,self.hidden_layers,self.output_length,ann1_weights)
        self.anns[2] = SingleMLP(self.input_length,self.hidden_layers,self.output_length,ann2_weights)
        self.anns[3] = SingleMLP(self.input_length,self.hidden_layers,self.output_length,ann3_weights)

        de1_population = self.optimiser_file.get('de1_population')
        de1_fitness = self.optimiser_file.get('de1_fitness')

        self.des[1] = DifferentialEvolver(self.inputs)
        self.des[1].setPopulation(de1_population)
        self.des[1].setFitness(de1_fitness)

        self.de_initialised = self.optimiser_file.get('de_initialised')
        self.de_init_trials = self.optimiser_file.get('de_init_trials')
        self.de_init_pop = Just(self.optimiser_file.get('de_init_pop'))
        self.de_init_fit = Just(self.optimiser_file.get('de_init_fit'))
        self.de_training_set = self.optimiser_file.get('de_training_set')
    
        self.de_trials = self.optimiser_file.get('de_trials')
        self.nn_trained = self.optimiser_file.get('nn_trained')
    
        self.nn_trials = self.optimiser_file.get('nn_trials')
        self.nn_fitnesses = self.optimiser_file.get('nn_fitnesses')
        
        self.active_trial = self.optimiser_file.get('active_trial')

    def saveFile(self,filename):

        if not self.de_initialised:
            return False
        
        data = {}
        
        data['inputs'] = self.inputs
        data['outputs'] = self.outputs
        data['bounds'] = self.bounds
        
        data['input_length'] = self.input_length
        data['hidden_layers'] = self.hidden_layers
        data['output_length'] = self.output_length
        
        data['ann1_weights'] = self.ann[1].get_weights()
        data['ann2_weights'] = self.ann[2].get_weights()
        data['ann3_weights'] = self.ann[3].get_weights()

        data['de1_population'] = self.des[1].getPopulation()
        data['de1_fitness'] = self.des[1].getFitnesses()

        data['de_initialised'] = self.de_initialised
        data['de_init_trials'] = self.de_init_trials
        data['de_init_pop'] = self.de_init_pop.value()
        data['de_init_fit'] = self.de_init_fit.value()
        data['de_training_set'] = self.de_training_set
    
        data['de_trials'] = self.de_trials
        data['nn_trained'] = self.nn_trained
    
        data['nn_trials'] = self.nn_trials
        data['nn_fitnesses'] = self.nn_fitnesses
        
        data['active_trial'] = self.active_trial

        np.save(filename,data)
    
    def normInputs(self,trial):
        normed =  (trial.flatten() - self.min_bound) / self.difference
        return normed
    
    def denormInputs(self,trial):
        denormed = self.min_bound + self.difference * trial
        denormed.reshape(self.inputs)
        return denormed
    
    def setGuess(self,guess,cost):
        population = self.des[1].getPopulation()
        fitnesses = self.des[1].getFitnesses()
        population[0] = self.normInputs(guess)
        fitnesses[0] = cost
        self.des[1].setPopulation(population)
        self.des[1].setFitnesses(fitnesses)
            
    def isDEInitialised(self):
        return self.de_initialised

    def isNNRTrained(self):
        return self.nn_trained

    def addTrainingData(self,trials,fitnesses):
        self.nn_trials = self.nn_trials + trials
        self.nn_fitnesses = self.nn_fitnesses + fitnesses
        
    def getBest(self):

        trial = np.zeros(3)
        cost = np.ones(3)

        guess = self.nn_trials[-1]

        for ii in range(1,4):
            trial[ii-1] = self.anns[ii].getNextTrial(guess,tolerance)
            cost[ii-1] = self.anns[ii].cost(trial)

        idx_min_cost = np.argmin(cost)

        return self.denormInputs(trial[idx_min_cost])
            
    def getTrial(self):

        if not self.de_initialised:
            if self.de_init_trials == 0:
                self.de_init_pop = Just(self.des[1].getPopulation())
            self.de_init_trials += 1

            pop = self.de_init_pop.value()

            if self.de_init_trials <= len(pop):
                trial = pop[self.de_init_trials]
                
            #else:
            #    self.de_initialised = True

        if not self.nn_trained:

            trial = self.des[1].getNextTrial()

            self.de_trials += 1
            self.de_training_set[0][self.de_trials] = trial
        
        else:

            self.active_trial += 1
            if self.active_trial > 3:
                if np.random.randint(0,20) == 0:
                    trial = self.getBest()
                    self.des[1].setThisTrial(trial)
                    self.nn_trials.append(trial)
                else:
                    trial = self.des[1].getNextTrial()
                    self.nn_trials.append(trial)
            else:
                trial = self.nns[self.active_trial].getNextTrial(self.nn_trials[-1],tolerance)
                self.nn_trials.append(trial)

        return self.denormInput(trial)

    def setFitness(self,fitness):
        
        if not self.de_initialised:
            if self.de_init_trials == 1:
                self.de_init_fit = np.zeros((len(self.de_init_pop.value())))

            if self.de_init_trials <= len(pop):
                self.de_init_fit[self.de_init_trials-1] = fitness
                
            else:
                self.de_initialised = True
                self.des[1].setFitness(self.de_init_fit)
                
        if not self.nn_trained:

            if self.de_trials <= 2*self.input_length:
                self.de_training_set[1][self.de_trials-1] = fitness

                if self.de_trials == 2*self.input_length:
                    for ii in range(1,4):
                        self.nns[ii].fit(*self.de_training_set)

                    self.nn_trained = True

        else:

            if self.active_trial > 3:
                self.des[1].setFitness(fitness)
                self.active_trial = 1
                self.nn_fitnesses.append(fitness)
            else:
                self.active_trial += 1
                self.nn_fitnesses.append(fitness)

                x_train = np.array([self.de_training_set[0],np.array(self.nn_trials)])
                y_train = np.array([self.de_training_set[1],np.array(self.nn_fitness)])
                for ii in range(1,4):
                    self.nns[ii].fit(x_train,y_train)
            

        # shouldn't reach here
