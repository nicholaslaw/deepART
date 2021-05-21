import numpy as np
import random
import logging

from .base import Flayer, generateComp, save_model, load_model
from .fuzzyART import FuzzyART


def encode(y,k):
    '''
    Returns a vector Y with Y[y] = 1, otherwise zero

    Args:
    -----------
    y : int
        class no, starting from 0
    k : int
        no of classes
    Y : numpy array of dimension k
        vector encoding the class y
    '''
    Y = np.zeros(k)
    Y[y] = 1
    return Y

def decode(Y):
    '''
    Returns the index, y, where Y vector is '1'

    Args:
    -----------
    Y : numpy array of dimension k
        vector encoding the class y
    '''
    index = np.where(Y==1)
    y = index[0][0]
    return y


class FuzzyARTMAPs:
    '''
    FuzzyARTMAP neural network class

    Args:
    -----------
    n : int
        Size of input
    m : int
        Maximum number of internal units 
    rho : float
        Vigilance parameter
    alpha : float
        Choice parameter
    beta : float
        Learning rate
    seed : float
        Random seed
    '''

    def __init__(self, n, k, m=20, rho=0.5, alpha=0.1, beta=0.5, eps=0.01, seed=0, comp=True):

        random.seed(seed)
        self.n = n
        self.m = m
        self.k = k
        # Enable complement coding
        self.comp = comp
        if self.comp:
            self.dim = 2
        else:
            self.dim = 1
        # Vigilance
        self.rho = rho
        self.eps = eps
        # Choice parameter
        self.alpha = alpha
        # Learning rate
        self.beta = beta
        # Number of active units in F2
        self.active = 0
        #initialize network
        self.fuzzyARTa = FuzzyART(self.n, self.m, self.rho, self.alpha, self.beta, seed=seed, comp=self.comp)
        self.wjk = np.ones((m,k))
        

        


    def fit(self,I, y, threading=None):
        '''
        Fit I to Category K

        Args
        -----------
        I : numpy array of dimension 1
            flattened numpy array of input vector
        y : int
            class no of the input

        '''
        if y > self.k:
            raise("{} class index exceeds predefined number of classes, {}".format(y,self.k))

        Y = encode(y,self.k) #encode y
        _rhoa = self.rho

        #encode 1st node if network is new
        if self.fuzzyARTa.active == 0:
            logging.info("Encode 1st node for this new network.")
            Zprop, J = self.fuzzyARTa.predict(I, threading=threading, learn=True)

        #initiate match-tracking learning    
        while True:
            self.fuzzyARTa.rho = _rhoa #set vigilance
            Zprop, J = self.fuzzyARTa.predict(I, threading=threading, learn=False) #train fuzzyARTa
            if J == None:
                #encode new node for unrecognized patterns
                Zprop, J = self.fuzzyARTa.predict(I, threading=threading, learn=True) #train fuzzyARTa
                if J == None:
                    #F2 nodes exhausted and learning can't be completed
                    logging.info("\n Learning can't be completed as F2 units have been exhausted.")
                    return None, None
            #Match tracking sub-system
            #check for match
            if self.wjk[J,y] == 1:
                #learn the pattern
                logging.info("Learn pattern and update match tracking weights...") 
                Zprop, J = self.fuzzyARTa.predict(I, threading=threading, learn=True)
                #update the match tracking weights
                self.wjk[J,...] = np.multiply(Y,self.wjk[J,...])
                return Zprop, decode(self.wjk[J,...])
            else:
                _rhoa += (np.sum(self.fuzzyARTa._V) / np.sum(self.fuzzyARTa.F1.nodes)) + self.eps
                logging.info("Current rhoa:{}\n".format(_rhoa))
                if _rhoa > 1:
                    logging.info("Unable to fit network. Match-tracking failed to map F2 nodes to a class.")
                    return None, None
                continue


    def predict(self,I, threading=None, learn=False):
        '''
        Returns: Zprop, 
        category, K, classified by network

        Args
        -----------
        I : numpy array of dimension 1
            flattened numpy array of input vector
        learn : boolean
            enable/disable learning
        '''
        if self.fuzzyARTa.active == 0:
            logging.info("Network is not trained. No active node found. Call fuzzyARTs.fit to train the network.")
            return None
        
        #Send I into fuzzyARTa
        Zprop, J = self.fuzzyARTa.predict(I, threading=threading, learn=False)

        if J == None:
            logging.info("Unidentified class\n")
            y = None
        else:
            Y = self.wjk[J,...]
            y=decode(Y)
        return Zprop, y
        
    def save(self, file_path):
        '''
        file_path: string, path to save model to

        saves the model
        '''
        return save_model(self, file_path)