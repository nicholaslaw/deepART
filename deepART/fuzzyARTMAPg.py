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


class FuzzyARTMAPg:
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

    def __init__(self, n1, n2, m1=20, m2=20, rhoa=0.5,rhob=0.5, alpha=0.1, beta=0.5, eps=0.01, seed=0, comp=True):

        random.seed(seed)
        self.n1 = n1
        self.m1 = m1
        self.n2 = n2
        self.m2 = m2
        # Enable complement coding
        self.comp = comp
        if self.comp:
            self.dim = 2
        else:
            self.dim = 1
        # Vigilance
        self.rhoa = rhoa
        self.rhob = rhob
        self.eps = eps
        # Choice parameter
        self.alpha = alpha
        # Learning rate
        self.beta = beta
        # Number of active units in F2
        self.active = 0
        #initialize network
        self.fuzzyARTa = FuzzyART(self.n1, self.m1, self.rhoa, self.alpha, self.beta, seed=seed, comp=self.comp)
        self.fuzzyARTb = FuzzyART(self.n2, self.m2, self.rhob, self.alpha, self.beta, seed=seed, comp=self.comp)
        self.wjk = np.ones((m1,m2))
        

        


    def fit(self,I, Y, threading=None):
        '''
        Fit I to Category K

        Args
        -----------
        I : numpy array of dimension 1
            flattened numpy array of input vector
        Y : numpy array of dimension 1
            flattened numpy array of label vector

        '''

        _rhoa = self.rhoa

        #encode 1st node if network is new
        if self.fuzzyARTa.active == 0:
            logging.info("Encode 1st node for this new network.")
            Zprota, Ja = self.fuzzyARTa.predict(I, threading=threading, learn=True)
            


        #initiate match-tracking learning    
        while True:
            #train fuzzyARTb with Y
            Zprotb, Jb = self.fuzzyARTb.predict(Y, threading=threading, learn=True)
            if Jb == None:
                logging.info("Label Y unrecognized. Increase fuzzyARTb F2 nodes to encode new class.")
                return [None, None], None

            self.fuzzyARTa.rho = _rhoa #set vigilance
            Zprota, Ja = self.fuzzyARTa.predict(I, threading=threading, learn=False) #train fuzzyARTa

            if Ja == None:
                #encode new node for unrecognized patterns
                Zprota, Ja = self.fuzzyARTa.predict(I, threading=threading, learn=True) #train fuzzyARTa
                if Ja == None:
                    #F2 nodes exhausted and learning can't be completed
                    logging.info("\n Learning can't be completed as F2 units have been exhausted.")
                    return [None, None], None
            #Match tracking sub-system
            #check for match
            if self.wjk[Ja,Jb] == 1:
                #learn the pattern
                logging.info("Learn pattern and update match tracking weights...") 
                Zprota, Ja = self.fuzzyARTa.predict(I, threading=threading, learn=True)
                #update the match tracking weights
                self.wjk[Ja,...] = np.multiply(encode(Jb,self.m2),self.wjk[Ja,...])
                return [Zprota, Zprotb], Jb
            else:
                _rhoa = (np.sum(self.fuzzyARTa._V) / np.sum(self.fuzzyARTa.F1.nodes)) + self.eps
                logging.info("Current rhoa:{}\n".format(_rhoa))
                if _rhoa > 1:
                    logging.info("Unable to fit network. Match-tracking failed to map F2 nodes to a class.")
                    return [None, None], None
                continue


    def predict(self,I, threading=None, learn=False):
        '''
        Returns: Zprot, 
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
        Zprota, Ja = self.fuzzyARTa.predict(I, threading=threading, learn=False)

        if Ja == None:
            logging.info("Unidentified class\n")
            Jb = None
            Zprotb = None
        else:
            F2b = self.wjk[Ja,...]
            Jb=decode(F2b)
            Zprotb=self.fuzzyARTb.z[0:self.dim,...,Jb]
        return [Zprota, Zprotb], Jb

    def save(self, file_path):
        '''
        file_path: string, path to save model to

        saves the model
        '''
        return save_model(self, file_path)

    def extract_F2(self, I, threading=None, learn=False):
        '''
        Returns: Zprot, 
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
        _, _ = self.fuzzyARTa.predict(I, threading=threading, learn=False)
        return self.fuzzyARTa._T