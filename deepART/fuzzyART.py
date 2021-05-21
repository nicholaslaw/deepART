import numpy as np
import random
import logging
from .base import Flayer, generateComp, save_model, load_model
import concurrent.futures
import time


class FuzzyART:
    '''
    FuzzyART(n, m=20, rho=0.5, alpha=0.1, beta=0.5, seed=0, comp=True)

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

    def __init__(self, n, m=20, rho=0.5, alpha=0.1, beta=0.5, seed=0, comp=True, fastEncode=1):

        random.seed(seed)
        self.n = n
        self.m = m
        # Enable complement coding
        self.comp = comp
        if self.comp:
            self.dim = 2
        else:
            self.dim = 1
        # Comparison layer
        self.F1 = Flayer((self.dim,n))
        # Recognition layer
        self.F2 = Flayer(m)
        # initializes memory traces
        self.z = np.ones((self.dim,n,1))
        if self.comp:
            self.z[1,0:self.n,:] = np.ones((n,1))
        # Vigilance
        self.rho = rho
        # Choice parameter
        self.alpha = alpha
        # Learning rate
        self.beta = beta
        # Number of active units in F2
        self.active = 0
        # Fast encoding for new nodes
        self.fastEncode = fastEncode

        


    def predict(self,I, threading=None, learn=True):
        '''Predict category of I
        '''

        #Complement coding
        if self.comp:
            Ic = 1 - I
            self.F1.nodes[0:self.dim, ...] = np.vstack((I,Ic))
        else:
            self.F1.nodes[0:self.dim,...] = I
        
        # Generate signal T    
        self._T = np.zeros(self.active)
        if threading is not None:
            future_to_T = {threading.submit(self.compute_T, i): i for i in range(self.active)}
            for future in concurrent.futures.as_completed(future_to_T):
                index = future_to_T[future]
                try:
                    vector = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (index, exc))
                else:
                    self._T[index] = vector
        else:
            for j in range(0,self.active):
                _zj = np.zeros((self.dim,self.n))
                _Tvec = np.minimum(self.F1.nodes[0:self.dim,...],self.z[0:self.dim,...,j])
                _zj = self.z[0:self.dim,...,j]

                self._T[j] = np.sum(_Tvec)/(self.alpha+np.sum(_zj))

        _TsortedIndex = np.argsort(self._T[:self.active].ravel())[::-1]
        logging.info("T signal ranking:\n{}\n".format(_TsortedIndex))

        #1st data into network
        if self.active == 0:
            index = self.active
            #Generate signal V
            self._V = np.ones((self.dim,self.n))
            self._V = np.minimum(self.F1.nodes[0:self.dim,...],self.z[0:self.dim,...,index])
            if learn:
                Zprot, k = self._learn(self._V)
                logging.info("New neruons coded into F2 node {}".format(k))
                return Zprot, k
            else:
                logging.info("No active neurons. Please set learn=True")
                return None, None
        
        if self.active>0:
            #top-down template matching
            for index in _TsortedIndex:
                #implement winner takes all contrast enhancement
                self.F2.nodes[...] = np.zeros(self.m, dtype =np.float32)
                self.F2.nodes[index] = 1
                #Generate signal V
                self._V = np.zeros((self.dim,self.n))
                self._V = np.minimum(self.F1.nodes[0:self.dim,...],self.z[0:self.dim,...,index])

                # Check if nearest memory is above the vigilance level
                d = np.sum(self._V)/np.sum(self.F1.nodes) #compute matching percentage
                if d >= self.rho:
                    if learn:
                        logging.info("Update LTM trace for node: \t {}\n".format(index))
                        self._learn(self._V, index)
                    return self.z[0:self.dim,...,index], index # return top-down LTM trace, category index
            
        # No match found
        if learn:
            #increase active units in F2 to learn new patterns
            logging.info("No match found. Attempting to encode new neurons:\tNeuron no {}".format(self.active+1))
            if self.active < self.m:
                Zprot, k = self._learn(self._V)
                logging.info("New neruons coded into class {}".format(k))
                return Zprot, k
            else:
                logging.info("\n Learning can't be completed as F2 units have been exhausted.")
                return None, None 
        else:
            return None, None # no match is found


    def _learn(self, V, J=None):
        ''' Learn I '''
        if J==None: #encode new node
            J = self.active
            self.active += 1
            if self.fastEncode != 0:
                learning_rate = self.fastEncode
            else:
                learning_rate = self.beta
            if self.z.shape[2] < (self.active+1):
                self.z = np.dstack((self.z, np.ones((self.dim, self.n, self.active + 1 - self.z.shape[2]))))
            self.z[0:self.dim,...,J] = learning_rate*self.F1.nodes[0:self.dim,...]+ (1-learning_rate)*(self.z[0:self.dim,...,J])
            logging.info("Number of active nodes:\t{}\n".format(self.active))
            return self.z[0:self.dim,...,J], J
            
        #update Jth node
        self.z[0:self.dim,...,J] = self.beta*(V[0:self.dim,...]) + (1-self.beta)*(self.z[0:self.dim,...,J])
        logging.info("Number of active nodes:\t{}\n".format(self.active))
        return self.z[0:self.dim,...,J], J
    
    def save(self, file_path):
        '''
        file_path: string, path to save model to

        saves the model
        '''
        return save_model(self, file_path)

    def compute_T(self, j):
        _zj = np.zeros((self.dim,self.n))
        _Tvec = np.minimum(self.F1.nodes[0:self.dim,...],self.z[0:self.dim,...,j])
        _zj = self.z[0:self.dim,...,j]
        return np.sum(_Tvec)/(self.alpha+np.sum(_zj))