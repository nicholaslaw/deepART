import numpy as np
import random

class Flayer:

    def __init__(self, n):

        self.nodes = np.ones(n)


class ART1:
    ''' ART1 class
    '''

    def __init__(self, n, m=20, rho=.5, seed=0):
        '''
        ART1(n, m=20, rho=0.5, seed=0)

        Args:
        -----------
        n : int
            Size of input
        m : int
            Maximum number of internal units 
        rho : float
            Vigilance parameter
        seed : float
            Random seed
        '''
        random.seed(seed)
        self.n = n
        self.m = m
        # Comparison layer
        self.F1 = Flayer(n)
        # Recognition layer
        self.F2 = Flayer(m)
        # Feed-forward weights
        self.Wf = np.random.random((m,n))
        # Feed-back weights
        self.Wb = np.random.random((n,m))
        # Vigilance
        self.rho = rho
        # Number of active units in F2
        self.active = 0

    def predict(self,I, learn=True):
        '''Predict category of I
        '''
        self.F1.nodes[...] = I
        
        self._T = np.dot(self.Wf, self.F1.nodes)
        _TsortedIndex = np.argsort(self._T[:self.active].ravel())[::-1]
        print(_TsortedIndex)
        if _TsortedIndex.size:
            #top-down template matching
            for index in _TsortedIndex:
                #implement winner takes all contrast enhancement
                self.F2.nodes[...] = np.zeros(self.m, dtype = int)
                self.F2.nodes[index] = 1
                _V = np.dot(self.Wb, self.F2.nodes[...])
                # Check if nearest memory is above the vigilance level
                self.F1.nodes[...] = self.F1.nodes[...] * _V
                d = (self.F1.nodes[...]).sum()/I.sum() #compute matching percentage
                if d >= self.rho:
                    return self.Wb[:,index], index # return top-down LTM trace, category index
            
        # No match found
        if learn:
            #increase active units in F2 to learn new patterns
            if self.active < self.F2.nodes.size:
                Z, k = self._learn(I)
                print("New neruons coded into class {}".format(k))
                return Z, k
            else:
                print("No match found.\n Learning can't be completed as F2 units have been exhausted.")
                return None, None 
        else:
            return None, None # no match is found


    def _learn(self, I):
        ''' Learn I '''
        i = self.active
        self.Wb[:,i] *= I
        self.Wf[i,:] = self.Wb[:,i]/(0.5+self.Wb[:,i].sum())
        self.active += 1
        return self.Wb[:,i], i
