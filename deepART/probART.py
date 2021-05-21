import concurrent.futures
import numpy as np
import random
import logging
from .base import Flayer, generateComp, softmax, normalize_sum, save_model, load_model





class ProbART:
    '''
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
    norm_T : string
        'softmax' or 'sum'
    comp : boolean
        Enable/disable complement coding
    camRule : dict
        'p' -> int: power of camRule, large p leads to higher contrast enhancement
        'q' -> int: top q nodes to activate, if q == None, average activation level will be set as threshold
    fasEncode : float
        encoding rate for new nodes. '1'-> fast encode, '0' -> use learning rate,'beta'
    '''

    def __init__(self, n, m=20, rho=0.5, alpha=1e-10, beta=0.5, seed=0, norm_T='softmax', comp=True, camRule={'p':1,'q':None}, fastEncode=1):

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
        self.z = np.ones((self.dim,n,m))
        if self.comp:
            self.z[1,0:self.n,0:self.m] = np.ones((n,m)) #np.zeros((n,m))
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
        # T signals normalization method
        self.norm_T = norm_T
        #CamRule
        self.camRule = camRule

        


    def fit_predict(self,I, threading = None, learn=True):
        '''Fit category of I and return prototypes and node indexes that pass vigilance
        '''
        #define return variables
        Zprot = []
        k = []        

        #Complement coding
        if self.comp:
            Ic = 1 - I
            self.F1.nodes[0:self.dim, ...] = np.vstack((I,Ic))
        else:
            self.F1.nodes[0:self.dim,...] = I
        
        
        #1st data into network
        if self.active == 0:
            index = self.active
            #Generate signal V
            self._x = np.ones((self.dim,self.n))
            self._x = np.minimum(self.F1.nodes[0:self.dim,...],self.z[0:self.dim,...,index])
            if learn:
                Zprot, k = self._learn(self._x)
                logging.debug("New neruons coded into F2 node {}".format(k))
                return Zprot, k
            else:
                logging.debug("No active neurons. Please set learn=True")
                return None, None

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
        logging.debug("T signal ranking:\n{}\nT signal: {}".format(_TsortedIndex, self._T))
        #implement CAM rule
        self.F2.nodes[0:self.active], nodes_act, nodes_sup = self._camRule(self._T[0:self.active], p=self.camRule['p'], q=self.camRule['q'])
        logging.debug("F2 activity after CAM rule:\n{}\n".format(self.F2.nodes[0:self.active]))

        '''
        #Generate signal V
        self._V = np.zeros((self.dim,self.n))
        for i in range(self.n):
            _Vvec = np.matmul(self.z[0:self.dim, i, 0:self.active], self.F2.nodes[0:self.active])
            self._V[0:self.dim,i] = np.sum(_Vvec,axis=0)
        logging.debug("V signal : {}".format(self._V))            
        '''

        if self.active>0:
            #top-down template matching
            for index in nodes_act:
                #implement winner takes all contrast enhancement & generate x activity at F1
                self._x = np.minimum(self.F1.nodes[0:self.dim,...].reshape((self.z[0:self.dim,...,index]).shape),self.z[0:self.dim,...,index])
                # Check if nearest memory is above the vigilance level
                d = np.sum(self._x)/np.sum(self.F1.nodes) #compute matching percentage
                if d >= self.rho:
                    if learn:
                        logging.debug("Update LTM trace for node: \t {}\n".format(index))
                        self._learn(self._x, self.F2.nodes[index], index)
                    Zprot.append(self.z[0:self.dim,...,index]) # record top-down LTM trace
                    k.append((index))#category index
                    continue
            

            k_scores = [(index,self.F2.nodes[index]) for index in k]
            
            #return if at least 1 node pass vigilance criteria
            if len(Zprot) > 0:
                return Zprot, k_scores
            
        # No match found
        if learn:
            #increase active units in F2 to learn new patterns
            logging.debug("No match found. Attempting to encode new neurons:\tNeuron no {}".format(self.active+1))
            if self.active < self.m:
                Zk, index = self._learn(self.F1.nodes[0:self.dim,...])
                logging.debug("New neruons coded into class {}".format(k))
                return Zprot.append(Zk), k.append((index,-1)) #return -1 as score for new node encoded
            else:
                logging.debug("\n Learning can't be completed as F2 units have been exhausted.")
                return None, None 
        else:
            return None, None # no match is found

    def compute_T(self, j):
        _zj = np.zeros((self.dim,self.n))
        _Tvec = np.minimum(self.F1.nodes[0:self.dim,...],self.z[0:self.dim,...,j])
        _zj = self.z[0:self.dim,...,j]
        return np.sum(_Tvec)/(self.alpha+np.sum(_zj))

    def _learn(self, x, abeta=1, J=None):
        ''' Learn I '''
        if J==None: #encode new node
            J = self.active
            self.active += 1
            if self.fastEncode != 0:
                learning_rate = self.fastEncode
            else:
                learning_rate = self.beta

            self.z[0:self.dim,...,J] = abeta*learning_rate*self.F1.nodes[0:self.dim,...]+ (1-abeta*learning_rate)*(self.z[0:self.dim,...,J])
            logging.debug("Number of active nodes:\t{}\n".format(self.active))
            return self.z[0:self.dim,...,J], J
            
        #update Jth node
        self.z[0:self.dim,...,J] = self.beta*self.F2.nodes[J]*(x[0:self.dim,...].reshape((self.z[0:self.dim,...,J]).shape)) + (1-self.beta*self.F2.nodes[J])*(self.z[0:self.dim,...,J])
        logging.debug("Number of active nodes:\t{}\n".format(self.active))
        return self.z[0:self.dim,...,J], J
    
    def _camRule(self, T , p=3, q=None):
        '''
        implement CAM rule
        '''
        y = np.zeros(self.active)
        _yp = np.power(T,p)
        #normalize _yp with scaled _yp's softmax
        y = softmax(_yp)
        #y = (_yp) / (_yp.sum()+self.alpha)
        #shutdown nodes
        if q == None or q == 0:
            _yavg = y.mean()
            indexes_sup = np.argwhere(y<_yavg) #suppressed nodes
            indexes_act = np.argwhere(y>=_yavg) #active nodes

            indexes_act = [idx[0] for idx in indexes_act]
            indexes_sup = [idx[0] for idx in indexes_sup]

        else:
            if q >=self.active:
                indexes_act= np.array([[self.active]])
                indexes_sup=np.array([None])
            else:
                indexes_sup = np.argpartition(y,-(self.active-q))[:(self.active-q)]
                indexes_act = np.argpartition(y,(self.active-q))[(self.active-q):]

        if not None in indexes_sup:
            y[indexes_sup] = 0 #suppressed nodes that doens't pass threshold

        return y, indexes_act, indexes_sup


    def save(self, file_path):
        '''
        file_path: string, path to save model to

        saves the model
        '''
        return save_model(self, file_path)