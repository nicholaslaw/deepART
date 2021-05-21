import torch
import random
import logging
from .base import generateComp, softmax, save_model, load_model
import numpy as np




class ProbART:
    '''
    ProbART(n, m=20, rho=0.5, alpha=0.1, beta=0.5, seed=0, norm_T='softmax', comp=True, fastEncode=1)

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
    fasEncode : float
        encoding rate for new nodes. '1'-> fast encode, '0' -> use learning rate,'beta'
    '''

    def __init__(self, n, m=20, rho=0.5, alpha=1e-10, beta=0.5, seed=0, norm_T='softmax', comp=True, fastEncode=1):

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
        self.F1 = torch.ones((self.dim,n), dtype=torch.float64)
        # Recognition layer
        self.F2 = torch.zeros(m, dtype=torch.float64)
        # initializes memory traces
        self.z = torch.ones((self.dim,n,m), dtype=torch.float64)
        if self.comp:
            self.z[1,0:self.n,0:self.m] = torch.ones((n,m), dtype=torch.float64) #np.zeros((n,m))
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
        #cuda flag
        self.cudaFlag = False

        


    def fit_predict(self,I, learn=True):
        '''Fit category of I and return prototypes and node indexes that pass vigilance
        '''
        #check if I is torch tensor
        if not torch.is_tensor(I):
            if isinstance(I, np.ndarray):
                I = torch.from_numpy(I)
            elif isinstance(I, list):
                torch.tensor(I)
            else:
                raise ValueError
        #define return variables
        Zprot = []
        k = []        

        #Complement coding
        if self.comp:
            Ic = 1 - I
            self.F1[0:self.dim, ...] = torch.cat((I,Ic),dim=0)
        else:
            self.F1[0:self.dim,...] = I

        #1st data into network
        if self.active == 0:
            index = self.active
            #Generate signal V
            self._x = torch.ones((self.dim,self.n), dtype=torch.float64)
            self._x = torch.min(self.F1[0:self.dim,...],self.z[0:self.dim,...,index])
            if learn:
                Zprot, k = self._learn(self._x)
                logging.debug("New neruons coded into F2 node {}".format(k))
                return Zprot, k
            else:
                logging.debug("No active neurons. Please set learn=True")
                return None, None
        
        # Generate signal T    
        self.genT()

        #Apply CAM rule
        self.F2[0:self.active], nodes_act, nodes_sup = self._camRule(self._T[0:self.active], p=1, q=None)
        logging.debug("F2 activity after CAM rule:\n{}\n".format(self.F2[0:self.active]))

        '''
        #Generate signal V
        self._V = np.zeros((self.dim,self.n))
        for i in range(self.n):
            _Vvec = np.matmul(self.z[0:self.dim, i, 0:self.active], self.F2[0:self.active])
            self._V[0:self.dim,i] = np.sum(_Vvec,axis=0)
        logging.debug("V signal : {}".format(self._V))            
        '''

        if self.active>0:
            #top-down template matching
            for index in nodes_act:
                #implement winner takes all contrast enhancement & generate x activity at F1
                self._x = torch.min(self.F1[0:self.dim,...].reshape((self.z[0:self.dim,...,index]).shape), self.z[0:self.dim,...,index])
                # Check if nearest memory is above the vigilance level
                d = torch.sum(self._x)/torch.sum(self.F1) #compute matching percentage
                if d >= self.rho:
                    if learn:
                        logging.debug("Update LTM trace for node: \t {}\n".format(index))
                        self._learn(self._x, index)
                    Zprot.append(self.z[0:self.dim,...,index]) # record top-down LTM trace
                    k.append(index)#category index
                    continue
            
            F2_copy = self.F2.clone()
            k_scores = [(index,F2_copy[index]) for index in k]
            
            #return if at least 1 node pass vigilance criteria
            if len(Zprot) > 0:
                return Zprot, k_scores
            
        # No match found
        if learn:
            #increase active units in F2 to learn new patterns
            logging.debug("No match found. Attempting to encode new neurons:\tNeuron no {}".format(self.active+1))
            if self.active < self.m:
                Zk, index = self._learn(self.F1[0:self.dim,...])
                logging.debug("New neruons coded into class {}".format(k))
                return Zprot.append(Zk), k.append((index,-1)) #return -1 as score for new node encoded
            else:
                logging.debug("\n Learning can't be completed as F2 units have been exhausted.")
                return None, None 
        else:
            return None, None # no match is found

    def genT(self):
        '''
        Generate signal T    
        '''
        if not self.cudaFlag:
            self._T = torch.zeros(self.active, dtype=torch.float64)
        else:
            self._T = torch.zeros(self.active, dtype=torch.float64, device = torch.device('cuda'))

        for j in range(0,self.active):
            if not self.cudaFlag:
                _zj = torch.zeros((self.dim,self.n), dtype=torch.float64)
            else:
                _zj = torch.zeros((self.dim,self.n), dtype=torch.float64, device = torch.device('cuda'))

            _Tvec = torch.min(self.F1[0:self.dim,...],self.z[0:self.dim,...,j])
            _zj = self.z[0:self.dim,...,j]

            self._T[j] = torch.sum(_Tvec)/(self.alpha+torch.sum(_zj))
            
        _, _TsortedIndex = torch.sort(self._T[:self.active], descending=True)
        logging.debug("T signal ranking:\n{}\nT signal: {}".format(_TsortedIndex, self._T))

    def _learn(self, x, J=-1):
        ''' Learn I '''
        if J==-1: #encode new node
            J = self.active
            self.active += 1
            if self.fastEncode != 0:
                learning_rate = self.fastEncode
            else:
                learning_rate = self.beta

            self.z[0:self.dim,...,J] = learning_rate*self.F1[0:self.dim,...]+ (1-learning_rate)*(self.z[0:self.dim,...,J])
            logging.debug("Number of active nodes:\t{}\n".format(self.active))
            return self.z[0:self.dim,...,J], J
            
        #update Jth node
        self.z[0:self.dim,...,J] = self.beta*self.F2[J]*(x[0:self.dim,...].reshape((self.z[0:self.dim,...,J]).shape)) + (1-self.beta*self.F2[J])*(self.z[0:self.dim,...,J])
        logging.debug("Number of active nodes:\t{}\n".format(self.active))
        return self.z[0:self.dim,...,J], J
    
    def _camRule(self, T , p=3, q=None):
        '''
        implement CAM rule
        '''
        if not self.cudaFlag:
            y = torch.zeros(self.active, dtype=torch.float64)
        else:
            y = torch.zeros(self.active, dtype=torch.float64, device=torch.device('cuda'))
        _yp = torch.pow(T,p)
        #normalize _yp with scaled _yp's softmax
        y = softmax(_yp)
        #y = (_yp) / (_yp.sum()+self.alpha)
        #shutdown nodes
        if q == None or q == 0:
            _yavg = y.mean()
            mask_sup = y<_yavg
            mask_act = y>_yavg
            indexes_sup = [ idx for idx, val in enumerate(mask_act) if val == 0]#suppressed nodes
            indexes_act = [ idx for idx, val in enumerate(mask_act) if val == 1] #active nodes
        else:
            #indexes_sup = np.argpartition(y,(self.active-q))[(self.active-q):]
            #indexes_act = np.argpartition(y,-(self.active-q))[-(self.active-q):]
            _, _act = torch.sort(y, descending=True)
            indexes_act = _act[0:q]
            indexes_sup = _[q:]
        y[indexes_sup] = 0 #suppressed nodes that doens't pass threshold

        return y, indexes_act, indexes_sup


    def cuda(self):
        '''
        move F1, F2, z and set CUDA flag to true
        '''
        try:
            if torch.cuda.is_available():
                self.F1cuda = self.F1.cuda()
                self.F2cuda = self.F2.cuda()
                self.zcuda = self.z.cuda()
                #self._Tcuda = torch.zeros(self.active,dtype=torch.float64, device=torch.device('cuda'))
                self.cudaFlag = True
            else:
                raise Exception("CUDA unavailable")
        except Exception as e:
            logging.exception(str(e), exc_info=True)


    def save(self, file_path):
        '''
        file_path: string, path to save model to

        saves the model
        '''
        return save_model(self, file_path)

    def load(self, file_path):
        '''
        file_path: string, file path to load the model from

        loads the model
        '''
        return load_model(file_path)