import sys, logging, joblib
import numpy as np
from time import time
from statistics import mean, stdev

#### Basic classes ###

class Flayer:

    def __init__(self, n):

        self.nodes = np.ones(n, dtype=np.float64)


### Helper functions ###

def generateComp(I):
    '''
    generateComp(I)

    Args:
    -----------
    I : numpy array
    '''
    Icomp = 1 - I

    return Icomp

def save_model(py_obj, file_path):
    '''
    py_obj: Python object
    file_path: string, file path to save

    saves ART model
    '''
    joblib.dump(py_obj, open(file_path, 'wb'))

def load_model(file_path):
    '''
    file_path: string, file path to save

    loads ART model
    '''
    return joblib.load(open(file_path, 'rb'))

def normalize_sum(lst):
    s = sum(lst)
    if s == 0:
        return [0 for _ in range(len(lst))] #if all elements in lst is zero -> return a list of zeros
    norm = [ float(x)/s for x in lst] #calculate the normalized values for the list
    return norm

def normalize_max(lst, alpha = 1e-30):
    x_max = max(lst)
    norm = [float(x)/x_max for x in lst]
    return norm


def softmax(lst, scale = True, alpha = 1e-30):
    """Compute softmax values for each sets of scores in x."""
    #scaled max x and multipley by 10 to increase by one order
    if scale:
        lst = 10*lst/(max(lst)+alpha)
    return np.exp(lst) / np.sum(np.exp(lst), axis=0)

### Other classes and functions ###
class Benchmark:

    @staticmethod
    def run(function):
        timings = []
        stdout = sys.stdout
        for i in range(100):
            sys.stdout = None
            startTime = time()
            function()
            seconds = time() - startTime 
            sys.stdout = stdout
            timings.append(seconds)
            mean_time = mean(timings)
            if i < 10 or i % 10 == 9:
                print("{} {:3.2f} {:3.2f}".format( 1 + i, mean_time,
                        stdev(timings, mean_time) 
                        if i > 1 else 0)) 