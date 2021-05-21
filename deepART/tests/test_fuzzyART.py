import unittest
import numpy as np
import random
from deepART import fuzzyART, base, dataset
import concurrent.futures

class Sample_data:

    def __init__(self, data):
        self.data = np.array(data,dtype=np.float32)
        self.data_normalized = np.array(data, dtype=np.float32)
        self.data_normalized[...,0] = self.data[...,0]/max(self.data[...,0])
        self.data_normalized[...,1] = self.data[...,1]/max(self.data[...,1])
        self.data_normalized[...,2] = self.data[...,2]/max(self.data[...,2])


class clusteringTest(unittest.TestCase):



    def fuzzyART_train(self, neurons, rho, beta, convergence,comp=False, threading=None):
        #Generate random clustered data
        sample_data = dataset.Clusters3d(nclusters=4, data_range=[0, 600])
        
        #initialize network
        network = fuzzyART.FuzzyART( sample_data.data.shape[1], neurons, rho=rho, beta=beta, comp=comp)
        print("Start testing...")
        #convergence tracking variables
        prev_active = 0
        converge = 0

        while True:
            for I in sample_data.data_normalized:
                Z, k = network.predict(I.ravel(), threading=threading) 
                if not k==None:
                    print("zJ:\t{}\n".format(Z))
                    print("Cluster:\t{}\n".format(k))
                    print("\n\n")
                else:
                    print("Unrecognized pattern:\n")
                    

            
            if (prev_active < network.active):
                prev_active = network.active
                continue
            else:
                converge +=1
                if converge > convergence:
                    print("Total Neurons Learned: {}\n\n".format(network.active))
                    break

        self.assertTrue(converge > convergence)
    
    def test_fuzzyART_training(self):
        self.fuzzyART_train(neurons=50, rho = 0.5, beta = 0.5, convergence = 5)
        self.fuzzyART_train(neurons=50, rho = 0.5, beta = 0.5, convergence = 5, threading=concurrent.futures.ThreadPoolExecutor(max_workers=20))
    
#    def test_fuzzyART_benchmark(self):
#        base.Benchmark.run(lambda: self.fuzzyART_train(neurons=50, rho = 0.5, beta = 0.5, convergence = 5))
    
    

if __name__ == '__main__':
    unittest.main()