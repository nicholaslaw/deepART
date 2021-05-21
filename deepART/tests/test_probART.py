import unittest
import numpy as np
import random
from deepART import ProbART, base, dataset




class clusteringTest(unittest.TestCase):



    def probART_train(self, neurons, rho, beta, convergence,comp=False):
        #Generate random clustered data
        sample_data = dataset.Clusters2d_overlap(nclusters=4, data_range=[0, 600])
        
        #initialize network
        network = ProbART( sample_data.data.shape[1], neurons, rho=rho, alpha = 0.1, beta=beta, comp=comp)
        print("Start testing...")
        #convergence tracking variables
        prev_active = 0
        converge = 0
        
        while True:
            for I in sample_data.data_normalized:
                Z, k = network.fit_predict(I.ravel()) 
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
    
    def test_probART_training(self):
        self.probART_train(neurons=50, rho = 0.5, beta = 0.5, convergence = 5, comp=False)
    
    

if __name__ == '__main__':
    unittest.main()