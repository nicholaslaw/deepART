import unittest
import numpy as np
import random
from deepART import fuzzyARTMAPs, base, dataset
import concurrent.futures
        

class fuzzyARTMAPsTest(unittest.TestCase):

    def fuzzyARTMAPs_2spirals(self, neurons, rho, alpha, beta, eps, epochs, comp=True, threading=None):

        #Generate data
        npoints = 200
        Xsample = dataset.TwoSpirals(npoints, noise=0.75)
        y = Xsample.y

        #train network
        network = fuzzyARTMAPs.FuzzyARTMAPs( 2, 2, m=neurons, rho=rho, alpha=alpha, beta=beta, eps=eps, comp=comp)

        for _ in range(epochs):
            order = np.random.choice(npoints*2,npoints*2)
            for i in order:
                I = Xsample.data_normalized[i,...]
                print("{} ---> class {}\n".format(I.ravel(),y[i]))
                
                Z, k = network.fit(I.ravel(),y[i], threading=threading) 
                if not k==None:
                    print("zJ:\t{}\n".format(Z))
                    print("Class:\t{}\n".format(k))
                    print("\n\n")
                else:
                    print("Unrecognized pattern:\n")
        
        #test the trained network
        #Generate test data
        npoints = 200
        Xtest = dataset.TwoSpirals(npoints, noise=0.78)
        ytest = Xtest.y

        #add a data point that can't be classified for testing (unidentified inputs labeled as class 2)
        Xtest.addOutlier()
        ytest = Xtest.y

        ypred = []
        for i in range(npoints*2+1):
            I = Xtest.data_normalized[i,...]
            print("{} ---> class {}\n".format(I.ravel(),ytest[i]))
            #predict class of I
            Z, k = network.predict(I.ravel(), threading=threading)
            if k == None:
                ypred.append(None)
            else:
                ypred.append(k)
            print("{} ---> predicted class {}\n".format(I.ravel(),k))
        ypred = [2 if pred==None else pred for pred in ypred ] #replace None with unidentified class -> 2
    

    def test_fuzzyARTMAPs(self):
        self.fuzzyARTMAPs_2spirals(neurons=50, rho=0.8, alpha=0.1, beta=0.8, eps=0.01, epochs=3, comp=True)
        self.fuzzyARTMAPs_2spirals(neurons=50, rho=0.8, alpha=0.1, beta=0.8, eps=0.01, epochs=3, comp=True, threading=concurrent.futures.ThreadPoolExecutor(max_workers=20))


if __name__ == '__main__':
    unittest.main()

