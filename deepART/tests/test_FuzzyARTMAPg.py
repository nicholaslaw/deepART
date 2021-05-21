import unittest
import numpy as np
import random
from deepART import fuzzyARTMAPg, base, dataset
from copy import deepcopy
import concurrent.futures


class fuzzyARTMAPgTest(unittest.TestCase):

    def fuzzyARTMAPg_2spirals_cluster(self, neuronsa, neuronsb, rhoa, rhob, alpha, beta, eps, epochs, comp=True, threading=None):

        #Generate data
        npoints = 200
        Xsample = dataset.TwoSpirals(npoints)
        y = Xsample.y

        #Generate label clusters
        label_data = dataset.Clusters3d(nclusters=2,npoints=npoints,spread=0.02, data_range=[0, 600])
        

        #Train network
        network = fuzzyARTMAPg.FuzzyARTMAPg(n1=2, n2=3, m1=neuronsa, m2=neuronsb, rhoa=rhoa, rhob=rhob, alpha=alpha, beta=beta, eps=0.01, seed=0, comp=True)
        print("Start training...")

        for _ in range(epochs):
            order = np.random.choice(npoints*2,npoints*2)
            for i in order:
                I = Xsample.data_normalized[i,...]
                Y = label_data.data_normalized[i,...]
                print("{} ---> class {}\n".format(I.ravel(),Y.ravel()))
                
                Z, k = network.fit(I.ravel(),Y.ravel(), threading=threading) 
                if not k==None:
                    print("zJa:\n{}\nZjb:\n{}\n".format(Z[0], Z[1]))
                    print("Class:\t{}\n".format(k))
                    print("\n\n")
                else:
                    print("Unrecognized pattern:\n")

        print("Training completed!")

        #test the trained network
        #Generate test data
        npoints = 200
        Xtest = dataset.TwoSpirals(npoints, noise=0.78)
        ytest = Xtest.y

        #add a data point that can't be classified for testing
        Xtest.addOutlier()
        ytest = Xtest.y

        #Generate test labels and add an outlier without trained class
        testLabels = deepcopy(label_data)
        testLabels.addOutlier()
        Ytest = testLabels
        
        #Predict class of test data
        ylab = []
        ypred = []

        for i in range(npoints*2+1):
            I = Xtest.data_normalized[i,...]
            Y = Ytest.data_normalized[i,...]
            
            #get the correct class of I from fuzzyARTb
            _, Jb = network.fuzzyARTb.predict(Y.ravel(),learn=False, threading=threading)
            ylab.append(Jb)
            print("{} ---> Y:{}\t class {}\n".format(I.ravel(),Y.ravel(),Jb))
            #predict class of I
            Z, k = network.predict(I.ravel())
            if k == None:
                ypred.append(None)
            else:
                ypred.append(k)
            print("{} ---> predicted class {}\n".format(I.ravel(),k))
            print("Zprota:\n{}\nZprotb:\n{}\nJb:\n{}\n".format(Z[0],Z[1],k))
        

    def test_fuzzyARTMAPg(self):
        self.fuzzyARTMAPg_2spirals_cluster(neuronsa=50, neuronsb=50, rhoa=0.85, rhob=0.80, alpha=0.1, beta=0.8, eps=0.01, epochs=3, comp=True)
        self.fuzzyARTMAPg_2spirals_cluster(neuronsa=50, neuronsb=50, rhoa=0.85, rhob=0.80, alpha=0.1, beta=0.8, eps=0.01, epochs=3, comp=True, threading=concurrent.futures.ThreadPoolExecutor(max_workers=20))


if __name__ == '__main__':
    unittest.main()

