import numpy as np
import unittest
import random
from .letters import Letters, print_letter, letter_to_array
from deepART import art1, base, fuzzyART
from math import floor



class lettersTest(unittest.TestCase):

    def ART1_train(self, neurons, rho, convergence, trainRandom=False):
        network = art1.ART1( 6*7, neurons, rho)

        prev_active = 0
        converge = 0
        samples = Letters()
        nsamples = floor(len(samples.letters)*0.75)
        
        while True:
            if trainRandom:
                train_set = samples.randSelect(nsamples)
            else:
                train_set = samples.letters[0:nsamples]
            for I in train_set:
                Z, k = network.predict(I.ravel()) 
                print_letter(I)
                print("\n","----------","\n", "class->", k)
                
                if not k==None:
                    print_letter(Z.reshape(7,6))
                    print("\n\n\n\n")
                else:
                    print("Unrecognized pattern:\n")
                    print_letter(I)

            
            if (prev_active < network.active):
                prev_active = network.active
                continue
            else:
                converge +=1
                if converge > convergence:
                    print("Total Neurons Learned: {}\n\n".format(network.active))
                    break

        self.assertTrue(converge > convergence)
    
    def test_ART1_training(self):
        self.ART1_train(neurons=50, rho = 0.5, convergence = 5, trainRandom=True)
    
#    def test_ART1_training_benchmark(self):
#        base.Benchmark.run(lambda: self.ART1_train(neurons=50, rho = 0.5, convergence = 5, trainRandom=True))
    
    


    def fuzzyART_train(self, neurons, rho, convergence, trainRandom=False):
        network = fuzzyART.FuzzyArt( 6*7, neurons, rho)

        prev_active = 0
        converge = 0
        samples = Letters()
        nsamples = floor(len(samples.letters)*0.75)
        
        while True:
            if trainRandom:
                train_set = samples.randSelect(nsamples)
            else:
                train_set = samples.letters[0:nsamples]
            for I in train_set:
                Z, k = network.predict(I.ravel()) 
                print_letter(I)
                print("\n","----------","\n", "class->", k)
                
                if not k==None:
                    print_letter(Z.reshape(7,6))
                    print("\n\n\n\n")
                else:
                    print("Unrecognized pattern:\n")
                    print_letter(I)

            
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
        self.ART1_train(neurons=50, rho = 0.5, convergence = 5, trainRandom=True)

if __name__ == '__main__':
    unittest.main()



