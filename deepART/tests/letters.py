import numpy as np
import random


def letter_to_array(letter):
        ''' Convert a letter to a numpy array '''
        shape = len(letter), len(letter[0])
        Z = np.zeros(shape, dtype=int)
        for row in range(Z.shape[0]):
            for column in range(Z.shape[1]):
                if letter[row][column] == '#':
                    Z[row][column] = 1
        return Z

def print_letter(Z):
    ''' Print an array as if it was a letter'''
    for row in range(Z.shape[0]):
        for col in range(Z.shape[1]):
            if Z[row,col]:
                print('#', end="")
            else:
                print(' ', end="")
        print( )




A = letter_to_array( [' #### ',
                      '#    #',
                      '#    #',
                      '######',
                      '#    #',
                      '#    #',
                      '#    #'] )
B = letter_to_array( ['##### ',
                      '#    #',
                      '#    #',
                      '##### ',
                      '#    #',
                      '#    #',
                      '##### '] )
C = letter_to_array( [' #### ',
                      '#    #',
                      '#     ',
                      '#     ',
                      '#     ',
                      '#    #',
                      ' #### '] )
D = letter_to_array( ['##### ',
                      '#    #',
                      '#    #',
                      '#    #',
                      '#    #',
                      '#    #',
                      '##### '] )
E = letter_to_array( ['######',
                      '#     ',
                      '#     ',
                      '####  ',
                      '#     ',
                      '#     ',
                      '######'] )
F = letter_to_array( ['######',
                      '#     ',
                      '#     ',
                      '####  ',
                      '#     ',
                      '#     ',
                      '#     '] )
G = letter_to_array( ['######',
                      '#     ',
                      '#     ',
                      '#  ###',
                      '#    #',
                      '#    #',
                      '######'] )
I = letter_to_array( ['######',
                      '  ##  ',
                      '  ##  ',
                      '  ##  ',
                      '  ##  ',
                      '  ##  ',
                      '######'] )
K = letter_to_array( ['      ',
                      '#    #',
                      '#   # ',
                      '####  ',
                      '####  ',
                      '#   # ',
                      '#    #'] )
O = letter_to_array( ['######',
                      '#    #',
                      '#    #',
                      '#    #',
                      '#    #',
                      '#    #',
                      '######'] )
Y = letter_to_array( ['#    #',
                      ' #  # ',
                      '  ##  ',
                      '  ##  ',
                      '  ##  ',
                      '  ##  ',
                      '  ##  '] )
Z = letter_to_array( ['######',
                      '     #',
                      '    # ',
                      '   #  ',
                      '  #   ',
                      ' #    ',
                      '######'] )

allLetters = [A,B,C,D,E,F,G,I,K,O,Y,Z]


class Letters:

    def __init__(self, letters = allLetters):
        self.letters = letters
    
    def randSelect(self, nsample=1, seed=0):
        '''
        return "nsample" random letters
        '''
        random.seed(seed)
        return random.sample(self.letters, nsample)