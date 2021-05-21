import numpy as np

def load_glove(gloveFile):
    '''  
    Requires packages: numpy
    
    gloveFile: string
        file path to txt file containing words and glove vectors
    
    returns a dictionary of words as keys and their corresponding vectors as values
    '''
    f = open(gloveFile,'r', encoding='utf8')
    
    word_vector = {}
    for line in f:
        splitLine = line.split(' ')
        word = splitLine[0]
        embedding = np.asarray(splitLine[1:], dtype='float32')
        word_vector[word] = embedding
    return word_vector