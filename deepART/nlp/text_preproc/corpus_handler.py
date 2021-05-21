import pandas as pd
import re


class Corpus_stream:
    '''
    Defines a custom iterator for streaming corpus from csv file. 
    '''
    def __init__(self, path, column):
        self.path = path
        self.column = column
    
    def __iter__(self):
        df = pd.read_csv(self.path,chunksize=1)
        for line in df:
            yield(line[self.column].iloc[0])