import pandas as pd
import re
import shutil
import os
from gensim import corpora
from .corpus_handler import Corpus_stream
from . import preprocessors


class BoWVectorizer:
    '''
    Defines a custom iterator for streaming corpus from csv file. 
    '''
    def __init__(self, shorts_thres = 3):
        self.dictionary = corpora.Dictionary()
        self.shorts_thres = shorts_thres # minimum length of word to keep
        self.d_w_nnz =[0,0,0]

    def fit(self, corpus_stream = None, update=True):
        '''
        Update dictionary with new documents from a corpus stream
        '''
        #Clear dictionary and refit if update set to false
        if not update:
            self.dictionary = corpora.Dictionary()

        if corpus_stream is not None:
            self.stream = corpus_stream
        else:
            print("No document stream was specified.")

        for text in self.stream:
            text = re.sub(r'[^\w\s]',' ', text)
            text = preprocessors.remove_stopwords(text.lower(), stopwords='nltk')
            text = preprocessors.remove_shorts(text,thres=self.shorts_thres)
            #tokenizer
            if tokenizer == None:
                tokens = text.split()
            else:
                tokens = tokenizer(text)
            self.dictionary.add_documents([tokens])
    
    def filter(self,stopwords=None,freq_thres=None, compactify=False):
        '''
        Filter out tokens that are unwanted using stopwords and/or by frequency
        '''
        # Filter by frequency
        if freq_thres is not None:
            thres_ids = [tokenid for tokenid, freq in self.dictionary.dfs.items() if freq <freq_thres]
            self.dictionary.filter_tokens(thres_ids)

        #filter by stopwords
        if stopwords is not None:
            stop_ids = [self.dictionary.token2id[stopword] for stopword in stopwords
                        if stopword in self.dictionary.token2id]
            self.dictionary.filter_tokens(stop_ids) 
        
        #Compactify dictionary
        if compactify:
            self.dictionary.compactify()
    
    def transform(self,corpus_stream=None,output_path = None, collection_name='default'):
        '''
        Transform a stream of documents into bow representation
        '''
        if corpus_stream is not None:
            self.stream = corpus_stream
        else:
            print("No document stream was specified.")

        if output_path == None:
            vectors =[]
        else:
            #make directory if not exist
            try:
                os.mkdir(output_path)
            except FileExistsError:
                pass

            try:
                #check if corpus exist, read in D_W_NNZ if exist
                fp = open(output_path+'docword.{}.txt'.format(collection_name),mode='r')
                line_no = 1
                for line in fp:
                    self.d_w_nnz[line_no-1]=int(line.split()[0])
                    line_no+=1
                    if line_no>3:
                        fp.close()
                        break
            except FileNotFoundError:
                #reinitialize D W NNZ
                self.d_w_nnz[:]=[0,0,0]
                fp = open(output_path+'docword.{}.txt'.format(collection_name),mode='w+')
                #write placeholder for D W NNZ
                for i in range(3):
                    fp.write('{}\n'.format(self.d_w_nnz[i]))
            else:
                fp = open(output_path+'docword.{}.txt'.format(collection_name),mode='a+')
        
        
        for text in self.stream:
            text = re.sub(r'[^\w\s]',' ', text)
            text = preprocessors.remove_stopwords(text.lower(), stopwords='nltk')
            text = preprocessors.remove_shorts(text,thres=self.shorts_thres)
            vector = self.dictionary.doc2bow(text.split())

            #output vector
            self.d_w_nnz[0]+=1
            if output_path == None:
                vectors.append(vector)
                print(vector)
            else:
                for w_id, count in vector:
                    #update nnz
                    if count > 0:
                        self.d_w_nnz[2]+=1
                    #write to file
                    fp.write('{} {} {}\n'.format(self.d_w_nnz[0],w_id+1,count))
        
        if output_path == None:
            return vectors
        else:
            fp.close()
            #update W in D_W_NNZ
            self.d_w_nnz[1] = len(self.dictionary.token2id)

            #Create temp file to update D, W, NNZ
            shutil.copy(output_path+'docword.{}.txt'.format(collection_name),output_path+'docword_temp')
            src_fp = open(output_path+'docword_temp')
            dest_fp = open(output_path+'docword.{}.txt'.format(collection_name), mode='w+')
            for i in range(3):
                line = src_fp.readline()
                dest_fp.write('{}\n'.format(self.d_w_nnz[i]))
            #form final file
            shutil.copyfileobj(src_fp,dest_fp)
            src_fp.close()
            dest_fp.close()
            os.remove(output_path+'docword_temp')

            #Create vocab file
            vocab_fp = open(output_path+'vocab.{}.txt'.format(collection_name),mode='w+')
            for token in self.dictionary.itervalues():
                vocab_fp.write('{}\n'.format(token))
            vocab_fp.close()

            return self.d_w_nnz #return D W NNZ stats of current corpus
                
    
