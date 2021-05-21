import spacy
import subprocess


class Dep_parser:

    def __init__(self, model='en'):
        #load model if exist, otherwise download from spacy
        try:
            self.nlp = spacy.load(model)
        except OSError:
            with subprocess.Popen('python -m spacy download {}'.format(model), shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
                for line in p.stdout.readlines():
                    print(line)
                retval = p.wait()
            self.nlp = spacy.load(model)

        self.doc = None

    def fit(self,text):
        self.doc = self.nlp(text)
    
    def transform(self,lemma = True):
        '''
        returns noun chunks with its dependencies as directed graph

        return {noun chunks: [src dst]}
        '''
        chunks = {}

        for chunk in self.doc.noun_chunks:
            #add noun chunk into dictionary and reversing the directed graph based on dependency tree
            chunks[chunk] = [chunk.lemma_, chunk.root.head.lemma_]
            if chunk.root.dep_ in ['pobj', 'dobj']:
                chunks[chunk].reverse()
        
        return chunks
    
    def fit_transform(self, text, lemma=True):
        '''
        process a text and returns nound chunks

        return {noun chunks: [src dst]}
        '''
        self.fit(text)

        return self.transform(lemma=lemma)

            
                


        
