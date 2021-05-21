# Example of a simple regular expression based NP chunker.# 
import nltk
from nltk import Tree
import os
from joblib import Parallel, delayed
import multiprocessing

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# import sys
# wd = sys.path.append('../')
# print(wd)


# wd=os.getcwd()
# direc=wd
# print(wd)


class ChunkPhrase(object):
    """
    This class involves two main phrase chunking: Noun and Verb phrase extraction. 
    Methodology-
    1) Input - takes in only an untokenised document with many sentences.
    2) Preprocess - tokenised & tagged & tree phrasing
    3) Parsing- Decode the tree into phrases
    4) Output- Return essay in list of dictionary of diffrent phrase grouping where NP, VP and PP are concatenate into one word
    """
    def __init__(self,method=None):
        self.method = method
    
    
    def pos_tag(self,sentence):
        tagged = nltk.pos_tag(nltk.word_tokenize(sentence))
        return tagged

    def _chunkParser(self,result):
        tree = self.chunkParser.parse(result)
        return tree
    
    def _sentencesplit(self,sentences):
        tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
        sentences = tokenizer.tokenize(sentences)
        sentences = [i.replace(".","") for i in sentences]
        return sentences
    
    def _sentenceloop(self,sentence):
        self.tmp_np=''
        self.tmp_vp=''
        self.tmp_pp=''
        self.sent_dic={}
        
        label, phrases = self._getNodes(sentence,newsent=1)

        # remove empty lists  
        self.array_result.append(self.sent_dic)
       
        return self.array_result
    
    def _dictionary(self,keys,phrases):
        # check if key existed in the dictionary
        if self.sent_dic.get(keys) == None and keys != None:
            self.sent_dic[keys]=[phrases]
            
        # Given that key existed check if value existing in the dictionary(duplication)
        else:
            list_phrases = self.sent_dic.get(keys,"")
            if phrases not in list_phrases and phrases != None:
                list_phrases.append(phrases)
                self.sent_dic[keys]=list_phrases
            else:
                pass

        return self.sent_dic
    
    def _parsing(self,method):
        
        if method == 'noun':
            parsing = 'NP: {<DT>?<JJ>*<NN>}'
            
        elif method == 'verb':
            parsing = 'VP: {<V> <NP|PP>*}'
        
        elif method == 'custom1':
            parsing = 'P: {<NN><VBD><JJ>(<CC><JJ>)?}'
        
        elif method == 'custom2':
            parsing = 'P1: {<JJ>? <NN>+ <CC>? <NN>* <VB>? <RB>* <JJ>+}'
            
        elif method == 'custom3':
            parsing = 'P2: {<JJ>+ <RB>? <JJ>* <NN>+ <VB>* <JJ>*}'
            
        elif method == 'custom4':
            parsing = 'P3: {<NP1><IN><NP2>}'
            
        elif method == 'custom5':
            parsing = 'P4: {<NP2><IN><NP1>}'
            
        elif method == None:
            parsing = """
            NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
                {<DT|PP\$>?<JJ>*<NN>}   # chunk determiner/possessive, adjectives and noun
                {<DT>?<JJ>*<NN>}       #  chunk determiners, adjectives and nouns
                {<NNP>+}                # chunk sequences of proper nouns
     
                }<[\.VI].*>+{       # unchunk any verbs, prepositions or periods
                <DT|JJ>{}<NN.*>     # merge det/adj with nouns
            PP: {<IN><NP>}               # Chunk prepositions followed by NP
            VP: {<VB.*><NP|PP>*}     # VP = verb words + NPs and PPs
                {<VB.*>?<VB.*>}
                
            
            """
            # VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
            # CLAUSE: {<NP><VP>} 
            #     NP: {<.*>*}             # start by chunking everything
            #    }<[\.VI].*>+{       # chink any verbs, prepositions or periods
            #    <.*>}{<DT>          # separate on determiners
            #    PP: {<IN><NP>}          # PP = preposition + noun phrase
            #    VP: {<VB.*><NP|PP>*}    # VP = verb words + NPs and PPs
            #    {<DT|PRP\$> <NN.*> <POS> <JJ>* <NN.*>+}
            #    {<DT|PRP\$>? <JJ>* <NN.*>+ }
            #           {<.*>}              # start by chunking each tag
            #{<MD>?<VB.*><NP|PP>}

        else:
            print("Input only noun, verb, custom1, custom2, custom3, custom4, custom5, None!")
            
        return parsing
    
    def _getNodes(self,parent,label=None,newsent= 0,phrases=None,np_phrases=None,vp_phrases=None,pp_phrases=None):
        """
        getNodes function parse the tree by finding nodes by using recursive function that will loop when it found tree
        Input: One sentence
        """
      #  print("parent:",parent)
        if newsent == 1:
            dic_sent= {}
            tmp=''  
     #       print(parent)


        
        for node in parent:
        ## break parents(tree) to each sentences -> break sentences to phrases
#             print("type:", type(node))
#             print("node:",node)
            if type(node) is nltk.Tree:

                self.tmp_np=''
                self.tmp_vp=''
                self.tmp_pp=''
                    
                if node.label() == 'ROOT':
                    print("======== Sentence =========")
#                     print("Sentence:", " ".join(node.leaves()))

                else:
                    label = node.label()
#                     print("------Non-Root-Tree-------")
#                     if node.label()== 'S':
#                         print("------Statement------")
                
                # Reloop into recursive if existing a tree
                label, phrases = self._getNodes(node,label=label, newsent=newsent,phrases=phrases,np_phrases=np_phrases,vp_phrases=vp_phrases,pp_phrases=pp_phrases)
                label=""
                np_phrases =""
                vp_phrases =""
                pp_phrases =""
            else: # not tree, join sentences together
#                 print("LABEL:",label)
#                 print("type:", type(node))
#                print("NODE:",node)
                if label == 'NP': # if input before is NP then concatenate
                    self.tmp_np += "_"+ node[0] # node should be in tuple
                    np_phrases = self.tmp_np

                elif label == 'VP':
                    self.tmp_vp += "_"+ node[0]
                    vp_phrases = self.tmp_vp
                
                elif label == 'PP':
                    self.tmp_pp += "_"+ node[0]
                    pp_phrases = self.tmp_pp
                
                else:
                    phrases = node[0]
                    try:
                        if phrases is not "" or phrases != None:
                            if phrases[0] == '_':
                                phrases=phrases[1:]
                            self._dictionary(node[1],phrases)
                    except:
                        pass

            
            
        if np_phrases is not "" or np_phrases != None: # if input before is NP then concatenate
            try:
                if np_phrases[0] == '_':
                    self._dictionary('NP',np_phrases[1:]) 
                else:
                    self._dictionary('NP',np_phrases)
            except:
                pass
        if vp_phrases is not "" or vp_phrases != None:
            try: 
                if vp_phrases[0] == '_':
                    self._dictionary('VP',vp_phrases[1:])
                else:
                    self._dictionary('NP',vp_phrases)
            except:
                pass
        if pp_phrases is not "" or pp_phrases != None:
            try:
                if pp_phrases[0] == '_':
                    self._dictionary('PP',pp_phrases[1:])
                else:
                    self._dictionary('NP',pp_phrases)
            except:
                pass
            
        return label, phrases



    
    def transform(self,sentences,method=None,chunk_label=None):
        
        sentences=self._sentencesplit(sentences)
        parsing=self._parsing(method)
        
        #Chunk Rule
        self.chunkParser = nltk.RegexpParser(parsing)
        ## RegexpParser customised rule
        #([chunk_rule, chink_rule, split_rule],chunk_label='NP')

        
        #Running multi-core tagged sentences
        num_cores = multiprocessing.cpu_count()
        tagged_sen = Parallel(n_jobs=num_cores)(delayed(self.pos_tag)(i) for i in sentences)
  #      print("tagged:",tagged_sen)
        
        # Return parsing in a tree
        num_cores = multiprocessing.cpu_count()
        chunk_results = Parallel(n_jobs=num_cores)(delayed(self._chunkParser)(i) for i in tagged_sen)
        
        final_result= []
        self.array_result = []
        
        
        for i in chunk_results:
            result= self._sentenceloop(i)
#            final_result.append(result)
#            print(result)
#         # Parsing the tree result to phrase
#         self.array_result = []
#         num_cores = multiprocessing.cpu_count()
#         final_result = Parallel(n_jobs=num_cores)(delayed(self._sentenceloop)(i) for i in chunk_results)
        
#         final_result = []
#         for sentence in results:
#             print("adjkhasghasdlihasd",sentence)
#             result = self._getNodes(sentence,sent=1)
#             final_result.append(result)
        
#        return final_result
        return result

            
