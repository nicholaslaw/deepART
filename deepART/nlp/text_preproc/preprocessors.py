import pandas as pd
import re
from .stopwords_lists import stopwords_nltk 


class Corpus_stream():
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




def remove_patterns(text, patterns = None):
    '''
    remove specified patterns for individual tweets
    '''
    if patterns == None:
        try:
        # UCS-4
            EMOJIS_PATTERN = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
        except re.error:
        # UCS-2
            EMOJIS_PATTERN = re.compile(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])')
        
        patterns = {
            "URL_PATTERN":re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))'),
            "HASHTAG_PATTERN" : re.compile(r'#\w*'),
            "MENTION_PATTERN" : re.compile(r'@\w*'),
            "RESERVED_WORDS_PATTERN" : re.compile(r'^(RT|FAV)'),
            "SMILEYS_PATTERN" : re.compile(r"(?:X|:|;|=)(?:-)?(?:\)|\(|O|D|P|S){1,}", re.IGNORECASE),
            "EMOJIS_PATTERN" : EMOJIS_PATTERN
            }
    
    clean_text = text
    for pattern in patterns:
        clean_text = re.sub(patterns[pattern],"",clean_text)
    
    return clean_text


def replace_apo(text, apo_dict = None):
    '''
    replace apostrophe in text
    '''

    if apo_dict == None:
        apo_dict = {"'s" : " is",
                "'m" : " am",
                "'re": " are",
                "'ve" : " have",
                "'d" :" had",
                "'ll" : " will",
                "n't": " not",
                    "’s" : " is",
                    "‘m" : " am",
                "‘re": " are",
                "‘ve" : " have",
                "‘d" :" had",
                "‘ll" : " will",
                "n‘t": " not"} ## Need a huge dictionary
        
    clean_text = text  
    
    matches = re.findall(r"[\'\’][\w]+", text)
    if matches:
        for apo in matches:
            if apo in apo_dict:
                try:
                    clean_text = re.sub(apo,apo_dict[apo],clean_text)
                except KeyError:
                    pass
    
    return clean_text


def replace_slang(text, slang_dict = None):
    '''
    replace slang in a tweet
    '''
    
    if slang_dict == None:
        slang_dict = {"lol" : "laughing out loud",
                "brb": "be right back",
                "btw" : "by the way",
                "lmk" :"let me know",
                "g2g" : "got to go",
                "TBH": "to be honest",
                    "IMHO" : "in my humble opinion",
                    "IRL" : "in real life",
                "TL;DR": "too long; did not read",
                    } ## Need a huge dictionary
        
    clean_text = text  
    
    for sl in slang_dict:
        clean_text = re.sub(sl,slang_dict[sl],clean_text)
    
    return clean_text


def remove_shorts(text, thres = 2, ignores = "default"):
    '''
    remove short words of length less than or equal to thres
    '''
    if ignores == "default":
        ignores = ["i","you","we","they","it",
                "be","am","is","are",
                "was","were",
                "for","in","on","into","onto","of","to","over", "at","by"
                "after","before",
                "who","what","when","where","how",
                "go","went",
                "not", "the",
        ]
    elif isinstance(ignores, list):
        pass
    else:
        ignores = []
    
    clean_text = text  

    try:
        matches = re.findall(re.compile(r'\W*\b\w{1,2}\b'), clean_text)
        if matches:
            for short in matches:
                word = re.findall(r'\w+',short)
                if word[0].lower() not in ignores:
                    try:
                        clean_text = re.sub(r'\W*\b{}\b'.format(short),"",clean_text)
                    except KeyError:
                        pass
    except Exception as e:
        print(clean_text)
        print("error:",e)
    
    return clean_text


def remove_stopwords(text, stopwords="default"):
    '''
    Remove stopwords in text defined by  stopwords list
    '''
    if stopwords == "default":
        stopwords = ['a','an','the',
                    'i','you','we','they',
                    'what', 'when','where','why','how',
                    'has','have','had'
                    'do','does',
                    'can', 'could'
                    ]
    elif stopwords == "nltk":
        stopwords = stopwords_nltk

    clean_text = text.split()
    clean_text = [word for word in clean_text if word not in stopwords]
    clean_text = ' '.join(clean_text)
    
    return clean_text
