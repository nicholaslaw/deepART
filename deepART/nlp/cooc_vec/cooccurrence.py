import numpy as np
import math, joblib
from deepART import base

class CoocVectorizer:
    def __init__(self, context_window=2, vocabulary=None, max_df=1, min_df=1, max_features=None, lowercase=True, normalize=2, cooc_likelihood=True):
        '''
        Requires packages:

        context_window: int, default=2
            Size of context window
        
        vocabulary: dict, default=None
            A mapping of terms to feature indices

        max_df: float in range [0.0, 1.0] or int, default=1.0
            When building the vocabulary ignore terms that have a document frequency strictly higher than the given threshold (corpus-specific stop words).
            If float, the parameter represents a proportion of documents, integer absolute counts. This parameter is ignored if vocabulary is not None.

        min_df: float in range [0.0, 1.0] or int, default=1
            When building the vocabulary ignore terms that have a document frequency strictly lower than the given threshold.
            This value is also called cut-off in the literature. If float, the parameter represents a proportion of documents, integer absolute counts.
            This parameter is ignored if vocabulary is not None.

        max_features: int or None, default=None
            If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus.
            This parameter is ignored if vocabulary is not None.

        lowercase: boolean, default=True
            Convert all characters to lowercase before tokenizing.

        normalize: int or None, default=2
            Use l1 or l2 normalization for rows of matrix

        cooc_likelihood: boolean, default=True
            Use ordinary co-occurrence statistics and measure the co-occurrence likelihood between two words by mutual information estimate, Church and Hanks 1989.
            I(X, Y) = log^{+} [P(X | Y) / P(X)]
            P(X): occurrence density of word X in the whole corpus
            P(X | Y): density of X in a neighborhood of word Y
        '''
        if context_window <= 0:
            raise ValueError('Context Window of Size > 0 Expected')
        self.context_window = context_window
        if isinstance(vocabulary, dict) or vocabulary is None:
            self.vocabulary = vocabulary
            self._validate_vocabulary()
        else:
            raise TypeError('vocabulary must be a dictionary or None')

        if max_df < 0 or min_df < 0:
            raise ValueError("negative value for max_df or min_df")
        if max_df > 1:
            if not isinstance(max_df, int):
                raise ValueError('max_df must be an integer when it is more than 1')
        if min_df > 1:
            if not isinstance(min_df, int):
                raise ValueError('max_df must be an integer when it is more than 1')
        self.max_df = max_df
        self.min_df = min_df

        if max_features is not None:
            if (not isinstance(max_features, int) or max_features<=0):
                raise ValueError('max_features must be a positive integer or None')
        self.max_features = max_features

        if isinstance(lowercase, bool):
            self.lowercase = lowercase
        else:
            raise TypeError('lowercase must be either True or False')

        if isinstance(normalize, int) or normalize is None:
            self.normalize = normalize
        else:
            raise ValueError('normalize must be 1 or 2, an integer')
            
        if isinstance(cooc_likelihood, bool):
            self.cooc_likelihood = cooc_likelihood
        else:
            raise TypeError('cooc_likelihood must be either True or False')
        self.fitted = False
        self.rejected_words = {}
        self.word_counts = {}
        self.word_counts = {}
        self.new_words = []
        self.counts_matrix = None
        self.matrix = None

    
    def _validate_vocabulary(self):
        '''
        Requires packages: N.A.

        Checks whether input vocabulary contains repeated indices
        '''
        vocabulary = self.vocabulary
        if vocabulary is not None:
            if isinstance(vocabulary, dict):
                if len(set(vocabulary.values())) != len(list(vocabulary)):
                    raise Exception('Input Dictionary Contains Repeated Indices')

    def _check_vocabulary(self, raw_documents):
        '''
        Requires packages: N.A.

        raw_documents: list of tokenized sentences

        Checks whether new documents contain words not contained in current vocabulary
        '''
        if self.vocabulary is None or len(self.vocabulary) == 0:
            raise Exception('Vocabulary is empty')
        new_doc_vocab = self.build_vocab_idx_dic(raw_documents)
        new_doc_vocab = list(new_doc_vocab.keys())
        old_doc_vocab = list(self.vocabulary.keys())
        new_words_lst = []
        for new_word in new_doc_vocab:
            if new_word not in old_doc_vocab:
                new_words_lst.append(new_word)
        if len(new_words_lst) != 0:
            raise Exception('Documents Consist of The Following Words Not Found in Current Vocabulary:\n {}'.format(new_words_lst))



    def build_vocab_idx_dic(self, raw_documents):
        '''
        Requires packages: N.A.

        raw_documents: list of tokenized sentences

        returns a dictionary of distinct words as keys and their corresponding indices
        '''
        print('Building Vocab-Index Dictionary...')
        result = set([])
        if self.lowercase:
            for splitted in raw_documents:
                for i in splitted:
                    result.add(i.lower())
        else:
            for splitted in raw_documents:
                for i in splitted:
                    result.add(i)
        final = {w: idx for idx, w in enumerate(list(result))}
        print('Number of Distinct Words: ', len(result), '\n\n')
        return final

    def update_vocab_idx_dic(self, raw_documents):
        '''
        Requires packages:

        raw_documents: list
            a list of tokenized sentences

        return an updated vocab dictionary
        '''
        print('Updating Vocabulary With New Words...')
        if self.lowercase:
            distinct_words = list(set([word.lower() for sent in raw_documents for word in sent]))
        else:
            distinct_words = list(set([word for sent in raw_documents for word in sent]))
        counter = self.vocabulary_size
        for word in distinct_words:
            if word not in self.vocabulary.keys():
                self.vocabulary[word] = counter
                counter += 1
        self.vocabulary_size = counter
        print('Done...\n')

    def tokenize_idx(self, raw_documents):
        '''
        Requires packages: N.A.

        raw_documents: list of tokenized sentences

        returns a list of lists where each list is a list of indices representing tokens
        '''
        print('Swapping Words with Their Respective Indices...\n\n')
        result = []
        if self.lowercase:
            for splitted in raw_documents:
                temp = [self.vocabulary.get(word.lower(), None) for word in splitted]
                result.append(temp)
        else:
            for splitted in raw_documents:
                temp = [self.vocabulary.get(word, None) for word in splitted]
                result.append(temp)
        print('Done...\n\n')
        return result

    def build_cooc_matrix(self, tokenized_idx_lst):
        '''
        Requires packages: numpy, math

        returns a computed co-occurence matrix
        '''
        print('Building Co-Occurence Matrix...')
        m = np.zeros((self.vocabulary_size, self.vocabulary_size))
        for sent in tokenized_idx_lst:
            sent_len = len(sent)
            for i, word in enumerate(sent):
                for j in range(i - self.context_window, i + self.context_window + 1):
                    if j < 0 or j > (sent_len - 1):
                        continue
                    else:
                        if word != sent[j]:
                            if word is not None and sent[j] is not None:
                                m[word,sent[j]]+=1
        return m

    def build_likelihood_matrix(self, counts_matrix):
        print('Converting All Matrix Entries to Mutual Information Estimates...')
        result = np.copy(counts_matrix)
        total_sum = np.sum(counts_matrix) # for P(X)
        row_sum = np.sum(counts_matrix, axis=0) # for P(X|Y)
        for idx, i in enumerate(counts_matrix):
            for idxlor, j in enumerate(i):
                if total_sum != 0:
                    P_X = j / total_sum
                else:
                    P_X = 0
                if row_sum[idxlor] != 0:
                    P_X_GIVEN_Y = j / row_sum[idxlor]
                else:
                    P_X_GIVEN_Y = 0
                try:
                    computed = P_X_GIVEN_Y / P_X 
                except ZeroDivisionError:
                    computed = 0
                if math.isnan(computed):
                    computed = 0
                if computed < 1:
                    result[idx, idxlor] = 0
                else:
                    result[idx, idxlor] = math.log10(computed)
        print('Done...\n')
        return result
    
    def normalize_matrix(self, matrix):
        print('Normalizing Matrix...')
        m = matrix.copy()
        for idx in range(len(m)):
            if np.linalg.norm(m[idx, :], ord=self.normalize) != 0:
                m[idx, :] /= np.linalg.norm(m[idx, :], ord=self.normalize) 
        print('Done...\n') 
        return m

    def update_cooc_matrix(self, tokenized_idx_lst):
        '''
        Updates cooccurrence matrix
        '''
        print('Updating Cooccurrence Matrix...\n')
        new_matrix = self.build_cooc_matrix(tokenized_idx_lst)
        previous_size = self.counts_matrix.shape[0]
        incremented = np.zeros((previous_size, self.vocabulary_size-previous_size))
        padded = np.hstack((self.counts_matrix, incremented))
        padded = np.vstack((padded, np.zeros((self.vocabulary_size-previous_size, self.vocabulary_size))))
        self.counts_matrix = padded + new_matrix
        print('Done...\n')

    def extract_word_vec(self, tokenized_sents):
        '''
        Requires packages: N.A.

        tokenized_sents: list
            a list of lists where each list contains indices representing tokens

        returns a list of lists where each list contains word vectors
        '''
        print('Extracting Cooccurrence Vectors...\n')
        if self.matrix is None:
            raise Exception('Cooccurrence Matrix is Empty, Check Whether Vectorizer Has Been Fitted')
        if not isinstance(tokenized_sents, list):
            raise TypeError("Expected List Type for Tokenized Sentences")
        if len(tokenized_sents) == 0:
            raise Exception('Empty List')
        result = []
        for sent in tokenized_sents:
            result.append([self.matrix[token_idx, :] if token_idx is not None else np.zeros(self.vocabulary_size) for token_idx in sent])
        print('Done...\n')
        return result

    def calc_word_count(self, raw_documents):
        if self.lowercase:
            for splitted in raw_documents:
                for word in splitted:
                    word = word.lower()
                    if word not in self.word_counts.keys():
                        self.word_counts[word] = 1
                    else:
                        self.word_counts[word] += 1
        else:
            for splitted in raw_documents:
                for word in splitted:
                    if word not in self.word_counts.keys():
                        self.word_counts[word] = 1
                    else:
                        self.word_counts[word] += 1

    def filter_by_count(self):
        '''
        Requires packages:

        raw_documents: list
            list of lists where each list contains tokens
        
        returns list of words that pass filter
        '''
        total = sum(self.word_counts.values())
        if isinstance(self.max_df, int) and isinstance(self.min_df, int):
            filtered_words = [word for word, count in self.word_counts.items() if self.min_df<=count<=self.max_df]
        elif isinstance(self.max_df, float) and isinstance(self.min_df, float):
            filtered_words = [word for word, count in self.word_counts.items() if self.min_df<=(count/total)<=self.max_df]
        elif isinstance(self.max_df, int):
            filtered_words = [(word, count) for word, count in self.word_counts.items() if count<=self.max_df]
            filtered_words = [word for word, count in filtered_words if (count/total)>=self.min_df]
        else:
            filtered_words = [(word, count) for word, count in self.word_counts.items() if count>=self.min_df]
            filtered_words = [word for word, count in filtered_words if (count/total)<=self.max_df]
        self.rejected_words = {}
        for word in self.word_counts.keys():
            if word not in filtered_words:
                temp = list(self.word_counts.keys())
                temp.remove(word)
                temp = {i: 0 for i in temp}
                self.rejected_words[word] = temp
        return filtered_words

    def update_word_count(self, raw_documents):
        '''
        Requires packages:

        raw_documents: list
            a list of lists where each list contains tokens

        updates the word count dictionary
        '''
        if self.lowercase:
            for splitted in raw_documents:
                for word in splitted:
                    word = word.lower()
                    if word not in self.word_counts.keys():
                        self.word_counts[word] = 1
                        self.new_words.append(word)
                    else:
                        self.word_counts[word] += 1
        else:
            for splitted in raw_documents:
                for word in splitted:
                    if word not in self.word_counts.keys():
                        self.word_counts[word] = 1
                        self.new_words.append(word)
                    else:
                        self.word_counts[word] += 1  

    def update_rejected_words_vocab(self, raw_documents):
        '''
        Requires packages: N.A.

        Obtains words which passed filter, see whether any of previously rejected words pass filter, update vocabulary,
        obtain newly rejected words and update rejected words dictionary
        '''
        # Obtain Words Which Passes Filter
        total = sum(self.word_counts.values())
        if isinstance(self.max_df, int) and isinstance(self.min_df, int):
            filtered_words = [word for word, count in self.word_counts.items() if self.min_df<=count<=self.max_df]
        elif isinstance(self.max_df, float) and isinstance(self.min_df, float):
            filtered_words = [word for word, count in self.word_counts.items() if self.min_df<=(count/total)<=self.max_df]
        elif isinstance(self.max_df, int):
            filtered_words = [(word, count) for word, count in self.word_counts.items() if count<=self.max_df]
            filtered_words = [word for word, count in filtered_words if (count/total)>=self.min_df]
        else:
            filtered_words = [(word, count) for word, count in self.word_counts.items() if count>=self.min_df]
            filtered_words = [word for word, count in filtered_words if (count/total)<=self.max_df]

        # print('FILTERED_WORDS: {}\n'.format(filtered_words))
        # obtain current list of rejected words, including those not seen before
        current_rejected = [word for word in self.word_counts.keys() if word not in filtered_words] 
        # print('CURRENT_REJECTED: {}\n'.format(current_rejected))
        # print('REJECTED KEYS: {}\n'.format(self.rejected_words.keys()))
        # obtain newly rejected words, including new words not seen by vectorizer as it can include previously accepted words
        newly_rejected_words = [word for word in current_rejected if word not in self.rejected_words.keys()] 
        # print('NEWLY REJECTED WORDS: {}\n'.format(newly_rejected_words))
        # update rejected words cooc dic with newly seen words
        if len(newly_rejected_words) > 0:
            lst = list(self.rejected_words.items())
            self.rejected_words = {}
            for word, dic in lst: #update existing rejected words cooc dic with newly seen words
                for new in self.new_words:
                    dic[new] = 0
                self.rejected_words[word] = dic
            sample_val = {word: 0 for word in list(self.word_counts.keys())}
            for new in newly_rejected_words: # update rejected words dictionary with newly rejected words
                temp = sample_val.copy()
                del temp[new]
                if new in self.vocabulary.keys():
                    for word in temp.keys():
                        if word in self.vocabulary.keys():
                            temp[word] = int(self.counts_matrix[self.vocabulary[new], self.vocabulary[word]])
                        elif word in self.rejected_words.keys():
                            temp[word] = self.rejected_words[word][new]
                self.rejected_words[new] = temp


        # Update cooccurrence rejected dic
        self.build_rejected_cooc(raw_documents)


        newly_accepted_words = [word for word in filtered_words if word in self.rejected_words.keys()] # newly accepted words which did not pass filter previously
        # print('NEWLY ACCEPTED WORDS: {}\n'.format(newly_accepted_words))
        
        # Add Newly Accepted Words into Dictionary and Update Cooccurrence Counts Matrix
        if len(newly_accepted_words) > 0:
            counter = self.vocabulary_size
            for idx, word in enumerate(newly_accepted_words):
                add_vector = np.zeros(self.vocabulary_size+1)
                vec_entries = [(i, count) for i, count in self.rejected_words[word].items() if i in self.vocabulary.keys()]
                for entry, count in vec_entries:
                    add_vector[self.vocabulary[entry]] = count
                assert add_vector[-1] == 0 # remove in the future
                self.counts_matrix = np.hstack((self.counts_matrix, np.zeros(self.vocabulary_size).reshape(-1, 1))) # horizontal stacking of column vector
                self.counts_matrix = np.vstack((self.counts_matrix, add_vector))
                self.counts_matrix[:, -1] = add_vector
                del self.rejected_words[word] # delete this key from the rejected words dic
                self.vocabulary[word] = counter + idx
                self.vocabulary_size = len(self.vocabulary)

        just_rejected = [word for word in self.rejected_words.keys() if word in self.vocabulary] # list of words which were not rejected previously but rejected in this run
        # print('just_rejected: {}\n'.format(just_rejected))
        if len(just_rejected) > 0:
            accepted_vocab_idx = {word: idx for word, idx in self.vocabulary.items() if word not in just_rejected} # obtain previously seen words not rejected in this run
            # print('ACCEPTED_VOCAB_IDX: {}\n'.format(accepted_vocab_idx))
            self.vocabulary = {word: idx for idx, word in enumerate(list(accepted_vocab_idx.keys()))}
            new_vocab = list(self.vocabulary.keys())
            self.vocabulary_size = len(self.vocabulary)
            init_mat = np.zeros((self.vocabulary_size, self.vocabulary_size))
            for i in range(self.vocabulary_size): # Create new matrix from previously seen and accepted words
                for j in range(i):
                    init_mat[i, j] = self.counts_matrix[accepted_vocab_idx[new_vocab[i]], accepted_vocab_idx[new_vocab[j]]]
            self.counts_matrix = init_mat

        # This code block below is for those words which are new and accepted by the vectorizer, update vocab dic
        all_seen_words = list(self.rejected_words.keys()) + list(self.vocabulary.keys())
        counter = self.vocabulary_size
        if self.lowercase:
            for idx, word in enumerate(filtered_words):
                if word not in all_seen_words:
                    self.vocabulary[word.lower()] = counter
                    counter += 1
        else:
            for idx, word in enumerate(filtered_words):
                if word not in all_seen_words:
                    self.vocabulary[word] = counter
                    counter += 1
        # Never update cooccurrence counts matrix here is because later would have an update cooc matrix function being called
        self.vocabulary_size = len(self.vocabulary)
    
    def build_rejected_cooc(self, raw_documents):
        '''
        Requires packages: 

        raw_documents: list
            a list of lists where each list contains tokens (not indices)

        returns a dictionary of words as keys and corresponding values are dictionaries containing other words as keys
        and values represent the cooccurrence value
        '''
        print('Building A Dictionary of Occurrences for Rejected Words for Storage...\n')
        if self.lowercase:
            for splitted in raw_documents:
                sent_len = len(splitted)
                for i, word in enumerate(splitted):
                    word = word.lower()
                    for j in range(i - self.context_window, i + self.context_window + 1):
                        if j < 0 or j > (sent_len - 1):
                            continue
                        else:
                            temp = splitted[j].lower()
                            if word != temp:
                                if word in self.rejected_words.keys():
                                    self.rejected_words[word][temp] += 1 
        else:
            for splitted in raw_documents:
                sent_len = len(splitted)
                for i, word in enumerate(splitted):
                    for j in range(i - self.context_window, i + self.context_window + 1):
                        if j < 0 or j > (sent_len - 1):
                            continue
                        else:
                            if word != splitted[j]:
                                if word in self.rejected_words.keys():
                                    self.rejected_words[word][splitted[j]] += 1 

    def fit(self, raw_documents):
        '''
        Requires packages:

        raw_documents: list
            A list of preprocessed documents, tokenized

        Allows vectorizer to learn vocabulary and build cooccurrence matrix
        '''
        print('Fitting of Vectorizer Starts...\n')
        if not self.fitted:
            self.fit_transform(raw_documents, just_fit=True)
        else:
            self.update_word_count(raw_documents)
            if self.min_df != 1 or self.max_df != 1:
                self.update_rejected_words_vocab(raw_documents)
            else:
                self.update_vocab_idx_dic(raw_documents)
            all_tokens = self.tokenize_idx(raw_documents)
            self.update_cooc_matrix(all_tokens)
            if self.cooc_likelihood:
                self.matrix = self.build_likelihood_matrix(self.counts_matrix)
            else:
                self.matrix = None
            if self.normalize is not None:
                if self.matrix is not None:
                    self.matrix = self.normalize_matrix(self.matrix)
                else:
                    self.matrix = self.normalize_matrix(self.counts_matrix)
        print('Done...\n')
        return self

    def fit_transform(self, raw_documents, just_fit=False):
        '''
        Requires packages:

        raw_documents: list
            A list of preprocessed documents, tokenized
        
        Allows vectorizer to learn vocabulary and build cooccurrence matrix
        '''
        if not isinstance(raw_documents, list):
            raise TypeError('Input of List Type Expected')
        if len(raw_documents) == 0:
            raise Exception('There Are No Documents Inserted')
        self.calc_word_count(raw_documents)
        if self.vocabulary is None:
            if self.min_df != 1 or self.max_df != 1:
                accepted_words = self.filter_by_count()
                self.build_rejected_cooc(raw_documents)
                self.vocabulary = {word: idx for idx, word in enumerate(accepted_words)}
            else:
                self.vocabulary = self.build_vocab_idx_dic(raw_documents)
        if len(self.vocabulary) == 0:
            raise Exception('Vocabulary is empty')
        self._validate_vocabulary()
        self.vocabulary_size = len(self.vocabulary)
        all_tokens = self.tokenize_idx(raw_documents)
        self.counts_matrix = self.build_cooc_matrix(all_tokens)
        self.fitted = True 
        if self.cooc_likelihood:
            self.matrix = self.build_likelihood_matrix(self.counts_matrix)
        else:
            self.matrix = self.counts_matrix
        if self.normalize is not None:
            self.matrix = self.normalize_matrix(self.matrix)
        if not just_fit:
            X = self.extract_word_vec(all_tokens)
            return X

    def transform(self, raw_documents): 
        '''
        Requires packages:

        raw_documents: list
            A list of preprocessed documents, tokenized
        '''
        if not self.fitted:
            raise Exception('Vectorizer Is Not Fitted')
        if not isinstance(raw_documents, list):
            raise TypeError('Input of List Type Expected')
        if self.vocabulary is None or len(self.vocabulary) == 0:
            raise Exception('Vocabulary is empty')
        self._validate_vocabulary()
        self._check_vocabulary(raw_documents)
        all_tokens = self.tokenize_idx(raw_documents)
        X = self.extract_word_vec(all_tokens)
        return X

    def save_cooc_matrix(self, save_path):
        '''
        Requires packages: N.A.

        matrix: co-occurrence matrix
        save_path: string, file path to save matrix
        '''
        print('Saving Matrix to {}\n'.format(save_path))
        vocab_lst = [word for word in self.vocabulary.keys()]
        with open(save_path, 'w') as f:
            for i in range(len(self.matrix)):
                f.write(str(vocab_lst[i]) + '\t' + " ".join(map(str, self.matrix[i, :])))
                f.write('\n')
        print('Saved Successfully to {}...\n\n'.format(save_path))

    def save(self, save_path):
        '''
        Saves vectorizer
        '''
        return base.save_model(self, save_path)