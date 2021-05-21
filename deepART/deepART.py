from .probART import ProbART
import numpy as np
from .base import save_model

class Sequential:

    def __init__(self, layers=None):
        self.layers_ = []
        self.all_params = {}
        self.all_active = {}
        self.load_helper = {'failed': False, 'submodel_idx': None}
        if layers:
            for layer in layers:
                self.add(layer)
    def add(self, layer):
        if not isinstance(layer, ProbART):
            raise TypeError('The added layer must be an instance of class ProbART')

        else: 
            temp = len(self.all_params) #HARDCODE
            self.all_params['SubModel_' + str(temp)] = {
                'm': layer.m,
                'rho': layer.rho,
                'alpha': layer.alpha,
                'beta': layer.beta,
                'norm_T': layer.norm_T,
                'comp': layer.comp,
                'camRule': layer.camRule,
                'fastEncode': layer.fastEncode
            }
            self.layers_.append(0)
            if temp == 0:
                self.all_params['SubModel_' + str(temp)]['n'] = layer.n

    def fit(self, x=None, epochs=1, shuffle=True, seed=0, threading=None, convergence=None, checkpoint=True, every_sample=100, file_path='', return_state=False, start_from=0):
        '''
        Requires packages: numpy

        x: array
            an array of arrays used to train the model
        
        epochs: int
            integer stating number of epochs

        shuffle: bool
            boolean stating whether to shuffle data after every epoch

        seed: int
            an integer representing the seed

        threading: concurrent.futures.ThreadPoolExecutor

        convergence: None or int
            integer which states the tolerance of number of epochs with same number of active nodes

        checkpoint: boolean
            boolean stating whether to take checkpoints

        every_sample: int
            integer larger than 1 stating the frequency of checkpoints with regard to number of samples

        file_path: str
            string stating the file path to the folder to store the model object

        return_state: boolean
            boolean to state whether to output the states between layers

        start_from: int
            integer between -1 and number of layers inclusive stating which layer to start training from
            -1 is used when user wants to continue training a fully trained model
        
        trains selected layers contained in deepART
        '''
        if len(self.layers_) == 0:
            raise ValueError('No layers have been added')
        if x.shape[1] != self.all_params['SubModel_0']['n']:
            raise ValueError('Incompatible shape of training data')
        if x is None or len(x) == 0:
            raise ValueError('Input Data is Empty')
        if not (isinstance(epochs, int) and epochs > 0):
            raise ValueError('Epochs should be an integer more than zero')
        if not isinstance(shuffle, bool):
            raise ValueError('shuffle should be a boolean')
        if convergence is not None and not isinstance(convergence, int):
            raise ValueError('convergence must be None or an integer')
        if not isinstance(checkpoint, bool):
            raise ValueError('checkpoint must be a boolean')
        if every_sample<=0 or not isinstance(every_sample, int):
            raise ValueError('every_sample must be an integer greater than or equal to zero')
        if not isinstance(file_path, str):
            raise ValueError('file_path must be a string')
        if not isinstance(return_state, bool):
            raise ValueError('return_state must be a boolean value')
        if not -1<=start_from<=len(self.layers_) or not isinstance(start_from, int):
            raise ValueError('start_from must be an integer from -1 to the number of layers (inclusive)')
        if 0 in self.layers_[:start_from] and start_from > 0:
            raise ValueError('One of the earlier layers is not trained')
        if self.layers_[start_from] == 0 and start_from > 0:
            raise ValueError('Selected layer is not trained, choose one of the earlier layers')       
        if start_from == -1:
            if 0 in self.layers_:
                raise ValueError('One of the layers is not trained')
        all_preds = []
        if shuffle:
            np.random.seed(seed)
        for idx, dic in enumerate(self.all_params.values()): # iterate through layers
            if idx >= start_from:
                if idx == start_from and self.load_helper['failed']: # checks for existence of previously trained layer
                    model = self.layers_[start_from]

                elif start_from == -1 and not self.load_helper['failed']:
                    model = self.layers_[idx]
                else:
                    dic['seed'] = seed
                    if idx > 0:
                        dic['n'] = self.layers_[idx - 1].active
                    model = ProbART(**dic)
                prev_active = 0
                converge = 0
                for epoch in range(epochs): # train a particular layer
                    self.layers_[idx] = model
                    self.load_helper['failed'] = True
                    self.save(file_path + 'submodel_{}_epoch_{}.p'.format(idx, epoch))
                    print('Epoch: {}'.format(epoch))
                    if shuffle:
                        if idx == 0:
                            np.random.shuffle(x)
                        else:
                            np.random.shuffle(all_preds[idx-1])
                    if idx == 0:
                        for index, I in enumerate(x):
                            model.fit_predict(I=I.ravel(), threading=threading, learn=True)
                            if checkpoint:
                                if index % every_sample == 0:
                                    self.layers_[idx] = model # puts model object in deepart so it would be saved with deepart
                                    self.load_helper['failed'] = True # helper attribute to check for whether layer's training been disrupted
                                    self.save(file_path + 'submodel_{}_epoch_{}_sample_{}.p'.format(idx, epoch, index))
                    else:
                        for index, I in enumerate(all_preds[idx-1]):
                            model.fit_predict(I=I.ravel(), threading=threading, learn=True)
                            if checkpoint:
                                if index % every_sample == 0:
                                    self.layers_[idx] = model
                                    self.load_helper['failed'] = True
                                    self.save(file_path + 'submodel_{}_epoch_{}_sample_{}.p'.format(idx, epoch, index))
                    if convergence is not None: # checks for convergence
                        if prev_active < model.active:
                            prev_active = model.active
                            continue
                        else:
                            converge += 1
                            if converge > convergence:
                                print('Converged')
                                break
            else:
                model = self.layers_[idx] # pretrained layers before user-stated layer to start from
            model_pred = []
            if idx == 0:
                for I in x:
                    _, _ = model.fit_predict(I.ravel(), threading=threading, learn=False) 
                    active_nodes = np.copy(model.F2.nodes[0:model.active])
                    model_pred.append(active_nodes)
            else:
                for I in all_preds[idx-1]:
                    _, _ = model.fit_predict(I.ravel(), threading=threading, learn=False) 
                    active_nodes = np.copy(model.F2.nodes[0:model.active])
                    model_pred.append(active_nodes)
            self.layers_[idx] = model
            self.all_active['SubModel_' + str(idx)] = model.active # store active nodes for all layers
            all_preds.append(model_pred) # store all predictions
        self.load_helper['failed'] = False 
        if return_state:
            return all_preds

    def predict(self, x=None, threading=None, return_state=False):
        '''
        Requires packages: N.A.

        x: array
            array of arrays containing data to predict

        threading: concurrent.futures.ThreadPoolExecutor

        return_state: boolean
            boolean stating whether to return predictions of all layers

        returns predictions of a trained deepART
        '''
        all_preds = []
        for idx, model in enumerate(self.layers_):
            model_pred = []
            if idx == 0:
               for I in x:
                    _, _ = model.fit_predict(I.ravel(), threading=threading, learn=False) 
                    active_nodes = np.copy(model.F2.nodes[0:model.active])
                    model_pred.append(active_nodes)
            else:
                for I in all_preds[idx-1]:
                    _, _ = model.fit_predict(I.ravel(), threading=threading, learn=False) 
                    active_nodes = np.copy(model.F2.nodes[0:model.active])
                    model_pred.append(active_nodes)
            all_preds.append(model_pred)
        result = []
        all_results = {'SubModel_' + str(idx): [] for idx in range(len(self.layers_))}
        for model_pred in all_preds:
            model_result = []
            for pred in model_pred:
                temp = []
                for idx_2, i in enumerate(pred):
                    if i == 0:
                        continue
                    temp.append((idx_2, i))
                temp.sort(key=lambda x: x[1], reverse=True)
                model_result.append(temp)
            result.append(model_result)
        for idx, i in enumerate(result):
            all_results['SubModel_' + str(idx)] = i
        if not return_state:
            return list(all_results.values())[-1]
        else:
            return all_results


    def fit_predict(self, x=None, epochs=1, shuffle=True, seed=0, threading=None, convergence=None, checkpoint=True, every_sample=100, file_path=''):
        '''
        Requires packages: numpy

        x: array
            an array of arrays used to train the model
        
        epochs: int
            integer stating number of epochs

        shuffle: bool
            boolean stating whether to shuffle data after every epoch

        seed: int
            an integer representing the seed

        threading: concurrent.futures.ThreadPoolExecutor

        convergence: None or int
            integer which states the tolerance of number of epochs with same number of active nodes

        checkpoint: boolean
            boolean stating whether to take checkpoints

        every_sample: int
            integer larger than 1 stating the frequency of checkpoints with regard to number of samples

        file_path: str
            string stating the file path to the folder to store the model object

        return_state: boolean
            boolean to state whether to output the states between layers

        start_from: int
            integer between 0 and number of layers inclusive stating which layer to start training from
        
        trains selected layers in deepART and outputs predictions from all layers
        '''
        all_preds = self.fit(x=x, epochs=epochs, shuffle=shuffle, seed=seed, threading=threading, convergence=convergence, checkpoint=checkpoint, every_sample=every_sample, file_path=file_path, return_state=True)
        result = []
        all_results = {'SubModel_' + str(idx): [] for idx in range(len(self.layers_))}
        for model_pred in all_preds:
            model_result = []
            for pred in model_pred:
                temp = []
                for idx_2, i in enumerate(pred):
                    if i == 0:
                        continue
                    temp.append((idx_2, i))
                temp.sort(key=lambda x: x[1], reverse=True)
                model_result.append(temp)
            result.append(model_result)
        for idx, i in enumerate(result):
            all_results['SubModel_' + str(idx)] = i
        return all_results


    def save(self, file_path):
        '''
        file_path: string, path to save model to

        saves the model
        '''
        return save_model(self, file_path) 