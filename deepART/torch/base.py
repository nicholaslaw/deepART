import torch

def generateComp(I):
    '''
    generateComp(I)

    Args:
    -----------
    I : numpy array
    '''
    Icomp = 1 - I

    return Icomp

def load_model(file_path):
    '''
    file_path: string, file path to save

    loads ART model
    '''
    return joblib.load(open(file_path, 'r'))


def save_model(py_obj, file_path):
    '''
    py_obj: Python object
    file_path: string, file path to save

    saves ART model
    '''
    joblib.dump(py_obj, open(file_path, 'w'))



def softmax(lst, scale = True, alpha = 1e-30):
    """Compute softmax values for each sets of scores in x."""
    #scaled max x and multipley by 10 to increase by one order
    if scale:
        lst = 10*lst/(max(lst)+alpha)
    exp_lst = torch.exp(lst)
    return exp_lst / torch.sum(exp_lst, dim=0)