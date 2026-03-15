import numpy as np
from scipy.optimize import minimize
from tinybenchmarks_utils import *
from py_irt_direct.py_irt_direct import *

# Modify tinyBenchmarks to not save the IRT parameters to disk
# in order to facilitate parallelization

def create_irt_dataset_no_saving(responses): 
    
    """
    Creates a dataset suitable for IRT analysis from a given set of responses and saves it in a JSON lines format.
    
    Parameters:
    - responses: A numpy array where each row represents a subject and each column a question.
    """
    
    dataset = []
    for i in range(responses.shape[0]):
        aux = {}
        aux_q = {}
        
        # Iterate over each question to create a response dict
        for j in range(responses.shape[1]):
            aux_q['q' + str(j)] = int(responses[i, j])
        aux['subject_id'] = str(i)
        aux['responses'] = aux_q
        dataset.append(aux)
    
    return dataset

def train_irt_model_no_saving(dataset, model_name, D, lr, epochs, device):
    
    """
    Trains an IRT model using the py-irt command-line tool.
    
    Parameters:
    - dataset_name: The name of the dataset file.
    - model_name: The desired name for the output model.
    - D: The number of dimensions for the IRT model.
    - lr: Learning rate for the model training.
    - epochs: The number of epochs to train the model.
    - device: The computing device ('cpu' or 'gpu') to use for training.
    """

    return train_no_saving('multidim_2pl', dataset, model_name, dims=D, lr=lr, epochs=epochs, device=device, priors='hierarchical', seed=42, deterministic=True, verbose=False)
        
def load_irt_parameters_no_saving(params):
    
    """
    Loads the parameters from a trained IRT model.
    
    Parameters:
    - model_name: The name of the file containing the model parameters.
    
    Returns: 
    - A, B, and Theta: The discrimination, difficulty, and ability parameters, respectively, from the IRT model.
    """
    
    A = np.array(params['disc']).T[None, :, :]
    B = np.array(params['diff']).T[None, :, :]
    Theta = np.array(params['ability'])[:,:,None]
    return A, B, Theta



def estimate_ability_parameters_no_saving(responses_test, A, B, theta_init=None, eps=1e-10, optimizer="BFGS"):
    
    """
    Estimates the ability parameters for a new set of test responses.
    
    Parameters:
    - responses_test: A 1D array of the test subject's responses.
    - A: The discrimination parameters of the IRT model.
    - B: The difficulty parameters of the IRT model.
    - theta_init: Initial guess for the ability parameters.
    - eps: A small value to avoid division by zero and log of zero errors.
    - optimizer: The optimization method to use.
    - weights: weighting for items according to their representativeness of the whole scenario
    
    Returns: 
    - optimal_theta: The estimated ability parameters for the test subject.
    """

    D = A.shape[1]
    
    # Define the negative log likelihood function
    def neg_log_like(x):
        P = item_curve(x.reshape(1, D, 1), A, B).squeeze()
        log_likelihood = np.sum(responses_test * np.log(P + eps) + (1 - responses_test) * np.log(1 - P + eps))
        return -log_likelihood
    
    # Ensure the initial theta is a numpy array with the correct shape
    if type(theta_init) == np.ndarray:
        theta_init = theta_init.reshape(-1)
        assert theta_init.shape[0] == D
    else:
        theta_init = np.zeros(D)

    # Use the minimize function to find the ability parameters that minimize the negative log likelihood
    optimal_theta = minimize(neg_log_like, theta_init, method = optimizer).x[None,:,None] 
    
    return optimal_theta