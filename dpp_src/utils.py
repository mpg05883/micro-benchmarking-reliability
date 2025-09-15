import numpy as np
from pickle import dump, load, HIGHEST_PROTOCOL
import matplotlib.pyplot as plt


###################
# Data generation #
###################

def get_hypercube_data(n,d, border=1):
    data = border*2*(np.random.rand(n, d) - .5)
    return data

def get_disk_data(n, d, radius=1):
    n = round(n * 4/(np.pi*radius**2)) # to compensate for filtering
    data = 2*(np.random.rand(n, d) - .5)
    filtered_data = []
    for point in data:
        if np.linalg.norm(point)<radius:
            filtered_data.append(point)
    return np.array(filtered_data)





def get_evenly_spaced_circle(radius, n_circles):
    angles = 2*np.pi*np.arange(n_circles)/n_circles + np.pi/4
    means = np.stack((radius*np.cos(angles), radius*np.sin(angles)))
    return means.T

def get_circle_data(n, d, n_circle, n_means=None, radius=0.5*2**.5, variance=1/49, border=1):
    means = get_evenly_spaced_circle(radius, n_circle)
    covariances = [variance*np.identity(d) for i in range(n_circle)]
    data = []
    if n_means is None:
        n_means = n_circle

    for i in range(n_means):
        data.extend(np.random.multivariate_normal(means[i], covariances[i], round(n/n_means)))
    filtered_data = []
    for point in data:
        if (np.abs(point) < border).all():
            filtered_data.append(point)
    return np.array(filtered_data)







def get_corners(d):
    corners = []
    for i in range(2**d):
        corner = np.zeros(d)
        for j in range(d):
            if (i & (1 << j)) != 0: corner[j] = 1
            else: corner[j] = -1
        corners.append(corner)
    return np.array(corners)

def get_corner_data(n, d, means=None, variance=1/25, border=1):
    if isinstance(means, int):
        k = means
        if k<=2**d:
            means = 0.5*get_corners(d)[:k]
        else: 
            raise Exception("Sorry, more centers are required than they are corners in the hypercube") 
    else: 
        k = len(means)
    covariances = [variance*np.identity(d) for i in range(k)]
    data = []
    for i in range(k):
        data.extend(np.random.multivariate_normal(means[i], covariances[i], int(n/k)))
    filtered_data = []
    for point in data:
        if (np.abs(point) < border).all():
            filtered_data.append(point)
    return np.array(filtered_data)















#################
# Loss and risk #
#################
from scipy.cluster.vq import vq

def risk(X, queries):
    code, dist = vq(X.reshape(-1, X.shape[-1]), queries)
    dist = dist.reshape(X.shape[:-1])
    return dist ** 2

def loss(X, query=None, samples=None, weights=None):
    if query is None:
        query = np.zeros((1,X.shape[-1]))
    if samples is None:
        return risk(X, query).mean(-1)
    elif weights is None:
        return (risk(X[samples], query)).mean(-1)
    else:
        return (risk(X[samples], query) * weights).sum(-1)

def relative_error(y_hat, y):
    return np.abs(1 - y_hat / y)



def logist_risk(X, y, queries):
    energy = y * (X @ queries)
    return np.log(1 + np.exp(energy))

def logist_loss(X, y, query, samples=None, weights=None):
    if samples is None:
        return logist_risk(X, y, query).mean(-1)
    elif weights is None:
        return (logist_risk(X[samples], y[samples], query)).mean(-1)
    else:
        return (logist_risk(X[samples], y[samples], query) * weights).sum(-1)



#########
# Misc. #
#########
from itertools import combinations

def get_true_sensit(X, k):
    max_sensit_query = np.zeros(len(X))
    for combin in combinations(X, k):
        queries = combin
        risk_query = risk(X, queries)
        sensit_query = risk_query / risk_query.sum()
        max_sensit_query = np.maximum(max_sensit_query, sensit_query)
    return max_sensit_query



def pickle_dump(obj, path):
    with open(path, 'wb') as file:
        dump(obj, file, protocol=HIGHEST_PROTOCOL)

def pickle_load(path):
    with open(path, 'rb') as file:
        return load(file)   

def set_plotting_options(usetex=False, font_size=None):
    params = {'text.usetex': usetex,
              'font.family': 'roman' if usetex else None,
              'font.size':font_size,}
    plt.rcParams.update(params)