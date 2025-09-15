import numpy as np

############
#   OPE    #
############
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE
from sklearn.neighbors import KernelDensity
from dppy.finite_dpps import FiniteDPP
from scipy.linalg import svd

# Function for estimating the Jacobi parameters
def generate_Jacobi_parameters(X):
    mu1 = X.mean(axis=0)
    Xs = X * X
    mu2 = Xs.mean(axis=0)
    tmp = (mu1 + 1) / 2 + (mu2 - 1) / 4 - (mu1 + 1) * (mu1 + 1) / 4
    t = (1 - mu1) / 2 * ((1 + mu1) * (1 - mu1) / 4 / tmp - 1)
    a = t - 1
    b = (1 + mu1) / (1 - mu1) * t - 1
    jac_params = np.column_stack((a,b))
    jac_params = np.minimum(np.maximum(jac_params,-0.5),0.5)
    return jac_params 

X = 2 * np.random.rand(10000, 2) -1
generate_Jacobi_parameters(X)

def draw_OPE(X, m:int, nb_samples:int, ab_coeff=-.5, gamma_X=None, prop_uniform=0.): 
    # complexity is n^2*m^2 if gamma_X is None then KDE needs to be computed, else n*m^2 
    n = len(X)
    m = int(m)
    d = X.shape[-1]
    if gamma_X is None: # /!\ induce n^2 complexity
        print("/!\ OPE sampling and gamma_X is None\nit will be KDE estimated, inducing n^2 complexity")
        # construct gamma tilde KDE estimation
        # alternatively, scipy.stats.gaussian_kde can be used
        kde = KernelDensity(kernel="epanechnikov", bandwidth="scott").fit(X)
        gamma_X = np.exp(kde.score_samples(X))

    # handle a proportion of samples drawn uniformly
    nb_total_uniform = round(prop_uniform * n)
    nb_sample_uniform = round(prop_uniform * m)
    if nb_total_uniform>0:
        m -= nb_sample_uniform
        if m==0:
            return draw_uniform(X, nb_sample_uniform, nb_samples)
        else:
            argsort_X = np.argsort(gamma_X)
            uniform_idX = argsort_X[:nb_total_uniform]
            OPE_idX = argsort_X[nb_total_uniform:]
            X_OPE = X[OPE_idX]
            gamma_X_OPE = gamma_X[OPE_idX] 
    else:
        X_OPE = X
        gamma_X_OPE = gamma_X

    # obtain P which decompose continuous DPP K = PP^T
    ab_coeff_array = np.zeros((d,2)) + ab_coeff
    dpp = MultivariateJacobiOPE(m, ab_coeff_array)
    polynom_X = dpp.eval_multiD_polynomials(X_OPE)
    ref_measure_X = dpp.eval_w(X_OPE)
    P = np.sqrt(ref_measure_X / gamma_X_OPE)[:,None] * polynom_X

    # extract m eigenvectors of K by SVD of P
    U, S, Vh = svd(P, full_matrices=False)
    eig_vals, eig_vecs = np.ones(m), U
    diag_K_tilde = (U**2).sum(-1)
    OPE_weights = (1 / n) / diag_K_tilde

    # draw from OPE
    DPP = FiniteDPP(kernel_type='correlation', projection=True, K_eig_dec=(eig_vals, eig_vecs))
    for _ in range(nb_samples):
        DPP.sample_exact(mode='GS')
    OPE_samples = np.array(DPP.list_of_samples, dtype=int)

    # return samples and associated weights
    if nb_sample_uniform>0:
        weights = np.zeros(n)
        weights[OPE_idX] = OPE_weights
        weights[uniform_idX] =  nb_total_uniform / (n * nb_sample_uniform)
        # draw nb_sample_uniform samples uniformly
        uniform_samples = np.random.choice(uniform_idX, (nb_samples, nb_sample_uniform))
        # concatenate OPE and uniform
        samples = np.concatenate((OPE_idX[OPE_samples], uniform_samples), axis=-1)
        return samples, weights[samples]
    else:
        return OPE_samples, OPE_weights[OPE_samples]
    




####################
#   discrete OPE   #
####################
import itertools
from scipy.linalg import qr

def compute_ordering(m, d):
    layer_max = np.floor(m**(1.0 / d)).astype(np.int16)
    ordering = itertools.chain.from_iterable(
                filter(lambda x: k in x,
                       itertools.product(range(k + 1), repeat=d))
                for k in range(layer_max + 1))
    return np.array(list(ordering)[:m])

def draw_discrete_OPE(X, m, nb_samples):
    n = len(X)
    # compute discrete OPE kernel
    ordering = compute_ordering(m, X.shape[-1])
    vander_matrix = np.prod(X[:,None,:]**ordering, axis=-1)
    q_matrix, r_matrix = qr(vander_matrix, mode="economic")
    eig_vals, eig_vecs = np.ones(m), q_matrix
    diag_K_tilde = (q_matrix**2).sum(-1)
    weights = (1 / n) / diag_K_tilde

    # draw from discrete OPE
    DPP = FiniteDPP(kernel_type='correlation', projection=True, K_eig_dec=(eig_vals, eig_vecs))
    for _ in range(nb_samples):
        DPP.sample_exact(mode='GS')
    samples = np.array(DPP.list_of_samples)
    return samples, weights[samples]





#####################
#   gaussian kDPP   #
#####################
def gaussian_kernel(X, sigma=1.):
    delta = X[:,None,:] - X[None,:,:]
    K = np.exp(-0.5 * np.sum(delta**2, axis=-1) / sigma**2)
    return K

def elementary_symmetric_polynomial(k:int, arr):
    n = len(arr)
    esp_eval = np.zeros((n+1,k+1))
    esp_eval[:,0] = np.ones(n+1)
    for i in range(1,k+1):
        for j in range(1,n+1):
            esp_eval[j,i] = esp_eval[j-1,i] + arr[j-1] * esp_eval[j-1,i-1]
    return esp_eval[-1,-1]

def get_kDPP_weights(likelihood, k:int):
    n = len(likelihood)
    U, S, Vh = svd(likelihood)
    eigvals = np.abs(S)
    elem_sym_pol_ratio = np.empty(n)
    for i_eigval in range(len(eigvals)):
        e_mn_kmo = elementary_symmetric_polynomial(k-1, np.concatenate((eigvals[:i_eigval], eigvals[i_eigval+1:])))
        e_n_k = elementary_symmetric_polynomial(k, eigvals)
        elem_sym_pol_ratio[i_eigval] = e_mn_kmo / e_n_k
    return (1 / n) / (U**2 * eigvals * elem_sym_pol_ratio).sum(-1)
    
def draw_gaussian_kDPP(X, m:int, nb_samples:int, bandwidth=.1):
    likelihood = gaussian_kernel(X, bandwidth)
    weights = get_kDPP_weights(likelihood, m)
    DPP = FiniteDPP(kernel_type='likelihood', L=likelihood)
    for _ in range(nb_samples):
        DPP.sample_exact_k_dpp(m, mode='GS')
    samples = np.array(DPP.list_of_samples)
    return samples, weights[samples]





##################
#   stratified   #
##################
from collections import defaultdict

def shuffle_cycle_array(uncompleted_stratas, m):
    # cycle through an array then shuffle and repeat until m elements (with repetitions) are selected
    completed_stratas = []
    for count in range(m):
        if count == len(uncompleted_stratas):
            np.random.shuffle(uncompleted_stratas)
        completed_stratas.append(uncompleted_stratas[count % len(uncompleted_stratas)])
    return completed_stratas

def draw_stratified(X, m, nb_samples):
    d = X.shape[-1]
    box_length = m**(-1/d)
    weights = None # return None weights for samplers API consistency

    # build stratas by cycling through each point and add it to the corresponding strata accordingly to its position.
    stratas = defaultdict(list)
    for i_x, x in enumerate((X+1)/2):
        key = ""
        for dim in range(d):
            key += str(int(x[dim]//box_length))
        stratas[key].append(i_x)
    stratas = np.array(list(stratas.values()), dtype=object)
    nb_stratas = len(stratas)

    # sample from each strata
    strata_samples = np.empty((nb_samples, nb_stratas), dtype=int)
    samples = np.empty((nb_samples, m), dtype=int)

    # in case there strictly less stratas than m, cycle through all stratas then shuffle and repeat until m stratas (with repetitions) are selected
    if nb_stratas<m:
        if nb_stratas<.9*m: print(f"/!\ m={m} but there are only {nb_stratas} stratas. Try increase n or reduce m.")
        for i_sample in range(nb_samples):
            completed_stratas = shuffle_cycle_array(stratas, m)
            for i_strata, strata in enumerate(completed_stratas):
                samples[i_sample,i_strata] = np.random.choice(strata)
        return samples, weights
    
    # else, select uniformly one point for each strata...
    for i_strata, strata in enumerate(stratas):
        strata_samples[:,i_strata] = np.random.choice(strata, nb_samples)
    if nb_stratas==m:
        return strata_samples, weights
    # ... and in case there are strictly more stratas than m, extract m stratas for each samples
    else: 
        for i_strata_sample, strata_sample in enumerate(strata_samples):
            samples[i_strata_sample] = np.random.choice(strata_sample, m, replace=False)

    return samples, weights





###################
#   sensitivity   #
###################
from scipy.spatial import distance
from scipy.cluster.vq import vq

def D_squared_sampling(X, k):
    n = len(X)
    B = np.zeros(k, dtype=int)
    B[0] = np.random.choice(n)
    sqdist_to_B = np.full(n, np.inf)
    for i in range(1,k):
        sqdist_to_sample = distance.cdist(X[None,B[i-1]], X, 'sqeuclidean')[0]
        sqdist_to_B = np.minimum(sqdist_to_B, sqdist_to_sample)
        sample = np.random.choice(n, p=sqdist_to_B/sqdist_to_B.sum())
        B[i] = sample
    return B        

def best_quant(X, k, delta):
    n_runs = np.ceil(10*np.log(1/delta)).astype(int)
    quant_error_min = np.inf
    for i_run in range(n_runs):
        B = D_squared_sampling(X, k)
        code, dist = vq(X, X[B])
        quant_error = (dist**2).sum()
        if quant_error < quant_error_min:
            quant_error_min = quant_error
            code_min, dist_min = code, dist
    return code_min, dist_min

def kmean_sensit_ub(X, k, delta):
    """
    cf. [Bachem+2017](http://arxiv.org/abs/1703.06476).
    """
    code, dist = best_quant(X, k, delta)
    sq_dist = dist**2
    alpha = 16*(np.log2(k) + 2)
    n = len(X)
    c_B = 1/n * sq_dist.sum()
    count_B = np.zeros(k, dtype=int)
    sq_dist_B = np.zeros(k)
    for c, sqd in zip(code, sq_dist):
        count_B[c] += 1
        sq_dist_B[c] += sqd
    sensit_ub = 2*alpha*sq_dist/c_B + 4*alpha*sq_dist_B[code]/(count_B[code]*c_B) + 4*n/count_B[code]
    return sensit_ub

def get_sensit_sample(X, m, k, delta):
    n = len(X)
    sensit_ub = kmean_sensit_ub(X, k, delta)
    sensit_proba = sensit_ub/sensit_ub.sum()
    samples = np.random.choice(n, m, p=sensit_proba)
    weights = 1/(sensit_proba[samples]*m*n)
    return samples, weights

def draw_sensitivity(X, m, nb_samples, k, delta):
    samples, weights = np.empty((nb_samples, m), dtype=int), np.empty((nb_samples, m))
    for i_sample in range(nb_samples):
        samples[i_sample], weights[i_sample] = get_sensit_sample(X, m, k, delta)
    return samples, weights





###############
#   uniform   #
###############
def draw_uniform(X, m, nb_samples):
    n = len(X)
    samples = np.random.choice(n, (nb_samples, m))
    weights = None
    return samples, weights


###########################
#      sensitivity        #
# for logistic regression #
###########################
from scipy.spatial import distance
from scipy.cluster.vq import vq

def D_squared_sampling(X, k):
    n = len(X)
    B = np.zeros(k, dtype=int)
    B[0] = np.random.choice(n)
    sqdist_to_B = np.full(n, np.inf)
    for i in range(1,k):
        sqdist_to_sample = distance.cdist(X[None,B[i-1]], X, 'sqeuclidean')[0]
        sqdist_to_B = np.minimum(sqdist_to_B, sqdist_to_sample)
        sample = np.random.choice(n, p=sqdist_to_B/sqdist_to_B.sum())
        B[i] = sample
    return B        


def best_quant(X, k, delta):
    n_runs = np.ceil(10*np.log(1/delta)).astype(int)
    quant_error_min = np.inf
    for i_run in range(n_runs):
        B = D_squared_sampling(X, k)
        code, dist = vq(X, X[B])
        quant_error = (dist**2).sum()
        if quant_error < quant_error_min:
            quant_error_min = quant_error
            code_min, dist_min = code, dist
    return code_min, dist_min


def logist_sensit_ub(X, k, delta, R):
    """
    cf. Lemma 3.1 from [Huggings+2017](http://arxiv.org/abs/1605.06423).
    """
    code, _ = best_quant(X, k, delta) # TODO: get best quant i.e. kmeans sol?
    n = len(X)
    count_B = np.zeros(k, dtype=int)
    sum_B = np.zeros((k, X.shape[-1]))
    for c in code:
        count_B[c] += 1
        sum_B[c] += X[c]

    sensit_ub = np.ones(n)
    for cb, sb in zip(count_B, sum_B):
        sensit_ub += cb * np.exp( - R * ((sb/cb - X)**2).sum(-1)**.5)
    
    for i_c, c in enumerate(code): # correct for the term where sample is in cluster
        cb, sb = count_B[c], sum_B[c]
        sensit_ub[i_c] -= cb * np.exp( - R * ((sb/cb - X[i_c])**2).sum(-1)**.5)
        sensit_ub[i_c] += (cb-1) * np.exp( - R * (((sb - X[i_c]) / (cb-1) - X[i_c])**2).sum(-1)**.5)
    return n / sensit_ub


def get_logist_sensit_sample(X, m, k, delta, R):
    n = len(X)
    sensit_ub = logist_sensit_ub(X, k, delta, R)
    sensit_proba = sensit_ub/sensit_ub.sum()
    samples = np.random.choice(n, m, p=sensit_proba)
    weights = 1/(sensit_proba[samples]*m*n)
    return samples, weights


def draw_logist_sensitivity(X, m, nb_samples, k, delta, R):
    samples, weights = np.empty((nb_samples, m), dtype=int), np.empty((nb_samples, m))
    for i_sample in range(nb_samples):
        samples[i_sample], weights[i_sample] = get_logist_sensit_sample(X, m, k, delta, R)
    return samples, weights