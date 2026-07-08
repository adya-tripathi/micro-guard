# this file will have the whole model's math 

import numpy as np

def fit(X_train):
    # center of our normal behaviour 
    mu = np.mean(X_train,axis =0)
    # will define the normal behaviour(ellipse) boundary
    # adding regularization term to avoid failure in case 2 features are to much co related
    epsilon_reg = 1e-5
    sigma = np.cov(X_train,rowvar = False)
    sigma += np.eye(sigma.shape[0]) * epsilon_reg
    return mu,sigma

def prob(X,mu,sigma):
    inv = np.linalg.inv(sigma)
    det = np.linalg.det(sigma)
    n = len(mu)
    diff = X - mu
    # einsum efficiently computes the Mahalanobis distance for each data point without using loops
    exp = -0.5*np.einsum('ij,jk,ik->i',diff,inv,diff)
    coef = 1/(((2*np.pi)**(n/2))*(det**0.5))
    return coef*np.exp(exp)

def log_gaussian(X, mu, sigma):
    X = np.asarray(X)
    mu = np.asarray(mu)
    sigma = np.asarray(sigma)

    n = len(mu)

    sign, log_det = np.linalg.slogdet(sigma)
    inv_sigma = np.linalg.pinv(sigma)

    diff = X - mu
    
    distance = -0.5 * np.sum((diff @ inv_sigma) * diff, axis=1)
    log_normalization = -0.5 * n * np.log(2 * np.pi) - 0.5 * log_det
    
    return log_normalization + distance
 
def load_model(path="model_params.npz"):
    data = np.load(path, allow_pickle=True)
    mu = data["mu"]
    sigma = data["sigma"]
    eps = float(data["eps"])
    features = list(data["features"])
    scaler_mean = data["scaler_mean"]
    scaler_scale = data["scaler_scale"]
    return mu, sigma, eps, features,scaler_mean,scaler_scale