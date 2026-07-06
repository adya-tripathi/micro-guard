# this file will have the whole model's math 

import numpy as np

def fit(X_train):
    # center of our normal behaviour 
    mu = np.mean(X_train,axis =0)
    # will define the normal behaviour(ellipse) boundary
    sigma = np.cov(X_train,rowvar = False)
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