# -*- coding: utf-8 -*-


import numpy as np
tol = 1e-8
# Calcul du proximal de la norme L1 : eps*lambda*norme1

def prox(eps,Lambda,u):
    n = u.shape[0]
    v = np.zeros(n)
    
    for i in range(n):
        
        if (u[i] > Lambda*eps + tol ): 
            v[i] = u[i]-Lambda*eps
        if (u[i] < -Lambda*eps - tol): 
            v[i] = u[i] +Lambda*eps
        if (u[i] >= -Lambda*eps - tol and u[i] <=Lambda*eps + tol): 
            v[i] = 0

    return v
   