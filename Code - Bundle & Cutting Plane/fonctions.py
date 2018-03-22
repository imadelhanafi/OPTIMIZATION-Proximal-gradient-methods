# -*- coding: utf-8 -*-

import numpy as np
import random


def X_generator(param):
    # Parametres
    nVariables = param.nVariables
    nCassures = param.nCassures
    xMin = param.xMin
    xMax = param.xMax
    
    # Initialisation
    X = np.empty((nVariables,nCassures))
    
    #Calcul
    for i in range(nVariables):
        for j in range(nCassures):
            X[i,j] = xMin[i]+random.uniform(0,1)*(xMax[i]-xMin[i])

    return X

def computations(param, X, x):
    # Parametres
    EPS = param.EPS
    nVariables = param.nVariables
    nCassures = param.nCassures
    
    # Initialisation
    terme_nlp_j = np.zeros(nVariables)
    terme_subg = np.zeros(nVariables)
    sub_gradients = np.zeros(nVariables)
        
    # Calcul
    for i in range(nVariables):
        for j in range(nCassures):
            if x[i] - X[i,j] < -EPS:
                terme_nlp_j[i] += (j+1)*(-2*(x[i] - X[i,j]))
                terme_subg[i] += (j+1)*(-2)
            elif x[i] - X[i,j] > +EPS:
                terme_nlp_j[i] += (j+1)*(+3*(x[i] - X[i,j]))
                terme_subg[i] += (j+1)*(+3)
            else:
                terme_nlp_j[i] += 0
                terme_subg[i] += 0
                
    # Resultats
    nlp_obj = (1/nVariables)*(x + (1/nCassures)*terme_nlp_j).sum()
    sub_gradients = (1/nVariables)*(1 + (1/nCassures)*terme_subg)
    
    result = dict()
    result['nlp_obj'] =  nlp_obj
    result['sub_gradients'] = sub_gradients
    return result
  