# -*- coding: utf-8 -*-

import numpy as np
from lasso import lasso
from lasso_accelerated import lasso_accelerated


# Etude de variation de Lambda
    
def Lambda_effect(A,b,u_0,erreur,Lambda_min,Lambda_max,Nb_Lambda = 10):
    '''
    Inputs: 
        - A, b, u_0, erreur
        - Lambda_min, Lambda_max
        - Nb_Lambda : le nombre de pas
    Outputs:
        - Non_zero : nombre d'elements non nuls dans la solution optimale
        - Solutions : les solutions optimales pour chaque Lambda
    '''
    
    step = (Lambda_max - Lambda_min)/Nb_Lambda
    Non_zero = [] # Contien le nombre d'element non nulle pour un lambda donné
    Solutions = []
    for i in range(Nb_Lambda):
        Lambda_i = Lambda_min + i*step
        
        res = lasso(A,b,Lambda_i, u_0,erreur)
        Solution = res['Sol_opt']
        Count_non_zero = np.count_nonzero(Solution)
        Non_zero.append(Count_non_zero)
        Solutions.append(Solution)
        
    return Non_zero,Solutions
    

    
def Lambda_effect_accelerated(A,b,u_0,v_0,erreur,Lambda_min,Lambda_max,Nb_Lambda = 10): 
    '''
    Inputs: 
        - A, b, u_0, v_0, erreur
        - Lambda_min, Lambda_max
        - Nb_Lambda : le nombre de pas
    Outputs:
        - Non_zero : nombre d'elements non nuls dans la solution optimale
        - Solutions : les solutions optimales pour chaque Lambda
    ''' 

    step = (Lambda_max - Lambda_min)/Nb_Lambda
    Non_zero = [] # Contien le nombre d'element non nulle pour un lambda donné
    Solutions = []
    for i in range(Nb_Lambda):
        Lambda_i = Lambda_min + i*step
        
        res = lasso_accelerated(A,b,Lambda_i, u_0,v_0,erreur)
        Solution = res['Sol_opt']
        Count_non_zero = np.count_nonzero(Solution)
        Non_zero.append(Count_non_zero)
        Solutions.append(Solution)


    return Non_zero,Solutions