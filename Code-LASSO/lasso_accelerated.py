# -*- coding: utf-8 -*-

import numpy as np
import prox as pr

def lasso_accelerated(A,b,Lambda, u_0,v_0,erreur,*args):
    '''
    Inputs:
        - matrice A, vecteur B
        - Lambda : la constante de LASSO
        - u_0, v_0 : le point initial
        - erreur : la tolérance
        - *args : le nombre maximal d'itérations
    Outputs: resultats = dict()
        - 'Err_hist' : valeurs de l'erreur à chaque itération
        - 'Sol_opt' : la solution optimale
        - 'nb_iterations' : le nombre d'itérations
        - 'Diff_hist' : la différence entre deux valeurs successives
    '''

    L,_ = np.linalg.eig(np.matmul(np.transpose(A),A)) # constant L
    Err = np.inf 
    u_k = u_0
    v_k = v_0
    
    Err_hist = []
    Diff_hist = []

    Nb_it = 1

    test_it_max = True

    if len(args) == 1:
        Nb_max_it = args[0]
        test_it_max = Nb_it < Nb_max_it
    
    test_err = Err > erreur

    
    while(test_err and test_it_max):
        
        eps_k = 1/np.max(np.abs(L))
        
        gradiant_F_k = np.dot(np.transpose(A),(np.dot(A,v_k)-b))
        
        u_k_new = pr.prox(eps_k,Lambda,v_k-eps_k*(gradiant_F_k))
        v_k_new = u_k_new + ((Nb_it-1)/(Nb_it+2))*(u_k_new-u_k)
        

        Err = np.linalg.norm(u_k-u_k_new)/(np.linalg.norm(u_k)+1e-10)
        Err_hist.append(Err)
        
        Diff = np.linalg.norm(u_k-u_k_new)
        Diff_hist.append(Diff)
        
        u_k = u_k_new
        v_k = v_k_new
        
        Nb_it += 1
        
        test_err = Err > erreur
        
        if len(args) ==1:
            test_it_max = Nb_it < Nb_max_it

    resultats = dict()
    resultats['Err_hist'] = Err_hist[1:]
    resultats['Sol_opt'] = u_k
    resultats['nb_iterations'] = Nb_it-1
    resultats['Diff_hist'] = Diff_hist
    return resultats
      
  
    
    