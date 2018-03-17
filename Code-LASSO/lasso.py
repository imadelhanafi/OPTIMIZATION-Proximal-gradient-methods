# -*- coding: utf-8 -*-


import numpy as np 
import prox as pr

 
# Resolution du problème Lasso avec Gradiant Proximal


def lasso(A,b,Lambda, u_0,erreur, *args): 
    '''
    Inputs:
        - matrice A, vecteur B
        - Lambda : la constante de LASSO
        - U_0 : le point initial
        - erreur : la tolérance
        - *args : le nombre maximal d'itérations
    Outputs: resultats = dict()
        - 'Err_hist' : valeurs de l'erreur à chaque itération
        - 'Sol_opt' : la solution optimale
        - 'nb_iterations' : le nombre d'itérations
        - 'Diff_hist' : la différence entre deux valeurs successives
    '''

    L,_ = np.linalg.eig(np.matmul(np.transpose(A),A)) # constant L de Lipschitz
    Err = np.inf # Erreur pour condition d'arrêt
    u_k = u_0
    k = 0 # Indice des itérations
    Err_hist = [] # Erreur historique
    Diff_hist = []

    #print("#### Début de la résolution du problème ####")
    Nb_it = 1
    test_it_max = True

    if len(args) == 1:
        Nb_max_it = args[0]
        test_it_max = Nb_it < Nb_max_it
    
    
    test_err = Err > erreur
    while(test_err and test_it_max):
        
        eps_k = 1/np.max(np.abs(L))
        
        gradiant_F_k = np.dot(np.transpose(A),(np.dot(A,u_k)-b))
        
        u_k_new = pr.prox(eps_k,Lambda,u_k-eps_k*(gradiant_F_k))

        Err = np.linalg.norm(u_k-u_k_new)/(np.linalg.norm(u_k)+1e-10)        
        Err_hist.append(Err)
        
        Diff = np.linalg.norm(u_k-u_k_new)
        Diff_hist.append(Diff)
        
        Nb_it += 1
        
        test_err = Err > erreur
        
        if len(args) ==1:
            test_it_max = Nb_it < Nb_max_it

        u_k = u_k_new
    #print("#### Fin de la résolution du problème ####")
    resultats = dict()
    resultats['Err_hist'] = Err_hist[1:]
    resultats['Sol_opt'] = u_k
    resultats['nb_iterations'] = Nb_it-1
    resultats['Diff_hist'] = Diff_hist
    return resultats